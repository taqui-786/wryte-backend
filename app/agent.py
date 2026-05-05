"""
agent.py — LangGraph agent with:
  • Short-term memory  → AsyncPostgresSaver (per-thread checkpointing)
  • Long-term memory   → AsyncPostgresStore (cross-thread, per-user memories)

The compiled `workflow` is built once during the FastAPI lifespan (see app.py)
and stored on the app state so every request shares the same graph object.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import AsyncGenerator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, messages_to_dict
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.store.postgres.aio import AsyncPostgresStore
from pydantic import BaseModel
from typing import Annotated, TypedDict
import os

load_dotenv()


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatNVIDIA(
    model="qwen/qwen3.5-122b-a10b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=1,
    top_p=0.95,
    max_completion_tokens=16384,
    model_kwargs={
        "enable_thinking": True,
        "reasoning_budget": 16384,
    },
)


# ---------------------------------------------------------------------------
# State & Context
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseModel], add_messages]


@dataclass
class UserContext:
    """Passed at invocation time to identify the current user for memory namespacing."""
    user_id: str


# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------

async def chat_node(state: ChatState, runtime: Runtime[UserContext]):
    """
    Main chat node.

    1. Retrieves relevant long-term memories for this user from the store.
    2. Injects them into the system prompt so the model is aware.
    3. If the user explicitly asks to remember something, extracts and stores it.
    4. Calls the LLM and returns the response.
    """
    user_id = runtime.context.user_id
    memory_namespace = ("memories", user_id)

    last_message = state["messages"][-1]
    last_content: str = last_message.content if hasattr(last_message, "content") else str(last_message)

    # --- Retrieve relevant long-term memories ---
    memories = await runtime.store.asearch(
        memory_namespace,
        query=last_content,
        limit=5,
    )
    memory_lines = [d.value["data"] for d in memories]
    memory_context = "\n".join(memory_lines) if memory_lines else "No memories yet."

    system_prompt = (
        "You are a helpful, thoughtful writing assistant called Wryte.\n"
        "You have access to the following long-term memories about the user:\n"
        f"{memory_context}\n\n"
        "Use these memories naturally when they are relevant. "
        "Do not mention the memory system explicitly unless asked."
    )

    # --- Store new memory if user explicitly asks ---
    if "remember" in last_content.lower():
        # Ask the LLM to extract what should be remembered
        extraction_prompt = (
            f"The user said: \"{last_content}\"\n\n"
            "Extract a single, concise fact to remember about the user from this message. "
            "Return only the fact, nothing else."
        )
        extraction = await llm.ainvoke([{"role": "user", "content": extraction_prompt}])
        memory_text = extraction.content.strip()
        if memory_text:
            await runtime.store.aput(
                memory_namespace,
                str(uuid.uuid4()),
                {"data": memory_text},
            )

    # --- Call the model ---
    response = await llm.ainvoke(
        [{"role": "system", "content": system_prompt}] + list(state["messages"])
    )
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    checkpointer: AsyncPostgresSaver,
    store: AsyncPostgresStore,
) -> CompiledStateGraph:
    """
    Compile the LangGraph workflow.

    Called once during the FastAPI lifespan after the store/checkpointer
    are set up and their tables have been created.
    """
    builder = StateGraph(ChatState, context_schema=UserContext)
    builder.add_node("chat_node", chat_node)
    builder.add_edge(START, "chat_node")
    builder.add_edge("chat_node", END)

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
    )


# ---------------------------------------------------------------------------
# Public API — called by app.py routes
# ---------------------------------------------------------------------------

async def my_agent(
    workflow: CompiledStateGraph,
    user_input: str,
    thread_id: str,
    user_id: str,
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from the agent for a given user + thread.

    Yields raw string chunks as they arrive.
    """
    config = {"configurable": {"thread_id": thread_id}}
    context = UserContext(user_id=user_id)

    async for chunk in workflow.astream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
        context=context,
    ):
        if chunk.get("type") == "messages":
            message_chunk, _metadata = chunk["data"]
            if message_chunk.content:
                yield message_chunk.content


async def get_chat_state(
    workflow: CompiledStateGraph,
    thread_id: str,
):
    """Return the persisted state for a thread, with messages serialised to dicts."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        res = await workflow.aget_state(config=config)
        if res is None:
            return None

        state_list = list(res)
        values = dict(state_list[0])
        if "messages" in values:
            values["messages"] = messages_to_dict(values["messages"])
        state_list[0] = values
        return state_list
    except Exception as exc:
        print(f"[get_chat_state] error: {exc}")
        return None
