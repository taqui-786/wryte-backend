from json import dumps
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import hashlib
import uuid
from dataclasses import dataclass
from typing import AsyncGenerator

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage, messages_to_dict
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
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
from langgraph.prebuilt import ToolNode, tools_condition
from tinyfish import TinyFish

load_dotenv()


client = TinyFish(api_key=os.getenv("TINYFISH_API_KEY"))
# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=1,
    top_p=0.95,
    max_completion_tokens=16384,
    model_kwargs={
        "enable_thinking": True,
        "reasoning_budget":3000
    },
)

llm_secondary = ChatNVIDIA(
    model="stepfun-ai/step-3.5-flash",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=1,
    top_p=0.95,
    max_completion_tokens=1024,
)

# Embedding model — 1024-dimensional vectors produced by nvidia/nv-embedqa-e5-v5.
EMBEDDING_DIMS = 1024
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=os.getenv("NVIDIA_API_KEY"),
    truncate="END",
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
# My Tools --- Taqui
# ---------------------------------------------------------------------------
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def search_agent(query: str) -> list:
    """Search the web for information. give a short query as input max 400 token"""
    response = client.search.query(query=query, location="US")
    top_urls = [r.url for r in response.results[:3]]

    # Step 2: Fetch full content
    pages = client.fetch.get_contents(urls=top_urls, format="markdown")
    return pages.results

my_tools = [multiply, search_agent]    
llm_with_tool = llm.bind_tools(my_tools)
# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------

tool_node = ToolNode(my_tools)

async def chat_node(state: ChatState, runtime: Runtime[UserContext]):
    """
    Main chat node.

    1. Retrieves relevant long-term memories for this user from the store.
    2. Injects them into the system prompt so the model is aware.
    3. If the user explicitly asks to remember something, extracts and stores it.
    4. Calls the LLM and returns the response.
    """
    user_email = runtime.context.user_id
    user_id = int(hashlib.md5(user_email.encode()).hexdigest(), 16) % 100000
    memory_namespace = ("memories", str(user_id))

    last_message = state["messages"][-1]
    last_content: str = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )

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
        "Reply in markdown format."
        "Don't be very lengthy in your responses, be concise and to the point."
        "Always do short thinking and then reply you final content"
        "You have access to tools that can help you with certain tasks. Use them when needed."
    )

    # --- Store new memory if user explicitly asks ---
    if "remember" in last_content.lower():
        # Ask the LLM to extract what should be remembered
        extraction_prompt = (
            f'The user said: "{last_content}"\n\n'
            "Extract a single, concise fact to remember about the user from this message. "
            "Return only the fact, nothing else."
        )
        extraction = await llm_with_tool.ainvoke([{"role": "user", "content": extraction_prompt}])
        memory_text = extraction.content.strip()
        if memory_text:
            await runtime.store.aput(
                memory_namespace,
                str(uuid.uuid4()),
                {"data": memory_text},
            )

    # --- Call the model ---
    response = await llm_with_tool.ainvoke(
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
    builder.add_node("tools", tool_node)
    # ----
    builder.add_edge(START, "chat_node")
    builder.add_conditional_edges("chat_node", tools_condition)
    builder.add_edge("tools","chat_node")
    builder.add_edge("chat_node",END)

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
        version="v2",
    ):
        # print(chunk)
        if chunk.get("type") == "messages":
            message_chunk, _metadata = chunk["data"]
            # Only yield if there is actual content
            if message_chunk:
                yield message_chunk.model_dump()
            # if isinstance(message_chunk,AIMessageChunk):
            #     yield {
            #         "content": message_chunk.content,
            #         "reasoning": message_chunk.additional_kwargs.get(
            #             "reasoning_content",
            #             "",
            #         ),
            #         "tool-calls": getattr(message_chunk, "tool_calls", []) or [],
            #     }
            # elif isinstance(message_chunk,ToolMessage):
            #     yield {
            #         "content": message_chunk.content,
            #         "name": message_chunk.name,
            #     }

async def get_chat_state(
    workflow: CompiledStateGraph,
    thread_id: str,
):
    """Return the persisted state for a thread, with messages serialised to dicts."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await workflow.aget_state(config=config)
        if snapshot is None:
            return None

        values = dict(snapshot.values)
        if "messages" in values:
            values["messages"] = messages_to_dict(values["messages"])
        return values
    except Exception as exc:
        print(f"[get_chat_state] error: {exc}")
        return None


async def generate_title_for_chat(conversation: str):
    try:
        prompt = ChatPromptTemplate.from_template(
            """
You are an AI assistant that generates concise chat titles for a writing editor application.

Your task is to create a short, natural, and meaningful title based on:
1. The user's first message
2. The assistant's initial response

The title should summarize the main intent or topic of the conversation, similar to how ChatGPT names chats.

Rules:
- Maximum 5 words
- Clear and human-friendly
- Do not use quotes
- Avoid generic titles like "New Chat" or "Conversation"
- Avoid unnecessary filler words
- Return only the title

Conversation:
{conversation}

Title:
"""
        )
        output_parser = StrOutputParser()
        chain = prompt | llm_secondary | output_parser
        response = await chain.ainvoke({"conversation": conversation})
        print("response----", response)
        return response
    except Exception as e:
        print(e)
        raise e
