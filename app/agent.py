from langgraph.graph.message import add_messages
from pydantic import BaseModel
from langgraph.constants import END
from langgraph.constants import START
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pprint import pprint
import os
from langchain_core.messages import messages_to_dict

load_dotenv()


llm = ChatNVIDIA(
    # model="stepfun-ai/step-3.5-flash",
    model="qwen/qwen3.5-122b-a10b",
    api_key=os.getenv("NVIDIA_API_KEY`"),
    temperature=1,
    top_p=0.95,
    max_completion_tokens=16384,  # ✅ renamed
    model_kwargs={
        "enable_thinking": True,
        "reasoning_budget": 16384,  # ✅ moved here
    },
)


class chat_state(TypedDict):
    messages: Annotated[list[BaseModel], add_messages]


def chat_node(state: chat_state):
    messages = state["messages"]
    res = llm.invoke(messages)
    return {"messages": [res]}


graph = StateGraph(chat_state)


graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

my_checkpointer = MemorySaver()
workflow = graph.compile(checkpointer=my_checkpointer)


def my_agent(user_input: str):
    config = {"configurable": {"thread_id": "786"}}
    # stream_mode="messages" yields (AIMessageChunk, metadata) tuples directly
    # — same as the JS example: for await (const [messageChunk, metadata] of graph.stream(...))
    for chunk in workflow.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
        version="v2",
    ):
        # print("chunk ", chunk)
        if chunk["type"] == "messages":
            message_chunk, metadata = chunk["data"]
            if message_chunk.content:
                # print(message_chunk.content, end="|", flush=True)
                yield message_chunk.content
            if hasattr(message_chunk, "additional_kwargs"):
                reasoning = message_chunk.additional_kwargs.get("reasoning")
                if reasoning:
                    print("REASONING:", reasoning)


def get_chat_state(thread_id: str):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        # stream_mode="messages" yields (AIMessageChunk, metadata) tuples directly
        # — same as the JS example: for await (const [messageChunk, metadata] of graph.stream(...))
        res = workflow.get_state(config=config)
        
        if res:
            state_list = list(res)
            values = dict(state_list[0])
            if "messages" in values:
                values["messages"] = messages_to_dict(values["messages"])
            state_list[0] = values
            return state_list
            
        return res
    except Exception as e:
        print(e)
        return None
