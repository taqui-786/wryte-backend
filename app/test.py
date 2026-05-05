
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from pydantic import BaseModel

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatNVIDIA(
    model="stepfun-ai/step-3.5-flash",
    api_key=os.getenv("NVIDIA_API_KEY`"),
    temperature=1,
    top_p=0.95,
    max_completion_tokens=1024,
)

# ── Graph ─────────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseModel], add_messages]

def chat_node(state: State):
    res = llm.invoke(state["messages"])
    return {"messages": [res]}

graph = (
    StateGraph(State)
    .add_node("chat_node", chat_node)
    .add_edge(START, "chat_node")
    .compile()
)

# ── Stream test ───────────────────────────────────────────────────────────────
print("--- streaming start ---")
for message_chunk, metadata in graph.stream(
    {"messages": [HumanMessage(content="Write a short note on debugger")]},
    stream_mode="messages",
):
    if message_chunk.content:
        print(message_chunk.content, end="|", flush=True)

print("\n--- streaming end ---")