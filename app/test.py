from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA,NVIDIAEmbeddings
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from pydantic import BaseModel

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatNVIDIA(
    # model="z-ai/glm-5.1",
    model="qwen/qwen3.5-122b-a10b",
    api_key=os.getenv("NVIDIA_API_KEY`"),
    temperature=1,
    top_p=0.95,
    max_completion_tokens=1024,
    model_kwargs={
        "enable_thinking": True,
    },
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

# # ── Stream test ───────────────────────────────────────────────────────────────
# print("--- streaming start ---")
# for message_chunk, metadata in graph.stream(
#     {"messages": [HumanMessage(content="Write a short note on debugger")]},
#     stream_mode="messages",
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end="|", flush=True)

# print("\n--- streaming end ---")


def generate_title_for_chat(conversation: str):
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
        chain = prompt | llm | output_parser
        response =  chain.invoke({"conversation": conversation})
        print("response----", response)
        return response
    except Exception as e:
        print(e)
        raise e

chat = """me - Hey how are you
    Ai - i am fine what about y"""
# generate_title_for_chat(chat)
# embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5",api_key=os.getenv("NVIDIA_API_KEY"))
# print(embeddings.embed_query("Hello, how are you?"))

# print(llm.invoke('Hey there'))






