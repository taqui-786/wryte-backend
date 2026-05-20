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
    model="mistralai/mixtral-8x22b-instruct-v0.1",
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



{'type': 'messages', 'ns': (), 'data': (AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='lc_run--019e441f-e91a-7b63-aca3-e3dd9d438ce8', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[], role='assistant'), {'thread_id': '1234567890', 'ls_integration': 'langchain_chat_model', 'langgraph_step': 17, 'langgraph_node': 'chat_node', 'langgraph_triggers': ('branch:to:chat_node',), 'langgraph_path': ('__pregel_pull', 'chat_node'), 'langgraph_checkpoint_ns': 'chat_node:c872c0ad-dd9b-910b-062d-b3475b083e0c', 'checkpoint_ns': 'chat_node:c872c0ad-dd9b-910b-062d-b3475b083e0c', 'ls_provider': 'NVIDIA', 'ls_model_name': 'nvidia/nemotron-3-super-120b-a12b', 'ls_model_type': 'chat', 'ls_temperature': 1.0, 'ls_max_tokens': 16384, 'ls_stop': None})}
{'type': 'messages', 'ns': (), 'data': (AIMessageChunk(content='', additional_kwargs={'reasoning_content': 'We need to use multiply tool. The user asked: "multiply two number 12 and 4". So compute 12*4=48. We\'ll call function.\n', 'reasoning': 'We need to use multiply tool. The user asked: "multiply two number 12 and 4". So compute 12*4=48. We\'ll call function.\n', '_reasoning_api_fields': ['reasoning_content', 'reasoning']}, response_metadata={}, id='lc_run--019e441f-e91a-7b63-aca3-e3dd9d438ce8', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[], role='assistant'), {'thread_id': '1234567890', 'ls_integration': 'langchain_chat_model', 'langgraph_step': 17, 'langgraph_node': 'chat_node', 'langgraph_triggers': ('branch:to:chat_node',), 'langgraph_path': ('__pregel_pull', 'chat_node'), 'langgraph_checkpoint_ns': 'chat_node:c872c0ad-dd9b-910b-062d-b3475b083e0c', 'checkpoint_ns': 'chat_node:c872c0ad-dd9b-910b-062d-b3475b083e0c', 'ls_provider': 'NVIDIA', 'ls_model_name': 'nvidia/nemotron-3-super-120b-a12b', 'ls_model_type': 'chat', 'ls_temperature': 1.0, 'ls_max_tokens': 16384, 'ls_stop': None})}
{'type': 'messages', 'ns': (), 'data': (AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='lc_run--019e441f-e91a-7b63-aca3-e3dd9d438ce8', tool_calls=[{'name': 'multiply', 'args': {'a': 12, 'b': 4}, 'id': 'chatcmpl-tool-985d5cdb1a31b516', 'type': 'tool_call'}], invalid_tool_calls=[], tool_call_chunks=[{'name': 'multiply', 'args': '{"a": 12, "b": 4}', 'id': 'chatcmpl-tool-985d5cdb1a31b516', 'index': 0, 'type': 'tool_call_chunk'}], role='assistant'), {'thread_id': '1234567890', 'ls_integration': 'langchain_chat_model', 'langgraph_step': 17, 'langgraph_node': 'chat_node', 'langgraph_triggers': ('branch:to:chat_node',), 'langgraph_path': ('__pregel_pull', 'chat_node'), 'langgraph_checkpoint_ns': 'chat_node:c872c0ad-dd9b-910b-062d-b3475b083e0c', 'checkpoint_ns': 'chat_node:c872c0ad-dd9b-910b-062d-b3475b083e0c', 'ls_provider': 'NVIDIA', 'ls_model_name': 'nvidia/nemotron-3-super-120b-a12b', 'ls_model_type': 'chat', 'ls_temperature': 1.0, 'ls_max_tokens': 16384, 'ls_stop': None})}
{'type': 'messages', 'ns': (), 'data': (AIMessageChunk(content='', additional_kwargs={}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'nvidia/nemotron-3-super-120b-a12b'}, id='lc_run--019e441f-e91a-7b63-aca3-e3dd9d438ce8', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[], role='assistant'), {'thread_id': '1234567890', 'ls_integration': 'langchain_chat_model', 'langgraph_step': 17, 'langgraph_node': 'chat_node', 'langgraph_triggers': ('branch:to:chat_node',), 'langgraph_path': ('__pregel_pull', 'chat_node'), 'langgraph_checkpoint_ns': 'chat_node:c872c0ad-dd9b-910b-062d-b3475b083e0c', 'checkpoint_ns': 'chat_node:c872c0ad-dd9b-910b-062d-b3475b083e0c', 'ls_provider': 'NVIDIA', 'ls_model_name': 'nvidia/nemotron-3-super-120b-a12b', 'ls_model_type': 'chat', 'ls_temperature': 1.0, 'ls_max_tokens': 16384, 'ls_stop': None})}
{'type': 'messages', 'ns': (), 'data': (AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='lc_run--019e441f-e91a-7b63-aca3-e3dd9d438ce8', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 550, 'output_tokens': 74, 'total_tokens': 624}, tool_call_chunks=[], role='assistant'), {'thread_id': '1234567890', 'ls_integration': 'langchain_chat_model', 'langgraph_step': 17, 'langgraph_node': 'chat_node', 'langgraph_triggers': ('branch:to:chat_node',), 'langgraph_path': ('__pregel_pull', 'chat_node'), 'langgraph_checkpoint_ns': 'chat_node:c872c0ad-dd9b-910b-062d-b3475b083e0c', 'checkpoint_ns': 'chat_node:c872c0ad-dd9b-910b-062d-b3475b083e0c', 'ls_provider': 'NVIDIA', 'ls_model_name': 'nvidia/nemotron-3-super-120b-a12b', 'ls_model_type': 'chat', 'ls_temperature': 1.0, 'ls_max_tokens': 16384, 'ls_stop': None})}
{'type': 'messages', 'ns': (), 'data': (AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='lc_run--019e441f-e91a-7b63-aca3-e3dd9d438ce8', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[], chunk_position='last'), {'thread_id': '1234567890', 'ls_integration': 'langchain_chat_model', 'langgraph_step': 17, 'langgraph_node': 'chat_node', 'langgraph_triggers': ('branch:to:chat_node',), 'langgraph_path': ('__pregel_pull', 'chat_node'), 'langgraph_checkpoint_ns': 'chat_node:c872c0ad-dd9b-910b-062d-b3475b083e0c', 'checkpoint_ns': 'chat_node:c872c0ad-dd9b-910b-062d-b3475b083e0c', 'ls_provider': 'NVIDIA', 'ls_model_name': 'nvidia/nemotron-3-super-120b-a12b', 'ls_model_type': 'chat', 'ls_temperature': 1.0, 'ls_max_tokens': 16384, 'ls_stop': None})}
{'type': 'messages', 'ns': (), 'data': (ToolMessage(content='48', name='multiply', id='31434e2e-426c-4fb3-809c-c980e22a972c', tool_call_id='chatcmpl-tool-985d5cdb1a31b516'), {'thread_id': '1234567890', 'ls_integration': 'langgraph', 'langgraph_step': 18, 'langgraph_node': 'tools', 'langgraph_triggers': ('branch:to:tools',), 'langgraph_path': ('__pregel_pull', 'tools'), 'langgraph_checkpoint_ns': 'tools:03ac6350-e6be-2731-00dd-dd338413f452'})}
{'type': 'messages', 'ns': (), 'data': (AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='lc_run--019e441f-f62e-7d90-8476-b74fae770560', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[], role='assistant'), {'thread_id': '1234567890', 'ls_integration': 'langchain_chat_model', 'langgraph_step': 19, 'langgraph_node': 'chat_node', 'langgraph_triggers': ('branch:to:chat_node',), 'langgraph_path': ('__pregel_pull', 'chat_node'), 'langgraph_checkpoint_ns': 'chat_node:9e7db8c8-da99-a5ca-228d-89311ee69ec0', 'checkpoint_ns': 'chat_node:9e7db8c8-da99-a5ca-228d-89311ee69ec0', 'ls_provider': 'NVIDIA', 'ls_model_name': 'nvidia/nemotron-3-super-120b-a12b', 'ls_model_type': 'chat', 'ls_temperature': 1.0, 'ls_max_tokens': 16384, 'ls_stop': None})}
{'type': 'messages', 'ns': (), 'data': (AIMessageChunk(content='', additional_kwargs={'reasoning_content': 'We have the result: 48. Need to output concise answer.\n', 'reasoning': 'We have the result: 48. Need to output concise answer.\n', '_reasoning_api_fields': ['reasoning_content', 'reasoning']}, response_metadata={}, id='lc_run--019e441f-f62e-7d90-8476-b74fae770560', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[], role='assistant'), {'thread_id': '1234567890', 'ls_integration': 'langchain_chat_model', 'langgraph_step': 19, 'langgraph_node': 'chat_node', 'langgraph_triggers': ('branch:to:chat_node',), 'langgraph_path': ('__pregel_pull', 'chat_node'), 'langgraph_checkpoint_ns': 'chat_node:9e7db8c8-da99-a5ca-228d-89311ee69ec0', 'checkpoint_ns': 'chat_node:9e7db8c8-da99-a5ca-228d-89311ee69ec0', 'ls_provider': 'NVIDIA', 'ls_model_name': 'nvidia/nemotron-3-super-120b-a12b', 'ls_model_type': 'chat', 'ls_temperature': 1.0, 'ls_max_tokens': 16384, 'ls_stop': None})}
{'type': 'messages', 'ns': (), 'data': (AIMessageChunk(content='48', additional_kwargs={}, response_metadata={}, id='lc_run--019e441f-f62e-7d90-8476-b74fae770560', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[], role='assistant'), {'thread_id': '1234567890', 'ls_integration': 'langchain_chat_model', 'langgraph_step': 19, 'langgraph_node': 'chat_node', 'langgraph_triggers': ('branch:to:chat_node',), 'langgraph_path': ('__pregel_pull', 'chat_node'), 'langgraph_checkpoint_ns': 'chat_node:9e7db8c8-da99-a5ca-228d-89311ee69ec0', 'checkpoint_ns': 'chat_node:9e7db8c8-da99-a5ca-228d-89311ee69ec0', 'ls_provider': 'NVIDIA', 'ls_model_name': 'nvidia/nemotron-3-super-120b-a12b', 'ls_model_type': 'chat', 'ls_temperature': 1.0, 'ls_max_tokens': 16384, 'ls_stop': None})}
{'type': 'messages', 'ns': (), 'data': (AIMessageChunk(content='', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'nvidia/nemotron-3-super-120b-a12b'}, id='lc_run--019e441f-f62e-7d90-8476-b74fae770560', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[], role='assistant'), {'thread_id': '1234567890', 'ls_integration': 'langchain_chat_model', 'langgraph_step': 19, 'langgraph_node': 'chat_node', 'langgraph_triggers': ('branch:to:chat_node',), 'langgraph_path': ('__pregel_pull', 'chat_node'), 'langgraph_checkpoint_ns': 'chat_node:9e7db8c8-da99-a5ca-228d-89311ee69ec0', 'checkpoint_ns': 'chat_node:9e7db8c8-da99-a5ca-228d-89311ee69ec0', 'ls_provider': 'NVIDIA', 'ls_model_name': 'nvidia/nemotron-3-super-120b-a12b', 'ls_model_type': 'chat', 'ls_temperature': 1.0, 'ls_max_tokens': 16384, 'ls_stop': None})}
{'type': 'messages', 'ns': (), 'data': (AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='lc_run--019e441f-f62e-7d90-8476-b74fae770560', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 610, 'output_tokens': 20, 'total_tokens': 630}, tool_call_chunks=[], role='assistant'), {'thread_id': '1234567890', 'ls_integration': 'langchain_chat_model', 'langgraph_step': 19, 'langgraph_node': 'chat_node', 'langgraph_triggers': ('branch:to:chat_node',), 'langgraph_path': ('__pregel_pull', 'chat_node'), 'langgraph_checkpoint_ns': 'chat_node:9e7db8c8-da99-a5ca-228d-89311ee69ec0', 'checkpoint_ns': 'chat_node:9e7db8c8-da99-a5ca-228d-89311ee69ec0', 'ls_provider': 'NVIDIA', 'ls_model_name': 'nvidia/nemotron-3-super-120b-a12b', 'ls_model_type': 'chat', 'ls_temperature': 1.0, 'ls_max_tokens': 16384, 'ls_stop': None})}
{'type': 'messages', 'ns': (), 'data': (AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='lc_run--019e441f-f62e-7d90-8476-b74fae770560', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[], chunk_position='last'), {'thread_id': '1234567890', 'ls_integration': 'langchain_chat_model', 'langgraph_step': 19, 'langgraph_node': 'chat_node', 'langgraph_triggers': ('branch:to:chat_node',), 'langgraph_path': ('__pregel_pull', 'chat_node'), 'langgraph_checkpoint_ns': 'chat_node:9e7db8c8-da99-a5ca-228d-89311ee69ec0', 'checkpoint_ns': 'chat_node:9e7db8c8-da99-a5ca-228d-89311ee69ec0', 'ls_provider': 'NVIDIA', 'ls_model_name': 'nvidia/nemotron-3-super-120b-a12b', 'ls_model_type': 'chat', 'ls_temperature': 1.0, 'ls_max_tokens': 16384, 'ls_stop': None})}







