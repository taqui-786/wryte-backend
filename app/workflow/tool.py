from typing import Annotated
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langgraph.prebuilt import ToolNode, InjectedState
from tinyfish import TinyFish

from app.config import settings
from app.workflow.state import ChatState


client = TinyFish(api_key=settings.TINYFISH_API_KEY)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def read_editor(
    state: Annotated[ChatState, InjectedState],
) -> str:
    """Read the full current content of the user's markdown editor.

    Use this whenever you need to see what the user is currently writing
    before you can help them (summarize, edit, review, continue, etc.).
    Returns the editor content as a single markdown string. If the editor
    is empty, returns an empty string.
    """
    content = state.get("editor_content", "") or ""
    if not content.strip():
        return "The editor is currently empty."
    return content


@tool
def search_agent(query: str) -> list:
    """Search the web for information. Always use this tool whenever you need
    current information or real-time data. Give a short query as input (max 400 tokens)."""
    response = client.search.query(query=query, location="US")
    top_urls = [r.url for r in response.results[:3]]
    pages = client.fetch.get_contents(urls=top_urls, format="markdown")
    return pages.results


my_tools = [multiply, search_agent, read_editor]

llm = ChatNVIDIA(
    # model="nvidia/nemotron-3-super-120b-a12b",
    model="stepfun-ai/step-3.5-flash",
    api_key=settings.NVIDIA_API_KEY,
    temperature=1,
    top_p=0.95,
    max_completion_tokens=16384,
    model_kwargs={"enable_thinking": True, "reasoning_budget": 3000},
)

llm_secondary = ChatNVIDIA(
    model="stepfun-ai/step-3.5-flash",
    api_key=settings.NVIDIA_API_KEY,
    temperature=1,
    top_p=0.95,
    max_completion_tokens=1024,
)
llm_classifier = ChatNVIDIA(
    model="nvidia/llama-3.3-nemotron-super-49b-v1",  # 8B params is plenty for classification
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,   # 0 = deterministic, no randomness. We want consistent YES/NO.
    max_completion_tokens=100,  # only need a short JSON
)

# A small, fast model for extracting the actual fact
llm_extractor = ChatNVIDIA(
    model="nvidia/llama-3.3-nemotron-super-49b-v1",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_completion_tokens=200,  # a fact is short
)

EMBEDDING_DIMS = 1024
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=settings.NVIDIA_API_KEY,
    truncate="END",
)

llm_with_tool = llm.bind_tools(my_tools)
tool_node = ToolNode(my_tools)



