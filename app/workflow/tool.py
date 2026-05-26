from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from langgraph.prebuilt import ToolNode
from tinyfish import TinyFish

from app.config import settings


client = TinyFish(api_key=settings.TINYFISH_API_KEY)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def search_agent(query: str) -> list:
    """Search the web for information. Always use this tool whenever you need
    current information or real-time data. Give a short query as input (max 400 tokens)."""
    response = client.search.query(query=query, location="US")
    top_urls = [r.url for r in response.results[:3]]
    pages = client.fetch.get_contents(urls=top_urls, format="markdown")
    return pages.results


my_tools = [multiply, search_agent]

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

EMBEDDING_DIMS = 1024
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=settings.NVIDIA_API_KEY,
    truncate="END",
)

llm_with_tool = llm.bind_tools(my_tools)
tool_node = ToolNode(my_tools)



