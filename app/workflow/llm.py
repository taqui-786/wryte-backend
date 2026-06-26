
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from app.config import settings


llm = ChatNVIDIA(
    model="stepfun-ai/step-3.5-flash",
    api_key=settings.NVIDIA_API_KEY,
    temperature=1,
    top_p=0.95,
    max_completion_tokens=16384,
    model_kwargs={"enable_thinking": True, "reasoning_budget": 3000},
)

llm_powerfull = ChatNVIDIA(
    model="qwen/qwen3.5-397b-a17b",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    top_p=0.95,
    max_completion_tokens=16384,
    # model_kwargs={"enable_thinking": False},
)

llm_secondary = ChatNVIDIA(
    model="stepfun-ai/step-3.5-flash",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0.3,
    max_completion_tokens=8192,
    model_kwargs={"enable_thinking": False},
)
llm_structure = ChatNVIDIA(
    model="openai/gpt-oss-120b",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_completion_tokens=8192,
)

EMBEDDING_DIMS = 1024
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=settings.NVIDIA_API_KEY,
    truncate="END",
)

