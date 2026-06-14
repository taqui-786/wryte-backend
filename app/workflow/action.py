from typing import Callable, Any
from pydantic import BaseModel

class ActionSpec(BaseModel):
    name: str
    description: str
    param_schema: dict  # JSON schema for params
    requires_subgraph: bool  # True if uses research/writer subgraph
    subgraph_entry: str | None = None  # Node to route to (e.g., "research_topics")
    state_setup: Callable[[dict, dict], dict]  # (state, params) -> state_updates

# Registry
ACTION_REGISTRY: dict[str, ActionSpec] = {}

def register_action(spec: ActionSpec):
    ACTION_REGISTRY[spec.name] = spec

# Built-in actions
def setup_deep_research(state: dict, params: dict) -> dict:
    return {"topic": params.get("topic", ""), "research_requested": True}

def setup_write_blog(state: dict, params: dict) -> dict:
    # Context building happens in dispatcher
    return {"writer_topic": params.get("topic", ""), "writer_requested": True}

def setup_search(state: dict, params: dict) -> dict:
    return {"topic": params.get("query", ""), "research_requested": True}

register_action(ActionSpec(
    name="deep_research",
    description="Comprehensive multi-source research",
    param_schema={"topic": {"type": "string"}},
    requires_subgraph=True,
    subgraph_entry="research_topics",
    state_setup=setup_deep_research,
))

register_action(ActionSpec(
    name="write_blog",
    description="Write a blog post/article",
    param_schema={"topic": {"type": "string"}},
    requires_subgraph=True,
    subgraph_entry="planning_node",
    state_setup=setup_write_blog,
))

register_action(ActionSpec(
    name="search",
    description="Quick web search for a fact",
    param_schema={"query": {"type": "string"}},
    requires_subgraph=True,
    subgraph_entry="research_topics",
    state_setup=setup_search,
))

# Add more: scrape_url, edit_content, code, analyze, etc.