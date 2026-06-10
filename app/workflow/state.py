from operator import add
from typing import Annotated, TypedDict
from dataclasses import dataclass
from pydantic import BaseModel
from langgraph.graph.message import add_messages



class ChatState(TypedDict):
    messages: Annotated[list[BaseModel], add_messages]
    memories:list[str]
    should_remember:bool
    memory_to_save:str | None
    editor_content: str

    # Research agent
    topic: str | None
    research_requested: bool
    research_topics: list[str] | None
    research_results: Annotated[list[str], add]
    final_research_report:str

    # Writer agent
    writer_requested: bool
    writer_topic: str | None
    writer_output: dict | None # Plan - draft - humanize
    writer_iteration: int 


@dataclass
class UserContext:
    user_id: str
