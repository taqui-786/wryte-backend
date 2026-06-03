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


@dataclass
class UserContext:
    user_id: str
