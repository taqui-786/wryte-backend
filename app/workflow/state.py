from typing import Annotated, TypedDict
from dataclasses import dataclass
from pydantic import BaseModel
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages: Annotated[list[BaseModel], add_messages]


@dataclass
class UserContext:
    user_id: str
