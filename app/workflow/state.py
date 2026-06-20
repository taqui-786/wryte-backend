from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages






class WorkflowState(TypedDict):
    messages:Annotated[list[AnyMessage], add_messages]



@dataclass
class UserContext:
    user_id:str