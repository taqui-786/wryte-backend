from operator import add
from typing import Annotated, Literal, TypedDict
from dataclasses import dataclass
from pydantic import BaseModel
from langgraph.graph.message import add_messages

from app.workflow.tool import PlanStep


class ChatState(TypedDict):
    messages: Annotated[list[BaseModel], add_messages]
    memories: list[str]
    should_remember: bool
    memory_to_save: str | None
    editor_content: str
    editor_changes: list[dict]

    # Research agent
    topic: str | None
    research_requested: bool
    research_topics: list[str] | None
    research_results: Annotated[list[str], add]
    final_research_report: str

    # Writer agent
    writer_requested: bool
    writer_topic: str | None
    writer_output: dict | None
    writer_iteration: int

    # Planning (DAG)
    current_step_id: str | None
    task_type: Literal["chat", "tool", "plan"] | None
    suggested_actions: list[str] | None
    plan: list[PlanStep] | None
    completed_step_ids: Annotated[
        list[str], add
    ]  # Track completed for dependency check
    current_executing_ids: list[str]  # Steps currently running
    needs_replan: bool
    replan_reason: str | None
    dispatch_ready: bool
    dispatch_done: bool
    dispatch_waiting: bool
    dispatch_ready_step_ids: list[str]
    step_error: str | None
    step_results: Annotated[list[str], add]


@dataclass
class UserContext:
    user_id: str
