from typing import Literal
from langgraph.types import Command
from app.workflow.state import ChatState


async def step_worker_node(state: ChatState) -> Command[Literal["research_topics", "writer_planning_node", "step_complete"]]:
    step_id = state.get("current_step_id")
    plan = state.get("plan", [])
    
    # Find this step
    step = next((s for s in plan if s["id"] == step_id), None)
    if not step:
        return Command(
            update={"step_error": "Step not found"},
            goto="step_complete")
    
    action = step["action"]
    params = step.get("params", {})

    if action == "deep_research":
          return Command(
              update={
                  "topic": params.get("topic", ""),
                  "research_requested": True,
                  "writer_requested": False,
                  "current_step_id": step_id,
              },
              goto="research_topics",
          )

    if action == "search":
          return Command(
              update={
                  "topic": params.get("query", ""),
                  "research_requested": True,
                  "writer_requested": False,
                  "current_step_id": step_id,
              },
              goto="research_topics",
          )

    if action == "write_blog":
          return Command(
              update={
                  "writer_topic": params.get("topic", ""),
                  "writer_requested": True,
                  "research_requested": False,
                  "current_step_id": step_id,
              },
              goto="writer_planning_node",
          )

    return Command(
        update={
            "step_error": f"Unknown action: {action}",
            "current_step_id": step_id,
        },
        goto="step_complete",
      )
