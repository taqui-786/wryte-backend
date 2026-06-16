from typing import Any
from langchain_core.messages import AIMessage
from langgraph.types import Send

from app.workflow.action import ACTION_REGISTRY
from app.workflow.state import ChatState


def get_ready_steps(plan: list[dict], completed_ids: set[str]) -> list[dict]:
    """Return steps ready to execute (deps satisfied, not running/done)."""
    ready = []
    for step in plan:
        if step["status"] != "pending":
            continue
        deps = step.get("depends_on", [])
        if all(dep in completed_ids for dep in deps):
            ready.append(step)
    return ready


async def step_dispatcher_node(state: ChatState) -> dict:
    plan = state.get("plan", [])
    if not plan:
        return {"messages": [AIMessage(content="No plan to dispatch.")]}

    completed = set(state.get("completed_step_ids", []))
    ready = get_ready_steps(plan, completed)

    if not ready:
        # No ready steps — either all done or waiting for deps
        if len(completed) >= len([s for s in plan if s["status"] != "failed"]):
            return {"plan":plan, "dispatch_ready":False, "dispatch_done":True, "messages": [AIMessage(content="All steps completed.")]}
        return {"plan":plan, "dispatch_ready":False, "dispatch_waiting":True, "messages": [AIMessage(content="Waiting for dependencies...")]}

    # Mark steps as running
    updated_plan = []
    executing_ids = []
    ready_step_ids = [r["id"] for r in ready]
    for step in plan:
        if step["id"] in ready_step_ids:
            updated_plan.append({**step, "status": "running"})
            executing_ids.append(step["id"])
        else:
            updated_plan.append(step)

    

    return {
        "plan":updated_plan,
        "current_executing_ids":executing_ids,
        "dispatch_ready":True,
        "dispatch_ready_step_ids":ready_step_ids,
        "messages": [AIMessage(content=f"Dispatching {len(ready)} ready step(s)...")]
    }
