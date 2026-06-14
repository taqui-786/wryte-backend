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


async def step_dispatcher_node(state: ChatState) -> list[Send] | dict:
    plan = state.get("plan", [])
    if not plan:
        return {}

    completed = set(state.get("completed_step_ids", []))
    ready = get_ready_steps(plan, completed)

    if not ready:
        # No ready steps — either all done or waiting for deps
        if len(completed) >= len([s for s in plan if s["status"] != "failed"]):
            return {}  # Plan complete
        return {}  # Waiting for running steps to complete

    # Mark steps as running
    updated_plan = []
    executing_ids = []
    for step in plan:
        if step["id"] in [r["id"] for r in ready]:
            updated_plan.append({**step, "status": "running"})
            executing_ids.append(step["id"])
        else:
            updated_plan.append(step)

    # Create Send for each ready step
    sends = []
    for step in ready:
        # Build sub-state for this step
        step_state = {
            **state,
            "plan": updated_plan,
            "current_executing_ids": executing_ids,
        }

        action_spec = ACTION_REGISTRY.get(step["action"])
        if action_spec and action_spec.state_setup:
            step_state.update(action_spec.state_setup(state, step["params"]))

        sends.append(Send("step_worker", {**step_state, "current_step_id": step["id"]}))

    return sends
