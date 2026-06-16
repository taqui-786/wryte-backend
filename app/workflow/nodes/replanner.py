from langchain_core.messages import AIMessage
from app.workflow.state import ChatState
from app.workflow.tool import Plan, llm_powerfull


REPLANNER_PROMPT = """You are a replanner. The original plan hit an issue.

ORIGINAL GOAL: {goal}
COMPLETED STEPS: {completed}
FAILED STEPS: {failed}
ERROR CONTEXT: {error}

Create a REVISED plan to achieve the goal. You may:
- Retry failed steps with different params
- Add new steps to work around failures
- Skip failed steps if goal still achievable
- Change approach entirely

Return a new Plan object.
"""

async def replanner_node(state: ChatState) -> dict:
    plan = state.get("plan", [])
    completed_ids = set(state.get("completed_step_ids", []))
    failed_steps = [s for s in plan if s["status"] == "failed"]
    completed_steps = [s for s in plan if s["id"] in completed_ids]
    
    goal = plan[0].get("goal", "") if plan else ""
    completed_str = "\n".join([f"- {s['description']}: {s['result'][:200]}" for s in completed_steps])
    failed_str = "\n".join([f"- {s['description']}: ERROR: {s['error']}" for s in failed_steps])
    error = state.get("replan_reason", "")
    
    replanner_llm = llm_powerfull.with_structured_output(Plan)
    result: Plan = await replanner_llm.ainvoke([
        {"role": "system", "content": REPLANNER_PROMPT.format(
            goal=goal, completed=completed_str, failed=failed_str, error=error
        )},
        {"role": "user", "content": "Create revised plan."},
    ])
    
    new_plan_dicts = [step.model_dump() for step in result.steps]
    
    return {
        "plan": new_plan_dicts,
        "completed_step_ids": [],  # Reset — new plan
        "current_executing_ids": [],
        "needs_replan": False,
        "replan_reason": None,
        "messages": [AIMessage(content=f"Replanned: {len(new_plan_dicts)} steps. Executing...")],
    }