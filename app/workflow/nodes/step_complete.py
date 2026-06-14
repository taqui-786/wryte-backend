from app.workflow.state import ChatState


async def step_complete_node(state: ChatState) -> dict:
    plan = list(state.get("plan", []) or [])
    step_id = state.get("current_step_id")
    completed = set(state.get("completed_step_ids", []))
    
    if not step_id:
        return {"research_requested": False, "writer_requested": False}
    
    # Find and update the step
    step = next((s for s in plan if s["id"] == step_id), None)
    if not step:
        return {}
    
    # Extract result based on action
    result = ""
    if step["action"] == "deep_research":
        result = state.get("final_research_report", "") or ""
    elif step["action"] == "write_blog":
        writer_out = state.get("writer_output", {}) or {}
        result = writer_out.get("humanized", "") or writer_out.get("draft", "") or ""
    elif step["action"] == "search":
        result = state.get("final_research_report", "") or ""
    
    # Determine status
    status = "done" if result else "failed"
    error = "No result produced" if not result else None
    
    # Update plan
    updated_plan = []
    for s in plan:
        if s["id"] == step_id:
            updated_plan.append({**s, "status": status, "result": result, "error": error})
        else:
            updated_plan.append(s)
    
    # Track completion
    new_completed = completed | {step_id}
    
    # Check if any step failed and needs replan
    failed_steps = [s for s in updated_plan if s["status"] == "failed"]
    needs_replan = len(failed_steps) > 0
    replan_reason = f"Steps failed: {[s['description'] for s in failed_steps]}" if needs_replan else None
    
    return {
        "plan": updated_plan,
        "completed_step_ids": list(new_completed),  
        "current_executing_ids": [sid for sid in state.get("current_executing_ids", []) if sid != step_id],
        "needs_replan": needs_replan,
        "replan_reason": replan_reason,
        "research_requested": False,
        "writer_requested": False,
        "step_results": [f"Step {step['step']} ({step['action']}): {result[:200]}"],
    }