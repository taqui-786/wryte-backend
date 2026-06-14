from app.workflow.state import ChatState


async def step_worker_node(state: ChatState) -> dict:
    """Execute a single plan step. Returns state updates for that step."""
    step_id = state.get("current_step_id")
    plan = state.get("plan", [])
    
    # Find this step
    step = next((s for s in plan if s["id"] == step_id), None)
    if not step:
        return {}
    
    action = step["action"]
    updates = {"plan": plan}  # Will be modified
    
    # The actual execution is delegated to existing subgraphs via flags
    # This node just sets up the state and lets the graph route
    if action == "deep_research":
        # Already set by dispatcher: topic, research_requested
        pass
    elif action == "write_blog":
        # Already set: writer_topic, writer_requested
        pass
    elif action == "search":
        # Quick search - could use lighter research or direct tool
        pass
    # ... other actions
    
    # The subgraphs (research, writer) will run and eventually hit step_complete
    # This worker just waits for them to finish
    return updates