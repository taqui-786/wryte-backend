import time

from langchain_core.messages import AIMessage

from app.workflow.state import ChatState
from app.workflow.tool import Plan, llm_powerfull

PLANNER_PROMPT = """You are a strategic planner for a writing assistant.

AVAILABLE ACTIONS:
- deep_research: Comprehensive multi-source research on a topic
- write_blog: Write a blog post/article (needs research context if topic requires facts)
- search: Quick web search for a specific fact
- scrape_url: Fetch full content from a single URL
- edit_content: Edit existing editor content (read_editor → update_editor)

RULES:
1. Break complex requests into MINIMAL steps (max 6)
2. Use depends_on to declare dependencies (enables parallel execution)
3. For "research X and write about it": research FIRST, then write (write depends_on research)
4. For independent research topics: run in PARALLEL (no depends_on between them)
5. Each step must have clear description and params

Return ONLY a Plan object.

Example for "Research AI agents and write a blog post":
{
  "goal": "Research AI agents and write a blog post",
  "steps": [
    {"id": "s1", "step": 1, "action": "deep_research", "params": {"topic": "AI agents"}, "description": "Research AI agents comprehensively", "depends_on": []},
    {"id": "s2", "step": 2, "action": "write_blog", "params": {"topic": "AI agents blog post"}, "description": "Write blog post on AI agents", "depends_on": ["s1"]}
  ]
}
"""


async def planner_node(state: ChatState) -> dict:
    last_usermessage = state["messages"][-1]
    content = last_usermessage.content if hasattr(last_usermessage, "content") else ""
    planner_llm = llm_powerfull.with_structured_output(Plan)
    suggested = state.get("suggested_actions", [])
    context_hint = f"\nsuggested actions: {suggested}" if suggested else ""
    result: Plan = await planner_llm.ainvoke(
        [
            {"role": "system", "content": PLANNER_PROMPT + context_hint},
            {"role": "user", "content": content},
        ]
    )
    plan_dicts = [step.model_dump() for step in result.steps]
    
    return {
        "plan": plan_dicts,
        "completed_step_ids": [],
        "current_executing_ids": [],
        "needs_replan": False,
        "replan_reason": None,
        "messages": [AIMessage(content=f"Created plan with {len(plan_dicts)} steps. Executing...")],
    }
