
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime
from app.workflow.llm import llm
from app.workflow.state import UserContext, WorkflowState


SYSTEM_PROMPT_TEMPLATE = """\
You are Wryte - a writing assistant built into a markdown editor.
You do NOT chat generically. You help the user write, edit, refine,
and research content inside their editor. Everything the user refers
to is about what's in that editor. If you are confuse just stop and Ask user to be more specific.

The user's content lives in the editor. That is your workspace.

YOUR IDENTITY:
- You are an extension of the editor, not a standalone chatbot.
- "Read", "check", "show", "review", "summarize", "what do I have"
  → means use `read_editor`.
- "Write", "update", "fix", "draft", "improve", "rewrite", "change"
  → means use `update_editor`.
- Never guess or invent editor content. Always read it first.


When relevant, draw on these memories about the user's writing style:
{memory_context}
"""


async def chat_node(state: WorkflowState, runtime: Runtime[UserContext]):
    try:
        memory_namespace = ("memories", runtime.context.user_id)
        memories = await runtime.store.asearch(
            (memory_namespace,), query=None, limit=10
        )
        if memories:
            memory_context = "\n".join(f"- {m.value['data']}" for m in memories)
        else:
            memory_context = "No Memories yet"
    except Exception:
        memory_context = "No Memories yet"

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(memory_context=memory_context)
    messages = state["messages"]
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        *messages,
    ])
    return {"messages": [response]}
    