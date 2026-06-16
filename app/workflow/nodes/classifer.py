from typing import Literal
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel

from app.workflow.state import ChatState, UserContext
from app.workflow.tool import llm_classifier


class TaskClassification(BaseModel):
    task_type: Literal["chat", "tool", "plan"]
    # reasoning:str
    # suggested_tools:list[str] = []
    suggested_actions:list[str] = [] # e.g.. deep_research, write_blog etc..

CLASSIFIER_PROMPT = """You are a task classifier for a writing assistant.

Classify the user's LAST MESSAGE into ONE category:

1. "chat" - Simple conversation, questions, follow-ups, greetings, clarifications
   Examples: "What's RAG?", "Make the intro shorter", "Thanks", "Continue"

2. "tool" - Single tool use that completes in one step
   Examples: "Search for X", "Scrape this URL", "Read my editor", "Fix typo in line 3"

3. "plan" - Multi-step task requiring coordination (2+ steps, or needs research+write)
   Examples: "Research X and write a blog", "Compare A vs B and summarize", 
             "Deep research on topic", "Write article with research"

Return ONLY the classification with brief reasoning.
"""

async def classifier_node(state:ChatState,runtime:Runtime[UserContext]) -> dict:
    last_usermessage = state["messages"][-1]
    content = last_usermessage.content if hasattr(last_usermessage,"content") else ""
    content_lower = content.lower().strip()
    if len(content_lower) < 10 or content_lower in {"hi", "hello", "thanks", "ok", "continue"}:
        return {"task_type": "chat"}
    llm_classifier_structured = llm_classifier.with_structured_output(TaskClassification)
    result: TaskClassification = await llm_classifier_structured.ainvoke([
        {"role":"system","content":CLASSIFIER_PROMPT},
        {"role":"user","content":content}
    ])
    return {
        "task_type":result.task_type,
        "suggested_actions":result.suggested_actions,
        "plan":None,
        "messages": [AIMessage(content=f"Classified task as: {result.task_type}")]
    }
