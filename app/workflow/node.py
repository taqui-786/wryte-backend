import asyncio
import uuid
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel
from app.workflow.state import ChatState, UserContext
from app.workflow.tool import (
    MARKDOWN_RULES,
    RememberDecision,
    llm_classifier_remeber_structured,
    llm_extractor,
    llm_with_tool,
)


# Classifier Node - Just to determine the msg need to be saved Yes / No
class MemoryDecision(BaseModel):
    should_remember: bool
    reason: str


async def classify_node(state: ChatState, runtime: Runtime[UserContext]):
    last_message = state["messages"][-1]
    last_content = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )
    
    if len(last_content) < 5:
        return {"should_remember": False}
    decision: RememberDecision = await llm_classifier_remeber_structured.ainvoke(
        f"Does this message contain personal information about the user worth remembering?\n\n"
        f"YES if the user reveals ANYTHING about themselves - facts, projects, work, "
        f"possessions, preferences, habits, life details, or instructions. "
        f"Even if it's casual or followed by a question (e.g. 'I built my portfolio at X, thoughts?' "
        f"contains 'portfolio at X' → YES).\n"
        f"NO for pure questions, greetings, chitchat, or statements about others.\n\n"
        f'User message: "{last_content}"'
    )
    print(decision.should_remember)
    return {"should_remember": decision.should_remember}


# Recall Node - To just fetch memory for us no reading just giving stuff


async def recall_node(state: ChatState, runtime: Runtime[UserContext]):
    user_id = runtime.context.user_id
    memory_namespace = ("memories", user_id)
    last_message = state["messages"][-1]
    last_content = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )
    memories = await runtime.store.asearch(
        memory_namespace, query=last_content, limit=5
    )
    memory_lines = [d.value["data"] for d in memories]
    return {"memories": memory_lines}


SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful, thoughtful writing assistant called Wryte.

You have access to the following long-term memories about the user:
{memory_context}

Use these memories naturally when they are relevant. Do not mention the
memory system explicitly unless asked.

Reply in markdown format. Be concise and to the point. Always think briefly,
then reply with your final content. You have access to tools that can
help you with certain tasks. Use them when needed.

EDITOR TOOLS:
- Use the `read_editor` tool whenever you need to see what the user is
  currently writing before you can help them (summarize, edit, continue,
  review, fix, count words, find typos, etc.). It returns the entire
  current editor content.
- Use the `write_editor` tool to write or replace content in the editor.
  The `content` argument REPLACES the current editor content entirely, so
  if you need to preserve existing text, call `read_editor` first and
  include the existing content in the new value.
- Do not invent or guess editor content. If you are unsure, call
  `read_editor` first.

EDITOR MARKDOWN RULES (the editor only supports the syntax listed below):
{MARKDOWN_RULES}

Always produce content that conforms to these rules. When you write to the
editor, the `content` you pass to `write_editor` MUST be a single
well-formed markdown string using only the inline and block forms above.
"""


# Father Node - Daddy Calling
async def chat_node(state: ChatState, runtime: Runtime[UserContext]):
    memory_lines = state.get("memories", [])
    if memory_lines:
        memory_context = "\n".join(f"- {memory}" for memory in memory_lines)
    else:
        memory_context = "No memories yet."

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(memory_context=memory_context,MARKDOWN_RULES=MARKDOWN_RULES)
    response = await llm_with_tool.ainvoke(
        [SystemMessage(content=system_prompt)] + list(state["messages"])
    )
    return {"messages": [response]}


async def remember_node(state: ChatState, runtime: Runtime[UserContext]):
    if not state.get("should_remember"):
        return {}

    # We "schedule" the slow work and return immediately.
    # The user gets their "DONE" signal NOW.
    # The memory saves in the background.
    asyncio.create_task(extract_and_save_node(state, runtime))
    return {}


# Extractor Node - Extract memories from user input
async def extract_and_save_node(state: ChatState, runtime: Runtime[UserContext]):
    if not state.get("should_remember"):
        return {"should_remember": False}

    last_message = state["messages"][-2]
    last_content = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )
    memory_namespace = ("memories", runtime.context.user_id)
    extraction_prompt = (
        f'User message: "{last_content}"\n\n'
        "Extract ONLY stable facts about the USER that may be useful in future conversations. "
        "Ignore information about assistants, bots, AI characters, quoted text, greetings, or other people. "
        "Do not guess or infer interests, traits, or preferences. "
        "Return only the extracted facts. If no clear user fact exists, return 'NONE'."
    )
    extraction = await llm_extractor.ainvoke(
        [{"role": "user", "content": extraction_prompt}]
    )
    memory_text = extraction.content.strip()
    if not memory_text:
        return {"should_remember": False}

    if memory_text.upper() == "NONE":
        return {"should_remember": False}

    # Defensive: if the LLM just echoed the user's message back, don't save it
    if memory_text.lower() == last_content.lower():
        return {"should_remember": False}

    if memory_text and memory_text.upper() != "NONE":
        await runtime.store.aput(
            memory_namespace,
            str(uuid.uuid4()),
            {"data": memory_text},
        )
    return {"should_remember": False}


# Extraction [Helper]
