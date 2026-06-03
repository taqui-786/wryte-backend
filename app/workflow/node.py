import asyncio
import uuid
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel
from app.workflow.state import ChatState, UserContext
from app.workflow.tool import llm_classifier, llm_extractor, llm_with_tool


# Classifier Node - Just to determine the msg need to be saved Yes / No
class MemoryDecision(BaseModel):
    should_remember: bool
    reason: str


async def classify_node(state: ChatState, runtime: Runtime[UserContext]):
    last_message = state["messages"][-1]
    last_content = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )
    prompt = (
        "Look at this user message and decide if it contains a fact worth "
        "remembering about the user for future conversations. Examples of "
        "worth-remembering: their name, job, preferences,website, ongoing projects, "
        "writing style, opinions. Examples of NOT worth remembering: "
        "questions, requests for help, chitchat. if that question doest not contain any personal user stuff\n\n"
        f'Message: "{last_content}"\n\n'
        "Return should_remember=True only if there's a clear, persistent fact."
    )
    classifier = llm_classifier.with_structured_output(MemoryDecision)
    decision: MemoryDecision = await classifier.ainvoke(
        [{"role": "user", "content": prompt}]
    )
    print("Decision: ", decision)
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
- Use the `read_editor` tool whenever the user asks you to do something
  with what they are currently writing (summarize, edit, continue, review,
  fix, count words, find typos, etc.) and you have not already been shown
  the full content. The tool returns the entire current editor content.
- Do not invent or guess editor content. If you are unsure, call
  `read_editor` first.
"""


# Father Node - Daddy Calling
async def chat_node(state: ChatState, runtime: Runtime[UserContext]):
    memory_lines = state.get("memories", [])
    if memory_lines:
        memory_context = "\n".join(f"- {memory}" for memory in memory_lines)
    else:
        memory_context = "No memories yet."

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(memory_context=memory_context)
    response = await llm_with_tool.ainvoke(
        [SystemMessage(content=system_prompt)] + list(state["messages"])
    )
    return {"messages": [response]}


# Extractor Node - Extract memories from user input
async def extract_and_save_node(state: ChatState, runtime: Runtime[UserContext]):
    if not state.get("should_remember"):
        return {"should_remember": False, "memory_to_save": None}

    last_message = state["messages"][-2]
    last_content = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )
    user_id = runtime.context.user_id
    memory_namespace = ("memories", user_id)
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
        return {"should_remember": False, "memory_to_save": None}

    if memory_text.upper() == "NONE":
        return {"should_remember": False, "memory_to_save": None}

    # Defensive: if the LLM just echoed the user's message back, don't save it
    if memory_text.lower() == last_content.lower():
        return {"should_remember": False, "memory_to_save": None}

    if memory_text and memory_text.upper() != "NONE":
        await runtime.store.aput(
            memory_namespace,
            str(uuid.uuid4()),
            {"data": memory_text},
        )
    return {"should_remember": False, "memory_to_save": memory_text}


# Extraction [Helper]


async def schedule_memory_save(workflow, state, config, context):
    """
    Spawns a background task that re-runs the graph for the extract step only.
    The user has already seen the response by the time this kicks in.
    """
    # We use `astream` with `interrupt_before` to run ONLY the extract step.
    # Actually, a simpler approach: just run the extraction function directly.

    # Easiest version: do it inline with asyncio.create_task
    # We pass the runtime context manually because we're outside the graph.
    async def _background_save():
        try:
            from langgraph.graph import StateGraph

            # Just call the extractor directly with a minimal state
            minimal_state = {
                "messages": state["messages"],
                "memories": [],
                "should_remember": state.get("should_remember", False),
                "memory_to_save": None,
            }

            # Build a tiny runtime-like object
            class _MiniRuntime:
                def __init__(self, ctx):
                    self.context = ctx

            await extract_and_save_node(minimal_state, _MiniRuntime(context))
        except Exception as e:
            # Never let a background failure crash anything
            print(f"[background memory save] failed: {e}")

    asyncio.create_task(_background_save())
