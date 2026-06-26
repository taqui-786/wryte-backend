import uuid
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from app.workflow.llm import llm_structure
from app.workflow.state import UserContext, WorkflowState


class MemorySaveOutput(BaseModel):
    save: bool = Field(description="Whether to save the memory True or False")
    memory: str = Field(description="The memory to save")


memory_save_llm = llm_structure.with_structured_output(MemorySaveOutput)


async def extract_and_save_memory_node(
    state: WorkflowState, runtime: Runtime[UserContext]
):
    last_message = state["messages"][-1]
    last_content = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )
    print("last_content", last_content)
    memory_namespace = ("memories", runtime.context.user_id)
    extraction_prompt = f"""
User message:
{last_content}

Task:
Decide whether this message contains any information worth remembering for future conversations.

Save information if it is:
- A personal fact
- An interest or curiosity
- A goal or aspiration
- A preference, opinion, or dislike
- A skill, profession, or area of study
- A project, task, or long-term activity
- A recurring behavior or habit
- An instruction about how the assistant should respond
- Any information likely to improve future interactions

Do NOT save:
- Greetings, small talk, filler
- Temporary requests with no lasting value
- Information about other people
- Facts that are already obvious from the current conversation alone
- Guesses or inferred information

Rules:
- Prefer saving too little over inventing facts.
- Extract only explicit information.
- If multiple memories exist, combine them into a concise summary.
- User interests, learning goals, and areas they want help with ARE valid memories.

Return:
save = true if at least one useful memory exists.
save = false otherwise.

memory = concise memory to store.
"""

    result: MemorySaveOutput = await memory_save_llm.ainvoke(extraction_prompt)
    print(result)
    if not result.save:
        return {}

    await runtime.store.aput(
        memory_namespace, str(uuid.uuid4()), {"data": result.memory}
    )
    return {"messages": [AIMessage(content="Memory extracted and saved")]}
