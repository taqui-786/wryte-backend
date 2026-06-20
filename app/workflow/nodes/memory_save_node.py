


from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from app.workflow.llm import llm_secondary
from app.workflow.state import UserContext, WorkflowState


class MemorySaveOutput(BaseModel):
    save:bool = Field(description="Whether to save the memory True or False")
    memory:str = Field(description="The memory to save")

memory_save_llm = llm_secondary.with_structured_output(MemorySaveOutput)
async def extract_and_save_memory_node(state:WorkflowState, runtime:Runtime[UserContext]):
    last_message = state["messages"][-2]
    last_content = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )
    memory_namespace = ("memories", runtime.context.user_id)
    extraction_prompt = (
        f'User message: "{last_content}"\n\n'
        "Extract stable, reusable facts about the USER only.\n\n"
        "EXTRACT:\n"
        "- Personal facts (name, age, location, job, company)\n"
        "- Projects or work they are doing\n"
        "- Skills, tools, or tech they use\n"
        "- Preferences, opinions, or dislikes\n"
        "- Habits or routines\n"
        "- Instructions for how the AI should behave\n"
        "- Relationships or team context\n\n"
        "WRITING STYLE (only if message is 30+ words or has a distinctive voice):\n"
        "- Note their tone, vocabulary, sentence structure, personality\n"
        "- Add a one-line replication note: how to write content in their exact style\n\n"
        "IGNORE:\n"
        "- Questions, greetings, filler\n"
        "- Statements about others, AI, or external topics\n"
        "- Anything inferred or guessed — only explicit signals\n\n"
        "Return a clean summary of extracted facts. "
        "If style profiling applies, append it at the end under 'Style:'. "
        "If nothing to extract, return 'save = False'."
    )
    
    result:MemorySaveOutput = await memory_save_llm.ainvoke(extraction_prompt)
    print(result)
    if not result.save:
        return {}

    await runtime.store.aput(memory_namespace, str(uuid.uuid4()), {"data":result.memory})
    return {"messages": [AIMessage(content="Memory extracted and saved")]}
