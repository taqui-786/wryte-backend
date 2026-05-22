import hashlib
import uuid
from langgraph.runtime import Runtime
from app.workflow.state import ChatState, UserContext
from app.workflow.tool import llm_with_tool


async def chat_node(state: ChatState, runtime: Runtime[UserContext]):
    user_id = runtime.context.user_id
    user_id = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100000
    memory_namespace = ("memories", str(user_id))

    last_message = state["messages"][-1]
    last_content = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )

    memories = await runtime.store.asearch(
        memory_namespace,
        query=last_content,
        limit=5,
    )
    memory_lines = [d.value["data"] for d in memories]
    memory_context = "\n".join(memory_lines) if memory_lines else "No memories yet."

    system_prompt = (
        "You are a helpful, thoughtful writing assistant called Wryte.\n"
        "You have access to the following long-term memories about the user:\n"
        f"{memory_context}\n\n"
        "Use these memories naturally when they are relevant. "
        "Do not mention the memory system explicitly unless asked."
        "Reply in markdown format."
        "Don't be very lengthy in your responses, be concise and to the point."
        "Always do short thinking and then reply you final content"
        "You have access to tools that can help you with certain tasks. Use them when needed."
    )

    if "remember" in last_content.lower():
        extraction_prompt = (
            f'The user said: "{last_content}"\n\n'
            "Extract a single, concise fact to remember about the user from this message. "
            "Return only the fact, nothing else."
        )
        extraction = await llm_with_tool.ainvoke([{"role": "user", "content": extraction_prompt}])
        memory_text = extraction.content.strip()
        if memory_text:
            await runtime.store.aput(
                memory_namespace,
                str(uuid.uuid4()),
                {"data": memory_text},
            )

    response = await llm_with_tool.ainvoke(
        [{"role": "system", "content": system_prompt}] + list(state["messages"])
    )
    return {"messages": [response]}
