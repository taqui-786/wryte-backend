from typing import AsyncGenerator
from langchain_core.messages import HumanMessage, messages_to_dict
from langgraph.graph.state import CompiledStateGraph

from app.workflow.state import UserContext


async def my_agent(
    workflow: CompiledStateGraph,
    user_input: str,
    thread_id: str,
    user_id: str,
    editor_content: str = "",
) -> AsyncGenerator[str,None]:
    config = {"configurable": {"thread_id": thread_id}}
    context = UserContext(user_id=user_id)
    async for chunk in workflow.astream(
        {
            "messages": [HumanMessage(content=user_input)],
            "editor_content": editor_content,
        },
        config=config,
        context=context,
        stream_mode="messages",
        version="v2",
    ):
        if chunk.get("type") != "messages":
            continue
        message_chunk, metadata = chunk["data"]
        if metadata.get("langgraph_node") in ["memory_save"]:
            continue
        if message_chunk:
            yield {
                "node": metadata.get("langgraph_node"),
                "message": message_chunk.model_dump(),
            }


async def get_chat_state(workflow: CompiledStateGraph, thread_id: str):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await workflow.aget_state(config=config)
        if snapshot is None:
            return None
        values = dict(snapshot.values)
        if "messages" in values:
            values["messages"] = messages_to_dict(values["messages"])
        return values
    except Exception as exc:
        print(f"[get_chat_state] error: {exc}")
        return None
