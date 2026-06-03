from json import dumps
from typing import AsyncGenerator
from langgraph.graph.state import CompiledStateGraph

from app.workflow.runner import get_chat_state, my_agent


class ChatService:
    """Orchestrates chat operations"""
    
    def __init__(self, workflow:CompiledStateGraph):
        self.workflow = workflow
        
    async def stream(
        self,
        user_input: str,
        thread_id: str,
        user_id: str,
        editor_content: str = "",
    ) -> AsyncGenerator[str, None]:
        async for chunk in my_agent(
            workflow=self.workflow,
            user_input=user_input,
            thread_id=thread_id,
            user_id=user_id,
            editor_content=editor_content,
        ):
            yield f"data: {dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    async def get_thread_state(self, thread_id: str):
        return await get_chat_state(workflow=self.workflow, thread_id=thread_id)


