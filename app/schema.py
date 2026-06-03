
from pydantic import BaseModel


class ChatPayload(BaseModel):
    message: str
    thread_id: str
    user_id: str
    editor_content: str = ""

class GenerateChatTitlePayload(BaseModel):
    conversation: str
    