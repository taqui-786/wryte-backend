
from pydantic import BaseModel


class ChatPayload(BaseModel):
    message: str
    thread_id: str
    user_id: str

class GenerateChatTitlePayload(BaseModel):
    conversation: str
    