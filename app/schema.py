
import uuid
from datetime import datetime
import uuid
from pydantic import BaseModel

class UserInfoResponse(BaseModel):
    id: str
    name: str
    given_name: str
    email: str
    verified_email: bool
    picture: str

class SaveUserPayload(BaseModel):
    name:str
    email:str
    image:str

class UserResponse(BaseModel):
    id: uuid.UUID
    name: str
    email: str
    avatar_url: str
    created_at: datetime
    class Config:
        from_attributes = True

class CreateDocumentPayload(BaseModel):
    title: str


class CreateThreadPayload(BaseModel):
    doc_id: str
    conversation:str
    thread_id:str

class ChatPayload(BaseModel):
    message: str
    thread_id: str
    user_id: str

class GenerateChatTitlePayload(BaseModel):
    conversation: str
    