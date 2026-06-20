from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from app.services.chat import ChatService
from app.schema import ChatPayload, GenerateChatTitlePayload
from app.services.title import TitleService




router = APIRouter()

@router.get("/")
async def return_jwt():
    return {"message": "Hello World"}



@router.post("/chat",status_code=status.HTTP_200_OK)
async def chat(payload:ChatPayload, request:Request):
    workflow = request.app.state.workflow
    service = ChatService(workflow=workflow,)
    return StreamingResponse(
        service.stream(
            payload.message,
            payload.thread_id,
            payload.user_id,
            payload.editor_content,
        ),
        media_type="text/event-stream",
    )

@router.post("/generate-chat-title",status_code=status.HTTP_200_OK)
async def generate_chat_title(payload:GenerateChatTitlePayload):
    conversation: str = payload.conversation
    title = await TitleService.generate_title(conversation)
    return {"title": title}

@router.get("/get-thread-messages/{thread_id}",status_code=status.HTTP_200_OK)
async def getThreadMessages(thread_id: str,request:Request):
    workflow = request.app.state.workflow
    service = ChatService(workflow=workflow)
    state = await service.get_thread_state(thread_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found",
        )
    return state


