from typing import Annotated
from app.agent import get_chat_state
from app.agent import my_agent
from app.action import get_user_by_email
from app.schema import SaveUserPayload, UserResponse, UserInfoResponse
from app.db import get_async_session
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import create_db_and_tables
from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.responses import RedirectResponse, StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
from app.db import User
from fastapi.middleware.cors import CORSMiddleware
from fastapi_nextauth_jwt import NextAuthJWTv4

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)
JWT = NextAuthJWTv4(secret=os.environ["NEXTAUTH_SECRET"],cookie_name="next-auth.session-token")




origins = [
    "http://localhost:3000",
    "https://wryte-ti.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler — ensures CORS headers are present even on 500 errors.
# Without this, an unhandled exception causes Starlette to return a bare 500
# with no CORS headers, and the browser reports a CORS error instead of the real one.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    origin = request.headers.get("origin", "")
    allowed = [
        "http://localhost:3000",
        "https://wryte-ti.vercel.app",
    ]
    headers = {}
    if origin in allowed:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
        headers=headers,
    )


@app.post(
    "/user/save", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def save_new_logged_in_user(
    payload: SaveUserPayload, session: AsyncSession = Depends(get_async_session)
):

    is_user = await get_user_by_email(payload.email, session)

    if is_user is not None:
        print("user is already exist")
        return is_user

    new_user = User(
        name=payload.name,
        email=payload.email,
        avatar_url=payload.image,
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    return new_user

@app.get("/")
async def return_jwt(jwt: Annotated[dict, Depends(JWT)]):
    return jwt


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_input = body.get("message", "")

    def stream_generator():
        for chunk in my_agent(user_input):
            yield str(chunk)
        yield "DONE"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.get("/get-state/{thread_id}",status_code=status.HTTP_200_OK)
async def get_state(thread_id:str,jwt: dict = Depends(JWT)):
    state = get_chat_state(thread_id)
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    return state