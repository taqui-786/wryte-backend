

from __future__ import annotations

from typing import Annotated, AsyncGenerator

from app.action import get_user_by_email
from app.agent import build_graph, get_chat_state, my_agent
from app.db import User, create_db_and_tables, get_async_session
from app.schema import SaveUserPayload, UserResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi_nextauth_jwt import NextAuthJWTv4
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from sqlalchemy.ext.asyncio import AsyncSession
import os

load_dotenv()

def _psycopg_uri(url: str) -> str:
    """
    Convert a SQLAlchemy-style DATABASE_URL to a plain psycopg connection string.

    Handles two common transformations:
      1. Strips dialect+driver prefix, e.g. "postgresql+asyncpg://" → "postgresql://"
      2. Normalises the SSL param, e.g. "?ssl=require" → "?sslmode=require"
    """
    # 1. Strip driver suffix from scheme (e.g. "+asyncpg", "+psycopg2")
    if "://" in url:
        scheme, rest = url.split("://", 1)
        if "+" in scheme:
            scheme = scheme.split("+")[0]
        url = f"{scheme}://{rest}"

    # 2. Replace bare ?ssl / ?ssl=require with ?sslmode=require
    if "?ssl=require" in url:
        url = url.replace("?ssl=require", "?sslmode=require")
    elif url.endswith("?ssl") or "&ssl" in url:
        url = url.replace("?ssl", "?sslmode=require").replace("&ssl", "&sslmode=require")

    return url


DB_URI = _psycopg_uri(os.environ["DATABASE_URL"])


# ---------------------------------------------------------------------------
# Lifespan — sets up store, checkpointer and compiled graph
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create application DB tables (SQLAlchemy models)
    await create_db_and_tables()

    async with (
        AsyncPostgresStore.from_conn_string(DB_URI) as store,
        AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        # Idempotent — creates tables if they don't exist yet
        await store.setup()
        await checkpointer.setup()

        app.state.workflow = build_graph(checkpointer=checkpointer, store=store)

        yield
        # Context managers clean up connections on exit


# --------------------------------------------------------------------------- 
# App & middleware
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)
JWT = NextAuthJWTv4(secret=os.environ["NEXTAUTH_SECRET"])

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://wryte-ti.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler — ensures CORS headers survive unhandled 500s.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    origin = request.headers.get("origin", "")
    headers = {}
    if origin in ALLOWED_ORIGINS:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/user/save",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def save_new_logged_in_user(
    payload: SaveUserPayload,
    session: AsyncSession = Depends(get_async_session),
):
    is_user = await get_user_by_email(payload.email, session)
    if is_user is not None:
        print("user already exists")
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
async def chat(request: Request, jwt: dict = Depends(JWT)):
    """
    Stream agent responses token-by-token.

    Expects JSON body:
        { "message": "...", "thread_id": "..." }

    The user_id is taken from the JWT so memories are always scoped
    to the authenticated user — no client-supplied user_id is trusted.
    """
    body = await request.json()
    user_input: str = body.get("message", "")
    thread_id: str = body.get("thread_id", "default")
    # JWT sub is the canonical user identifier
    user_id: str = jwt.get("email") or "anonymous"

    workflow = request.app.state.workflow

    async def stream_generator() -> AsyncGenerator[str, None]:
        async for chunk in my_agent(
            workflow=workflow,
            user_input=user_input,
            thread_id=thread_id,
            user_id=user_id,
        ):
            yield f"data: {chunk}\n\n"   # ← SSE format: must have "data: " prefix + double newline
        yield "data: DONE\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@app.get("/get-state/{thread_id}", status_code=status.HTTP_200_OK)
async def get_state(
    thread_id: str,
    request: Request,
    jwt: dict = Depends(JWT),
):
    # print(jwt.get("email"))
    workflow = request.app.state.workflow
    state = await get_chat_state(workflow=workflow, thread_id=thread_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found",
        )
    return state

