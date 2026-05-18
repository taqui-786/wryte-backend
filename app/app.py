


from sqlalchemy import text
# from app.agent import generate_title_for_chat
# from app.action import add_new_thread
# from app.schema import CreateThreadPayload
# from app.action import get_document_by_id
# from app.action import get_all_documents
# from app.action import create_new_document
# from app.action import get_user_id_by_email
from app.schema import ChatPayload, CreateDocumentPayload, GenerateChatTitlePayload

from typing import Annotated, AsyncGenerator

# from app.action import get_user_by_email
from app.agent import build_graph, embeddings, EMBEDDING_DIMS, generate_title_for_chat, my_agent
from app.db import get_async_session
from app.schema import SaveUserPayload, UserResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi_nextauth_jwt import NextAuthJWTv4,NextAuthJWT
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from sqlalchemy.ext.asyncio import AsyncSession
import json
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
    # await create_db_and_tables()

    async with (
        # index= tells LangGraph to embed every memory on write and use
        # cosine-similarity when asearch() is called (semantic search).
        # dims must match the output of your embedding model (1044 here).
        AsyncPostgresStore.from_conn_string(
            DB_URI,
            index={
                "embed": embeddings,   # NVIDIAEmbeddings instance from agent.py
                "dims": EMBEDDING_DIMS,  # 1024 — must match the model
            },
        ) as store,
        AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        # Idempotent — creates tables if they don't exist yet.
        # The store.setup() also creates the vector-index column in Postgres.
        await store.setup()
        await checkpointer.setup()

        app.state.workflow = build_graph(checkpointer=checkpointer, store=store)

        yield
        # Context managers clean up connections on exit


# --------------------------------------------------------------------------- 
# App & middleware
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)
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





# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def return_jwt():
    return {"message": "Hello World"}

@app.get("/health")
async def health(session: AsyncSession = Depends(get_async_session)):
   user = await session.execute(text('SELECT * FROM "user"'))
   result = user.fetchall()
   print('res - ',result)
   # Convert Row objects to dictionaries for JSON serialization
   result_dict = [dict(row._mapping) for row in result]
   return {"status": "ok", "user": result_dict}

@app.post("/chat")
async def chat(payload:ChatPayload):

    user_input: str = payload.message
    thread_id: str = payload.thread_id
    user_id: str = payload.user_id
    workflow = app.state.workflow
    async def stream_generator() -> AsyncGenerator[str, None]:
        async for chunk in my_agent(
            workflow=workflow,
            user_input=user_input,
            thread_id=thread_id,
            user_id=user_id,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: DONE\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.post("/generate-chat-title")
async def generate_chat_title(payload:GenerateChatTitlePayload):
    conversation: str = payload.conversation
    title = await generate_title_for_chat(conversation)
    return {"title": title}

# @app.post("/chat")
# async def chat(request: Request, jwt: dict = Depends(JWT)):
#     """
#     Stream agent responses token-by-token.

#     Expects JSON body:
#         { "message": "...", "thread_id": "..." }

#     The user_id is taken from the JWT so memories are always scoped
#     to the authenticated user — no client-supplied user_id is trusted.
#     """
#     body = await request.json()
#     user_input: str = body.get("message", "")
#     thread_id: str = body.get("thread_id", "default")
#     # JWT sub is the canonical user identifier
#     user_id: str = jwt.get("email") or "anonymous"

    # workflow = request.app.state.workflow

    # async def stream_generator() -> AsyncGenerator[str, None]:
    #     async for chunk in my_agent(
    #         workflow=workflow,
    #         user_input=user_input,
    #         thread_id=thread_id,
    #         user_id=user_id,
    #     ):
    #         yield f"data: {json.dumps(chunk)}\n\n"
    #     yield "data: DONE\n\n"

    # return StreamingResponse(stream_generator(), media_type="text/event-stream")


# @app.get("/get-state/{thread_id}", status_code=status.HTTP_200_OK)
# async def get_state(
#     thread_id: str,
#     request: Request,
#     jwt: dict = Depends(JWT),
# ):
#     # print(jwt.get("email"))
#     workflow = request.app.state.workflow
#     state = await get_chat_state(workflow=workflow, thread_id=thread_id)
#     if state is None:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail="Thread not found",
#         )
#     return state

