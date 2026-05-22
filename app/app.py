from app.agent import build_graph, embeddings, EMBEDDING_DIMS
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from app.config import settings


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
        url = url.replace("?ssl", "?sslmode=require").replace(
            "&ssl", "&sslmode=require"
        )

    return url


DB_URI = _psycopg_uri(settings.DATABASE_URL)


# ---------------------------------------------------------------------------
# Lifespan — sets up store, checkpointer and compiled graph
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):

    async with (
        AsyncPostgresStore.from_conn_string(
            DB_URI,
            index={
                "embed": embeddings,  # NVIDIAEmbeddings instance from agent.py
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

app = FastAPI(lifespan=lifespan,title=settings.PROJECT_NAME,version=settings.VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORE_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
