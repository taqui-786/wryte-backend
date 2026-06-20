from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import settings
from app.routes import router
from app.workflow.graph import build_graph
from app.workflow.llm import EMBEDDING_DIMS, embeddings


def _psycopg_uri(url: str) -> str:
    """
    Convert a SQLAlchemy-style DATABASE_URL to a plain psycopg connection string.

    Handles two common transformations:
      1. Strips dialect+driver prefix, e.g. "postgresql+asyncpg://" → "postgresql://"
      2. Normalises the SSL param, e.g. "?ssl=require" → "?sslmode=require"
    """
    if "://" in url:
        scheme, rest = url.split("://", 1)
        if "+" in scheme:
            scheme = scheme.split("+")[0]
        url = f"{scheme}://{rest}"

    if "?ssl=require" in url:
        url = url.replace("?ssl=require", "?sslmode=require")
    elif url.endswith("?ssl") or "&ssl" in url:
        url = url.replace("?ssl", "?sslmode=require").replace(
            "&ssl", "&sslmode=require"
        )

    return url


DB_URI = _psycopg_uri(settings.DATABASE_URL)

_POOL_CONFIG = {
    "min_size": 2,
    "max_size": 10,
    "max_idle": 120,
    "max_lifetime": 1800,
    "check": AsyncConnectionPool.check_connection,
    "kwargs": {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
    },
}


# ---------------------------------------------------------------------------
# Lifespan — sets up store, checkpointer and compiled graph
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):

    pool = AsyncConnectionPool(DB_URI, **_POOL_CONFIG)

    async with (
        pool,
        AsyncPostgresStore.from_conn_string(
            DB_URI,
            pool_config={**_POOL_CONFIG},
            index={
                "embed": embeddings,
                "dims": EMBEDDING_DIMS,
            },
        ) as store,
    ):
        checkpointer = AsyncPostgresSaver(conn=pool)

        await store.setup()
        await checkpointer.setup()

        app.state.workflow = build_graph(checkpointer=checkpointer, store=store)

        yield


# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan,title=settings.PROJECT_NAME,version=settings.VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router=router, prefix=settings.API_V1_STR)

