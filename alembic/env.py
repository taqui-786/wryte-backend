import os
from logging.config import fileConfig

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from alembic import context

# ---------------------------------------------------------------------------
# Load environment and build a SYNC database URL for Alembic.
# (The app uses postgresql+asyncpg://, but Alembic needs a sync driver.)
# ---------------------------------------------------------------------------
load_dotenv()

# Import only Base — never import app.app or app.agent here, because those
# modules spin up the async engine at import time and crash in a sync context.
from app.db import Base  # noqa: E402  (after load_dotenv so DATABASE_URL is set)


def _make_sync_url(async_url: str) -> str:
    """Replace the asyncpg driver with psycopg2 for Alembic's sync engine.

    asyncpg uses ?ssl=require but psycopg2 uses ?sslmode=require, so we
    translate that query parameter too.
    """
    url = async_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    url = url.replace("?ssl=require", "?sslmode=require")
    return url


DATABASE_URL = os.environ["DATABASE_URL"]
SYNC_DATABASE_URL = _make_sync_url(DATABASE_URL)

# ---------------------------------------------------------------------------
# Alembic Config
# ---------------------------------------------------------------------------
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", SYNC_DATABASE_URL)

target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Migration runners
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """Run migrations without an active DB connection (generates SQL only)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against the live database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
