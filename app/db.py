import os
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


class Base(DeclarativeBase):
    pass


# class Post(Base):
#     __tablename__ = "posts"

#     id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     caption = Column(Text)
#     url = Column(String, nullable=False)
#     file_type = Column(String, nullable=False)
#     file_name = Column(String, nullable=False)
#     created_at = Column(DateTime, default=datetime.utcnow)


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)
    avatar_url = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    docs = relationship("Doc", back_populates="user")

class Thread(Base):
    __tablename__ = "threads"
    id = Column(UUID(as_uuid=True), primary_key=True)
    title = Column(String, nullable=False)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    doc_id = Column(
        UUID(as_uuid=True),
        ForeignKey("docs.id"),
        nullable=False
    )
    doc = relationship("Doc", back_populates="threads")

class Doc(Base):
    __tablename__ = "docs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=False)
    content = Column(String, nullable=True)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False
    )
    user = relationship("User", back_populates="docs")
    threads = relationship("Thread", back_populates="doc")


engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session






# Migration commands

# 1. Generate migration (diffs your models vs the real DB)

# uv run alembic revision --autogenerate -m "describe your change"

# 2. Review the file in alembic/versions/ — make sure it looks correct

# 3. Apply to the database
# uv run alembic upgrade head
