
from app.db import Doc
import uuid
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from app.schema import UserInfoResponse
from app.db import User 



async def get_user_by_email(email: str, session: AsyncSession) -> User:
    result = await session.execute(
        select(User).where(User.email == email)
    )
    user = result.scalar_one_or_none()  # returns User or None
    return user

async def get_user_id_by_email(email: str, session: AsyncSession) -> uuid.UUID:
    try:
        result = await session.execute(
            select(User.id).where(User.email == email)
        )
        user = result.scalar_one_or_none()  # returns User or None
        print("user-id",user)
        return user
    except Exception as e:
        print(e)
        raise e
    
    


async def save_user_to_db(credentials:UserInfoResponse, session:AsyncSession):
    is_user = await get_user_by_email(credentials.email, session)
    print("user - ",is_user)

    if  is_user is not None:
        return is_user

    new_user = User(
        name = credentials.name,
        email = credentials.email,
        avatar_url = credentials.picture,
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    return new_user 


async def create_new_document(user_id:uuid.UUID,title:str, session:AsyncSession):
    try:
        new_doc = Doc(
            user_id = user_id,
            title = title,
        )
        session.add(new_doc)
        await session.commit()
        await session.refresh(new_doc)
        return new_doc.id
    except Exception as e:
        await session.rollback()
        print(e)
        raise e


async def get_user_documents(user_id:uuid.UUID, session:AsyncSession):
    try:
        result = await session.execute(
            select(Doc).where(Doc.user_id == user_id)
        )
        return result.scalars().all()
    except Exception as e:
        print(e)
        raise e

async def get_all_documents(user_id:uuid.UUID,session:AsyncSession):
    try:
        result = await session.execute(
            select(Doc.id,Doc.title).where(Doc.user_id == user_id )
        )
        return result.mappings().all()
    except Exception as e:
        print(e)
        raise e

async def get_document_by_id(document_id:str,session:AsyncSession):
    try:
        result = await session.execute(
            select(Doc).options(selectinload(Doc.user)).where(Doc.id == document_id)
        )
        return result.scalar_one_or_none()
    except Exception as e:
        print(e)
        raise e
