
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.schema import UserInfoResponse
from app.db import User 



async def get_user_by_email(email: str, session: AsyncSession) -> User:
    result = await session.execute(
        select(User).where(User.email == email)
    )
    user = result.scalar_one_or_none()  # returns User or None
    return user


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