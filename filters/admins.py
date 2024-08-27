from aiogram.filters import BaseFilter
from aiogram.types import Message


from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database.models import Users

async def get_admins_list(session: AsyncSession, user_id = None):
    subscribers_query = select(Users.user_id)

    result = await session.execute(subscribers_query)
    ADMINS_LIST = result.all()
    return ADMINS_LIST



class AdminFilter(BaseFilter):
    async def __call__(self, message: Message) -> bool:
        user_id = message.from_user.id
        from database.engine import session_maker
        async with session_maker() as session:
            ADMINS_LIST = await get_admins_list(session, user_id)
        

        return message.from_user.id in ADMINS_LIST[0]