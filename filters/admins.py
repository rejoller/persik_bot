from aiogram.filters import BaseFilter
from aiogram.types import Message, CallbackQuery


from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database.models import Users
from database.engine import session_maker




async def get_admins_list(session: AsyncSession, user_id = None):
    subscribers_query = select(Users.user_id).where(Users.is_admin == True)
    result = await session.execute(subscribers_query)
    ADMINS_LIST = [row[0] for row in result.all()] 
    return ADMINS_LIST



class AdminFilter(BaseFilter):
    async def __call__(self, message: Message) -> bool:
        user_id = message.from_user.id
        async with session_maker() as session:
            ADMINS_LIST = await get_admins_list(session, user_id)
        if user_id not in ADMINS_LIST:
            await message.answer('У вас нет прав на выполнение этой команды')
        return message.from_user.id in ADMINS_LIST



class CallbackQueryAdminFilter(BaseFilter):
    async def __call__(self, query: CallbackQuery) -> bool:
        user_id = query.from_user.id  
        async with session_maker() as session:
            ADMINS_LIST = await get_admins_list(session, user_id)
        if user_id not in ADMINS_LIST:
            await query.answer('У вас нет прав на выполнение этой команды', show_alert=True)
        return query.from_user.id in ADMINS_LIST