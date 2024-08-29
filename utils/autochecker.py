from aiogram import Bot
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Messages




# async def badwords_autochecker(session: AsyncSession):
#     messages_query = select(func.max(Messages.message_tg_id))

#     messages = await session.execute(messages_query)

#     result = messages.all()
#     msg_id = ''