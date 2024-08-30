import logging

from aiogram.types import Message
from aiogram import Router, F
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext

from database.models import Badphrases
from filters.admins import AdminFilter, CallbackQueryAdminFilter
from states import Form

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError




router = Router()


@router.message(StateFilter(Form.one_word))
async def one_word_handler(message: Message, session: AsyncSession, state: FSMContext):
    try:          
        save_query = insert(Badphrases).values(phrase_text = message.text)
        await session.execute(save_query)
        await session.commit()
        await message.answer('Новая фраза загружена в базу данных')
    except SQLAlchemyError as db_err:
        logging.error(
            f"Ошибка базы данных при сохранении фразы: {db_err}"
        )
        await message.answer('Ошибка базы данных при сохранении фразы {db_err}')
    await state.clear()

            
            
        
