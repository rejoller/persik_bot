from aiogram import Router, F
from aiogram.types import  Message
from aiogram.fsm.context import FSMContext
from aiogram.filters import CommandStart


from sqlalchemy.ext.asyncio import AsyncSession


from user_manager import UserManager
from kb.start import markup




router = Router()


@router.message(CommandStart(), F.chat.type == 'private')
async def handle_start(message: Message, state: FSMContext, session: AsyncSession):
    await state.clear()
    user_manager = UserManager(session)
    user_data = user_manager.extract_user_data_from_message(message)
    await user_manager.add_user_if_not_exists(user_data)

    await message.answer('Добро пожаловать в бота!', reply_markup=markup)