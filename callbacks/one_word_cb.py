from aiogram import Router, F, types
from aiogram.fsm.context import FSMContext
from filters.admins import AdminFilter, CallbackQueryAdminFilter
from icecream import ic
from states import Form

router = Router()


@router.callback_query(F.data == "one_word", CallbackQueryAdminFilter())
async def handle_one_word(query: types.CallbackQuery, state: FSMContext):
    print('one_word_handler')
    await query.message.answer('Напишите одно (фразу), которое хотите добавить в базу данных')
    await state.set_state(Form.one_word)