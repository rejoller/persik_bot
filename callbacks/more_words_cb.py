from aiogram import Router, F, types
from aiogram.fsm.context import FSMContext
from filters.admins import CallbackQueryAdminFilter
from icecream import ic
from states import Form

router = Router()


@router.callback_query(F.data == "more_words", CallbackQueryAdminFilter())
async def handle_more_words(query: types.CallbackQuery, state: FSMContext):
    print('more_words_handler')
    await query.message.answer('отправьте файл со списком слов(фраз) c названием "список.xlsx".\nТаблица должна содержать только один столбец с названием "words"')
    await state.set_state(Form.more_words)