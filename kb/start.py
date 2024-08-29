from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton



markup = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text="Добавить одно слово(фразу)", callback_data="one_word"),
         InlineKeyboardButton(
            text="Добавить файл", callback_data="more_words")],
])