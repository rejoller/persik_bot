from aiogram.fsm.state import State, StatesGroup


class Form(StatesGroup):
    one_word = State()
    more_words = State()