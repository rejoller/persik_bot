import logging
import string
from aiogram import F, Router, types, Bot
import os
import pandas as pd
from icecream import ic
from datetime import datetime as dt

from sqlalchemy.dialects.postgresql import insert

from database.models import Badphrases, Messages
from filters.admins import AdminFilter
from nltk.tokenize import word_tokenize
import re
import pandas as pd
from icecream import ic
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from filters.admins import AdminFilter
from nltk.stem import SnowballStemmer
from fuzzywuzzy import process


router = Router()


def stem_words(words):
    stemmer = SnowballStemmer("russian")
    return [stemmer.stem(word) for word in words]


def remove_special_characters_and_digits(text):
    return re.sub(f"[{re.escape(string.punctuation + string.digits)}]", "", text)


async def find_similar_words(search_value, words_list, threshold=70):
    best_matches = process.extract(search_value, words_list, limit=5)
    result = [match for match in best_matches if match[1] >= threshold]
    return result


async def check_message_for_bad_words(message_words, bad_words, threshold=70):
    for word in message_words:
        similar_bad_words = await find_similar_words(word, bad_words, threshold)
        if similar_bad_words:
            return True
    return False


@router.message(F.text, F.chat.id == -1002219094742)
async def badwords_handler(message: types.Message, session: AsyncSession, bot: Bot):
    add_user_query = (
        insert(Messages)
        .values(
            message_tg_id=message.message_id,
            date_send=dt.now(),
            message_text=message.text,
        )
        .on_conflict_do_nothing()
    )
    await session.execute(add_user_query)
    await session.commit()

    check_word_query = select(Badphrases.phrase_text)
    result = await session.execute(check_word_query)
    result = result.all()

    df = pd.DataFrame(result)
    bad_words = []
    for i, row in df.iterrows():
        bad_words.append(row["phrase_text"])

    if message.text.lower() in bad_words:
        await message.answer("ты точно сматерился")
        await bot.delete_message(chat_id=-1002219094742, message_id=message.message_id)
        return

    cleaned_text = remove_special_characters_and_digits(message.text.lower())
    message_words = word_tokenize(cleaned_text)

    if await check_message_for_bad_words(message_words, bad_words, threshold=95):
        await message.answer("ты сматерился с вероятностью более 95 процентов")
        return

    if await check_message_for_bad_words(message_words, bad_words, threshold=93):
        await message.answer(
            "ты скорее всего сматерился с вероятностью более 90 процентов"
        )
        return

    await message.answer("мата нет")
