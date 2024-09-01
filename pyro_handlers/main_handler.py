import logging
import unidecode
from config import CHAT_ID_MODERATORS, INTERVAL_MIN, PYRO_API_HASH, PYRO_API_ID, TARGET_CHANNEL_ID, TARGET_CHAT_ID
from database.models import Badphrases
from database.engine import session_maker
from sqlalchemy import select


from datetime import datetime as dt
import pandas as pd
from utils.unidecoder import unidecoder
from nltk.tokenize import word_tokenize
import re
import string
from nltk.stem import SnowballStemmer
from fuzzywuzzy import process
from icecream import ic

from pyrogram import Client
from pyrogram.handlers import MessageHandler
from pyrogram import filters
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler


def stem_words(words):
    stemmer = SnowballStemmer("russian")
    return [stemmer.stem(word) for word in words]


def remove_special_characters_and_digits(text):
    return re.sub(f"[{re.escape(string.punctuation + string.digits)}]", "", text)


async def find_similar_words(search_value, words_list, threshold=70):
    best_matches = process.extract(search_value, words_list, limit=5)

    result = [match for match in best_matches if match[1] >= threshold]
    return result


async def badwords_autochecker(app, bad_words=None, unidecoded_bad_words=None):
    async for message in app.get_chat_history(chat_id=TARGET_CHAT_ID, limit=10):

        await asyncio.sleep(1)


        if message.text and any(word in message.text for word in bad_words) or message.caption and any(word in message.caption for word in bad_words):
            print("точно мат")
            await app.send_message(
                chat_id=CHAT_ID_MODERATORS,
                text=f"Найден мат❗️\nhttps://t.me/c/{TARGET_CHANNEL_ID}/{message.id}",
            )
            await message.forward(chat_id=CHAT_ID_MODERATORS)
            try:
                await message.delete()
            except Exception as e:
                logging.error(f"Ошибка при удалении сообщения: {e}")
            return
        
        if message.text and any(word in unidecoder(message.text) for word in unidecoded_bad_words) or message.caption and any(word in unidecoder(message.caption) for word in unidecoded_bad_words):
            unidecoder(message.text)
            print("найден мат с помощью юнидекодера")
            await app.send_message(
                chat_id=CHAT_ID_MODERATORS,
                text=f"Найден мат с помощью юнидекодера❗️\nhttps://t.me/c/{TARGET_CHANNEL_ID}/{message.id}",
            )
            await message.forward(chat_id=CHAT_ID_MODERATORS)
            try:
                await message.delete()
            except Exception as e:
                logging.error(f"Ошибка при удалении сообщения: {e}")
            return


async def check_message_for_bad_words(message_words, bad_words, threshold=70):
    for word in message_words:
        similar_bad_words = await find_similar_words(word, bad_words, threshold)
        if similar_bad_words:
            return True
    return False





async def pyro_main_handler(app, message):
    async with session_maker() as session:
        check_word_query = select(Badphrases.phrase_text, Badphrases.unicoded_phrase_text)
        result = await session.execute(check_word_query)
    result = result.all()

    df = pd.DataFrame(result)

    bad_words = []
    unidecoded_bad_words = []
    for i, row in df.iterrows():
        bad_words.append(row["phrase_text"])
        unidecoded_bad_words.append(row["unicoded_phrase_text"])

    if message.text and any(word in message.text for word in bad_words) or message.caption and any(word in message.caption for word in bad_words):
        print("точно мат")
        await app.send_message(
            chat_id=CHAT_ID_MODERATORS,
            text=f"Найден мат❗️\nhttps://t.me/c/{TARGET_CHANNEL_ID}/{message.id}",
        )
        await message.forward(chat_id=CHAT_ID_MODERATORS)
        try:
            await message.delete()
        except Exception as e:
            logging.error(f"Ошибка при удалении сообщения: {e}")
        return
    
    if message.text and any(word in unidecoder(message.text) for word in unidecoded_bad_words) or message.caption and any(word in unidecoder(message.caption) for word in unidecoded_bad_words):
        unidecoder(message.text)
        print("найден мат с помощью юнидекодера")
        await app.send_message(
            chat_id=CHAT_ID_MODERATORS,
            text=f"Найден мат с помощью юнидекодера❗️\nhttps://t.me/c/{TARGET_CHANNEL_ID}/{message.id}",
        )
        await message.forward(chat_id=CHAT_ID_MODERATORS)
        try:
            await message.delete()
        except Exception as e:
            logging.error(f"Ошибка при удалении сообщения: {e}")
        return

        

   


async def run_pyrogram():
    # app = Client("my_account", PYRO_API_ID, PYRO_API_HASH)
    app = Client("my_account", PYRO_API_ID, PYRO_API_HASH)
    await app.start()

    print("пиро работает")

    async with session_maker() as session:
        check_word_query = select(Badphrases.phrase_text, Badphrases.unicoded_phrase_text)
        result = await session.execute(check_word_query)
        result = result.all()
        df = pd.DataFrame(result)
        bad_words = []
        unidecoded_bad_words = []
        for i, row in df.iterrows():
            bad_words.append(row["phrase_text"])
            unidecoded_bad_words.append(row["unicoded_phrase_text"])

        async def scheduled_badwords_autochecker():
            await badwords_autochecker(app, bad_words, unidecoded_bad_words)

        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            scheduled_badwords_autochecker, "interval", minutes=INTERVAL_MIN
        )
        scheduler.start()
        # await badwords_autochecker(app, bad_words)
    # my_handler = MessageHandler(pyro_main_handler, filters.chat([964635576]))
    my_handler = MessageHandler(pyro_main_handler, filters.chat([TARGET_CHAT_ID, CHAT_ID_MODERATORS]))
    app.add_handler(my_handler)
