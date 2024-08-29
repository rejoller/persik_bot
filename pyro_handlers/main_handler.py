from sqlalchemy.dialects.postgresql import insert
from config import CHAT_ID_MODERATORS, INTERVAL_MIN, PYRO_API_HASH, PYRO_API_ID
from database.models import Badphrases, Messages
from database.engine import session_maker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select


from datetime import datetime as dt
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import re
import string
from nltk.stem import SnowballStemmer
from fuzzywuzzy import process
from icecream import ic

from pyrogram import Client
from pyrogram.handlers import MessageHandler
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler



def stem_words(words):
    stemmer = SnowballStemmer("russian")
    return [stemmer.stem(word) for word in words]


def remove_special_characters_and_digits(text):
    return re.sub(f"[{re.escape(string.punctuation + string.digits)}]", "", text)


async def find_similar_words(search_value, words_list, threshold=70):
    best_matches = process.extract(search_value, words_list, limit=1)

    result = [match for match in best_matches if match[1] >= threshold]
    return result




app = Client("my_account", PYRO_API_ID, PYRO_API_HASH)


async def badwords_autochecker(app, bad_words = None):
    
    
    async for message in app.get_chat_history(chat_id=-1001867239318, limit=3):
        await asyncio.sleep(1)
        print(f'https://t.me/c/1867239318/{message.id}')
        await app.send_message(chat_id=964635576, text=f"https://t.me/c/1867239318/{message.id}")
        if message.text is not None:
            if message.text.lower() in bad_words:
                print("точно мат")
                # await app.send_message(chat_id=-CHAT_ID_MODERATORS, text='Точно мат❗️')
                # await message.forward(chat_id=CHAT_ID_MODERATORS)

                # await bot.delete_message(chat_id=-1002219094742, message_id=message.message_id)
                return

            cleaned_text = remove_special_characters_and_digits(message.text.lower())
            message_words = word_tokenize(cleaned_text)

            if await check_message_for_bad_words(message_words, bad_words, threshold=95):
                print("на 95% мат")
                # await app.send_message(chat_id=CHAT_ID_MODERATORS, text='👇на 95% мат❗️👇')
                # await message.forward(chat_id=CHAT_ID_MODERATORS)


            else:
                print('мата нет')
                # await app.send_message(chat_id=CHAT_ID_MODERATORS, text='👇мата нет👇')
                # await message.forward(chat_id=CHAT_ID_MODERATORS)



async def check_message_for_bad_words(message_words, bad_words, threshold=70):
    for word in message_words:
        similar_bad_words = await find_similar_words(word, bad_words, threshold)
        if similar_bad_words:
            return True
    return False

@app.on_message(filters=(-1001867239318))
async def pyro_main_handler(app, message):
    ic(message.id, message.text)
    

    async with session_maker() as session:
        check_word_query = select(Badphrases.phrase_text)
        result = await session.execute(check_word_query)
    result = result.all()

    df = pd.DataFrame(result)

    bad_words = []
    for i, row in df.iterrows():
        bad_words.append(row["phrase_text"])

    if message.text.lower() in bad_words:
        print("точно мат")
        # await app.send_message(chat_id=CHAT_ID_MODERATORS, text="Точно мат❗️")
        # await message.forward(chat_id=CHAT_ID_MODERATORS)

        # await bot.delete_message(chat_id=-1002219094742, message_id=message.message_id)
        return

    cleaned_text = remove_special_characters_and_digits(message.text.lower())
    message_words = word_tokenize(cleaned_text)

    if await check_message_for_bad_words(message_words, bad_words, threshold=95):
        print("на 95% мат")
        # await app.send_message(chat_id=CHAT_ID_MODERATORS, text="на 95% мат❗️")
        # await message.forward(chat_id=CHAT_ID_MODERATORS)
        return

    if await check_message_for_bad_words(message_words, bad_words, threshold=93):
        print("на 93% мат")
        # await app.send_message(chat_id=CHAT_ID_MODERATORS, text="на 93% мат❗️")
        # await message.forward(chat_id=CHAT_ID_MODERATORS)
        return

    # await app.send_message(chat_id=CHAT_ID_MODERATORS, text="мата нет")
    # await message.forward(chat_id=CHAT_ID_MODERATORS)





async def run_pyrogram():
    # app = Client("my_account", PYRO_API_ID, PYRO_API_HASH)
    await app.start()

    print("пиро работает")
    
    async with session_maker() as session:
        
        check_word_query = select(Badphrases.phrase_text)
        result = await session.execute(check_word_query)
        result = result.all()
        df = pd.DataFrame(result)
        bad_words = []
        for i, row in df.iterrows():
            bad_words.append(row['phrase_text'])
        async def scheduled_badwords_autochecker():
            await badwords_autochecker(app, bad_words)
        
        scheduler = AsyncIOScheduler()
        scheduler.add_job(scheduled_badwords_autochecker, 'interval', minutes=INTERVAL_MIN)
        scheduler.start()
        await badwords_autochecker(app, bad_words)
    
    my_handler = MessageHandler(pyro_main_handler)
    app.add_handler(my_handler)
