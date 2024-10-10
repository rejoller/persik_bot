import logging
from config import (
    CHAT_ID_MODERATORS,
    INTERVAL_MIN,
    PYRO_API_HASH,
    PYRO_API_ID,
    TARGET_CHANNEL_ID,
    TARGET_CHAT_ID,
    MODEL_DESTINATION_5,
    VECT_DESTINATION_5,
)
from database.models import Badphrases
from database.engine import session_maker
from sqlalchemy import select
import pandas as pd
from utils.unidecoder import unidecoder
from utils.spamcheckerv5.spamchecker import spamchecker5
import re
import string
from nltk.stem import SnowballStemmer
from fuzzywuzzy import process
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




async def full_phrase_analyse(message_text, bad_words):
    for word in bad_words:
        if re.search(rf"\b{word}\b", message_text, re.IGNORECASE):
            return word



async def badwords_autochecker(app, bad_words=None, unidecoded_bad_words=None):
    async for message in app.get_chat_history(chat_id=TARGET_CHAT_ID, limit=15):
        await asyncio.sleep(1)

        message_text = message.text.lower() if message.text else ""
        message_caption = message.caption.lower() if message.caption else ""

        if message.animation:
            await app.send_message(
                chat_id=CHAT_ID_MODERATORS,
                text="обнаружена анимация",
            )
            await message.forward(chat_id=CHAT_ID_MODERATORS)
            try:
                await message.delete()
            except Exception as e:
                logging.error(f"Ошибка при удалении сообщения с анимацией: {e}")
            return

        if any(word in message_text.split() for word in bad_words) or any(
            word in message_caption.split() for word in bad_words
        ):
            found_words = [
                word
                for word in bad_words
                if word in message_text.split() or word in message_caption.split()
            ]

            await app.send_message(
                chat_id=CHAT_ID_MODERATORS,
                text=f"Найден мат\n{', '.join(found_words)}",
            )
            await message.forward(chat_id=CHAT_ID_MODERATORS)
            try:
                await message.delete()
                # await app.ban_chat_member(chat_id=CHAT_ID_MODERATORS, user_id = message.from_user.id, until_date =dt.now() + timedelta(days=1))
            except Exception as e:
                logging.error(f"Ошибка при удалении сообщения: {e}")
            return
        
        
        
        if message_text:
            found_phrase = await full_phrase_analyse(message_text, bad_words)
            if found_phrase:
                await app.send_message(
                    chat_id=CHAT_ID_MODERATORS,
                    text=f"Найден мат\n{found_phrase}",
                )
                await message.forward(chat_id=CHAT_ID_MODERATORS)
                # try:
                #     # await app.ban_chat_member(chat_id=CHAT_ID_MODERATORS, user_id = message.from_user.id, until_date =dt.now() + timedelta(days=1))
                #     await message.delete()
                # except Exception as e:
                #     logging.error(f"Ошибка при удалении сообщения: {e}")
                pass
            
            
                
                
        if message_caption:
            found_phrase = await full_phrase_analyse(message_caption, bad_words)
            if found_phrase:
                await app.send_message(
                    chat_id=CHAT_ID_MODERATORS,
                    text=f"Найден мат\n{found_phrase}",
                )
                await message.forward(chat_id=CHAT_ID_MODERATORS)
                # try:
                #     # await app.ban_chat_member(chat_id=CHAT_ID_MODERATORS, user_id = message.from_user.id, until_date =dt.now() + timedelta(days=1))
                #     await message.delete()
                # except Exception as e:
                #     logging.error(f"Ошибка при удалении сообщения: {e}")
                pass
                
                
                

        decoded_message_text = [unidecoder(word) for word in message_text.split()]
        decoded_message_caption = [unidecoder(word) for word in message_caption.split()]

        if any(word in decoded_message_text for word in unidecoded_bad_words) or any(
            word in decoded_message_caption for word in unidecoded_bad_words
        ):
            found_words = [
                word
                for word in unidecoded_bad_words
                if word in decoded_message_text or word in decoded_message_caption
            ]

            await app.send_message(
                chat_id=CHAT_ID_MODERATORS,
                text=f"Найден мат с помощью юнидекодера\n{', '.join(found_words)}",
            )
            await message.forward(chat_id=CHAT_ID_MODERATORS)
            try:
                # await app.ban_chat_member(chat_id=CHAT_ID_MODERATORS, user_id = message.from_user.id, until_date =dt.now() + timedelta(days=1))
                await message.delete()
            except Exception as e:
                logging.error(f"Ошибка при удалении сообщения: {e}")
            return
        
        
        
        if decoded_message_text:
            found_phrase = await full_phrase_analyse(decoded_message_text, unidecoded_bad_words)
            if found_phrase:
                await app.send_message(
                    chat_id=CHAT_ID_MODERATORS,
                    text=f"Найден мат с помощью юнидекодера\n{found_phrase}",
                )
                await message.forward(chat_id=CHAT_ID_MODERATORS)
                # try:
                #     # await app.ban_chat_member(chat_id=CHAT_ID_MODERATORS, user_id = message.from_user.id, until_date =dt.now() + timedelta(days=1))
                #     await message.delete()
                # except Exception as e:
                #     logging.error(f"Ошибка при удалении сообщения: {e}")
                pass
                
                
        if decoded_message_caption:
            found_phrase = await full_phrase_analyse(decoded_message_caption, unidecoded_bad_words)
            if found_phrase:
                await app.send_message(
                    chat_id=CHAT_ID_MODERATORS,
                    text=f"Найден мат с помощью юнидекодера\n{found_phrase}",
                )
                await message.forward(chat_id=CHAT_ID_MODERATORS)
                # try:
                #     # await app.ban_chat_member(chat_id=CHAT_ID_MODERATORS, user_id = message.from_user.id, until_date =dt.now() + timedelta(days=1))
                #     await message.delete()
                # except Exception as e:
                #     logging.error(f"Ошибка при удалении сообщения: {e}")
                pass


async def check_message_for_bad_words(message_words, bad_words, threshold=70):
    for word in message_words:
        similar_bad_words = await find_similar_words(word, bad_words, threshold)
        if similar_bad_words:
            return True
    return False


async def pyro_main_handler(app, message):
    async with session_maker() as session:
        check_word_query = select(
            Badphrases.phrase_text, Badphrases.unicoded_phrase_text
        )
        result = await session.execute(check_word_query)
    result = result.all()

    df = pd.DataFrame(result)

    bad_words = []
    unidecoded_bad_words = []

    for i, row in df.iterrows():
        bad_words.append(row["phrase_text"])
        unidecoded_bad_words.append(row["unicoded_phrase_text"])

    message_text = message.text.lower() if message.text else ""
    message_caption = message.caption.lower() if message.caption else ""

    if message.animation:
        await app.send_message(
            chat_id=CHAT_ID_MODERATORS,
            text="обнаружена анимация",
        )
        await message.forward(chat_id=CHAT_ID_MODERATORS)

        try:
            await message.delete()
        except Exception as e:
            logging.error(f"Ошибка при удалении сообщения с анимацие: {e}")
        return

    if any(word in message_text.split() for word in bad_words) or any(
        word in message_caption.split() for word in bad_words
    ):
        found_words = [
            word
            for word in bad_words
            if word in message_text.split() or word in message_caption.split()
        ]
        await app.send_message(
            chat_id=CHAT_ID_MODERATORS,
            text=f"Найден мат\n{', '.join(found_words)}",
        )
        await message.forward(chat_id=CHAT_ID_MODERATORS)
        try:
            # await app.ban_chat_member(chat_id=CHAT_ID_MODERATORS, user_id = message.from_user.id, until_date =dt.now() + timedelta(days=1))
            await message.delete()
        except Exception as e:
            logging.error(f"Ошибка при удалении сообщения: {e}")
        return

    decoded_message_text = [unidecoder(word) for word in message_text.split()]
    decoded_message_caption = [unidecoder(word) for word in message_caption.split()]

    if any(word in decoded_message_text for word in unidecoded_bad_words) or any(
        word in decoded_message_caption for word in unidecoded_bad_words
    ):
        found_words = [
            word
            for word in unidecoded_bad_words
            if word in decoded_message_text or word in decoded_message_caption
        ]
        await app.send_message(
            chat_id=CHAT_ID_MODERATORS,
            text=f"Найден мат с помощью юнидекодера\n{', '.join(found_words)}",
        )
        await message.forward(chat_id=CHAT_ID_MODERATORS)
        try:
            await message.delete()
        except Exception as e:
            logging.error(f"Ошибка при удалении сообщения: {e}")
        return

    if message_text:
        spam_checkv5 = await spamchecker5(message_text)
        if spam_checkv5 == 1:
            await app.send_message(
                chat_id=CHAT_ID_MODERATORS,
                text="обнаружен спам с помощью алгоритма Feodor Cherepakhin",
            )
            await message.forward(chat_id=CHAT_ID_MODERATORS)

            try:
                await message.delete()
            except Exception as e:
                logging.error(f"Ошибка при удалении сообщения со спамом: {e}")
            return

        for word in bad_words:
            found_phrase = await full_phrase_analyse(message_text, bad_words)
            if found_phrase:
                await app.send_message(
                    chat_id=CHAT_ID_MODERATORS,
                    text=f"Найден мат\n{found_phrase}",
                )
                await message.forward(chat_id=CHAT_ID_MODERATORS)
                # try:
                #     # await app.ban_chat_member(chat_id=CHAT_ID_MODERATORS, user_id = message.from_user.id, until_date =dt.now() + timedelta(days=1))
                #     await message.delete()
                # except Exception as e:
                #     logging.error(f"Ошибка при удалении сообщения: {e}")
                pass

    if message_caption:
        spam_checkv5 = await spamchecker5(message_caption)
        if spam_checkv5 == 1:
            await app.send_message(
                chat_id=CHAT_ID_MODERATORS,
                text="обнаружен спам с помощью алгоритма Feodor Cherepakhin",
            )
            await message.forward(chat_id=CHAT_ID_MODERATORS)

            try:
                await message.delete()
            except Exception as e:
                logging.error(f"Ошибка при удалении сообщения со спамом: {e}")

            return

        for word in bad_words:
            found_phrase = await full_phrase_analyse(message_caption, bad_words)
            if found_phrase:
                await app.send_message(
                    chat_id=CHAT_ID_MODERATORS,
                    text=f"Ниден мат\n{found_phrase}",
                )
                await message.forward(chat_id=CHAT_ID_MODERATORS)
                # try:
                #     # await app.ban_chat_member(chat_id=CHAT_ID_MODERATORS, user_id = message.from_user.id, until_date =dt.now() + timedelta(days=1))
                #     await message.delete()
                # except Exception as e:
                #     logging.error(f"Ошибка при удалении сообщения: {e}")
                pass
            
            
        if decoded_message_text:
            found_phrase = await full_phrase_analyse(decoded_message_text, unidecoded_bad_words)
            if found_phrase:
                await app.send_message(
                    chat_id=CHAT_ID_MODERATORS,
                    text=f"Ниден мат с помощью юнидекодера\n{found_phrase}",
                )
                await message.forward(chat_id=CHAT_ID_MODERATORS)
                # try:
                #     # await app.ban_chat_member(chat_id=CHAT_ID_MODERATORS, user_id = message.from_user.id, until_date =dt.now() + timedelta(days=1))
                #     await message.delete()
                # except Exception as e:
                #     logging.error(f"Ошибка при удалении сообщения: {e}")
                pass

                
                
        if decoded_message_caption:
            found_phrase = await full_phrase_analyse(decoded_message_caption, unidecoded_bad_words)
            if found_phrase:
                await app.send_message(
                    chat_id=CHAT_ID_MODERATORS,
                    text=f"Ниден мат с помощью юнидекодера\n{found_phrase}",
                )
                await message.forward(chat_id=CHAT_ID_MODERATORS)
                # try:
                #     # await app.ban_chat_member(chat_id=CHAT_ID_MODERATORS, user_id = message.from_user.id, until_date =dt.now() + timedelta(days=1))
                #     await message.delete()
                # except Exception as e:
                #     logging.error(f"Ошибка при удалении сообщения: {e}")
                pass



async def run_pyrogram():
    app = Client("my_account", PYRO_API_ID, PYRO_API_HASH)
    await app.start()

    print("пиро работает")

    async with session_maker() as session:
        check_word_query = select(
            Badphrases.phrase_text, Badphrases.unicoded_phrase_text
        )
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
    # my_handler = MessageHandler(pyro_main_handler, filters.chat([964635576]))
    my_handler = MessageHandler(
        pyro_main_handler, filters.chat([TARGET_CHAT_ID, CHAT_ID_MODERATORS])
    )
    app.add_handler(my_handler)