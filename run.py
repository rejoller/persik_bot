import logging

from sqlalchemy import select
from database.db import DataBaseSession
from database.models import Badphrases
from handlers import setup_routers
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.redis import RedisStorage
from logger.logging_config import setup_logging
from logger.logging_middleware import LoggingMiddleware
from config import BOT_TOKEN, PYRO_API_ID, PYRO_API_HASH, CHAT_ID_MODERATORS
import asyncio
from database.engine import session_maker



from pyro_handlers.main_handler import run_pyrogram
import pandas as pd

from utils.sanya_spamchecker_loader import train_and_save_model
from utils.spamcheckerv3.spamchecker_loader import train_and_save_model3
from utils.spamcheckerv4.spamchecker_loader import train_and_save_model4

bot = Bot(BOT_TOKEN)




storage = RedisStorage.from_url("redis://localhost:6379/7")











async def run_aiogram():
    setup_logging()
    # train_and_save_model()
    # train_and_save_model3()
    # train_and_save_model4()
    print('модель загружена')
    import nltk
    nltk.download('punkt_tab')
    dp = Dispatcher(storage=storage)
    
    dp.update.middleware(DataBaseSession(session_pool=session_maker))
    router = setup_routers()
    dp.include_router(router)
    
    dp.message.middleware(LoggingMiddleware())
    from database.engine import create_db
    await create_db()

    print('Бот запущен и готов к приему сообщений')

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types(), skip_updates=True)
    
    
async def main():
    await asyncio.gather(run_aiogram(), run_pyrogram())
    
if __name__ == '__main__':
    asyncio.run(main())