import logging
from aiogram import F, Router, types, Bot
import os
import pandas as pd
import random
from icecream import ic

from database.engine import engine
from database.models import Badphrases
from filters.admins import AdminFilter

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import delete

router = Router()




@router.message(F.document, AdminFilter(), F.chat.type == "private")
async def documents_handler(message: types.Message, session: AsyncSession, bot: Bot):
    document = message.document
    file_name = document.file_name.lower()
    if "список" in file_name:
        directory = 'saved_data'
        if not os.path.exists(directory):
            os.mkdir(directory)
        destination = os.path.join(os.getcwd(), directory, file_name)
        file_info = await bot.get_file(document.file_id)
        await bot.download_file(file_info.file_path, destination)

        df = pd.read_excel(destination)
        
        await message.answer('Файл загружен на сервер, пробую загрузить в базу данных🧐')
        
        dele = delete(Badphrases)
        await session.execute(dele)
        await session.commit()
        await message.answer('удален прежний список')
        
        
        try:
            for i, row in df.iterrows():
                
                save_query = insert(Badphrases).values(phrase_text = row['мат'])
                await session.execute(save_query)
            await session.commit()
            await message.answer('Новый список загружен в базу данных')
        except SQLAlchemyError as db_err:
            logging.error(
                f"Ошибка базы данных при сохранении фразы: {db_err}"
            )
            
            
        
    if "список" not in file_name:
        await message.answer('Имя файла не подходит для сохранения')