import logging
from aiogram import F, Router, types, Bot
import os
import pandas as pd
from icecream import ic

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Badphrases
from filters.admins import AdminFilter





router = Router()

# @router.message(F.document, ~AdminFilter(), F.chat.type == "group")
@router.message()
async def badwords_handler(message: types.Message, session: AsyncSession, bot: Bot):
    print('авмвм')
    check_word_query = select(Badphrases.phrase_text)
    result = await session.execute(check_word_query)
    result = result.all()
    
    
    
    
    df = pd.DataFrame(result)
    bad_words = []
    for i, row in df.iterrows():
        bad_words.append(row['phrase_text'])
        
        
    if message.text in bad_words:
        await message.answer('ты сматерился')
    else:
        await message.answer('мата нет')