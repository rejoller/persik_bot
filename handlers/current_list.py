import os
from aiogram import Router, F
from aiogram.types import Message, FSInputFile
from aiogram.fsm.context import FSMContext
from aiogram.filters import Command




from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

import pandas as pd


from database.models import Badphrases



router = Router()


@router.message(Command('current_list'), F.chat.type == 'private')
async def handle_currentlist(message: Message, state: FSMContext, session: AsyncSession):
    await state.clear()
    
    current_list_query = select(Badphrases.phrase_text)
    result = await session.execute(current_list_query)
    result = result.all()
    
    df = pd.DataFrame(result)
    
    current_list = df['phrase_text']
    directory = 'saved_data'
    filename = 'текущий_список.xlsx'
    
    destination = os.join(os.getcwd(), directory, filename)
    
    writer = pd.ExcelWriter(destination, engine='xlsxwriter')
    
    df.to_excel(writer, index=False, sheet_name='список')
    
    workbook = writer.book
    worksheet = writer.sheets['список']
    for i, col in enumerate(df.columns):
        width = max(df[col].apply(lambda x: len(str(x))).max(), len(col))
        worksheet.set_column(i, i, width)
    writer.close()

    try:
        await message.answer_document(document=FSInputFile('текущий_список.xlsx'), caption=f'Текущий список\n{len(df)} матов')
    except Exception as e:
        await message.answer(f'Ошибка при отправке файла{e}')
