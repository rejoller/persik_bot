import aiohttp

from config import SPAM_CHECKER_API


async def api_spam_check(user_id):
    async with aiohttp.ClientSession() as session:
        async with session.get(SPAM_CHECKER_API+f'{user_id}') as resp:
            res = await resp.json()   

    return res