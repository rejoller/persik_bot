import aiohttp

from config import SPAM_CHECKER_API


async def api_spam_check(user_id):
    answer = []
    async with aiohttp.ClientSession() as session:
        async with session.get(SPAM_CHECKER_API+f'id={user_id}') as resp:
            res = await resp.json()
            answer.append(int(res['offenses']))
            answer.append(res['spam_factor'])
            
    return answer