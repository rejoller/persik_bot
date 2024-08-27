from aiogram import Router


def setup_routers() -> Router:

    from handlers import start_command, badwords_file_handler, badword_handler
    



    router = Router()

    router.include_router(start_command.router)
    router.include_router(badword_handler.router)
    router.include_router(badwords_file_handler.router)


    return router
