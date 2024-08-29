from aiogram import Router




def setup_routers() -> Router:

    from handlers import start_command, badwords_file_handler, one_word, more_words
    from callbacks import one_word_cb, more_words_cb
    



    router = Router()
    router.include_router(one_word_cb.router)
    router.include_router(more_words_cb.router)
    router.include_router(start_command.router)
    # router.include_router(badword_handler.router)
    # router.include_router(badwords_file_handler.router)
    router.include_router(one_word.router)
    router.include_router(more_words.router)
    


    return router
