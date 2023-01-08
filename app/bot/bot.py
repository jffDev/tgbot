import logging

from aiogram import Bot, Dispatcher, executor, types

import inline_keyboard as ik
import config

from app.service.predict import predict

bot = Bot(token=config.BOT_API_TOKEN)
dp = Dispatcher(bot=bot)

logging.basicConfig(level=logging.INFO)


@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(message: types.Message):
    img_name = str(message.message_id) + '.jpg'
    await message.photo[-1].download(img_name)
    filename = predict(img_name)['filename']
    img = open(filename, "rb")
    await bot.send_photo(message.from_user.id, photo=img)


@dp.message_handler(commands='image')
async def show_img(message: types.Message):
    await message.answer(text='messages.img',
                         reply_markup=ik.IMAGE)


@dp.message_handler(commands='info')
async def show_img(message: types.Message):
    await message.answer(text='messages.info',
                         reply_markup=ik.INFO)


@dp.callback_query_handler(text='image')
async def process_callback_img(callback_query: types.CallbackQuery):
    logging.info(msg='process_callback_img')
    img = open('../web_app/out.png', "rb")
    # print(img)
    await bot.answer_callback_query(callback_query.id)
    await bot.send_photo(callback_query.from_user.id, photo=img)


@dp.callback_query_handler(text='info')
async def process_callback_info(callback_query: types.CallbackQuery):
    logging.info(msg='process_callback_info')
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(
        callback_query.from_user.id,
        text='messages.info',
        reply_markup=ik.INFO
    )


async def start_handler(message: types.Message):
    await message.answer(
        text="Choose btn",
        reply_markup=ik.IMAGE)


if __name__ == '__main__':
    dp.register_message_handler(start_handler, commands={"start", "restart"})
    executor.start_polling(dp, skip_updates=True)
