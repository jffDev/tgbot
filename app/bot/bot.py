import logging, shutil

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ParseMode

import inline_keyboard as ik
import config
from app.data.messages import Messages

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


@dp.message_handler(commands='description')
async def show_img(message: types.Message):
    await message.answer(
        text=Messages.message_description(),
        reply_markup=ik.DESCRIPTION)


@dp.message_handler(commands='info')
async def show_img(message: types.Message):
    await message.answer(
        text=Messages.message_info(),
        reply_markup=ik.INFO)


@dp.message_handler(commands='test')
async def show_img(message: types.Message):
    shutil.copy(config.TEST_IMG_PATH + config.TEST_IMG_NAME, '.')
    filename = predict(config.TEST_IMG_NAME)['filename']
    img = open(filename, "rb")
    await bot.send_photo(message.from_user.id, photo=img)


@dp.callback_query_handler(text='description')
async def process_callback_img(callback_query: types.CallbackQuery):
    logging.info(msg='process callback: description')
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(
        callback_query.from_user.id,
        text=Messages.message_description(),
        reply_markup=ik.INFO
    )


@dp.callback_query_handler(text='test')
async def process_callback_test(callback_query: types.CallbackQuery):
    logging.info(msg='process callback: test')
    shutil.copy(config.TEST_IMG_PATH + config.TEST_IMG_NAME, '.')
    filename = predict(config.TEST_IMG_NAME)['filename']
    img = open(filename, "rb")
    await bot.answer_callback_query(callback_query.id)
    await bot.send_photo(callback_query.from_user.id, photo=img)


@dp.callback_query_handler(text='info')
async def process_callback_info(callback_query: types.CallbackQuery):
    logging.info(msg='process callback: info')
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(
        callback_query.from_user.id,
        text=Messages.message_info(),
        parse_mode=ParseMode.HTML,
        reply_markup=ik.INFO
    )


@dp.message_handler(commands='start')
async def start_handler(message: types.Message):
    await message.answer(
        text="Выберите действие",
        reply_markup=ik.INFO)


if __name__ == '__main__':
    dp.register_message_handler(start_handler, commands={"start"})
    executor.start_polling(dp, skip_updates=True)
