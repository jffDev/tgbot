from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

BTN_DESCRIPTION = InlineKeyboardButton('Как работает', callback_data='description')
BTN_INFO = InlineKeyboardButton('Описание бота', callback_data='info')
BTN_TEST = InlineKeyboardButton('Тест', callback_data='test')

DESCRIPTION = InlineKeyboardMarkup().add(BTN_DESCRIPTION, BTN_INFO, BTN_TEST)
INFO = InlineKeyboardMarkup().add(BTN_DESCRIPTION, BTN_INFO, BTN_TEST)