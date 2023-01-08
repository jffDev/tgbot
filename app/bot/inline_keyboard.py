from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

BTN_IMG = InlineKeyboardButton('Image', callback_data='image')
BTN_INFO = InlineKeyboardButton('Info', callback_data='info')

IMAGE = InlineKeyboardMarkup().add(BTN_IMG, BTN_INFO)
INFO = InlineKeyboardMarkup().add(BTN_INFO, BTN_IMG)