class Messages:

    @staticmethod
    def message_description():
        return "Для работы бота необходимо послать изображение. После обработки нейросетью на базе обученной модели " \
               "FAST R-CNN вернется изображение с выделенным подписанным фруктом"

    @staticmethod
    def message_info():
        return 'Данный бот использует модель FAST R-CNN по распознаванию яблок, апельсинов и бананов.\n' \
               'Также распознаются команды:\n' \
               '/start\n' \
               '/info\n' \
               '/test\n' \
               '/description'
