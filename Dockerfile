FROM python:3.9

EXPOSE 80
WORKDIR /tgbot

COPY requirements.txt .
COPY app/ .
COPY data/ .

RUN pip install -r /requirements.txt
RUN pip install --user torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

CMD [ "python", "./app/bot/bot.py" ]