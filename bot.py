import telebot
import requests
import pandas as pd
import io
import gigaam
from transformers import pipeline
from bs4 import BeautifulSoup
from telebot.types import Message, Voice
from pydub import AudioSegment

TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
API_URL = "http://127.0.0.1:8000"  # URL FastAPI
bot = telebot.TeleBot(TOKEN)
model_name = "v2_rnnt"
asr_model = gigaam.load_model(model_name)

LABEL_MAPPING = {
    "LABEL_0": "bad",
    "LABEL_1": "neutral",
    "LABEL_2": "good",
}

class SentimentAnalyzer:
    def __init__(self):
        self.model = pipeline(model="sismetanin/xlm_roberta_large-ru-sentiment-sentirueval2016", trust_remote_code=True)

    def predict(self, text):
        result = self.model(text)[0]
        return result["label"], result["score"]

analyzer = SentimentAnalyzer()

def check_emotion(text):
    posit = [":)", ":d", "поздрав", "рождения", "подар", "днюхой", "=)", "счасть", " норм", "спасибо", "отличн", "молодец", "молодцы"]
    bad = [":(", "=(", "проблема", "спор", "эх,", "эх)", "forbid", "нечего"]
    for word in posit:
        if word in text.lower():
            return "good"
    for word in bad:
        if word in text.lower():
            return "bad"
    return None

@bot.message_handler(commands=['start'])
def start_message(message: Message):
    bot.send_message(message.chat.id, "Привет! Отправь мне текст или голосовое сообщение, и я проанализирую его тональность.")

@bot.message_handler(content_types=['text'])
def handle_text(message: Message):
    sentiment, confidence = analyzer.predict(message.text)
    sentiment = LABEL_MAPPING.get(sentiment, "unknown")
    emotion_fix = check_emotion(message.text)
    if emotion_fix:
        sentiment = emotion_fix
    
    bot.send_message(message.chat.id, f"Тональность: {sentiment}\nУверенность: {round(confidence * 100, 2)}%")

@bot.message_handler(content_types=['voice'])
def handle_voice(message: Voice):
    file_info = bot.get_file(message.voice.file_id)
    file = bot.download_file(file_info.file_path)
    
    audio = AudioSegment.from_ogg(io.BytesIO(file))
    if len(audio) > 30000:
        audio = audio[:30000]
    
    audio_path = "voice.wav"
    audio.export(audio_path, format="wav")
    
    text = asr_model.transcribe(audio_path)
    if not text:
        bot.send_message(message.chat.id, "Не удалось распознать речь.")
        return
    
    sentiment, confidence = analyzer.predict(text)
    sentiment = LABEL_MAPPING.get(sentiment, "unknown")
    emotion_fix = check_emotion(text)
    if emotion_fix:
        sentiment = emotion_fix
    
    bot.send_message(message.chat.id, f"Распознанный текст: {text}\nТональность: {sentiment}\nУверенность: {round(confidence * 100, 2)}%")

@bot.message_handler(content_types=['document'])
def handle_document(message: Message):
    file_info = bot.get_file(message.document.file_id)
    file = bot.download_file(file_info.file_path)
    df = pd.read_excel(io.BytesIO(file))
    if "MessageText" not in df.columns:
        bot.send_message(message.chat.id, "Файл должен содержать колонку 'MessageText'")
        return
    
    df["text"] = df["MessageText"].fillna("").apply(lambda x: BeautifulSoup(x, "html.parser").get_text(strip=True)[:512])
    df["sentiment"], df["confidence"] = zip(*df["text"].map(analyzer.predict))
    df["sentiment"] = df["sentiment"].map(LABEL_MAPPING)
    df["emotion_fix"] = df["text"].map(check_emotion)
    df["sentiment"] = df.apply(lambda row: row["emotion_fix"] if row["emotion_fix"] else row["sentiment"], axis=1)
    
    response_text = "Результаты анализа:\n"
    for index, row in df.iterrows():
        response_text += f"{index + 1}. {row['sentiment']} (уверенность: {round(row['confidence'] * 100, 2)}%)\n"
    bot.send_message(message.chat.id, response_text)

bot.polling(none_stop=True)
