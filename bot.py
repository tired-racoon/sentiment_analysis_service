import asyncio
import io
import gigaam
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import CommandStart
from pydub import AudioSegment
from backend.model import analyzer

# Инициализация
TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
model = gigaam.load_model("v2_rnnt")
bot = Bot(token=TOKEN)
dp = Dispatcher()

LABEL_MAPPING = {
    "LABEL_0": "bad",
    "LABEL_1": "neutral",
    "LABEL_2": "good",
}

# Функция обрезки аудио до 30 секунд
def trim_audio(file: bytes) -> bytes:
    audio = AudioSegment.from_file(io.BytesIO(file))
    if len(audio) > 30000:
        audio = audio[:30000]  # Обрезка до 30 секунд
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue()

# Функция анализа текста
def analyze_text(text: str):
    sentiment, confidence = analyzer.predict(text)
    sentiment = LABEL_MAPPING.get(sentiment, "unknown")
    return sentiment, round(confidence * 100, 2)

# Обработчик команды /start
@dp.message(CommandStart())
async def start_handler(message: Message):
    await message.answer("Привет! Отправь мне текст или голосовое сообщение, и я определю его тональность.")

# Обработчик текстовых сообщений
@dp.message()
async def text_handler(message: Message):
    sentiment, confidence = analyze_text(message.text)
    await message.answer(f"**Тональность:** {sentiment}\n**Уверенность:** {confidence}%")

# Обработчик голосовых сообщений
@dp.message(content_types=types.ContentType.VOICE)
async def voice_handler(message: Message):
    voice = await message.voice.get_file()
    file = await bot.download_file(voice.file_path)
    trimmed_audio = trim_audio(file.read())
    text = model.transcribe(io.BytesIO(trimmed_audio))
    
    if not text.strip():
        await message.answer("Не удалось распознать голосовое сообщение.")
        return
    
    sentiment, confidence = analyze_text(text)
    await message.answer(f"Распознанный текст: {text}\n**Тональность:** {sentiment}\n**Уверенность:** {confidence}%")

# Запуск бота
async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
