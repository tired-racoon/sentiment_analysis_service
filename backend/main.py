from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from backend.model import SentimentAnalyzer
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
import time
import io

app = FastAPI()
analyzer = SentimentAnalyzer()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LABEL_MAPPING = {
    "LABEL_0": "bad",
    "LABEL_1": "neutral",
    "LABEL_2": "good",
}

def check_emotion(text):
    posit = [':)', ':d', 'поздрав', 'рождения', 'подар', 'днюхой', '=)']
    bad = [':(', '=(', '...', 'проблема', 'спор', 'эх,', 'эх)']
    for word in posit:
        if word in text.lower():
            return 'good'
    for word in bad:
        if word in text.lower():
            return 'bad'
    return None

class TextRequest(BaseModel):
    text: str

@app.post("/analyze/")
async def analyze_text(request: TextRequest):
    sentiment, confidence = analyzer.predict(request.text)
    sentiment = LABEL_MAPPING.get(sentiment, "unknown")
    
    # Проверка эмоций и возможная замена результата
    emotion_fix = check_emotion(request.text)
    if emotion_fix:
        sentiment = emotion_fix
    
    return {"sentiment": sentiment, "confidence": confidence}

@app.post("/analyze-file/")
async def analyze_file(file: UploadFile = File(...)):
    contents = await file.read()  # Читаем содержимое файла в байты
    df = pd.read_excel(io.BytesIO(contents))  # Создаём объект BytesIO и передаём его в read_excel

    if "MessageText" not in df.columns:
        return {"error": "Файл должен содержать колонку 'MessageText'"}

    df["text"] = df["MessageText"].fillna("").apply(
        lambda x: BeautifulSoup(x, "html.parser").get_text(strip=True)[:512]
    )

    df["sentiment"], df["confidence"] = zip(*df["text"].map(analyzer.predict))
    df["sentiment"] = df["sentiment"].map(LABEL_MAPPING)

    df["emotion_fix"] = df["text"].map(check_emotion)
    df["sentiment"] = df.apply(lambda row: row["emotion_fix"] if row["emotion_fix"] else row["sentiment"], axis=1)

    result = df[["UserSenderId", "SubmitDate", "MessageText", "sentiment", "confidence"]]

    return result.to_dict(orient="records")
