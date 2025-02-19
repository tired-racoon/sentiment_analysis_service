import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import time
import io
from transformers import pipeline
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
from starlette.responses import JSONResponse
from fastapi.testclient import TestClient
from backend.model import analyzer
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter

STOP_WORDS = {"и", "в", "на", "с", "по", "за", "к", "под", "от", "что", "как", "для", "то", "а", "ли", "будет", "меня", "будет", "пока", "может", "уже", "раз", "мы"
              "не", "но", "до", "из", "у", "же", "так", "вы", "он", "она", "они", "это", "все", "при", "я", "есть", "днём", "не", "почему", "только", "непример", "нас"
              "мы", "мне", "мой", "моя", "моё", "мои", "твой", "твоя", "твоё", "твои", "-", "спасибо", "ты", "очень", "ни", "их", "всем", "такие", "их", "или", "если", "эти", "кроме", "про"}

# Запуск FastAPI внутри Streamlit
app = FastAPI()

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
    posit = [':)', ':d', 'поздрав', 'рождения', 'подар', 'днюхой', '=)', 'счасть', ' норм', 'спасибо', 'отличн', 'молодец', 'молодцы']
    bad = [':(', '=(', 'проблема', 'спор', 'эх,', 'эх)', 'forbid', 'нечего']
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
def analyze_text(request: TextRequest):
    sentiment, confidence = analyzer.predict(request.text)
    sentiment = LABEL_MAPPING.get(sentiment, "unknown")
    emotion_fix = check_emotion(request.text)
    if emotion_fix:
        sentiment = emotion_fix
    return JSONResponse(content={"sentiment": sentiment, "confidence": confidence})

@app.post("/analyze-file/")
async def analyze_file(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))
    if "MessageText" not in df.columns:
        return JSONResponse(content={"error": "Файл должен содержать колонку 'MessageText'"})
    df["text"] = df["MessageText"].fillna("").apply(lambda x: BeautifulSoup(x, "html.parser").get_text(strip=True)[:512])
    df["sentiment"], df["confidence"] = zip(*df["text"].map(analyzer.predict))
    df["sentiment"] = df["sentiment"].map(LABEL_MAPPING)
    df["emotion_fix"] = df["text"].map(check_emotion)
    df["sentiment"] = df.apply(lambda row: row["emotion_fix"] if row["emotion_fix"] else row["sentiment"], axis=1)

    result = df[["UserSenderId", "MessageText", "sentiment", "confidence"]]
    return JSONResponse(content=result.to_dict(orient="records"))

client = TestClient(app)

st.title("🔍 Анализ тональности текста")
if "df" not in st.session_state:
    st.session_state.df = None

text_input = st.text_area("Введите текст для анализа:")
if st.button("Анализировать текст"):
    if text_input.strip():
        response = client.post("/analyze/", json={"text": text_input})
        if response.status_code == 200:
            data = response.json()
            st.write(f"**Тональность:** {data['sentiment']}")
            st.write(f"**Уверенность модели:** {round(data['confidence'] * 100, 2)}%")
        else:
            st.error("Ошибка API!")
    else:
        st.warning("Введите текст!")

uploaded_file = st.file_uploader("Загрузите XLSX-файл с колонкой 'MessageText'", type=["xlsx"])
if uploaded_file and st.button("Анализировать файл"):
    files = {"file": ("file.xlsx", uploaded_file.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    response = client.post("/analyze-file/", files=files)
    if response.status_code == 200:
        st.session_state.df = pd.DataFrame(response.json())
    else:
        st.error("Ошибка API!")

if st.session_state.df is not None:
    df = st.session_state.df
    df["MessageText"] = df["MessageText"].apply(lambda x: BeautifulSoup(x, "html.parser").get_text(strip=True))
    fig = px.pie(df, names="sentiment", title="Распределение тональности", hole=0.3)
    st.plotly_chart(fig)
    keyword = st.text_input("🔍 Поиск по ключевым словам:", "")
    all_classes = ["Все классы"] + list(df["sentiment"].unique())
    selected_class = st.selectbox("Фильтр по классу:", all_classes)
    filtered_df = df
    if keyword:
        filtered_df = filtered_df[filtered_df["MessageText"].str.contains(keyword, case=False, na=False)]
    if selected_class != "Все классы":
        filtered_df = filtered_df[filtered_df["sentiment"] == selected_class]
    st.write(filtered_df)


    # Функция для подсчета частоты слов по классам
    def get_top_words_by_class(df, top_n=15):
        word_counts = []
        
        for sentiment_class in df["sentiment"].unique():
            subset = df[df["sentiment"] == sentiment_class]
        
            # Объединяем весь текст этого класса и разбиваем на слова
            words = " ".join(subset["MessageText"]).lower().split()

            # Фильтруем слова, убирая стоп-слова
            words = [word for word in words if word not in STOP_WORDS]

            word_freq = Counter(words).most_common(top_n)  # ТОП-10 слов
        
            for word, count in word_freq:
                word_counts.append({"word": word, "count": count, "sentiment": sentiment_class})
    
        return pd.DataFrame(word_counts)

    # Получаем таблицу с частотой слов
    word_df = get_top_words_by_class(df)

    # Строим Treemap
    fig_treemap = px.treemap(word_df, 
                          path=["sentiment", "word"], 
                          values="count", 
                          title="Популярные слова по классам",
                          color="sentiment", 
                          color_discrete_map={"positive": "green", "negative": "red", "neutral": "blue"})

    # Отображаем в Streamlit
    st.plotly_chart(fig_treemap)
