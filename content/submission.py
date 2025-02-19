import pandas as pd
import io
from backend.model import analyzer
from bs4 import BeautifulSoup
from total import check_emotion

LABEL_MAPPING = {
    "LABEL_0": "B",
    "LABEL_1": "N",
    "LABEL_2": "G",
}

def main():
    try:
        df = pd.read_csv('/contest/data.csv')
    except Exception as e:
        print('Ошибка чтения data.csv:', e)
        return
    if 'UserSenderId' not in df.columns or 'MessageText' not in df.columns:
        print('Отсутствуют необходимые колонки')
        return

    df["text"] = df["MessageText"].fillna("").apply(lambda x: BeautifulSoup(x, "html.parser").get_text(strip=True)[:512])
    df["sentiment"], _ = zip(*df["text"].map(analyzer.predict))
    df["sentiment"] = df["sentiment"].map(LABEL_MAPPING)
    df["emotion_fix"] = df["text"].map(check_emotion)
    df["Class"] = df.apply(lambda row: row["emotion_fix"] if row["emotion_fix"] else row["sentiment"], axis=1)
    
    submission = df[['UserSenderId', 'Class']]
    submission.to_csv('/contest/submission.csv', index=False)
    print('Submission успешно сохранён')

if __name__ == '__main__':
    main()
