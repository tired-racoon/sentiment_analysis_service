import threading
import time
import subprocess
import uvicorn
from backend.main import app

# Функция для запуска FastAPI
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)

# Функция для запуска Streamlit
def run_streamlit():
    # Ждем, пока запустится FastAPI (чтобы избежать ошибок подключения)
    time.sleep(2)  
    subprocess.run(["streamlit", "run", "frontend/app.py"])

# Создаем потоки
t1 = threading.Thread(target=run_fastapi)
t2 = threading.Thread(target=run_streamlit)

# Запускаем оба потока
t1.start()
t2.start()

# Ждем завершения обоих потоков
t1.join()
t2.join()
