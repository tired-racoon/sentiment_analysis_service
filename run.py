import threading
import time
import subprocess
import uvicorn
from backend.main import app

# Функция для запуска FastAPI
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8888)

# Функция для запуска Streamlit
def run_streamlit():
    # Ждем, пока запустится FastAPI (чтобы избежать ошибок подключения)
    time.sleep(2)  
    subprocess.run(["streamlit", "run", "frontend/app.py", "--server.port", "8889", "--server.address", "0.0.0.0"])

# Создаем потоки
t1 = threading.Thread(target=run_fastapi)
t2 = threading.Thread(target=run_streamlit)

# Запускаем оба потока
t1.start()
t2.start()

# Ждем завершения обоих потоков
t1.join()
t2.join()
