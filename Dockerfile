FROM python:3.10

RUN apt-get update \
    && apt-get install -y \
        cmake libsm6 libxext6 libxrender-dev protobuf-compiler git \
    && rm -r /var/lib/apt/lists/*

WORKDIR /app

# Клонируем GigaAM и устанавливаем его
RUN git clone https://github.com/salute-developers/GigaAM.git /app/GigaAM \
    && pip install -e /app/GigaAM

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код бота
COPY . .

CMD ["python", "bot.py"]
