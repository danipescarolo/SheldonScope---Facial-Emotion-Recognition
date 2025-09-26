# --- Base image ---
FROM python:3.11-slim

# --- Installazione librerie di sistema necessarie per OpenCV e MoviePy ---
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Copia requirements e installazione librerie Python ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Patch per FER: commentiamo import di MoviePy per evitare errori ---
RUN sed -i 's/^from moviepy.editor import \*/# from moviepy.editor import */' /usr/local/lib/python3.11/site-packages/fer/classes.py

# --- Imposta la working directory e copia il tuo codice ---
WORKDIR /app
COPY . /app

# --- Comando di default all'avvio del container ---
CMD ["python", "main.py"]
