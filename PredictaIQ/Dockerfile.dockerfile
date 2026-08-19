FROM python:3.11-slim

WORKDIR /app

# Gerekli sistem bağımlılıklarını kur
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Bağımlılık dosyasını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm uygulama kodlarını kopyala
COPY . .

EXPOSE 8000

# Shell form kullanarak $PORT değişkeninin dinamik okunmasını sağla
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
