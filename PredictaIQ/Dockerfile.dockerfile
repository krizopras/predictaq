FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY PredictaIQ/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ⚠️ BURASI DEĞİŞTİ: PredictaIQ/ klasörünün içindeki her şeyi kopyala
COPY PredictaIQ/ .

# Model/data directories
RUN mkdir -p /app/models /app/data

EXPOSE 8000

# ⚠️ BURASI DEĞİŞTİ: app.main:app kullan
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
