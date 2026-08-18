FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY PredictaIQ/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY . .

# Model/data directories -- NOT: settings.model_path varsayılanı "./models"
# (WORKDIR /app'e göre göreli), yani gerçek yol /app/models. Eskiden burada
# yanlışlıkla /models (kök dizin) oluşturuluyordu ve hiçbir zaman
# kullanılmıyordu. Railway/Render'da kalıcı bir Volume'u /app/models'e
# mount edin, aksi halde her deploy'da eğitilmiş modeller silinir.
RUN mkdir -p /app/models /app/data

EXPOSE 8000

CMD ["uvicorn", "PredictaIQ.main:app", "--host", "0.0.0.0", "--port", "8000"]
