# PredictaIQ — Deployment

Mimari:

```
Netlify (frontend/, React/Vite)  ── GET /matches, /predictions, /odds ──►  FastAPI backend
                                                                             (Railway / Render / Fly)
                                                                                    │
                                                                                    ▼
                                                                            Supabase PostgreSQL
                                                                                    ▲
                                                                                    │
GitHub Actions (cron) ── POST /admin/sync/live (15 dk) ────────────────────────────┤
                     └── POST /admin/models/train (günlük) ────────────────────────┘
```

Netlify Functions Python'u desteklemediği (sadece JS/TS/Go) ve bu projenin
xgboost/lightgbm/catboost gibi native Python bağımlılıkları olduğu için
backend Netlify'da DEĞİL, ayrı bir Python host'unda (Railway/Render/Fly)
çalışır. Netlify sadece `frontend/`'i statik olarak sunar. Supabase bu
mimaride sadece PostgreSQL sağlayıcısıdır — Edge Function kullanılmıyor.

## 1) Supabase (PostgreSQL)

1. supabase.com üzerinde yeni proje oluştur.
2. Project Settings → Database → Connection string'i al (URI formatı,
   `?sslmode=require` ekle).
3. Bu değeri backend'in `DATABASE_URL` ortam değişkenine ver.

Tablolar backend ilk açıldığında otomatik oluşturulur (`Base.metadata.create_all`,
`app/main.py`). Ayrıca alembic migration'ları varsa (`alembic upgrade head`)
onları da çalıştırabilirsin.

## 2) Backend (Railway örneği, `railway.toml` hazır)

1. Railway'de yeni proje → "Deploy from GitHub repo" → bu repo.
2. Root olarak proje kökünü seç (Dockerfile: `Dockerfile.dockerfile`).
3. **Volume ekle** ve `/app/models`'e mount et — aksi halde her deploy'da
   eğitilmiş modeller silinir, `MODEL_PATH` env değişkeni varsayılan
   olarak `./models` (yani `/app/models`) kullanır.
4. Environment variable'ları `.env.example`'a göre gir. En kritik olanlar:
   - `DATABASE_URL` → Supabase connection string
   - `SPORTSDATA_API_KEY` → sportsdata.io anahtarın
   - `ADMIN_API_KEY` → uzun rastgele bir secret (GitHub Actions bunu kullanacak)
   - `SECRET_KEY` → uzun rastgele bir secret
   - `CORS_ALLOW_ORIGINS` → Netlify domain'in (örn. `https://predictaiq.netlify.app`)
   - `SYNC_LEAGUES` → örn. `EPL`
5. Deploy sonrası `https://<railway-url>/health` çağrısının `{"status":"healthy"}`
   döndüğünü doğrula.

İlk kurulumda veritabanı boş olacağı için modeller eğitilmemiş durumda
başlar (bu normaldir, hata değildir). Veri toplanıp yeterli maç birikince
(bkz. `MIN_MATCHES_TO_TRAIN_SIMILARITY` / `MIN_MATCHES_TO_TRAIN_ML`)
`/admin/models/train` çağrısı ile eğitim tamamlanır.

## 3) Frontend (Netlify)

1. Netlify'da "Add new site" → bu repo, `netlify.toml` (`frontend/`
   içinde) otomatik algılanır (`base = "frontend"`).
2. Site configuration → Environment variables → `VITE_API_BASE_URL` =
   Railway backend URL'in (örn. `https://predictaiq-production.up.railway.app`).
3. Deploy. `CORS_ALLOW_ORIGINS` backend tarafında bu Netlify domain'ini
   içermezse istekler tarayıcıda CORS hatası verir — Railway env'ini
   güncelleyip yeniden deploy et.

## 4) GitHub Actions (canlı veri + model eğitimi)

Repo → Settings → Secrets and variables → Actions → şu secret'ları ekle:

- `API_BASE_URL` → Railway backend URL'in (sonunda `/` OLMADAN)
- `ADMIN_API_KEY` → backend'deki `ADMIN_API_KEY` ile AYNI değer

İki workflow zaten `.github/workflows/` altında hazır:

- `sync-live.yml` — her 15 dakikada bir `/admin/sync/live`'ı tetikler
  (canlı maç + oran verisini çeker, scheduled/live maçların tahminlerini
  tazeler).
- `train-models.yml` — günde bir kez (03:00 UTC) rating'leri ve
  similarity/ML Engine'i yeniden eğitir, sonucu diske (`MODEL_PATH`) ve
  DB'ye (`ModelVersion`) yazar.

İlk deploy'dan sonra her iki workflow'u da "Run workflow" ile elle bir kez
tetiklemen, veritabanını hemen doldurup ilk modeli eğitmen için faydalı
olur.

## 5) İlk veri yüklemesi (opsiyonel ama önerilir)

Similarity/ML modelleri (`MIN_MATCHES_TO_TRAIN_SIMILARITY=30`,
`MIN_MATCHES_TO_TRAIN_ML=150`) yeterli geçmiş maç olmadan eğitilmez.
`app/scripts/init_db.py`, Football-Data.co.uk'tan (API anahtarı
gerektirmez) tarihsel veri yükler:

```bash
DATABASE_URL=... python -m app.scripts.init_db --league E0 --seasons 2223 2324
```

Bunu backend'in çalıştığı ortamda (Railway shell, veya `DATABASE_URL`'i
Supabase'e işaret eden yerel bir Python ortamında) çalıştırıp ardından
`/admin/models/train`'i tetikle.

## Bilinen sınırlamalar / doğrulanması gerekenler

- `app/services/live_sync_service.py`, SportsData.io'nun ham JSON alan
  adlarını (`GameId`, `HomeTeamId`, ...) yaygın konvansiyonlara göre
  tahmin ediyor -- bu ortamda sportsdata.io dokümantasyonuna ağ erişimi
  olmadığı için gerçek API yanıtıyla doğrulanamadı. İlk `/admin/sync/live`
  çağrısından sonra loglardaki eşleşmeyen/atlanan fixture uyarılarını
  kontrol edip gerekirse `_FIELD_CANDIDATES` sözlüğünü güncelle.
- Odds eşleştirmesi şu an tek kaynak (`sportsdata_client.get_odds`)
  üzerinden yapılıyor; `OddsAPIClient` (The Odds API) bağlı ama
  sync akışına henüz otomatik entegre değil -- istersen ikinci kaynağı
  da `live_sync_service.py`'ye ekleyebiliriz.
