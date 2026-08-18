# PredictaIQ v2 — Revizyon Notları

Bu revizyon, önerilen mimari dokümanla önceki kod arasındaki tüm boşlukları
kapatmayı ve tespit edilen gerçek hataları düzeltmeyi hedefledi. Aşağıdaki
her madde **çalıştırılarak** doğrulandı (bkz. `tests/`).

## 🔴 Düzeltilen kırıcı hatalar (uygulama önceden hiç açılmıyordu)

1. **`app/main.py`** `matches`, `historical`, `odds`, `admin` router'larını
   import ediyordu ama bu dosyalar hiç yoktu → `ImportError`. Dördü de
   yazıldı.
2. **`app/routers/predictions.py`** `schemas.PredictionDetail` import
   ediyordu ama bu sınıf tanımlı değildi → `ImportError`. Ayrıca
   `predict_batch` içinde `Season`/`Competition` hiç import edilmeden
   kullanılıyordu → `NameError`.
3. **`app/services/xg_service.py`** `calculate_shot_xg` içinde tanımsız
   `angle_factor` adı kullanılıyordu (olması gereken: hesaplanan
   `angle_xg`) → çalıştırıldığında `NameError`.
4. **`app/services/calibration_service.py`** `calculate_log_loss`,
   Python listesi ile numpy aritmetiği yapmaya çalışıyordu →
   `TypeError`.
5. **`app/services/sportsdata_client.py`** `batch_get_fixtures`, `offset`
   parametresini API'ye hiç göndermiyordu → aynı sayfayı sonsuz döngüde
   tekrar isteyebilen bir kod.
6. Veri modelleri (`models.py`) PostgreSQL'e özgü `UUID` tipi kullanıyordu;
   SQLite ile (yerel geliştirme/test) hiç çalışmıyordu. Platform bağımsız
   `GUID` tipiyle değiştirildi.

## 🟡 Bağlanan "ölü kod" (vardı ama hiçbir yerden çağrılmıyordu)

- **Similarity/Historical Engine**: `similarity_service.train()` artık
  uygulama açılışında ve `/api/v1/admin/models/train` üzerinden gerçekten
  çağrılıyor.
- **Calibration Engine**: Artık `PredictionService` her tahminin sonunda
  (varsa) çok sınıflı kalibrasyon uyguluyor.
- **Market Movement**: Ayrı bir "Model 6" olarak ensemble'a katıldı
  (`market_movement_service.py`).

## 🟢 Yeni eklenen motorlar

| Motor | Dosya | Plan maddesi |
|---|---|---|
| ML Engine (XGBoost+LightGBM+CatBoost) | `services/ml_service.py` | 13 |
| Zaman ağırlıklı form/rating + hücum/savunma | `services/team_rating_service.py` | 2-4 |
| Oyuncu etkisi / sakatlık motoru | `services/player_service.py` | 5 |
| Market Movement Model | `services/market_movement_service.py` | 13, 18 |
| Walk-forward backtest + öğrenilen ensemble ağırlıkları | `services/backtest_service.py` | 3, 15, 26 |
| Merkezi, sızıntısız feature engineering | `services/feature_engineering.py` | 22 |
| The Odds API istemcisi (ikincil kaynak) | `services/odds_api_client.py` | 7, 24 |
| Football-Data.co.uk yükleyici (üçüncü kaynak) | `services/football_data_client.py` | 9, 24 |

## ⚠️ En kritik mimari düzeltme: veri sızıntısı (leakage)

Eski `PredictionService`, henüz oynanmamış bir maçı tahmin ederken o maçın
**gerçekleşmiş** `home_xg`/`away_xg` değerini girdi olarak kullanıyordu —
bu değer maç oynanmadan bilinemez. `Match` modeline artık ayrı
`home_xg_pre`/`away_xg_pre` alanları eklendi; bunlar SADECE maçtan önceki
hücum/savunma rating'lerinden türetiliyor. `home_xg`/`away_xg` (gerçekleşen)
alanları sadece görüntüleme amaçlı kalıyor, hiçbir modele girdi olarak
verilmiyor. Aynı ilke `TeamRatingService.recompute_all_ratings` içinde de
uygulanıyor: her maçın Elo/form snapshot'ı, O MAÇTAN ÖNCEKİ bilgiyle
yazılıyor.

## Veritabanı şeması eklenenler

`injuries`, `lineup_entries`, `bookmakers`, `markets`, `team_rating_history`,
`historical_features`, `prediction_results` tabloları eklendi (plan
madde 23'teki 19 tablonun tamamına yakını artık mevcut).

## Nasıl test edildi

```bash
pip install -r requirements.txt
PYTHONPATH=. python tests/test_smoke_synthetic.py   # sentetik veriyle uçtan uca pipeline
PYTHONPATH=. python tests/test_api_endpoints.py     # gerçek FastAPI + TestClient ile tüm endpoint'ler
```

İki script de SQLite kullanır (Postgres gerekmez), ~175 sentetik maç
üretir, rating/form hesaplar, walk-forward backtest çalıştırır, similarity
+ ML Engine'i eğitir, kalibrasyonu uygular ve bir maç için tam tahmin
üretip tüm sağlamlık kontrollerini (olasılıkların 1'e toplanması,
JSON serileştirme, vb.) doğrular.

**Not:** Gerçek Sportsdata.io / Football-Data.co.uk verisiyle uçtan uca
(gerçek API anahtarlarıyla) test edilmedi -- bu ortamda dış ağ erişimi
kısıtlı. `.env.example` dosyasını doldurup
`python -m app.scripts.init_db --league E0 --seasons 2223 2324` ile
gerçek veri yüklemesini kendi ortamınızda doğrulamanız gerekir.

## Bilinen sınırlamalar / sonraki adımlar

- Ensemble ağırlıkları küçük/sentetik veri setinde bazı modellere (elo)
  aşırı ağırlık verebilir; gerçek veri hacmi arttıkça
  `/api/v1/admin/models/train` periyodik olarak yeniden çalıştırılmalı.
- `PlayerImpactService.impact_score` şu an manuel/harici olarak
  doldurulmalı (otomatik xG-katkı hesaplama modeli plan kapsamı dışında
  bırakıldı; ileride şut/asist verisiyle genişletilebilir).
- TimescaleDB hypertable migration'ı eklenmedi (plan madde 23'te
  opsiyonel olarak belirtilmişti); PostgreSQL + normal indexleme yeterli
  ölçekte çalışır.
