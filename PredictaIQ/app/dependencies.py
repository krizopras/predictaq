"""Uygulama genelinde paylaşılan servis singleton'ları.

Eski kodda her router kendi `PredictionService()` örneğini oluşturuyordu.
Bu, similarity/ml modelleri eğitildiğinde (örn. admin router üzerinden)
bu eğitimin predictions router'ındaki AYRI bir örneğe yansımaması
anlamına gelirdi. Burada tek bir paylaşılan örnek tutuluyor.
"""
from app.services.backtest_service import BacktestService
from app.services.prediction_service import PredictionService

prediction_service = PredictionService()
backtest_service = BacktestService(prediction_service)
