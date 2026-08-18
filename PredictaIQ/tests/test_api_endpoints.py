import os

os.environ["SPORTSDATA_API_KEY"] = "test-key"
os.environ["DATABASE_URL"] = "sqlite:////tmp/predictaiq_test.db"  # smoke_test.py'nin oluşturduğu DB
os.environ["SECRET_KEY"] = "test-secret"
os.environ["CORS_ALLOW_ORIGINS"] = "http://localhost:3000"
os.environ["MIN_MATCHES_TO_TRAIN_SIMILARITY"] = "30"
os.environ["MIN_MATCHES_TO_TRAIN_ML"] = "100"

from fastapi.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    print("== / ==")
    r = client.get("/")
    print(r.status_code, r.json())
    assert r.status_code == 200

    print("== /health ==")
    r = client.get("/health")
    assert r.status_code == 200

    print("== /api/v1/admin/models/status (otomatik bootstrap sonrası) ==")
    r = client.get("/api/v1/admin/models/status")
    print(r.status_code, r.json())
    assert r.status_code == 200
    status = r.json()
    assert status["similarity_trained"] is True, "Otomatik bootstrap ile similarity eğitilmemiş!"
    assert status["ml_trained"] is True, "Otomatik bootstrap ile ML Engine eğitilmemiş!"

    print("== /api/v1/matches/ ==")
    r = client.get("/api/v1/matches/", params={"limit": 5})
    print(r.status_code, len(r.json()), "maç döndü")
    assert r.status_code == 200
    assert len(r.json()) > 0

    match_id = r.json()[0]["id"]

    print(f"== /api/v1/matches/{match_id} ==")
    r = client.get(f"/api/v1/matches/{match_id}")
    print(r.status_code, r.json())
    assert r.status_code == 200

    print("== /api/v1/odds/analysis/{match_id} ==")
    r = client.get(f"/api/v1/odds/analysis/{match_id}")
    print(r.status_code)
    assert r.status_code == 200
    print(r.json())

    print("== /api/v1/historical/same-odds ==")
    r = client.post("/api/v1/historical/same-odds", json={
        "home_odds": 1.85, "draw_odds": 3.60, "away_odds": 4.20, "tolerance_pct": 15,
    })
    print(r.status_code, r.json())
    assert r.status_code == 200

    print("== /api/v1/historical/similar/{match_id} ==")
    r = client.get(f"/api/v1/historical/similar/{match_id}")
    print(r.status_code, {"count": r.json()["count"], "home": r.json()["home"]})
    assert r.status_code == 200

    # Bekleyen (scheduled) maç bul
    r = client.get("/api/v1/matches/", params={"status": "scheduled", "limit": 5})
    scheduled = r.json()
    if scheduled:
        sid = scheduled[0]["id"]
        print(f"== /api/v1/predictions/match/{sid} ==")
        r = client.get(f"/api/v1/predictions/match/{sid}")
        print(r.status_code)
        assert r.status_code == 200, r.text
        pred = r.json()
        print("Probabilities:", pred["probabilities"])
        print("Confidence:", pred["confidence"])
        print("Data quality:", pred["data_quality"])
        print("Model details keys:", list(pred["model_details"].keys()))
        total = sum(pred["probabilities"].values())
        assert abs(total - 1.0) < 1e-6, f"Olasılıklar 1'e toplanmıyor: {total}"
    else:
        print("[SKIP] Bekleyen maç yok (smoke_test.py çalıştırılmamış olabilir)")

    print("== /api/v1/admin/models/train ==")
    r = client.post("/api/v1/admin/models/train", json={"n_folds": 3, "min_train": 60, "persist_weights": True})
    print(r.status_code, r.json())
    assert r.status_code == 200

print("\n[PASS] Tüm API endpoint testleri geçti.")
