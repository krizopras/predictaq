import os
import random
from datetime import datetime, timedelta

os.environ["SPORTSDATA_API_KEY"] = "test-key"
os.environ["DATABASE_URL"] = "sqlite:////tmp/predictaiq_test.db"  # gecici test DB
os.environ["SECRET_KEY"] = "test-secret"
os.environ["CORS_ALLOW_ORIGINS"] = "http://localhost:3000"

if os.path.exists("/tmp/predictaiq_test.db"):
    os.remove("/tmp/predictaiq_test.db")

from app.database import Base, engine, SessionLocal
from app.models import Competition, Season, Team, Match, Injury, Player
from app.services.team_rating_service import TeamRatingService
from app.dependencies import prediction_service, backtest_service

random.seed(42)

Base.metadata.create_all(bind=engine)
db = SessionLocal()

# --- Sentetik lig / takımlar ---
comp = Competition(name="Test League", country="XX", level=1)
db.add(comp)
db.flush()
season = Season(competition_id=comp.id, name="2425", current=True)
db.add(season)
db.flush()

team_names = [f"Team {i}" for i in range(1, 13)]
teams = []
true_strength = {}
for name in team_names:
    t = Team(name=name)
    db.add(t)
    teams.append(t)
    true_strength[name] = random.uniform(0.7, 1.4)  # gizli "gerçek güç"
db.flush()

# --- 400 sentetik geçmiş maç üret (gerçek güce göre skor + tutarlı oranlar) ---
start_date = datetime.utcnow() - timedelta(days=730)
matches_created = 0
for day_offset in range(0, 700, 4):
    match_date = start_date + timedelta(days=day_offset)
    home, away = random.sample(teams, 2)
    hs = true_strength[home.name]
    as_ = true_strength[away.name]
    home_xg = max(0.3, 1.4 * hs / as_)
    away_xg = max(0.3, 1.1 * as_ / hs)
    home_goals = min(int(random.gauss(home_xg, 1.1) + 0.5), 6)
    away_goals = min(int(random.gauss(away_xg, 1.0) + 0.5), 6)
    home_goals = max(0, home_goals)
    away_goals = max(0, away_goals)

    # Oranları "gerçek" güce göre türet (tutarlı sentetik piyasa)
    home_prob = hs / (hs + as_ + 0.5)
    draw_prob = 0.25
    away_prob = max(0.05, 1 - home_prob - draw_prob)
    total = home_prob + draw_prob + away_prob
    home_prob, draw_prob, away_prob = home_prob / total, draw_prob / total, away_prob / total
    margin = 1.06
    opening_home = margin / home_prob * random.uniform(0.95, 1.05)
    opening_draw = margin / draw_prob * random.uniform(0.95, 1.05)
    opening_away = margin / away_prob * random.uniform(0.95, 1.05)
    closing_home = margin / home_prob * random.uniform(0.97, 1.03)
    closing_draw = margin / draw_prob * random.uniform(0.97, 1.03)
    closing_away = margin / away_prob * random.uniform(0.97, 1.03)

    m = Match(
        season_id=season.id, home_team_id=home.id, away_team_id=away.id,
        date=match_date, status="finished",
        home_score=home_goals, away_score=away_goals,
        opening_home_odds=opening_home, opening_draw_odds=opening_draw, opening_away_odds=opening_away,
        closing_home_odds=closing_home, closing_draw_odds=closing_draw, closing_away_odds=closing_away,
    )
    db.add(m)
    matches_created += 1

db.commit()
print(f"[OK] {matches_created} sentetik geçmiş maç oluşturuldu")

# --- Rating recompute ---
rating_service = TeamRatingService()
n = rating_service.recompute_all_ratings(db)
db.commit()
print(f"[OK] Rating recompute: {n} maç işlendi")

sample_team = db.query(Team).first()
print(f"[OK] Örnek takım rating: elo={sample_team.elo:.1f} attack={sample_team.attack:.1f} "
      f"defense={sample_team.defense:.1f} form5={sample_team.form_last5:.1f}")

# --- Sakatlık senaryosu için bir oyuncu + injury ekle ---
key_player = Player(team_id=teams[0].id, name="Star Forward", position="ST",
                     impact_score=0.35, is_key_player=True)
db.add(key_player)
db.flush()
db.add(Injury(player_id=key_player.id, team_id=teams[0].id, status="out", reason="hamstring"))
db.commit()
print("[OK] Sakatlık senaryosu eklendi")

# --- Walk-forward backtest + model eğitimi ---
finished = db.query(Match).filter(Match.status == "finished").all()
result = backtest_service.walk_forward(db, finished, n_folds=4, min_train=80)
print(f"[OK] Backtest status: {result.get('status')}")
if result.get("status") == "completed":
    print(f"     n_matches_evaluated={result['n_matches_evaluated']}")
    print(f"     brier={result['overall_brier_score']:.4f} logloss={result['overall_log_loss']:.4f}")
    print(f"     learned_weights={result['learned_ensemble_weights']}")
    prediction_service.set_learned_weights(result["learned_ensemble_weights"])
    print(f"[OK] Calibration fitted: {prediction_service.calibration.is_fitted}")
else:
    print(f"     DETAY: {result}")

# Nihai modelleri TÜM veriyle eğit
prediction_service.similarity.train(finished)
X, y = prediction_service.build_training_matrix(finished)
ml_result = prediction_service.ml.train(X, y, min_matches=100)
print(f"[OK] ML Engine train sonucu: {ml_result}")
print(f"[OK] Similarity trained: {prediction_service.similarity.nn is not None}, "
      f"match_count={len(prediction_service.similarity.match_data or [])}")

# --- Yeni (henüz oynanmamış) maç için tahmin ---
future_match = Match(
    season_id=season.id, home_team_id=teams[0].id, away_team_id=teams[1].id,
    date=datetime.utcnow() + timedelta(days=3), status="scheduled",
    opening_home_odds=2.10, opening_draw_odds=3.40, opening_away_odds=3.20,
    closing_home_odds=1.85, closing_draw_odds=3.60, closing_away_odds=4.20,
)
db.add(future_match)
db.commit()
db.refresh(future_match)

from app.services.player_service import PlayerImpactService
pis = PlayerImpactService()
home_injury = pis.team_injury_impact(db, teams[0].id)
print(f"[OK] Ev sahibi sakatlık etkisi: {home_injury}")

prediction = prediction_service.predict_match(
    future_match,
    home_injury_impact=home_injury["elo_penalty"],
    away_injury_impact=0.0,
)
print("[OK] TAHMİN SONUCU:")
print(f"     Probabilities: {prediction['model_probability']}")
print(f"     Fair odds: {prediction['fair_odds']}")
print(f"     Confidence: {prediction['confidence']:.1f}")
print(f"     Data quality: {prediction['data_quality']}")
print(f"     Calibration applied: {prediction['calibration_applied']}")
print(f"     Weights used: {prediction['weights_used']}")
print(f"     Similarity count: {prediction['model_details']['similarity'].get('count')}")
print(f"     Market movement: {prediction['model_details']['market_movement']}")
print(f"     ML probs: {prediction['model_details']['ml']}")
print(f"     Value: {prediction['value']}")
print(f"     Recommendation: {prediction['recommendation']}")

probs = prediction["model_probability"]
assert abs(sum(probs.values()) - 1.0) < 1e-6, "Olasılıklar 1'e toplanmıyor!"
assert all(0 <= v <= 1 for v in probs.values()), "Olasılık aralık dışı!"
print("\n[PASS] Tüm sağlamlık kontrolleri geçti.")

db.close()
