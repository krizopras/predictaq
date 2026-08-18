import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, JSON,
    String, Text, UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.types import CHAR, TypeDecorator

from app.database import Base


class GUID(TypeDecorator):
    """Platform bağımsız UUID tipi.

    PostgreSQL'de native UUID kullanır, diğer veritabanlarında (örn. test
    ortamındaki SQLite) 32 karakterlik hex string olarak saklar. Eski kod
    doğrudan postgresql.dialects.UUID kullanıyordu; bu, SQLite ile test
    edilemez ve alembic migration'larını Postgres'e kilitlerdi.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import UUID as PG_UUID
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return str(value)
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(value)
        return value.hex

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(value)
        return value


class Competition(Base):
    __tablename__ = "competitions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    country = Column(String(50))
    level = Column(Integer)
    current_season = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

    seasons = relationship("Season", back_populates="competition")


class Season(Base):
    __tablename__ = "seasons"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    competition_id = Column(GUID(), ForeignKey("competitions.id"))
    name = Column(String(20), nullable=False)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    current = Column(Boolean, default=False)

    competition = relationship("Competition", back_populates="seasons")
    matches = relationship("Match", back_populates="season")


class Team(Base):
    __tablename__ = "teams"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    sportsdata_id = Column(String(20), unique=True, index=True)
    name = Column(String(100), nullable=False)
    short_name = Column(String(30))
    country = Column(String(50))
    city = Column(String(50))
    stadium = Column(String(100))
    founded_year = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Güncel rating'ler (anlık görünüm; tarihçesi team_rating_history'de tutulur)
    elo = Column(Float, default=1500)
    attack = Column(Float, default=50)
    defense = Column(Float, default=50)
    home_power = Column(Float, default=50)
    away_power = Column(Float, default=50)
    form = Column(Float, default=50)
    form_last3 = Column(Float, default=50)
    form_last5 = Column(Float, default=50)
    form_last10 = Column(Float, default=50)
    spi = Column(Float)

    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")
    rating_history = relationship("TeamRatingHistory", back_populates="team")


class TeamRatingHistory(Base):
    """Zaman ağırlıklı rating/form hesaplaması için gerekli tarihsel snapshot'lar.

    Plan madde 2-3: 'Son maç 6 ay öncekinden daha önemli' -- bunu uygulayabilmek
    için her maç sonrası rating anlık görüntüsünü buraya yazıyoruz.
    """
    __tablename__ = "team_rating_history"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    team_id = Column(GUID(), ForeignKey("teams.id"), nullable=False)
    match_id = Column(GUID(), ForeignKey("matches.id"))
    as_of = Column(DateTime, nullable=False, default=datetime.utcnow)
    elo = Column(Float)
    attack = Column(Float)
    defense = Column(Float)
    form_last3 = Column(Float)
    form_last5 = Column(Float)
    form_last10 = Column(Float)
    home_power = Column(Float)
    away_power = Column(Float)

    team = relationship("Team", back_populates="rating_history")
    __table_args__ = (Index("idx_team_rating_asof", "team_id", "as_of"),)


class Player(Base):
    __tablename__ = "players"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    sportsdata_id = Column(String(20), unique=True, index=True)
    team_id = Column(GUID(), ForeignKey("teams.id"))
    name = Column(String(100), nullable=False)
    position = Column(String(20))
    jersey_number = Column(Integer)
    nationality = Column(String(50))
    date_of_birth = Column(DateTime)
    height = Column(Float)
    weight = Column(Float)
    market_value = Column(Float)

    # Player Impact Score (plan madde 5)
    impact_score = Column(Float, default=0)
    xg_contribution = Column(Float, default=0)
    xga_contribution = Column(Float, default=0)
    minutes_last5 = Column(Float, default=0)
    is_key_player = Column(Boolean, default=False)

    team = relationship("Team")


class Injury(Base):
    """Sakatlık / cezalı durumu (plan madde 5)."""
    __tablename__ = "injuries"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    player_id = Column(GUID(), ForeignKey("players.id"), nullable=False)
    team_id = Column(GUID(), ForeignKey("teams.id"), nullable=False)
    status = Column(String(20))  # out, doubtful, suspended, questionable
    reason = Column(String(100))
    reported_at = Column(DateTime, default=datetime.utcnow)
    expected_return = Column(DateTime)

    player = relationship("Player")
    team = relationship("Team")
    __table_args__ = (Index("idx_injury_team_status", "team_id", "status"),)


class LineupEntry(Base):
    """Maç kadrosu / ilk 11 (plan madde 5)."""
    __tablename__ = "lineup_entries"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    match_id = Column(GUID(), ForeignKey("matches.id"), nullable=False)
    team_id = Column(GUID(), ForeignKey("teams.id"), nullable=False)
    player_id = Column(GUID(), ForeignKey("players.id"), nullable=False)
    is_starting = Column(Boolean, default=True)
    position = Column(String(20))

    match = relationship("Match")
    team = relationship("Team")
    player = relationship("Player")
    __table_args__ = (Index("idx_lineup_match_team", "match_id", "team_id"),)


class Bookmaker(Base):
    __tablename__ = "bookmakers"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False)
    is_sharp = Column(Boolean, default=False)  # Pinnacle vb. "keskin" kitapçılar
    source = Column(String(50))  # hangi API'den geliyor (odds_api, sportsdata, ...)


class Market(Base):
    """1X2 dışındaki pazarlar (O/U, BTTS, AH) için genel amaçlı tablo."""
    __tablename__ = "markets"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    match_id = Column(GUID(), ForeignKey("matches.id"), nullable=False)
    bookmaker_id = Column(GUID(), ForeignKey("bookmakers.id"))
    market_type = Column(String(30))  # "1x2", "ou_2.5", "btts", "ah_-1"
    timestamp = Column(DateTime, default=datetime.utcnow)
    selections = Column(JSON)  # {"home": 1.85, "draw": 3.6, "away": 4.5} gibi

    match = relationship("Match")
    bookmaker = relationship("Bookmaker")
    __table_args__ = (Index("idx_market_match_type", "match_id", "market_type"),)


class Match(Base):
    __tablename__ = "matches"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    sportsdata_id = Column(String(20), unique=True, index=True)
    season_id = Column(GUID(), ForeignKey("seasons.id"))
    home_team_id = Column(GUID(), ForeignKey("teams.id"))
    away_team_id = Column(GUID(), ForeignKey("teams.id"))

    date = Column(DateTime, nullable=False)
    round = Column(String(20))
    venue = Column(String(100))
    referee = Column(String(100))
    status = Column(String(20))  # scheduled, live, finished, postponed, cancelled

    # Sonuç
    home_score = Column(Integer)
    away_score = Column(Integer)
    home_shots = Column(Integer)
    away_shots = Column(Integer)
    home_shots_on_target = Column(Integer)
    away_shots_on_target = Column(Integer)
    home_possession = Column(Float)
    away_possession = Column(Float)
    home_corners = Column(Integer)
    away_corners = Column(Integer)
    home_fouls = Column(Integer)
    away_fouls = Column(Integer)
    home_yellow_cards = Column(Integer)
    away_yellow_cards = Column(Integer)
    home_red_cards = Column(Integer)
    away_red_cards = Column(Integer)

    # xG -- GERÇEKLEŞEN (post-match, sağlayıcıdan gelir veya şut verisinden
    # hesaplanır). SADECE görüntüleme / geriye dönük analiz için kullanılır.
    # Tahmin modellerine ASLA doğrudan girdi olarak verilmez çünkü bu değer
    # maçın kendisi oynanmadan bilinemez (veri sızıntısı olur).
    home_xg = Column(Float)
    away_xg = Column(Float)
    home_xga = Column(Float)
    away_xga = Column(Float)

    # xG -- MAÇ ÖNCESİ TAHMİN (TeamRatingService tarafından, maçtan önceki
    # hücum/savunma rating'lerinden üretilir). Similarity/ML/Poisson
    # modellerinin GİRDİSİ budur.
    home_xg_pre = Column(Float)
    away_xg_pre = Column(Float)

    # Rating snapshot (maç anındaki değerler -- similarity/ML feature'ları için sabitlenir)
    home_elo = Column(Float)
    away_elo = Column(Float)

    # Oranlar (Opening)
    opening_home_odds = Column(Float)
    opening_draw_odds = Column(Float)
    opening_away_odds = Column(Float)

    # Oranlar (Closing)
    closing_home_odds = Column(Float)
    closing_draw_odds = Column(Float)
    closing_away_odds = Column(Float)

    # Metrikler
    elo_difference = Column(Float)
    form_difference = Column(Float)
    xg_difference = Column(Float)

    # Tahminler (son ensemble sonucu -- ayrıntılı geçmiş predictions tablosunda)
    model_home_prob = Column(Float)
    model_draw_prob = Column(Float)
    model_away_prob = Column(Float)
    model_confidence = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    season = relationship("Season", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    odds_snapshots = relationship("OddsSnapshot", back_populates="match")


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    match_id = Column(GUID(), ForeignKey("matches.id"))
    bookmaker = Column(String(50))
    source = Column(String(50))  # sportsdata / odds_api / football_data
    timestamp = Column(DateTime, default=datetime.utcnow)
    home_odds = Column(Float)
    draw_odds = Column(Float)
    away_odds = Column(Float)

    match = relationship("Match", back_populates="odds_snapshots")
    __table_args__ = (Index("idx_match_bookmaker_time", "match_id", "bookmaker", "timestamp"),)


class HistoricalSimilarity(Base):
    __tablename__ = "historical_similarities"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    match_id = Column(GUID(), ForeignKey("matches.id"))
    similar_match_id = Column(GUID(), ForeignKey("matches.id"))
    similarity_score = Column(Float)
    rank = Column(Integer)

    match = relationship("Match", foreign_keys=[match_id])
    similar_match = relationship("Match", foreign_keys=[similar_match_id])


class HistoricalFeature(Base):
    """Bir maça ait, eğitim/tahmin anında kullanılan donmuş (frozen) feature
    vektörü. Bunu ayrı tabloda tutmak, ileride sızıntısız (leakage-free)
    walk-forward backtest yapabilmek için kritik: bir maçın feature'ları,
    o maçın kickoff anındaki bilgiyle sabitlenir, sonradan değişmez.
    """
    __tablename__ = "historical_features"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    match_id = Column(GUID(), ForeignKey("matches.id"), unique=True, nullable=False)
    feature_vector = Column(JSON, nullable=False)
    feature_names = Column(JSON, nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    match_id = Column(GUID(), ForeignKey("matches.id"))
    model_name = Column(String(50))
    home_prob = Column(Float)
    draw_prob = Column(Float)
    away_prob = Column(Float)
    confidence = Column(Float)
    value_score = Column(Float)
    model_details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match")


class PredictionResult(Base):
    """Tahmin ile gerçekleşen sonucun eşleştirilmiş hali -- calibration,
    Brier score ve log loss hesapları bu tablodan beslenir. Eski şemada
    bu geri besleme döngüsü hiç yoktu.
    """
    __tablename__ = "prediction_results"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(GUID(), ForeignKey("predictions.id"), nullable=False)
    match_id = Column(GUID(), ForeignKey("matches.id"), nullable=False)
    predicted_home_prob = Column(Float, nullable=False)
    predicted_draw_prob = Column(Float, nullable=False)
    predicted_away_prob = Column(Float, nullable=False)
    actual_outcome = Column(String(10))  # "home" | "draw" | "away"
    brier_component = Column(Float)
    log_loss_component = Column(Float)
    resolved_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("Prediction")
    match = relationship("Match")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), nullable=False)
    version = Column(String(20))
    parameters = Column(JSON)
    ensemble_weights = Column(JSON)  # ML ile öğrenilen ağırlıklar (madde 3)
    brier_score = Column(Float)
    log_loss = Column(Float)
    calibration_error = Column(Float)
    trained_on_matches = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)

    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_name_version"),)
