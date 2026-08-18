import logging
import os

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session, joinedload
from typing import Dict, List, Optional, Tuple

from app.models import Match, Season
from app.services.feature_engineering import FEATURE_NAMES, build_match_feature_vector

logger = logging.getLogger(__name__)


class SimilarityService:
    def __init__(self):
        self.scaler = StandardScaler()
        self.nn = None
        self.match_data = None
        self.feature_names: List[str] = list(FEATURE_NAMES)

    def _create_feature_vector(self, match: Match, home_injury_impact: float = 0.0,
                                away_injury_impact: float = 0.0) -> np.ndarray:
        """Bir maçtan, MAÇ ÖNCESİ bilinen bilgilerden özellik vektörü oluşturur.

        NOT: match.home_xg/away_xg (gerçekleşen xG) DEĞİL, TeamRatingService
        tarafından üretilen match.home_xg_pre/away_xg_pre kullanılır --
        aksi halde henüz oynanmamış bir maçı, o maçın kendi sonucundan
        türetilmiş bir bilgiyle "tahmin etmiş" gibi görünürüz (leakage).
        """
        return build_match_feature_vector(
            home_elo=match.home_elo,
            away_elo=match.away_elo,
            home_form5=match.home_team.form_last5 if match.home_team else None,
            away_form5=match.away_team.form_last5 if match.away_team else None,
            home_xg_pre=match.home_xg_pre,
            away_xg_pre=match.away_xg_pre,
            opening_home_odds=match.opening_home_odds,
            opening_draw_odds=match.opening_draw_odds,
            opening_away_odds=match.opening_away_odds,
            closing_home_odds=match.closing_home_odds,
            closing_draw_odds=match.closing_draw_odds,
            closing_away_odds=match.closing_away_odds,
            home_injury_impact=home_injury_impact,
            away_injury_impact=away_injury_impact,
        )

    def train(self, matches: List[Match]) -> None:
        """Benzerlik modelini SADECE sonucu bilinen (bitmiş) maçlarla eğitir."""
        finished = [m for m in matches if m.home_score is not None and m.away_score is not None]
        if len(finished) < 10:
            print("Yeterli veri yok, model eğitilemedi")
            return

        # Özellik vektörlerini oluştur
        feature_vectors = []
        valid_matches = []

        for match in finished:
            if match.home_elo is not None and match.away_elo is not None:
                vector = self._create_feature_vector(match)
                feature_vectors.append(vector)
                valid_matches.append(match)

        if len(feature_vectors) < 5:
            print("Yetersiz geçerli maç")
            return

        feature_vectors = np.array(feature_vectors)
        self.match_data = valid_matches

        # Özellikleri normalize et
        normalized = self.scaler.fit_transform(feature_vectors)

        # Nearest Neighbors modeli
        self.nn = NearestNeighbors(n_neighbors=min(100, len(normalized)), metric='euclidean')
        self.nn.fit(normalized)

        print(f"Benzerlik modeli {len(feature_vectors)} maç ile eğitildi")
    
    def find_similar_matches(self, match: Match, n_neighbors: int = 50,
                              home_injury_impact: float = 0.0,
                              away_injury_impact: float = 0.0,
                              league_id: Optional[str] = None) -> Dict:
        """Benzer maçları bulur.

        Plan madde 8'deki basamaklı filtreye (aynı lig -> benzer güç farkı
        -> benzer oran hareketi) yaklaşık bir uygulama: önce geniş bir KNN
        havuzu (n_neighbors*4) çekilir, sonra mümkünse aynı lige ait
        maçlarla önceliklendirilir; yeterince aynı-lig maçı yoksa tüm
        havuza geri düşülür (soft-fallback).
        """
        if self.nn is None or self.match_data is None:
            return {
                "count": 0,
                "home": 0.33,
                "draw": 0.33,
                "away": 0.34,
                "matches": [],
                "error": "Model eğitilmemiş (yeterli geçmiş veri yok)"
            }

        # Maç vektörünü oluştur
        vector = self._create_feature_vector(match, home_injury_impact, away_injury_impact).reshape(1, -1)
        normalized = self.scaler.transform(vector)

        # Geniş bir havuz çek, sonra basamaklı filtrele
        pool_size = min(len(self.match_data), max(n_neighbors * 4, n_neighbors))
        distances, indices = self.nn.kneighbors(normalized, n_neighbors=pool_size)

        pool = [self.match_data[idx] for idx in indices[0] if idx < len(self.match_data)]

        similar_matches = pool
        if league_id is not None:
            same_league = [m for m in pool if str(getattr(m.season, "competition_id", None)) == str(league_id)]
            if len(same_league) >= max(10, n_neighbors // 2):
                similar_matches = same_league[:n_neighbors]
            else:
                similar_matches = pool[:n_neighbors]
        else:
            similar_matches = pool[:n_neighbors]

        # Sonuç dağılımı
        total = len(similar_matches)
        if total == 0:
            return {"count": 0, "home": 0.33, "draw": 0.33, "away": 0.34, "matches": []}

        home_wins = sum(1 for m in similar_matches if m.home_score > m.away_score)
        draws = sum(1 for m in similar_matches if m.home_score == m.away_score)
        away_wins = total - home_wins - draws
        
        # Benzer maç bilgileri
        match_info = []
        for m in similar_matches[:10]:
            match_info.append({
                "home_team": m.home_team.name if m.home_team else "Unknown",
                "away_team": m.away_team.name if m.away_team else "Unknown",
                "score": f"{m.home_score}-{m.away_score}" if m.home_score is not None else "Unknown",
                "home_odds": m.closing_home_odds,
                "draw_odds": m.closing_draw_odds,
                "away_odds": m.closing_away_odds
            })
        
        return {
            "count": total,
            "home": home_wins / total,
            "draw": draws / total,
            "away": away_wins / total,
            "home_wins": home_wins,
            "draws": draws,
            "away_wins": away_wins,
            "matches": match_info
        }
    
    # ------------------------------------------------------------------
    # Kalıcılık (joblib)
    # ------------------------------------------------------------------
    # NOT: self.match_data içindeki SQLAlchemy ORM Match nesnelerini
    # doğrudan pickle'lamak (session/engine'e bağımlılık, şema değişikliği
    # riskleri yüzünden) kırılgan olur. Bunun yerine sadece match ID'lerini
    # (sırayla) diske yazıyoruz; yüklerken bu ID'lerle DB'den, similarity
    # hesaplamasının ihtiyaç duyduğu ilişkilerle (home_team/away_team/
    # season->competition) EAGER olarak tekrar çekiyoruz.
    def save(self, dir_path: str) -> None:
        if self.nn is None or not self.match_data:
            logger.info("Similarity: eğitilmemiş model, kaydedilecek bir şey yok")
            return
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(dir_path, "scaler.joblib"))
        joblib.dump(self.nn, os.path.join(dir_path, "nn.joblib"))
        match_ids = [str(m.id) for m in self.match_data]
        joblib.dump(
            {"match_ids": match_ids, "feature_names": self.feature_names},
            os.path.join(dir_path, "meta.joblib"),
        )
        logger.info("Similarity: %s içine kaydedildi (%d maç)", dir_path, len(match_ids))

    def load(self, dir_path: str, db: Session) -> bool:
        meta_path = os.path.join(dir_path, "meta.joblib")
        if not os.path.exists(meta_path):
            return False
        try:
            meta = joblib.load(meta_path)
            match_ids = meta.get("match_ids", [])
            if not match_ids:
                return False

            rows = (
                db.query(Match)
                .options(
                    joinedload(Match.home_team),
                    joinedload(Match.away_team),
                    joinedload(Match.season).joinedload(Season.competition),
                )
                .filter(Match.id.in_(match_ids))
                .all()
            )
            by_id = {str(m.id): m for m in rows}
            # Kaydedilen sırayı (nn indeksleriyle hizalı) koru; DB'den
            # silinmiş bir maç varsa atla (nn indeksleri kayar ama bu,
            # hiç yüklenememekten daha iyi bir best-effort durumdur).
            ordered = [by_id[mid] for mid in match_ids if mid in by_id]
            if len(ordered) != len(match_ids):
                logger.warning(
                    "Similarity: %d/%d kaydedilmiş maç DB'de bulunamadı, model tutarsız olabilir -- yeniden eğitim önerilir",
                    len(match_ids) - len(ordered), len(match_ids),
                )
            if not ordered:
                return False

            self.scaler = joblib.load(os.path.join(dir_path, "scaler.joblib"))
            self.nn = joblib.load(os.path.join(dir_path, "nn.joblib"))
            self.feature_names = meta.get("feature_names", list(FEATURE_NAMES))
            self.match_data = ordered
            logger.info("Similarity: %s içinden yüklendi (%d maç)", dir_path, len(ordered))
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("Similarity model yüklenemedi (yoksayılıyor): %s", exc)
            return False

    def calculate_similarity_score(self, match1: Match, match2: Match) -> float:
        """İki maç arasındaki benzerlik skorunu hesaplar"""
        v1 = self._create_feature_vector(match1)
        v2 = self._create_feature_vector(match2)
        
        # Eksik değerleri doldur
        v1 = np.nan_to_num(v1)
        v2 = np.nan_to_num(v2)
        
        # Euclidean mesafesi
        distance = np.linalg.norm(v1 - v2)
        
        # Benzerlik skoruna çevir (0-1 arası)
        similarity = 1 / (1 + distance)
        return similarity