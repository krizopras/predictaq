import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Shot:
    x: float
    y: float
    distance: float
    angle: float
    shot_type: str  # header, foot, free_kick, penalty
    assisted: bool
    chance_type: str  # big, half, low

class XGService:
    def __init__(self):
        # xG model parametreleri (basitleştirilmiş)
        self.distance_factor = 0.05
        self.angle_factor = 0.02
        self.type_factors = {
            "header": 0.85,
            "foot": 1.0,
            "free_kick": 0.7,
            "penalty": 0.76
        }
        self.chance_factors = {
            "big": 1.3,
            "half": 0.7,
            "low": 0.4
        }
    
    def calculate_shot_xg(self, shot: Shot) -> float:
        """Tek bir şutun xG değerini hesaplar"""
        # Mesafe faktörü
        distance_xg = 1 / (1 + np.exp(shot.distance / 10 - 2))
        
        # Açı faktörü
        angle_xg = 1 / (1 + np.exp(-shot.angle / 10 + 1))
        
        # Tip faktörü
        type_factor = self.type_factors.get(shot.shot_type, 1.0)
        
        # Şans faktörü
        chance_factor = self.chance_factors.get(shot.chance_type, 1.0)
        
        # Temel xG
        # NOT: önceki sürümde burada tanımsız 'angle_factor' adı kullanılıyordu
        # (self.angle_factor bir katsayı olup şut bazlı hesapla ilgisizdi) ve bu
        # NameError'a yol açıyordu. Doğrusu yukarıda hesaplanan 'angle_xg'.
        base_xg = distance_xg * angle_xg * type_factor * chance_factor
        
        # Assist bonus
        if shot.assisted:
            base_xg *= 1.05
        
        return min(1.0, base_xg)  # Max 1.0
    
    def calculate_team_xg(self, shots: List[Shot]) -> float:
        """Takımın toplam xG'sini hesaplar"""
        return sum(self.calculate_shot_xg(shot) for shot in shots)
    
    def calculate_match_xg(self, home_shots: List[Shot], away_shots: List[Shot]) -> Tuple[float, float]:
        """Maçın xG ve xGA değerlerini hesaplar"""
        home_xg = self.calculate_team_xg(home_shots)
        away_xg = self.calculate_team_xg(away_shots)
        return home_xg, away_xg
    
    def estimate_pre_match_xg(
        self,
        home_attack: float, home_defense: float,
        away_attack: float, away_defense: float,
        league_avg_home_goals: float = 1.45,
        league_avg_away_goals: float = 1.15,
    ) -> Tuple[float, float]:
        """Maç ÖNCESİ beklenen gol sayısını, takımların hücum/savunma
        rating'lerinden (0-100, 50=lig ortalaması) türetir.

        KRİTİK: Bu fonksiyon `Match.home_xg`/`away_xg` (maçın gerçekleşmiş
        xG'si) KULLANMAZ -- o değerler maç oynandıktan sonra bilinir ve
        tahmin girdisi olarak kullanılırsa veri sızıntısına yol açar. Bunun
        yerine TeamRatingService'in ürettiği, maçtan önceki üstel ortalama
        hücum/savunma güçlerini kullanır. Dixon-Coles tarzı çarpımsal model:

            home_xg = lig_ort_ev_gol * (ev_hücüm/50) * (dep_savunma/50)
            away_xg = lig_ort_dep_gol * (dep_hücüm/50) * (ev_savunma/50)
        """
        home_attack = home_attack or 50.0
        home_defense = home_defense or 50.0
        away_attack = away_attack or 50.0
        away_defense = away_defense or 50.0

        home_xg = league_avg_home_goals * (home_attack / 50.0) * (away_defense / 50.0)
        away_xg = league_avg_away_goals * (away_attack / 50.0) * (home_defense / 50.0)

        # Aşırı uçları kırp (0.2 - 4.5 arası makul gol beklentisi)
        home_xg = float(np.clip(home_xg, 0.2, 4.5))
        away_xg = float(np.clip(away_xg, 0.2, 4.5))
        return home_xg, away_xg

    def normalize_xg(self, xg: float, avg_xg: float = 1.5) -> float:
        """xG'yi normalize eder (genel ortalamaya göre)"""
        return xg / avg_xg
    
    def calculate_xg_diff(self, home_xg: float, away_xg: float) -> Dict:
        """xG farkını ve güç göstergelerini hesaplar"""
        diff = home_xg - away_xg
        total = home_xg + away_xg
        
        if total == 0:
            return {"diff": 0, "home_ratio": 0.5, "away_ratio": 0.5}
        
        return {
            "diff": diff,
            "home_ratio": home_xg / total,
            "away_ratio": away_xg / total,
            "total_xg": total
        }
    
    def expected_goals_to_probability(self, xg: float, xga: float, 
                                      avg_goals_per_match: float = 2.5) -> Dict:
        """xG verilerinden maç sonucu olasılığı çıkarır"""
        # Poisson dağılımı ile gol olasılıkları
        from scipy.stats import poisson
        
        # Beklenen gol sayılarını normalize et
        home_lambda = xg * avg_goals_per_match / 1.5
        away_lambda = xga * avg_goals_per_match / 1.5
        
        max_goals = 6
        home_probs = [poisson.pmf(i, home_lambda) for i in range(max_goals + 1)]
        away_probs = [poisson.pmf(i, away_lambda) for i in range(max_goals + 1)]
        
        home_win = 0
        draw = 0
        away_win = 0
        
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob = home_probs[h] * away_probs[a]
                if h > a:
                    home_win += prob
                elif h == a:
                    draw += prob
                else:
                    away_win += prob
        
        return {
            "home": home_win,
            "draw": draw,
            "away": away_win,
            "expected_home_goals": home_lambda,
            "expected_away_goals": away_lambda
        }