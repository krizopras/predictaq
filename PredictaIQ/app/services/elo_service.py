import math
import numpy as np
from typing import Tuple, Dict

class EloService:
    def __init__(self, k_factor: float = 32, home_advantage: float = 100):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
    
    def expected_score(self, rating_a: float, rating_b: float, home_advantage: bool = False) -> float:
        """A takımının B takımına karşı beklenen skorunu hesaplar"""
        if home_advantage:
            rating_a += self.home_advantage
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))
    
    def update_rating(self, rating: float, expected: float, actual: float) -> float:
        """Elo rating güncellemesi"""
        return rating + self.k_factor * (actual - expected)
    
    def calculate_match_elo_change(self, home_elo: float, away_elo: float, 
                                   home_score: int, away_score: int) -> Tuple[float, float]:
        """Maç sonucuna göre Elo değişimini hesaplar"""
        expected_home = self.expected_score(home_elo, away_elo, home_advantage=True)
        expected_away = 1 - expected_home
        
        # Gerçek sonuç: galibiyet=1, beraberlik=0.5, mağlubiyet=0
        if home_score > away_score:
            actual_home, actual_away = 1, 0
        elif home_score == away_score:
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0, 1
        
        new_home_elo = self.update_rating(home_elo, expected_home, actual_home)
        new_away_elo = self.update_rating(away_elo, expected_away, actual_away)
        
        return new_home_elo, new_away_elo
    
    def calculate_win_probability(self, home_elo: float, away_elo: float) -> Dict[str, float]:
        """Elo'ya göre maç kazanma olasılıklarını hesaplar"""
        expected_home = self.expected_score(home_elo, away_elo, home_advantage=True)
        expected_away = 1 - expected_home
        
        # Beraberlik olasılığı için Elo farkını kullan (ampirik)
        elo_diff = home_elo - away_elo + self.home_advantage
        draw_prob = 0.25 - 0.0001 * abs(elo_diff)  # Basitleştirilmiş yaklaşım
        draw_prob = max(0.15, min(0.30, draw_prob))  # Sınırlandır
        
        # Galibiyet olasılıklarını beraberliğe göre ayarla
        home_win_prob = expected_home * (1 - draw_prob)
        away_win_prob = expected_away * (1 - draw_prob)
        
        return {
            "home": home_win_prob,
            "draw": draw_prob,
            "away": away_win_prob
        }
    
    def calculate_multi_match_elo(self, matches: list) -> Dict:
        """Birden fazla maçın Elo'sunu günceller"""
        ratings = {}
        
        for match in matches:
            home = match["home_team"]
            away = match["away_team"]
            
            # Takımların güncel Elo'larını al
            home_elo = ratings.get(home, 1500)
            away_elo = ratings.get(away, 1500)
            
            # Yeni Elo'ları hesapla
            new_home, new_away = self.calculate_match_elo_change(
                home_elo, away_elo, 
                match["home_score"], match["away_score"]
            )
            
            ratings[home] = new_home
            ratings[away] = new_away
        
        return ratings