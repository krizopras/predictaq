import numpy as np
from scipy.stats import poisson
from typing import Dict, List, Tuple

class PoissonService:
    def __init__(self, max_goals: int = 6):
        self.max_goals = max_goals
    
    def predict_match(self, home_xg: float, away_xg: float) -> Dict:
        """Poisson modeli ile maç tahmini yapar"""
        home_probs = [poisson.pmf(i, home_xg) for i in range(self.max_goals + 1)]
        away_probs = [poisson.pmf(i, away_xg) for i in range(self.max_goals + 1)]
        
        # Sonuç olasılıkları
        home_win = 0
        draw = 0
        away_win = 0
        
        # Skor matrisi
        score_matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
        
        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
                prob = home_probs[h] * away_probs[a]
                score_matrix[h][a] = prob
                
                if h > a:
                    home_win += prob
                elif h == a:
                    draw += prob
                else:
                    away_win += prob
        
        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "score_matrix": score_matrix.tolist(),
            "most_likely_home_goals": np.argmax(np.sum(score_matrix, axis=1)),
            "most_likely_away_goals": np.argmax(np.sum(score_matrix, axis=0))
        }
    
    def calculate_over_under(self, home_xg: float, away_xg: float, threshold: float = 2.5) -> Dict:
        """Over/Under olasılıklarını hesaplar"""
        home_probs = [poisson.pmf(i, home_xg) for i in range(self.max_goals + 1)]
        away_probs = [poisson.pmf(i, away_xg) for i in range(self.max_goals + 1)]
        
        over = 0
        under = 0
        
        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
                prob = home_probs[h] * away_probs[a]
                total = h + a
                if total > threshold:
                    over += prob
                else:
                    under += prob
        
        return {"over": over, "under": under, "threshold": threshold}
    
    def calculate_btts(self, home_xg: float, away_xg: float) -> Dict:
        """Her iki takımın da gol atması olasılığını hesaplar"""
        home_no_goal = poisson.pmf(0, home_xg)
        away_no_goal = poisson.pmf(0, away_xg)
        
        btts = (1 - home_no_goal) * (1 - away_no_goal)
        no_btts = 1 - btts
        
        return {"btts": btts, "no_btts": no_btts}
    
    def calculate_exact_score(self, home_xg: float, away_xg: float) -> Dict:
        """En olası skorları hesaplar"""
        home_probs = [poisson.pmf(i, home_xg) for i in range(self.max_goals + 1)]
        away_probs = [poisson.pmf(i, away_xg) for i in range(self.max_goals + 1)]
        
        scores = []
        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
                prob = home_probs[h] * away_probs[a]
                scores.append({
                    "home": h,
                    "away": a,
                    "probability": prob
                })
        
        scores.sort(key=lambda x: x["probability"], reverse=True)
        return scores[:10]  # Top 10 skor