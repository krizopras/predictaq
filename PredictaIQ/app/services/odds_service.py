import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class OddsService:
    def __init__(self):
        self.movement_threshold = 0.05
        self.sharp_bookmakers = ["Pinnacle", "Bet365", "William Hill", "Sportingbet"]
    
    def calculate_implied_probability(self, odds: float, overround: float = 0) -> float:
        """Orandan implied probability hesaplar"""
        if odds <= 0:
            return 0
        return 1 / odds / (1 + overround)
    
    def normalize_odds(self, home_odds: float, draw_odds: float, away_odds: float) -> Dict:
        """Oranları normalize eder (marjı dağıtır)"""
        if any(o <= 0 for o in [home_odds, draw_odds, away_odds]):
            return {"home": 0.33, "draw": 0.33, "away": 0.33}
        
        implied_home = 1 / home_odds
        implied_draw = 1 / draw_odds
        implied_away = 1 / away_odds
        
        total = implied_home + implied_draw + implied_away
        
        return {
            "home": implied_home / total,
            "draw": implied_draw / total,
            "away": implied_away / total,
            "overround": total - 1,
            "fair_home_odds": 1 / (implied_home / total),
            "fair_draw_odds": 1 / (implied_draw / total),
            "fair_away_odds": 1 / (implied_away / total)
        }
    
    def calculate_movement(self, opening_odds: float, closing_odds: float) -> Dict:
        """Oran hareketini analiz eder"""
        if opening_odds <= 0 or closing_odds <= 0:
            return {"change": 0, "percentage": 0, "direction": "none"}
        
        change = closing_odds - opening_odds
        percentage = (change / opening_odds) * 100
        
        direction = "none"
        if abs(change) < self.movement_threshold:
            direction = "stable"
        elif change > 0:
            direction = "up"
        else:
            direction = "down"
        
        return {
            "change": change,
            "percentage": percentage,
            "direction": direction,
            "magnitude": abs(percentage)
        }
    
    def calculate_bookmaker_consensus(self, odds_list: List[Dict]) -> Dict:
        """Birden fazla bookmaker'ın oranlarından konsensus hesaplar"""
        if not odds_list:
            return {}
        
        home_odds = []
        draw_odds = []
        away_odds = []
        
        for bookmaker in odds_list:
            if bookmaker.get("home_odds"):
                home_odds.append(bookmaker["home_odds"])
            if bookmaker.get("draw_odds"):
                draw_odds.append(bookmaker["draw_odds"])
            if bookmaker.get("away_odds"):
                away_odds.append(bookmaker["away_odds"])
        
        result = {}
        
        if home_odds:
            result["home"] = {
                "mean": np.mean(home_odds),
                "std": np.std(home_odds),
                "min": np.min(home_odds),
                "max": np.max(home_odds),
                "sharpest": min(home_odds)  # En düşük oran genelde en keskin
            }
        
        if draw_odds:
            result["draw"] = {
                "mean": np.mean(draw_odds),
                "std": np.std(draw_odds),
                "min": np.min(draw_odds),
                "max": np.max(draw_odds)
            }
        
        if away_odds:
            result["away"] = {
                "mean": np.mean(away_odds),
                "std": np.std(away_odds),
                "min": np.min(away_odds),
                "max": np.max(away_odds)
            }
        
        return result
    
    def calculate_ev(self, model_prob: float, market_odds: float) -> Dict:
        """Beklenen değeri (Expected Value) hesaplar.

        NOT: model_prob genelde numpy.float64 olarak gelebilir (ensemble
        hesaplamalarından); bu fonksiyon FastAPI/Pydantic response'larında
        kullanıldığı için tüm dönüş değerleri native Python tipine
        (float/bool) dönüştürülür -- aksi halde `numpy.bool_`/`numpy.float64`
        JSON serileştirmesi PydanticSerializationError ile patlar.
        """
        model_prob = float(model_prob)
        market_odds = float(market_odds)
        if market_odds <= 0:
            return {"ev": 0.0, "edge": 0.0, "is_positive": False}
        
        implied_prob = 1 / market_odds
        edge = model_prob - implied_prob
        ev = model_prob * market_odds - 1
        
        return {
            "ev": float(ev * 100),  # Yüzde olarak
            "edge": float(edge * 100),  # Yüzde olarak
            "is_positive": bool(ev > 0),
            "model_prob": model_prob,
            "implied_prob": implied_prob,
            "fair_odds": float(1 / model_prob) if model_prob > 0 else None
        }
    
    def identify_sharp_movement(self, odds_history: List[Dict]) -> Dict:
        """Sharp bookmaker'ların hareketlerini tespit eder"""
        if not odds_history:
            return {"has_sharp_movement": False}
        
        sharp_movements = []
        for snapshot in odds_history:
            bookmaker = snapshot.get("bookmaker", "")
            if bookmaker in self.sharp_bookmakers:
                sharp_movements.append(snapshot)
        
        if not sharp_movements:
            return {"has_sharp_movement": False}
        
        # Zaman serisi analizi
        result = {
            "has_sharp_movement": True,
            "sharp_bookmakers": list(set(s["bookmaker"] for s in sharp_movements)),
            "movements": []
        }
        
        for bookmaker in self.sharp_bookmakers:
            bookmaker_data = [s for s in sharp_movements if s.get("bookmaker") == bookmaker]
            if len(bookmaker_data) >= 2:
                first = bookmaker_data[0]
                last = bookmaker_data[-1]
                result["movements"].append({
                    "bookmaker": bookmaker,
                    "home_change": self.calculate_movement(
                        first.get("home_odds", 0), 
                        last.get("home_odds", 0)
                    ),
                    "draw_change": self.calculate_movement(
                        first.get("draw_odds", 0), 
                        last.get("draw_odds", 0)
                    ),
                    "away_change": self.calculate_movement(
                        first.get("away_odds", 0), 
                        last.get("away_odds", 0)
                    )
                })
        
        return result