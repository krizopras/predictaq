"""Market Movement Model (plan madde 13 - "Model 6").

Eski koddaki OddsService zaten `calculate_movement` ve
`identify_sharp_movement` fonksiyonlarını sağlıyordu, ama bunlar hiçbir
zaman ensemble'a bir OLASILIK olarak katılmıyordu -- sadece görüntüleme
amaçlı ham sayılardı. Bu servis, oran hareketini (opening -> closing)
bağımsız bir "modelin" tahminine çeviriyor:

Fikir: piyasa, ev sahibinin oranını açılıştan kapanışa doğru
düşürüyorsa (yani ev sahibine para giriyorsa), bu genellikle piyasanın
modelin henüz bilmediği bir bilgiye (kadro, hava durumu, sharp bahisçi
hareketi) sahip olduğunun sinyalidir. Model, kapanış olasılığını baz
alır ve hareketin büyüklüğü/yönüyle orantılı küçük bir ek kaymayı o
yöne doğru uygular. Bu kayma sınırlıdır (max +-6 puan) ki piyasayı
kopyalamak yerine yalnızca "momentum" bilgisini eklemiş olsun.
"""
from __future__ import annotations

from typing import Dict, Optional

from app.services.odds_service import OddsService

MAX_SHIFT = 0.06  # olasılık puanı cinsinden üst sınır


class MarketMovementModel:
    def __init__(self, odds_service: Optional[OddsService] = None):
        self.odds = odds_service or OddsService()

    def predict(
        self,
        opening_home: Optional[float], opening_draw: Optional[float], opening_away: Optional[float],
        closing_home: Optional[float], closing_draw: Optional[float], closing_away: Optional[float],
    ) -> Dict:
        closing_probs = self.odds.normalize_odds(
            closing_home or 0, closing_draw or 0, closing_away or 0
        )
        if not all([opening_home, opening_draw, opening_away, closing_home, closing_draw, closing_away]):
            # Açılış verisi yoksa, kapanış olasılığını olduğu gibi döndür
            # (hareket bilgisi katkısı sıfır).
            return {
                "home": closing_probs["home"],
                "draw": closing_probs["draw"],
                "away": closing_probs["away"],
                "shift_applied": {"home": 0.0, "draw": 0.0, "away": 0.0},
                "has_movement_data": False,
            }

        opening_probs = self.odds.normalize_odds(opening_home, opening_draw, opening_away)

        # Olasılık bazında hareket (implied prob artışı = para o tarafa akıyor)
        shift = {
            key: closing_probs[key] - opening_probs[key]
            for key in ("home", "draw", "away")
        }
        # Hareketi hafiflet ve sınırla -- amaç piyasayı kopyalamak değil,
        # sadece momentum sinyali eklemek.
        shift = {k: max(-MAX_SHIFT, min(MAX_SHIFT, v * 0.5)) for k, v in shift.items()}

        adjusted = {
            "home": closing_probs["home"] + shift["home"],
            "draw": closing_probs["draw"] + shift["draw"],
            "away": closing_probs["away"] + shift["away"],
        }
        # Negatifleri temizle ve yeniden normalize et
        adjusted = {k: max(0.001, v) for k, v in adjusted.items()}
        total = sum(adjusted.values())
        adjusted = {k: v / total for k, v in adjusted.items()}

        return {
            **adjusted,
            "shift_applied": shift,
            "has_movement_data": True,
            "opening_implied": opening_probs,
            "closing_implied": closing_probs,
        }
