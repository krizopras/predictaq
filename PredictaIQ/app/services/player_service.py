"""Kadro ve oyuncu etkisi motoru.

Plan madde 5: "Takım normalde %54 kazanıyor fakat ana golcünün yokluğunda
%47'ye düşüyor" örneğini uygulanabilir kılan servis. Eski kodda `Player`
tablosunda impact_score/xg_contribution alanları vardı ama hiçbir yerde
kullanılmıyordu; bu servis onları gerçek bir olasılık düzeltmesine
çeviriyor.

Yöntem (basit ama savunulabilir bir yaklaşım):
- Her oyuncunun `impact_score`'u (-1..+1 aralığında, pozitif = takıma
  katkısı yüksek) var.
- `is_key_player=True` ve güncel bir 'out'/'suspended' sakatlık kaydı
  varsa, o oyuncunun impact_score'u kadar takımın toplam gücünden
  düşülür.
- Sonuç, PredictionService'te doğrudan Elo/xG'ye eklenecek bir
  "team_strength_adjustment" (yaklaşık +-150 Elo puanı ölçeğinde) olarak
  döner; böylece mevcut Elo/Poisson modellerine minimal invaziv şekilde
  entegre edilebilir.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from sqlalchemy.orm import Session

from app.models import Injury, Player


ACTIVE_INJURY_STATUSES = {"out", "suspended"}
ELO_POINTS_PER_IMPACT_UNIT = 120.0  # impact_score=1.0 kaybı ~ 120 Elo puanı


class PlayerImpactService:
    def get_active_injuries(self, db: Session, team_id) -> List[Injury]:
        return (
            db.query(Injury)
            .filter(Injury.team_id == team_id, Injury.status.in_(ACTIVE_INJURY_STATUSES))
            .filter((Injury.expected_return.is_(None)) | (Injury.expected_return >= datetime.utcnow()))
            .all()
        )

    def team_injury_impact(self, db: Session, team_id) -> Dict:
        """Bir takımın güncel sakatlık/ceza durumuna göre toplam güç kaybını
        hesaplar. Dönen `elo_penalty`, EloService'e doğrudan eklenebilir
        (negatif değer)."""
        injuries = self.get_active_injuries(db, team_id)
        if not injuries:
            return {"elo_penalty": 0.0, "missing_key_players": [], "total_impact": 0.0}

        player_ids = [inj.player_id for inj in injuries]
        players = db.query(Player).filter(Player.id.in_(player_ids)).all()
        players_by_id = {p.id: p for p in players}

        total_impact = 0.0
        missing_key_players = []
        for inj in injuries:
            player = players_by_id.get(inj.player_id)
            if not player:
                continue
            weight = 1.0 if player.is_key_player else 0.35
            impact = (player.impact_score or 0.0) * weight
            total_impact += impact
            if player.is_key_player:
                missing_key_players.append({
                    "name": player.name,
                    "position": player.position,
                    "status": inj.status,
                    "impact_score": player.impact_score,
                })

        return {
            "elo_penalty": -total_impact * ELO_POINTS_PER_IMPACT_UNIT,
            "missing_key_players": missing_key_players,
            "total_impact": total_impact,
        }
