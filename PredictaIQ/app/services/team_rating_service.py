"""Zaman ağırlıklı takım gücü / form motoru.

Plan madde 2-3'ün ana talebi: "Son maç 6 ay önceki maçtan daha önemli"
ve son3/son5/son10/sezon form bileşenlerinin ayrı ayrı hesaplanması.
Eski koddaki `Team.form` tek bir statik sütundu ve hiçbir yerde
güncellenmiyordu. Bu servis:

1. Bir takımın kronolojik maç geçmişini gezerek Elo'yu adım adım günceller
   (EloService kullanarak) ve her maçtan sonra bir `TeamRatingHistory`
   satırı yazar (leakage-free backtest için gerekli: t anındaki rating,
   sadece t'den önceki maçlardan hesaplanmış olmalı).
2. Üstel zaman ağırlıklandırması (exponential half-life decay) ile
   son N maçın "form skorunu" hesaplar -- yakın maçlar daha ağır basar.
3. Team tablosundaki güncel alanları (elo, form_last3/5/10, home_power,
   away_power) senkronize eder.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Match, Team, TeamRatingHistory
from app.services.elo_service import EloService
from app.services.xg_service import XGService


class TeamRatingService:
    def __init__(
        self,
        elo_service: Optional[EloService] = None,
        xg_service: Optional[XGService] = None,
        half_life_days: Optional[int] = None,
    ):
        self.elo = elo_service or EloService()
        self.xg = xg_service or XGService()
        self.half_life_days = half_life_days or settings.elo_time_decay_half_life_days

    # ------------------------------------------------------------------
    # Zaman ağırlıklı form
    # ------------------------------------------------------------------
    def _decay_weight(self, match_date: datetime, as_of: datetime) -> float:
        """Üstel yarı ömür ağırlığı: match_date, as_of'tan half_life kadar
        önceyse ağırlık 0.5 olur, iki katı kadar önceyse 0.25 olur, vb."""
        days_ago = max((as_of - match_date).days, 0)
        return math.pow(0.5, days_ago / max(self.half_life_days, 1))

    def _match_points(self, team_id, match: Match) -> float:
        """0-100 arası form puanı: galibiyet=100, beraberlik=50, mağlubiyet=0."""
        if match.home_score is None or match.away_score is None:
            return 50.0
        is_home = str(match.home_team_id) == str(team_id)
        gf = match.home_score if is_home else match.away_score
        ga = match.away_score if is_home else match.home_score
        if gf > ga:
            return 100.0
        if gf == ga:
            return 50.0
        return 0.0

    def weighted_form(self, team_id, matches: List[Match], as_of: datetime, n: int) -> float:
        """Bir takımın belirli tarihe kadarki son n maçtan, zaman ağırlıklı form
        skorunu hesaplar (0-100 arası)."""
        past = [m for m in matches if m.date < as_of and m.home_score is not None]
        past.sort(key=lambda m: m.date, reverse=True)
        recent = past[:n]
        if not recent:
            return 50.0

        total_w = 0.0
        weighted_sum = 0.0
        for m in recent:
            w = self._decay_weight(m.date, as_of)
            weighted_sum += w * self._match_points(team_id, m)
            total_w += w
        if total_w == 0:
            return 50.0
        return weighted_sum / total_w

    def compute_form_snapshot(self, team_id, matches: List[Match], as_of: datetime) -> dict:
        return {
            "form_last3": self.weighted_form(team_id, matches, as_of, 3),
            "form_last5": self.weighted_form(team_id, matches, as_of, 5),
            "form_last10": self.weighted_form(team_id, matches, as_of, 10),
        }

    # ------------------------------------------------------------------
    # Elo + form geçmişini kronolojik olarak yeniden hesaplama
    # ------------------------------------------------------------------
    def recompute_all_ratings(self, db: Session) -> int:
        """Tüm bitmiş maçları tarih sırasıyla gezip her takım için Elo'yu ve
        zaman ağırlıklı formu adım adım hesaplar, TeamRatingHistory'ye yazar
        ve Team tablosundaki güncel değerleri senkronize eder.

        Geriye kaç maçın işlendiğini döner.
        """
        finished = (
            db.query(Match)
            .filter(Match.status == "finished", Match.home_score.isnot(None))
            .order_by(Match.date.asc())
            .all()
        )
        if not finished:
            return 0

        all_by_team: dict = {}
        for m in finished:
            all_by_team.setdefault(str(m.home_team_id), []).append(m)
            all_by_team.setdefault(str(m.away_team_id), []).append(m)

        current_elo: dict = {}
        home_power: dict = {}
        away_power: dict = {}
        attack: dict = {}   # üstel ortalama, gol/maç -- lig ortalamasına göre ileride ölçeklenir
        defense: dict = {}  # üstel ortalama, yenilen gol/maç

        league_goals_avg = self._league_avg_goals_per_team(finished)

        for m in finished:
            home_id, away_id = str(m.home_team_id), str(m.away_team_id)
            h_elo = current_elo.get(home_id, 1500.0)
            a_elo = current_elo.get(away_id, 1500.0)

            # Bu maçtan ÖNCEKİ bilgiyle snapshot al (sızıntısız -- m'in kendi
            # sonucu henüz rating'e karışmadan önce yazılıyor). Bu, prediction
            # tarafında kullanılan home_elo/away_elo/attack/defense alanlarının
            # ASLA o maçın kendi sonucundan etkilenmemesini garanti eder.
            m.home_elo = h_elo
            m.away_elo = a_elo
            m.elo_difference = h_elo - a_elo

            h_form = self.weighted_form(home_id, all_by_team.get(home_id, []), m.date, 5)
            a_form = self.weighted_form(away_id, all_by_team.get(away_id, []), m.date, 5)
            m.form_difference = h_form - a_form

            # Pre-match hücum/savunma gücünü (0-100, 50=lig ortalaması) bu
            # maçtan ÖNCEKİ üstel ortalamalardan al ve maça yaz. xg_service
            # bunu kullanarak sızıntısız bir "beklenen xG" üretebilecek.
            h_att = attack.get(home_id, 50.0)
            a_def = defense.get(away_id, 50.0)
            a_att = attack.get(away_id, 50.0)
            h_def = defense.get(home_id, 50.0)

            # Maç öncesi beklenen gol (sızıntısız) -- sadece bu ana kadarki
            # hücum/savunma rating'lerinden türetilir, maçın kendi skorundan
            # ETKİLENMEZ.
            home_xg_pre, away_xg_pre = self.xg.estimate_pre_match_xg(h_att, h_def, a_att, a_def)
            m.home_xg_pre = home_xg_pre
            m.away_xg_pre = away_xg_pre

            # Elo'yu bu maçın sonucuna göre güncelle (sıradaki maçlar için)
            new_h, new_a = self.elo.calculate_match_elo_change(h_elo, a_elo, m.home_score, m.away_score)
            current_elo[home_id] = new_h
            current_elo[away_id] = new_a

            # Ev/deplasman gücü: basit üstel ortalama (galibiyet oranı bazlı)
            home_result = 1.0 if m.home_score > m.away_score else (0.5 if m.home_score == m.away_score else 0.0)
            away_result = 1.0 - home_result
            home_power[home_id] = 0.85 * home_power.get(home_id, 50.0) + 0.15 * (home_result * 100)
            away_power[away_id] = 0.85 * away_power.get(away_id, 50.0) + 0.15 * (away_result * 100)

            # Hücum/savunma üstel ortalamalarını BU maçın sonucuyla güncelle
            # (sıradaki maçlar için) -- lig ortalamasına göre 0-100 ölçeğine
            # normalize edilir (50 = tam lig ortalaması).
            if league_goals_avg > 0:
                home_attack_sample = (m.home_score / league_goals_avg) * 50.0
                away_defense_sample = (m.home_score / league_goals_avg) * 50.0  # rakibe yedirdiği gol
                away_attack_sample = (m.away_score / league_goals_avg) * 50.0
                home_defense_sample = (m.away_score / league_goals_avg) * 50.0
                attack[home_id] = 0.80 * attack.get(home_id, 50.0) + 0.20 * home_attack_sample
                defense[away_id] = 0.80 * defense.get(away_id, 50.0) + 0.20 * away_defense_sample
                attack[away_id] = 0.80 * attack.get(away_id, 50.0) + 0.20 * away_attack_sample
                defense[home_id] = 0.80 * defense.get(home_id, 50.0) + 0.20 * home_defense_sample

            db.add(TeamRatingHistory(
                team_id=m.home_team_id, match_id=m.id, as_of=m.date,
                elo=new_h, attack=h_att, defense=h_def,
                form_last3=None, form_last5=h_form, form_last10=None,
                home_power=home_power.get(home_id, 50.0),
            ))
            db.add(TeamRatingHistory(
                team_id=m.away_team_id, match_id=m.id, as_of=m.date,
                elo=new_a, attack=a_att, defense=a_def,
                form_last3=None, form_last5=a_form, form_last10=None,
                away_power=away_power.get(away_id, 50.0),
            ))

        # Güncel (bugünkü) değerlerle Team tablosunu senkronize et
        now = datetime.utcnow()
        teams = db.query(Team).all()
        for team in teams:
            tid = str(team.id)
            team.elo = current_elo.get(tid, team.elo or 1500.0)
            team.attack = attack.get(tid, team.attack or 50.0)
            team.defense = defense.get(tid, team.defense or 50.0)
            team.home_power = home_power.get(tid, team.home_power or 50.0)
            team.away_power = away_power.get(tid, team.away_power or 50.0)
            snapshot = self.compute_form_snapshot(tid, all_by_team.get(tid, []), now)
            team.form_last3 = snapshot["form_last3"]
            team.form_last5 = snapshot["form_last5"]
            team.form = snapshot["form_last5"]
            team.form_last10 = snapshot["form_last10"]

        db.flush()
        return len(finished)

    @staticmethod
    def _league_avg_goals_per_team(matches: List[Match]) -> float:
        scores = [m.home_score for m in matches if m.home_score is not None]
        scores += [m.away_score for m in matches if m.away_score is not None]
        if not scores:
            return 1.35
        return sum(scores) / len(scores)
