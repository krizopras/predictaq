import React, { useEffect, useMemo, useState } from "react";
import { api } from "./api.js";

const STATUS_LABEL = {
  scheduled: "Planlanan",
  live: "Canlı",
  finished: "Bitti",
  postponed: "Ertelendi",
  cancelled: "İptal",
};

function pct(v) {
  if (v === null || v === undefined) return "—";
  return `${Math.round(v * 100)}%`;
}

function odds(v) {
  if (v === null || v === undefined) return "—";
  return v.toFixed(2);
}

function ProbBar({ home, draw, away }) {
  const h = Math.max(0, home ?? 0) * 100;
  const d = Math.max(0, draw ?? 0) * 100;
  const a = Math.max(0, away ?? 0) * 100;
  return (
    <div className="prob-bar" title={`Ev ${pct(home)} · Berabere ${pct(draw)} · Deplasman ${pct(away)}`}>
      <span className="prob-bar__seg prob-bar__seg--home" style={{ width: `${h}%` }} />
      <span className="prob-bar__seg prob-bar__seg--draw" style={{ width: `${d}%` }} />
      <span className="prob-bar__seg prob-bar__seg--away" style={{ width: `${a}%` }} />
    </div>
  );
}

function MatchRow({ match }) {
  const hasPrediction = match.model_home_prob !== null && match.model_home_prob !== undefined;
  const date = new Date(match.date);
  return (
    <div className="row">
      <div className="row__status">
        <span className={`dot dot--${match.status}`} />
        <span className="row__status-label">{STATUS_LABEL[match.status] || match.status}</span>
        <span className="row__time">
          {date.toLocaleDateString("tr-TR", { day: "2-digit", month: "short" })}{" "}
          {date.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" })}
        </span>
      </div>

      <div className="row__teams">
        <span className="row__team">{match.home_team}</span>
        <span className="row__score">
          {match.home_score ?? "-"} : {match.away_score ?? "-"}
        </span>
        <span className="row__team">{match.away_team}</span>
      </div>

      <div className="row__prediction">
        {hasPrediction ? (
          <>
            <ProbBar home={match.model_home_prob} draw={match.model_draw_prob} away={match.model_away_prob} />
            <div className="row__probs">
              <span>1: {pct(match.model_home_prob)}</span>
              <span>X: {pct(match.model_draw_prob)}</span>
              <span>2: {pct(match.model_away_prob)}</span>
            </div>
            {match.model_confidence !== null && match.model_confidence !== undefined && (
              <span className="row__confidence">güven {Math.round(match.model_confidence)}</span>
            )}
          </>
        ) : (
          <span className="row__no-prediction">tahmin bekleniyor</span>
        )}
      </div>

      <div className="row__odds">
        <div className="odds-grid">
          <span className="odds-grid__label">1</span>
          <span className="odds-grid__label">X</span>
          <span className="odds-grid__label">2</span>
          <span className="odds-grid__value">{odds(match.closing_home_odds)}</span>
          <span className="odds-grid__value">{odds(match.closing_draw_odds)}</span>
          <span className="odds-grid__value">{odds(match.closing_away_odds)}</span>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [matches, setMatches] = useState([]);
  const [status, setStatus] = useState("scheduled");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const data = await api.matches({ status, limit: 50 });
        if (!cancelled) {
          setMatches(data);
          setLastUpdated(new Date());
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    // Backend zaten 15 dakikada bir GitHub Actions ile tazeleniyor; frontend
    // burada sadece son yazılan veriyi okumak için hafif bir polling yapıyor.
    const interval = setInterval(load, 60_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [status]);

  const counts = useMemo(() => {
    return { total: matches.length };
  }, [matches]);

  return (
    <div className="app">
      <header className="header">
        <div className="header__brand">
          <span className="header__mark">PIQ</span>
          <div>
            <h1>PredictaIQ</h1>
            <p className="header__tagline">Ensemble futbol tahmin motoru — canlı skor · piyasa oranı · model olasılığı</p>
          </div>
        </div>
        <div className="header__meta">
          {lastUpdated && (
            <span className="header__updated">
              son güncelleme {lastUpdated.toLocaleTimeString("tr-TR")}
            </span>
          )}
        </div>
      </header>

      <nav className="tabs">
        {["scheduled", "live", "finished"].map((s) => (
          <button
            key={s}
            className={`tabs__item ${status === s ? "tabs__item--active" : ""}`}
            onClick={() => setStatus(s)}
          >
            {STATUS_LABEL[s]}
          </button>
        ))}
      </nav>

      <main className="board">
        {loading && matches.length === 0 && <p className="board__empty">Yükleniyor…</p>}
        {error && (
          <p className="board__error">
            Veri alınamadı: {error}. Backend adresini (VITE_API_BASE_URL) kontrol edin.
          </p>
        )}
        {!loading && !error && matches.length === 0 && (
          <p className="board__empty">Bu kategoride maç yok.</p>
        )}

        <div className="board__list">
          {matches.map((m) => (
            <MatchRow key={m.id} match={m} />
          ))}
        </div>
      </main>

      <footer className="footer">
        <span>{counts.total} maç · veri kaynağı: SportsData.io / The Odds API</span>
      </footer>
    </div>
  );
}
