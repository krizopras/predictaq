// Backend her zaman Supabase/Netlify Edge Function DEĞİL, ayrı barındırılan
// FastAPI servisidir (Railway/Render/Fly). Bkz. proje kökündeki README.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function get(path) {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`${path} -> HTTP ${res.status}`);
  }
  return res.json();
}

// Frontend sadece GET /matches, /predictions, /odds gibi hazır veriyi okur --
// canlı veri toplama ve model eğitimi tamamen arka planda (GitHub Actions ->
// /api/v1/admin/*) gerçekleşir, kullanıcı sayfayı açtığında hiçbir ağır iş
// tetiklenmez.
export const api = {
  matches: (params = {}) => {
    const qs = new URLSearchParams(params).toString();
    return get(`/api/v1/matches/${qs ? `?${qs}` : ""}`);
  },
  predictionForMatch: (matchId) => get(`/api/v1/predictions/match/${matchId}`),
  oddsAnalysis: (matchId) => get(`/api/v1/odds/analysis/${matchId}`),
  health: () => get("/health"),
};
