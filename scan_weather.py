#!/usr/bin/env python3
"""Kalshi weather edge scanner (read-only, free data).

For a city's daily-high market it:
  1. pulls the live bracket prices from Kalshi,
  2. pulls the ensemble daily-high distribution from Open-Meteo,
  3. turns the members into a smoothed probability for each bracket,
  4. compares model prob vs market price, fee-adjusted, and flags edges.

This finds *candidate* edges and sizes them. It does NOT trade.
Usage:  ./.venv/bin/python scan_weather.py NY [--date 2026-06-09] [--edge 0.03]
"""
from __future__ import annotations

import argparse
import math
import re

import numpy as np

from kalshi import Kalshi, taker_fee

# --- City config -------------------------------------------------------------
# station coords MUST match the NWS station Kalshi settles on, or the model is
# biased. NY (Central Park / KNYC) is confirmed. Others are best-effort and
# should be verified against each market's rulebook before trusting.
CITIES = {
    "NY":  {"series": "KXHIGHNY",  "lat": 40.78, "lon": -73.97, "tz": "America/New_York",    "verified": True,  "station": "Central Park (KNYC)", "nws": "KNYC"},
    "CHI": {"series": "KXHIGHCHI", "lat": 41.79, "lon": -87.75, "tz": "America/Chicago",     "verified": False, "station": "Midway? verify",      "nws": None},
    "MIA": {"series": "KXHIGHMIA", "lat": 25.79, "lon": -80.32, "tz": "America/New_York",    "verified": False, "station": "Miami Intl? verify",  "nws": None},
    "AUS": {"series": "KXHIGHAUS", "lat": 30.18, "lon": -97.68, "tz": "America/Chicago",     "verified": False, "station": "Camp Mabry? verify",  "nws": None},
    "LAX": {"series": "KXHIGHLAX", "lat": 33.94, "lon": -118.40, "tz": "America/Los_Angeles", "verified": False, "station": "LAX? verify",        "nws": None},
    "DEN": {"series": "KXHIGHDEN", "lat": 39.85, "lon": -104.66, "tz": "America/Denver",     "verified": False, "station": "DEN? verify",         "nws": None},
    "PHIL":{"series": "KXHIGHPHIL","lat": 39.87, "lon": -75.23, "tz": "America/New_York",    "verified": False, "station": "PHL? verify",         "nws": None},
}
CALIB_PATH = "data/weather_calib.json"
_MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bracket_bounds(subtitle: str) -> tuple[float, float] | None:
    """'84° to 85°' -> (83.5, 85.5); '86° or above' -> (85.5, inf); '77° or below' -> (-inf, 77.5)."""
    s = subtitle.replace("°", " ")
    nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", s)]
    low_kw = "below" in s.lower() or "under" in s.lower()
    high_kw = "above" in s.lower() or "over" in s.lower()
    if not nums:
        return None
    if high_kw:
        return (nums[0] - 0.5, math.inf)
    if low_kw:
        return (-math.inf, nums[0] + 0.5)
    if len(nums) >= 2:
        return (min(nums) - 0.5, max(nums) + 0.5)
    return (nums[0] - 0.5, nums[0] + 0.5)


def kde_bracket_prob(highs: np.ndarray, lo: float, hi: float, bw: float,
                     weights: np.ndarray | None = None) -> float:
    """P(lo < high < hi) under a (weighted) Gaussian KDE over the member highs."""
    if len(highs) == 0:
        return float("nan")
    w = np.ones(len(highs)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    def cdf(x):
        if x == math.inf:
            return 1.0
        if x == -math.inf:
            return 0.0
        return float(np.sum(w * np.array([_norm_cdf((x - h) / bw) for h in highs])))
    return max(0.0, min(1.0, cdf(hi) - cdf(lo)))


def calibrated_members(by_model: dict[str, list[float]], calib: dict | None,
                       floor_f: float | None) -> tuple[np.ndarray, np.ndarray, str]:
    """Apply per-model bias shifts + skill weights (+ same-day floor).

    Returns (member_highs, member_weights, summary_line). Without calibration,
    members pool unweighted — the configuration the first live scan showed to
    be untrustworthy (raw multi-model spread >> tradable edge).
    """
    highs, weights, parts = [], [], []
    models = (calib or {}).get("models", {})
    for model, hs in by_model.items():
        st = models.get(model)
        bias = st["bias"] if st else 0.0
        w = 1.0 / max(st["mae"], 0.5) if st else 1.0
        for h in hs:
            v = h - bias
            if floor_f is not None:
                v = max(v, floor_f)
            highs.append(v)
            weights.append(w)
        parts.append(f"{model.split('_')[0]}(n={len(hs)},b={bias:+.1f},w={w:.2f})")
    note = " ".join(parts) + (f"  floor={floor_f:.0f}F" if floor_f is not None else "")
    return np.array(highs), np.array(weights), note


def date_from_ticker(ticker: str) -> str | None:
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker)
    if not m:
        return None
    yy, mon, dd = m.groups()
    return f"20{yy}-{_MONTHS.get(mon, 0):02d}-{int(dd):02d}"


def run(code: str, date: str | None, edge_thr: float, models: list[str] | None,
        log: bool = False):
    import json as _json
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from weather import ensemble_daily_highs_by_model, station_running_max_f

    cfg = CITIES[code]
    k = Kalshi()
    markets = k.markets(cfg["series"])
    if not markets:
        print(f"No open markets for {cfg['series']}."); return

    calib = None
    try:
        with open(CALIB_PATH) as f:
            calib = _json.load(f).get(code)
    except (OSError, ValueError):
        pass
    if calib is None:
        print(f"⚠️  No calibration for {code} (run: python calibrate.py {code}) — "
              f"using raw pooled ensemble, which the live 06-09 scan showed is untrustworthy.\n")

    # group brackets by settlement date
    by_date: dict[str, list] = {}
    for m in markets:
        d = date_from_ticker(m["ticker"])
        if d:
            by_date.setdefault(d, []).append(m)
    dates = [date] if date else sorted(by_date)
    if not cfg["verified"]:
        print(f"⚠️  Station coords for {code} are UNVERIFIED — confirm against Kalshi's rulebook before trusting probabilities.\n")

    today_local = datetime.now(ZoneInfo(cfg["tz"])).date().isoformat()
    for d in dates:
        ms = by_date.get(d, [])
        if not ms:
            continue
        by_model = ensemble_daily_highs_by_model(cfg["lat"], cfg["lon"], d, cfg["tz"], models)
        floor_f = None
        if d == today_local and cfg.get("nws"):
            floor_f = station_running_max_f(cfg["nws"], d, cfg["tz"])
        highs, w, note = calibrated_members(by_model, calib, floor_f)
        print("=" * 96)
        print(f"{code}  {cfg['station']}   high-temp market for {d}   ({cfg['series']})")
        if len(highs) == 0:
            print("  No ensemble data returned for this date — skipping.\n"); continue
        print(f"  calib: {note}")
        # Silverman-ish bandwidth with a floor (rounding + model granularity)
        std = float(np.std(highs, ddof=1)) if len(highs) > 1 else 2.0
        bw = max(0.75, 0.9 * std * len(highs) ** (-1 / 5))
        print(f"  ensemble: n={len(highs)} members | mean={np.mean(highs):.1f}°F  std={std:.1f}  "
              f"range=[{np.min(highs):.0f},{np.max(highs):.0f}]  bw={bw:.2f}")
        print(f"  {'bracket':<16}{'model_P':>8}{'mkt_yes':>9}{'mkt_no':>8}{'edge_YES':>10}{'edge_NO':>9}   action")
        print("  " + "-" * 92)

        rows = []
        for m in ms:
            sub = m.get("yes_sub_title") or m.get("subtitle") or m.get("ticker")
            b = bracket_bounds(sub)
            if not b:
                continue
            p = kde_bracket_prob(highs, b[0], b[1], bw, w)
            q = k.orderbook(m["ticker"])
            ya, na = q.yes_ask, q.no_ask
            e_yes = (p - (ya + taker_fee(ya))) if ya is not None else None
            e_no = ((1 - p) - (na + taker_fee(na))) if na is not None else None
            rows.append((sub, p, ya, na, e_yes, e_no, q))

        journal_rows = []
        for sub, p, ya, na, e_yes, e_no, q in rows:
            action = "—"
            # tradeable zone: avoid fee-tax longshots (<0.15) and near-certain (>0.90)
            best = None
            if e_yes is not None and ya is not None and 0.15 <= ya <= 0.90 and e_yes >= edge_thr:
                best = ("BUY YES", e_yes, ya, q.yes_ask_size)
            if e_no is not None and na is not None and 0.15 <= na <= 0.90 and e_no >= edge_thr:
                if best is None or e_no > best[1]:
                    best = ("BUY NO", e_no, na, q.no_ask_size)
            if best:
                sz = f"~{int(best[3])}" if best[3] else "?"
                action = f"✅ {best[0]} @ {best[2]:.2f}  (+{best[1]*100:.1f}¢/ct, {sz} ct)"
            ys = f"{ya:.2f}" if ya is not None else " n/a"
            ns = f"{na:.2f}" if na is not None else " n/a"
            eys = f"{e_yes*100:+.1f}¢" if e_yes is not None else "  n/a"
            ens = f"{e_no*100:+.1f}¢" if e_no is not None else "  n/a"
            print(f"  {sub:<16}{p*100:>7.1f}%{ys:>9}{ns:>8}{eys:>10}{ens:>9}   {action}")
            journal_rows.append({
                "strategy": "weather", "ticker": q.ticker, "model_p": round(p, 4),
                "mkt_yes_ask": round(ya * 100) if ya is not None else None,
                "mkt_no_ask": round(na * 100) if na is not None else None,
                "flagged": best[0] if best else "",
                "note": f"{code} {d} {note}",
            })
        if log and journal_rows:
            from journal import log_rows
            print(f"  journal: logged {log_rows(journal_rows)} brackets")
        print()
    print("Edge = model_prob − (price + Kalshi taker fee). Flagged only inside the 0.15–0.90 price band.")
    print("⚠️  Raw ensemble has known warm/cool bias by station — calibrate against history before sizing up.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="NY", choices=list(CITIES))
    ap.add_argument("--date", help="YYYY-MM-DD (defaults to all open dates)")
    ap.add_argument("--edge", type=float, default=0.03, help="min edge in $ to flag (default 0.03 = 3¢)")
    ap.add_argument("--models", help="comma list of Open-Meteo ensemble models")
    ap.add_argument("--log", action="store_true", help="append all brackets to the edge journal")
    a = ap.parse_args()
    run(a.city, a.date, a.edge, a.models.split(",") if a.models else None, log=a.log)
