#!/usr/bin/env python3
"""Per-station, per-model bias/skill calibration for Kalshi weather markets.

For each city and forecast model we compare:
  forecast: the model's DAY-AHEAD hourly temps (Open-Meteo previous-runs API,
            `temperature_2m_previous_day1`), maxed over the local day —
            i.e. "what this model said yesterday about today's high",
  observed: the official station daily max from ACIS (data.rcc-acis.org),
            the same climate record NWS daily climate reports are built on.

The per-model bias estimated this way absorbs model bias, grid-vs-station
representativeness error, AND the hourly-sampling underestimate of the true
max — because it is measured on exactly the quantity we later predict.

Outputs data/weather_calib.json:
  {city: {"models": {model: {bias, mae, n}}, "asof": ..., "walkforward": {...}}}

A walk-forward check (fit on days < d, predict day d) reports whether the
bias-corrected, skill-weighted blend actually beats each raw model
out-of-sample before we trust it with money.

Usage: python calibrate.py NY [--days 60]
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, timedelta

import requests

from scan_weather import CITIES  # station coords + tz live in one place

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "weather_calib.json")
MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"]
PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ACIS_URL = "https://data.rcc-acis.org/StnData"

# ACIS station ids for the settlement station of each Kalshi market.
# Only set when verified against the market rulebook.
ACIS_SID = {"NY": "NYC"}


def observed_highs(sid: str, start: str, end: str) -> dict[str, float]:
    """Official daily max temps (F) from ACIS for [start, end]."""
    r = requests.post(ACIS_URL, json={
        "sid": sid, "sdate": start, "edate": end, "elems": [{"name": "maxt"}]},
        timeout=30)
    r.raise_for_status()
    out = {}
    for day, val in r.json().get("data", []):
        try:
            out[day] = float(val)
        except (TypeError, ValueError):
            continue  # 'M' (missing) / 'T' etc.
    return out


def day_ahead_forecast_highs(lat: float, lon: float, tz: str, past_days: int,
                             models: list[str]) -> dict[str, dict[str, float]]:
    """{model: {local_date: forecast_high_F}} from the previous-runs archive."""
    r = requests.get(PREV_RUNS_URL, params={
        "latitude": lat, "longitude": lon, "timezone": tz,
        "temperature_unit": "fahrenheit",
        "past_days": min(past_days, 92), "forecast_days": 1,
        "hourly": "temperature_2m_previous_day1", "models": ",".join(models)},
        timeout=30)
    r.raise_for_status()
    h = r.json()["hourly"]
    times = h["time"]
    out: dict[str, dict[str, float]] = {}
    for key, series in h.items():
        if key == "time":
            continue
        model = key.replace("temperature_2m_previous_day1_", "")
        days: dict[str, float] = {}
        for t, v in zip(times, series):
            if v is None:
                continue
            d = t[:10]
            days[d] = max(days.get(d, -999.0), v)
        out[model] = days
    return out


def fit(pairs: dict[str, list[tuple[float, float]]]) -> dict[str, dict]:
    """pairs: {model: [(forecast, observed)]} -> {model: {bias, mae, n}}."""
    res = {}
    for m, ps in pairs.items():
        if not ps:
            continue
        errs = [f - o for f, o in ps]
        bias = sum(errs) / len(errs)
        mae = sum(abs(e) for e in errs) / len(errs)
        res[m] = {"bias": round(bias, 2), "mae": round(mae, 2), "n": len(ps)}
    return res


def walk_forward(pairs: dict[str, list[tuple[str, float, float]]], min_fit: int = 10) -> dict:
    """Out-of-sample check: for each day d, fit bias/skill on days < d, then
    predict d with (a) each raw model, (b) the corrected skill-weighted blend."""
    dates = sorted({d for ps in pairs.values() for d, _, _ in ps})
    raw_err = {m: [] for m in pairs}
    blend_err = []
    for d in dates[min_fit:]:
        hist = {m: [(f, o) for (dd, f, o) in ps if dd < d] for m, ps in pairs.items()}
        today = {m: next(((f, o) for (dd, f, o) in ps if dd == d), None) for m, ps in pairs.items()}
        stats = fit(hist)
        num = den = 0.0
        obs = None
        for m, fo in today.items():
            if fo is None or m not in stats or stats[m]["n"] < min_fit:
                continue
            f, obs = fo
            raw_err[m].append(abs(f - obs))
            w = 1.0 / max(stats[m]["mae"], 0.5)
            num += w * (f - stats[m]["bias"])
            den += w
        if den and obs is not None:
            blend_err.append(abs(num / den - obs))
    out = {f"raw_{m}": round(sum(e) / len(e), 2) for m, e in raw_err.items() if e}
    if blend_err:
        out["corrected_blend"] = round(sum(blend_err) / len(blend_err), 2)
        out["eval_days"] = len(blend_err)
    return out


def calibrate(city: str, days: int) -> dict:
    cfg = CITIES[city]
    sid = ACIS_SID.get(city)
    if not sid:
        raise SystemExit(f"No verified ACIS station for {city} — verify the market "
                         f"rulebook's settlement station first, then add it to ACIS_SID.")
    end = date.today() - timedelta(days=1)       # today’s obs is incomplete
    start = end - timedelta(days=days)
    obs = observed_highs(sid, start.isoformat(), end.isoformat())
    fcs = day_ahead_forecast_highs(cfg["lat"], cfg["lon"], cfg["tz"], days + 2, MODELS)

    pairs_dated = {m: [(d, f, obs[d]) for d, f in sorted(dmap.items()) if d in obs]
                   for m, dmap in fcs.items()}
    stats = fit({m: [(f, o) for _, f, o in ps] for m, ps in pairs_dated.items()})
    wf = walk_forward(pairs_dated)
    return {"models": stats, "walkforward": wf,
            "station": cfg["station"], "acis_sid": sid,
            "asof": date.today().isoformat(), "window_days": days}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", nargs="?", default="NY", choices=list(CITIES))
    ap.add_argument("--days", type=int, default=60)
    a = ap.parse_args()
    result = calibrate(a.city, a.days)

    all_cal = {}
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH) as f:
            all_cal = json.load(f)
    all_cal[a.city] = result
    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
    with open(CALIB_PATH, "w") as f:
        json.dump(all_cal, f, indent=1)

    print(f"{a.city} ({result['station']}) calibrated on {result['window_days']}d "
          f"ending yesterday -> {CALIB_PATH}")
    for m, s in result["models"].items():
        print(f"  {m:<16} bias={s['bias']:+.2f}F  mae={s['mae']:.2f}F  n={s['n']}")
    print("  walk-forward MAE (out-of-sample):",
          json.dumps(result["walkforward"]))


if __name__ == "__main__":
    main()
