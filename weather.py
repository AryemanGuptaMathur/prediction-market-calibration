"""Open-Meteo ensemble client -> per-member daily-high distribution.

Free, no API key for non-commercial use. We pull every ensemble member's hourly
2m temperature, then take each member's max over the target *local* calendar day.
That set of member-highs IS our probability distribution for the day's high.

Endpoint verified live: ensemble-api.open-meteo.com/v1/ensemble
"""
from __future__ import annotations

import requests

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Multiple ensemble systems -> more members -> smoother tails. Falls back to GFS
# alone if a model name is rejected.
DEFAULT_MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"]


def ensemble_daily_highs_by_model(
    lat: float,
    lon: float,
    target_date: str,        # "YYYY-MM-DD" in the station's local timezone
    tz: str,                 # e.g. "America/New_York"
    models: list[str] | None = None,
    timeout: int = 25,
) -> dict[str, list[float]]:
    """{model: [daily-high °F per ensemble member]} for target_date.

    One request per model: the API renames models in combined responses
    (gfs_seamless -> ncep_gefs_seamless), so per-model calls keep attribution
    unambiguous — required for per-model bias correction and skill weights.
    """
    out: dict[str, list[float]] = {}
    for model in (models or DEFAULT_MODELS):
        data = _fetch({
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m", "temperature_unit": "fahrenheit",
            "timezone": tz, "start_date": target_date, "end_date": target_date,
            "models": model,
        }, timeout)
        if not data or "hourly" not in data:
            continue
        hourly = data["hourly"]
        times = hourly.get("time", [])
        highs: list[float] = []
        for k, series in hourly.items():
            if not k.startswith("temperature_2m"):
                continue
            vals = [v for t, v in zip(times, series) if t[:10] == target_date and v is not None]
            if vals:
                highs.append(max(vals))
        if highs:
            out[model] = highs
    return out


def ensemble_daily_highs(lat, lon, target_date, tz, models=None, timeout=25) -> list[float]:
    """Pooled member highs (back-compat wrapper)."""
    by_model = ensemble_daily_highs_by_model(lat, lon, target_date, tz, models, timeout)
    return [h for highs in by_model.values() for h in highs]


def station_running_max_f(nws_station: str, local_date: str, tz: str,
                          timeout: int = 25) -> float | None:
    """Observed running max (°F) at an NWS station over the given LOCAL date.

    Used as a same-day floor: once the station has touched T, no ensemble
    member's daily high can honestly be below T. NWS timestamps are UTC, so
    they are converted to the station's timezone before date-matching.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    try:
        r = requests.get(
            f"https://api.weather.gov/stations/{nws_station}/observations",
            params={"limit": 60},
            headers={"User-Agent": "prediction-market-research"},
            timeout=timeout,
        )
        r.raise_for_status()
        zone = ZoneInfo(tz)
        temps = []
        for f in r.json().get("features", []):
            p = f.get("properties", {})
            ts, v = p.get("timestamp", ""), (p.get("temperature") or {}).get("value")
            if v is None or not ts:
                continue
            local_day = datetime.fromisoformat(ts).astimezone(zone).date().isoformat()
            if local_day == local_date:
                temps.append(v * 9 / 5 + 32)
        return max(temps) if temps else None
    except (requests.RequestException, ValueError):
        return None


def _fetch(params, timeout):
    try:
        r = requests.get(ENSEMBLE_URL, params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.RequestException:
        return None
