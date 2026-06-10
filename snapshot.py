#!/usr/bin/env python3
"""Kalshi calibration-ledger snapshotter.

The data spine for all prediction-market research in this project: continuously
logs market quotes and eventual outcomes into SQLite, producing a labeled
(price, time-to-close, category) -> outcome dataset for calibration studies,
divergence half-life measurement, and strategy backtests.

Design (validated against the live API, June 2026):
  - /events?status=open excludes the ~tens-of-thousands of auto-generated
    KXMVE* parlay markets that flood /markets — so the universe sweep paginates
    /events with with_nested_markets=true (~40 pages, ~85MB) every FULL_SWEEP_S.
  - Between sweeps, a HOT tier (markets closing soon or actively trading) is
    re-quoted every CYCLE_S via batched /markets?tickers=... calls.
  - Rows are deduped: a snapshot is written only when the quote tuple changed
    or the last write is older than FORCE_WRITE_S.
  - Unresolved markets past close are swept in batches until a terminal
    status/result appears.
  - Unauthenticated access is rate-limited (429s observed live): global pacer
    + exponential backoff with Retry-After support.

Usage:
  python snapshot.py run      # main loop (long-running daemon)
  python snapshot.py sweep    # one full universe sweep, then exit
  python snapshot.py status   # print ledger stats
"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ledger.db")

CYCLE_S = 300            # hot-tier cadence
FULL_SWEEP_S = 7200      # full universe sweep cadence
FORCE_WRITE_S = 3600     # write a row even if unchanged after this long
HOT_CLOSE_H = 48         # markets closing within this many hours are hot
HOT_MAX = 4000           # cap on hot-tier size (closest-to-close + most active first)
SETTLE_BATCH = 300       # unresolved tickers checked per cycle
REQ_GAP_S = 0.25         # global pacing between requests (~4 req/s, unauthenticated)
PAGE_LIMIT = 200


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cents(dollar_str) -> int | None:
    """'0.3200' -> 32. None/'' -> None."""
    if dollar_str in (None, ""):
        return None
    try:
        return round(float(dollar_str) * 100)
    except (TypeError, ValueError):
        return None


def fnum(s) -> float | None:
    if s in (None, ""):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


class Api:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self._last = 0.0

    def get(self, path: str, params: dict | None = None, tries: int = 6):
        for k in range(tries):
            wait = self._last + REQ_GAP_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.s.get(f"{BASE}{path}", params=params, timeout=30)
                self._last = time.monotonic()
            except requests.RequestException as e:
                if k == tries - 1:
                    raise
                time.sleep(min(60, 5 * 2**k))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                if k == tries - 1:
                    r.raise_for_status()
                retry_after = fnum(r.headers.get("Retry-After"))
                time.sleep(retry_after if retry_after else min(60, 5 * 2**k))
                continue
            r.raise_for_status()
        raise RuntimeError("unreachable")


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  event_ticker TEXT PRIMARY KEY,
  series_ticker TEXT, category TEXT, title TEXT, sub_title TEXT,
  mutually_exclusive INTEGER, strike_period TEXT,
  first_seen TEXT, last_seen TEXT
);
CREATE TABLE IF NOT EXISTS markets (
  ticker TEXT PRIMARY KEY,
  event_ticker TEXT, market_type TEXT, title TEXT,
  yes_sub_title TEXT, no_sub_title TEXT, strike_type TEXT,
  open_time TEXT, close_time TEXT,
  status TEXT, result TEXT, settled_ts TEXT,
  rules_primary TEXT,
  first_seen TEXT, last_seen TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (
  ts TEXT NOT NULL, ticker TEXT NOT NULL,
  yes_bid INTEGER, yes_ask INTEGER, no_bid INTEGER, no_ask INTEGER,
  last_price INTEGER,
  volume REAL, volume_24h REAL, open_interest REAL, liquidity REAL,
  yes_bid_size REAL, yes_ask_size REAL
);
CREATE INDEX IF NOT EXISTS idx_snap_ticker_ts ON snapshots(ticker, ts);
CREATE INDEX IF NOT EXISTS idx_snap_ts ON snapshots(ts);
CREATE INDEX IF NOT EXISTS idx_mkt_unresolved ON markets(result, close_time);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class Ledger:
    def __init__(self, path: str = DB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        # ticker -> (quote_hash, last_write_monotonic) for dedup
        self._last_write: dict[str, tuple[int, float]] = {}

    def upsert_event(self, e: dict, now: str):
        self.db.execute(
            """INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(event_ticker) DO UPDATE SET last_seen=excluded.last_seen""",
            (e.get("event_ticker"), e.get("series_ticker"), e.get("category"),
             e.get("title"), e.get("sub_title"),
             1 if e.get("mutually_exclusive") else 0, e.get("strike_period"),
             now, now),
        )

    def upsert_market(self, m: dict, now: str):
        self.db.execute(
            """INSERT INTO markets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(ticker) DO UPDATE SET
                 status=excluded.status, close_time=excluded.close_time,
                 last_seen=excluded.last_seen,
                 result=CASE WHEN excluded.result!='' THEN excluded.result ELSE markets.result END,
                 settled_ts=CASE WHEN excluded.result!='' AND (markets.result IS NULL OR markets.result='')
                                 THEN excluded.settled_ts ELSE markets.settled_ts END""",
            (m.get("ticker"), m.get("event_ticker"), m.get("market_type"), m.get("title"),
             m.get("yes_sub_title"), m.get("no_sub_title"), m.get("strike_type"),
             m.get("open_time"), m.get("close_time"),
             m.get("status"), m.get("result") or "",
             now if m.get("result") else None,
             m.get("rules_primary"), now, now),
        )

    def snap(self, m: dict, now: str) -> bool:
        """Write a quote row if changed or stale. Returns True if written."""
        row = (
            cents(m.get("yes_bid_dollars")), cents(m.get("yes_ask_dollars")),
            cents(m.get("no_bid_dollars")), cents(m.get("no_ask_dollars")),
            cents(m.get("last_price_dollars")),
            fnum(m.get("volume_fp")), fnum(m.get("volume_24h_fp")),
            fnum(m.get("open_interest_fp")), fnum(m.get("liquidity_dollars")),
            fnum(m.get("yes_bid_size_fp")), fnum(m.get("yes_ask_size_fp")),
        )
        if all(v in (None, 0, 0.0) for v in row):
            return False  # dead listing: no quotes, no trades yet
        ticker = m["ticker"]
        h = hash(row[:5] + row[5:7])  # quotes + volume define "changed"
        prev = self._last_write.get(ticker)
        if prev and prev[0] == h and time.monotonic() - prev[1] < FORCE_WRITE_S:
            return False
        self.db.execute("INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (now, ticker) + row)
        self._last_write[ticker] = (h, time.monotonic())
        return True

    def unresolved_past_close(self, limit: int) -> list[str]:
        cur = self.db.execute(
            """SELECT ticker FROM markets
               WHERE (result IS NULL OR result='') AND close_time < ?
               ORDER BY close_time LIMIT ?""",
            (utcnow(), limit))
        return [r[0] for r in cur.fetchall()]

    def hot_tickers(self) -> list[str]:
        """Markets closing within HOT_CLOSE_H, then most-active, capped at HOT_MAX."""
        cur = self.db.execute(
            """SELECT m.ticker FROM markets m
               LEFT JOIN (SELECT ticker, MAX(ts) mts FROM snapshots GROUP BY ticker) s
                 ON s.ticker = m.ticker
               LEFT JOIN snapshots sn ON sn.ticker = s.ticker AND sn.ts = s.mts
               WHERE (m.result IS NULL OR m.result='') AND m.close_time > ?
               ORDER BY (m.close_time < datetime('now', ?)) DESC,
                        COALESCE(sn.volume_24h, 0) DESC
               LIMIT ?""",
            (utcnow(), f"+{HOT_CLOSE_H} hours", HOT_MAX))
        return [r[0] for r in cur.fetchall()]

    def commit(self):
        self.db.commit()

    def stats(self) -> dict:
        q = lambda sql: self.db.execute(sql).fetchone()[0]
        return {
            "events": q("SELECT COUNT(*) FROM events"),
            "markets": q("SELECT COUNT(*) FROM markets"),
            "resolved": q("SELECT COUNT(*) FROM markets WHERE result IN ('yes','no')"),
            "snapshots": q("SELECT COUNT(*) FROM snapshots"),
            "latest_snapshot": q("SELECT MAX(ts) FROM snapshots"),
            "db_mb": round(os.path.getsize(DB_PATH) / 1e6, 1) if os.path.exists(DB_PATH) else 0,
        }


def full_sweep(api: Api, led: Ledger) -> tuple[int, int, int]:
    """Paginate all open events with nested markets. Returns (events, markets, rows)."""
    now = utcnow()
    cursor, n_ev, n_mk, n_rows, pages = "", 0, 0, 0, 0
    while True:
        params = {"status": "open", "limit": PAGE_LIMIT, "with_nested_markets": "true"}
        if cursor:
            params["cursor"] = cursor
        d = api.get("/events", params)
        evs = d.get("events", [])
        for e in evs:
            led.upsert_event(e, now)
            n_ev += 1
            for m in e.get("markets") or []:
                led.upsert_market(m, now)
                n_mk += 1
                if led.snap(m, now):
                    n_rows += 1
        led.commit()
        cursor = d.get("cursor")
        pages += 1
        if not cursor or not evs or pages > 300:
            break
    return n_ev, n_mk, n_rows


def batch_quotes(api: Api, led: Ledger, tickers: list[str]) -> tuple[int, int]:
    """Re-quote tickers via /markets?tickers=... (50 per call). Returns (fetched, rows)."""
    now, fetched, rows = utcnow(), 0, 0
    for i in range(0, len(tickers), 50):
        d = api.get("/markets", {"tickers": ",".join(tickers[i:i + 50]), "limit": 50})
        for m in d.get("markets", []):
            led.upsert_market(m, now)
            fetched += 1
            if led.snap(m, now):
                rows += 1
    led.commit()
    return fetched, rows


def settle_sweep(api: Api, led: Ledger) -> int:
    """Check unresolved past-close markets for terminal results."""
    tickers = led.unresolved_past_close(SETTLE_BATCH)
    if not tickers:
        return 0
    now, resolved = utcnow(), 0
    for i in range(0, len(tickers), 50):
        d = api.get("/markets", {"tickers": ",".join(tickers[i:i + 50]), "limit": 50})
        for m in d.get("markets", []):
            led.upsert_market(m, now)
            if m.get("result") in ("yes", "no"):
                resolved += 1
    led.commit()
    return resolved


_stop = False


def _sig(_n, _f):
    global _stop
    _stop = True


def run():
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    api, led = Api(), Ledger()
    last_full = 0.0
    print(f"[{utcnow()}] snapshotter starting (db={DB_PATH})", flush=True)
    while not _stop:
        t0 = time.monotonic()
        try:
            if time.monotonic() - last_full > FULL_SWEEP_S or last_full == 0.0:
                ev, mk, rows = full_sweep(api, led)
                last_full = time.monotonic()
                print(f"[{utcnow()}] FULL sweep: {ev} events, {mk} markets, {rows} rows "
                      f"({time.monotonic()-t0:.0f}s)", flush=True)
            else:
                hot = led.hot_tickers()
                fetched, rows = batch_quotes(api, led, hot)
                print(f"[{utcnow()}] hot cycle: {len(hot)} hot, {fetched} fetched, "
                      f"{rows} rows ({time.monotonic()-t0:.0f}s)", flush=True)
            resolved = settle_sweep(api, led)
            if resolved:
                print(f"[{utcnow()}] settle sweep: +{resolved} resolved", flush=True)
        except Exception as e:  # log and keep the daemon alive
            print(f"[{utcnow()}] ERROR {type(e).__name__}: {e}", flush=True)
            time.sleep(30)
        sleep_left = CYCLE_S - (time.monotonic() - t0)
        while sleep_left > 0 and not _stop:
            time.sleep(min(5, sleep_left))
            sleep_left -= 5
    led.commit()
    print(f"[{utcnow()}] stopped cleanly", flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "sweep":
        api, led = Api(), Ledger()
        t0 = time.monotonic()
        ev, mk, rows = full_sweep(api, led)
        print(f"sweep done: {ev} events, {mk} markets, {rows} snapshot rows "
              f"in {time.monotonic()-t0:.0f}s")
        print(json.dumps(led.stats(), indent=2))
    elif cmd == "status":
        print(json.dumps(Ledger().stats(), indent=2))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
