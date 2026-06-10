#!/usr/bin/env python3
"""Edge journal: the gate between "model prints edges" and "model earns money".

Every scan can log its full probability surface (not just flagged trades) next
to the market's prices at that moment. Outcomes auto-fill from the ledger as
markets settle. `score` then answers the only question that matters:

    Did the model's probabilities beat the market's implied probabilities
    (lower Brier score) on markets that have since resolved?

Until that answer is YES over a meaningful sample, no real money.

Usage:
  python journal.py score [--strategy weather] [--days 30]
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone

from snapshot import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS edge_log (
  ts TEXT NOT NULL,              -- when the scan ran
  strategy TEXT NOT NULL,        -- 'weather', 'basket_arb', ...
  ticker TEXT NOT NULL,
  model_p REAL,                  -- model P(YES)
  mkt_yes_ask INTEGER,           -- cents at scan time
  mkt_no_ask INTEGER,
  flagged TEXT,                  -- 'BUY YES'/'BUY NO'/'' (what the scanner said)
  note TEXT
);
CREATE INDEX IF NOT EXISTS idx_edge_ticker ON edge_log(ticker);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_rows(rows: list[dict], db_path: str = DB_PATH):
    """rows: dicts with strategy/ticker/model_p/mkt_yes_ask/mkt_no_ask/flagged/note."""
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    ts = utcnow()
    db.executemany(
        "INSERT INTO edge_log VALUES (?,?,?,?,?,?,?,?)",
        [(ts, r["strategy"], r["ticker"], r.get("model_p"),
          r.get("mkt_yes_ask"), r.get("mkt_no_ask"),
          r.get("flagged", ""), r.get("note", "")) for r in rows])
    db.commit()
    db.close()
    return len(rows)


def score(strategy: str | None, days: int, db_path: str = DB_PATH):
    """Brier score of model vs market on logged rows whose markets resolved.

    Market implied prob uses the yes/no ask midpoint ((yes_ask + (100-no_ask))/2)
    — the fair-value estimate a taker faces. One (model, market) pair per
    ticker per scan-ts; resolved outcome joins from the ledger.
    """
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    q = """
      SELECT e.ts, e.strategy, e.ticker, e.model_p, e.mkt_yes_ask, e.mkt_no_ask,
             e.flagged, m.result
      FROM edge_log e JOIN markets m ON m.ticker = e.ticker
      WHERE m.result IN ('yes','no') AND e.model_p IS NOT NULL
        AND e.ts >= datetime('now', ?)
    """
    args = [f"-{days} days"]
    if strategy:
        q += " AND e.strategy = ?"
        args.append(strategy)
    rows = db.execute(q, args).fetchall()
    db.close()
    if not rows:
        print("No resolved logged rows yet — keep scanning with --log and let markets settle.")
        return

    n = len(rows)
    b_model = b_mkt = 0.0
    flagged_pnl_c = 0.0
    n_flagged = 0
    for (_ts, _s, _tk, p, ya, na, flagged, result) in rows:
        y = 1.0 if result == "yes" else 0.0
        if ya is not None and na is not None:
            mkt_p = (ya + (100 - na)) / 200.0
        elif ya is not None:
            mkt_p = ya / 100.0
        else:
            mkt_p = 1 - na / 100.0 if na is not None else 0.5
        b_model += (p - y) ** 2
        b_mkt += (mkt_p - y) ** 2
        if flagged in ("BUY YES", "BUY NO"):
            n_flagged += 1
            from arb_scan import fee_cents
            if flagged == "BUY YES" and ya is not None:
                flagged_pnl_c += (100.0 * y) - ya - fee_cents(ya)
            elif flagged == "BUY NO" and na is not None:
                flagged_pnl_c += (100.0 * (1 - y)) - na - fee_cents(na)

    print(f"Resolved logged rows: {n}")
    print(f"  Brier (model):  {b_model / n:.4f}")
    print(f"  Brier (market): {b_mkt / n:.4f}   "
          f"{'MODEL BEATS MARKET' if b_model < b_mkt else 'market still better — do not trade real money'}")
    if n_flagged:
        print(f"  Flagged-trade paper PnL: {flagged_pnl_c:+.1f}c over {n_flagged} trades "
              f"({flagged_pnl_c / n_flagged:+.2f}c/trade)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    sc = sub.add_parser("score")
    sc.add_argument("--strategy", default=None)
    sc.add_argument("--days", type=int, default=30)
    a = ap.parse_args()
    if a.cmd == "score":
        score(a.strategy, a.days)
    else:
        ap.print_help()
