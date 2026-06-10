#!/usr/bin/env python3
"""Within-Kalshi basket arbitrage scanner over mutually-exclusive events.

For an event whose markets are mutually exclusive (Kalshi's `mutually_exclusive`
flag, captured in the ledger), with n open legs:

  NO-basket:  buy 1 NO on every leg at ask q_i. At most one leg can resolve YES,
              so at least n-1 NOs pay $1. If no listed leg wins (partition not
              exhaustive) all n pay — strictly better. Hence
                  min_profit = (n-1)*100 - sum(q_i + fee(q_i))   [cents/basket]
              is RISK-FREE w.r.t. outcomes (execution/settlement risk remains).

  YES-basket: buy 1 YES on every leg at ask p_i. Pays $1 only if some listed
              leg wins — requires the partition to be EXHAUSTIVE, which the API
              flag does not guarantee. Flagged separately; verify the rulebook
              before trading these.

Events with any already-resolved-YES leg are excluded (the remaining legs are
then certain NOs — a different trade). Resolved-NO legs simply shrink n.

Pipeline: scan ledger (fast, possibly stale) -> live re-verify candidates via
batched /markets -> fetch orderbook depth for survivors -> report.

Usage:
  python arb_scan.py [--min-edge 0.5] [--top 10] [--no-live] [--json]
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass, field

from snapshot import DB_PATH, Api, cents


def fee_cents(price_cents: int | None, multiplier: float = 0.07) -> int:
    """Kalshi taker fee in cents for one contract at the given price (cents).

    ceil(multiplier * P * (1-P) * 100) with P in dollars — rounded UP per cent.
    """
    if not price_cents or price_cents <= 0 or price_cents >= 100:
        return 0
    p = price_cents / 100.0
    return math.ceil(multiplier * p * (1 - p) * 100)


@dataclass
class Leg:
    ticker: str
    subtitle: str
    yes_ask: int | None
    no_ask: int | None
    age_s: float | None = None
    no_ask_size: float | None = None
    yes_ask_size: float | None = None


@dataclass
class Basket:
    event_ticker: str
    title: str
    category: str
    side: str                    # "NO" or "YES"
    legs: list[Leg] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.legs)

    @property
    def cost(self) -> int:
        return sum((l.no_ask if self.side == "NO" else l.yes_ask) or 0 for l in self.legs)

    @property
    def fees(self) -> int:
        return sum(fee_cents(l.no_ask if self.side == "NO" else l.yes_ask) for l in self.legs)

    @property
    def profit(self) -> float:
        """NO: guaranteed min profit. YES: profit IF the partition is exhaustive."""
        payout = (self.n - 1) * 100 if self.side == "NO" else 100
        return payout - self.cost - self.fees

    @property
    def max_age_s(self) -> float | None:
        ages = [l.age_s for l in self.legs if l.age_s is not None]
        return max(ages) if ages else None

    @property
    def implied_other(self) -> float | None:
        """YES-baskets only: market-implied P(no listed leg wins), in percent.

        On an exhaustive partition the YES asks sum to ~100c, so a large
        (100 - sum) means the market believes the listed legs are NOT
        exhaustive — the apparent 'arb' is a trap, not free money.
        """
        if self.side != "YES":
            return None
        return max(0.0, 100.0 - self.cost)


def scan_ledger(db_path: str = DB_PATH) -> list[Basket]:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = db.execute("""
        WITH latest AS (SELECT ticker, MAX(ts) mts FROM snapshots GROUP BY ticker),
        q AS (SELECT s.ticker, s.yes_ask, s.no_ask,
                     (julianday('now') - julianday(s.ts)) * 86400.0 AS age_s
              FROM snapshots s JOIN latest l ON s.ticker = l.ticker AND s.ts = l.mts)
        SELECT e.event_ticker, e.title, e.category,
               m.ticker, COALESCE(m.yes_sub_title, ''),
               q.yes_ask, q.no_ask, q.age_s,
               m.result, m.status
        FROM events e
        JOIN markets m USING (event_ticker)
        LEFT JOIN q ON q.ticker = m.ticker
        WHERE e.mutually_exclusive = 1
        ORDER BY e.event_ticker""").fetchall()
    db.close()

    by_event: dict[str, dict] = {}
    for (ev, title, cat, tk, sub, ya, na, age, result, status) in rows:
        d = by_event.setdefault(ev, {"title": title, "cat": cat, "legs": [],
                                     "yes_resolved": False, "bad": False})
        if result == "yes":
            d["yes_resolved"] = True
        elif result == "no":
            continue  # eliminated leg: shrink the basket
        elif status in ("active", "open"):
            if na is None and ya is None:
                d["bad"] = True  # an unquoted open leg breaks the basket
            else:
                d["legs"].append(Leg(tk, sub, ya, na, age))
        else:
            d["bad"] = True  # paused/initialized leg: can't execute the full basket

    out: list[Basket] = []
    for ev, d in by_event.items():
        if d["yes_resolved"] or d["bad"] or len(d["legs"]) < 2:
            continue
        if all(l.no_ask not in (None, 0) for l in d["legs"]):
            out.append(Basket(ev, d["title"], d["cat"], "NO", d["legs"]))
        if all(l.yes_ask not in (None, 0) for l in d["legs"]):
            out.append(Basket(ev, d["title"], d["cat"], "YES", d["legs"]))
    return out


def live_verify(api: Api, baskets: list[Basket]) -> list[Basket]:
    """Re-price every leg from the live API; drop baskets that no longer clear."""
    tickers = sorted({l.ticker for b in baskets for l in b.legs})
    fresh: dict[str, dict] = {}
    for i in range(0, len(tickers), 50):
        d = api.get("/markets", {"tickers": ",".join(tickers[i:i + 50]), "limit": 50})
        for m in d.get("markets", []):
            fresh[m["ticker"]] = m
    out = []
    for b in baskets:
        ok = True
        for l in b.legs:
            m = fresh.get(l.ticker)
            if not m or m.get("result") or m.get("status") not in ("active", "open"):
                ok = False
                break
            l.yes_ask, l.no_ask, l.age_s = cents(m.get("yes_ask_dollars")), cents(m.get("no_ask_dollars")), 0.0
            ask = l.no_ask if b.side == "NO" else l.yes_ask
            if ask in (None, 0):
                ok = False
                break
        if ok and b.profit > 0:
            out.append(b)
    return out


def add_depth(b: Basket) -> float:
    """Fetch orderbooks and return executable basket count (min depth across legs)."""
    from kalshi import Kalshi
    k = Kalshi()
    depths = []
    for l in b.legs:
        q = k.orderbook(l.ticker)
        l.no_ask_size, l.yes_ask_size = q.no_ask_size, q.yes_ask_size
        depths.append((q.no_ask_size if b.side == "NO" else q.yes_ask_size) or 0)
    return min(depths) if depths else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-edge", type=float, default=0.5, help="min profit in cents/basket (default 0.5)")
    ap.add_argument("--top", type=int, default=10, help="max candidates to live-verify/report")
    ap.add_argument("--no-live", action="store_true", help="skip live re-verification (ledger only)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    baskets = scan_ledger()
    # Risk-free NO-baskets first; YES-baskets only if the market itself prices
    # the partition as near-exhaustive (small implied 'other'), else they are
    # almost always non-exhaustiveness traps, not arbs.
    no_side = [b for b in baskets if b.side == "NO" and b.profit >= a.min_edge]
    yes_side = [b for b in baskets if b.side == "YES" and b.profit >= a.min_edge]
    yes_clean = [b for b in yes_side if (b.implied_other or 0) <= 10.0]
    yes_traps = len(yes_side) - len(yes_clean)
    cands = (sorted(no_side, key=lambda b: -b.profit)
             + sorted(yes_clean, key=lambda b: -b.profit))[:a.top]
    scanned = len({b.event_ticker for b in baskets})
    if not a.no_live and cands:
        cands = live_verify(Api(), cands)
        cands.sort(key=lambda b: (b.side != "NO", -b.profit))
        for b in cands[:5]:
            b.depth = add_depth(b)

    if a.json:
        print(json.dumps([{
            "event": b.event_ticker, "title": b.title, "side": b.side, "legs": b.n,
            "cost_c": b.cost, "fees_c": b.fees, "profit_c": round(b.profit, 1),
            "depth": getattr(b, "depth", None),
            "exhaustiveness_required": b.side == "YES",
        } for b in cands], indent=1))
        return

    print(f"Scanned {scanned} mutually-exclusive events "
          f"({len(baskets)} candidate baskets, {'ledger+live' if not a.no_live else 'ledger only'})")
    if yes_traps:
        print(f"Suppressed {yes_traps} YES-baskets whose own prices imply a >10% 'other' outcome "
              f"(non-exhaustive partitions, not arbs).")
    if not cands:
        print(f"No risk-free baskets clear {a.min_edge}c after fees. "
              f"Books are tight today — that's the honest result.")
        return
    for b in cands:
        depth = getattr(b, "depth", None)
        print(f"\n{'='*78}\n{b.side}-basket  {b.event_ticker}  [{b.category}]  {b.title}")
        print(f"  legs={b.n}  cost={b.cost}c  fees={b.fees}c  "
              f"min_profit={b.profit:.1f}c/basket"
              + (f"  executable~{depth:.0f} baskets (${depth*b.profit/100:.2f} total)" if depth else ""))
        if b.side == "YES":
            print(f"  ⚠️  YES-basket: pays only if a LISTED leg wins "
                  f"(market-implied 'other' ≈ {b.implied_other:.0f}% — verify the rulebook covers it).")
        for l in b.legs:
            ask = l.no_ask if b.side == "NO" else l.yes_ask
            sz = l.no_ask_size if b.side == "NO" else l.yes_ask_size
            print(f"    {l.ticker:<38} {b.side} ask={ask:>3}c"
                  + (f" size={sz:.0f}" if sz is not None else ""))


if __name__ == "__main__":
    main()
