"""Minimal read-only Kalshi public-market-data client.

No auth needed for market data. Trading would require RSA-signed requests
(api key id + private key) — deliberately out of scope here; this module is
for *discovering and pricing* edges, not executing them.

Verified host: api.elections.kalshi.com  (api.kalshi.com does NOT resolve)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def taker_fee(price: float, multiplier: float = 0.07) -> float:
    """Kalshi taker fee per contract, in dollars.

    fee = ceil(multiplier * P * (1 - P) * 100) / 100   (rounded UP to next cent)
    Default multiplier 0.07 covers most categories; some (e.g. sports) are higher.
    """
    if price is None or price <= 0 or price >= 1:
        return 0.0
    return math.ceil(multiplier * price * (1 - price) * 100) / 100


@dataclass
class Quote:
    ticker: str
    subtitle: str
    yes_bid: float | None   # best price someone will pay for YES
    yes_ask: float | None   # cost to BUY one YES contract now
    no_bid: float | None    # best price someone will pay for NO
    no_ask: float | None    # cost to BUY one NO contract now
    yes_ask_size: float | None  # contracts available at the YES ask
    no_ask_size: float | None   # contracts available at the NO ask


class Kalshi:
    def __init__(self, base: str = BASE, timeout: int = 20):
        self.base = base
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update(_HEADERS)

    def _get(self, path: str, **params):
        r = self.s.get(f"{self.base}{path}", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def markets(self, series_ticker: str, status: str = "open", limit: int = 100):
        """All markets (brackets) in a series, e.g. KXHIGHNY-26JUN09 or KXHIGHNY."""
        data = self._get("/markets", series_ticker=series_ticker, status=status, limit=limit)
        return data.get("markets", [])

    def orderbook(self, ticker: str) -> Quote:
        """Best bid/ask on both sides, derived from the resting-order ladders.

        Kalshi books are one-sided-per-contract: a resting NO bid at price q is
        equivalent to a YES offer at (1 - q). So:
            yes_ask = 1 - best_no_bid   (lift cheapest YES = best NO bid)
            no_ask  = 1 - best_yes_bid
        """
        data = self._get(f"/markets/{ticker}/orderbook")
        ob = data.get("orderbook_fp") or data.get("orderbook") or {}
        yes = _levels(ob.get("yes") or ob.get("yes_dollars"))
        no = _levels(ob.get("no") or ob.get("no_dollars"))

        yes_bid = max((p for p, _ in yes), default=None)
        no_bid = max((p for p, _ in no), default=None)
        yes_ask = round(1 - no_bid, 2) if no_bid is not None else None
        no_ask = round(1 - yes_bid, 2) if yes_bid is not None else None
        # size available at the best ask = size resting at the matching opposite bid
        yes_ask_size = next((sz for p, sz in no if no_bid is not None and abs(p - no_bid) < 1e-9), None)
        no_ask_size = next((sz for p, sz in yes if yes_bid is not None and abs(p - yes_bid) < 1e-9), None)
        return Quote(ticker, "", yes_bid, yes_ask, no_bid, no_ask, yes_ask_size, no_ask_size)


def _levels(raw) -> list[tuple[float, float]]:
    """Normalize a ladder into [(price_dollars, size), ...]. Handles dollar-string
    and integer-cent formats."""
    out: list[tuple[float, float]] = []
    if not raw:
        return out
    for entry in raw:
        try:
            price, size = entry[0], entry[1]
            price = float(price)
            if price > 1.5:  # integer-cent format (e.g. 55 -> 0.55)
                price /= 100.0
            out.append((round(price, 4), float(size)))
        except (TypeError, ValueError, IndexError):
            continue
    return out
