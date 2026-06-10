"""No-network unit tests for the pure math the strategies depend on.

The fee formulas, bracket parsing, basket arithmetic, and scoring logic are
where a silent sign/rounding error costs real money — they get exact tests.
"""
import math
import os
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arb_scan import Basket, Leg, fee_cents
from calibrate import fit
from kalshi import taker_fee, _levels
from scan_weather import bracket_bounds, date_from_ticker, kde_bracket_prob
from snapshot import cents, fnum


# --- Kalshi fee math (ceil-to-next-cent is the part people get wrong) -------

@pytest.mark.parametrize("price_c,expected_c", [
    (50, 2),   # 7*0.5*0.5 = 1.75c -> ceil 2c (the maximum)
    (20, 2),   # 1.12c -> 2c
    (10, 1),   # 0.63c -> 1c
    (5, 1),    # 0.3325c -> 1c : a 20% tax on a 5c contract
    (95, 1),   # symmetric tail
    (1, 1),    # even 1c contracts pay 1c fee
    (0, 0), (100, 0),
])
def test_fee_cents(price_c, expected_c):
    assert fee_cents(price_c) == expected_c


def test_taker_fee_dollars_matches_cents():
    for c in range(1, 100):
        assert round(taker_fee(c / 100) * 100) == fee_cents(c)


# --- snapshot field parsing ---------------------------------------------------

def test_cents_and_fnum():
    assert cents("0.3200") == 32
    assert cents("1.0000") == 100
    assert cents("") is None and cents(None) is None
    assert fnum("10780.64") == 10780.64
    assert fnum(None) is None


def test_orderbook_levels_both_formats():
    assert _levels([["0.0100", "3607.60"]]) == [(0.01, 3607.60)]
    assert _levels([[55, 10]]) == [(0.55, 10.0)]  # integer-cent format
    assert _levels(None) == []


# --- weather bracket parsing --------------------------------------------------

@pytest.mark.parametrize("sub,expected", [
    ("84° to 85°", (83.5, 85.5)),
    ("86° or above", (85.5, math.inf)),
    ("77° or below", (-math.inf, 77.5)),
])
def test_bracket_bounds(sub, expected):
    assert bracket_bounds(sub) == expected


def test_date_from_ticker():
    assert date_from_ticker("KXHIGHNY-26JUN09-B82.5") == "2026-06-09"
    assert date_from_ticker("NODATE") is None


def test_kde_partition_sums_to_one():
    highs = np.array([78.0, 79.0, 80.0, 81.0, 82.0])
    brackets = [(-math.inf, 77.5), (77.5, 79.5), (79.5, 81.5), (81.5, 83.5), (83.5, math.inf)]
    total = sum(kde_bracket_prob(highs, lo, hi, bw=1.0) for lo, hi in brackets)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_kde_weights_shift_mass():
    highs = np.array([70.0, 90.0])
    hot = kde_bracket_prob(highs, 85.0, math.inf, bw=0.5, weights=np.array([1.0, 9.0]))
    assert hot == pytest.approx(0.9, abs=0.01)


# --- basket arb math ----------------------------------------------------------

def _basket(side, asks):
    legs = [Leg(f"T{i}", "", a, a) for i, a in enumerate(asks)]
    return Basket("EV", "t", "c", side, legs)


def test_no_basket_min_profit():
    # 3 legs, NO asks 60/70/80 => payout floor 200, cost 210, fees 2+2+2=6 -> -16
    assert _basket("NO", [60, 70, 80]).profit == 200 - 210 - 6
    # profitable: NO asks 90/95 on 2 legs => payout 100, cost 185, fees 1+1 -> wait,
    # fee(90)=1, fee(95)=1 => 100-185-2 = -87 (2-leg NO baskets need cost<98)
    assert _basket("NO", [90, 95]).profit == 100 - 185 - 2
    b = _basket("NO", [49, 48])
    assert b.profit == 100 - 97 - 4  # fee(49)=fee(48)=2c: 'cheap' 2-leg baskets die on fees


def test_yes_basket_and_implied_other():
    b = _basket("YES", [2, 2])
    assert b.profit == 100 - 4 - 2
    assert b.implied_other == 96.0  # the market screams non-exhaustive
    assert _basket("NO", [50, 50]).implied_other is None


# --- calibration math ---------------------------------------------------------

def test_fit_bias_and_mae():
    stats = fit({"m": [(82.0, 80.0), (78.0, 80.0), (84.0, 80.0)]})["m"]
    assert stats["bias"] == pytest.approx(round((2 - 2 + 4) / 3, 2))
    assert stats["mae"] == pytest.approx(round((2 + 2 + 4) / 3, 2))
    assert stats["n"] == 3


# --- journal scoring (temp DB round-trip) --------------------------------------

def test_journal_score_brier(tmp_path, capsys):
    import journal
    db_path = str(tmp_path / "t.db")
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE markets (ticker TEXT PRIMARY KEY, result TEXT)")
    db.execute("INSERT INTO markets VALUES ('A','yes'), ('B','no')")
    db.commit(); db.close()
    journal.log_rows([
        {"strategy": "weather", "ticker": "A", "model_p": 0.8,
         "mkt_yes_ask": 60, "mkt_no_ask": 42, "flagged": "BUY YES"},
        {"strategy": "weather", "ticker": "B", "model_p": 0.1,
         "mkt_yes_ask": 30, "mkt_no_ask": 72, "flagged": ""},
    ], db_path=db_path)
    journal.score(None, 30, db_path=db_path)
    out = capsys.readouterr().out
    # model Brier: ((0.8-1)^2 + (0.1-0)^2)/2 = 0.025
    # market: A mid=(60+58)/200=0.59 -> 0.168; B mid=(30+28)/200=0.29 -> 0.0841/2-> total 0.1261
    assert "Brier (model):  0.0250" in out
    assert "MODEL BEATS MARKET" in out
    # flagged BUY YES on A at 60c: payout 100 - 60 - fee(60)=2 -> +38c
    assert "+38.0c" in out
