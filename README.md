# Prediction Market Microstructure & Probabilistic Calibration

Research platform for retail prediction markets (Kalshi), built around one discipline:
**no model trades real money until it demonstrably beats the market's own probabilities
on a logged, out-of-sample record.**

Three pillars:

1. **Data asset** — a continuously-running snapshotter builds a tick-history of quotes and
   settled outcomes across every real Kalshi market (~7.6k events / ~65k markets), producing a
   labeled `(price, time-to-close, category) → outcome` dataset for calibration research.
2. **Probabilistic forecasting** — a weather model converts multi-model ensemble forecasts
   (122 members across GFS/ECMWF/ICON) into bracket probabilities for Kalshi's daily
   high-temperature markets, with per-station bias correction and skill weighting fitted
   against the official climate record.
3. **Strategy + honest evaluation** — a fee-aware basket-arbitrage scanner and an edge
   journal that scores every model claim against what actually settled (Brier score vs.
   the market) before any capital is risked.

## Architecture

| Component | What it does |
|---|---|
| [`snapshot.py`](snapshot.py) | Daemon: 2h full-universe sweeps + 5min hot-tier re-quoting (~4k markets near close/active) + settlement capture, into SQLite (`data/ledger.db`). Change-deduped rows; rate-limit aware. |
| [`kalshi.py`](kalshi.py) | Public market-data client (correct host, orderbook→bid/ask, exact `ceil`-rounded fee formula). |
| [`weather.py`](weather.py) | Open-Meteo ensemble client → per-member daily-high distributions, per model; NWS station running-max (same-day floor). |
| [`calibrate.py`](calibrate.py) | Fits per-station, per-model bias + MAE from day-ahead forecast errors vs. the official ACIS climate record, with a walk-forward out-of-sample check. |
| [`scan_weather.py`](scan_weather.py) | Bracket probabilities (weighted KDE over bias-corrected members + observed floor) vs. live market prices, fee-adjusted, journaled. |
| [`arb_scan.py`](arb_scan.py) | Mutually-exclusive basket arbitrage: risk-free NO-baskets ranked first; YES-baskets auto-screened by market-implied "other" probability (non-exhaustiveness traps). |
| [`journal.py`](journal.py) | The gate: logs every scan's full probability surface; `score` computes model-vs-market Brier + flagged-trade paper PnL on resolved markets. |

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python snapshot.py run &        # start the data spine (long-running)
.venv/bin/python calibrate.py NY          # fit weather calibration (~60d history)
.venv/bin/python scan_weather.py NY --log # bracket probabilities vs market, journaled
.venv/bin/python arb_scan.py              # risk-free basket scan (ledger + live verify)
.venv/bin/python journal.py score         # model vs market on settled rows
.venv/bin/python -m pytest tests/ -q      # 21 tests, no network
```

## Day-one findings (kept honest)

- **The market beat the raw model.** On 2026-06-09 the raw pooled ensemble put 39% on the
  82–83°F bracket (market: 4%) and 22% on the winner. Observed high: **79°F** — the bracket
  the market had at ~50%. Verdict: market 1, raw model 0.
- **Why: model skill is station-specific.** Calibration against the official Central Park
  record shows ICON at **1.49°F** walk-forward MAE vs ~2.17 for GFS/ECMWF — the market had
  already internalized what the naive pooled ensemble ignores. The skill-weighted,
  bias-corrected blend (1.54°F) roughly matches the best single model out-of-sample;
  its value is robustness and distribution shape, not point accuracy.
- **Risk-free arbitrage is mostly competed flat.** A full scan of 3,235 mutually-exclusive
  events found zero NO-baskets clearing fees; 27 seductive YES-baskets were auto-rejected
  because their own prices implied a >10% unlisted outcome (e.g. jungle-primary "nominee"
  markets where no listed candidate is likely to qualify).
- **Fees dominate small edges.** Kalshi's taker fee `ceil(0.07·P·(1−P)·100)¢` rounds *up*:
  a 5¢ contract pays a 1¢ fee — a 20% tax. Sub-15¢ "edges" are usually fee illusions.

## Methodology notes

- **Settlement is a proxy, not the event.** Contracts pay on a named source's report
  (NWS climate report, AP/league feeds), and disputed markets can settle off the rulebook's
  edge cases — basket "min profit" is risk-free w.r.t. outcomes, not w.r.t. settlement rules.
- **Bias correction is measured on the deliverable.** Day-ahead forecast errors are computed
  against the same official record the contracts settle on, so the fitted bias absorbs model
  bias, grid-vs-station representativeness, and hourly-sampling error in one term.
- **Known gap:** calibration is fitted at day-ahead lead but also applied same-day, where true
  uncertainty is tighter — lead-time-specific calibration is the next refinement.

## Status / roadmap

- [x] Data spine, weather calibration v1, basket scanner, edge journal, tests
- [ ] 2+ weeks of journal data → first model-vs-market Brier verdict
- [ ] Lead-time-specific calibration + intraday METAR assimilation
- [ ] Calibration study across categories (favorite-longshot / underconfidence)
- [ ] Cross-venue divergence half-life measurement (Polymarket US)

## Data sources & terms

[Open-Meteo](https://open-meteo.com/) (CC-BY-4.0; non-commercial tier) · 
[ACIS / NOAA RCCs](https://www.rcc-acis.org/) · [NWS API](https://api.weather.gov/) · 
Kalshi public market-data API. Personal research; not financial advice. Trading involves
risk of loss; respect each venue's terms of service.
