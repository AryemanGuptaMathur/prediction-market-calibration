# Prediction market calibration research

This is my research platform for trading prediction markets, mainly Kalshi for now.
It exists because of a rule I keep having to relearn: the market is sharper than my
models until proven otherwise, and "proven" means out of sample, on settled outcomes,
not on backtests I can fool myself with.

Prediction markets quote probabilities. If you can produce better-calibrated
probabilities than the market in some niche, you have an edge. Plenty of people
believe their model; very few measure whether its probabilities actually beat the
prices. So the centerpiece of this repo is not a trading bot. It is a journal that
records every probability my models emit next to the market price at that moment,
waits for markets to settle, and scores both sides. The standing rule: no real money
until the journal shows the model beating the market on Brier score over at least
two weeks of resolved markets. As of day one, it does not.

## What's in here

- `snapshot.py` collects the data. A daemon sweeps the full Kalshi universe every two
  hours (about 7,600 events and 65,000 markets) and requotes the ~4,000 markets that
  are near close or actively trading every five minutes, into SQLite. Settlements get
  recorded as they finalize, which is what turns price history into labeled training
  data. Rows are only written when a quote actually changes.
- `kalshi.py` is a small public-data client. The working API host is
  `api.elections.kalshi.com` (the documented `api.kalshi.com` does not resolve), and
  the fee formula is `ceil(0.07 * P * (1-P) * 100)` cents, rounded up. That rounding
  matters more than it looks: a 5 cent contract pays a 1 cent fee, a 20% tax.
- `weather.py` and `calibrate.py` are the forecasting side. Kalshi's daily
  high-temperature markets settle on the NWS climate report for one specific station,
  so the model pulls free ensemble forecasts (122 members across GFS, ECMWF, and ICON
  via Open-Meteo) and calibrates them per station against the official ACIS climate
  record: per-model bias and error measured on exactly the quantity being predicted.
- `scan_weather.py` turns calibrated members into bracket probabilities (weighted
  kernel density, skill-weighted by model, clamped to the station's observed running
  max on same-day scans) and compares them to live market prices, fee-adjusted.
- `arb_scan.py` looks for risk-free baskets on mutually-exclusive events.
- `journal.py` is the gate described above. `score` joins logged predictions to
  settled outcomes and prints the verdict.
- `tests/` covers the math that would lose money silently if wrong: fees, bracket
  parsing, basket arithmetic, scoring. 21 tests, no network.

## What day one taught me

I scanned the NYC high-temperature market on June 9, 2026 with the naive version of
the model (pool all ensemble members, read off bracket probabilities). It put 39% on
the 82-83°F bracket. The market priced that bracket at 4 cents and split its money
between 78-79 and 80-81. The observed high was 79°F. The market had roughly 50% on
the winning bracket; my model had 22%, and its favorite bracket lost.

The post-mortem was more useful than a win would have been. The three weather models
flatly disagreed that day (GFS and ECMWF said 82-83, ICON said 78), and pooling them
produced a confident-looking blend of two opinions, one of which was wrong. Fitting
each model against the official Central Park record explained the rest: ICON's
day-ahead error at that station is 1.49°F walk-forward MAE, versus roughly 2.17°F for
GFS and ECMWF. The market already knew which model to trust. My calibration learned
it one day late, which is exactly the kind of lesson the journal is designed to
catch before it costs anything.

The arbitrage side got its own reality check. A full scan of 3,235 mutually-exclusive
events found zero risk-free NO-baskets clearing fees. It did find 27 YES-baskets
showing 80 to 94 cents of apparent profit, every one of which was a trap: markets
like Louisiana primary "nominee" contracts where the listed candidates sum to a few
cents because the likely outcome is that no listed candidate qualifies. The prices
themselves tell you this (100 minus the sum of asks is the market's estimate of the
unlisted outcome), so the scanner now computes that and suppresses the traps instead
of reporting them as opportunities.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python snapshot.py run &         # data daemon, leave running
.venv/bin/python calibrate.py NY           # fit weather calibration (~60 days)
.venv/bin/python scan_weather.py NY --log  # bracket probabilities vs market, journaled
.venv/bin/python arb_scan.py               # basket scan, ledger + live verification
.venv/bin/python journal.py score          # model vs market on settled rows
.venv/bin/python -m pytest tests/ -q
```

## Things I do not trust yet

- The calibration is fitted on day-ahead forecast errors but the scanner also runs
  same-day, where true uncertainty is tighter. Until calibration is lead-time
  specific, same-day disagreements with the market are suspect, and the model's
  large flagged "edges" should be read as model warnings.
- Settlement pays on what a named source reports, not on what happened. Disputed
  Kalshi markets have settled at last-traded price under rule 6.3(c). Basket profits
  are risk-free with respect to outcomes, not with respect to settlement rules.
- Displayed prices are not fillable size. The scanner walks order books for its top
  candidates, but thin books move when touched.
- 51 evaluation days of weather calibration is a small sample. The model-vs-market
  question gets answered by the journal, slowly, the only way it can be.

## Roadmap

- [x] Data spine, weather calibration v1, basket scanner, edge journal, tests
- [ ] Two weeks of journal data, then the first model-vs-market verdict
- [ ] Lead-time-specific calibration and intraday observation assimilation
- [ ] Calibration study across market categories from the accumulated ledger
- [ ] Cross-venue divergence measurement against Polymarket US

## Data sources

[Open-Meteo](https://open-meteo.com/) (CC-BY-4.0, non-commercial tier),
[ACIS](https://www.rcc-acis.org/) for official station climate records,
the [NWS API](https://api.weather.gov/) for live observations, and Kalshi's public
market-data API. Personal research, not financial advice. Trading risks loss.
