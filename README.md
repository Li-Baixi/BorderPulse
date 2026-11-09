# BorderPulse 口岸脉搏

A border-crossing congestion forecaster for Hong Kong — an offline-first PWA that
predicts passenger flow at 8 land border control points (both directions) for the
next 14 days, recommends the best day / port / time window, quantifies queue risk
with an in-browser Monte Carlo simulation, and ranks off-peak attractions by
sentiment.

**Live app (installable PWA):** <https://li-baixi.github.io/BorderPulse/>

Built from Hong Kong open data only. All core algorithms are hand-written
NumPy / vanilla JS — gradient-boosted trees, Local Outlier Factor, Markov chains,
Erlang-C + Monte Carlo, TF-IDF + logistic regression. scikit-learn is used
**only** as an external benchmark, never in the shipped pipeline.

## Features

| Tab | What it does |
|---|---|
| **日历 Calendar** | 14-day × 8-port congestion heatmap (self-calibrated quartile bands). Tap a day: expected flow with P10–P90 band, P(busy), Markov state probabilities, latest published actual, and an hourly strip with the best 2-hour window to cross. |
| **对比 Compare** | All ports side by side for a chosen date: weekday base, P85/P95, 60-day sparkline, ×-multiple vs the busiest port, and a switch recommendation. |
| **风险 Risk** | Monte Carlo queue simulation (3,000 trials) running live in the browser: Erlang-C waits below saturation, capped triangular dwell above; outputs P(wait > 30 min), median and P95. |
| **景点 Attractions** | Off-peak alternatives ranked by TF-IDF + logistic-regression sentiment trained on 161k real TripAdvisor reviews, with per-attraction top terms (negation-aware) and crowd-level tips. |

Everything works **offline** after first load — service worker caches the shell
and the data bundle, so the app is usable at the border with no signal.

## Pipeline

```
ImmD daily CSV ──► series filter (≥45 days, median ≥50/day)
                    │
                    ├─► LOF anomaly cleaning (k=20, two-gate: LOF>1.3 ∧ top-1.5%)
                    │     └─ independently catches Saola 2023-09-01/02,
                    │        record black rainstorm 2023-09-08, Yagi 2024-09-06
                    ▼
              HistGBR (own NumPy, 400 trees, 56 features, 21.5k rows)
              log-flow target, early stopping, 14-day recursive forecast
                    │
                    ├─► Markov chain over 4 congestion bands (P50/P85/P95 cuts,
                    │     Laplace-smoothed, started from today's observed state)
                    ├─► daily forecast × port intraday template → Erlang-C / MC
                    │     queue risk per hour
                    └─► docs/data.json  ──►  PWA
                                          │
TripAdvisor corpus ──► TF-IDF + class-balanced logistic regression ──► docs/nlp.json
```

### Rolling-origin backtest (n = 1,120, five 14-day windows, expanding training)

| Model | RMSE | MAPE | band acc. | ±1 band |
|---|---|---|---|---|
| Weekday-median baseline | 12,743 | 30.9% | 32.8% | 86.1% |
| **HistGBR (this repo, own NumPy)** | **8,764** | **19.3%** | **64.3%** | **95.2%** |
| scikit-learn HistGradientBoosting (benchmark) | 13,209 | 24.0% | 56.4% | 90.6% |

The own model beats both the baseline (−31% RMSE) and the sklearn benchmark:
early stopping and quantile-binned histogram splits regularise the recursive
14-day rollout, where small biases compound.

## Data sources

| Dataset | Role |
|---|---|
| [ImmD — Daily Passenger Traffic](https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic) | core time series, 2021-01 → present, per control point × direction × resident type |
| [ImmD — Festive Period Traffic](https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-passenger-traffic-festive-period) | festive-calendar cross-check |
| [C&ED — Control Point Throughput](https://data.gov.hk/en-data/dataset/hk-customs-ced_stat-passenger-clearance) | annual capacity anchor (sanity check) |
| Kaggle — TripAdvisor reviews (Singapore attractions) | sentiment training corpus (161,442 train / 17,939 test) |

The in-app Hong Kong attraction corpus is a **clearly labelled synthetic demo
corpus** (course-spec showcase of generation-based good-to-haves); the sentiment
model itself is trained exclusively on the real labelled corpus above. Details in
`data/SOURCES.md`.

## Repository layout

```
docs/               PWA static site (GitHub Pages root; no framework, no build step)
  app.js, index.html, styles.css, sw.js, data.json, nlp.json, icons/
src/
  build_docs.py     main pipeline: clean → calibrate → forecast → docs/data.json
  backtest.py       rolling-origin evaluation vs baseline + sklearn benchmark
  make_figures.py   report figure set (matplotlib)
  make_icons.py     PWA icon generation
  models/
    features.py     panel loading, 56-feature calendar/lag engineering
    lof.py          Local Outlier Factor (own NumPy) + two-gate flagging
    gbt.py          histogram gradient-boosted trees (own NumPy)
    markov.py       4-band transition matrix, h-step distributions
    mc_queue.py     Erlang-C + capped Monte Carlo queue risk
    nlp.py          TF-IDF + class-balanced logistic regression sentiment
data/
  SOURCES.md        verified source catalogue
  raw/immd_daily.csv
```

## Reproduce

Requires Python 3.10+ with `numpy`, `pandas`, `matplotlib` (figures),
`pillow` (icons), `scikit-learn` (benchmarks only).

```bash
python src/build_docs.py        # rebuild docs/data.json from the CSV
python src/backtest.py          # rolling-origin backtest → report/backtest.json
python src/make_figures.py      # figure set → report/figs/
python src/models/nlp.py        # sentiment model → docs/nlp.json
                                #   (needs the Kaggle TripAdvisor CSV at
                                #    data/raw/sg_tripadvisor.csv — not
                                #    redistributed here for licensing reasons)
python -m http.server -d docs   # serve the app locally
```

## Limitations & disclaimer

- Hourly curves are port-specific intraday **templates** scaled by the daily
  forecast — ImmD publishes no official hourly series.
- Queue constants (service rate, peak share, counter count) are disclosed,
  calibrated assumptions, not fitted to a queue-length feed.
- Anomalies are cleaned from training, not forecast: a typhoon next Tuesday is
  out of scope by design.
- Course project (MATH7005, data-product design). Forecasts are informational,
  not official ImmD figures.
