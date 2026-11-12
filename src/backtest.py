# -*- coding: utf-8 -*-
"""Rolling-origin backtest of the 14-day recursive forecast.
Primary author: Li Baixi (rolling-origin protocol and analysis).


Competitors:
  baseline  weekday-of-week median x festive multiplier (the v0.2 product model)
  gbt       own NumPy histogram GBRT (models/gbt.py), recursive strategy
  sk        scikit-learn GradientBoostingRegressor, same features (benchmark only)

Origins: the last 5 non-overlapping 14-day windows before the data end.
Training data at each origin is strictly pre-origin (expanding window), LOF
cleaning recomputed pre-origin. Metrics pooled over all series x days:
RMSE, MAPE, exact band accuracy, +-1 band accuracy.
Writes report/backtest.json for the report generator.
"""
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from models.features import (Panel, REOPEN, load_panel, logf, holiday_windows,
                             phase_of)
from models.lof import flag_anomalies
from models.gbt import HistGBR
from models.markov import state_of as level_of   # same band definition as the product

CPS8 = ["Lo Wu", "Lok Ma Chau Spur Line", "Lok Ma Chau", "Shenzhen Bay",
        "Man Kam To", "Heung Yuen Wai",
        "Hong Kong-Zhuhai-Macao Bridge", "Express Rail Link West Kowloon"]
N_ORIGINS = 5
HORIZON = 14


def weekday_medians(p, end_ts):
    """Per-series len-7 weekday medians from post-reopening data <= end."""
    out = {}
    for s in p.columns:
        v = p[s][p.index <= end_ts]
        v = v[v.index >= pd.Timestamp(REOPEN)]
        out[s] = v.groupby(v.index.dayofweek).median().reindex(range(7)).fillna(v.median()).values
    return out


def festive_multipliers(p, end_ts):
    """median(flow / weekday base) per (direction, code, phase), pre-origin."""
    wins = holiday_windows()
    rows = []
    wm = weekday_medians(p, end_ts)
    for s in p.columns:
        col = p[s][p.index <= end_ts]
        col = col[col.index >= pd.Timestamp(REOPEN)]
        for ts, v in col.items():
            d = ts.date()
            hit = next(((c, (d - s0).days, ln) for c, s0, s1, ln in wins if s0 <= d <= s1), None)
            if hit:
                code, off, ln = hit
                rows.append((s[1], code, phase_of(off, ln), v / max(wm[s][d.weekday()], 1)))
    df = pd.DataFrame(rows, columns=["dir", "code", "phase", "ratio"])
    g = df.groupby(["dir", "code", "phase"])["ratio"].agg(["median", "count"])
    return {k: (r["median"] if r["count"] >= 10 else 1.0) for k, r in g.iterrows()}


def series_quantiles(p, end_ts):
    out = {}
    for s in p.columns:
        v = p[s][(p.index <= end_ts) & (p.index >= pd.Timestamp(REOPEN))]
        out[s] = {q: float(np.percentile(v, q)) for q in (50, 85, 95)}
    return out


def recursive_forecast(model, panel, p, origin_ts, horizon=HORIZON):
    """model-agnostic recursive forecast: predict one day, feed back as lags."""
    preds = {}
    for s in p.columns:
        hist = logf(p[s][p.index <= origin_ts].values)
        out = []
        d0 = (origin_ts + pd.Timedelta(days=1)).date()
        for h in range(horizon):
            x = panel.row(s, d0 + dt.timedelta(days=h), hist)
            yhat = float(model.predict(x[None, :])[0])
            hist = np.append(hist, yhat)
            out.append(float(np.exp(yhat)))
        preds[s] = out
    return preds


def evaluate(preds, p, test_days, q):
    rows = []
    for s, vals in preds.items():
        actual = p[s].reindex(test_days).values.astype(float)
        for h, (pv, av) in enumerate(zip(vals, actual)):
            if not np.isfinite(av):
                continue
            rows.append({
                "series": str(s), "horizon": h + 1,
                "pred": float(pv), "actual": float(av),
                "level_pred": int(level_of(pv, q[s])),
                "level_true": int(level_of(av, q[s])),
            })
    return rows


def summarise(rows):
    pv = np.array([r["pred"] for r in rows]); av = np.array([r["actual"] for r in rows])
    lp = np.array([r["level_pred"] for r in rows]); lt = np.array([r["level_true"] for r in rows])
    return {
        "rmse": float(np.sqrt(np.mean((pv - av) ** 2))),
        "mape": float(np.mean(np.abs(pv / av - 1))),
        "band_acc": float((lp == lt).mean()),
        "band_acc1": float((np.abs(lp - lt) <= 1).mean()),
        "n": len(rows),
    }


def main():
    p = load_panel("data/raw/immd_daily.csv", keep_cps=CPS8)
    panel = Panel(p, list(p.columns))
    from sklearn.ensemble import GradientBoostingRegressor  # benchmark ONLY

    all_rows = {"baseline": [], "gbt": [], "sk": []}
    for k in range(N_ORIGINS, 0, -1):
        origin_ts = p.index[-1] - pd.Timedelta(days=14 * k)
        end_ts = origin_ts
        test_days = pd.date_range(origin_ts + pd.Timedelta(days=1), periods=HORIZON, freq="D")

        # ---- LOF flags on pre-origin data ----
        wm = weekday_medians(p, end_ts)
        anomalies = {}
        for s in p.columns:
            flow = p[s][p.index <= end_ts].values.astype(float)
            base = np.take(wm[s], np.arange(len(flow)) % 7)
            m, _ = flag_anomalies(flow, base)
            anomalies[s] = np.concatenate([m, np.zeros(len(p) - len(m), bool)])
        X, y, keys = panel.matrices(anomalies)
        dates_key = np.array([kd for _, kd in keys])
        keep = dates_key <= origin_ts.date()
        X, y = X[keep], y[keep]
        ntr = int(len(X) * 0.97)

        # ---- competitors ----
        t0 = time.time()
        ours = HistGBR(n_estimators=400, eval_set=(X[ntr:], y[ntr:]), patience=40, seed=7005)
        ours.fit(X[:ntr], y[:ntr])
        sk = GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                       learning_rate=0.05, subsample=0.8, random_state=7005)
        sk.fit(X, y)
        fmult = festive_multipliers(p, end_ts)
        q = series_quantiles(p, end_ts)
        print("origin %s: train %d rows, ours %.0fs (%d trees)" %
              (origin_ts.date(), len(X), time.time() - t0, ours.best_m_))

        wins = holiday_windows()

        def baseline_pred():
            out = {}
            for s in p.columns:
                vals = []
                for ts in test_days:
                    d = ts.date()
                    b = wm[s][d.weekday()]
                    hit = next(((c, (d - s0).days, ln) for c, s0, s1, ln in wins
                                if s0 <= d <= s1), None)
                    m = fmult.get((s[1], hit[0], phase_of(hit[1], hit[2])), 1.0) if hit else 1.0
                    vals.append(float(b * m))
                out[s] = vals
            return out

        all_rows["baseline"] += evaluate(baseline_pred(), p, test_days, q)
        all_rows["gbt"] += evaluate(recursive_forecast(ours, panel, p, origin_ts), p, test_days, q)
        all_rows["sk"] += evaluate(recursive_forecast(sk, panel, p, origin_ts), p, test_days, q)

    result = {name: summarise(rows) for name, rows in all_rows.items()}
    result["per_series_gbt"] = {}
    for s in p.columns:
        rows = [r for r in all_rows["gbt"] if r["series"] == str(s)]
        result["per_series_gbt"][str(s)] = summarise(rows)
    os.makedirs("report", exist_ok=True)
    with open("report/backtest.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("per_")}, indent=1))
    print("wrote report/backtest.json")


if __name__ == "__main__":
    main()
