# -*- coding: utf-8 -*-
"""BorderPulse docs builder — v0.3 model-driven pipeline.

Reads data/raw/immd_daily.csv and writes docs/data.json for the static PWA:

  1. panel      8 control points x 2 directions, post-reopening focus
  2. LOF        per-series anomaly flags (own implementation) -> training filter
  3. GBT        hand-written histogram gradient-boosted trees, pooled over all
                series, recursive 14-day forecast starting TODAY
  4. Markov     4-band state transition chain per series -> per-day state
                distribution as a second, probabilistic opinion
  5. MC queue   runs client-side in the PWA's live risk tab (same model as
                models/mc_queue.py, which the report figures use)
  6. NLP        attraction sentiment page (docs/nlp.json produced separately by
                models/nlp.py) is merged in here so the app fetches one file

Run:  python src/build_docs.py          (models/nlp.py first if reviews moved)
"""
import datetime as dt
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.features import Panel, REOPEN, FEATURE_NAMES, load_panel, logf
from models.lof import flag_anomalies
from models.gbt import HistGBR
from models.markov import fit_transition, forecast_dist, state_of

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "immd_daily.csv")
OUT = os.path.join(ROOT, "docs", "data.json")
NLP = os.path.join(ROOT, "docs", "nlp.json")
HORIZON = 14

CPS = {
    "Lo Wu":                       {"cn": "罗湖",           "short": "LoWu"},
    "Lok Ma Chau Spur Line":       {"cn": "落马洲支线",      "short": "LMC-Spur"},
    "Lok Ma Chau":                 {"cn": "落马洲",          "short": "LMC"},
    "Shenzhen Bay":                {"cn": "深圳湾",          "short": "SZB"},
    "Man Kam To":                  {"cn": "文锦渡",          "short": "MKT"},
    "Heung Yuen Wai":              {"cn": "香园围",          "short": "HYW"},
    "Hong Kong-Zhuhai-Macao Bridge": {"cn": "港珠澳大桥",    "short": "HZMB"},
    "Express Rail Link West Kowloon": {"cn": "高铁西九龙",   "short": "ERL"},
}

# --- intraday model (unchanged from v0.2; see report Methods) ----------------
PROFILE_OF = {
    "Lo Wu": "rail", "Lok Ma Chau Spur Line": "rail",
    "Express Rail Link West Kowloon": "erl",
    "Lok Ma Chau": "road", "Man Kam To": "road", "Heung Yuen Wai": "road",
    "Shenzhen Bay": "road", "Hong Kong-Zhuhai-Macao Bridge": "road",
}
OPEN = {
    "LoWu": (6.5, 24.0), "LMC-Spur": (6.5, 22.5), "LMC": (0.0, 24.0),
    "MKT": (7.0, 22.0), "SZB": (6.5, 24.0), "HYW": (7.0, 22.0),
    "HZMB": (0.0, 24.0), "ERL": (6.0, 24.0),
}


def curve(bumps, base=0.0):
    """24-hour weight vector = flat base + sum of circular gaussian bumps."""
    w = [base] * 24
    for center, width, mass in bumps:
        for h in range(24):
            dh = min(abs(h - center), 24 - abs(h - center))
            w[h] += mass * math.exp(-0.5 * (dh / width) ** 2)
    return [round(v, 4) for v in w]


TEMPLATES = {
    "rail": {
        "Departure": {
            "wk": curve([(7.8, 1.2, 1.0), (12.5, 2.2, 0.8), (18.6, 2.0, 2.4)]),
            "sa": curve([(10.8, 2.6, 2.2), (18.2, 2.4, 1.0)]),
            "su": curve([(10.8, 2.6, 1.8), (18.2, 2.4, 0.8)]),
            "hf": curve([(10.0, 2.4, 2.6), (16.0, 2.5, 0.5)]),
            "hm": curve([(13.5, 3.4, 2.2)]),
            "hl": curve([(13.0, 3.0, 1.4)]),
        },
        "Arrival": {
            "wk": curve([(7.6, 1.1, 2.4), (12.5, 2.2, 0.7), (18.2, 1.8, 1.3)]),
            "sa": curve([(14.5, 3.0, 2.2)]),
            "su": curve([(14.0, 2.6, 1.2), (18.6, 1.8, 2.6)]),
            "hf": curve([(13.5, 3.0, 1.6)]),
            "hm": curve([(13.5, 3.4, 2.2)]),
            "hl": curve([(11.0, 2.4, 0.8), (18.8, 2.0, 2.8)]),
        },
    },
    "erl": {
        "Departure": {
            "wk": curve([(8.4, 1.1, 1.4), (12.5, 2.0, 1.0), (18.8, 1.6, 2.4)]),
            "sa": curve([(9.6, 1.8, 2.2), (16.5, 2.2, 1.2)]),
            "su": curve([(10.2, 2.0, 1.8), (16.5, 2.2, 0.8)]),
            "hf": curve([(9.4, 1.8, 2.6), (15.5, 2.2, 0.6)]),
            "hm": curve([(12.5, 3.0, 2.2)]),
            "hl": curve([(12.5, 2.8, 1.4)]),
        },
        "Arrival": {
            "wk": curve([(8.6, 1.2, 2.2), (12.5, 2.0, 1.0), (19.4, 1.8, 2.0)]),
            "sa": curve([(13.0, 2.8, 2.0), (19.0, 1.8, 1.0)]),
            "su": curve([(13.0, 2.6, 1.4), (19.6, 1.6, 2.2)]),
            "hf": curve([(13.5, 2.8, 1.6)]),
            "hm": curve([(13.5, 3.2, 2.2)]),
            "hl": curve([(11.5, 2.2, 0.8), (19.4, 1.8, 2.6)]),
        },
    },
    "road": {
        "Departure": {
            "wk": curve([(8.2, 1.8, 1.6), (13.0, 2.8, 1.4), (18.4, 2.2, 2.2)], 0.25),
            "sa": curve([(10.5, 2.8, 2.2), (16.5, 2.4, 1.2)], 0.25),
            "su": curve([(10.5, 2.8, 1.8), (16.5, 2.4, 0.8)], 0.25),
            "hf": curve([(9.8, 2.6, 2.6), (15.5, 2.4, 0.8)], 0.25),
            "hm": curve([(13.5, 3.4, 2.2)], 0.25),
            "hl": curve([(13.0, 3.0, 1.4)], 0.25),
        },
        "Arrival": {
            "wk": curve([(8.0, 1.6, 2.0), (13.0, 2.8, 1.2), (19.0, 2.2, 2.0)], 0.25),
            "sa": curve([(14.0, 3.0, 2.2)], 0.25),
            "su": curve([(13.5, 2.8, 1.4), (18.8, 2.2, 2.0)], 0.25),
            "hf": curve([(13.5, 3.0, 1.6)], 0.25),
            "hm": curve([(13.5, 3.4, 2.2)], 0.25),
            "hl": curve([(11.0, 2.4, 0.8), (19.2, 2.2, 2.6)], 0.25),
        },
    },
}


def lognormal_p(x, thr, sigma):
    if thr <= 0 or x <= 0:
        return 0.0
    z = (math.log(thr) - math.log(x)) / max(sigma, 1e-6)
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def main():
    p = load_panel(RAW, keep_cps=list(CPS))
    panel = Panel(p, list(p.columns))
    through = p.index[-1].date()
    post_mask = np.asarray(p.index >= pd.Timestamp(REOPEN))

    # ---------- 2. LOF anomaly flags per series ----------
    anomalies, flag_counts, weekday_base = {}, {}, {}
    for s in p.columns:
        flow = p[s].values.astype(float)
        wm = (pd.Series(flow[post_mask])
              .groupby(p.index[post_mask].dayofweek).median())
        wm = wm.reindex(range(7)).fillna(wm.median())
        weekday_base[s] = wm.values
        base = np.take(wm.values, p.index.dayofweek.values)
        m, _ = flag_anomalies(flow, base)
        anomalies[s] = m
        flag_counts[str(s)] = int((m & post_mask).sum())

    # ---------- 3. GBT: pooled training + recursive forecast ----------
    X, y, keys = panel.matrices(anomalies)
    ntr = int(len(X) * 0.97)
    model = HistGBR(n_estimators=400, eval_set=(X[ntr:], y[ntr:]),
                    patience=40, seed=7005).fit(X[:ntr], y[:ntr])

    # per-series residual sigma on training rows (honest uncertainty bands)
    pred_tr = model.predict(X)
    resid = y - pred_tr
    sigma_s = {}
    for s in p.columns:
        mask = np.array([k[0] == s for k in keys])
        sigma_s[s] = float(resid[mask].std()) if mask.sum() > 30 else float(resid.std())

    forecast = {}
    mk_chain, mk_state = {}, {}
    for s in p.columns:
        flow_s = p[s]
        post = flow_s[post_mask]
        # stats block (quantile bands etc.)
        q = {qq: float(np.percentile(post, qq)) for qq in (50, 85, 95, 99)}

        # Markov chain on LOF-cleaned post-reopening states
        clean = post[~anomalies[s][post_mask]]
        states = np.array([state_of(v, q) for v in clean])
        P = fit_transition(states)
        mk_chain[s] = P
        mk_state[s] = state_of(float(flow_s.iloc[-1]), q)   # today's last actual

        # recursive forecast starting today
        hist = logf(flow_s.values)
        d0 = dt.date.today()
        rows = []
        for h in range(HORIZON):
            d = d0 + dt.timedelta(days=h)
            xrow = panel.row(s, d, hist)
            yhat = float(model.predict(xrow[None, :])[0])
            hist = np.append(hist, yhat)
            mu = float(np.exp(yhat))
            sig = sigma_s[s]
            level = 1 + (mu > q[50]) + (mu > q[85]) + (mu > q[95])
            hit, code, ph = panel.hol_info(d)[0], "", ""
            if hit:
                code, off, ln = hit
                ph = ("first" if off <= 2 else "last" if off >= ln - 3 else "mid")
            steps = (d - through).days          # Markov steps from last actual
            dist = forecast_dist(P, mk_state[s], max(steps, 1))[steps - 1]
            rows.append({
                "date": d.isoformat(), "dow": d.weekday(),
                "base": round(mu), "festive": code, "ph": ph,
                "level": int(level),
                "p_busy": round(lognormal_p(mu, q[85], sig), 3),
                "lo": round(mu * math.exp(-1.2816 * sig)),
                "hi": round(mu * math.exp(1.2816 * sig)),
                "mk": [round(float(v), 3) for v in dist],
            })
        forecast["%s|%s" % (CPS[s[0]]["short"], s[1])] = rows

    # ---------- stats / spark ----------
    stats, spark = {}, {}
    for s in p.columns:
        flow_s = p[s]
        post = flow_s[post_mask]
        q = {qq: float(np.percentile(post, qq)) for qq in (50, 85, 95, 99)}
        key = "%s|%s" % (CPS[s[0]]["short"], s[1])
        stats[key] = {
            "cp": s[0], "cn": CPS[s[0]]["cn"], "short": CPS[s[0]]["short"],
            "dir": s[1], "profile": PROFILE_OF[s[0]],
            "q": q, "weekday_base": [round(float(v)) for v in weekday_base[s]],
            "sigma": round(sigma_s[s], 4), "n_days": int(post.notna().sum()),
        }
        tail = flow_s.tail(60)
        spark[key] = {
            "dates": [d.strftime("%m-%d") for d in tail.index],
            "flow": [round(float(v)) for v in tail],
        }

    # ---------- model card (backtest numbers if present) ----------
    card = {"name": "HistGBR (own NumPy, %d trees)" % model.best_m_,
            "target": "log daily flow", "features": len(FEATURE_NAMES),
            "train_rows": int(len(X))}
    bt_path = os.path.join(ROOT, "report", "backtest.json")
    if os.path.exists(bt_path):
        with open(bt_path, encoding="utf-8") as fh:
            bt = json.load(fh)
        card["backtest"] = {k: {kk: round(vv, 3) for kk, vv in v.items()}
                            for k, v in bt.items() if isinstance(v, dict) and "rmse" in v}

    # ---------- NLP page (merged for one-fetch app) ----------
    attractions = None
    if os.path.exists(NLP):
        with open(NLP, encoding="utf-8") as fh:
            attractions = json.load(fh)

    out = {
        "app": "BorderPulse", "version": "0.3.0",
        "built_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_through": through.isoformat(),
        "reopen": REOPEN.isoformat(),
        "horizon": HORIZON,
        "model": card,
        "lof_flagged": flag_counts,
        "intraday": {"templates": TEMPLATES, "open": OPEN},
        "stats": stats, "spark": spark, "forecast": forecast,
        "notes": "v0.3: LOF-cleaned GBT (own NumPy) recursive forecast + Markov "
                 "state chain + intraday templates + Erlang-C queue; GBT replaces "
                 "the v0.2 weekday-median baseline.",
    }
    if attractions:
        out["nlp"] = attractions
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print("series:", len(stats), "| LOF flagged:", sum(flag_counts.values()),
          "| trees:", model.best_m_, "| through:", through)
    print("wrote", OUT, "%.0f KB" % (os.path.getsize(OUT) / 1024))


if __name__ == "__main__":
    main()
