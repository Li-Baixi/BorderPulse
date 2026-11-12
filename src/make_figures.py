# -*- coding: utf-8 -*-
"""EDA + model figures for the report (English labels, PNG into report/figs).
Primary author: Li Baixi (evaluation figures); EDA panels Lin Jingyu.


Reuses the project's own implementations (no new modelling):
  fig01  16-series panel overview with the 2023-02-06 reopening break
  fig02  Lo Wu Departure weekly medians around the reopening
  fig03  festive multipliers (flow / weekday median) by holiday, phase, direction
  fig04  LOF score timeline + flagged days (typhoon validation)
  fig05  rolling-backtest: baseline vs own GBT vs sklearn, + per-series RMSE
  fig06  GBT split-gain feature importance (one-hot families aggregated)
  fig07  14-day recursive forecast fan chart (product data.json)
  fig08  Markov transition matrix + h-step state distribution
  fig09  Monte Carlo wait-time distribution on a P90 day
  fig10  intraday hourly templates (profile x direction)
  fig11  NLP: attraction ranking + own-vs-sklearn metrics

Run:  python src/make_figures.py        (~3 min: refits HistGBR for fig06)
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.features import Panel, REOPEN, FEATURE_NAMES, load_panel
from models.lof import flag_anomalies, LOF
from models.gbt import HistGBR
from models.markov import fit_transition, forecast_dist, state_of, stationary
from models import mc_queue
from backtest import festive_multipliers
from build_docs import CPS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "immd_daily.csv")
FIGS = os.path.join(ROOT, "report", "figs")

# palette (validated categorical trio for model comparisons: ours / baseline / sklearn)
BLUE, BASE, ORANGE = "#2a78d6", "#b0568f", "#c96a2b"
GRAY = "#898781"          # reference lines / decoration only, never a series
GOOD, WARN, SERI, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
INK, GRID = "#0b0b0b", "#e1e0d9"
BAND_COLORS = [GOOD, WARN, SERI, CRIT]
BAND_NAMES = ["Smooth", "Normal", "Busy", "Packed"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 150, "savefig.bbox": "tight",
})


def short(series):
    cp, direction = series
    return "%s %s" % (CPS[cp]["short"], "Dep" if direction == "Departure" else "Arr")


def decorate(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- fig01
def fig01_overview(p):
    order = sorted(p.columns, key=lambda s: -p[s].tail(400).median())
    fig, axes = plt.subplots(4, 4, figsize=(7.4, 6.4), sharex=True)
    reopen_ts = pd.Timestamp(REOPEN)
    for ax, s in zip(axes.flat, order):
        v = p[s] / 1000.0
        ax.plot(v.index, v.values, lw=0.7, color=BLUE)
        ax.axvline(reopen_ts, lw=0.8, color=GRAY, ls="--", alpha=0.8)
        ax.set_title(short(s), loc="left", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.set_ylim(bottom=0)
    axes.flat[0].annotate("reopening\n2023-02-06", xy=(reopen_ts, 0.86),
                          xycoords=("data", "axes fraction"), fontsize=6.5,
                          color=GRAY, xytext=(3, 0), textcoords="offset points")
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Daily passenger flow by control point and direction, 2021-2026",
                 x=0.02, ha="left", fontsize=10.5)
    fig.supxlabel("thousands of passengers per day", fontsize=8, color=GRAY)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    save(fig, "fig01_series_overview.png")


# ---------------------------------------------------------------- fig02
def fig02_reopen(p):
    s = ("Lo Wu", "Departure")
    v = p[s]["2022-09-01":"2023-10-31"]
    wk = v.resample("W").median() / 1000.0
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(wk.index, wk.values, lw=1.6, color=BLUE)
    ax.axvline(pd.Timestamp(REOPEN), lw=1.0, color=INK, ls="--", alpha=0.6)
    pre = v[v.index < pd.Timestamp(REOPEN)].median() / 1000.0
    post = v[v.index >= pd.Timestamp(REOPEN)].median() / 1000.0
    ax.annotate("reopening 2023-02-06", xy=(pd.Timestamp(REOPEN), wk.max() * 0.97),
                fontsize=8, color=INK, xytext=(6, 0), textcoords="offset points",
                va="top")
    ax.axhline(pre, lw=0.9, color=GRAY, ls=":")
    ax.axhline(post, lw=0.9, color=GRAY, ls=":")
    ax.text(wk.index[2], pre + wk.max() * 0.03, "pre median %.1fk" % pre,
            fontsize=7.5, color=GRAY)
    ax.text(wk.index[2], post + wk.max() * 0.03, "post median %.1fk (%.1fx)" % (post, post / max(pre, 1e-9)),
            fontsize=7.5, color=GRAY)
    decorate(ax, "Lo Wu departures around the 2023 reopening (weekly medians)",
             "", "thousands per day")
    save(fig, "fig02_reopen_break.png")


# ---------------------------------------------------------------- fig03
def fig03_festive(p):
    fmult = festive_multipliers(p, p.index[-1])
    codes = ["CNY", "ChingMing", "Easter", "MayDay", "NatDay", "Xmas", "NewYear"]
    phases = ["first", "mid", "last"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    x = np.arange(len(codes))
    w = 0.26
    for ax, direction, ttl in zip(axes, ["Departure", "Arrival"],
                                  ["Departures", "Arrivals"]):
        for i, ph in enumerate(phases):
            vals = [fmult.get((direction, c, ph), np.nan) for c in codes]
            ax.bar(x + (i - 1) * w, vals, width=w * 0.9,
                   color=[BLUE, BASE, ORANGE][i], label=ph)
        ax.axhline(1.0, lw=0.9, color=INK, ls=":", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(codes, rotation=30, ha="right")
        decorate(ax, ttl, "", "flow / weekday median")
    axes[0].legend(title="phase", frameon=False, loc="upper left")
    fig.suptitle("Festive multipliers by holiday window, phase and direction (medians)",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig03_festive_multipliers.png")


# ---------------------------------------------------------------- fig04
def fig04_lof(p):
    s = ("Lo Wu", "Departure")
    post_mask = np.asarray(p.index >= pd.Timestamp(REOPEN))
    flow = p[s].values.astype(float)
    wm = (pd.Series(flow[post_mask]).groupby(p.index[post_mask].dayofweek).median())
    wm = wm.reindex(range(7)).fillna(wm.median())
    base = np.take(wm.values, p.index.dayofweek.values)
    mask, lof = flag_anomalies(flow, base)
    idx = p.index[post_mask]
    sc, fl = lof[post_mask], mask[post_mask]

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.scatter(idx[~fl], sc[~fl], s=5, color=BLUE, alpha=0.55, lw=0)
    ax.scatter(idx[fl], sc[fl], s=14, color=CRIT, lw=0)
    ax.axhline(1.3, lw=0.9, color=INK, ls="--", alpha=0.6)
    ax.text(idx[5], 1.32, "LOF = 1.3 gate", fontsize=7.5, color=INK, va="bottom")
    notes = [("2023-09-01", "Typhoon Saola"), ("2023-09-08", "Black rainstorm"),
             ("2024-09-06", "Typhoon Yagi")]
    for ds, label in notes:
        ts = pd.Timestamp(ds)
        hit = (idx.date == ts.date()) & fl
        if hit.any():
            ax.annotate(label, xy=(ts, sc[hit][0]), xytext=(4, 10),
                        textcoords="offset points", fontsize=7, color=CRIT,
                        arrowprops=dict(arrowstyle="-", color=CRIT, lw=0.7))
    decorate(ax, "LOF anomaly scores, Lo Wu departures (flagged days in red; "
                 "top-1.5% rank + score > 1.3 gates)", "", "LOF score")
    save(fig, "fig04_lof_timeline.png")


# ---------------------------------------------------------------- fig05
def fig05_backtest():
    with open(os.path.join(ROOT, "report", "backtest.json"), encoding="utf-8") as fh:
        bt = json.load(fh)
    models = [("baseline", "Weekday median x festive", BASE),
              ("gbt", "Own NumPy GBT", BLUE),
              ("sk", "sklearn GBR (benchmark)", ORANGE)]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    panels = [
        ("rmse", "RMSE (passengers/day)", "{:,.0f}", False),
        ("mape", "MAPE", "{:.1%}", False),
        ("band_acc1", "Band accuracy within +/-1", "{:.1%}", True),
    ]
    for ax, (key, ttl, fmt, _) in zip(axes, panels):
        vals = [bt[m][key] for m, _, _ in models]
        bars = ax.bar(range(3), vals, width=0.62,
                      color=[c for _, _, c in models])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["baseline", "own GBT", "sklearn"], fontsize=7.5)
        decorate(ax, ttl)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.suptitle("Rolling-origin backtest, 5 x 14-day windows, 16 series pooled (n = %s)"
                 % format(bt["gbt"]["n"], ","),
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig05_backtest.png")

    # per-series RMSE panel (own GBT)
    per = bt["per_series_gbt"]
    items = sorted(per.items(), key=lambda kv: kv[1]["rmse"])
    names = [short(eval(k)) for k, _ in items]          # noqa: S307 - own json
    vals = [v["rmse"] for _, v in items]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.barh(range(len(vals)), vals, height=0.62, color=BLUE)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=7.5)
    for i, v in enumerate(vals):
        ax.text(v, i, " {:,.0f}".format(v), va="center", fontsize=7)
    decorate(ax, "Own GBT backtest RMSE by series (pooled fit, recursive 14-day forecast)",
             "RMSE (passengers/day)")
    save(fig, "fig05b_per_series_rmse.png")


# ---------------------------------------------------------------- fig06
def fig06_importance(p, panel, anomalies, model):
    imp = model.importance_ / model.importance_.sum()
    fams = {"dow_": "day-of-week one-hots", "m_": "month one-hots",
            "code_": "holiday-code one-hots", "series_": "series one-hots",
            "ph_": "holiday phase one-hots"}
    rows = {}
    for name, v in zip(FEATURE_NAMES, imp):
        fam = next((f for k, f in fams.items() if name.startswith(k)), None)
        if fam:
            rows[fam] = rows.get(fam, 0.0) + float(v)
        else:
            rows[name] = float(v)
    pretty = {"lag_1": "lag-1 log flow", "lag_7": "lag-7 log flow",
              "lag_14": "lag-14 log flow", "lag_28": "lag-28 log flow",
              "roll7_mean": "7-day mean (log)", "roll7_std": "7-day std (log)",
              "roll28_mean": "28-day mean (log)", "hol": "in-holiday flag",
              "d_to_start": "days to holiday start", "d_from_end": "days from holiday end",
              "trend": "days since reopening"}
    items = sorted(rows.items(), key=lambda kv: kv[1])[-14:]
    names = [pretty.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.barh(range(len(vals)), vals, height=0.62, color=BLUE)
    for b, v in zip(bars, vals):
        ax.text(v, b.get_y() + b.get_height() / 2, " {:.1%}".format(v),
                va="center", fontsize=7.5)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, max(vals) * 1.16)
    decorate(ax, "GBT split-gain importance (aggregated one-hot families)",
             "share of total gain")
    save(fig, "fig06_feature_importance.png")


# ---------------------------------------------------------------- fig07
def fig07_forecast(p):
    with open(os.path.join(ROOT, "docs", "data.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    rows = d["forecast"]["LoWu|Departure"]
    sig = d["stats"]["LoWu|Departure"]["sigma"]
    hist = p[("Lo Wu", "Departure")].tail(120) / 1000.0
    fd = [pd.Timestamp(r["date"]) for r in rows]
    base = np.array([r["base"] for r in rows]) / 1000.0
    lo = np.array([r["lo"] for r in rows]) / 1000.0
    hi = np.array([r["hi"] for r in rows]) / 1000.0

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(hist.index, hist.values, lw=1.3, color=BLUE, label="actual")
    ax.plot(fd, base, lw=1.3, color=ORANGE, ls="--", label="forecast (recursive)")
    ax.fill_between(fd, lo, hi, color=ORANGE, alpha=0.18, label="P10-P90 band")
    ax.axvline(pd.Timestamp(rows[0]["date"]), lw=0.9, color=INK, ls=":", alpha=0.6)
    ax.text(pd.Timestamp(rows[0]["date"]), ax.get_ylim()[1], " today",
            fontsize=7.5, va="top", color=INK)
    decorate(ax, "Lo Wu departures: 120 days of actuals + 14-day recursive forecast",
             "", "thousands per day")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    save(fig, "fig07_forecast_fan.png")
    return sig


# ---------------------------------------------------------------- fig08
def fig08_markov(p):
    s = ("Lo Wu", "Departure")
    post_mask = np.asarray(p.index >= pd.Timestamp(REOPEN))
    post = p[s][post_mask]
    # replicate build_docs: LOF-cleaned states
    flow = p[s].values.astype(float)
    wm = (pd.Series(flow[post_mask]).groupby(p.index[post_mask].dayofweek).median())
    wm = wm.reindex(range(7)).fillna(wm.median())
    base = np.take(wm.values, p.index.dayofweek.values)
    mask, _ = flag_anomalies(flow, base)
    q = {qq: float(np.percentile(post, qq)) for qq in (50, 85, 95, 99)}
    clean = post[~mask[post_mask]]
    states = np.array([state_of(v, q) for v in clean])
    Pm = fit_transition(states)
    start = state_of(float(p[s].iloc[-1]), q)
    dist = forecast_dist(Pm, start, 14)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.1),
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    im = ax1.imshow(Pm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, "%.2f" % Pm[i, j], ha="center", va="center",
                     fontsize=7.5, color=INK if Pm[i, j] < 0.6 else "white")
    ax1.set_xticks(range(4)); ax1.set_xticklabels(BAND_NAMES, fontsize=7.5)
    ax1.set_yticks(range(4)); ax1.set_yticklabels(BAND_NAMES, fontsize=7.5)
    ax1.grid(False)
    decorate(ax1, "Transition matrix P (Laplace-smoothed)")
    hs = np.arange(1, 15)
    for b in range(4):
        ax2.plot(hs, dist[:, b], lw=1.6, color=BAND_COLORS[b], label=BAND_NAMES[b])
        ax2.text(14.15, dist[-1, b], BAND_NAMES[b], fontsize=7.5,
                 color=BAND_COLORS[b], va="center")
    pi = stationary(Pm)
    for b in range(4):
        ax2.axhline(pi[b], lw=0.7, ls=":", color=BAND_COLORS[b], alpha=0.6)
    decorate(ax2, "h-step state distribution from today (state %d); dotted = stationary"
                 % start, "days ahead", "probability")
    ax2.set_xlim(1, 15.9)
    fig.suptitle("Markov congestion-state chain, Lo Wu departures",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig08_markov.png")


# ---------------------------------------------------------------- fig09
def fig09_mc(p, sigma):
    s = ("Lo Wu", "Departure")
    post = p[s][p.index >= pd.Timestamp(REOPEN)]
    q90 = float(np.percentile(post, 90))
    c = mc_queue.counter_count(q90)
    w = mc_queue.simulate(q90, sigma, c, trials=6000)
    p30 = float((w > 30).mean())
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.hist(w, bins=48, range=(0, 120), color=BLUE, alpha=0.75)
    ax.axvline(30, lw=1.2, color=CRIT)
    ax.text(31, ax.get_ylim()[1] * 0.95, "30 min line: P(wait > 30) = %.0f%%" % (100 * p30),
            fontsize=8, color=CRIT, va="top")
    med, p95 = float(np.median(w)), float(np.percentile(w, 95))
    ax.axvline(med, lw=1.0, color=INK, ls="--", alpha=0.7)
    ax.axvline(p95, lw=1.0, color=INK, ls=":", alpha=0.7)
    ax.text(med - 1, ax.get_ylim()[1] * 0.62, "median %.0f" % med, fontsize=7.5,
            ha="right", color=INK)
    ax.text(p95 + 1, ax.get_ylim()[1] * 0.62, "P95 %.0f" % p95, fontsize=7.5, color=INK)
    decorate(ax, "Monte Carlo queue wait times, P90 day at Lo Wu "
                 "(%d counters, mu = %.0f/day, sigma = %.2f)" % (c, q90, sigma),
             "wait in the peak hour (minutes)", "trials")
    save(fig, "fig09_mc_wait.png")


# ---------------------------------------------------------------- fig10
def fig10_intraday():
    with open(os.path.join(ROOT, "docs", "data.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    T = d["intraday"]["templates"]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.6), sharex=True)
    hours = np.arange(24)
    for row, prof in enumerate(["rail", "erl", "road"]):
        for col, direction in enumerate(["Departure", "Arrival"]):
            ax = axes[row][col]
            t = T[prof][direction]
            ax.plot(hours, t["wk"], lw=1.4, color=BLUE, label="weekday")
            ax.plot(hours, t["su"], lw=1.4, color=ORANGE, ls="--", label="Sunday")
            ax.fill_between(hours, 0, t["wk"], color=BLUE, alpha=0.10)
            decorate(ax, "%s - %ss" % (prof, direction[:-3].lower()))
            ax.set_xlim(0, 23)
            ax.set_xticks([0, 6, 12, 18, 23])
    axes[0][0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Intraday arrival templates by port profile and direction "
                 "(share of daily flow per hour)", x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, "fig10_intraday_templates.png")


# ---------------------------------------------------------------- fig11
def fig11_nlp():
    with open(os.path.join(ROOT, "docs", "nlp.json"), encoding="utf-8") as fh:
        nlp = json.load(fh)
    at = sorted(nlp["attractions"], key=lambda a: a["pos"])
    names = [a["en"] for a in at]
    vals = [a["pos"] for a in at]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    ax1.barh(range(len(vals)), vals, height=0.62, color=BLUE)
    ax1.set_yticks(range(len(vals)))
    ax1.set_yticklabels(names, fontsize=8)
    for i, v in enumerate(vals):
        ax1.text(v, i, " {:.0%}".format(v), va="center", fontsize=7.5)
    ax1.set_xlim(0, 1.02)
    decorate(ax1, "Share of predicted-positive reviews (synthetic demo corpus)",
             "")
    mets = [("accuracy", "Accuracy", "{:.1%}"), ("auc", "AUC", "{:.3f}")]
    x = np.arange(2)
    w = 0.34
    for i, (key, _, _) in enumerate(mets):
        own = [nlp["acc"] if key == "accuracy" else nlp["auc"]]
        sk = [nlp["acc_sk"] if key == "accuracy" else nlp["auc_sk"]]
        ax2.bar(i - w / 2, own, width=w, color=BLUE, label="own NumPy" if i == 0 else None)
        ax2.bar(i + w / 2, sk, width=w, color=ORANGE, label="sklearn (benchmark)" if i == 0 else None)
        ax2.text(i - w / 2, own[0], mets[i][2].format(own[0]), ha="center",
                 va="bottom", fontsize=7.5)
        ax2.text(i + w / 2, sk[0], mets[i][2].format(sk[0]), ha="center",
                 va="bottom", fontsize=7.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[1] for m in mets], fontsize=8)
    ax2.legend(frameon=False, fontsize=7)
    decorate(ax2, "Held-out test set")
    fig.suptitle("Attraction sentiment: ranking and model quality "
                 "(trained on %s real TripAdvisor reviews)" % format(nlp["train_reviews"], ","),
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig11_nlp.png")


# ---------------------------------------------------------------- main
def main():
    p = load_panel(RAW, keep_cps=list(CPS))
    panel = Panel(p, list(p.columns))
    post_mask = np.asarray(p.index >= pd.Timestamp(REOPEN))

    fig01_overview(p)
    fig02_reopen(p)
    fig03_festive(p)
    fig04_lof(p)
    fig05_backtest()
    fig10_intraday()
    fig11_nlp()

    # ---- shared heavy state: LOF + pooled GBT (as in build_docs) ----
    anomalies = {}
    for s in p.columns:
        flow = p[s].values.astype(float)
        wm = (pd.Series(flow[post_mask])
              .groupby(p.index[post_mask].dayofweek).median())
        wm = wm.reindex(range(7)).fillna(wm.median())
        base = np.take(wm.values, p.index.dayofweek.values)
        anomalies[s], _ = flag_anomalies(flow, base)
    X, y, keys = panel.matrices(anomalies)
    ntr = int(len(X) * 0.97)
    model = HistGBR(n_estimators=400, eval_set=(X[ntr:], y[ntr:]),
                    patience=40, seed=7005).fit(X[:ntr], y[:ntr])
    print("GBT refit for importance: %d trees, %d rows" % (model.best_m_, len(X)))

    fig06_importance(p, panel, anomalies, model)
    sigma = fig07_forecast(p)
    fig08_markov(p)
    fig09_mc(p, sigma)
    print("all figures done")


if __name__ == "__main__":
    main()
