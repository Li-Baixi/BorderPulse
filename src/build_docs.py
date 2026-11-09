# -*- coding: utf-8 -*-
"""BorderPulse docs builder.

Reads data/raw/immd_daily.csv (ImmD Statistics on Daily Passenger Traffic) and
writes docs/data.json consumed by the static PWA:

  * per control-point x direction stats self-calibrated from post-reopening data
    (P50/P85/P95/P99 daily flow -> 4 congestion bands + effective capacity)
  * v1 baseline forecast for the next 14 days: weekday-of-week median x festive
    multiplier, with lognormal uncertainty (sigma from residual log-deviations)
  * recent 60-day sparkline series

Run:  python src/build_docs.py
"""
import datetime as dt
import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "immd_daily.csv")
FESTIVE = "https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_passenger_traffic_festival_periods.csv"
OUT = os.path.join(ROOT, "docs", "data.json")

# --- control-point configuration (land boundary + cross-boundary rail) -------
CPS = {
    "Lo Wu":                       {"cn": "罗湖",           "short": "LoWu"},
    "Lok Ma Chau Spur Line":       {"cn": "落马洲支线",      "short": "LMC-Spur"},
    "Lok Ma Chau":                 {"cn": "落马洲",          "short": "LMC"},
    "Huanggang":                   {"cn": "皇岗",           "short": "HG"},
    "Shenzhen Bay":                {"cn": "深圳湾",          "short": "SZB"},
    "Man Kam To":                  {"cn": "文锦渡",          "short": "MKT"},
    "Sha Tau Kok":                 {"cn": "沙头角",          "short": "STK"},
    "Heung Yuen Wai":              {"cn": "香园围",          "short": "HYW"},
    "Hong Kong-Zhuhai-Macao Bridge": {"cn": "港珠澳大桥",    "short": "HZMB"},
    "Express Rail Link West Kowloon": {"cn": "高铁西九龙",   "short": "ERL"},
}
# spelling variants that appear across the years
ALIASES = {"Macau Ferry Terminal": "Macao Ferry Terminal"}

REOPEN = dt.date(2023, 2, 6)      # full border reopening
HORIZON = 14                      # forecast horizon (days)

# built-in cross-border holiday calendar (HK + mainland), 2023-2027.
# The ImmD festive-period dataset stops at 2020-12-31, so festival factors are
# learned against this calendar instead (see report: data-stale finding).
HOLIDAYS = {
    "CNY":      [(2023, 1, 19, 2023, 1, 28), (2024, 2, 7, 2024, 2, 16),
                 (2025, 1, 26, 2025, 2, 4), (2026, 2, 13, 2026, 2, 23),
                 (2027, 2, 3, 2027, 2, 13)],
    "ChingMing": [(y, 4, 3, y, 4, 6) for y in range(2023, 2028)],
    "Easter":   [(2023, 4, 6, 2023, 4, 11), (2024, 3, 28, 2024, 4, 2),
                 (2025, 4, 17, 2025, 4, 22), (2026, 4, 2, 2026, 4, 7),
                 (2027, 3, 25, 2027, 3, 30)],
    "MayDay":   [(y, 4, 30, y, 5, 5) for y in range(2023, 2028)],
    "NatDay":   [(2023, 9, 28, 2023, 10, 7), (2024, 9, 30, 2024, 10, 8),
                 (2025, 9, 30, 2025, 10, 9), (2026, 9, 30, 2026, 10, 9),
                 (2027, 9, 30, 2027, 10, 9)],
    "Xmas":     [(y, 12, 22, y, 12, 27) for y in range(2023, 2028)],
    "NewYear":  [(y, 12, 30, y + 1, 1, 3) for y in range(2023, 2027)],
}


def load_daily():
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Arrival / Departure": "Dir"})
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df["cp"] = df["Control Point"].str.strip().replace(ALIASES)
    df = df[df["cp"].isin(CPS)].copy()
    df["flow"] = df["Total"].astype(float)
    return df[["Date", "cp", "Dir", "flow"]]


def load_festive():
    """Return {date: (code, offset, length)} from the built-in calendar."""
    out = {}
    for code, wins in HOLIDAYS.items():
        for y0, m0, d0, y1, m1, d1 in wins:
            start, end = dt.date(y0, m0, d0), dt.date(y1, m1, d1)
            length = (end - start).days + 1
            d = start
            while d <= end:
                out[d] = (code, (d - start).days, length)
                d += dt.timedelta(days=1)
    return out


def phase_of(offset, length):
    if offset <= 2:
        return "first"
    if offset >= length - 3:
        return "last"
    return "mid"


def quantiles(x):
    return {q: float(np.percentile(x, q)) for q in (50, 85, 95, 99)}


def lognormal_p(x, thr, sigma):
    """P(X > thr) for X ~ LogNormal(ln x, sigma)."""
    if thr <= 0 or x <= 0:
        return 0.0
    z = (math.log(thr) - math.log(x)) / max(sigma, 1e-6)
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def main():
    df = load_daily()
    festive = load_festive()
    through = df["Date"].max().date()

    stats, series, spark = {}, {}, {}
    for (cp, dire), g in df.groupby(["cp", "Dir"]):
        g = g.sort_values("Date")
        post = g[g["Date"].dt.date >= REOPEN]
        if len(post) < 45:             # Huanggang: reopened 2026, ~8 weeks history
            continue
        recent = post[post["Date"] >= post["Date"].max() - pd.Timedelta(days=365)]
        if recent["flow"].median() < 50:   # STK: passenger clearance suspended, all-zero
            continue
        # weekday-of-week baseline (median), trained on last 365 days
        wk = recent.groupby(recent["Date"].dt.dayofweek)["flow"].median()
        base = wk.reindex(range(7)).fillna(wk.median())
        # residual spread in log space -> forecast uncertainty
        pred = recent["Date"].dt.dayofweek.map(base).astype(float)
        resid = np.log(recent["flow"].clip(lower=1) / pred.clip(lower=1))
        sigma = float(resid.std()) if len(resid) > 30 else 0.25
        key = "%s|%s" % (CPS[cp]["short"], dire)
        stats[key] = {
            "cp": cp, "cn": CPS[cp]["cn"], "short": CPS[cp]["short"], "dir": dire,
            "q": quantiles(recent["flow"].values), "weekday_base": [round(float(v)) for v in base],
            "sigma": round(sigma, 4), "n_days": int(len(post)),
        }
        tail = g.tail(60)
        spark[key] = {
            "dates": [d.strftime("%m-%d") for d in tail["Date"]],
            "flow": [round(float(v)) for v in tail["flow"]],
        }
        series[key] = (recent, base, sigma)

    # festive multipliers learned per (direction, code, phase): northbound and
    # southbound peaks sit at opposite ends of a holiday, so pooling directions
    # cancels the signal.
    mult = {}
    for key, (recent, base, sigma) in series.items():
        dire = stats[key]["dir"]
        for _, r in recent.iterrows():
            hit = festive.get(r["Date"].date())
            if hit:
                code, off, ln = hit
                b = float(base[r["Date"].dayofweek])
                mk = "%s|%s|%s" % (dire, code, phase_of(off, ln))
                mult.setdefault(mk, []).append(r["flow"] / max(b, 1.0))
    fmult = {c: round(float(np.median(v)), 3) for c, v in mult.items() if len(v) >= 20}

    # 14-day baseline forecast
    forecast = {}
    today = dt.date.today()
    for key, (recent, base, sigma) in series.items():
        s = stats[key]
        q = s["q"]
        dire = s["dir"]
        rows = []
        for i in range(1, HORIZON + 1):
            d = today + dt.timedelta(days=i)
            b = float(base[d.weekday()])
            hit = festive.get(d)
            m = 1.0
            code = ""
            if hit:
                c0, off, ln = hit
                mk = "%s|%s|%s" % (dire, c0, phase_of(off, ln))
                m = fmult.get(mk, 1.0)
                code = c0
            mu = b * m
            level = 1 + (mu > q[50]) + (mu > q[85]) + (mu > q[95])
            rows.append({
                "date": d.isoformat(), "dow": d.weekday(),
                "base": round(mu), "festive": code if m != 1.0 else "",
                "level": int(level),
                "p_busy": round(lognormal_p(mu, q[85], sigma), 3),
                "lo": round(mu * math.exp(-1.2816 * sigma)),
                "hi": round(mu * math.exp(1.2816 * sigma)),
            })
        forecast[key] = rows

    out = {
        "app": "BorderPulse", "version": "0.1.3",
        "built_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_through": through.isoformat(),
        "reopen": REOPEN.isoformat(),
        "horizon": HORIZON,
        "festive_mult": fmult,
        "levels": ["Smooth", "Normal", "Busy", "Packed"],
        "stats": stats, "spark": spark, "forecast": forecast,
        "notes": "v1 baseline: weekday median x festive multiplier, self-calibrated "
                 "P50/P85/P95/P99 bands (post-2023-02-06). GBT model to replace.",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print("cps x dir:", len(stats), "| festive codes:", fmult, "| through:", through)
    print("wrote", OUT, "%.0f KB" % (os.path.getsize(OUT) / 1024))


if __name__ == "__main__":
    main()
