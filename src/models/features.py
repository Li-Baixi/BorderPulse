# -*- coding: utf-8 -*-
"""Panel construction + feature engineering for the GBT forecaster.

One pooled panel over all 16 control-point x direction series (about 22k
series-days since the 2023-02-06 reopening). Pooled training lets small
control points borrow strength from the big ones via the series one-hots.

Target: log(flow). Multiplicative noise (weather, holidays, holidays missped)
is close to lognormal, and log-space residuals give the direct sigma for the
product's uncertainty bands.

Features (all plain floats; one-hot blocks are 0/1):
  lags      log-flow at d-1, d-7, d-14, d-28      (recursive strategy: during
            forecasting, lag values beyond the last observed day come from
            the model's own predictions)
  rolls     mean/std of log-flow over d-7..d-1, mean over d-28..d-1
  calendar  day-of-week (7), month (12)
  festive   in-holiday flag, phase one-hot (first/mid/last), holiday-code
            one-hot (7 codes), signed days to holiday start / from end
  series    control-point-direction one-hot (16)
  trend    (date - reopening) / 400 days
"""
import datetime as dt

import numpy as np
import pandas as pd

REOPEN = dt.date(2023, 2, 6)
HORIZON = 14

HOL_CODES = ["CNY", "ChingMing", "Easter", "MayDay", "NatDay", "Xmas", "NewYear"]
DOW_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
MONTHS = list(range(1, 13))

FEATURE_NAMES = (
    [f"lag_{l}" for l in (1, 7, 14, 28)]
    + ["roll7_mean", "roll7_std", "roll28_mean"]
    + [f"dow_{d}" for d in DOW_NAMES]
    + [f"m_{m}" for m in MONTHS]
    + ["hol", "ph_first", "ph_mid", "ph_last"]
    + [f"code_{c}" for c in HOL_CODES]
    + ["d_to_start", "d_from_end"]
    + [f"series_{i}" for i in range(16)]
    + ["trend"]
)
N_FEAT = len(FEATURE_NAMES)


def logf(flow):
    """log of daily flow, safe on closed days (typhoon shutdowns, partial
    suspensions) where the published total is 0: clip to 1 first so log is 0."""
    return np.log(np.clip(np.asarray(flow, dtype=float), 1.0, None))


def load_panel(csv_path, keep_cps=None):
    """Wide panel: index=date, columns=MultiIndex(cp_full_name, direction).

    keep_cps: restrict to these control points (the product's 8 land /
    cross-boundary ports; None keeps every series that passes the filters).
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Arrival / Departure": "Dir"})
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df["cp"] = df["Control Point"].str.strip()
    df = df[df["Dir"].isin(["Arrival", "Departure"])]
    p = df.pivot_table(index="Date", columns=["cp", "Dir"], values="Total", aggfunc="sum")
    p = p.sort_index().asfreq("D")                     # daily continuous index
    if keep_cps is not None:
        p = p[[c for c in p.columns if c[0] in keep_cps]]
    # series kept: >=45 days history and median >= 50 (drops STK/HG, see report)
    keep = []
    for c in p.columns:
        s = p[c].dropna()
        recent = s[s.index >= pd.Timestamp(REOPEN)]
        if len(recent) >= 45 and recent.median() >= 50:
            keep.append(c)
    p = p[keep]
    # rare missing days inside a kept series -> forward-fill (documented)
    p = p.ffill()
    return p


def holiday_windows():
    """HK + mainland cross-border holiday calendar 2023-2027 (tuple ranges)."""
    H = {
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
    out = []
    for code, wins in H.items():
        for y0, m0, d0, y1, m1, d1 in wins:
            start = dt.date(y0, m0, d0)
            end = dt.date(y1, m1, d1)
            out.append((code, start, end, (end - start).days + 1))
    return out


def phase_of(offset, length):
    if offset <= 2:
        return "first"
    if offset >= length - 3:
        return "last"
    return "mid"


class Panel:
    """Owns the wide panel and produces model matrices for any (series, date).

    row(series_idx, date, hist_log) builds one feature row; hist_log is the
    series' log-flow history UP TO date-1 (actual values during training,
    actual-then-predicted during recursive forecasting).
    """

    def __init__(self, panel, series_order):
        self.p = panel
        self.series = list(series_order)               # [(cp_short, dir), ...]
        self.sidx = {s: i for i, s in enumerate(self.series)}
        self.wins = holiday_windows()
        self._hol_cache = {}

    def hol_info(self, d):
        if d not in self._hol_cache:
            hit = None
            for code, s0, s1, ln in self.wins:
                if s0 <= d <= s1:
                    hit = (code, (d - s0).days, ln)
                    break
            d_to = min([abs((s0 - d).days) for _, s0, _, _ in self.wins
                        if s0 > d] or [0])
            d_fr = min([abs((d - s1).days) for _, _, s1, _ in self.wins
                        if s1 < d] or [0])
            self._hol_cache[d] = (hit, min(d_to, 14), min(d_fr, 14))
        return self._hol_cache[d]

    def row(self, series, d, hist_log):
        """Feature row for date d given hist_log (log flows through d-1)."""
        x = np.zeros(N_FEAT)
        base = 0
        for j, lag in enumerate((1, 7, 14, 28)):
            x[base + j] = hist_log[-lag]
        base += 4
        w7 = hist_log[-7:]
        w28 = hist_log[-28:]
        x[base] = w7.mean(); x[base + 1] = w7.std(); x[base + 2] = w28.mean()
        base += 3
        x[base + d.weekday()] = 1                       # dow one-hot (7)
        base += 7
        x[base + d.month - 1] = 1                       # month one-hot (12)
        base += 12
        hit, d_to, d_fr = self.hol_info(d)
        if hit:
            code, off, ln = hit
            ph = phase_of(off, ln)
            x[base] = 1
            x[base + 1 + ("first", "mid", "last").index(ph)] = 1
            x[base + 4 + HOL_CODES.index(code)] = 1
        x[base + 11] = d_to / 14                        # signed dist to window
        x[base + 12] = d_fr / 14
        base += 13
        x[base + self.sidx[series]] = 1                 # series one-hot (16)
        base += 16
        x[base] = (d - REOPEN).days / 400               # slow trend
        return x

    def matrices(self, anomalies=None):
        """Training matrices over all series-days after a 28-day warm-up.

        anomalies: {(series): boolean mask by panel position} — LOF-flagged
        days are excluded from training (they are exactly the abnormal days
        the product wants to detect, not learn).
        """
        rows, ys, keys = [], [], []
        dates = self.p.index
        reopen_ts = pd.Timestamp(REOPEN)
        for s in self.series:
            col = logf(self.p[s].values)
            anom = anomalies.get(s) if anomalies else None
            for i in range(28, len(dates)):
                if dates[i] < reopen_ts:               # train on post-reopening only
                    continue
                if anom is not None and anom[i]:
                    continue
                d = dates[i].date()
                rows.append(self.row(s, d, col[:i]))
                ys.append(col[i])
                keys.append((s, d))
        return np.array(rows), np.array(ys), keys
