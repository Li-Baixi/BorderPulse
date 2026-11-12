# -*- coding: utf-8 -*-
"""Local Outlier Factor (Breunig et al., SIGMOD 2000) — own NumPy implementation.
Primary author: Lin Jingyu (anomaly cleaning / two-gate design).


Used to flag anomalous border-crossing days (typhoon closures, system outages,
data glitches) before fitting the forecaster, so the model trains on days that
reflect normal travel behaviour.

Run per control-point x direction series (not pooled): absolute flow levels
differ by 20x across control points and would dominate the distance metric.
The caller standardises features per series; this module only computes LOF.

Algorithm (k = neighbourhood size):
  kdist(p)      distance from p to its k-th nearest neighbour
  reach(p, o)   max(kdist(o), d(p, o))          "reachability distance"
  lrd(p)        1 / mean_o reach(p, o)  over the k neighbours of p
  LOF(p)        mean_o lrd(o) / lrd(p)          ~1 inside a dense region,
                                                >> 1 in a sparse one
"""
import numpy as np


class LOF:
    def __init__(self, k=20):
        self.k = k

    def fit_predict(self, X):
        """Return LOF scores for the rows of X (n x d). O(n^2) distances —
        fine here: n <= ~1,400 days per series."""
        X = np.asarray(X, dtype=float)
        n = len(X)
        k = min(self.k, n - 1)
        d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)

        nn = np.argsort(d, axis=1)[:, :k]                # k neighbours
        kdist = d[np.arange(n)[:, None], nn][:, -1]      # distance to k-th
        # reachability distance to each neighbour
        reach = np.maximum(d[np.arange(n)[:, None], nn], kdist[nn])
        lrd = 1.0 / (reach.mean(axis=1) + 1e-12)
        lof = (lrd[nn] / lrd[:, None]).mean(axis=1)
        return lof


def anomalous_features(flow, weekday_base):
    """Per-day feature vector for anomaly detection within one series.

    flow          : array of daily totals (chronological)
    weekday_base  : len-7 array of weekday medians for this series

    Features (all scale-free within the series):
      x0  log(flow)                       overall level (drops/outages)
      x1  log(flow / weekday baseline)    residual vs own weekly pattern
      x2  same residual, 7-day median     slow shifts (suspension creep)
    """
    dow = np.arange(len(flow)) % 7
    ratio = flow / np.take(weekday_base, dow)
    resid = np.log(np.clip(ratio, 1e-3, None))
    med7 = np.array([np.median(resid[max(0, i - 6):i + 1]) for i in range(len(resid))])
    return np.column_stack([np.log(np.clip(flow, 1, None)), resid, med7])


def flag_anomalies(flow, weekday_base, k=20, thresh=1.3, contamination=0.015):
    """Boolean mask of anomalous days for one series + their LOF scores.

    Two gates: score above `thresh` AND in the top `contamination` fraction
    of the series' own scores — the absolute threshold alone is not
    comparable across series, the rank gate alone would flag a fixed quota
    even in a perfectly clean series. Together: ~1.5% worst days that also
    look sparse to their neighbours.
    """
    X = anomalous_features(flow, weekday_base)
    lof = LOF(k=k).fit_predict(X)
    quota = max(1, int(len(flow) * contamination))
    cut = np.sort(lof)[-quota]
    return (lof > thresh) & (lof >= cut), lof
