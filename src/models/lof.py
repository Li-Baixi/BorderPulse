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

【中文说明 —— 局部离群因子 LOF（异常日清洗）】
作用：在训练预测模型之前，先找出"不像正常日子"的异常天（台风封关、
系统故障、数据毛刺），让模型只学习正常出行规律。
原理（k = 邻居数）：
  kdist(p)   p 到第 k 个近邻的距离
  reach      可达距离 = max(邻居的 kdist, 实际距离) —— 抹平近邻自身
             密度带来的偏差
  lrd(p)     局部可达密度 = 1 / 邻居可达距离均值
  LOF(p)     邻居密度与自己密度之比 —— 落在密集区 ≈ 1（正常），
             落在稀疏区 >> 1（异常）
要点：每个"口岸×方向"序列单独跑 LOF（不合并）——各口岸客流相差
可达 20 倍，合并后距离会被大口岸完全支配。调用方已按序列标准化特征，
本模块只负责算 LOF 分数。
"""
import numpy as np


class LOF:
    def __init__(self, k=20):
        self.k = k

    def fit_predict(self, X):
        """Return LOF scores for the rows of X (n x d). O(n^2) distances —
        fine here: n <= ~1,400 days per series."""
        # 中文：一次性算出所有点的 LOF 分数。两两距离是 O(n^2)，
        # 但每个序列最多约 1,400 天，完全跑得动（向量化后秒级）。
        X = np.asarray(X, dtype=float)
        n = len(X)
        k = min(self.k, n - 1)
        # 中文：广播算 n×n 欧氏距离矩阵；对角线填 +inf，排除"自己是自己的邻居"
        d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)

        nn = np.argsort(d, axis=1)[:, :k]                # k neighbours
        kdist = d[np.arange(n)[:, None], nn][:, -1]      # distance to k-th
        # reachability distance to each neighbour
        # 中文：nn = 每个点的 k 个近邻；kdist = 到第 k 近邻的距离；
        # reach = 可达距离（见模块说明）——取"实际距离"与"邻居自身 kdist"的较大者
        reach = np.maximum(d[np.arange(n)[:, None], nn], kdist[nn])
        lrd = 1.0 / (reach.mean(axis=1) + 1e-12)         # 中文：局部可达密度（加小量防除零）
        lof = (lrd[nn] / lrd[:, None]).mean(axis=1)      # 中文：邻居密度/自己密度 的均值 = LOF
        return lof


def anomalous_features(flow, weekday_base):
    """Per-day feature vector for anomaly detection within one series.

    flow          : array of daily totals (chronological)
    weekday_base  : len-7 array of weekday medians for this series

    Features (all scale-free within the series):
      x0  log(flow)                       overall level (drops/outages)
      x1  log(flow / weekday baseline)    residual vs own weekly pattern
      x2  same residual, 7-day median     slow shifts (suspension creep)
    【中文】三个特征都在序列内部无量纲，便于跨序列比较：
      x0  log(客流)                     总体水平（捕捉骤降/停摆）
      x1  log(客流/星期基线)             相对自身周规律的残差
      x2  该残差的 7 天滚动中位数        缓慢漂移（如口岸渐进式缩减）
    """
    # 中文：星期基线是该序列每个星期几的客流中位数，先把绝对量级除掉
    dow = np.arange(len(flow)) % 7
    ratio = flow / np.take(weekday_base, dow)
    resid = np.log(np.clip(ratio, 1e-3, None))          # 中文：log 残差（clip 防极端值）
    # 中文：截至第 i 天（含）最近 7 天残差的中位数——对单日毛刺不敏感
    med7 = np.array([np.median(resid[max(0, i - 6):i + 1]) for i in range(len(resid))])
    return np.column_stack([np.log(np.clip(flow, 1, None)), resid, med7])


def flag_anomalies(flow, weekday_base, k=20, thresh=1.3, contamination=0.015):
    """Boolean mask of anomalous days for one series + their LOF scores.

    Two gates: score above `thresh` AND in the top `contamination` fraction
    of the series' own scores — the absolute threshold alone is not
    comparable across series, the rank gate alone would flag a fixed quota
    even in a perfectly clean series. Together: ~1.5% worst days that also
    look sparse to their neighbours.
    【中文】双闸门设计（两者必须同时满足才标记为异常）：
      闸门一  LOF 分数 > 1.3 —— 绝对意义上"够稀疏"；
      闸门二  排进本序列自身分数的前 1.5% —— 相对意义上"够反常"。
      只用闸门一：各序列分数尺度不一，无法跨序列比较；
      只用闸门二：哪怕序列完全干净也会强行凑满配额。
      两者结合 = 只标记"既稀疏又排在自家最差 1.5%"的那几天。
    """
    X = anomalous_features(flow, weekday_base)
    lof = LOF(k=k).fit_predict(X)                       # 中文：算全序列 LOF 分数
    quota = max(1, int(len(flow) * contamination))       # 中文：前 1.5% 至少 1 天
    cut = np.sort(lof)[-quota]                           # 中文：配额线上的分数
    # 中文：返回（异常掩码, 全部分数）。分数也返回，供报告画分布图
    return (lof > thresh) & (lof >= cut), lof
