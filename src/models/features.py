# -*- coding: utf-8 -*-
"""Panel construction + feature engineering for the GBT forecaster.
Primary author: Lin Jingyu (series filters, 56-feature scheme, holiday calendar).


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

【中文说明 —— 特征工程（GBT 预测器的输入面板）】
作用：把原始日客流整理成"一行 = 某口岸某方向某天"的训练面板，并造出
56 个特征供梯度提升树使用。全部序列合并成一个面板训练（约 2.2 万行），
小口岸能通过"序列独热"借用大口岸的经验（借力）。
目标值：log(客流)。客流波动近似对数正态，且 log 空间的残差标准差
正好可以直接当产品里"不确定性区间"的 σ。
特征分七组（全部是普通浮点数，独热块为 0/1）：
  滞后      前 1/7/14/28 天的 log 客流（预测时超出实测期的滞后值
            用模型自己的预测递归补上）
  滚动      过去 7 天均值/标准差、过去 28 天均值
  日历      星期几（7 个）、月份（12 个）
  节假日    是否在假期内、假期阶段（头/中/尾）、假期代码（7 种）、
            距假期开始/结束的天数
  序列      口岸×方向独热（16 个）
  趋势      (日期 - 复关日) / 400 天
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
    # 中文：log 客流。封关日（台风、局部停运）公布的总数是 0，
    # 先把下限截到 1，log(1)=0，避免 log(0) 报错。
    return np.log(np.clip(np.asarray(flow, dtype=float), 1.0, None))


def load_panel(csv_path, keep_cps=None):
    """Wide panel: index=date, columns=MultiIndex(cp_full_name, direction).

    keep_cps: restrict to these control points (the product's 8 land /
    cross-boundary ports; None keeps every series that passes the filters).
    【中文】keep_cps 可限定只保留产品用到的 8 个陆路/跨境口岸；
    None = 保留所有能通过下面筛选条件的序列。
    """
    # 中文：读 ImmD 原始 CSV 并转成"行=日期、列=(口岸, 方向)"的宽面板
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Arrival / Departure": "Dir"})
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df["cp"] = df["Control Point"].str.strip()
    df = df[df["Dir"].isin(["Arrival", "Departure"])]
    p = df.pivot_table(index="Date", columns=["cp", "Dir"], values="Total", aggfunc="sum")
    p = p.sort_index().asfreq("D")                     # daily continuous index
    # 中文：asfreq("D") 把索引补成连续逐日——缺失日变 NaN，后面统一处理
    if keep_cps is not None:
        p = p[[c for c in p.columns if c[0] in keep_cps]]
    # series kept: >=45 days history and median >= 50 (drops STK/HG, see report)
    # 中文：序列准入门槛：复关后至少 45 天数据且中位日客流 >= 50
    # （太短/太小的序列学不出规律，会拖累合并训练——STK/HG 因此被剔除）
    keep = []
    for c in p.columns:
        s = p[c].dropna()
        recent = s[s.index >= pd.Timestamp(REOPEN)]
        if len(recent) >= 45 and recent.median() >= 50:
            keep.append(c)
    p = p[keep]
    # rare missing days inside a kept series -> forward-fill (documented)
    # 中文：保留序列内部偶发的缺日 → 用前一天的值补上（报告中已说明）
    p = p.ffill()
    return p


def holiday_windows():
    """HK + mainland cross-border holiday calendar 2023-2027 (tuple ranges)."""
    # 中文：港陆两地跨境相关假期日历 2023-2027（春节/清明/复活节/五一/
    # 国庆/圣诞/元旦）。窗口取"客流影响的整段"而非法定假日本身——
    # 例如国庆含前后周末拼接的长假出行期。
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
            # 中文：摊平成 (代码, 开始, 结束, 总天数) 列表，供 hol_info 查询
            out.append((code, start, end, (end - start).days + 1))
    return out


def phase_of(offset, length):
    # 中文：假期阶段——头 2 天算"头"，最后 3 天算"尾"（返程高峰），
    # 其余算"中"。不同阶段客流形态差异大（出程 vs 返程），模型需要区分。
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
    【中文】Panel 类持有宽面板，负责给任意(序列, 日期)生成模型输入行。
    row() 的 hist_log 是该序列截至"前一天"的 log 客流历史——训练时
    全部用实测值；预测未来时，越界部分用模型自己的预测递归补上。
    """

    def __init__(self, panel, series_order):
        self.p = panel
        self.series = list(series_order)               # [(cp_short, dir), ...]
        self.sidx = {s: i for i, s in enumerate(self.series)}
        self.wins = holiday_windows()
        self._hol_cache = {}

    def hol_info(self, d):
        # 中文：查询日期 d 的假期信息，结果缓存（同一日期只算一次）。
        # 返回 (命中窗口: (代码, 第几天, 总长) 或 None,
        #       距下一个假期开始的天数, 距上一个假期结束的天数)
        # 距离都封顶 14 天——太远的假期对当天客流没有影响。
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
        # 中文：按 FEATURE_NAMES 的顺序逐块填 56 维特征向量。
        # base 游标逐段右移，每块填完就跳到下一块的起点。
        x = np.zeros(N_FEAT)
        base = 0
        # 块 1：滞后特征——前 1/7/14/28 天的 log 客流
        for j, lag in enumerate((1, 7, 14, 28)):
            x[base + j] = hist_log[-lag]
        base += 4
        # 块 2：滚动统计——近 7 天均值/标准差、近 28 天均值
        w7 = hist_log[-7:]
        w28 = hist_log[-28:]
        x[base] = w7.mean(); x[base + 1] = w7.std(); x[base + 2] = w28.mean()
        base += 3
        x[base + d.weekday()] = 1                       # dow one-hot (7)
        # 中文：块 3-4：星期几独热（7）、月份独热（12）——捕捉周规律和季节性
        base += 7
        x[base + d.month - 1] = 1                       # month one-hot (12)
        base += 12
        # 块 5：假期特征——是否在假期内、阶段、代码、距窗口距离
        hit, d_to, d_fr = self.hol_info(d)
        if hit:
            # 中文：当天落在某个假期窗口内 → 点亮 hol 标志、阶段独热、假期代码独热
            code, off, ln = hit
            ph = phase_of(off, ln)
            x[base] = 1
            x[base + 1 + ("first", "mid", "last").index(ph)] = 1
            x[base + 4 + HOL_CODES.index(code)] = 1
        x[base + 11] = d_to / 14                        # signed dist to window
        x[base + 12] = d_fr / 14
        # 中文：不在假期内的日子用连续距离特征——假期前抢跑、假期后回落
        base += 13
        x[base + self.sidx[series]] = 1                 # series one-hot (16)
        # 中文：块 6：序列独热——让树知道这行属于哪个口岸×方向（借力的关键）
        base += 16
        x[base] = (d - REOPEN).days / 400               # slow trend
        # 中文：块 7：缓慢趋势——复关以来的长期恢复/衰减（400 天归一化）
        return x

    def matrices(self, anomalies=None):
        """Training matrices over all series-days after a 28-day warm-up.

        anomalies: {(series): boolean mask by panel position} — LOF-flagged
        days are excluded from training (they are exactly the abnormal days
        the product wants to detect, not learn).
        【中文】anomalies 是 LOF 标记的异常掩码——异常日被排除在训练外：
        它们是产品想要"识别"的异常，不是想要"学习"的规律。
        """
        # 中文：三层循环——每个序列、复关后的每一天（前 28 天作暖机期，
        # 因为最长的滞后/滚动特征需要 28 天历史才完整）
        rows, ys, keys = [], [], []
        dates = self.p.index
        reopen_ts = pd.Timestamp(REOPEN)
        for s in self.series:
            col = logf(self.p[s].values)
            anom = anomalies.get(s) if anomalies else None
            for i in range(28, len(dates)):
                if dates[i] < reopen_ts:               # train on post-reopening only
                    continue                           # 中文：只用复关后的数据训练
                if anom is not None and anom[i]:
                    continue                           # 中文：跳过 LOF 标记的异常日
                d = dates[i].date()
                rows.append(self.row(s, d, col[:i]))   # 中文：用截至 d-1 的历史造特征行
                ys.append(col[i])                      # 中文：目标 = 当天 log 客流
                keys.append((s, d))                    # 中文：记下键，供回测按日期对齐
        return np.array(rows), np.array(ys), keys
