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

【中文说明 —— 滚动起点回测（14 天递归预测的公正评估）】
作用：像"时光倒流"一样检验模型——只允许使用每个起点之前的数据
训练，再预测其后的 14 天，与真实值对比。这样评估不含任何未来信息。
对比三个选手：
  baseline  星期中位数 × 节假日系数（v0.2 产品模型，必须赢它）
  gbt       自写 NumPy 直方图梯度提升（gbt.py），递归策略
  sk        sklearn GradientBoostingRegressor，同特征（仅作基准）
协议：数据末端往前取 5 个互不重叠的 14 天窗口作测试段；每个起点
都用"该起点之前的全部数据"重新训练（扩张窗口），LOF 清洗也在
起点之前的数据上重算——完全模拟产品上线时的真实信息集。
指标（所有序列×天合并统计）：RMSE、MAPE、档位 exact 命中率、
档位 ±1 命中率。结果写入 report/backtest.json 供报告生成器使用。
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
# 中文：CPS8 = 产品覆盖的 8 个陆路/跨境口岸（与前端一致）
N_ORIGINS = 5    # 中文：回测起点数（数据末端往前 5 个不重叠的 14 天窗）
HORIZON = 14     # 中文：预测视野 = 14 天（与产品"未来两周"一致）


def weekday_medians(p, end_ts):
    """Per-series len-7 weekday medians from post-reopening data <= end."""
    # 中文：每个序列的"星期几中位数"基线（只用 end_ts 之前的数据，
    # 防泄漏）。缺的星期几用该序列总中位数兜底。
    out = {}
    for s in p.columns:
        v = p[s][p.index <= end_ts]
        v = v[v.index >= pd.Timestamp(REOPEN)]
        out[s] = v.groupby(v.index.dayofweek).median().reindex(range(7)).fillna(v.median()).values
    return out


def festive_multipliers(p, end_ts):
    """median(flow / weekday base) per (direction, code, phase), pre-origin."""
    # 中文：节假日系数 = 假期内客流相对"星期基线"的倍数，按
    # (方向, 假期代码, 阶段) 分组取中位数。样本 <10 的组退回 1.0
    # （系数不可靠时宁可不乘）。
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
    # 中文：每个序列的 P50/P85/P95 分位数（档位切分点），同样只用
    # 起点之前的数据——评估档位命中率时和产品用同一套切分定义。
    out = {}
    for s in p.columns:
        v = p[s][(p.index <= end_ts) & (p.index >= pd.Timestamp(REOPEN))]
        out[s] = {q: float(np.percentile(v, q)) for q in (50, 85, 95)}
    return out


def recursive_forecast(model, panel, p, origin_ts, horizon=HORIZON):
    """model-agnostic recursive forecast: predict one day, feed back as lags."""
    # 中文：递归多步预测（与模型无关：gbt 和 sk 走同一条路，公平）。
    # 预测下一天 → 把预测值追加进历史 → 再预测再下一天……
    # log 空间预测，最后 exp 回人数。
    preds = {}
    for s in p.columns:
        hist = logf(p[s][p.index <= origin_ts].values)  # 中文：起点前的真实历史
        out = []
        d0 = (origin_ts + pd.Timedelta(days=1)).date()
        for h in range(horizon):
            x = panel.row(s, d0 + dt.timedelta(days=h), hist)
            yhat = float(model.predict(x[None, :])[0])
            hist = np.append(hist, yhat)                # 中文：预测值回填为下一天的滞后特征
            out.append(float(np.exp(yhat)))             # 中文：log → 人数
        preds[s] = out
    return preds


def evaluate(preds, p, test_days, q):
    # 中文：把预测对齐到真实值，逐天记录（序列, 步长, 预测, 真实,
    # 预测档位, 真实档位）。真实值缺失的天跳过。
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
    # 中文：把逐天记录汇总成 4 个指标 + 样本数。
    # band_acc = 档位完全命中；band_acc1 = 档位误差 ≤1（产品语境更宽容）
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
    # 中文：主流程：读数据 → 5 个起点逐一（重算 LOF → 重训练 → 预测评估）
    # → 汇总写 backtest.json。每个起点都严格"只用过去"。
    p = load_panel("data/raw/immd_daily.csv", keep_cps=CPS8)
    panel = Panel(p, list(p.columns))
    from sklearn.ensemble import GradientBoostingRegressor  # benchmark ONLY
    # 中文：sklearn 只作基准对照（课程规则），不进产品

    all_rows = {"baseline": [], "gbt": [], "sk": []}
    for k in range(N_ORIGINS, 0, -1):
        origin_ts = p.index[-1] - pd.Timedelta(days=14 * k)   # 中文：从数据末端往前的第 k 个起点
        end_ts = origin_ts
        test_days = pd.date_range(origin_ts + pd.Timedelta(days=1), periods=HORIZON, freq="D")

        # ---- LOF flags on pre-origin data ----
        # 中文：只用起点之前的数据重算 LOF 异常标记——和真实上线时的信息集一致
        wm = weekday_medians(p, end_ts)
        anomalies = {}
        for s in p.columns:
            flow = p[s][p.index <= end_ts].values.astype(float)
            base = np.take(wm[s], np.arange(len(flow)) % 7)
            m, _ = flag_anomalies(flow, base)
            anomalies[s] = np.concatenate([m, np.zeros(len(p) - len(m), bool)])
        X, y, keys = panel.matrices(anomalies)
        dates_key = np.array([kd for _, kd in keys])
        keep = dates_key <= origin_ts.date()           # 中文：训练矩阵只留起点之前的行
        X, y = X[keep], y[keep]
        ntr = int(len(X) * 0.97)                       # 中文：97/3 切训练/验证（早停用）

        # ---- competitors ----
        # 中文：自写 GBT 用早停（400 棵上限、耐心 40）；sklearn 用相近配置
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
            # 中文：基线 = 星期中位数 ×（若逢假期）节假日系数。
            # 简单、可解释、无参数——机器学习模型必须赢过它才有存在感。
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
    # 中文：另给 gbt 按序列单独汇总（报告的 per-series RMSE 图用它）
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
