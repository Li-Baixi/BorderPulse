# -*- coding: utf-8 -*-
"""Monte Carlo queue simulation for a border-control hall — own NumPy implementation.
Primary author: Li Baixi (Erlang-C core, Monte Carlo layer).


Model (identical constants to the live PWA so the two stay comparable):
  * daily flow F ~ LogNormal(ln mu, sigma)   (mu from the forecaster,
    sigma = residual spread of the model that produced mu)
  * the busiest hour carries PEAK_SHARE of the day -> arrival rate
    lambda = F * PEAK_SHARE / 60 per minute
  * c counters, each M/M with mean service 30 s (MU = 2 per minute)
  * if rho = lambda / (c MU) < 0.97: expected wait = Erlang-C / (c MU - lambda)
    (steady-state M/M/c mean waiting time, closed form)
  * if rho >= 0.97 the linear steady-state formula diverges while the real
    hall saturates: model the dwell as a triangular distribution
    (20 / 45 / 120 min) -> capped queueing
  * counters c calibrated so the historical P90 day runs at 95% utilisation:
    c = ceil((q90 * PEAK_SHARE / 60 / MU) / 0.95), min 4

Outputs P(wait > 30 min), median and P95 wait over TRIALS draws.

【中文说明 —— 排队风险模拟（Erlang-C + 蒙特卡洛）】
作用：把"那天预计有多少人"翻译成旅客真正关心的决策变量——"要排多久"。
流程：
  1) 日客流 F 不是定值而是对数正态随机变量（均值来自 GBT 预测，
     波动 σ 取该模型的残差），先抽样 3000 个"可能的当天客流"。
  2) 最忙 1 小时约占全天 12%（PEAK_SHARE）→ 换算成每分钟到达率 λ。
  3) 柜台数 c 用该口岸历史 P90 日负荷反推：让 P90 那天跑到 95% 利用率。
  4) 利用率 ρ = λ/(c·MU) 不足 0.97 时，用 M/M/c 稳态公式（Erlang-C）
     算平均等待；ρ ≥ 0.97 时稳态公式发散，而现实中大厅只会"饱和"，
     改用三角分布（20/45/120 分钟）封顶模拟滞留时间。
  5) 汇总 3000 次抽样 → P(排队>30分钟)、中位等待、P95 等待。
前端展示：排队页在浏览器里跑的正是同一套逻辑（同常数、同种子，可复现）。
"""
import numpy as np

MU = 2.0            # service rate per counter (per minute)      中文：每柜台每分钟服务 2 人（30 秒/人）
PEAK_SHARE = 0.12   # share of daily flow in the busiest hour    中文：最忙 1 小时占全天客流的 12%
TRIALS = 3000       #                                             中文：蒙特卡洛抽样次数
CAP = 120.0         # minutes — physical saturation cap (triangular max)  中文：饱和滞留时间上限（分钟）


def erlang_c(c, rho):
    """P(all servers busy) for M/M/c; rho = offered erlangs / c. Iterative, O(c)."""
    # 中文：Erlang-C 公式——M/M/c 排队系统中"所有柜台都忙"的概率。
    # 用递推式算泊松项之和（避免大数阶乘），复杂度 O(c)。
    a = rho * c
    s, term = 1.0, 1.0
    for i in range(1, c):
        term *= a / i
        s += term
    last = term * a / c
    return last / ((1 - rho) * s + last)


def counter_count(q90):
    # 中文：由历史 P90 日客流反推柜台数——让 P90 那天达到 95% 利用率，
    # 至少 4 个柜台（小口岸保底）。
    return max(4, int(np.ceil((q90 * PEAK_SHARE / 60 / MU) / 0.95)))


def simulate(mu, sigma, c, trials=TRIALS, rng=None):
    """Vectorised draws of peak-hour queueing time (minutes)."""
    # 中文：一次完整的蒙特卡洛抽样，返回 trials 个"当天排队分钟数"。
    rng = rng or np.random.default_rng(7005)   # 中文：固定种子 → 演示可复现
    flow = mu * np.exp(rng.normal(0, max(sigma, 1e-6), trials))  # 中文：抽日客流（对数正态）
    lam = flow * PEAK_SHARE / 60               # 中文：→ 最忙小时到达率（每分钟）
    rho = lam / (c * MU)                       # 中文：利用率
    wait = np.empty(trials)
    sat = rho >= 0.97                          # 中文：哪些抽样日会饱和
    # unsaturated: steady-state mean waiting time (loop: erlang_c is scalar)
    # 中文：未饱和的抽样日用 Erlang-C 稳态平均等待公式
    for i in np.where(~sat)[0]:
        wait[i] = erlang_c(c, rho[i]) / (c * MU - lam[i])
    # saturated: triangular dwell 20/45/120 min via inverse CDF
    # 中文：饱和的抽样日用三角分布（20/45/120 分钟）逆 CDF 抽滞留时间，并裁剪到 [20, CAP]
    u = rng.uniform(0, 1, trials)
    tri = np.where(u < 0.25, 20 + 50 * np.sqrt(u), CAP - 86.6 * np.sqrt(1 - u))
    wait[sat] = np.minimum(CAP, np.maximum(20, tri[sat]))
    return wait


def risk(mu, sigma, q90, t_line=30.0, trials=TRIALS):
    """Summary for one (control point, day): P(wait>t), median, P95."""
    # 中文：对"某口岸某天"的完整风险摘要——前端三块数字瓦片对应这里的
    # p_over / median / p95（排队超 30 分钟概率 / 典型等待 / 基本不会超过）。
    c = counter_count(q90)
    w = simulate(mu, sigma, c, trials)
    return {
        "counters": c,
        "p_over": float((w > t_line).mean()),
        "median": float(np.median(w)),
        "p95": float(np.percentile(w, 95)),
    }
