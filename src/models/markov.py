# -*- coding: utf-8 -*-
"""First-order Markov chain over the 4 congestion states — own NumPy implementation.
Primary author: Li Baixi (state chain, transition fitting).


States are the product's own congestion bands (1 Smooth / 2 Normal / 3 Busy /
4 Packed), cut at each series' own P50/P85/P95 daily flow. The chain gives a
probabilistic second opinion on the GBT point forecast: starting from today's
OBSERVED state, the h-step distribution is row s of P^h.

Smoothing: one pseudo-count per transition (Laplace) — with ~1,300 usable days
per series the raw counts are mostly full, but sparse CPs (e.g. MKT) benefit.

【中文说明 —— 马尔可夫拥挤状态链】
作用：给 GBT 点预测提供一个"第二意见"。它不预测具体人数，只回答：
  "从今天观测到的拥挤档位出发，未来第 h 天落在各档位的概率是多少？"
做法：
  1) 把每个口岸×方向序列的日客流按自身历史分位数 P50/P85/P95 切成 4 档
     （1 畅顺 / 2 正常 / 3 繁忙 / 4 爆满）——用自身分位数而非全局阈值，
     是因为各口岸体量相差几十倍，统一阈值会让小口岸永远"畅顺"。
  2) 统计历史"今天 a 档→明天 b 档"的频次，得到 4×4 转移矩阵（每行和=1），
     加拉普拉斯平滑（每个转移先垫 1 个伪计数）防止稀疏序列出零。
  3) 从今天实际观测档位出发，逐步乘转移矩阵得到未来每天的档位分布。
前端展示：日历详情弹层里的四段彩色概率条（畅顺37% 正常42% …）就是它。
"""
import numpy as np


def state_of(flow_value, q):
    """Map a daily flow to band 1..4 using quantile cut points."""
    # 中文：把一天的客流转成 1..4 档。q 是该序列的 P50/P85/P95 三个分位数，
    # 依次比较即可（比 Python if-else 链更向量化、更快）。
    return 1 + (flow_value > q[50]) + (flow_value > q[85]) + (flow_value > q[95])


def fit_transition(states, n_states=4, alpha=1.0):
    """Row-stochastic transition matrix with Laplace smoothing."""
    # 中文：由历史档位序列统计转移矩阵。C 先全部填 alpha（伪计数），
    # 再对每个相邻日对 (今天, 明天) 加 1；最后每行除以行和 → 行随机矩阵。
    C = np.full((n_states, n_states), alpha)
    for a, b in zip(states[:-1], states[1:]):
        C[a - 1, b - 1] += 1
    return C / C.sum(axis=1, keepdims=True)


def forecast_dist(P, start_state, horizon):
    """(horizon, 4) array: state distribution h steps ahead for each h."""
    # 中文：从 start_state 出发递推 h = 1..horizon 天的档位分布。
    # p 是当前分布行向量，每乘一次 P 就往前走一天；out 按天收集结果。
    out = np.zeros((horizon, 4))
    p = np.eye(4)[start_state - 1]          # 中文：one-hot 起点（今天确定在某档）
    for h in range(horizon):
        p = p @ P
        out[h] = p
    return out


def stationary(P):
    """Stationary distribution (for the report: does the chain mix to the
    unconditional band frequencies?). Power iteration."""
    # 中文：平稳分布——反复乘 P 直到收敛（幂迭代）。
    # 用于报告检查：链长期混合后是否回到无条件的档位频率。
    p = np.ones(4) / 4
    for _ in range(200):
        p = p @ P
    return p
