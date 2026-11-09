# -*- coding: utf-8 -*-
"""First-order Markov chain over the 4 congestion states — own NumPy implementation.

States are the product's own congestion bands (1 Smooth / 2 Normal / 3 Busy /
4 Packed), cut at each series' own P50/P85/P95 daily flow. The chain gives a
probabilistic second opinion on the GBT point forecast: starting from today's
OBSERVED state, the h-step distribution is row s of P^h.

Smoothing: one pseudo-count per transition (Laplace) — with ~1,300 usable days
per series the raw counts are mostly full, but sparse CPs (e.g. MKT) benefit.
"""
import numpy as np


def state_of(flow_value, q):
    """Map a daily flow to band 1..4 using quantile cut points."""
    return 1 + (flow_value > q[50]) + (flow_value > q[85]) + (flow_value > q[95])


def fit_transition(states, n_states=4, alpha=1.0):
    """Row-stochastic transition matrix with Laplace smoothing."""
    C = np.full((n_states, n_states), alpha)
    for a, b in zip(states[:-1], states[1:]):
        C[a - 1, b - 1] += 1
    return C / C.sum(axis=1, keepdims=True)


def forecast_dist(P, start_state, horizon):
    """(horizon, 4) array: state distribution h steps ahead for each h."""
    out = np.zeros((horizon, 4))
    p = np.eye(4)[start_state - 1]
    for h in range(horizon):
        p = p @ P
        out[h] = p
    return out


def stationary(P):
    """Stationary distribution (for the report: does the chain mix to the
    unconditional band frequencies?). Power iteration."""
    p = np.ones(4) / 4
    for _ in range(200):
        p = p @ P
    return p
