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
"""
import numpy as np

MU = 2.0            # service rate per counter (per minute)
PEAK_SHARE = 0.12   # share of daily flow in the busiest hour
TRIALS = 3000
CAP = 120.0         # minutes — physical saturation cap (triangular max)


def erlang_c(c, rho):
    """P(all servers busy) for M/M/c; rho = offered erlangs / c. Iterative, O(c)."""
    a = rho * c
    s, term = 1.0, 1.0
    for i in range(1, c):
        term *= a / i
        s += term
    last = term * a / c
    return last / ((1 - rho) * s + last)


def counter_count(q90):
    return max(4, int(np.ceil((q90 * PEAK_SHARE / 60 / MU) / 0.95)))


def simulate(mu, sigma, c, trials=TRIALS, rng=None):
    """Vectorised draws of peak-hour queueing time (minutes)."""
    rng = rng or np.random.default_rng(7005)
    flow = mu * np.exp(rng.normal(0, max(sigma, 1e-6), trials))
    lam = flow * PEAK_SHARE / 60
    rho = lam / (c * MU)
    wait = np.empty(trials)
    sat = rho >= 0.97
    # unsaturated: steady-state mean waiting time (loop: erlang_c is scalar)
    for i in np.where(~sat)[0]:
        wait[i] = erlang_c(c, rho[i]) / (c * MU - lam[i])
    # saturated: triangular dwell 20/45/120 min via inverse CDF
    u = rng.uniform(0, 1, trials)
    tri = np.where(u < 0.25, 20 + 50 * np.sqrt(u), CAP - 86.6 * np.sqrt(1 - u))
    wait[sat] = np.minimum(CAP, np.maximum(20, tri[sat]))
    return wait


def risk(mu, sigma, q90, t_line=30.0, trials=TRIALS):
    """Summary for one (control point, day): P(wait>t), median, P95."""
    c = counter_count(q90)
    w = simulate(mu, sigma, c, trials)
    return {
        "counters": c,
        "p_over": float((w > t_line).mean()),
        "median": float(np.median(w)),
        "p95": float(np.percentile(w, 95)),
    }
