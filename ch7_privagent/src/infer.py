"""Inference of weights from answers: vote accumulation (Chapter 7, Eqs. 7.1-7.3) and Bayesian
posterior mean (ablation)."""
from __future__ import annotations

import numpy as np
from scipy.special import log_expit, logsumexp

from .dialogue import PAIRS


def _deltas(contrast: np.ndarray, used: np.ndarray, kappa_tab: np.ndarray) -> np.ndarray:
    kap = kappa_tab[contrast] * used
    tot = kap.sum(axis=1, keepdims=True)
    return np.where(tot > 0, kap / np.where(tot > 0, tot, 1.0), 0.0)


def vote_accumulate(pair, contrast, answer, used, kappa_tab) -> np.ndarray:
    """Accumulated (unprojected) w_hat: start at (1/3, 1/3, 1/3); for each used answer add
    Delta_p = kappa_p / sum(kappa) to the winning criterion and subtract it from the losing one."""
    N, S = pair.shape
    w = np.full((N, 3), 1.0 / 3.0)
    delta = _deltas(contrast, used, kappa_tab)
    pj = np.array([p[0] for p in PAIRS])[pair]
    pk = np.array([p[1] for p in PAIRS])[pair]
    win = np.where(answer == 1, pj, pk)
    lose = np.where(answer == 1, pk, pj)
    rows = np.arange(N)
    for s in range(S):
        w[rows, win[:, s]] += delta[:, s]
        w[rows, lose[:, s]] -= delta[:, s]
    return w


def project(w: np.ndarray) -> np.ndarray:
    """w_j = max(w_j, 0) / sum_k max(w_k, 0)."""
    p = np.maximum(w, 0.0)
    return p / p.sum(axis=1, keepdims=True)


def vote(d: dict, kappa_tab) -> np.ndarray:
    return project(vote_accumulate(d["pair"], d["contrast"], d["answer"], d["used"], kappa_tab))


def bayes(d: dict, h_tab, l_tab, draws: np.ndarray, tau_a: float, chunk: int = 500) -> np.ndarray:
    """Posterior mean under the Dirichlet(1,1,1) prior (represented by `draws`, prior samples) and the
    logistic response likelihood with the analyst's tau, by self-normalized importance sampling."""
    N = d["pair"].shape[0]
    D = draws.shape[0]
    # log-likelihood of each (pair, contrast, answer) for each draw: (3, 3, 2, D)
    T = np.empty((3, 3, 2, D))
    for q, (j, k) in enumerate(PAIRS):
        diff = draws[:, j] - draws[:, k]
        for c in range(3):
            z = diff * (h_tab[c] - l_tab[c]) / tau_a
            T[q, c, 1] = log_expit(z)        # chose A
            T[q, c, 0] = log_expit(-z)       # chose B
    Tm = T.reshape(18, D)
    # counts of each (pair, contrast, answer) per user
    code = (d["pair"] * 3 + d["contrast"]) * 2 + d["answer"]
    counts = np.zeros((N, 18))
    for s in range(code.shape[1]):
        np.add.at(counts, (np.arange(N), code[:, s]), d["used"][:, s].astype(float))
    out = np.empty((N, 3))
    for a in range(0, N, chunk):
        L = counts[a:a + chunk] @ Tm                                  # (n, D)
        L -= logsumexp(L, axis=1, keepdims=True)
        out[a:a + chunk] = np.exp(L) @ draws
    return out
