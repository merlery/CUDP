"""Simulated users: true weights, logistic (Luce) response model, anchored slider."""
from __future__ import annotations

import numpy as np
from scipy.special import expit


def draw_weights(rng: np.random.Generator, alpha, n: int) -> np.ndarray:
    """True weights w* ~ Dirichlet(alpha), n users (columns LP, LA, CR)."""
    return rng.dirichlet(np.asarray(alpha, dtype=float), size=n)


def choice_prob(W: np.ndarray, pair: tuple[int, int], h: float, l: float, tau: float) -> np.ndarray:
    """P(choose option A) for a prompt on pair (j, j'), A = (m_j, m_j') = (h, l), B = (l, h), third
    criterion equal in both: sigma((U(A) - U(B)) / tau) with U(A) - U(B) = (w_j - w_j')(h - l)."""
    j, k = pair
    return expit((W[:, j] - W[:, k]) * (h - l) / tau)


def answer(W: np.ndarray, pair, h, l, tau, u: np.ndarray) -> np.ndarray:
    """1 = the user chose option A (favoring j), from uniform draws u (common random numbers)."""
    return (u < choice_prob(W, pair, h, l, tau)).astype(np.int8)


def anchored_slider(W: np.ndarray, lam: float, anchor, eta: np.ndarray) -> np.ndarray:
    """w_rep = lam * a + (1 - lam) * (w* + eta), clipped at 0 and renormalized (uniform if all zero)."""
    a = np.asarray(anchor, dtype=float)[None, :]
    rep = lam * a + (1.0 - lam) * (W + eta)
    rep = np.maximum(rep, 0.0)
    s = rep.sum(axis=1, keepdims=True)
    out = np.where(s > 0, rep / np.where(s > 0, s, 1.0), 1.0 / 3.0)
    return out
