"""ULDP: TOPSIS selection of the privacy budget, exactly as Chapter 6 (Eqs. 6.6-6.12).

Decision matrices are hard-coded from Chapter 6, Table 6.1 (not recomputed).
All functions are vectorized over many weight vectors.
"""
from __future__ import annotations

import numpy as np

CANDIDATES = np.array([0.1, 0.3, 1.0, 5.0, 10.0])
LP = {
    "linear": np.array([1.0, 0.979798, 0.909091, 0.505051, 0.0]),
    "exponential": np.array([1.0, 0.818722, 0.406540, 0.007397, 0.0]),
}
LA = {
    "adult": np.array([0.109371, 0.168731, 0.478616, 0.746903, 0.754148]),
    "cdc": np.array([0.0, 0.329227, 0.586387, 0.891900, 0.918694]),
}
SETTINGS = [("adult", "linear"), ("adult", "exponential"), ("cdc", "linear"), ("cdc", "exponential")]
UNDEFINED = -1          # index returned when every closeness coefficient is 0/0


def matrix(dataset: str, lp_map: str) -> np.ndarray:
    """Decision matrix M (5 candidates x 3 criteria: LP, LA, CR), CR = 1 for every candidate."""
    return np.column_stack([LP[lp_map], LA[dataset], np.ones(len(CANDIDATES))])


def normalize(M: np.ndarray) -> np.ndarray:
    """Divide each column by its Euclidean norm; a column that is zero for every candidate stays zero."""
    norms = np.sqrt((M ** 2).sum(axis=0))
    R = np.zeros_like(M, dtype=float)
    nz = norms > 0
    R[:, nz] = M[:, nz] / norms[nz]
    return R


def closeness(M: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Closeness coefficients C (n x K) for weight vectors W (n x 3, or 3). NaN where d+ + d- = 0."""
    W = np.atleast_2d(np.asarray(W, dtype=float))
    R = normalize(M)
    V = W[:, None, :] * R[None, :, :]                       # (n, K, 3)
    A_pos = V.max(axis=1, keepdims=True)
    A_neg = V.min(axis=1, keepdims=True)
    d_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=2))
    d_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=2))
    den = d_pos + d_neg
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(den > 0, d_neg / np.where(den > 0, den, 1.0), np.nan)
    return C


def select(M: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of eps* = argmax C (ties -> smallest eps, i.e. the first index) and the C matrix.
    Returns UNDEFINED (-1) where every C is undefined (0/0)."""
    C = closeness(M, W)
    undefined = np.all(np.isnan(C), axis=1)
    idx = np.argmax(np.where(np.isnan(C), -np.inf, C), axis=1)
    idx = np.where(undefined, UNDEFINED, idx)
    return idx, C


def privacy_share(W: np.ndarray) -> np.ndarray:
    """pi = w_LP / (w_LP + w_LA), with pi = 0.5 when w_LP + w_LA = 0."""
    W = np.atleast_2d(np.asarray(W, dtype=float))
    s = W[:, 0] + W[:, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, W[:, 0] / np.where(s > 0, s, 1.0), 0.5)
