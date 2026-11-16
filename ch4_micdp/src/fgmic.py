"""Dependence scores with small, provable sensitivity (enhanced MIC-DP, "v2").

1. Fixed-grid MIC (FG-MIC)
   MIC restricted to grids whose cut points come from a candidate set fixed in advance
   (quantiles of the PUBLIC split): the x-axis may merge the candidate bins into any k contiguous
   columns (exact dynamic programme), the y-axis uses l public-quantile rows (regression) or the
   classes (classification), k * l <= B(n) = n ** alpha, and each grid's mutual information is
   divided by log min(k, l). Because the family of grids does not depend on the private data,

       |FG-MIC(D) - FG-MIC(D')| <= 3 (1 + ln n) / (n ln 2)      for neighbouring D, D' of size n.

   Proof sketch: for a fixed grid, I = ln n + (S_xy - S_x - S_y) / n with S = sum_cells c ln c.
   Replacing one record moves one unit of count in the joint table and in each margin; with
   g(c) = c ln c, 0 <= g(c+1) - g(c) <= 1 + ln n, so each S changes by at most 1 + ln n and
   |dI| <= 3 (1 + ln n) / n. Dividing by log min(k, l) >= ln 2 and taking a maximum over a fixed
   family keeps the bound. The d scores move together, so their L2 sensitivity is sqrt(d) times it.

2. Pearson and partial correlation from privatized moments
   z = [x, y01] in [0, 1]^(d+1). The released vector is (mean(z), upper triangle of mean(z z^T)).
   Every entry moves by at most 1/n when one record is replaced, so the L2 sensitivity is
   sqrt((d+1) + (d+1)(d+2)/2) / n. Correlations are post-processing of the noisy moments.
"""
from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------------------- cut points

def public_cuts(v_public: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior cut points at the 1/n_bins, ..., (n_bins-1)/n_bins quantiles of a public sample."""
    if n_bins < 2:
        return np.array([])
    qs = np.quantile(np.asarray(v_public, dtype=float), np.arange(1, n_bins) / n_bins)
    return np.unique(qs)


def bin_index(v: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Bin id 0..len(cuts) of every value: values <= cut go left."""
    return np.searchsorted(cuts, v, side="left")


def row_partitions(task: str, y_public: np.ndarray, B: float, l_max: int) -> list[tuple[int, np.ndarray]]:
    """Fixed row partitions: [(nominal number of rows, cut points)]. Classification: the classes."""
    if task == "classification":
        return [(2, np.array([0.5]))]
    out, seen = [], set()
    for l in range(2, max(2, min(l_max, int(math.floor(B / 2)))) + 1):
        cuts = public_cuts(y_public, l)
        key = tuple(np.round(cuts, 12))
        if len(cuts) < 1 or key in seen:
            continue
        seen.add(key)
        out.append((len(cuts) + 1, cuts))
    return out


# ----------------------------------------------------------------------------- core DP

def optimize_columns(cnt: np.ndarray, max_cols: int) -> np.ndarray:
    """Largest MI (nats) when merging the rows of cnt (candidate x-bins, in order) into at most k
    contiguous columns, for k = 0..max_cols. cnt: (bins x rows) counts. Empty bins are dropped first
    (merging an empty bin into a neighbour does not change any count)."""
    out = np.zeros(max_cols + 1)
    cnt = cnt[cnt.sum(axis=1) > 0]
    n = cnt.sum()
    if cnt.shape[0] < 2 or n == 0:
        return out
    rows_tot = cnt.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = rows_tot / n
        hq = -np.nansum(np.where(pr > 0, pr * np.log(pr), 0.0))
    P = cnt.shape[0]
    cum = np.vstack([np.zeros(cnt.shape[1]), np.cumsum(cnt, axis=0)])
    N = cum[None, :, :] - cum[:, None, :]
    tot = N.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where(N > 0, N * np.log(N / tot[..., None]), 0.0).sum(axis=-1)
    W = np.where(np.triu(np.ones((P + 1, P + 1), dtype=bool)), W, -np.inf)
    G = W[0].copy()
    for k in range(2, max_cols + 1):
        G = np.max(G[:, None] + W, axis=0)
        out[k] = hq + G[P] / n
    return np.maximum(out, 0.0)


def fgmic(x: np.ndarray, y: np.ndarray, x_cuts: np.ndarray, rows: list[tuple[int, np.ndarray]],
          alpha: float = 0.55) -> float:
    """Fixed-grid MIC of one feature. x_cuts / rows come from public data only."""
    n = x.size
    B = max(n ** alpha, 4.0)
    xb = bin_index(x, x_cuts)
    gx = len(x_cuts) + 1
    best = 0.0
    for l, ycuts in rows:
        kmax = min(gx, int(math.floor(B / l)))
        if kmax < 2:
            continue
        yr = bin_index(y, ycuts)
        cnt = np.zeros((gx, l))
        np.add.at(cnt, (xb, yr), 1.0)
        I = optimize_columns(cnt, kmax)
        for k in range(2, kmax + 1):
            best = max(best, I[k] / math.log(min(k, l)))
    return float(min(best, 1.0))


def fgmic_scores(X: np.ndarray, y: np.ndarray, x_cuts: list[np.ndarray], rows, alpha: float = 0.55) -> np.ndarray:
    return np.array([fgmic(X[:, j], y, x_cuts[j], rows, alpha) for j in range(X.shape[1])])


def fgmic_sensitivity(n: int) -> float:
    """Per-score sensitivity bound 3 (1 + ln n) / (n ln 2)."""
    return 3.0 * (1.0 + math.log(n)) / (n * math.log(2.0))


# ----------------------------------------------------------------------------- moments

def moment_vector(X01: np.ndarray, y01: np.ndarray) -> np.ndarray:
    Z = np.column_stack([X01, y01])
    n, p = Z.shape
    iu = np.triu_indices(p)
    return np.concatenate([Z.mean(axis=0), ((Z.T @ Z) / n)[iu]])


def moment_sensitivity(n: int, d: int) -> float:
    p = d + 1
    return math.sqrt(p + p * (p + 1) / 2) / n


def correlations_from_moments(v: np.ndarray, d: int, ridge: float = 1e-6):
    """|Pearson r_j|, |partial corr pc_j| and the PSD-projected covariance from a moment vector."""
    p = d + 1
    m = v[:p]
    M = np.zeros((p, p))
    M[np.triu_indices(p)] = v[p:]
    M = M + M.T - np.diag(np.diag(M))
    C = M - np.outer(m, m)
    ev, U = np.linalg.eigh((C + C.T) / 2)
    ev = np.maximum(ev, 1e-10)
    C = (U * ev) @ U.T
    sd = np.sqrt(np.diag(C))
    r = C[:d, d] / (sd[:d] * sd[d])
    P = np.linalg.inv(C + ridge * np.eye(p))
    pc = -P[:d, d] / np.sqrt(np.diag(P)[:d] * P[d, d])
    clean = lambda a: np.clip(np.abs(np.where(np.isfinite(a), a, 0.0)), 0.0, 1.0)
    return clean(r), clean(pc), C


def scale_target(y: np.ndarray, task: str, lo: float, hi: float) -> np.ndarray:
    """Target in [0, 1] for the moment release (classification: 0/1; regression: public range)."""
    if task == "classification":
        return np.asarray(y, dtype=float)
    return (np.clip(y, lo, hi) - lo) / (hi - lo)
