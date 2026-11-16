"""Maximal information coefficient, implemented from the published algorithms (numpy only).

No minepy / mictools code is used here. References:

* D. N. Reshef et al., "Detecting novel associations in large data sets", Science 334 (2011),
  Supplementary Online Material, Algorithms 1-5: EquipartitionYAxis, GetClumpsPartition,
  GetSuperclumpsPartition, OptimizeXAxis, ApproxMaxMI / ApproxCharacteristicMatrix.
* Y. A. Reshef, D. N. Reshef et al., "Measuring dependence powerfully and equitably", JMLR 17 (2016):
  the estimator MIC_e = max over the *equicharacteristic* matrix, whose (k, l) entry is the largest
  mutual information over k-by-l grids in which the axis with MORE bins is equipartitioned,
  normalised by log min(k, l).

Conventions (fixed in config.yaml):
  B(n) = max(n ** alpha, 4); grids satisfy k * l <= B, k, l >= 2.
  For every number of equipartitioned rows l in 2..floor(B/2) and each orientation (X optimised /
  Y equipartitioned, and the transpose), the optimised axis starts from at most floor(c * floor(B/l))
  superclumps and may use up to K(l) columns: K(l) = min(l, floor(B/l)) for MIC_e (only k <= l
  enters the equicharacteristic matrix) and K(l) = floor(B/l) for the 2011 approximation.
  A grid's mutual information is normalised by log min(k, q), where q is the number of rows the
  equipartition actually produced (q < l when ties force bins to merge): the grid found really is
  a k-by-q grid. Natural logarithms throughout (MIC is base-free because of the normalisation).
  For k = l both orientations are admissible and the larger value is kept.
"""
from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------------------- partitions

def _equipartition_groups(sizes: np.ndarray, n_bins: int) -> np.ndarray:
    """EquipartitionYAxis on consecutive groups of tied points (Reshef 2011, SOM Alg. 3).

    sizes: number of points in each group of identical values, in sorted order.
    Returns the bin label (0..q-1) of every group; tied points always share a bin.
    """
    n = int(sizes.sum())
    labels = np.empty(sizes.size, dtype=np.int64)
    rowsize = n / n_bins
    h = 0          # points in the current bin
    i = 0          # points already assigned
    curr = 0
    for g, s in enumerate(sizes.tolist()):
        if h != 0 and abs(h + s - rowsize) >= abs(h - rowsize):
            curr += 1
            h = 0
            rowsize = (n - i) / (n_bins - curr)
        labels[g] = curr
        i += s
        h += s
    return labels


def _tie_groups(v_sorted: np.ndarray) -> np.ndarray:
    """Start indices of runs of identical values in a sorted vector."""
    return np.flatnonzero(np.r_[True, v_sorted[1:] != v_sorted[:-1]])


def equipartition(v_sorted: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin label of every point of a sorted vector, equipartitioned into (at most) n_bins bins."""
    starts = _tie_groups(v_sorted)
    sizes = np.diff(np.r_[starts, v_sorted.size])
    return np.repeat(_equipartition_groups(sizes, n_bins), sizes)


def clumps(x_sorted: np.ndarray, q_in_x_order: np.ndarray) -> np.ndarray:
    """GetClumpsPartition (Reshef 2011, SOM Alg. 4): clump id of each point, points sorted by x.

    Consecutive points with the same row form a clump; points sharing an x value always share a
    clump (a tie group whose points lie in different rows becomes its own clump).
    """
    qt = q_in_x_order.astype(np.int64).copy()
    starts = _tie_groups(x_sorted)
    ends = np.r_[starts[1:], x_sorted.size]
    multi = np.flatnonzero(ends - starts > 1)
    if multi.size:
        mn = np.minimum.reduceat(qt, starts)[multi]
        mx = np.maximum.reduceat(qt, starts)[multi]
        mixed = multi[mn != mx]
        for c, g in enumerate(mixed):              # unique negative label per mixed tie group
            qt[starts[g]:ends[g]] = -(c + 1)
    change = np.r_[False, qt[1:] != qt[:-1]]
    return np.cumsum(change)


def superclumps(x_sorted: np.ndarray, q_in_x_order: np.ndarray, k_hat: int) -> np.ndarray:
    """GetSuperclumpsPartition (Reshef 2011, SOM Alg. 5): at most k_hat superclumps."""
    p = clumps(x_sorted, q_in_x_order)
    n_clumps = int(p[-1]) + 1
    if n_clumps > k_hat:
        sizes = np.bincount(p, minlength=n_clumps)
        p = np.repeat(_equipartition_groups(sizes, k_hat), sizes)
    return p


# ----------------------------------------------------------------------------- optimisation

def _xlogx_ratio(counts: np.ndarray, totals: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = counts * np.log(counts / totals[..., None])
    return np.where(counts > 0, r, 0.0).sum(axis=-1)


def optimize_x_axis(p: np.ndarray, q: np.ndarray, max_cols: int) -> np.ndarray:
    """OptimizeXAxis (Reshef 2011, SOM Alg. 2), exact dynamic programme over superclumps.

    p: superclump id (0..P-1, non-decreasing) of each point sorted by x; q: row of each point.
    Returns I[k] for k = 0..max_cols: the largest mutual information (nats) between the fixed rows
    and any partition of the superclumps into at most k columns (I[0] = I[1] = 0).

    Uses -H(Q|P) = sum_cols sum_rows n_cr log(n_cr / n_c) / n, which is additive over columns,
    so G(t, k) = max_{s <= t} G(s, k-1) + w(s, t) and I_k = H(Q) + G(P, k) / n.
    """
    n = p.size
    n_sc = int(p[-1]) + 1
    n_rows = int(q.max()) + 1
    out = np.zeros(max_cols + 1)
    rows_tot = np.bincount(q, minlength=n_rows).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        hq = -np.nansum((rows_tot / n) * np.log(rows_tot / n))
    if n_sc == 1 or n_rows == 1:
        return out
    cnt = np.zeros((n_sc, n_rows))
    np.add.at(cnt, (p, q), 1.0)
    cum = np.vstack([np.zeros(n_rows), np.cumsum(cnt, axis=0)])        # (P+1) x rows
    N = cum[None, :, :] - cum[:, None, :]                               # N[s, t, r] for column (s, t]
    tot = N.sum(axis=-1)
    W = _xlogx_ratio(N, tot)                                            # w(s, t); 0 when s >= t
    valid = np.triu(np.ones((n_sc + 1, n_sc + 1), dtype=bool))          # s <= t (s == t: empty column)
    W = np.where(valid, W, -np.inf)
    G = W[0].copy()                                                     # one column: (0, t]
    out[1] = 0.0
    for k in range(2, max_cols + 1):
        G = np.max(G[:, None] + W, axis=0)
        out[k] = hq + G[n_sc] / n
    return np.maximum(out, 0.0)


# ----------------------------------------------------------------------------- MIC

def _orientation(a: np.ndarray, b: np.ndarray, B: float, c: float, est: str) -> list[tuple[int, int, int, float]]:
    """Grid scores with b equipartitioned into l rows and a optimised into k columns.

    Returns (k, l, q, I*) for every admissible grid of this orientation, q = rows actually formed."""
    n = a.size
    ob = np.argsort(b, kind="mergesort")
    b_sorted = b[ob]
    oa = np.argsort(a, kind="mergesort")
    a_sorted = a[oa]
    rank_in_b = np.empty(n, dtype=np.int64)
    rank_in_b[ob] = np.arange(n)
    res = []
    for l in range(2, max(int(math.floor(B / 2)), 2) + 1):
        cols_b = int(math.floor(B / l))
        kmax = min(l, cols_b) if est == "mic_e" else cols_b
        if kmax < 2:
            continue
        q_b = equipartition(b_sorted, l)                 # row of each point, in b order
        q_a = q_b[rank_in_b[oa]]                         # rows in a-sorted order
        q = int(q_b[-1]) + 1
        k_hat = max(int(c * cols_b), 1)
        p = superclumps(a_sorted, q_a, k_hat)
        I = optimize_x_axis(p, q_a, kmax)
        for k in range(2, kmax + 1):
            res.append((k, l, q, I[k]))
    return res


def characteristic_matrix(x: np.ndarray, y: np.ndarray, alpha: float = 0.55, c: float = 5,
                          est: str = "mic_e") -> dict[tuple[int, int, str], float]:
    """Normalised scores I*/log min(k, l) keyed by (k_optimised, l_equipartitioned, orientation)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if est not in ("mic_e", "mic_approx"):
        raise ValueError("est must be 'mic_e' or 'mic_approx'")
    n = x.size
    B = max(n ** alpha, 4.0) if alpha <= 1 else float(alpha)
    M = {}
    for tag, (a, b) in (("x|y", (x, y)), ("y|x", (y, x))):
        for k, l, q, I in _orientation(a, b, B, c, est):
            M[(k, l, tag)] = I / math.log(min(k, q)) if q >= 2 else 0.0
    return M


def mic(x: np.ndarray, y: np.ndarray, alpha: float = 0.55, c: float = 5, est: str = "mic_e") -> float:
    """MIC_e (default) or the 2011 approximate MIC of two vectors; in [0, 1]."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    M = characteristic_matrix(x, y, alpha, c, est)
    return float(min(max(M.values(), default=0.0), 1.0))


def alpha_for_n(n: int) -> float:
    """Sample-size rule for alpha used by MICtools (Albanese et al. 2018), reimplemented:
    0.85 (n<25), 0.80, 0.75, 0.70, 0.65, 0.60, 0.55 (2,500<=n<5,000), 0.50, 0.45, 0.40 (n>=40,000)."""
    bins = [1, 25, 50, 250, 500, 1000, 2500, 5000, 10000, 40000]
    alphas = [0.85, 0.80, 0.75, 0.70, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]
    if n < 1:
        raise ValueError("n must be >= 1")
    return alphas[int(np.digitize([n], bins)[0]) - 1]
