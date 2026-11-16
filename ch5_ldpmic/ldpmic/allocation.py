"""Phase 2: precision shares p (p_j > 0, sum_j p_j = 1) from dependence scores.

The guarantee of Phase 3 holds for ANY valid share vector, so ``validate_shares``
is the only thing a client must trust; everything else here is a utility
heuristic.  Under the ``global`` allocation mode the server broadcasts p and
every client re-validates it with ``validate_shares`` before use.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

MODES = ("uniform", "exp", "linear", "random", "pearson", "spearman")

SUM_TOL = 1e-9


def validate_shares(p: np.ndarray, allow_zero: bool = False) -> np.ndarray:
    """Raise ``ValueError`` unless p is finite, > 0 (>= 0 if ``allow_zero``)
    and sums to 1 within ``SUM_TOL``.  Returns p as a float array."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p must be a non-empty 1-d vector")
    if not np.all(np.isfinite(p)):
        raise ValueError("p contains non-finite entries")
    if allow_zero:
        if np.any(p < 0):
            raise ValueError("p contains negative entries")
        if not np.any(p > 0):
            raise ValueError("p must have at least one positive entry")
    elif np.any(p <= 0):
        raise ValueError("every p_j must be strictly positive")
    if abs(p.sum() - 1.0) > SUM_TOL:
        raise ValueError(f"shares must sum to 1 (got {p.sum()!r})")
    return p


def _normalize(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    tot = w.sum()
    if tot <= 0:
        raise ValueError("weights sum to zero")
    return w / tot


def apply_cap(p: np.ndarray, sensitive: Sequence[int], p_max: float, max_iter: int = 1000) -> np.ndarray:
    """Enforce ``p_j <= p_max`` for ``j in sensitive`` and redistribute the
    surplus proportionally over the remaining (unfixed) features.

    Done by water-filling: features at the cap are fixed to exactly
    ``p_max``; the rest is rescaled so that the total stays 1; repeat until
    no sensitive feature exceeds the cap (each pass fixes at least one more
    feature, so at most ``len(sensitive)`` passes are needed).  Raises if
    the constraint set is infeasible (all features sensitive and
    ``d * p_max < 1``) or if nothing is left to absorb the surplus.
    """
    p = validate_shares(p).copy()
    sensitive = sorted(set(int(j) for j in sensitive))
    if not sensitive:
        return p
    if not (0 < p_max <= 1):
        raise ValueError("p_max must be in (0, 1]")
    if any(j < 0 or j >= p.size for j in sensitive):
        raise ValueError("sensitive index out of range")
    fixed = np.zeros(p.size, dtype=bool)
    for _ in range(max_iter):
        over = [j for j in sensitive if (not fixed[j]) and p[j] > p_max + 1e-15]
        if not over:
            break
        for j in over:
            fixed[j] = True
            p[j] = p_max
        free = ~fixed
        remaining = 1.0 - p[fixed].sum()
        if remaining <= 0 or not np.any(free):
            raise ValueError("cap infeasible: nothing left to absorb the surplus")
        free_sum = p[free].sum()
        if free_sum <= 0:
            raise ValueError("cap infeasible: free features have zero mass")
        p[free] *= remaining / free_sum
    else:  # pragma: no cover
        raise RuntimeError("apply_cap did not converge")
    p = p / p.sum()  # remove floating drift
    return validate_shares(p)


def shares(scores: np.ndarray, mode: str = "exp", beta: float = 2.0, floor: float = 0.05,
           deltas: Optional[np.ndarray] = None, sensitive: Optional[Sequence[int]] = None,
           p_max: Optional[float] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Compute shares from scores.

    uniform : p_j = 1/d (isotropic Gaussian baseline; ``scores`` only sets d)
    exp     : p_j ∝ exp(beta * s_j)
    linear  : p_j ∝ max(s_j, floor) * Delta_j     (needs ``deltas``)
    random  : random permutation of the exp allocation (needs ``rng``)
    pearson / spearman : same formula as exp; the caller passes |corr| scores
              (see ``scores.correlation_scores``) instead of MI scores.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("scores must be a non-empty 1-d vector")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    d = s.size
    if mode == "uniform":
        p = np.full(d, 1.0 / d)
    elif mode in ("exp", "pearson", "spearman"):
        p = _normalize(np.exp(beta * (s - s.max())))  # shift for stability; same ratios
    elif mode == "linear":
        if deltas is None:
            raise ValueError("linear mode needs deltas")
        deltas = np.asarray(deltas, dtype=float)
        if deltas.shape != (d,) or np.any(deltas <= 0):
            raise ValueError("deltas must be positive with shape (d,)")
        p = _normalize(np.maximum(s, floor) * deltas)
    elif mode == "random":
        if rng is None:
            raise ValueError("random mode needs rng")
        p_exp = _normalize(np.exp(beta * (s - s.max())))
        p = p_exp[rng.permutation(d)]
    else:
        raise ValueError(f"unknown allocation mode {mode!r}; choose from {MODES}")
    if sensitive:
        if p_max is None:
            raise ValueError("p_max is required when sensitive features are given")
        p = apply_cap(p, sensitive, p_max)
    return validate_shares(p)


def dimsample_shares(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """``dimsample`` baseline: p_j = 1/k on k uniformly sampled coordinates,
    0 elsewhere.  Zero-share coordinates get infinite sigma in Phase 3 and
    are replaced by the public midpoint."""
    if not (1 <= k <= d):
        raise ValueError("need 1 <= k <= d")
    p = np.zeros(d)
    p[rng.choice(d, size=k, replace=False)] = 1.0 / k
    return validate_shares(p, allow_zero=True)
