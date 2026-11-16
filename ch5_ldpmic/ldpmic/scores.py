"""Phase 1: per-feature dependence scores and their one-shot private release.

``fixed_grid_scores`` is the default "fixed-grid dependence score": normalized
mutual information between the feature discretized on G equal-width bins over
its PUBLIC range [a_j, b_j] and the label.  Bin edges depend only on the
public range, never on the data, so the score of feature j is a function of
the empirical joint histogram of (bin_j, y).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np

from .privacy import sigma0_analytic


def _check_ranges(ranges: np.ndarray, d: int) -> np.ndarray:
    ranges = np.asarray(ranges, dtype=float)
    if ranges.shape != (d, 2):
        raise ValueError(f"ranges must have shape ({d}, 2), got {ranges.shape}")
    if not np.all(ranges[:, 1] > ranges[:, 0]):
        raise ValueError("every range must satisfy b_j > a_j")
    return ranges


def discretize(X: np.ndarray, ranges: np.ndarray, G: int) -> np.ndarray:
    """Map each coordinate to a bin index in {0,...,G-1} on the fixed public
    grid; values are clipped to the range first (so out-of-range records land
    in the first/last bin)."""
    X = np.asarray(X, dtype=float)
    ranges = _check_ranges(ranges, X.shape[1])
    a, b = ranges[:, 0], ranges[:, 1]
    Xc = np.clip(X, a, b)
    idx = np.floor((Xc - a) / (b - a) * G).astype(int)
    return np.clip(idx, 0, G - 1)


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def fixed_grid_scores(X: np.ndarray, y: np.ndarray, ranges: np.ndarray, G: int = 10,
                      n_classes: Optional[int] = None) -> np.ndarray:
    """Normalized MI score ``s_j = I(bin_j ; y) / log(min(G, n_classes))``.

    Natural logarithms; the normalization makes ``s_j`` in [0, 1] because
    ``I(A;B) <= min(H(A), H(B)) <= log(min(|A|, |B|))``.  Tiny floating
    overshoots are clipped.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    m, d = X.shape
    if y.shape != (m,):
        raise ValueError("y must have shape (m,)")
    if n_classes is None:
        n_classes = int(y.max()) + 1
    if y.min() < 0 or y.max() >= n_classes:
        raise ValueError("labels must be in [0, n_classes)")
    if G < 2 or n_classes < 2:
        raise ValueError("need G >= 2 and n_classes >= 2")
    bins = discretize(X, ranges, G)
    norm = math.log(min(G, n_classes))
    hy = _entropy(np.bincount(y, minlength=n_classes).astype(float))
    s = np.empty(d)
    for j in range(d):
        joint = np.zeros((G, n_classes))
        np.add.at(joint, (bins[:, j], y), 1.0)
        hx = _entropy(joint.sum(axis=1))
        hxy = _entropy(joint.ravel())
        mi = hx + hy - hxy
        s[j] = mi / norm
    return np.clip(s, 0.0, 1.0)


def correlation_scores(X: np.ndarray, y: np.ndarray, method: str = "pearson") -> np.ndarray:
    """|corr(x_j, y)| in [0,1] for the ``pearson``/``spearman`` ablations.

    These are NOT covered by ``score_sensitivity`` (their sensitivity is not
    bounded by the histogram argument); they are only used in the ``oracle``
    allocation mode or must be released with ``sensitivity=1.0``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if method == "spearman":
        from scipy.stats import rankdata
        X = np.apply_along_axis(rankdata, 0, X)
        y = rankdata(y)
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    num = Xc.T @ yc
    den = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    return np.clip(np.abs(r), 0.0, 1.0)


def mic_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """MIC via ``minepy`` if installed; raises ImportError otherwise (the
    caller falls back to ``fixed_grid_scores``)."""
    from minepy import MINE  # type: ignore  # optional dependency

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    s = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        mine.compute_score(X[:, j], y)
        s[j] = mine.mic()
    return np.clip(s, 0.0, 1.0)


def score_sensitivity(G: int, m: int, n_classes: int, mode: str = "conservative") -> float:
    """Per-score sensitivity ``Delta_s`` of ``fixed_grid_scores`` under the
    replacement of ONE record among ``m``.

    mode="conservative": 1.0.  Always valid because s_j in [0, 1].

    mode="analytic": the bound obtained from the continuity of entropy in
    total variation (Cover & Thomas, *Elements of Information Theory*, 2nd
    ed., Theorem 17.3.3: for distributions P, Q on k outcomes with
    ``||P-Q||_1 = theta <= 1/2``, ``|H(P)-H(Q)| <= theta * log(k/theta)``).
    Replacing one record moves at most two cells of the empirical joint
    histogram by 1/m, so the joint, the feature marginal and the label
    marginal each move by at most ``theta = 2/m`` in l1, and with
    ``I = H(bin) + H(y) - H(bin, y)``:

        |Delta I| <= theta*log(G/theta) + theta*log(C/theta) + theta*log(G*C/theta),

    divided by the normalizer ``log(min(G, C))`` and capped at 1.0.  Requires
    ``m >= 4`` (so that theta <= 1/2); for smaller m returns 1.0.

    This is a *proven but loose* bound (it does not use that the three
    histograms move together).  The author's own tighter proof, once
    finalized, should replace it; see QUESTIONS.md.  ``tests/test_scores.py``
    checks it empirically on random replacements (a sanity check, not a
    proof).
    """
    if mode == "conservative":
        return 1.0
    if mode != "analytic":
        raise ValueError("mode must be 'conservative' or 'analytic'")
    if G < 2 or n_classes < 2 or m < 1:
        raise ValueError("need G >= 2, n_classes >= 2, m >= 1")
    if m < 4:
        return 1.0
    theta = 2.0 / m
    bound = (
        theta * math.log(G / theta)
        + theta * math.log(n_classes / theta)
        + theta * math.log(G * n_classes / theta)
    ) / math.log(min(G, n_classes))
    return float(min(1.0, bound))


def score_noise_std(d: int, sensitivity: float, eps1: float, delta1: float,
                    sigma0_fn: Callable[[float, float], float] = sigma0_analytic) -> float:
    """``sigma_s = sqrt(d) * Delta_s * sigma0(eps1, delta1)``: the whole score
    vector has l2 sensitivity ``sqrt(d) * Delta_s`` (each coordinate moves by
    at most ``Delta_s``) and is released with ONE Gaussian mechanism."""
    if d < 1 or sensitivity <= 0:
        raise ValueError("need d >= 1 and sensitivity > 0")
    return math.sqrt(d) * sensitivity * sigma0_fn(eps1, delta1)


def privatize_scores(s: np.ndarray, eps1: float, delta1: float, sensitivity: float,
                     rng: np.random.Generator,
                     sigma0_fn: Callable[[float, float], float] = sigma0_analytic
                     ) -> Tuple[np.ndarray, float]:
    """One-shot Gaussian release of the WHOLE score vector.

    Returns ``(clip(s + N(0, sigma_s^2 I), 0, 1), sigma_s)``.  Must be called
    exactly once per client per run (tested); clipping to [0,1] is
    post-processing.
    """
    s = np.asarray(s, dtype=float)
    if s.ndim != 1:
        raise ValueError("s must be a 1-d score vector")
    sigma_s = score_noise_std(s.shape[0], sensitivity, eps1, delta1, sigma0_fn)
    noisy = s + sigma_s * rng.standard_normal(s.shape[0])
    return np.clip(noisy, 0.0, 1.0), sigma_s
