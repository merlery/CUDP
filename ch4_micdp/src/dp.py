"""Privacy primitives for the Chapter 4 (MIC-DP) experiment.

* ``noise_multiplier``  - analytic Gaussian mechanism calibration (Balle & Wang, 2018).
* ``privatize_scores``  - Gaussian release of a score vector in [0, 1]^d (L2 sensitivity sqrt(d)).
* ``weighted_gaussian`` - per-feature Gaussian noise with sigma_j = c(eps_m, delta_m) / sqrt(w_j)
                          on features scaled to [0, 1] (Delta_j = 1).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.stats import norm


def _privacy_loss_tail(c: float, eps: float) -> float:
    """delta(c) for the Gaussian mechanism with noise multiplier c = sigma / Delta."""
    return float(norm.cdf(1.0 / (2.0 * c) - eps * c)
                 - math.exp(eps) * norm.cdf(-1.0 / (2.0 * c) - eps * c))


@lru_cache(maxsize=None)
def noise_multiplier(eps: float, delta: float, lo: float = 1e-6, hi: float = 1e6,
                     iters: int = 200) -> float:
    """c(eps, delta) = min{c > 0 : Phi(1/(2c) - eps c) - e^eps Phi(-1/(2c) - eps c) <= delta}.

    Bisection on log(c) over [lo, hi]. The returned value is always on the feasible
    side (delta(c) <= delta), so the resulting mechanism is never under-noised.
    """
    eps = float(eps)
    delta = float(delta)
    if not eps > 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    a, b = math.log(lo), math.log(hi)
    if _privacy_loss_tail(math.exp(b), eps) > delta:
        raise ValueError("upper bisection bound is not feasible")
    if _privacy_loss_tail(math.exp(a), eps) <= delta:
        return float(math.exp(a))
    for _ in range(iters):
        m = 0.5 * (a + b)
        if _privacy_loss_tail(math.exp(m), eps) <= delta:
            b = m
        else:
            a = m
    return float(math.exp(b))


def classical_multiplier(eps: float, delta: float) -> float:
    """Classical Gaussian-mechanism bound sqrt(2 ln(1.25/delta)) / eps (valid for eps < 1)."""
    return math.sqrt(2.0 * math.log(1.25 / delta)) / eps


def split_budget(eps: float, delta: float, rho: float) -> tuple[float, float, float, float]:
    """Return (eps_s, delta_s, eps_m, delta_m) for a score share rho (basic composition)."""
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1)")
    if rho == 0:
        return 0.0, 0.0, float(eps), float(delta)
    return rho * eps, rho * delta, (1 - rho) * eps, (1 - rho) * delta


def score_sigma(d: int, eps_s: float, delta_s: float) -> float:
    """sigma_s = sqrt(d) * c(eps_s, delta_s) (each score in [0,1] moves by <= 1)."""
    return math.sqrt(d) * noise_multiplier(eps_s, delta_s)


def privatize_scores(s: np.ndarray, eps_s: float, delta_s: float,
                     rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """s_hat = clip(s + N(0, sigma_s^2 I), 0, 1) with sigma_s = sqrt(d) c(eps_s, delta_s)."""
    s = np.asarray(s, dtype=float)
    if s.ndim != 1:
        raise ValueError("scores must be a vector")
    if np.any(s < 0) or np.any(s > 1):
        raise ValueError("raw scores must lie in [0, 1]")
    sigma_s = score_sigma(s.size, eps_s, delta_s)
    s_hat = np.clip(s + rng.normal(0.0, sigma_s, size=s.size), 0.0, 1.0)
    return s_hat, sigma_s


def mechanism_sigmas(w: np.ndarray, eps_m: float, delta_m: float) -> np.ndarray:
    """sigma_j = c(eps_m, delta_m) / sqrt(w_j) for Delta_j = 1."""
    w = np.asarray(w, dtype=float)
    assert np.all(w > 0), "all weights must be strictly positive"
    assert abs(w.sum() - 1.0) < 1e-9, f"weights must sum to 1 (got {w.sum()!r})"
    return noise_multiplier(eps_m, delta_m) / np.sqrt(w)


def weighted_gaussian(X: np.ndarray, w: np.ndarray, eps_m: float, delta_m: float,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Release X_tilde = X + N(0, diag(sigma^2)) entrywise. No clipping after noise.

    X must already be clipped and scaled to [0, 1], so Delta_j = 1 for every feature.
    Replacing one record changes column j by at most 1, and
    sum_j (Delta_j / sigma_j)^2 = sum_j w_j / c^2 = 1 / c^2, which is exactly the
    privacy loss of a single Gaussian query with noise multiplier c(eps_m, delta_m).
    """
    X = np.asarray(X, dtype=float)
    if X.min() < 0 or X.max() > 1:
        raise ValueError("X must be clipped and scaled to [0, 1] before the mechanism")
    sigma = mechanism_sigmas(w, eps_m, delta_m)
    Z = rng.standard_normal(size=X.shape)
    return X + Z * sigma[None, :], sigma


def privatize_vector(v: np.ndarray, l2_sensitivity: float, eps: float, delta: float,
                     rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Gaussian mechanism on a vector with the given L2 sensitivity. No clipping (any clipping or
    shrinkage downstream is post-processing)."""
    v = np.asarray(v, dtype=float)
    sigma = l2_sensitivity * noise_multiplier(eps, delta)
    return v + rng.normal(0.0, sigma, size=v.shape), sigma


def weighted_gaussian_subset(X: np.ndarray, w: np.ndarray, eps_m: float, delta_m: float,
                             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Release only the columns with w_j > 0 (weights over the released columns sum to 1).

    Returns (X_tilde of the released columns, sigma over all d columns with inf for dropped
    columns, boolean mask of released columns). Not releasing a column is free, and
    sum_{released} 1/sigma_j^2 = 1/c^2 keeps the same (eps_m, delta_m) guarantee.
    """
    w = np.asarray(w, dtype=float)
    assert np.all(w >= 0) and abs(w.sum() - 1.0) < 1e-9, "weights must be >= 0 and sum to 1"
    keep = w > 0
    Xt, s_keep = weighted_gaussian(X[:, keep], w[keep], eps_m, delta_m, rng)
    sigma = np.full(w.size, np.inf)
    sigma[keep] = s_keep
    return Xt, sigma, keep
