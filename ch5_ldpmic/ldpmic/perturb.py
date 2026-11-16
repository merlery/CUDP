"""Phase 3: one-shot record perturbation.

``sigmas`` turns shares into per-coordinate noise levels and ``perturb_once``
clips every record to its public range and adds Gaussian noise EXACTLY ONCE.
The sanitized array is hashed so that later code can assert it never changes.

Privacy: for a record x and its replacement x' (both clipped to the public
box), ``sum_j (x_j - x'_j)^2 / sigma_j^2 <= sum_j Delta_j^2 / sigma_j^2
= sum_j p_j / sigma0^2 = 1 / sigma0^2``, i.e. the whitened l2 sensitivity is
1 and the mechanism is ``(eps2, delta2)``-DP for the record.  Releasing all
records of a client is still ``(eps2, delta2)``-DP under the replace-one-record
neighboring relation because only one record changes (parallel composition
over disjoint records).
"""

from __future__ import annotations

import hashlib
from typing import Callable, Optional, Tuple

import numpy as np

from .allocation import validate_shares
from .privacy import sigma0_analytic

INVARIANT_TOL = 1e-9


def sigmas(p: np.ndarray, deltas: np.ndarray, eps2: float, delta2: float,
           sigma0_fn: Callable[[float, float], float] = sigma0_analytic) -> np.ndarray:
    """``sigma_j = Delta_j * sigma0(eps2, delta2) / sqrt(p_j)``.

    ``p_j = 0`` (dimsample) gives ``sigma_j = inf``; ``perturb_once`` then
    replaces that coordinate with the public midpoint.
    """
    p = validate_shares(p, allow_zero=True)
    deltas = np.asarray(deltas, dtype=float)
    if deltas.shape != p.shape or np.any(deltas <= 0) or not np.all(np.isfinite(deltas)):
        raise ValueError("deltas must be positive, finite, same shape as p")
    s0 = sigma0_fn(eps2, delta2)
    with np.errstate(divide="ignore"):
        sig = deltas * s0 / np.sqrt(p)
    sig = np.where(p > 0, sig, np.inf)
    check_invariant(sig, deltas, s0)
    return sig


def precision_sum(sig: np.ndarray, deltas: np.ndarray) -> float:
    """``sum_j Delta_j^2 / sigma_j^2`` (inf-sigma coordinates contribute 0)."""
    sig = np.asarray(sig, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    with np.errstate(divide="ignore"):
        terms = np.where(np.isfinite(sig), deltas**2 / sig**2, 0.0)
    return float(terms.sum())


def check_invariant(sig: np.ndarray, deltas: np.ndarray, s0: float, tol: float = INVARIANT_TOL) -> None:
    """Assert ``sum_j Delta_j^2 / sigma_j^2 == 1 / sigma0^2`` (relative tol)."""
    lhs = precision_sum(sig, deltas)
    rhs = 1.0 / s0**2
    if abs(lhs - rhs) > tol * max(1.0, abs(rhs)):
        raise AssertionError(f"precision invariant violated: {lhs!r} != {rhs!r}")


def array_sha256(X: np.ndarray) -> str:
    """SHA256 of the array's float64 C-contiguous bytes (shape included)."""
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    h = hashlib.sha256()
    h.update(repr(X.shape).encode())
    h.update(X.tobytes())
    return h.hexdigest()


def clip_to_ranges(X: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    ranges = np.asarray(ranges, dtype=float)
    return np.clip(np.asarray(X, dtype=np.float64), ranges[:, 0], ranges[:, 1])


def perturb_once(X: np.ndarray, ranges: np.ndarray, sig: np.ndarray, rng: np.random.Generator
                 ) -> Tuple[np.ndarray, str]:
    """Clip every record to ``[a_j, b_j]`` coordinate-wise, then add
    ``N(0, sigma_j^2)`` to coordinate j of every record, once.

    Coordinates with ``sigma_j = inf`` (dimsample's unsampled coordinates)
    are set to the public midpoint ``(a_j + b_j)/2`` instead.  Returns the
    sanitized float64 array and its SHA256.
    """
    X = np.asarray(X, dtype=np.float64)
    ranges = np.asarray(ranges, dtype=float)
    sig = np.asarray(sig, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-d (m, d)")
    d = X.shape[1]
    if ranges.shape != (d, 2) or sig.shape != (d,):
        raise ValueError("ranges must be (d,2) and sig (d,)")
    if np.any(sig <= 0):
        raise ValueError("sigmas must be positive")
    Xc = clip_to_ranges(X, ranges)
    finite = np.isfinite(sig)
    noise = rng.standard_normal(Xc.shape) * np.where(finite, sig, 0.0)
    Xs = Xc + noise
    if not finite.all():
        mid = 0.5 * (ranges[:, 0] + ranges[:, 1])
        Xs[:, ~finite] = mid[~finite]
    return Xs, array_sha256(Xs)


def perturb_once_sampled(X: np.ndarray, ranges: np.ndarray, k: int, s0: float, rng: np.random.Generator
                         ) -> Tuple[np.ndarray, str]:
    """``dimsample`` (spec: "k uniformly sampled coordinates per record"):
    for EVERY record independently, sample k of the d coordinates uniformly
    (independent of the data), clip, and add ``N(0, (Delta_j s0 sqrt(k))^2)``
    to each sampled coordinate (shares p_j = 1/k on the sampled set); every
    other coordinate is replaced by the public midpoint.  Per record the
    precision invariant ``sum_j Delta_j^2 / sigma_j^2 = k / (k s0^2) = 1/s0^2``
    holds exactly, so each record's release is the Gaussian mechanism of
    Phase 3 with a data-independent share vector.  Returns the sanitized
    array and its SHA256."""
    X = np.asarray(X, dtype=np.float64)
    ranges = np.asarray(ranges, dtype=float)
    m, d = X.shape
    if not (1 <= k <= d):
        raise ValueError("need 1 <= k <= d")
    deltas = ranges[:, 1] - ranges[:, 0]
    sig = deltas * s0 * np.sqrt(k)
    Xc = clip_to_ranges(X, ranges)
    mid = 0.5 * (ranges[:, 0] + ranges[:, 1])
    Xs = np.broadcast_to(mid, (m, d)).copy()
    idx = np.argsort(rng.random((m, d)), axis=1)[:, :k]          # k distinct uniform coordinates per record
    rows = np.arange(m)[:, None]
    Xs[rows, idx] = Xc[rows, idx] + rng.standard_normal((m, k)) * sig[idx]
    if m:
        one = np.full(d, np.inf); one[idx[0]] = sig[idx[0]]
        check_invariant(one, deltas, s0)                             # same value for every sampled set
    return Xs, array_sha256(Xs)
