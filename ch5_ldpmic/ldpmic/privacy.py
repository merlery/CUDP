"""Privacy calibration: Gaussian-mechanism multipliers, composition, RDP accounting.

Everything here is pure numpy/scipy and is unit-tested in tests/test_privacy.py.

Conventions
-----------
``sigma0(eps, delta)`` is the *unit-sensitivity* Gaussian multiplier: adding
``N(0, (S * sigma0)^2)`` to a query with l2 sensitivity ``S`` is
``(eps, delta)``-DP.  All callers scale by their own sensitivity.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy import special
from scipy.optimize import brentq


# --------------------------------------------------------------------------- #
# Gaussian mechanism calibration
# --------------------------------------------------------------------------- #

def _check_eps_delta(eps: float, delta: float) -> None:
    if not (eps > 0) or not math.isfinite(eps):
        raise ValueError(f"eps must be a positive finite number, got {eps}")
    if not (0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1), got {delta}")


def sigma0_classical(eps: float, delta: float, strict: bool = True) -> float:
    """Classical Gaussian multiplier ``sqrt(2 ln(1.25/delta)) / eps``.

    Dwork & Roth (2014), Theorem A.1.  The theorem is only proven for
    ``eps < 1``; with ``strict=True`` (default) larger ``eps`` raises
    ``ValueError``.  ``strict=False`` evaluates the formula anyway, which is
    useful only for comparisons -- the returned value is then NOT a valid
    calibration (see ``gaussian_delta`` to check it).
    """
    _check_eps_delta(eps, delta)
    if eps >= 1.0 and strict:
        raise ValueError(
            "sigma0_classical is only valid for eps < 1 (Dwork & Roth, Thm A.1); "
            "use sigma0_analytic"
        )
    return math.sqrt(2.0 * math.log(1.25 / delta)) / eps


def gaussian_delta(sigma: float, eps: float, sensitivity: float = 1.0) -> float:
    """Exact ``delta`` of the Gaussian mechanism with noise ``sigma`` at ``eps``.

    Balle & Wang (ICML 2018), "Improving the Gaussian Mechanism for
    Differential Privacy: Analytical Calibration and Optimal Denoising",
    Theorem 8: the mechanism ``M(x) = f(x) + N(0, sigma^2 I)`` with l2
    sensitivity ``S`` is ``(eps, delta)``-DP iff

        delta >= Phi(S/(2 sigma) - eps sigma / S)
                 - exp(eps) * Phi(-S/(2 sigma) - eps sigma / S).

    Evaluated in log space with ``scipy.special.log_ndtr`` so that the
    ``exp(eps) * Phi(.)`` term does not overflow/underflow for large ``eps``.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    a = sensitivity / (2.0 * sigma)
    b = eps * sigma / sensitivity
    t1 = special.ndtr(a - b)
    t2 = math.exp(eps + special.log_ndtr(-a - b))
    return float(t1 - t2)


def sigma0_analytic(eps: float, delta: float, tol: float = 1e-12) -> float:
    """Analytic Gaussian mechanism multiplier (Balle & Wang 2018, Alg. 1).

    Returns the smallest ``sigma`` (unit sensitivity) such that
    ``gaussian_delta(sigma, eps) <= delta``.  Instead of the paper's
    closed-form bracketing we solve the exact condition of Theorem 8 by
    bracketing + Brent's method (a safeguarded bisection), which is what the
    paper's Algorithm 1 does numerically as well.  ``gaussian_delta`` is
    strictly decreasing in ``sigma`` so the root is unique.
    """
    _check_eps_delta(eps, delta)

    def g(sig: float) -> float:
        return gaussian_delta(sig, eps) - delta

    lo, hi = 1e-6, 1.0
    while g(lo) < 0:  # extremely large eps: shrink lower bracket
        lo /= 10.0
        if lo < 1e-300:
            raise RuntimeError("could not bracket sigma from below")
    while g(hi) > 0:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("could not bracket sigma from above")
    sigma = brentq(g, lo, hi, xtol=tol, rtol=4 * np.finfo(float).eps, maxiter=500)
    # brentq returns a point within tol of the root; nudge to the safe side so
    # the returned sigma is guaranteed to satisfy the (eps, delta) condition.
    while gaussian_delta(sigma, eps) > delta:
        sigma *= 1.0 + 1e-12
    return float(sigma)


def sigma0(eps: float, delta: float, method: str = "analytic") -> float:
    """Dispatch: ``method`` in {"analytic", "classical"}."""
    if method == "analytic":
        return sigma0_analytic(eps, delta)
    if method == "classical":
        return sigma0_classical(eps, delta)
    raise ValueError(f"unknown sigma0 method {method!r}")


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #

def compose_basic(budgets: Iterable[Tuple[float, float]]) -> Tuple[float, float]:
    """Basic (adaptive) sequential composition: sum eps, sum delta.

    Dwork & Roth (2014), Theorem 3.16.  Holds for adaptively chosen
    mechanisms, which is what Phase 1 -> Phase 3 is.
    """
    eps_tot, delta_tot = 0.0, 0.0
    for eps, delta in budgets:
        _check_eps_delta(eps, delta)
        eps_tot += eps
        delta_tot += delta
    return eps_tot, delta_tot


def split_budget(eps: float, delta: float, frac1: float = 0.2) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Default split: ``eps1 = frac1*eps, eps2 = (1-frac1)*eps, delta1=delta2=delta/2``.
    ``frac1 = 0`` means no Phase 1 release at all: ``(0, 0)`` and ``(eps, delta)``
    (only valid when the scores are public / oracle)."""
    _check_eps_delta(eps, delta)
    if frac1 == 0:
        return (0.0, 0.0), (eps, delta)
    if not (0 < frac1 < 1):
        raise ValueError("frac1 must be in [0,1)")
    return (frac1 * eps, delta / 2.0), ((1.0 - frac1) * eps, delta / 2.0)


# --------------------------------------------------------------------------- #
# Renyi DP accounting for DP-SGD (Poisson-subsampled Gaussian mechanism)
# --------------------------------------------------------------------------- #
#
# The subsampled-Gaussian RDP bound below is the one of
#   Mironov, Talwar, Zhang, "Renyi Differential Privacy of the Sampled
#   Gaussian Mechanism", arXiv:1908.10530 (2019),
# as implemented in TensorFlow Privacy
#   tensorflow_privacy/privacy/analysis/rdp_accountant.py  (functions
#   _compute_log_a_int, _compute_log_a_frac, _compute_rdp)
# and, identically, in Opacus
#   opacus/accountants/analysis/rdp.py.
# The code is a line-by-line transcription of those reference
# implementations (integer-order closed form; fractional-order series);
# no bound was invented here.  ``tests/test_privacy.py`` cross-checks it
# against the installed Opacus implementation when Opacus is importable.

DEFAULT_ORDERS: Tuple[float, ...] = tuple(
    [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
)


def _log_add(logx: float, logy: float) -> float:
    a, b = min(logx, logy), max(logx, logy)
    if a == -math.inf:
        return b
    return math.log1p(math.exp(a - b)) + b


def _log_sub(logx: float, logy: float) -> float:
    if logx < logy:
        raise ValueError("log_sub of a negative number")
    if logy == -math.inf:
        return logx
    if logx == logy:
        return -math.inf
    return math.log(math.expm1(logx - logy)) + logy


def _log_erfc(x: float) -> float:
    return float(special.log_ndtr(-x * 2**0.5)) + math.log(2.0)


def _compute_log_a_int(q: float, sigma: float, alpha: int) -> float:
    log_a = -math.inf
    for i in range(alpha + 1):
        log_coef_i = (
            math.log(special.binom(alpha, i)) + i * math.log(q) + (alpha - i) * math.log(1 - q)
        )
        s = log_coef_i + (i * i - i) / (2 * sigma**2)
        log_a = _log_add(log_a, s)
    return float(log_a)


def _compute_log_a_frac(q: float, sigma: float, alpha: float) -> float:
    log_a0, log_a1 = -math.inf, -math.inf
    i = 0
    z0 = sigma**2 * math.log(1 / q - 1) + 0.5
    while True:
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i
        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)
        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))
        log_s0 = log_t0 + (i * i - i) / (2 * sigma**2) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * sigma**2) + log_e1
        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)
        i += 1
        if max(log_s0, log_s1) < -30:
            break
    return _log_add(log_a0, log_a1)


def rdp_gaussian(alpha: float, sigma: float) -> float:
    """RDP of order ``alpha`` of the (non-subsampled) Gaussian mechanism with
    unit sensitivity and noise ``sigma``: ``alpha / (2 sigma^2)``
    (Mironov, CSF 2017, Proposition 7)."""
    if alpha <= 1 or sigma <= 0:
        raise ValueError("need alpha > 1 and sigma > 0")
    return alpha / (2.0 * sigma**2)


def rdp_sampled_gaussian(q: float, sigma: float, alpha: float) -> float:
    """RDP of order ``alpha`` for ONE step of the Poisson-subsampled Gaussian
    mechanism with sampling rate ``q`` and noise multiplier ``sigma``
    (noise std = ``sigma`` * clipping norm).  See module comment for source."""
    if not (0 <= q <= 1):
        raise ValueError("q must be in [0,1]")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if q == 0:
        return 0.0
    if q == 1.0:
        return rdp_gaussian(alpha, sigma)
    if math.isinf(alpha):
        return math.inf
    try:
        if float(alpha).is_integer():
            return _compute_log_a_int(q, sigma, int(alpha)) / (alpha - 1)
        return _compute_log_a_frac(q, sigma, alpha) / (alpha - 1)
    except OverflowError:
        # exp((i^2 - i) / (2 sigma^2)) overflows for tiny sigma: the RDP is
        # astronomically large; report it as infinite (conservative).
        return math.inf


def rdp_sgm_steps(q: float, sigma: float, steps: int, orders: Sequence[float] = DEFAULT_ORDERS) -> np.ndarray:
    """RDP vector (one entry per order) for ``steps`` composed SGM steps."""
    return np.array([rdp_sampled_gaussian(q, sigma, a) * steps for a in orders], dtype=float)


def rdp_to_dp(
    orders: Sequence[float],
    rdp: Sequence[float],
    delta: float,
    conversion: str = "mironov",
) -> Tuple[float, float]:
    """Convert an RDP curve to ``(eps, best_order)`` at a target ``delta``.

    conversion="mironov":  eps = rdp(alpha) + log(1/delta)/(alpha-1)
        Mironov (CSF 2017), Proposition 3.  This is what TF-Privacy's
        ``get_privacy_spent`` used up to v0.5 and what the reference value in
        ``tests/test_privacy.py::test_dpsgd_accountant`` was computed with.
    conversion="balle":    eps = rdp(alpha) + log((alpha-1)/alpha)
                                 - (log(delta) + log(alpha))/(alpha-1)
        Balle, Barthe, Gaboardi, Hsu, Sato (AISTATS 2020), Theorem 21 /
        Eq. (6) as implemented in Opacus ``get_privacy_spent`` and newer
        TF-Privacy.  Never larger than the Mironov conversion.
    The minimum over the given orders is returned.
    """
    orders_v = np.asarray(orders, dtype=float)
    rdp_v = np.asarray(rdp, dtype=float)
    if orders_v.shape != rdp_v.shape:
        raise ValueError("orders and rdp must have the same shape")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    if conversion == "mironov":
        eps = rdp_v + math.log(1.0 / delta) / (orders_v - 1)
    elif conversion == "balle":
        eps = rdp_v + np.log((orders_v - 1) / orders_v) - (math.log(delta) + np.log(orders_v)) / (orders_v - 1)
    else:
        raise ValueError(f"unknown conversion {conversion!r}")
    eps = np.where(np.isfinite(eps), eps, np.inf)
    idx = int(np.argmin(eps))
    return float(eps[idx]), float(orders_v[idx])


class RDPAccountant:
    """Per-client RDP accountant for DP-SGD.

    Every call to ``step`` composes one Poisson-subsampled Gaussian step; the
    history persists across FL rounds so that ALL local steps of a client are
    composed.  ``get_epsilon`` converts at a target delta.
    """

    def __init__(self, orders: Sequence[float] = DEFAULT_ORDERS, conversion: str = "balle"):
        self.orders = tuple(float(a) for a in orders)
        self.rdp = np.zeros(len(self.orders))
        self.conversion = conversion
        self.history: list[tuple[float, float, int]] = []  # (sigma, q, steps)

    def step(self, noise_multiplier: float, sample_rate: float, num_steps: int = 1) -> None:
        if num_steps <= 0:
            return
        self.rdp += rdp_sgm_steps(sample_rate, noise_multiplier, num_steps, self.orders)
        self.history.append((noise_multiplier, sample_rate, num_steps))

    def total_steps(self) -> int:
        return sum(h[2] for h in self.history)

    def get_epsilon(self, delta: float) -> float:
        if self.total_steps() == 0:
            return 0.0
        return rdp_to_dp(self.orders, self.rdp, delta, self.conversion)[0]


def dpsgd_epsilon(noise_multiplier: float, sample_rate: float, steps: int, delta: float,
                  orders: Sequence[float] = DEFAULT_ORDERS, conversion: str = "balle") -> float:
    """eps spent by ``steps`` SGM steps (convenience wrapper)."""
    rdp = rdp_sgm_steps(sample_rate, noise_multiplier, steps, orders)
    return rdp_to_dp(orders, rdp, delta, conversion)[0]


def calibrate_noise_multiplier(target_eps: float, delta: float, sample_rate: float, steps: int,
                               orders: Sequence[float] = DEFAULT_ORDERS, conversion: str = "balle",
                               tol: float = 1e-4, sigma_max: float = 1e4) -> float:
    """Smallest noise multiplier (to ``tol``) whose ``steps`` composed SGM
    steps spend at most ``target_eps`` at ``delta``.  Bisection; eps is
    decreasing in sigma."""
    _check_eps_delta(target_eps, delta)
    if steps <= 0:
        raise ValueError("steps must be positive")
    lo, hi = 1e-3, 1.0
    while dpsgd_epsilon(hi, sample_rate, steps, delta, orders, conversion) > target_eps:
        hi *= 2.0
        if hi > sigma_max:
            raise RuntimeError("target eps unreachable with sigma <= sigma_max")
    while dpsgd_epsilon(lo, sample_rate, steps, delta, orders, conversion) <= target_eps:
        lo /= 2.0
        if lo < 1e-9:
            return lo
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if dpsgd_epsilon(mid, sample_rate, steps, delta, orders, conversion) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


def eps_for_sigma(sigma: float, delta: float, sensitivity: float = 1.0, tol: float = 1e-10) -> float:
    """Inverse of ``sigma0_analytic``: the smallest eps at which the Gaussian
    mechanism with noise ``sigma`` and l2 sensitivity ``sensitivity`` is
    ``(eps, delta)``-DP (Balle & Wang Thm. 8; ``gaussian_delta`` is decreasing
    in eps for fixed sigma).  Used to report a per-coordinate
    "eps-equivalent" ``eps_j`` of ``sigma_j`` (the budget a coordinate alone
    would consume); it is a diagnostic, not part of the accounting."""
    if sigma <= 0 or not (0 < delta < 1):
        raise ValueError("need sigma > 0 and delta in (0,1)")
    s = sigma / sensitivity
    lo, hi = 1e-6, 1.0
    while gaussian_delta(s, hi) > delta:
        hi *= 2.0
        if hi > 1e6:
            return math.inf
    while gaussian_delta(s, lo) <= delta:
        lo /= 10.0
        if lo < 1e-12:
            return 0.0
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if gaussian_delta(s, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi
