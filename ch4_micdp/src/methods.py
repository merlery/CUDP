"""Weight rules for every method (enhanced MIC-DP, "v2").

All DP methods use the same weighted Gaussian mechanism on the released columns. A method turns
(cached exact statistics, budget, rng) into weights w (w_j >= 0, sum w = 1; w_j = 0 means the column
is not released) and the budget left for the mechanism.

Changes from the original design (see config.yaml, methods.v2):
* scores have small, provable sensitivity (src/fgmic.py) and are computed on the full
  private-train split;
* noisy scores are shrunk toward their mean before use (empirical Bayes, post-processing), so the
  allocation falls back to uniform when the scores carry no information;
* relevance-type rules use w_j proportional to q_j ** alpha with q_j = s_j / max_k s_k, and drop
  columns with q_j < theta once the shrunken scores are reliable (B >= prune_min_B);
* Random-Alloc permutes the MIC-DP weights of the same trial (same shape, signal removed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import dp
from . import fgmic as F


@dataclass
class Allocation:
    w: np.ndarray
    eps_m: float
    delta_m: float
    sigma_s: float
    s_hat: np.ndarray | None
    s_tilde: np.ndarray | None
    B: float

    @property
    def keep(self) -> np.ndarray:
        return self.w > 0


# ----------------------------------------------------------------------------- post-processing

def shrink(s_hat: np.ndarray, sigma: float, tau0: float) -> tuple[np.ndarray, float]:
    """Shrink noisy scores toward their mean. B = min(positive-part James-Stein factor,
    tau0^2 / (tau0^2 + sigma^2)); B = 1 for exact scores. Returns (max(shrunken, 0), B)."""
    s_hat = np.asarray(s_hat, dtype=float)
    if sigma <= 0:
        return np.maximum(s_hat, 0.0), 1.0
    d = s_hat.size
    m = s_hat.mean()
    ss = float(((s_hat - m) ** 2).sum())
    b_js = max(0.0, 1.0 - (d - 3) * sigma ** 2 / ss) if (ss > 0 and d > 3) else 0.0
    b_prior = tau0 ** 2 / (tau0 ** 2 + sigma ** 2)
    B = min(b_js, b_prior)
    return np.maximum(m + B * (s_hat - m), 0.0), float(B)


def relevance_weights(s_t: np.ndarray, B: float, alpha: float, theta: float,
                      prune_min_B: float) -> np.ndarray:
    """w_j proportional to q_j ** alpha, q_j = s_j / max s. Columns with q_j < theta are dropped
    when B >= prune_min_B; otherwise q is floored at theta (nothing dropped)."""
    d = s_t.size
    if not np.isfinite(s_t).all() or s_t.max() <= 0:
        return np.full(d, 1.0 / d)
    q = s_t / s_t.max()
    if B >= prune_min_B:
        v = np.where(q >= theta, q ** alpha, 0.0)
    else:
        v = np.maximum(q, theta) ** alpha
    return v / v.sum()


def inv_sq_weights(s: np.ndarray) -> np.ndarray:
    """w proportional to (1 + s)^(-2)."""
    v = (1.0 + np.asarray(s, dtype=float)) ** -2
    return v / v.sum()


def override_weights(w: np.ndarray, protected_idx: list[int]) -> np.ndarray:
    """Cap protected features at the uniform share 1/d; rescale the other released features."""
    w = np.asarray(w, dtype=float).copy()
    d = w.size
    prot = np.zeros(d, dtype=bool)
    prot[list(protected_idx)] = True
    w[prot] = np.minimum(w[prot], 1.0 / d)
    rest = ~prot & (w > 0)
    if w[rest].sum() > 0:
        w[rest] *= (1.0 - w[prot].sum()) / w[rest].sum()
    else:                                  # every unprotected column was dropped: spread evenly
        w[~prot] = (1.0 - w[prot].sum()) / (~prot).sum()
    return w


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


# ----------------------------------------------------------------------------- allocation

def _mic_dp(raw: dict, d: int, eps_s: float, delta_s: float, rng, p: dict) -> tuple:
    s_hat, sigma_s = dp.privatize_vector(raw["fgmic"], math.sqrt(d) * raw["fgmic_sens"], eps_s, delta_s, rng)
    s_t, B = shrink(s_hat, sigma_s, p["tau0"])
    w = relevance_weights(s_t, B, p["alpha"], p["theta"], p["prune_min_B"])
    return w, sigma_s, s_hat, s_t, B


def _moment_scores(raw: dict, d: int, eps_s: float, delta_s: float, rng, which: str, p: dict) -> tuple:
    v_hat, sigma_e = dp.privatize_vector(raw["moments"], raw["moment_sens"], eps_s, delta_s, rng)
    r, pc, C = F.correlations_from_moments(v_hat, d)
    s = r if which == "pearson" else pc
    # delta-method noise scale of a correlation: sigma_e / sqrt(var_j var_y), median over features
    sig_r = float(np.median(sigma_e / np.sqrt(np.maximum(np.diag(C)[:d] * C[d, d], 1e-12))))
    s_t, B = shrink(s, sig_r, p["tau0"])
    return s, sig_r, s_t, B


def allocate(name: str, mcfg: dict, eps: float, delta: float, raw: dict, d: int,
             rng: np.random.Generator, p: dict, score_rng: np.random.Generator | None = None,
             protected_idx: list[int] | None = None) -> Allocation:
    """p: {'alpha', 'rho', 'theta', 'tau0', 'prune_min_B'} (rho is ignored for rho = 0 methods)."""
    rho = p["rho"] if mcfg["rho"] > 0 else 0.0
    eps_s, delta_s, eps_m, delta_m = dp.split_budget(eps, delta, rho)
    rule = mcfg["weights"]
    nan = float("nan")

    if rule == "uniform":
        return Allocation(np.full(d, 1.0 / d), eps_m, delta_m, nan, None, None, nan)

    if rule in ("relevance", "relevance_override", "permute"):
        score = mcfg["score"]
        if score == "fgmic":
            w, sigma_s, s_hat, s_t, B = _mic_dp(raw, d, eps_s, delta_s,
                                                score_rng if score_rng is not None else rng, p)
        elif score in ("fgmic_public", "fgmic_oracle"):
            s_hat = raw["fgmic_public"] if score == "fgmic_public" else raw["fgmic"]
            s_t, B = np.maximum(s_hat, 0.0), 1.0
            sigma_s = nan
            w = relevance_weights(s_t, B, p["alpha"], p["theta"], p["prune_min_B"])
        elif score == "pearson":
            s_hat, sigma_s, s_t, B = _moment_scores(raw, d, eps_s, delta_s, rng, "pearson", p)
            w = relevance_weights(s_t, B, p["alpha"], p["theta"], p["prune_min_B"])
        else:
            raise ValueError(f"unknown score {score}")
        if rule == "permute":                      # Random-Alloc: MIC-DP's weights, shuffled
            w = w[rng.permutation(d)]
        elif rule == "relevance_override":
            w = override_weights(w, protected_idx or [])
        return Allocation(w, eps_m, delta_m, sigma_s, s_hat, s_t, B)

    if rule == "inv_sq":
        s_hat, sigma_s, s_t, B = _moment_scores(raw, d, eps_s, delta_s, rng, mcfg["score"], p)
        return Allocation(inv_sq_weights(s_t), eps_m, delta_m, sigma_s, s_hat, s_t, B)

    raise ValueError(f"unknown weight rule {rule}")
