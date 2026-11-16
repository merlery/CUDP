"""Noise-aware prediction for HistGradientBoosting models trained on Gaussian-perturbed features.

A model trained on X_tilde = X + eta, eta ~ N(0, diag(sigma^2)), is applied to clean test points.
When sigma is much larger than the [0, 1] feature range, every clean test value falls into the same
histogram bin and the model returns one constant prediction. Because sigma is public, we instead
return the model's expected output under the release noise,

    f_bar(x) = E_eta[ f(x + eta) ],

computed exactly: for a tree with axis-aligned splits "x_f <= t", the probability that x + eta
reaches a leaf is a product of Gaussian CDFs along its path. This is post-processing of the released
data and the trained model (it uses only sigma), so it costs no privacy. For classification the
expected raw score is passed through the logistic link (rank-preserving for ROC-AUC).
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, ndtr


def _tree_arrays(model):
    trees = []
    for it in model._predictors:
        nodes = it[0].nodes
        trees.append((nodes["is_leaf"].astype(bool).tolist(), nodes["feature_idx"].tolist(),
                      nodes["num_threshold"].tolist(), nodes["left"].tolist(), nodes["right"].tolist(),
                      nodes["value"].tolist()))
    return trees


def expected_raw(model, X: np.ndarray, sigma: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """E[raw prediction of model at X + eta], eta ~ N(0, diag(sigma^2)), computed exactly.

    A path may split on the same feature several times, so the constraints on each feature are
    collected into one interval (lo, hi] per path; the leaf probability is the product over the
    constrained features of Phi((hi - x_f)/sigma_f) - Phi((lo - x_f)/sigma_f).
    """
    if any(len(it) != 1 for it in model._predictors):
        raise ValueError("only single-output models (regression / binary classification) are supported")
    X = np.asarray(X, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    base = float(np.ravel(model._baseline_prediction)[0])
    trees = _tree_arrays(model)
    out = np.full(X.shape[0], base)
    for start in range(0, X.shape[0], chunk):
        Xc = X[start:start + chunk]
        acc = np.zeros(Xc.shape[0])
        cache: dict = {}

        def cdf(f, t):
            if t == np.inf:
                return 1.0
            if t == -np.inf:
                return 0.0
            v = cache.get((f, t))
            if v is None:
                s = sigma[f]
                v = ndtr((t - Xc[:, f]) / s) if s > 0 else (Xc[:, f] <= t).astype(float)
                cache[(f, t)] = v
            return v

        for is_leaf, feat, thr, left, right, value in trees:
            stack = [(0, {})]
            while stack:
                i, bounds = stack.pop()
                if is_leaf[i]:
                    if not bounds:
                        acc += value[i]
                        continue
                    pr = None
                    for f, (lo, hi) in bounds.items():
                        q = cdf(f, hi) - cdf(f, lo)
                        pr = q if pr is None else pr * q
                    acc += value[i] * pr
                    continue
                f, t_ = feat[i], thr[i]
                lo, hi = bounds.get(f, (-np.inf, np.inf))
                bl = dict(bounds)
                bl[f] = (lo, min(hi, t_))           # left:  x_f + eta_f <= t
                br = dict(bounds)
                br[f] = (max(lo, t_), hi)           # right: x_f + eta_f >  t
                stack.append((right[i], br))
                stack.append((left[i], bl))
        out[start:start + Xc.shape[0]] += acc
    return out


def predict_noise_aware(model, X: np.ndarray, sigma: np.ndarray, task: str) -> np.ndarray:
    """Probability of class 1 (classification) or predicted value (regression)."""
    raw = expected_raw(model, X, sigma)
    return expit(raw) if task == "classification" else raw
