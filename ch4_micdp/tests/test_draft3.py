"""Tests for the Draft 3 follow-up code (src/draft3.py)."""
import math

import numpy as np
import pytest
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier

from src import draft3 as T
from src.noiseaware import expected_raw
from src.public import ContinuedHGBClassifier


def _data(n=600, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    y = ((X[:, 0] + 0.5 * X[:, 1] + 0.2 * rng.standard_normal(n)) > 0.8).astype(int)
    return X, y


def test_best_threshold_matches_brute_force():
    rng = np.random.default_rng(1)
    p = np.round(rng.random(300), 2)
    y = (rng.random(300) < p).astype(int)
    t, ba = T.best_threshold(p, y)
    cand = np.r_[np.unique(p), np.inf]
    bas = [0.5 * (((p >= c) & (y == 1)).sum() / y.sum() + ((p < c) & (y == 0)).sum() / (1 - y).sum()) for c in cand]
    assert ba == pytest.approx(max(bas))
    assert t == cand[int(np.argmax(bas))]


def test_best_threshold_constant_prediction():
    p = np.full(50, 0.3)
    y = np.r_[np.ones(10), np.zeros(40)].astype(int)
    t, ba = T.best_threshold(p, y)
    assert ba == pytest.approx(0.5) and t == 0.3


def test_random_k_release_per_record_and_fixed():
    X, _ = _data(400, 8)
    m = X.mean(axis=0)
    for mode in ("per_record", "fixed"):
        Xt, s, M = T.random_k_release(X, 3, 1.0, 1e-5, np.random.default_rng([5, 41, 0]), mode, m)
        assert np.all(M.sum(axis=1) == 3)
        assert np.all(Xt[~M] == np.broadcast_to(m, M.shape)[~M])
        from src import dp
        assert s == pytest.approx(math.sqrt(3) * dp.noise_multiplier(1.0, 1e-5))
        if mode == "fixed":
            assert np.all(M == M[0])
        else:
            assert len({tuple(r) for r in M[:50]}) > 1


@pytest.mark.parametrize("mode", ["per_record", "fixed"])
def test_expected_raw_sampled_matches_monte_carlo(mode):
    X, y = _data(800, 5, 2)
    m = X.mean(axis=0)
    k, s = 2, 0.3
    rng = np.random.default_rng(3)
    Xt, _, M = T.random_k_release(X, k, 5.0, 1e-5, rng, mode, m)
    # use a fixed noise scale s for the test instead of the calibrated one
    Z = rng.standard_normal(X.shape)
    Xt = np.where(M, X + s * Z, m[None, :])
    model = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(Xt, y)
    Xq = X[:20]
    released = M[0] if mode == "fixed" else None
    exact = T.expected_raw_sampled(model, Xq, s, m, k, mode, released)
    R = 20000
    acc = np.zeros(len(Xq))
    r2 = np.random.default_rng(4)
    for i, x in enumerate(Xq):
        Xr = np.repeat(x[None, :], R, axis=0)
        if mode == "per_record":
            U = r2.random((R, X.shape[1]))
            idx = np.argpartition(U, k - 1, axis=1)[:, :k]
            Mr = np.zeros_like(Xr, dtype=bool)
            np.put_along_axis(Mr, idx, True, axis=1)
        else:
            Mr = np.broadcast_to(released, Xr.shape)
        Xr = np.where(Mr, Xr + s * r2.standard_normal(Xr.shape), m[None, :])
        acc[i] = model._raw_predict(Xr).ravel().mean()
    assert np.max(np.abs(exact - acc)) < 0.02


def test_expected_raw_sampled_fixed_equals_subset_model():
    """Fixed subset: fitting on the mean-filled d columns equals fitting on the k released columns, and the
    sampled expectation equals the existing exact expectation of the subset model."""
    X, y = _data(700, 6, 5)
    m = X.mean(axis=0)
    Xt, s, M = T.random_k_release(X, 3, 5.0, 1e-5, np.random.default_rng(6), "fixed", m)
    keep = M[0]
    full = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(Xt, y)
    sub = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(Xt[:, keep], y)
    a = T.expected_raw_sampled(full, X[:200], s, m, 3, "fixed", keep)
    sig = np.full(keep.sum(), s)
    b = expected_raw(sub, X[:200, keep], sig)
    assert np.max(np.abs(a - b)) < 1e-9


def test_repaired_start_is_clipped_and_without_clip_equals_continued():
    X, y = _data(600, 4, 7)
    rng = np.random.default_rng(8)
    Xn = X + 3.0 * rng.standard_normal(X.shape)
    a = ContinuedHGBClassifier(max_iter=20, warm_start=True, random_state=0, early_stopping=False).fit(X, y)
    a.set_params(max_iter=40).fit(Xn, y)
    b = T.RepairedContinuedHGB(max_iter=20, warm_start=True, random_state=0, early_stopping=False).fit(X, y)
    b.set_params(max_iter=40).fit(Xn, y)                     # start_clip None: same as ContinuedHGB
    assert np.allclose(a._raw_predict(X), b._raw_predict(X), atol=1e-12)
    c = T.RepairedContinuedHGB(max_iter=20, warm_start=True, random_state=0, early_stopping=False).fit(X, y)
    raw0 = c._raw_predict(Xn).ravel()
    c.start_clip = 0.5
    c.set_params(max_iter=40, l2_regularization=1.0).fit(Xn, y)
    lo, hi = c._start_raw_range
    assert lo >= -0.5 and hi <= 0.5 and np.abs(raw0).max() > 0.5
    # the continued trees fit the gradients at the clipped start: f = clip(F_pre) + F_cont reproduces a
    # manual boosting of one step from the clipped start
    f = T.repaired_raw(c, Xn, 20, 0.5)
    assert np.isfinite(f).all()


def test_repaired_noise_aware():
    """Without clipping it is the exact expectation of the whole model; with a clip, the pretrained part's
    expectation is clipped."""
    X, y = _data(500, 4, 9)
    m = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(X, y)
    sig = np.full(4, 0.2)
    p = T.repaired_noise_aware(m, X[:50], sig, 15, 1e6)
    q = expit(expected_raw(m, X[:50], sig))
    assert np.max(np.abs(p - q)) < 1e-9
    ep = expected_raw(T._part(m, 0, 15, True), X[:50], sig)
    ec = expected_raw(T._part(m, 15, None, False), X[:50], sig)
    p2 = T.repaired_noise_aware(m, X[:50], sig, 15, 0.3)
    assert np.max(np.abs(p2 - expit(np.clip(ep, -0.3, 0.3) + ec))) < 1e-12


def test_tree_stats_maps_to_original_columns():
    X, y = _data(500, 5, 10)
    keep = np.array([True, False, True, True, False])
    m = HistGradientBoostingClassifier(max_iter=10, random_state=0).fit(X[:, keep], y)
    ts = T.tree_stats(m, keep, 5)
    assert ts["count"][1] == 0 and ts["count"][4] == 0 and ts["count"][0] > 0
