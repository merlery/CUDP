import math

import numpy as np
import pytest

from src import dp


DELTA = 1e-5


@pytest.mark.parametrize("eps,expected", [(0.1, 30.750), (0.5, 7.032), (0.9, 4.107), (5.0, 0.892)])
def test_reference_values(eps, expected):
    assert abs(dp.noise_multiplier(eps, DELTA) - expected) < 1e-3


@pytest.mark.parametrize("eps", np.linspace(0.05, 0.99, 25))
def test_tighter_than_classical_bound(eps):
    assert dp.noise_multiplier(float(eps), DELTA) <= math.sqrt(2 * math.log(1.25 / DELTA)) / eps


def test_decreasing_in_eps():
    vals = [dp.noise_multiplier(e, DELTA) for e in np.geomspace(0.01, 20, 40)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_decreasing_in_delta():
    vals = [dp.noise_multiplier(1.0, d) for d in np.geomspace(1e-10, 1e-2, 30)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_returned_c_is_feasible_and_tight():
    for eps in [0.1, 0.3, 1.0, 5.0, 10.0]:
        c = dp.noise_multiplier(eps, DELTA)
        assert dp._privacy_loss_tail(c, eps) <= DELTA
        assert dp._privacy_loss_tail(c * (1 - 1e-6), eps) > DELTA


def test_uniform_weights_sigma():
    for d in [5, 12, 27]:
        w = np.full(d, 1.0 / d)
        sig = dp.mechanism_sigmas(w, 0.8, 0.8 * DELTA)
        assert np.allclose(sig, math.sqrt(d) * dp.noise_multiplier(0.8, 0.8 * DELTA))


def test_privacy_invariant_random_weights():
    rng = np.random.default_rng(0)
    for _ in range(200):
        d = int(rng.integers(2, 40))
        w = rng.dirichlet(rng.uniform(0.05, 5, size=d))
        w = np.maximum(w, 1e-12); w /= w.sum()
        eps = float(rng.choice([0.1, 0.3, 1.0, 5.0]))
        c = dp.noise_multiplier(eps, DELTA)
        sig = dp.mechanism_sigmas(w, eps, DELTA)
        assert math.isclose((1 / sig ** 2).sum(), 1 / c ** 2, rel_tol=1e-9)


def test_weight_assertions():
    with pytest.raises(AssertionError):
        dp.mechanism_sigmas(np.array([0.5, 0.5, 0.0]), 1.0, DELTA)
    with pytest.raises(AssertionError):
        dp.mechanism_sigmas(np.array([0.5, 0.4]), 1.0, DELTA)


def test_weighted_gaussian_no_clipping_and_scale():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(200000, 3))
    w = np.array([0.6, 0.3, 0.1])
    Xt, sig = dp.weighted_gaussian(X, w, 1.0, DELTA, rng)
    noise = Xt - X
    assert Xt.min() < 0 and Xt.max() > 1          # no clipping after noise
    assert np.allclose(noise.std(axis=0), sig, rtol=0.01)


def test_weighted_gaussian_rejects_unscaled():
    with pytest.raises(ValueError):
        dp.weighted_gaussian(np.array([[0.0, 2.0]]), np.array([0.5, 0.5]), 1.0, DELTA,
                             np.random.default_rng(0))


def test_score_privatization():
    rng = np.random.default_rng(2)
    s = np.array([0.1, 0.5, 0.9, 0.0])
    s_hat, sigma_s = dp.privatize_scores(s, 0.2, 0.2 * DELTA, rng)
    assert math.isclose(sigma_s, 2.0 * dp.noise_multiplier(0.2, 0.2 * DELTA))
    assert np.all((s_hat >= 0) & (s_hat <= 1))
    # s_hat is exactly clip(s + N(0, sigma_s^2), 0, 1) with sigma_s = sqrt(d) c(eps_s, delta_s)
    s_hat2, _ = dp.privatize_scores(s, 0.2, 0.2 * DELTA, np.random.default_rng(9))
    expect = np.clip(s + np.random.default_rng(9).normal(0.0, sigma_s, size=4), 0, 1)
    assert np.allclose(s_hat2, expect)


def test_split_budget():
    assert dp.split_budget(1.0, 1e-5, 0.0) == (0.0, 0.0, 1.0, 1e-5)
    es, ds, em, dm = dp.split_budget(1.0, 1e-5, 0.2)
    assert math.isclose(es + em, 1.0) and math.isclose(ds + dm, 1e-5)
    assert math.isclose(es, 0.2) and math.isclose(dm, 0.8e-5)


def test_subset_release_keeps_invariant():
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(1000, 5))
    w = np.array([0.5, 0.0, 0.3, 0.2, 0.0])
    Xt, sig, keep = dp.weighted_gaussian_subset(X, w, 1.0, 1e-5, rng)
    assert Xt.shape == (1000, 3) and keep.tolist() == [True, False, True, True, False]
    assert np.isinf(sig[[1, 4]]).all()
    c = dp.noise_multiplier(1.0, 1e-5)
    assert math.isclose((1 / sig[keep] ** 2).sum(), 1 / c ** 2, rel_tol=1e-12)
