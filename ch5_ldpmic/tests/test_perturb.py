import numpy as np
import pytest

from ldpmic.allocation import dimsample_shares, shares
from ldpmic.perturb import array_sha256, check_invariant, perturb_once, sigmas
from ldpmic.privacy import sigma0_analytic


def test_clip_before_noise():
    """Records outside [a_j, b_j] are clipped BEFORE the noise is added: with a
    frozen (zero) noise stream the output equals the clipped input, and with
    real noise the residual output - clip(X) is exactly the noise draw."""
    rng = np.random.default_rng(0)
    d = 4
    ranges = np.array([[0.0, 1.0], [-1.0, 1.0], [10.0, 20.0], [0.0, 0.5]])
    X = rng.uniform(-30, 30, size=(50, d))
    deltas = ranges[:, 1] - ranges[:, 0]
    sig = sigmas(shares(np.zeros(d), "uniform"), deltas, 1.0, 5e-6)

    class ZeroRng:
        def standard_normal(self, shape):
            return np.zeros(shape)

    Xs, _ = perturb_once(X, ranges, sig, ZeroRng())
    np.testing.assert_array_equal(Xs, np.clip(X, ranges[:, 0], ranges[:, 1]))
    assert np.any(X != Xs)  # some records really were out of range

    seed = 123
    Xs2, _ = perturb_once(X, ranges, sig, np.random.default_rng(seed))
    noise = np.random.default_rng(seed).standard_normal(X.shape) * sig
    np.testing.assert_allclose(Xs2 - np.clip(X, ranges[:, 0], ranges[:, 1]), noise, rtol=0, atol=1e-12)


def test_noise_std_matches_sigma():
    rng = np.random.default_rng(1)
    d = 3
    ranges = np.tile([[0.0, 1.0]], (d, 1))
    deltas = np.ones(d)
    p = np.array([0.6, 0.3, 0.1])
    sig = sigmas(p, deltas, 1.0, 5e-6)
    X = np.full((200000, d), 0.5)
    Xs, _ = perturb_once(X, ranges, sig, rng)
    emp = (Xs - 0.5).std(axis=0)
    np.testing.assert_allclose(emp, sig, rtol=0.01)


def test_sha256_is_stable_and_sensitive():
    X = np.random.default_rng(0).standard_normal((10, 3))
    h = array_sha256(X)
    assert h == array_sha256(X.copy())
    Y = X.copy()
    Y[0, 0] += 1e-12
    assert h != array_sha256(Y)
    assert h != array_sha256(X.reshape(3, 10))


def test_dimsample_midpoint_replacement():
    rng = np.random.default_rng(5)
    d, k = 10, 3
    ranges = np.stack([np.arange(d, dtype=float), np.arange(d, dtype=float) + 2], axis=1)
    deltas = ranges[:, 1] - ranges[:, 0]
    p = dimsample_shares(d, k, rng)
    sig = sigmas(p, deltas, 1.0, 5e-6)
    assert np.isinf(sig[p == 0]).all() and np.isfinite(sig[p > 0]).all()
    X = rng.uniform(0, 12, size=(20, d))
    Xs, _ = perturb_once(X, ranges, sig, rng)
    mid = ranges.mean(axis=1)
    for j in np.where(p == 0)[0]:
        assert np.all(Xs[:, j] == mid[j])
    for j in np.where(p > 0)[0]:
        assert not np.all(Xs[:, j] == mid[j])


def test_check_invariant_detects_violation():
    deltas = np.ones(3)
    sig = sigmas(shares(np.zeros(3), "uniform"), deltas, 1.0, 5e-6)
    s0 = sigma0_analytic(1.0, 5e-6)
    check_invariant(sig, deltas, s0)
    with pytest.raises(AssertionError):
        check_invariant(sig * 1.001, deltas, s0)


def test_dimsample_per_record_sampling():
    """Spec: k coordinates sampled per RECORD; sampled ones get N(0, (Delta s0 sqrt k)^2),
    the others the public midpoint; the sampled sets differ across records."""
    from ldpmic.perturb import perturb_once_sampled
    rng = np.random.default_rng(0)
    m, d, k, s0 = 4000, 12, 3, 0.7
    ranges = np.stack([np.zeros(d), np.linspace(1, 5, d)], 1)
    X = rng.uniform(0, 1, (m, d)) * ranges[:, 1]
    Xs, h = perturb_once_sampled(X, ranges, k, s0, np.random.default_rng(1))
    mid = ranges.mean(1)
    released = Xs != mid
    assert (released.sum(1) == k).all()                                   # exactly k per record
    assert len({tuple(np.flatnonzero(r)) for r in released[:200]}) > 50   # sets vary by record
    j = d - 1
    resid = (Xs[:, j] - X[:, j])[released[:, j]]
    assert abs(resid.std() / (ranges[j, 1] * s0 * np.sqrt(k)) - 1) < 0.1
    Xs2, h2 = perturb_once_sampled(X, ranges, k, s0, np.random.default_rng(1))
    assert h == h2 and np.array_equal(Xs, Xs2)                            # deterministic given the seed
