import numpy as np
import pytest

from ldpmic.allocation import MODES, apply_cap, dimsample_shares, shares, validate_shares
from ldpmic.perturb import precision_sum, sigmas
from ldpmic.privacy import sigma0_analytic, sigma0_classical


def _all_mode_shares(d, rng):
    s = rng.uniform(0, 1, size=d)
    deltas = rng.uniform(0.1, 10.0, size=d)
    out = {}
    for mode in MODES:
        out[mode] = shares(s, mode=mode, beta=2.0, floor=0.05, deltas=deltas, rng=rng)
    if d >= 3:
        out["exp_capped"] = shares(s, mode="exp", sensitive=[0, 1], p_max=0.5 / d, deltas=deltas)
    out["dimsample"] = dimsample_shares(d, min(4, d), rng)
    return out, deltas


@pytest.mark.parametrize("d", [1, 2, 7, 50, 784])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_precision_invariant(d, seed):
    """sum_j Delta_j^2 / sigma_j^2 == 1 / sigma0^2 for every allocation mode."""
    rng = np.random.default_rng(seed)
    all_p, deltas = _all_mode_shares(d, rng)
    for eps2, delta2 in [(0.8, 5e-6), (4.0, 5e-6), (0.3, 1e-7)]:
        s0 = sigma0_analytic(eps2, delta2)
        for mode, p in all_p.items():
            sig = sigmas(p, deltas, eps2, delta2)
            lhs = precision_sum(sig, deltas)
            assert abs(lhs - 1.0 / s0**2) <= 1e-9 * max(1.0, 1.0 / s0**2), mode
            assert np.all(sig[p > 0] > 0)


def test_precision_invariant_classical_sigma0():
    rng = np.random.default_rng(3)
    p = shares(rng.uniform(0, 1, 20), "exp")
    deltas = rng.uniform(0.5, 2, 20)
    sig = sigmas(p, deltas, 0.5, 5e-6, sigma0_fn=sigma0_classical)
    assert abs(precision_sum(sig, deltas) - 1 / sigma0_classical(0.5, 5e-6) ** 2) <= 1e-9 / sigma0_classical(0.5, 5e-6) ** 2


@pytest.mark.parametrize("d", [1, 5, 64])
def test_uniform_equals_isotropic(d):
    """Uniform shares give sigma_j = Delta_j * sqrt(d) * sigma0.  With equal
    Delta_j = Delta this is the isotropic Gaussian mechanism calibrated to l2
    sensitivity ||Delta||_2 = Delta * sqrt(d)."""
    eps2, delta2 = 1.0, 5e-6
    s0 = sigma0_analytic(eps2, delta2)
    deltas = np.full(d, 3.0)
    p = shares(np.zeros(d), "uniform")
    sig = sigmas(p, deltas, eps2, delta2)
    np.testing.assert_allclose(sig, deltas * np.sqrt(d) * s0, rtol=1e-12)
    np.testing.assert_allclose(sig, np.linalg.norm(deltas) * s0, rtol=1e-12)
    # unequal Delta_j: still sigma_j = Delta_j sqrt(d) sigma0
    deltas2 = np.linspace(1, 4, d)
    np.testing.assert_allclose(sigmas(p, deltas2, eps2, delta2), deltas2 * np.sqrt(d) * s0, rtol=1e-12)


def test_shares_basic_properties():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 12)
    deltas = rng.uniform(0.5, 2, 12)
    for mode in MODES:
        p = shares(s, mode, deltas=deltas, rng=rng)
        assert p.shape == (12,) and np.all(p > 0) and abs(p.sum() - 1) < 1e-12
    # exp ordering follows scores; uniform is flat
    p_exp = shares(s, "exp")
    assert np.all(np.diff(p_exp[np.argsort(s)]) >= 0)
    np.testing.assert_allclose(shares(s, "uniform"), 1 / 12)
    # random is a permutation of exp
    assert np.allclose(np.sort(shares(s, "random", rng=rng)), np.sort(p_exp))
    # linear uses floor and Delta_j
    s2 = np.array([0.0, 0.5])
    np.testing.assert_allclose(shares(s2, "linear", floor=0.05, deltas=np.array([2.0, 1.0])),
                               np.array([0.1, 0.5]) / 0.6)
    # beta=0 -> uniform
    np.testing.assert_allclose(shares(s, "exp", beta=0.0), 1 / 12)


def test_cap_redistribution():
    """With a sensitive feature capped at p_max, the sum of shares is 1 and
    the capped feature's share equals p_max; uncapped ratios are preserved."""
    s = np.array([0.95, 0.4, 0.3, 0.2, 0.1])
    p_free = shares(s, "exp", beta=4.0)
    assert p_free[0] > 0.3
    p_max = 0.3
    p = shares(s, "exp", beta=4.0, sensitive=[0], p_max=p_max)
    assert abs(p.sum() - 1) < 1e-12
    assert p[0] == pytest.approx(p_max, abs=1e-12)
    np.testing.assert_allclose(p[1:] / p[1:].sum(), p_free[1:] / p_free[1:].sum(), rtol=1e-12)
    # a sensitive feature already below the cap is untouched
    p2 = shares(s, "exp", beta=4.0, sensitive=[4], p_max=p_max)
    np.testing.assert_allclose(p2, p_free)


def test_cap_iterates_when_redistribution_overflows_second_sensitive():
    s = np.array([0.9, 0.85, 0.1, 0.1])
    p = shares(s, "exp", beta=5.0, sensitive=[0, 1], p_max=0.3)
    assert abs(p.sum() - 1) < 1e-12
    assert p[0] == pytest.approx(0.3) and p[1] == pytest.approx(0.3)
    assert np.all(p[2:] > 0)


def test_cap_infeasible_raises():
    with pytest.raises(ValueError):
        apply_cap(np.full(4, 0.25), sensitive=[0, 1, 2, 3], p_max=0.2)


def test_validate_shares_rejects_bad_vectors():
    for bad in ([0.5, 0.5, 0.1], [1.0, 0.0], [np.nan, 1.0], [-0.1, 1.1]):
        with pytest.raises(ValueError):
            validate_shares(np.array(bad))
    validate_shares(np.array([1.0, 0.0]), allow_zero=True)
