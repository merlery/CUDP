import math

import numpy as np
import pytest

from ldpmic import privacy
from ldpmic.privacy import (
    DEFAULT_ORDERS,
    RDPAccountant,
    calibrate_noise_multiplier,
    compose_basic,
    dpsgd_epsilon,
    gaussian_delta,
    rdp_sgm_steps,
    rdp_to_dp,
    sigma0_analytic,
    sigma0_classical,
)


@pytest.mark.parametrize("eps", [0.1, 0.5, 0.9])
def test_analytic_le_classical_small_eps(eps):
    """For eps < 1 (where the classical theorem applies) the analytic
    calibration is never worse, and both satisfy the exact condition."""
    delta = 1e-5
    s_an = sigma0_analytic(eps, delta)
    s_cl = sigma0_classical(eps, delta)
    assert s_an <= s_cl
    assert gaussian_delta(s_an, eps) <= delta
    assert gaussian_delta(s_cl, eps) <= delta


@pytest.mark.parametrize("eps", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_analytic_satisfies_exact_condition(eps):
    """The analytic sigma satisfies the exact (eps, delta) condition of Balle &
    Wang Thm. 8 when plugged back in, and is tight (a slightly smaller sigma
    violates it)."""
    delta = 1e-5
    s = sigma0_analytic(eps, delta)
    assert gaussian_delta(s, eps) <= delta
    assert gaussian_delta(s * (1 - 1e-6), eps) > delta


@pytest.mark.parametrize("eps", [1.0, 2.0, 5.0, 10.0])
def test_classical_formula_invalid_for_large_eps(eps):
    """Documented fact (Balle & Wang 2018, Sec. 1): the classical formula is
    not a valid calibration for eps >= 1.  We check that it raises by default
    and that, when forced, eps = 10 yields a sigma that violates the exact
    condition (and is SMALLER than the analytic sigma) -- so the inequality
    ``sigma0_analytic <= sigma0_classical`` cannot be asserted for all of
    eps in {0.5, 1, 2, 5, 10} (see QUESTIONS.md Q10)."""
    with pytest.raises(ValueError):
        sigma0_classical(eps, 1e-5)
    forced = sigma0_classical(eps, 1e-5, strict=False)
    assert forced > 0
    if eps == 10.0:
        assert gaussian_delta(forced, eps) > 1e-5
        assert forced < sigma0_analytic(eps, 1e-5)


def test_gaussian_delta_monotone():
    eps = 1.0
    sig = np.linspace(0.3, 5.0, 50)
    d = np.array([gaussian_delta(s, eps) for s in sig])
    assert np.all(np.diff(d) < 0)


def test_compose_basic():
    assert compose_basic([(0.2, 5e-6), (0.8, 5e-6)]) == pytest.approx((1.0, 1e-5))
    with pytest.raises(ValueError):
        compose_basic([(0.0, 1e-5)])


def test_default_split_composes_to_total():
    (e1, d1), (e2, d2) = privacy.split_budget(1.0, 1e-5)
    assert (e1, d1) == pytest.approx((0.2, 5e-6))
    assert compose_basic([(e1, d1), (e2, d2)]) == pytest.approx((1.0, 1e-5))


def test_rdp_gaussian_no_subsampling_matches_closed_form():
    for a in (1.5, 2, 8, 32):
        assert privacy.rdp_sampled_gaussian(1.0, 2.0, a) == pytest.approx(a / 8.0)


def test_dpsgd_accountant():
    """Reference value: TensorFlow Privacy,
    tensorflow_privacy/privacy/analysis/rdp_accountant_test.py,
    ``test_get_privacy_spent_check_target_delta``: q=0.01, sigma=4,
    steps=10000, orders=range(2,33), delta=1e-5 -> eps = 1.258575 at order 20
    (Mironov Prop. 3 conversion).  Same setting as Abadi et al. 2016, Sec. 3.2
    ("eps approx 1.26 for sigma=4, q=0.01, T=10^4").
    """
    orders = list(range(2, 33))
    rdp = rdp_sgm_steps(0.01, 4.0, 10000, orders)
    eps, order = rdp_to_dp(orders, rdp, 1e-5, conversion="mironov")
    assert abs(eps - 1.258575) / 1.258575 < 0.01
    assert order == 20


def test_dpsgd_accountant_matches_opacus():
    """Cross-check the transcribed SGM-RDP formulas against the installed
    Opacus implementation on integer and fractional orders."""
    rdp_ref = pytest.importorskip("opacus.accountants.analysis.rdp")
    q, sigma, steps = 0.05, 1.1, 300
    orders = [1.5, 2, 2.5, 4, 8, 16.3, 32, 63]
    ours = rdp_sgm_steps(q, sigma, steps, orders)
    theirs = np.asarray(rdp_ref.compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=orders))
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-12)
    eps_ours, _ = rdp_to_dp(orders, ours, 1e-5, conversion="balle")
    eps_theirs, _ = rdp_ref.get_privacy_spent(orders=orders, rdp=theirs, delta=1e-5)
    assert eps_ours == pytest.approx(float(eps_theirs), rel=1e-9)


def test_balle_conversion_never_worse_than_mironov():
    rdp = rdp_sgm_steps(0.02, 1.5, 500, DEFAULT_ORDERS)
    e_m, _ = rdp_to_dp(DEFAULT_ORDERS, rdp, 1e-5, "mironov")
    e_b, _ = rdp_to_dp(DEFAULT_ORDERS, rdp, 1e-5, "balle")
    assert e_b <= e_m


def test_calibrate_noise_multiplier_hits_target():
    target, delta, q, steps = 2.0, 1e-5, 0.1, 400
    sigma = calibrate_noise_multiplier(target, delta, q, steps)
    assert dpsgd_epsilon(sigma, q, steps, delta) <= target
    assert dpsgd_epsilon(sigma - 2e-4, q, steps, delta) > target


def test_accountant_composes_across_rounds():
    acc = RDPAccountant()
    for _ in range(5):
        acc.step(1.2, 0.1, 20)
    assert acc.total_steps() == 100
    assert acc.get_epsilon(1e-5) == pytest.approx(dpsgd_epsilon(1.2, 0.1, 100, 1e-5))


@pytest.mark.parametrize("eps", [0.1, 0.8, 4.0, 20.0])
def test_eps_for_sigma_inverts_sigma0(eps):
    delta = 5e-6
    s = sigma0_analytic(eps, delta)
    assert privacy.eps_for_sigma(s, delta) == pytest.approx(eps, rel=1e-6)
    # with sensitivity Delta the same eps is obtained for sigma = Delta * s0
    assert privacy.eps_for_sigma(3.0 * s, delta, sensitivity=3.0) == pytest.approx(eps, rel=1e-6)


def test_rdp_tiny_sigma_does_not_overflow():
    """Tiny noise multipliers make the SGM series overflow; the RDP is then
    reported as infinite (conservative) instead of raising."""
    for sigma in (1e-3, 1e-5, 1e-9):
        for a in (1.5, 8, 63):
            v = privacy.rdp_sampled_gaussian(0.2, sigma, a)   # must not raise
            assert v > 0
    assert dpsgd_epsilon(1e-9, 0.2, 10, 1e-5) > 1e6   # astronomically large or infinite, never an exception
    # calibration at a very large target bisects through tiny sigmas without raising
    nm = calibrate_noise_multiplier(50.0, 1e-5, 0.2, 100)
    assert 0 < nm < 1.0 and math.isfinite(nm)
