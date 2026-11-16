import math

import numpy as np
import pytest

from ldpmic import pipeline, scores
from ldpmic.privacy import sigma0_analytic
from ldpmic.scores import fixed_grid_scores, privatize_scores, score_noise_std, score_sensitivity

from .conftest import synthetic_gaussian_dataset


def test_fixed_grid_scores_range_and_informativeness(synthetic_tabular):
    X, y, ranges, C = synthetic_tabular
    s = fixed_grid_scores(X, y, ranges, G=10, n_classes=C)
    assert s.shape == (X.shape[1],)
    assert np.all((s >= 0) & (s <= 1))
    # informative features (first 3) score higher than pure-noise features
    assert s[:3].mean() > s[3:].mean()


def test_fixed_grid_scores_use_public_grid_only():
    """Bin edges depend only on the public range: shifting the data outside
    the range (clipped into the edge bins) or inside it changes bins, but
    passing a different data-derived range is never consulted."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(500, 2))
    X[:, 0] = rng.permutation((np.arange(500) + 0.5) / 500)  # exactly balanced halves
    y = (X[:, 0] > 0.5).astype(int)
    ranges = np.array([[0.0, 1.0], [0.0, 1.0]])
    s = fixed_grid_scores(X, y, ranges, 10, 2)
    assert s[0] > 0.9 and s[1] < 0.1
    # balanced labels, perfectly separable at the grid boundary 0.5 -> normalized MI == 1
    assert s[0] == pytest.approx(1.0, abs=1e-12)


def test_score_sensitivity_modes():
    assert score_sensitivity(10, 1000, 2, "conservative") == 1.0
    a = score_sensitivity(10, 1000, 2, "analytic")
    assert 0 < a < 1
    assert score_sensitivity(10, 50, 2, "analytic") > a
    assert score_sensitivity(10, 3, 2, "analytic") == 1.0
    with pytest.raises(ValueError):
        score_sensitivity(10, 100, 2, "bogus")


@pytest.mark.parametrize("m", [50, 200, 1000])
def test_empirical_score_sensitivity(m):
    """Sanity check, NOT a proof: over 200 random single-record replacements
    the max change of any s_j never exceeds the analytic bound (nor 1.0).
    A proof must cover all datasets; this only samples random ones."""
    rng = np.random.default_rng(m)
    d, C, G = 5, 2, 10
    X, y = synthetic_gaussian_dataset(m, d, C, seed=m)
    ranges = np.tile([[-5.0, 5.0]], (d, 1))
    base = fixed_grid_scores(X, y, ranges, G, C)
    bound = score_sensitivity(G, m, C, "analytic")
    worst = 0.0
    for _ in range(200):
        X2, y2 = X.copy(), y.copy()
        i = rng.integers(m)
        X2[i] = rng.uniform(-5, 5, size=d)
        y2[i] = rng.integers(C)
        s2 = fixed_grid_scores(X2, y2, ranges, G, C)
        worst = max(worst, float(np.abs(s2 - base).max()))
    assert worst <= bound <= 1.0
    assert worst <= score_sensitivity(G, m, C, "conservative")


def test_score_vector_single_release_std():
    """privatize_scores adds noise with std exactly
    sqrt(d) * Delta_s * sigma0(eps1, delta1) in ONE vector release."""
    d, eps1, delta1, Ds = 7, 0.2, 5e-6, 1.0
    s = np.linspace(0, 1, d)
    seed = 7
    noisy, sigma_s = privatize_scores(s, eps1, delta1, Ds, np.random.default_rng(seed))
    expected = math.sqrt(d) * Ds * sigma0_analytic(eps1, delta1)
    assert sigma_s == pytest.approx(expected, rel=1e-12)
    assert score_noise_std(d, Ds, eps1, delta1) == pytest.approx(expected, rel=1e-12)
    # the draw is exactly sigma_s * standard normals (before clipping to [0,1])
    z = np.random.default_rng(seed).standard_normal(d)
    np.testing.assert_allclose(noisy, np.clip(s + sigma_s * z, 0, 1), atol=1e-12)
    assert np.all((noisy >= 0) & (noisy <= 1))
    # large sample: empirical std matches (sigma_s ~ 0.03, so clipping is immaterial)
    big = np.full(200000, 0.5)
    out, sig = privatize_scores(big, 5.0, 1e-3, 1e-4, np.random.default_rng(1))
    assert sig < 0.05
    assert (out - 0.5).std() == pytest.approx(sig, rel=0.02)


@pytest.mark.parametrize("alloc", ["per_client", "global"])
def test_score_vector_single_release_called_once_per_client(synthetic_federated, monkeypatch, alloc):
    """Phase 1 releases the score vector exactly once per client per run."""
    clients_data, _, ranges, C = synthetic_federated
    calls = []
    real = scores.privatize_scores

    def counting(s, eps1, delta1, sensitivity, rng, **kw):
        calls.append((len(s), eps1, delta1, sensitivity))
        return real(s, eps1, delta1, sensitivity, rng, **kw)

    monkeypatch.setattr(scores, "privatize_scores", counting)
    mcfg = pipeline.MethodConfig(method="ldpmic", eps=1.0, delta=1e-5, allocation=alloc)
    fl_cfg = pipeline.FLConfig(rounds=3, participation=1.0, local_epochs=1, batch_size=32, lr=0.1,
                               clip_update=None, seed=0)
    clients, info = pipeline.build_clients(mcfg, clients_data, ranges, C, seed=0, fl_cfg=fl_cfg)
    assert len(calls) == len(clients_data)
    assert all(c == (ranges.shape[0], 0.2, 5e-6, 1.0) for c in calls)
    assert info.eps1 == pytest.approx(0.2) and info.eps2 == pytest.approx(0.8)
    assert (info.eps_realized, info.delta_realized) == pytest.approx((1.0, 1e-5))
    assert info.private


def test_oracle_is_flagged_non_private(synthetic_federated, monkeypatch):
    clients_data, _, ranges, C = synthetic_federated
    monkeypatch.setattr(scores, "privatize_scores", lambda *a, **k: pytest.fail("oracle must not call privatize"))
    mcfg = pipeline.MethodConfig(method="ldpmic", allocation="oracle")
    fl_cfg = pipeline.FLConfig(rounds=1, participation=1.0, local_epochs=1, batch_size=32, lr=0.1, clip_update=None, seed=0)
    _, info = pipeline.build_clients(mcfg, clients_data, ranges, C, seed=0, fl_cfg=fl_cfg)
    assert not info.private


def test_iso_spends_all_eps_in_phase3(synthetic_federated, monkeypatch):
    clients_data, _, ranges, C = synthetic_federated
    monkeypatch.setattr(scores, "privatize_scores", lambda *a, **k: pytest.fail("iso has no Phase 1"))
    mcfg = pipeline.MethodConfig(method="iso", eps=1.0, delta=1e-5)
    fl_cfg = pipeline.FLConfig(rounds=1, participation=1.0, local_epochs=1, batch_size=32, lr=0.1, clip_update=None, seed=0)
    clients, info = pipeline.build_clients(mcfg, clients_data, ranges, C, seed=0, fl_cfg=fl_cfg)
    assert info.eps1 == 0 and info.eps2 == 1.0 and info.delta2 == 1e-5
    d = ranges.shape[0]
    for c in clients:
        np.testing.assert_allclose(c.sig, (ranges[:, 1] - ranges[:, 0]) * np.sqrt(d) * sigma0_analytic(1.0, 1e-5))
