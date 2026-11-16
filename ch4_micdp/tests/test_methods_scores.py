import numpy as np
import pytest

from src import dp, methods, scores


P = {"rho": 0.2, "alpha": 2.0, "theta": 0.2, "tau0": 0.1, "prune_min_B": 0.5}
MCFG = {
    "uniform": {"rho": 0.0, "weights": "uniform"},
    "mic_dp": {"rho": 0.2, "weights": "relevance", "score": "fgmic"},
    "mic_public": {"rho": 0.0, "weights": "relevance", "score": "fgmic_public"},
    "mic_oracle": {"rho": 0.0, "weights": "relevance", "score": "fgmic_oracle"},
    "pearson_alloc": {"rho": 0.2, "weights": "relevance", "score": "pearson"},
    "pearson_adj": {"rho": 0.2, "weights": "inv_sq", "score": "pearson"},
    "partialcorr_adj": {"rho": 0.2, "weights": "inv_sq", "score": "partialcorr"},
    "random_alloc": {"rho": 0.2, "weights": "permute", "score": "fgmic"},
    "mic_dp_override": {"rho": 0.2, "weights": "relevance_override", "score": "fgmic"},
}


def _raw(d, rng, n=20000):
    from src import fgmic as F
    X = rng.uniform(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=n) > 0.8).astype(float)
    return {"fgmic": rng.uniform(0, 0.3, size=d), "fgmic_public": rng.uniform(0, 0.3, size=d),
            "fgmic_sens": F.fgmic_sensitivity(n), "moments": F.moment_vector(X, y),
            "moment_sens": F.moment_sensitivity(n, d)}


@pytest.mark.parametrize("name", list(MCFG))
def test_weights_valid(name):
    rng = np.random.default_rng(0)
    d = 12
    raw = _raw(d, rng)
    for eps in [0.1, 1.0, 10.0]:
        al = methods.allocate(name, MCFG[name], eps, 1e-5, raw, d, np.random.default_rng(1), P,
                              score_rng=np.random.default_rng(2), protected_idx=[0, 6, 7])
        assert np.all(al.w >= 0) and abs(al.w.sum() - 1) < 1e-9 and al.w.max() > 0
        if MCFG[name]["rho"] > 0:
            assert np.isclose(al.eps_m, 0.8 * eps) and np.isclose(al.delta_m, 0.8e-5)
        else:
            assert al.eps_m == eps and al.delta_m == 1e-5


def test_shrinkage_falls_back_to_uniform_when_noise_is_large():
    rng = np.random.default_rng(3)
    s = np.array([0.3, 0.2, 0.1, 0.05, 0.01, 0.0, 0.15, 0.25])
    s_t, B = methods.shrink(s + rng.normal(0, 50, size=8), 50.0, 0.1)
    w = methods.relevance_weights(s_t, B, 2.0, 0.2, 0.5)
    assert B < 1e-4 and np.all(w > 0) and w.max() / w.min() < 1.01
    s_t, B = methods.shrink(s, 0.0, 0.1)                 # exact scores: no shrinkage, pruning active
    w = methods.relevance_weights(s_t, B, 2.0, 0.2, 0.5)
    assert B == 1.0 and w[5] == 0 and w[4] == 0 and np.argmax(w) == 0
    assert np.isclose(w[1] / w[0], (0.2 / 0.3) ** 2)


def test_random_alloc_is_a_permutation_of_mic_dp():
    rng = np.random.default_rng(4)
    d = 10
    raw = _raw(d, rng)
    a = methods.allocate("mic_dp", MCFG["mic_dp"], 5.0, 1e-5, raw, d, np.random.default_rng([1000, 2, 4]), P)
    r = methods.allocate("random_alloc", MCFG["random_alloc"], 5.0, 1e-5, raw, d,
                         np.random.default_rng([1000, 7, 4]), P, score_rng=np.random.default_rng([1000, 2, 4]))
    assert np.allclose(np.sort(a.w), np.sort(r.w)) and np.allclose(a.s_hat, r.s_hat)


def test_override_caps_protected_and_renormalises():
    w = np.array([0.5, 0.2, 0.15, 0.1, 0.05, 0.0])
    wo = methods.override_weights(w, [0, 2])
    d = 6
    assert np.all(wo[[0, 2]] <= 1 / d + 1e-15) and abs(wo.sum() - 1) < 1e-12
    assert wo[5] == 0                                    # dropped columns stay dropped
    keep = [1, 3, 4]
    assert np.allclose(wo[keep] / wo[keep].sum(), w[keep] / w[keep].sum())


def test_inv_sq():
    w = methods.inv_sq_weights(np.array([0.0, 1.0]))
    assert np.allclose(w, np.array([1.0, 0.25]) / 1.25)


def test_partial_correlation_known_case():
    rng = np.random.default_rng(5)
    n = 200000
    x1 = rng.standard_normal(n)
    x2 = x1 + rng.standard_normal(n)          # correlated with x1 but no direct effect on y
    y = x1 + 0.5 * rng.standard_normal(n)
    X = np.column_stack([x1, x2])
    pc = scores.partialcorr_scores(X, y)
    r = scores.pearson_scores(X, y)
    assert pc[1] < 0.01 and r[1] > 0.5        # x2 is marginally but not partially correlated
    # analytic partial correlation of x1 with y given x2: corr(y, x1 | x2)
    assert abs(pc[0] - 0.8165) < 0.01


def test_scores_in_unit_interval_and_mic_detects_nonlinear():
    rng = np.random.default_rng(6)
    x = rng.uniform(-1, 1, size=(2000, 3))
    y = x[:, 1] ** 2 + 0.05 * rng.standard_normal(2000)
    m = scores.mic_scores(x, y)
    r = scores.pearson_scores(x, y)
    assert np.all((m >= 0) & (m <= 1)) and np.all((r >= 0) & (r <= 1))
    assert m[1] > 0.5 and r[1] < 0.1           # MIC sees the quadratic term, Pearson does not
    assert m[0] < 0.2 and m[2] < 0.2
    const = scores.mic_scores(np.ones((100, 1)), rng.standard_normal(100))
    assert const[0] == 0.0


def test_subsample_rows():
    a = scores.subsample_rows(10000, 3000, 7)
    b = scores.subsample_rows(10000, 3000, 7)
    assert len(a) == 3000 and np.array_equal(a, b) and len(np.unique(a)) == 3000
    assert np.array_equal(scores.subsample_rows(100, 3000, 7), np.arange(100))

