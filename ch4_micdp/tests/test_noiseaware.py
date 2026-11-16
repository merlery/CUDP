import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.noiseaware import expected_raw, predict_noise_aware


def test_matches_plain_prediction_when_noise_vanishes():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(3000, 4))
    y = X[:, 0] + np.sin(4 * X[:, 1]) + 0.1 * rng.normal(size=3000)
    m = HistGradientBoostingRegressor(max_iter=30, random_state=0).fit(X, y)
    Xt = rng.uniform(size=(300, 4))
    assert np.allclose(expected_raw(m, Xt, np.full(4, 1e-12)), m.predict(Xt))


def test_matches_monte_carlo_average():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(3000, 3)) + rng.normal(scale=0.5, size=(3000, 3))
    y = (X[:, 0] + rng.normal(scale=0.3, size=3000) > 0.5).astype(int)
    m = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(X, y)
    Xt = rng.uniform(size=(50, 3))
    sig = np.array([0.5, 0.5, 0.5])
    K = 40000
    draws = m.decision_function((Xt[None] + rng.normal(size=(K,) + Xt.shape) * sig).reshape(-1, 3)).reshape(K, -1)
    se = draws.std(axis=0) / np.sqrt(K)
    # exact smoothing agrees with the Monte Carlo average up to Monte Carlo error
    assert np.all(np.abs(draws.mean(axis=0) - expected_raw(m, Xt, sig)) < 4.5 * se)
    p = predict_noise_aware(m, Xt, sig, "classification")
    assert np.all((p > 0) & (p < 1))
