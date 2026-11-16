"""Unit tests for the public-data baselines (src/public.py)."""
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src import data
from src import public as PB
from src.noiseaware import expected_raw
from src.scores import v2_statistics

CFG = data.load_config()


def _toy(n=3000, d=5, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d)) + shift
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n) > 0.9 + shift * 1.5).astype(int)
    return X, y


def test_continued_hgb_same_data_equals_one_long_fit():
    X, y = _toy()
    a = PB.ContinuedHGBClassifier(max_iter=30, warm_start=True, early_stopping=False, random_state=1).fit(X, y)
    a.set_params(max_iter=60)
    a.fit(X, y)
    b = HistGradientBoostingClassifier(max_iter=60, early_stopping=False, random_state=1).fit(X, y)
    assert len(a._predictors) == 60
    assert np.max(np.abs(a.decision_function(X) - b.decision_function(X))) < 1e-12


def test_continued_hgb_starts_from_the_models_output_on_new_rows():
    X, y = _toy()
    Xn, yn = _toy(seed=1)
    Xn = Xn + np.random.default_rng(2).normal(0, 3.0, Xn.shape)
    m = PB.ContinuedHGBClassifier(max_iter=20, warm_start=True, early_stopping=False, random_state=1).fit(X, y)
    want = m.decision_function(Xn).copy()
    seen = {}
    orig = PB.ContinuedHGBClassifier._raw_predict

    def spy(self, Xb, n_threads=None):
        in_fit = getattr(self, "_in_fit", False)
        out = orig(self, Xb, n_threads)
        if in_fit and "first" not in seen:
            seen["first"] = out.ravel().copy()
        return out

    PB.ContinuedHGBClassifier._raw_predict = spy
    try:
        m.set_params(max_iter=40)
        m.fit(Xn, yn)
    finally:
        PB.ContinuedHGBClassifier._raw_predict = orig
    assert np.max(np.abs(seen["first"] - want)) < 1e-10
    assert len(m._predictors) == 40
    # the continued ensemble keeps the first 20 trees unchanged
    staged = list(m.staged_decision_function(X))
    m0 = HistGradientBoostingClassifier(max_iter=20, early_stopping=False, random_state=1).fit(X, y)
    assert np.max(np.abs(staged[19] - m0.decision_function(X))) < 1e-12


def test_noise_aware_runs_on_continued_model():
    X, y = _toy(n=800)
    m = PB.ContinuedHGBClassifier(max_iter=10, warm_start=True, early_stopping=False, random_state=1).fit(X, y)
    m.set_params(max_iter=20)
    m.fit(X + np.random.default_rng(0).normal(0, 0.5, X.shape), y)
    zero = expected_raw(m, X[:50], np.zeros(X.shape[1]))
    assert np.allclose(zero, m.decision_function(X[:50]), atol=1e-10)


def _fake_prepared(n_pub=1500, n_total=10000, seed=0):
    rng = np.random.default_rng(seed)
    Xp, yp = _toy(n=n_pub, seed=seed)
    Xp = np.clip(Xp, 0, 1)
    Xt, yt = _toy(n=2000, seed=seed + 5)
    return SimpleNamespace(n_total=n_total, idx_public=np.arange(n_pub), X_public=Xp, y_public=yp.astype(float),
                           features=["age", "f1", "f2", "f3", "f4"], a=np.array([17.0, 0, 0, 0, 0]),
                           b=np.array([90.0, 1, 1, 1, 1]), task="classification", d=5,
                           X_train=np.clip(Xt, 0, 1), y_train=yt.astype(float))


def test_public_subsets_nested_stratified_and_shifted():
    p = _fake_prepared()
    s = PB.public_subsets(p, CFG, "adult")
    s15, s5, s1, sS = (set(s[k].tolist()) for k in ("p15", "p5", "p1", "shift5"))
    assert len(s5) == 500 and len(s1) == 100 and len(sS) == 500
    assert s1 <= s5 <= s15 and sS <= s15
    for k in ("p5", "p1"):
        assert abs(p.y_public[s[k]].mean() - p.y_public.mean()) < 0.01
    raw = 17.0 + p.X_public[:, 0] * 73.0
    assert np.all(raw[s["shift5"]] <= np.median(raw) + 1e-9)
    s2 = PB.public_subsets(p, CFG, "adult")
    assert all(np.array_equal(s[k], s2[k]) for k in ("p15", "p5", "p1", "shift5"))


def test_fgmic_on_full_public_split_equals_existing_mic_public_scores():
    p = _fake_prepared()
    raw = v2_statistics(p, CFG)
    assert np.allclose(PB.fgmic_on(p.X_public, p.y_public, p.task, CFG), raw["fgmic_public"], atol=0, rtol=0)
