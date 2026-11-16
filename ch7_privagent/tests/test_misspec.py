"""Tests for the exploratory misspecification run (src/misspec.py)."""
import numpy as np

from src import dialogue as DLG
from src import misspec as MS
from src.run import load_config


def _W(n=400, seed=0):
    return np.random.default_rng(seed).dirichlet([1, 1, 1], size=n)


def test_logistic_rule_reproduces_stored_dialogue():
    cfg = load_config()
    W = _W()
    for name, rep in (("pa6", True), ("lpla", False)):
        a = DLG.run(W, cfg, name, 0.1, np.random.default_rng(5), repair=rep)
        b = MS.run_dialogue(W, cfg, name, 0.1, np.random.default_rng(5), rep, {"rule": "logistic"}, np.zeros(len(W), bool))
        for k in ("pair", "contrast", "answer", "used", "repair"):
            assert np.array_equal(a[k], b[k]), k


def test_rules():
    W = _W(1000, 1)
    pair = np.zeros(1000, int)              # LP-LA
    pos = np.ones(1000, int)
    p_log = MS.prob_A(W, pair, 0.9, 0.1, 0.1, {"rule": "logistic"}, pos, None)
    p_lap = MS.prob_A(W, pair, 0.9, 0.1, 0.1, {"rule": "lapse", "p": 0.2}, pos, None)
    assert np.allclose(p_lap, 0.8 * p_log + 0.1)
    aff = np.arange(1000) < 300
    p_pa = MS.prob_A(W, pair, 0.9, 0.1, 0.1, {"rule": "privacy_always", "f": 0.3}, pos, aff)
    assert np.all(p_pa[aff] == 1.0) and np.allclose(p_pa[~aff], p_log[~aff])
    p_pa_lacr = MS.prob_A(W, np.full(1000, 2), 0.9, 0.1, 0.1, {"rule": "privacy_always"}, pos, aff)
    assert np.allclose(p_pa_lacr, MS.prob_A(W, np.full(1000, 2), 0.9, 0.1, 0.1, {"rule": "logistic"}, pos, None))
    p_lex = MS.prob_A(W, pair, 0.6, 0.4, 0.1, {"rule": "lexicographic"}, pos, None)
    assert np.array_equal(p_lex, (W[:, 0] > W[:, 1]).astype(float))
    sb = {"rule": "side_bias", "b": 1.0, "sides": "alternating"}
    z = (W[:, 0] - W[:, 1]) * 0.8 / 0.1
    assert np.allclose(MS.prob_A(W, pair, 0.9, 0.1, 0.1, sb, pos, None), 1 / (1 + np.exp(-(z + 1))))
    assert np.allclose(MS.prob_A(W, pair, 0.9, 0.1, 0.1, sb, pos + 1, None), 1 / (1 + np.exp(-(z - 1))))
    sf = dict(sb, sides="fixed")
    assert np.allclose(MS.prob_A(W, pair, 0.9, 0.1, 0.1, sf, pos + 1, None), 1 / (1 + np.exp(-(z + 1))))
