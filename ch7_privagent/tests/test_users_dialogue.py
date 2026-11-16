import numpy as np
import yaml

from src import dialogue as DLG
from src import users as USR
from src.run import ROOT

CFG = yaml.safe_load(open(ROOT / "config.yaml"))


def test_response_model_frequency():
    rng = np.random.default_rng(1)
    W = np.tile([[0.5, 0.3, 0.2]], (200000, 1))
    p = USR.choice_prob(W, (0, 1), 0.9, 0.1, 0.1)[0]
    assert np.isclose(p, 1 / (1 + np.exp(-(0.2 * 0.8) / 0.1)))
    a = USR.answer(W, (0, 1), 0.9, 0.1, 0.1, rng.random(200000))
    assert abs(a.mean() - p) < 0.005


def test_slider():
    W = np.array([[0.7, 0.2, 0.1], [0.05, 0.05, 0.9]])
    assert np.allclose(USR.anchored_slider(W, 0.0, [1 / 3] * 3, np.zeros_like(W)), W)
    eta = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.0]])
    out = USR.anchored_slider(W, 0.0, [1 / 3] * 3, eta)
    assert out[0, 2] == 0 and np.allclose(out.sum(1), 1)
    out = USR.anchored_slider(W, 0.4, [1 / 3] * 3, np.zeros_like(W))
    assert np.allclose(out, 0.4 / 3 + 0.6 * W)


def test_profile_population_mean():
    rng = np.random.default_rng(2)
    W = USR.draw_weights(rng, [12, 4, 4], 100000)
    assert np.allclose(W.mean(0), [0.6, 0.2, 0.2], atol=0.005)


def test_prompt_sets():
    p6 = DLG.prompt_set(CFG, "pa6")
    assert sorted(map(tuple, p6.tolist())) == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    pl = DLG.prompt_set(CFG, "lpla")
    assert (pl[:, 0] == 0).all() and np.bincount(pl[:, 1]).tolist() == [2, 2, 2]


def test_cycle_and_repair_pair():
    res = np.array([[1, -1, 1], [-1, 1, -1], [1, 1, 1], [1, 0, 1], [0, 0, 0]])
    assert DLG.is_cycle(res).tolist() == [True, True, False, False, False]
    assert DLG.repair_pair(res).tolist() == [0, 0, 0, 1, 0]


def test_pair_results():
    pairs = np.array([[0, 0, 1, 1, 2, 2]])
    ans = np.array([[1, 1, 0, 0, 1, 0]])
    assert DLG.pair_results(pairs, ans).tolist() == [[1, -1, 0]]


def test_dialogue_repair_triggered_only_on_cycles_and_truncation():
    rng = np.random.default_rng(3)
    W = rng.dirichlet([1, 1, 1], size=5000)
    d = DLG.run(W, CFG, "pa6", 0.4, np.random.default_rng(4), repair=True)
    res = DLG.pair_results(d["pair"][:, :6], d["answer"][:, :6])
    assert np.array_equal(d["repair"], DLG.is_cycle(res))
    assert np.array_equal(d["used"][:, 6], d["repair"]) and d["repair"].any()
    assert (d["pair"][d["repair"], 6] == 0).all() and (d["contrast"][:, 6] == 0).all()
    t = DLG.truncate(d, 3)
    assert (t["used"].sum(1) == 3).all() and not t["repair"].any()
    # same stream -> same answers (coupling across calls)
    d2 = DLG.run(W, CFG, "pa6", 0.4, np.random.default_rng(4), repair=True)
    assert np.array_equal(d["answer"], d2["answer"])
