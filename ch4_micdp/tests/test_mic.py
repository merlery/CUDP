"""Unit tests for the own MIC implementation (src/mic.py). No external MIC library is used."""
import itertools
import math

import numpy as np
import pytest

from src import mic as M


def _mi_of_partition(col: np.ndarray, row: np.ndarray) -> float:
    n = col.size
    tab = np.zeros((col.max() + 1, row.max() + 1))
    np.add.at(tab, (col, row), 1)
    p = tab / n
    pc, pr = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = p * np.log(p / (pc * pr))
    return float(np.nansum(v))


def test_equipartition_balanced_and_keeps_ties():
    rng = np.random.default_rng(0)
    v = np.sort(rng.uniform(size=1000))
    lab = M.equipartition(v, 7)
    sizes = np.bincount(lab)
    assert sizes.size == 7 and sizes.max() - sizes.min() <= 1
    v2 = np.sort(rng.integers(0, 20, size=500).astype(float))
    lab2 = M.equipartition(v2, 6)
    for val in np.unique(v2):                       # tied values share a bin
        assert np.unique(lab2[v2 == val]).size == 1
    assert np.all(np.diff(lab2) >= 0)


def test_clumps_respect_rows_and_ties():
    x = np.array([0., 1, 2, 3, 3, 4, 5, 6])
    q = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    p = M.clumps(x, q)
    # 0,1 -> clump A ; 2 -> B (row 1) ; tie group {3,3} mixed rows -> own clump ; 4 -> row 0 ; 5,6 -> row 1
    assert p.tolist() == [0, 0, 1, 2, 2, 3, 4, 4]


@pytest.mark.parametrize("seed", range(6))
def test_optimize_x_axis_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n, n_sc, rows = 60, 7, 3
    p = np.sort(rng.integers(0, n_sc, size=n))
    p = np.unique(p, return_inverse=True)[1]          # consecutive ids 0..P-1
    q = rng.integers(0, rows, size=n)
    P = p.max() + 1
    I = M.optimize_x_axis(p, q, 4)
    for k in range(2, 5):
        best = 0.0
        for cuts in itertools.combinations(range(1, P), min(k, P) - 1):
            col = np.searchsorted(np.array(cuts), p, side="right")
            best = max(best, _mi_of_partition(col, q))
        assert abs(I[k] - best) < 1e-10


def test_noiseless_functions_score_one():
    x = np.linspace(-1, 1, 1000)
    for f in (lambda t: 2 * t + 1, lambda t: t ** 3, np.exp, lambda t: t ** 2):
        assert M.mic(x, f(x)) > 0.99


def test_independent_is_small_and_dependence_is_detected():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=3000)
    assert M.mic(x, rng.uniform(size=3000)) < 0.05
    assert M.mic(x, np.sin(8 * x) + 0.2 * rng.normal(size=3000)) > 0.4


def test_symmetric_range_and_rank_invariant():
    rng = np.random.default_rng(2)
    x = rng.normal(size=1500)
    y = x ** 2 + rng.normal(size=1500)
    m = M.mic(x, y)
    assert 0 <= m <= 1
    assert M.mic(y, x) == pytest.approx(m, abs=1e-12)
    assert M.mic(np.exp(x), y ** 3) == pytest.approx(m, abs=1e-12)   # strictly monotone maps


def test_mic_e_le_mic_approx_and_constant_columns():
    rng = np.random.default_rng(3)
    x = rng.uniform(size=2000)
    y = x + 0.5 * rng.normal(size=2000)
    assert M.mic(x, y, est="mic_e") <= M.mic(x, y, alpha=0.6, c=15, est="mic_approx") + 1e-12
    assert M.mic(np.ones(100), rng.normal(size=100)) == 0.0


def test_alpha_rule():
    assert M.alpha_for_n(3000) == 0.55 and M.alpha_for_n(2961) == 0.55
    assert M.alpha_for_n(10) == 0.85 and M.alpha_for_n(50000) == 0.4
