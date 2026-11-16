"""FG-MIC, moment release and their sensitivity bounds."""
import itertools
import math

import numpy as np
import pytest

from src import fgmic as F


def _mi(col, row):
    tab = np.zeros((col.max() + 1, row.max() + 1))
    np.add.at(tab, (col, row), 1)
    p = tab / tab.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum(p * np.log(p / (p.sum(1, keepdims=True) * p.sum(0, keepdims=True)))))


@pytest.mark.parametrize("seed", range(5))
def test_optimize_columns_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    cnt = rng.integers(0, 6, size=(7, 3)).astype(float)
    I = F.optimize_columns(cnt, 4)
    bins = np.repeat(np.arange(7), cnt.sum(1).astype(int))
    rows = np.concatenate([np.repeat(np.arange(3), cnt[b].astype(int)) for b in range(7)])
    nz = np.flatnonzero(cnt.sum(1) > 0)
    for k in range(2, 5):
        best = 0.0
        for cuts in itertools.combinations(range(1, len(nz)), min(k, len(nz)) - 1):
            lab = np.searchsorted(np.array(cuts), np.searchsorted(nz, bins), side="right")
            best = max(best, _mi(lab, rows))
        assert abs(I[k] - best) < 1e-10


def _setup(n, task, rng):
    xpub = rng.uniform(size=500)
    ypub = rng.integers(0, 2, 500).astype(float) if task == "classification" else rng.normal(size=500)
    cuts = F.public_cuts(xpub, 8)
    rows = F.row_partitions(task, ypub, max(n ** 0.55, 4), 6)
    return cuts, rows


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_fgmic_sensitivity_bound_holds(task):
    rng = np.random.default_rng(0 if task == "classification" else 1)
    n = 150
    cuts, rows = _setup(n, task, rng)
    bound = F.fgmic_sensitivity(n)
    worst = 0.0
    for _ in range(40):
        x = rng.uniform(size=n)
        y = (x + 0.3 * rng.normal(size=n) > 0.5).astype(float) if task == "classification" else x ** 2 + 0.2 * rng.normal(size=n)
        base = F.fgmic(x, y, cuts, rows)
        for _ in range(25):                                   # replace one record
            i = rng.integers(n)
            x2, y2 = x.copy(), y.copy()
            x2[i] = rng.choice([0.0, 1.0, rng.uniform()])
            y2[i] = rng.integers(0, 2) if task == "classification" else rng.choice([-3.0, 3.0, rng.normal()])
            worst = max(worst, abs(F.fgmic(x2, y2, cuts, rows) - base))
    assert worst <= bound
    assert worst > 0


def test_fgmic_detects_dependence_and_is_bounded():
    rng = np.random.default_rng(2)
    n = 5000
    x = rng.uniform(size=n)
    cuts = F.public_cuts(rng.uniform(size=1000), 32)
    rows = F.row_partitions("regression", np.sin(6 * rng.uniform(size=1000)), n ** 0.55, 16)
    dep = F.fgmic(x, np.sin(6 * x) + 0.05 * rng.normal(size=n), cuts, rows)
    ind = F.fgmic(x, rng.normal(size=n), cuts, rows)
    assert 0 <= ind < 0.05 < 0.5 < dep <= 1


def test_moment_sensitivity_bound_holds():
    rng = np.random.default_rng(3)
    n, d = 60, 4
    bound = F.moment_sensitivity(n, d)
    for _ in range(200):
        X = rng.uniform(size=(n, d))
        y = rng.uniform(size=n)
        X2, y2 = X.copy(), y.copy()
        i = rng.integers(n)
        X2[i] = rng.choice([0.0, 1.0], size=d)
        y2[i] = rng.choice([0.0, 1.0])
        assert np.linalg.norm(F.moment_vector(X, y) - F.moment_vector(X2, y2)) <= bound + 1e-12


def test_correlations_from_exact_moments():
    rng = np.random.default_rng(4)
    n = 50000
    X = rng.uniform(size=(n, 3))
    y = 0.6 * X[:, 0] + 0.2 * rng.uniform(size=n)
    r, pc, _ = F.correlations_from_moments(F.moment_vector(X, y), 3)
    assert abs(r[0] - abs(np.corrcoef(X[:, 0], y)[0, 1])) < 1e-6 and r[1] < 0.02 and pc[2] < 0.02
