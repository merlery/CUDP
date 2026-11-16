import numpy as np
import pytest

from src import prefselect as U

# Chapter 6, Table 6.2: closeness coefficients (linear LP) and eps* (linear, exponential)
TABLE62 = [
    ("adult", (0.6, 0.2, 0.2), (0.759, 0.772, 0.850, 0.546, 0.241), 1, 0.1),
    ("adult", (0.2, 0.6, 0.2), (0.259, 0.281, 0.605, 0.853, 0.741), 5, 10),
    ("adult", (1 / 3, 1 / 3, 1 / 3), (0.512, 0.532, 0.718, 0.683, 0.488), 1, 0.1),
    ("adult", (0.2, 0.2, 0.6), (0.512, 0.532, 0.718, 0.683, 0.488), 1, 0.1),
    ("cdc", (0.6, 0.2, 0.2), (0.730, 0.806, 0.853, 0.556, 0.270), 1, 0.1),
    ("cdc", (0.2, 0.6, 0.2), (0.231, 0.420, 0.657, 0.866, 0.769), 5, 10),
    ("cdc", (1 / 3, 1 / 3, 1 / 3), (0.474, 0.598, 0.737, 0.705, 0.526), 1, 0.3),
    ("cdc", (0.2, 0.2, 0.6), (0.474, 0.598, 0.737, 0.705, 0.526), 1, 0.3),
]


@pytest.mark.parametrize("ds,w,C,e_lin,e_exp", TABLE62)
def test_table_6_2(ds, w, C, e_lin, e_exp):
    idx, Cm = U.select(U.matrix(ds, "linear"), w)
    assert np.allclose(np.round(Cm[0], 3), C, atol=1e-12)
    assert U.CANDIDATES[idx[0]] == e_lin
    idx, _ = U.select(U.matrix(ds, "exponential"), w)
    assert U.CANDIDATES[idx[0]] == e_exp


def test_switch_points_adult_linear():
    """Chapter 6 reports the switches on a grid of step 0.001 of the privacy share (w_CR = 0): the first
    grid value at which the new budget is selected is 0.013, 0.472, 0.887 and 0.946."""
    M = U.matrix("adult", "linear")
    g = np.round(np.arange(0, 1.0005, 0.001), 3)
    idx, _ = U.select(M, np.column_stack([g, 1 - g, np.zeros_like(g)]))
    ch = np.nonzero(np.diff(idx))[0] + 1
    assert [round(float(g[i]), 3) for i in ch] == [0.013, 0.472, 0.887, 0.946]
    assert [float(U.CANDIDATES[idx[i]]) for i in ch] == [5.0, 1.0, 0.3, 0.1]


def test_scaling_invariance():
    rng = np.random.default_rng(0)
    W = rng.dirichlet([1, 1, 1], size=200)
    for ds, lp in U.SETTINGS:
        M = U.matrix(ds, lp)
        C1 = U.closeness(M, W)
        for c in (0.01, 3.0, 250.0):
            assert np.allclose(U.closeness(M, c * W), C1, atol=1e-12)


def test_undefined_and_zero_column_and_ties():
    M = U.matrix("adult", "linear")
    idx, C = U.select(M, (0.0, 0.0, 1.0))
    assert idx[0] == U.UNDEFINED and np.all(np.isnan(C))
    # a column that is zero for every candidate stays zero
    Z = np.column_stack([np.zeros(5), U.LA["adult"], np.ones(5)])
    assert np.all(U.normalize(Z)[:, 0] == 0)
    # ties: identical rows -> smallest eps (first index)
    T = np.array([[0.5, 0.5, 1.0], [0.5, 0.5, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.2, 0.2, 1.0]])
    idx, C = U.select(T, (0.5, 0.5, 0.0))
    assert C[0, 0] == C[0, 1] and idx[0] == 0


def test_privacy_share():
    assert np.allclose(U.privacy_share([[0.6, 0.2, 0.2], [0, 0, 1], [0.2, 0.6, 0.2]]), [0.75, 0.5, 0.25])
