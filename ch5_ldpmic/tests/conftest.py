"""Shared fixtures.

Everything here is SYNTHETIC test data (Gaussian clusters) and exists only to
exercise the code paths.  It must never be written under results/ and is not
used by any experiment script.
"""

from __future__ import annotations

import numpy as np
import pytest


def synthetic_gaussian_dataset(m: int, d: int, n_classes: int, seed: int, informative: int = 3,
                               centers_seed: int = 0):
    """Synthetic Gaussian blobs: the first ``informative`` features carry the
    label, the rest are noise.  The class centers depend only on
    ``centers_seed`` (shared by all clients and the test set); ``seed`` draws
    the samples.  Features live roughly in [-4, 4]; the public range used in
    tests is [-5, 5]."""
    centers = np.random.default_rng(centers_seed).uniform(-2.5, 2.5, size=(n_classes, informative))
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=m)
    X = rng.standard_normal((m, d))
    X[:, :informative] += centers[y]
    X = np.clip(X, -4.5, 4.5)
    return X, y


@pytest.fixture
def synthetic_tabular():
    """(X, y, ranges, n_classes) with m=600, d=8, binary labels."""
    X, y = synthetic_gaussian_dataset(600, 8, 2, seed=0)
    ranges = np.tile(np.array([[-5.0, 5.0]]), (8, 1))
    return X, y, ranges, 2


@pytest.fixture
def synthetic_federated():
    """4 synthetic clients with unequal sizes + a synthetic test set."""
    sizes = [120, 80, 150, 100]
    clients = [synthetic_gaussian_dataset(m, 6, 3, seed=10 + i) for i, m in enumerate(sizes)]
    X_test, y_test = synthetic_gaussian_dataset(300, 6, 3, seed=99)
    ranges = np.tile(np.array([[-5.0, 5.0]]), (6, 1))
    return clients, (X_test, y_test), ranges, 3
