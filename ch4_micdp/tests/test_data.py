import numpy as np
import pytest

from src import data

CFG = data.load_config()


def test_clip_scale_maps_range_to_unit_interval():
    X = np.array([[-5.0, 0.0], [0.0, 5.0], [10.0, 20.0]])
    Z = data.clip_scale(X, np.array([0.0, 0.0]), np.array([5.0, 10.0]))
    assert np.allclose(Z, [[0, 0], [0, 0.5], [1, 1]])


def test_split_proportions_and_disjoint():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=10000) < 0.25).astype(int)
    pub, tr, te = data.split_indices(10000, y, "classification", CFG)
    assert len(set(pub) | set(tr) | set(te)) == 10000
    assert not (set(pub) & set(tr) or set(pub) & set(te) or set(tr) & set(te))
    assert abs(len(pub) / 1e4 - 0.15) < 0.002 and abs(len(te) / 1e4 - 0.20) < 0.002
    for idx in (pub, tr, te):                  # stratified
        assert abs(y[idx].mean() - y.mean()) < 0.01


@pytest.mark.parametrize("name", list(CFG["datasets"].keys()))
def test_prepared_dataset(name):
    p = data.prepare(name, CFG)
    for X in (p.X_public, p.X_train, p.X_test):
        assert X.min() >= 0 and X.max() <= 1 and X.shape[1] == p.d
    assert CFG["privacy"]["delta"] < 1 / len(p.idx_train)
    assert len(p.idx_public) + len(p.idx_train) + len(p.idx_test) == p.n_total
    assert p.n_total <= CFG["data"]["max_rows"]
    # fallback ranges equal the public-split percentiles (never private-train or test)
    q = CFG["ranges"]["fallback_percentiles"]
    X, y, _ = data.load_frame(name, CFG)
    if p.n_raw > CFG["data"]["max_rows"]:
        rng = np.random.default_rng(CFG["seeds"]["subsample"])
        keep = np.sort(rng.choice(p.n_raw, size=CFG["data"]["max_rows"], replace=False))
        X = X.iloc[keep].reset_index(drop=True)
    for j, (f, s) in enumerate(zip(p.features, p.range_sources)):
        if s == "public-split":
            lo, hi = np.percentile(X.iloc[p.idx_public][f].to_numpy(), q)
            assert np.isclose(p.a[j], lo) and np.isclose(p.b[j], hi)


def test_synthetic_variants_share_x_and_splits():
    v = data.make_synthetic(CFG)
    X0 = v["syn_lin"][0].to_numpy()
    for k in v:
        assert np.array_equal(v[k][0].to_numpy(), X0)
    p1, p2 = data.prepare("syn_nl_low", CFG), data.prepare("syn_nl_high", CFG)
    assert np.array_equal(p1.idx_test, p2.idx_test)
    assert np.array_equal(p1.X_train, p2.X_train)
    C = np.corrcoef(X0[:, :8], rowvar=False)
    assert abs(C[0, 1] - 0.5) < 0.03 and abs(C[0, 4]) < 0.03
