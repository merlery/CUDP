"""Dataset-independent tests of ldpmic.data (partitions, encoding, scaler).
Uses small synthetic frames only; no download."""
import numpy as np
import pandas as pd
import pytest

from ldpmic.data import common
from ldpmic.data.common import (CategoricalFeature, NumericFeature, dirichlet_partition, encode,
                                group_partition, public_scaler)


def test_encode_clips_and_one_hots_on_documented_categories():
    df = pd.DataFrame({"a": [-5.0, 0.5, 100.0], "c": ["x", "y ", "zzz"]})
    feats = [NumericFeature("a", 0, 10, "declared"), CategoricalFeature("c", ("x", "y"))]
    X, ranges, names = encode(df, feats)
    np.testing.assert_array_equal(X[:, 0], [0.0, 0.5, 10.0])
    np.testing.assert_array_equal(X[:, 1:], [[1, 0], [0, 1], [0, 0]])  # unlisted value -> all zeros
    np.testing.assert_array_equal(ranges, [[0, 10], [0, 1], [0, 1]])
    assert names == ["a", "c=x", "c=y"]
    f = public_scaler(ranges)
    np.testing.assert_allclose(f(X)[:, 0], [0.0, 0.05, 1.0])


def test_group_partition_covers_everything_one_group_per_client():
    rng = np.random.default_rng(0)
    groups = rng.choice(["a", "b", "c", "d"], size=1000, p=[0.5, 0.3, 0.15, 0.05])
    parts = group_partition(groups, 10, rng)
    assert len(parts) == 10
    allidx = np.concatenate(parts)
    assert sorted(allidx.tolist()) == list(range(1000))
    for p in parts:
        assert len(p) > 0 and len(set(groups[p])) == 1
    # every group has at least one client; big groups get more
    owners = [groups[p][0] for p in parts]
    assert set(owners) == {"a", "b", "c", "d"}
    assert owners.count("a") > owners.count("d")
    with pytest.raises(ValueError):
        group_partition(groups, 3, rng)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_dirichlet_partition_skews_labels_and_is_seeded(n_classes):
    y = np.random.default_rng(1).integers(0, n_classes, size=3000)
    p1 = dirichlet_partition(y, 50, 0.1, np.random.default_rng(5), min_size=10)
    p2 = dirichlet_partition(y, 50, 0.1, np.random.default_rng(5), min_size=10)
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))
    assert sorted(np.concatenate(p1).tolist()) == list(range(3000))
    assert min(len(p) for p in p1) == 60 and max(len(p) for p in p1) == 60
    # alpha=0.1 gives much more skew than alpha=100
    def skew(parts):
        return np.mean([np.bincount(y[p], minlength=n_classes).max() / len(p) for p in parts])
    p3 = dirichlet_partition(y, 50, 100.0, np.random.default_rng(5))
    assert skew(p1) > 0.9 > skew(p3) + 0.2


def test_make_partition_caches_by_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA_DIR", tmp_path)
    y = np.random.default_rng(0).integers(0, 2, size=500)
    ds = common.TabularDataset("synthetic_test", np.zeros((500, 2)), y, np.zeros((5, 2)), np.zeros(5),
                               np.array([[0, 1], [0, 1]]), ["a", "b"], np.array(["g"] * 250 + ["h"] * 250), "g", 2, {})
    a = common.make_partition(ds, "dirichlet", 5, seed=1, alpha=0.5)
    assert (tmp_path / "partitions").exists()
    b = common.make_partition(ds, "dirichlet", 5, seed=1, alpha=0.5)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    c = common.make_partition(ds, "dirichlet", 5, seed=2, alpha=0.5)
    assert not all(np.array_equal(x, z) for x, z in zip(a, c))
    g = common.make_partition(ds, "group", 4, seed=1)
    assert len(g) == 4 and sorted(np.concatenate(g).tolist()) == list(range(500))


def test_download_verifies_sha256(tmp_path, monkeypatch):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    good = common.sha256_file(f)
    assert common.download("unused://", f, good) == f
    with pytest.raises(RuntimeError):
        common.download("unused://", f, "0" * 64)
    monkeypatch.delenv("LDPMIC_ALLOW_UNVERIFIED", raising=False)
    with pytest.raises(RuntimeError):
        common.download("unused://", f, None)


def test_kdd_schema_parsing_and_encoding():
    """The KDD loader takes attribute lists and category values from the
    documentation file; dropped attributes vanish, numerics get public ranges."""
    import pandas as pd
    from ldpmic.data import kdd
    names = ("| header comment\n"
             "age: continuous.\n"
             "class of worker: Not in universe, Private, Never worked.\n"
             "detailed industry recode: 0, 40, 44.\n"
             "sex: Female, Male.\n"
             "weeks worked in year: continuous.\n"
             "year: 94, 95.\n")
    schema = kdd.parse_schema(names)
    assert list(schema) == ["age", "class of worker", "detailed industry recode", "sex", "weeks worked in year", "year"]
    assert schema["age"] is None and schema["sex"] == ["Female", "Male"]
    feats = kdd.build_features(schema)
    assert [f.name for f in feats] == ["age", "class of worker", "sex", "weeks worked in year"]   # recode and year dropped
    df = pd.DataFrame({"age": [30.0, 95.0], "class of worker": ["Private", "Self-employed"], "sex": ["Female", "Male"],
                       "weeks worked in year": [52.0, 60.0]})
    X, ranges, cols = kdd.encode(df, feats)
    assert cols[:3] == ["age", "class of worker=Not in universe", "class of worker=Private"]
    assert X[1, 0] == 90.0 and X[1, cols.index("weeks worked in year")] == 52.0       # clipped to the public range
    assert X[1, 1:4].sum() == 0.0                                                  # undocumented category -> all zeros
    assert ranges[0].tolist() == [0.0, 90.0]


def test_cdc_encoding():
    """CDC: binary indicators and scales clipped to the documented ranges, BMI to
    its declared range, Age / Income / Sex one-hot (Sex 0 = female, 1 = male)."""
    import pandas as pd
    from ldpmic.data import cdc
    from ldpmic.data.common import encode
    row = {b: 1.0 for b in cdc.BINARY}
    rows = [dict(row, BMI=120.0, GenHlth=5.0, MentHlth=40.0, PhysHlth=0.0, Education=6.0, Age=13.0, Income=1.0, Sex=0),
            dict(row, BMI=5.0, GenHlth=1.0, MentHlth=3.0, PhysHlth=30.0, Education=1.0, Age=1.0, Income=8.0, Sex=1)]
    X, ranges, cols = encode(cdc.prepare(pd.DataFrame(rows)), cdc.FEATURES)
    assert X.shape == (2, len(cdc.BINARY) + 5 + 13 + 8 + 2) == (2, 41)
    assert X[0, cols.index("BMI")] == 100.0 and X[1, cols.index("BMI")] == 10.0
    assert X[0, cols.index("MentHlth")] == 30.0
    assert X[0, cols.index("Age=13")] == 1.0 and X[0, [i for i, c in enumerate(cols) if c.startswith("Age=")]].sum() == 1.0
    assert X[0, cols.index("Sex=Female")] == 1.0 and X[1, cols.index("Sex=Male")] == 1.0
    assert X[1, cols.index("Income=8")] == 1.0
    assert ranges[cols.index("BMI")].tolist() == [10.0, 100.0]
