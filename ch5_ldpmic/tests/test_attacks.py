"""Attack code on SYNTHETIC data only (tmp_path; nothing reaches results/)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ldpmic.attacks import attribute_inference as ai
from ldpmic.attacks import membership as mi
from ldpmic.models import MLP

from .conftest import synthetic_gaussian_dataset
from .test_run_sweep import _load, synthetic_dataset  # noqa: F401  (fixture re-export)

ROOT = Path(__file__).resolve().parents[1]


def _sensitive_dataset(n_clients=8, m=150, seed=0):
    """Synthetic clients with a binary one-hot attribute `sex` that is
    predictable from feature 0 (and nothing else)."""
    rng = np.random.default_rng(seed)
    raw = []
    for i in range(n_clients):
        x = rng.standard_normal((m, 4))
        s = (x[:, 0] + 0.3 * rng.standard_normal(m) > 0).astype(float)
        X = np.concatenate([x, np.stack([1 - s, s], axis=1)], axis=1)  # cols: f0..f3, sex=F, sex=M
        raw.append(X)
    names = ["f0", "f1", "f2", "f3", "sex=F", "sex=M"]
    ranges = np.array([[-5, 5]] * 4 + [[0, 1]] * 2, dtype=float)
    return raw, names, ranges


def test_attribute_inference_reads_signal_and_noise_destroys_it():
    raw, names, ranges = _sensitive_dataset()
    rng = np.random.default_rng(1)
    # "sanitized" copies: tiny noise -> attacker works; huge noise on f0 -> attacker ~ majority
    small = [X + 0.01 * rng.standard_normal(X.shape) for X in raw]
    big = [X + np.array([50, 0.01, 0.01, 0.01, 0.01, 0.01]) * rng.standard_normal(X.shape) for X in raw]
    sig_small = [np.full(6, 0.01)] * len(raw)
    sig_big = [np.array([50, 0.01, 0.01, 0.01, 0.01, 0.01])] * len(raw)
    r1 = ai.attribute_inference(raw, small, names, "sex", ranges, sig_small, seed=0)
    r2 = ai.attribute_inference(raw, big, names, "sex", ranges, sig_big, seed=0)
    assert r1.n_classes == 2 and r1.n_train + r1.n_eval == 8 * 150
    assert r1.accuracy > 0.85 and r1.balanced_accuracy > 0.85
    assert r1.raw_accuracy > 0.85
    assert r2.accuracy < r1.accuracy - 0.2
    assert abs(r2.balanced_accuracy - 0.5) < 0.1
    assert set(r1.snr) == {"sex=F", "sex=M"} and r1.snr["sex=F"] == pytest.approx(100.0)
    # the attribute's own columns are excluded from the attacker's input:
    # a sanitized copy where ONLY the sex columns are clean must not help
    leak = [X.copy() for X in big]
    r3 = ai.attribute_inference(raw, leak, names, "sex", ranges, sig_big, seed=0)
    assert abs(r3.accuracy - r2.accuracy) < 1e-9


def test_attribute_labels_and_missing_values():
    X = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
    np.testing.assert_array_equal(ai.attribute_labels(X, [0, 1]), [0, 1, -1])
    with pytest.raises(ValueError):
        ai.attribute_columns(["a", "b"], "sex")


def test_membership_attack_detects_overfit_model():
    X, y = synthetic_gaussian_dataset(300, 6, 2, seed=0)
    Xn, yn = synthetic_gaussian_dataset(300, 6, 2, seed=1)
    rng = np.random.default_rng(0)
    y, yn = rng.integers(0, 2, 300), rng.integers(0, 2, 300)  # random labels: only memorization separates members
    torch.manual_seed(0)
    model = MLP(6, 2, hidden=(64, 64))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
    for _ in range(400):  # memorize the members
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(model(Xt), yt); loss.backward(); opt.step()
    res = mi.loss_threshold_attack(model, X, y, Xn, yn, seed=0)
    assert res.n_members == 300 and res.auc > 0.7 and res.advantage > 0.3
    assert res.member_loss_mean < res.nonmember_loss_mean
    fresh = MLP(6, 2)
    res0 = mi.loss_threshold_attack(fresh, X, y, Xn, yn, seed=0, n=200)
    assert res0.n_members == 200 and abs(res0.auc - 0.5) < 0.15


def test_run_attacks_and_cap_resolution_end_to_end(tmp_path, monkeypatch, synthetic_dataset):  # noqa: F811
    """sensitive_prefixes / p_max=uniform resolve through run_sweep; the
    attribute attack rebuilds the same sanitized data as the FL run; the
    membership attack loads the saved model."""
    from ldpmic.data import common
    rs = _load("run_sweep")
    ra = _load("run_attacks")
    ds = synthetic_dataset
    # give the synthetic dataset a one-hot "sex" attribute in its last two columns
    rng = np.random.default_rng(0)
    for attr in ("X_train", "X_test"):
        X = getattr(ds, attr)
        s = (X[:, 0] > 0).astype(float)
        setattr(ds, attr, np.concatenate([X[:, :4], np.stack([1 - s, s], axis=1)], axis=1))
    ds.feature_names = ["f0", "f1", "f2", "f3", "sex=F", "sex=M"]
    monkeypatch.setattr(rs, "RUNLOG", tmp_path / "RUNLOG.md")
    monkeypatch.setattr(ra.rs, "RUNLOG", tmp_path / "RUNLOG.md")
    monkeypatch.setattr(rs, "get_dataset", lambda name: ds)
    monkeypatch.setattr(ra.rs, "get_dataset", lambda name: ds)
    monkeypatch.setattr(common, "DATA_DIR", tmp_path / "cache")
    monkeypatch.setattr(rs.ldata, "DATA_DIR", tmp_path / "cache")
    out = tmp_path / "raw"
    common_args = ["--seeds", "0", "--eps", "1", "--results", str(out), "-o", "fl.rounds=2", "-o", "n_clients=4",
                   "-o", "fl.lr=0.05", "-o", "fl.participation=1.0"]
    cap = str(ROOT / "configs" / "adult_occ_ldpmic_global.yaml")
    assert rs.main([cap] + common_args + ["-o", "method.sensitive_prefixes=[sex]", "-o", "method.p_max=uniform"]) == 0
    prov = json.loads(next(p for p in out.glob("*.json") if "ldpmic" in p.name).read_text())
    assert prov["sensitive_indices"] == [4, 5] and prov["p_max"] == pytest.approx(1 / 6)
    assert all(prov["privacy"]["shares"]["global"][j] <= 1 / 6 + 1e-12 for j in (4, 5))
    # F instrumentation present
    assert prov["privacy"]["phase1_time_s"] > 0 and prov["privacy"]["phase3_time_s"] > 0
    assert prov["privacy"]["phase1_bytes_per_client"] == 6 * 8
    import pandas as pd
    df = pd.read_csv(next(p for p in out.glob("*.csv") if "ldpmic" in p.name))
    assert {"client_time_mean_s", "client_time_max_s", "eval_time_s"} <= set(df.columns) and df.client_time_mean_s.gt(0).all()
    # attribute attack on the same config: must match the FL run's sanitized hashes
    assert ra.main(["attribute", "--configs", cap, "--attributes", "sex"] + common_args
                   + ["-o", "method.sensitive_prefixes=[sex]", "-o", "method.p_max=uniform"]) == 0
    ap = json.loads(next(out.glob("attack_attribute_*.json")).read_text())
    assert ap["status"] == "completed" and ap["matches_fl_run"] is True
    assert set(ap["results"]["sex"]["snr"]) == {"sex=F", "sex=M"}
    # membership: needs save_model
    nodp = str(ROOT / "configs" / "adult_occ_nodp.yaml")
    assert rs.main([nodp] + common_args + ["-o", "save_model=true"]) == 0
    assert ra.main(["membership", "--configs", nodp] + common_args + ["-o", "save_model=true"]) == 0
    mp = json.loads(next(out.glob("attack_membership_*.json")).read_text())
    assert mp["status"] == "completed" and 0 <= mp["results"]["auc"] <= 1
    # aggregate_attacks runs on the directory
    ag = _load("aggregate_attacks")
    assert ag.main(["--raw", str(out), "--out", str(tmp_path / "summary")]) == 0
    assert (tmp_path / "summary" / "attribute_inference.csv").exists()
    assert (tmp_path / "summary" / "membership.csv").exists()


def test_attribute_inference_on_selected_release():
    """Selected release: the attacker only sees the released columns; the
    attribute's own columns are reported as not released when absent."""
    raw, names, ranges = _sensitive_dataset()
    rng = np.random.default_rng(2)
    released = [0, 2, 3]                       # f0 released (the signal), sex columns not released
    san = [X[:, released] + 0.01 * rng.standard_normal((X.shape[0], 3)) for X in raw]
    sig = [np.full(3, 0.01)] * len(raw)
    res = ai.attribute_inference(raw, san, names, "sex", ranges, sig, seed=0, released=released)
    assert res.released_features == released and not res.attribute_released
    assert res.snr == {"sex=F": None, "sex=M": None}
    assert res.accuracy > 0.8 and res.raw_accuracy > 0.8      # f0 carries the attribute
    # without f0 the attacker is at the majority rate
    released2 = [1, 2, 3, 4, 5]
    san2 = [X[:, released2] + 0.01 * rng.standard_normal((X.shape[0], 5)) for X in raw]
    sig2 = [np.full(5, 0.01)] * len(raw)
    res2 = ai.attribute_inference(raw, san2, names, "sex", ranges, sig2, seed=0, released=released2)
    assert res2.attribute_released and res2.snr["sex=F"] == pytest.approx(100.0)
    assert abs(res2.accuracy - res2.majority_accuracy) < 0.1


def test_attribute_inference_direct_and_all_channels():
    """The direct channel reads the attribute's own noisy columns (clean ->
    near-perfect; noise of 50x the range -> near majority); the all-column
    channel is at least as strong as the proxy channel on clean data."""
    raw, names, ranges = _sensitive_dataset()
    rng = np.random.default_rng(3)
    clean_sex = [X + np.array([50, 50, 50, 50, 0.01, 0.01]) * rng.standard_normal(X.shape) for X in raw]
    sig = [np.array([50, 50, 50, 50, 0.01, 0.01])] * len(raw)
    r = ai.attribute_inference(raw, clean_sex, names, "sex", ranges, sig, seed=0)
    assert r.direct_balanced_accuracy > 0.95 and abs(r.balanced_accuracy - 0.5) < 0.1   # proxies destroyed, own columns clean
    assert r.all_balanced_accuracy > 0.9
    noisy_sex = [X + np.array([0.01, 0.01, 0.01, 0.01, 50, 50]) * rng.standard_normal(X.shape) for X in raw]
    sig2 = [np.array([0.01, 0.01, 0.01, 0.01, 50, 50])] * len(raw)
    r2 = ai.attribute_inference(raw, noisy_sex, names, "sex", ranges, sig2, seed=0)
    assert abs(r2.direct_balanced_accuracy - 0.5) < 0.1 and r2.balanced_accuracy > 0.85
    # selected release without the attribute's columns: no direct channel
    rel = [0, 2, 3]
    san = [X[:, rel] + 0.01 * rng.standard_normal((X.shape[0], 3)) for X in raw]
    r3 = ai.attribute_inference(raw, san, names, "sex", ranges, [np.full(3, 0.01)] * len(raw), seed=0, released=rel)
    assert r3.direct_balanced_accuracy is None and r3.all_balanced_accuracy > 0.8
