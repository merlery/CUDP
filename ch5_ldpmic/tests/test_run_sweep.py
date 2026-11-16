"""End-to-end test of scripts/run_sweep.py + scripts/aggregate.py on a
SYNTHETIC in-memory dataset (monkeypatched loader).  Everything is written
under pytest's tmp_path: nothing reaches results/ or RUNLOG.md."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ldpmic.data import common

from .conftest import synthetic_gaussian_dataset

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def synthetic_dataset():
    X, y = synthetic_gaussian_dataset(600, 6, 2, seed=1)
    Xt, yt = synthetic_gaussian_dataset(200, 6, 2, seed=2)
    ranges = np.tile([[-5.0, 5.0]], (6, 1))
    groups = np.array(["g%d" % (i % 3) for i in range(600)])
    return common.TabularDataset("synthetic_test", X, y, Xt, yt, ranges, [f"f{i}" for i in range(6)],
                                 groups, "g", 2, {"files": {}})


def test_run_sweep_and_aggregate_end_to_end(tmp_path, monkeypatch, synthetic_dataset):
    rs = _load("run_sweep")
    ag = _load("aggregate")
    monkeypatch.setattr(rs, "RUNLOG", tmp_path / "RUNLOG.md")
    monkeypatch.setattr(rs, "get_dataset", lambda name: synthetic_dataset)
    monkeypatch.setattr(common, "DATA_DIR", tmp_path / "cache")
    monkeypatch.setattr(rs.ldata, "DATA_DIR", tmp_path / "cache")
    out = tmp_path / "raw"
    configs = [str(ROOT / "configs" / f"adult_occ_{m}.yaml") for m in
               ("nodp", "iso", "ldpmic_global", "ldpmic_per_client", "local_dpsgd")]
    argv = configs + ["--seeds", "0", "1", "--eps", "1", "--results", str(out),
                      "-o", "fl.rounds=2", "-o", "n_clients=3", "-o", "fl.lr=0.05", "-o", "fl.participation=1.0"]
    assert rs.main(argv) == 0
    jsons = sorted(out.glob("*.json"))
    assert len(jsons) == 10 and len(list(out.glob("*.csv"))) == 10
    for jp in jsons:
        prov = json.loads(jp.read_text())
        assert prov["status"] == "completed"
        for k in ("config", "seed", "config_hash", "git_commit", "start_time", "end_time", "host", "final", "privacy"):
            assert k in prov
        assert "hostname" not in json.dumps(prov["host"]).lower()
        df = pd.read_csv(jp.with_suffix(".csv"))
        assert len(df) == 2 and {"round", "accuracy", "balanced_accuracy", "auc", "train_loss", "wall_clock_s",
                                 "bytes_per_client"} <= set(df.columns)
        m = prov["config"]["method"]["method"]
        pv = prov["privacy"]
        if m == "nodp":
            assert pv["private"] is False
        elif m in ("iso", "ldpmic"):
            assert pv["eps_realized"] == pytest.approx(1.0) and pv["delta_realized"] == pytest.approx(1e-5)
            assert len(pv["sigmas"]) == 3 and all(len(v) == 6 for v in pv["sigmas"].values())
        elif m == "local_dpsgd":
            assert 0 < pv["eps_realized"] <= 1.0 + 1e-9
    log = (tmp_path / "RUNLOG.md").read_text()
    assert log.count("| completed |") == 10
    # resumable: a second invocation skips everything
    assert rs.main(argv) == 0
    assert (tmp_path / "RUNLOG.md").read_text().count("| completed |") == 10
    # a changed config (different hash) is a new run, not a skip
    assert rs.main(argv[:-1] + ["fl.participation=0.5"]) == 0
    assert len(list(out.glob("*.json"))) == 20
    # aggregate
    summ_dir = tmp_path / "summary"
    assert ag.main(["--raw", str(out), "--out", str(summ_dir)]) == 0
    final = pd.read_csv(summ_dir / "adult_final.csv")
    assert set(final.n_seeds) == {2}
    assert {"accuracy_mean", "accuracy_std", "auc_mean", "eps_realized"} <= set(final.columns)
    assert len(final) == 10  # 5 methods x 2 participation settings
    assert (summ_dir / "adult_curves.csv").exists()


def test_run_sweep_records_failures(tmp_path, monkeypatch, synthetic_dataset):
    rs = _load("run_sweep")
    monkeypatch.setattr(rs, "RUNLOG", tmp_path / "RUNLOG.md")
    monkeypatch.setattr(rs, "get_dataset", lambda name: synthetic_dataset)
    monkeypatch.setattr(common, "DATA_DIR", tmp_path / "cache")
    monkeypatch.setattr(rs.ldata, "DATA_DIR", tmp_path / "cache")
    out = tmp_path / "raw"
    cfg = str(ROOT / "configs" / "adult_occ_nodp.yaml")
    # lr unset -> refuses to run and records the failure
    assert rs.main([cfg, "--seeds", "0", "--results", str(out), "-o", "n_clients=3", "-o", "fl.rounds=1", "-o", "fl.lr=null"]) == 1
    prov = json.loads(next(out.glob("*.json")).read_text())
    assert prov["status"] == "failed" and "fl.lr is not set" in prov["traceback"]
    assert "| failed |" in (tmp_path / "RUNLOG.md").read_text()


def test_score_l1_diagnostics_and_vector_files(tmp_path, monkeypatch, synthetic_dataset):
    """ldpmic runs record the L1 distance between private and oracle scores;
    oracle runs record 0; per-client vectors move to an .npz when N*d is large."""
    rs = _load("run_sweep")
    from ldpmic import pipeline
    monkeypatch.setattr(rs, "RUNLOG", tmp_path / "RUNLOG.md")
    monkeypatch.setattr(rs, "get_dataset", lambda name: synthetic_dataset)
    monkeypatch.setattr(common, "DATA_DIR", tmp_path / "cache")
    monkeypatch.setattr(rs.ldata, "DATA_DIR", tmp_path / "cache")
    monkeypatch.setattr(pipeline, "INLINE_VECTOR_LIMIT", 10)  # 3 clients x 6 features = 18 > 10
    out = tmp_path / "raw"
    common_args = ["--seeds", "0", "--eps", "1", "--results", str(out), "-o", "fl.rounds=1", "-o", "n_clients=3",
                   "-o", "fl.lr=0.05", "-o", "fl.participation=1.0"]
    for name in ("adult_occ_ldpmic_global", "adult_occ_ldpmic_per_client", "adult_occ_ldpmic_oracle", "adult_occ_iso"):
        assert rs.main([str(ROOT / "configs" / f"{name}.yaml")] + common_args) == 0
    seen = {}
    for jp in out.glob("*.json"):
        prov = json.loads(jp.read_text())
        pv = prov["privacy"]
        label = prov["config"]["name"]
        seen[label] = pv
        if prov["config"]["method"]["method"] == "ldpmic":
            assert len(pv["score_l1_per_client"]) == 3
            assert len(pv["score_raw_global"]) == 6 and len(pv["score_private_global"]) == 6
            if prov["config"]["method"]["allocation"] == "oracle":
                assert pv["score_l1_mean"] == 0.0 and pv["score_l1_global"] == 0.0 and not pv["private"]
            else:
                assert pv["score_l1_mean"] > 0 and pv["score_l1_global"] > 0 and pv["private"]
        vec = jp.with_name(jp.stem + "_vectors.npz")
        if prov["config"]["method"]["method"] in ("ldpmic", "iso"):
            assert vec.exists() and pv["sigmas"] == {}  # moved out of the JSON
            z = np.load(vec)
            assert z["sigmas"].shape == (3, 6) and z["shares"].shape == (3, 6)
            np.testing.assert_allclose(z["shares"].sum(axis=1), 1.0, rtol=1e-6)
    assert len(seen) == 4
    ag = _load("aggregate")
    assert ag.main(["--raw", str(out), "--out", str(tmp_path / "summary")]) == 0
    dec = pd.read_csv(tmp_path / "summary" / "decomposition.csv")
    assert len(dec) == 4 and "score_l1_mean" in dec.columns


def test_clip_grid_tags_runs(tmp_path, monkeypatch, synthetic_dataset):
    rs = _load("run_sweep")
    monkeypatch.setattr(rs, "RUNLOG", tmp_path / "RUNLOG.md")
    monkeypatch.setattr(rs, "get_dataset", lambda name: synthetic_dataset)
    monkeypatch.setattr(common, "DATA_DIR", tmp_path / "cache")
    monkeypatch.setattr(rs.ldata, "DATA_DIR", tmp_path / "cache")
    out = tmp_path / "raw"
    cfg = str(ROOT / "configs" / "adult_occ_nodp.yaml")
    assert rs.main([cfg, "--seeds", "0", "--results", str(out), "-o", "n_clients=3", "-o", "fl.rounds=1",
                    "-o", "fl.lr=0.05", "--clip-grid", "0.5", "2"]) == 0
    names = sorted(p.name for p in out.glob("*.json"))
    assert len(names) == 2 and any("_C0.5_" in n for n in names) and any("_C2_" in n for n in names)
    for jp in out.glob("*.json"):
        prov = json.loads(jp.read_text())
        assert prov["config"]["fl"]["clip_update"] in (0.5, 2.0) and prov["config"]["tag_clip"] is True
