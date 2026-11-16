#!/usr/bin/env python
"""Central-on-Privatized baseline (DECISIONS.md D53, results/ch5/central/config.yaml).

Every client privatizes its records exactly as in the recorded federated run
of the same setting and seed (Phases 1 and 3 through the same code path, with
the SHA-256 of every client's sanitized array asserted equal to the one in
the federated run's provenance), uploads them once, and the server trains the
same MLP centrally with plain SGD for the number of epochs that matches the
federated run's total number of local gradient steps.  Central-on-Clean does
the same on the clean records.  The test set is built only after training.

    python scripts/run_central.py            # all settings of config.yaml, resumable
    python scripts/run_central.py --report   # tables, figure, summary from the recorded runs

Per-run provenance: results/ch5/central/runs/<run_id>.json.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_sweep as rs  # noqa: E402
import run_attacks as ra  # noqa: E402
from ldpmic import data as ldata  # noqa: E402
from ldpmic import fl  # noqa: E402
from ldpmic.models import build_model  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

OUT = ROOT / "results" / "ch5" / "central"
RUNS = OUT / "runs"
BATCH = 128
TOL = 0.05
LR = {"adult": 0.1, "bank": 0.05}           # non-private federated model's lr (D17)
NODP_CONFIG = {"adult": "configs/adult_occ_nodp.yaml"}

SETTINGS = [  # (release label, config, eps list)
    ("all_features", "configs/adult_occ_iso.yaml", [1.0, 10.0]),
    ("all_features", "configs/adult_occ_v7_iso.yaml", [50.0, 100.0, 200.0]),
    ("private_sel_k20", "configs/adult_occ_v7_sel_mi_k20.yaml", [50.0, 100.0, 200.0]),
    ("oracle_sel_k20", "configs/adult_occ_v7_sel_mi_oracle_k20.yaml", [50.0, 100.0, 200.0]),
    ("ext_bank_private_sel_k10", "configs/bank_occ_v9_sel_k10.yaml", [200.0]),
]
SEEDS = [0, 1, 2, 3, 4]


def federated_steps(schedule, sizes, local_epochs, batch_size) -> int:
    return int(sum(local_epochs * math.ceil(sizes[i] / batch_size) for part in schedule for i in part))


def train_central(X, y, model_name, n_classes, seed, epochs, lr):
    fl.seed_everything(seed)
    model = build_model(model_name, X.shape[1], n_classes)
    gen = fl.torch_gen(seed, "central")
    t0 = time.perf_counter()
    loss = fl.local_sgd(model, X, y, epochs, BATCH, lr, gen, momentum=0.0)
    return model, time.perf_counter() - t0, loss


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        out = model(X_test)
    finite = bool(torch.isfinite(out).all())
    pred = out.argmax(1).numpy()
    yt = y_test.numpy()
    return float((pred == yt).mean()), float(balanced_accuracy_score(yt, pred)), finite


def fed_comm(fl_json: Path):
    rows = list(csv.DictReader(open(fl_json.with_suffix(".csv"))))
    return float(sum(float(r["bytes_per_client"]) * float(r["n_participants"]) for r in rows))


def run_one(release, config, eps, seed, clean=False):
    cfg = rs.load_config(ROOT / config, [])
    r = rs.resolve(cfg, eps, seed)
    fl_id = rs.run_id_of(r)
    kind = "central_clean" if clean else "central_priv"
    run_id = f"{kind}_{fl_id}"
    jp = RUNS / f"{run_id}.json"
    if jp.exists() and json.loads(jp.read_text()).get("status") == "completed":
        return json.loads(jp.read_text())
    prov = {"run_id": run_id, "kind": kind, "release": release, "federated_run_id": fl_id, "config_file": config,
            "eps": eps, "seed": seed, "git_commit": rs.git_commit(), "host": rs.host_info(),
            "start_time": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "running"}
    try:
        fl_json = ROOT / "results" / "raw" / f"{fl_id}.json"
        fl_prov = json.loads(fl_json.read_text())
        if fl_prov.get("status") != "completed":
            raise RuntimeError(f"federated reference {fl_id} not completed")
        ds = rs.get_dataset(r["dataset"])
        r_build = dict(r); r_build["cache_sanitized"] = False          # no disk cache (D51); the noise does not depend on it
        ds_, parts, cdata, clients, info, schedule, scaler, mcfg = ra._prepare(r_build)
        sizes = [int(len(p)) for p in parts]
        # sanity 1: byte-identical privatized arrays
        if not clean:
            rec = fl_prov["privacy"]["sanitized_sha256"]
            mism = [c.cid for c in clients if rec.get(str(c.cid)) != c.sanitized_hash]
            if mism:
                raise RuntimeError(f"SANITY 1 FAILED: sanitized SHA-256 differs from {fl_id} for clients {mism}")
            prov["sha256_match"] = f"{len(clients)}/{len(clients)} clients"
            X = torch.cat([c.X for c in clients]); y = torch.cat([c.y for c in clients])
            sel = info.selected_features
            n_feat = X.shape[1]
            central_bytes = float(X.shape[0] * n_feat * 4 + fl_prov["privacy"].get("phase1_bytes_per_client", 0) * len(clients))
        else:
            X = torch.as_tensor(scaler(ds.X_train), dtype=torch.float32); y = torch.as_tensor(ds.y_train, dtype=torch.long)
            sel = []; n_feat = X.shape[1]; central_bytes = float(X.shape[0] * n_feat * 4)
        if X.shape[0] != sum(sizes):
            raise RuntimeError("central training set size differs from the union of the client datasets")
        # sanity 3: matched gradient steps
        fed = federated_steps(schedule, sizes, int(r["fl"]["local_epochs"]), int(r["fl"]["batch_size"]))
        spe = math.ceil(X.shape[0] / BATCH)
        epochs = max(1, round(fed / spe)); cen = epochs * spe
        prov.update({"federated_steps": fed, "central_steps": cen, "central_epochs": epochs, "steps_per_epoch": spe,
                     "step_rel_diff": (cen - fed) / fed})
        print(f"  steps: federated {fed}, central {cen} ({epochs} epochs x {spe}), rel diff {(cen - fed) / fed:+.4f}", flush=True)
        if abs(cen - fed) / fed > TOL:
            raise RuntimeError(f"SANITY 3 FAILED: central steps {cen} vs federated {fed}")
        lr = LR[r["dataset"]]
        model, train_s, loss = train_central(X, y, r["model"], ds.n_classes, seed, epochs, lr)
        # test set built only now (sanity 2: no test record is read during training)
        Xt = scaler(ds.X_test)
        if sel:
            Xt = Xt[:, np.array(sel)]
        acc, bacc, finite = evaluate(model, torch.as_tensor(Xt, dtype=torch.float32), torch.as_tensor(ds.y_test, dtype=torch.long))
        prov.update({"status": "completed", "accuracy": acc, "balanced_accuracy": bacc, "finite_outputs": finite,
                     "train_loss": loss, "train_seconds": train_s, "lr": lr, "batch_size": BATCH, "n_records": int(X.shape[0]),
                     "n_released_features": int(n_feat), "selected_features": list(sel),
                     "central_upload_bytes": central_bytes, "federated_bytes": fed_comm(fl_json),
                     "federated_accuracy": fl_prov["final"]["accuracy"], "federated_balanced_accuracy": fl_prov["final"]["balanced_accuracy"],
                     "federated_duration_s": fl_prov["duration_s"], "test_set_built_after_training": True})
    except Exception:
        prov["status"] = "failed"; prov["traceback"] = traceback.format_exc().replace(str(ROOT), "<repo>")
        print(prov["traceback"], file=sys.stderr)
    prov["end_time"] = dt.datetime.now(dt.timezone.utc).isoformat()
    RUNS.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(prov, indent=1, default=str))
    return prov


def sanity_reproduce():
    """Sanity 4: re-run one federated setting for one seed and compare with the stored accuracy."""
    import tempfile
    cfgp, eps, seed = "configs/adult_occ_v7_sel_mi_k20.yaml", 200.0, 0
    cfg = rs.load_config(ROOT / cfgp, []); r = rs.resolve(cfg, eps, seed)
    stored = json.loads((ROOT / "results" / "raw" / f"{rs.run_id_of(r)}.json").read_text())["final"]["accuracy"]
    with tempfile.TemporaryDirectory() as td:
        old = rs.RUNLOG; rs.RUNLOG = Path(td) / "RUNLOG.md"
        try:
            rs.main([str(ROOT / cfgp), "--seeds", str(seed), "--eps", str(int(eps)), "--results", td])
        finally:
            rs.RUNLOG = old
        new = json.loads(next(Path(td).glob("*_s0_*.json")).read_text())["final"]["accuracy"]
    ok = abs(new - stored) < 1e-12
    print(f"SANITY 4: re-run of {rs.run_id_of(r)}: stored {stored:.6f}, re-run {new:.6f} -> {'OK' if ok else 'FAILED'}", flush=True)
    return {"run": rs.run_id_of(r), "stored": stored, "rerun": new, "ok": ok}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    if args.report:
        import central_report
        return central_report.main()
    t0 = time.perf_counter()
    s4 = sanity_reproduce()
    if not s4["ok"]:
        raise SystemExit("SANITY 4 FAILED: the federated run does not reproduce; stopping")
    (OUT / "sanity_reproduce.json").write_text(json.dumps(s4, indent=1))
    failed = 0
    for release, config, eps_list in SETTINGS:
        for eps in eps_list:
            for seed in SEEDS:
                print(f"central {release} {config} eps={eps:g} seed={seed}", flush=True)
                p = run_one(release, config, eps, seed)
                if p["status"] != "completed":
                    failed += 1
                    raise SystemExit("a sanity check or run failed; stopping (see the provenance file)")
                print(f"  done: acc {p['accuracy']:.4f} bal {p['balanced_accuracy']:.4f} (federated {p['federated_accuracy']:.4f}) "
                      f"train {p['train_seconds']:.1f}s", flush=True)
    for seed in SEEDS:
        print(f"central clean adult seed={seed}", flush=True)
        p = run_one("clean", NODP_CONFIG["adult"], None, seed, clean=True)
        if p["status"] != "completed":
            raise SystemExit("central-on-clean failed; stopping")
        print(f"  done: acc {p['accuracy']:.4f} (federated nodp {p['federated_accuracy']:.4f})", flush=True)
    print(f"all central runs finished in {time.perf_counter() - t0:.0f} s, {failed} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
