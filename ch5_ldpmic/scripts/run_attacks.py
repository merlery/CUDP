#!/usr/bin/env python
"""Privacy attacks on recorded runs.

    python scripts/run_attacks.py attribute --configs configs/adult_occ_iso.yaml ... --attributes sex race
    python scripts/run_attacks.py membership --configs configs/adult_occ_mi_*.yaml

attribute (E6): for every (config, eps, seed) rebuilds the sanitized client
records exactly as the FL run did (same seeds => same noise; the SHA256 of
each client's sanitized array is checked against the FL run's provenance
when that run exists) and runs ``ldpmic.attacks.attribute_inference``.
No model is trained.

membership: loads the final global model saved by a run with
``save_model: true`` and runs the loss-threshold attack with members =
training records of clients that participated in >= 1 round and
non-members = test-split records.

Each attack writes ``<results>/attack_<kind>_<run_id>.json`` (resumable) and
appends to RUNLOG.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_sweep as rs  # noqa: E402
from ldpmic import data as ldata  # noqa: E402
from ldpmic import fl, pipeline  # noqa: E402
from ldpmic.attacks import attribute_inference as ai  # noqa: E402
from ldpmic.attacks import membership as mi  # noqa: E402
from ldpmic.models import build_model  # noqa: E402


def _prepare(r: dict):
    """Dataset, partition, clients and privacy info of a resolved run config
    (identical construction to run_sweep.execute)."""
    ds = rs.get_dataset(r["dataset"])
    n_clients = int(r["n_clients"])
    parts = ldata.make_partition(ds, r["partition"], n_clients, r["seed"], r.get("alpha"))
    cdata = ldata.client_data(ds, parts)
    flc = r["fl"]
    fl_cfg = fl.FLConfig(rounds=int(flc["rounds"]), participation=float(flc["participation"]),
                         local_epochs=int(flc["local_epochs"]), batch_size=int(flc["batch_size"]),
                         lr=float(flc["lr"] or 0.0), clip_update=flc.get("clip_update"), seed=r["seed"])
    mdict = dict(r["method"])
    prefixes = mdict.pop("sensitive_prefixes", None)
    if prefixes:
        mdict["sensitive"] = tuple(j for j, n in enumerate(ds.feature_names)
                                   if any(n == q or n.startswith(q + "=") for q in prefixes))
    if mdict.get("p_max") == "uniform":
        mdict["p_max"] = 1.0 / ds.d
    mcfg = pipeline.MethodConfig(**{k: (tuple(v) if k == "sensitive" else v) for k, v in mdict.items()})
    schedule = fl.participation_schedule(n_clients, fl_cfg.rounds, fl_cfg.participation, fl_cfg.seed)
    scaler = ldata.public_scaler(ds.ranges)
    cache_dir = (ldata.DATA_DIR / "sanitized" / ds.name) if r.get("cache_sanitized", True) else None
    clients, info = pipeline.build_clients(mcfg, cdata, ds.ranges, ds.n_classes, r["seed"], fl_cfg,
                                           schedule=schedule, cache_dir=cache_dir, transform=scaler)
    return ds, parts, cdata, clients, info, schedule, scaler, mcfg


def attack_attribute(r: dict, out_dir: Path, attributes, command: str) -> dict:
    run_id = rs.run_id_of(r)
    jp = out_dir / f"attack_attribute_{run_id}.json"
    prov = {"attack": "attribute_inference", "run_id": run_id, "config": r, "config_hash": rs.config_hash(r),
            "seed": r["seed"], "git_commit": rs.git_commit(), "host": rs.host_info(), "command": command,
            "start_time": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "running"}
    t0 = time.perf_counter()
    try:
        ds, parts, cdata, clients, info, _, scaler, mcfg = _prepare(r)
        if mcfg.method not in ("iso", "ldpmic", "dimsample"):
            raise ValueError("attribute inference needs a method that releases sanitized records")
        # consistency with the FL run of the same config, if it exists
        fl_json = out_dir / f"{run_id}.json"
        if fl_json.exists():
            fl_prov = json.loads(fl_json.read_text())
            if fl_prov.get("status") == "completed":
                recorded = fl_prov["privacy"]["sanitized_sha256"]
                mism = [c.cid for c in clients if recorded.get(str(c.cid)) != c.sanitized_hash]
                if mism:
                    raise RuntimeError(f"sanitized data differs from the FL run for clients {mism}")
                prov["matches_fl_run"] = True
        raw = [X for X, _ in cdata]
        san = [c.X_san for c in clients]
        sig = [c.sig for c in clients]
        released = info.selected_features or None          # selected release: k columns of every record
        san_transform = clients[0].transform if released else None
        prov["results"] = {}
        for attr in attributes:
            res = ai.attribute_inference(raw, san, ds.feature_names, attr, ds.ranges, sig, r["seed"], transform=scaler,
                                         released=released, san_transform=san_transform)
            prov["results"][attr] = res.as_dict()
        prov["selected_features"] = list(info.selected_features)
        prov["privacy"] = {k: v for k, v in info.as_dict().items()
                           if k in ("eps_realized", "delta_realized", "eps1", "eps2", "private", "sigma0", "score_l1_mean")}
        prov["sensitive_indices"] = list(mcfg.sensitive)
        prov["p_max"] = mcfg.p_max
        prov["status"] = "completed"
    except Exception:
        prov["status"] = "failed"
        prov["traceback"] = traceback.format_exc().replace(str(ROOT), "<repo>")
        print(prov["traceback"], file=sys.stderr)
    prov["end_time"] = dt.datetime.now(dt.timezone.utc).isoformat()
    prov["duration_s"] = time.perf_counter() - t0
    jp.write_text(json.dumps(prov, indent=1, default=str))
    rs.append_runlog(f"| {prov['start_time']} | attack_attribute_{run_id} | {prov['status']} | {prov['config_hash'][:12]} | "
                     f"{prov['git_commit'][:12]} | {prov['duration_s']:.1f} | `{command}` |")
    return prov


def attack_membership(r: dict, out_dir: Path, command: str) -> dict:
    run_id = rs.run_id_of(r)
    jp = out_dir / f"attack_membership_{run_id}.json"
    prov = {"attack": "membership_loss_threshold", "run_id": run_id, "config": r, "config_hash": rs.config_hash(r),
            "seed": r["seed"], "git_commit": rs.git_commit(), "host": rs.host_info(), "command": command,
            "start_time": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "running"}
    t0 = time.perf_counter()
    try:
        model_path = out_dir / f"{run_id}_model.pt"
        fl_json = out_dir / f"{run_id}.json"
        if not model_path.exists() or not fl_json.exists():
            raise FileNotFoundError(f"run {run_id} has no saved model/provenance (needs save_model: true)")
        fl_prov = json.loads(fl_json.read_text())
        if fl_prov.get("status") != "completed":
            raise RuntimeError("FL run not completed")
        ds = rs.get_dataset(r["dataset"])
        parts = ldata.make_partition(ds, r["partition"], int(r["n_clients"]), r["seed"], r.get("alpha"))
        scaler = ldata.public_scaler(ds.ranges)
        model = build_model(r["model"], ds.d, ds.n_classes)
        model.load_state_dict(torch.load(model_path))
        participated = sorted(set(int(c) for part in fl_prov["participation_schedule"] for c in part))
        member_idx = np.concatenate([parts[c] for c in participated])
        res = mi.loss_threshold_attack(model, scaler(ds.X_train[member_idx]), ds.y_train[member_idx],
                                       scaler(ds.X_test), ds.y_test, seed=r["seed"])
        prov["results"] = res.as_dict()
        prov["n_participating_clients"] = len(participated)
        prov["privacy"] = {k: v for k, v in fl_prov["privacy"].items() if k in ("eps_realized", "delta_realized", "private")}
        prov["final_accuracy"] = fl_prov["final"].get("accuracy")
        prov["diverged"] = fl_prov.get("diverged")
        prov["status"] = "completed"
    except Exception:
        prov["status"] = "failed"
        prov["traceback"] = traceback.format_exc().replace(str(ROOT), "<repo>")
        print(prov["traceback"], file=sys.stderr)
    prov["end_time"] = dt.datetime.now(dt.timezone.utc).isoformat()
    prov["duration_s"] = time.perf_counter() - t0
    jp.write_text(json.dumps(prov, indent=1, default=str))
    rs.append_runlog(f"| {prov['start_time']} | attack_membership_{run_id} | {prov['status']} | {prov['config_hash'][:12]} | "
                     f"{prov['git_commit'][:12]} | {prov['duration_s']:.1f} | `{command}` |")
    return prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["attribute", "membership"])
    ap.add_argument("--configs", nargs="+", type=Path, required=True)
    ap.add_argument("--attributes", nargs="*", default=["sex", "race"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--eps", nargs="*", type=float, default=None)
    ap.add_argument("--override", "-o", action="append", default=[])
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "raw")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args(argv)
    command = "python scripts/run_attacks.py " + " ".join(sys.argv[1:] if argv is None else argv)
    args.results.mkdir(parents=True, exist_ok=True)
    n_done = n_skip = n_fail = 0
    for cpath in args.configs:
        cfg = rs.load_config(cpath, args.override)
        seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [0])
        is_private = cfg["method"]["method"] != "nodp"
        eps_list = (args.eps if args.eps is not None else cfg.get("eps_grid", [cfg["method"].get("eps")])) if is_private else [None]
        for eps in eps_list:
            for seed in seeds:
                r = rs.resolve(cfg, eps, seed)
                rid = rs.run_id_of(r)
                jp = args.results / f"attack_{args.kind}_{rid}.json"
                if jp.exists() and not args.rerun and json.loads(jp.read_text()).get("status") == "completed":
                    n_skip += 1
                    continue
                print(f"attack {args.kind} {rid}", flush=True)
                prov = (attack_attribute(r, args.results, args.attributes, command) if args.kind == "attribute"
                        else attack_membership(r, args.results, command))
                if prov["status"] == "completed":
                    n_done += 1
                    print("  ", json.dumps(prov["results"], default=lambda v: round(v, 4) if isinstance(v, float) else v)[:300])
                else:
                    n_fail += 1
    print(f"attacks finished: {n_done} completed, {n_skip} skipped, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
