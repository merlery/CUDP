#!/usr/bin/env python
"""Run a grid of (config x eps x seed) FL experiments.

    python scripts/run_sweep.py configs/adult_occ_*.yaml --seeds 0 1 2 3 4
    python scripts/run_sweep.py configs/adult_occ_nodp.yaml --seeds 0 --lr-grid 0.01 0.05 0.1

Every run is fully specified by one config file + seed (+ eps for private
methods).  For each run it writes

    <results>/<run_id>.csv    per-round metrics
    <results>/<run_id>.json   provenance: full resolved config, seed, config
                              hash, git commit, start/end time, host info
                              (no hostname), durations, privacy info
                              (per-feature sigma_j / p_j, realized (eps, delta)),
                              status and traceback on failure

and appends one line to RUNLOG.md.  Completed runs (JSON with
status == "completed" and the same config hash) are skipped, so the sweep is
resumable.  Failed runs are recorded and retried on the next invocation.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from ldpmic import data as ldata  # noqa: E402
from ldpmic import fl, pipeline  # noqa: E402
from ldpmic.models import build_model, num_params  # noqa: E402

RUNLOG = ROOT / "RUNLOG.md"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def host_info() -> dict:
    # Deliberately no hostname / user name (anonymity rule 4).
    return {
        "os": platform.system(), "os_release": platform.release(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
    }


HASH_NEUTRAL = ("record_updates", "record_norms")   # observer flags that do not change the run


def config_hash(cfg: dict) -> str:
    cfg = {k: v for k, v in cfg.items() if k not in HASH_NEUTRAL}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def set_override(cfg: dict, key: str, value: str) -> None:
    """``a.b.c=value`` with YAML-parsed value."""
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = yaml.safe_load(value)


def load_config(path: Path, overrides) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("name", path.stem)
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        set_override(cfg, k, v)
    return cfg


def resolve(cfg: dict, eps, seed: int) -> dict:
    """The fully specified config of ONE run (what the hash covers)."""
    r = copy.deepcopy(cfg)
    r.pop("eps_grid", None)
    r.pop("seeds", None)
    r.pop("lr_grid", None)
    r.pop("clip_grid", None)
    r["seed"] = int(seed)
    if eps is not None:
        r["method"]["eps"] = float(eps)
    return r


def run_id_of(r: dict) -> str:
    eps = r["method"].get("eps")
    tag = f"_eps{eps:g}" if r["method"]["method"] != "nodp" else ""
    lr_tag = f"_lr{r['fl']['lr']:g}" if r.get("tag_lr") else ""
    c_tag = f"_C{r['fl']['clip_update']:g}" if r.get("tag_clip") else ""
    return f"{r['name']}{tag}{lr_tag}{c_tag}_s{r['seed']}_{config_hash(r)[:8]}"


def append_runlog(line: str) -> None:
    new = not RUNLOG.exists()
    with open(RUNLOG, "a") as f:
        if new:
            f.write("# Run log (appended automatically by scripts/run_sweep.py)\n\n"
                    "| time (UTC) | run_id | status | config hash | commit | duration (s) | command |\n"
                    "|---|---|---|---|---|---|---|\n")
        f.write(line + "\n")


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #

_DATASET_CACHE: dict = {}


def get_dataset(name: str):
    if name not in _DATASET_CACHE:
        _DATASET_CACHE[name] = ldata.load(name)
    return _DATASET_CACHE[name]


def execute(r: dict, out_dir: Path, command: str) -> dict:
    run_id = run_id_of(r)
    csv_path, json_path = out_dir / f"{run_id}.csv", out_dir / f"{run_id}.json"
    prov = {
        "run_id": run_id, "config": r, "config_hash": config_hash(r), "seed": r["seed"],
        "git_commit": git_commit(), "host": host_info(), "command": command,
        "start_time": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "running",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        ds = get_dataset(r["dataset"])
        n_clients = int(r["n_clients"])
        parts = ldata.make_partition(ds, r["partition"], n_clients, r["seed"], r.get("alpha"))
        cdata = ldata.client_data(ds, parts)
        if r.get("smoke_max_records_per_client"):
            k = int(r["smoke_max_records_per_client"])
            cdata = [(X[:k], y[:k]) for X, y in cdata]
        # Optional post-hoc validation split (absent in every reported run): a
        # fixed share of EACH client's training records is held out, the model
        # trains on the rest and is evaluated on the pooled held-out records
        # instead of the test split.  The test split is not read.
        X_eval_raw, y_eval = ds.X_test, ds.y_test
        if r.get("validation"):
            vfrac, vseed = float(r["validation"]["frac"]), int(r["validation"]["seed"])
            vrng = np.random.default_rng(ldata.hash_seed(vseed, "validation", ds.name, r["partition"], n_clients))
            keep, held = [], []
            for X, y in cdata:
                perm = vrng.permutation(len(y))
                n_val = int(round(vfrac * len(y)))
                keep.append((X[perm[n_val:]], y[perm[n_val:]]))
                held.append((X[perm[:n_val]], y[perm[:n_val]]))
            cdata = keep
            X_eval_raw = np.concatenate([X for X, _ in held])
            y_eval = np.concatenate([y for _, y in held])
            prov["validation"] = {"frac": vfrac, "seed": vseed, "n_train": int(sum(len(y) for _, y in keep)),
                                  "n_validation": int(len(y_eval)), "evaluated_on": "validation (test split not used)"}
        flc = r["fl"]
        if flc.get("lr") is None:
            raise ValueError("fl.lr is not set: tune it first (--lr-grid) and freeze it in the config")
        fl_cfg = fl.FLConfig(rounds=int(flc["rounds"]), participation=float(flc["participation"]),
                             local_epochs=int(flc["local_epochs"]), batch_size=int(flc["batch_size"]),
                             lr=float(flc["lr"]), clip_update=flc.get("clip_update"), seed=r["seed"],
                             momentum=float(flc.get("momentum", 0.0)), eval_every=int(flc.get("eval_every", 1)))
        mdict = dict(r["method"])
        # Sensitive features may be named by column prefix (e.g. "sex", "race"
        # select every one-hot column "sex=*", "race=*"); p_max "uniform" is 1/d.
        prefixes = mdict.pop("sensitive_prefixes", None)
        if prefixes:
            idx = [j for j, n in enumerate(ds.feature_names) if any(n == q or n.startswith(q + "=") for q in prefixes)]
            if not idx:
                raise ValueError(f"no feature matches sensitive_prefixes {prefixes}")
            mdict["sensitive"] = tuple(idx)
        if mdict.get("p_max") == "uniform":
            mdict["p_max"] = 1.0 / ds.d
        # Group-wise l2 sensitivity of a record (uniform release only): numeric columns
        # count 1, every one-hot group ("name=value" columns) counts 2.
        if mdict.pop("sensitivity", "full") == "group":
            groups = {n.split("=", 1)[0] for n in ds.feature_names if "=" in n}
            n_num = sum(1 for n in ds.feature_names if "=" not in n)
            mdict["l2_sens_sq"] = float(n_num + 2 * len(groups))
            prov["group_sensitivity"] = {"n_numeric": n_num, "n_categorical": len(groups), "l2_sens_sq": mdict["l2_sens_sq"]}
        mcfg = pipeline.MethodConfig(**{k: (tuple(v) if k == "sensitive" else v) for k, v in mdict.items()})
        schedule = fl.participation_schedule(n_clients, fl_cfg.rounds, fl_cfg.participation, fl_cfg.seed)
        scaler = ldata.public_scaler(ds.ranges)
        t_build = time.perf_counter()
        # The on-disk cache of sanitized data is for reuse across runs of the
        # same (seed, sigmas); it is disabled for large datasets (images) where
        # it would cost ~0.4 GB per run -- the in-memory copy is still made
        # exactly once per run and hash-checked every round.
        cache_dir = (ldata.DATA_DIR / "sanitized" / ds.name) if r.get("cache_sanitized", True) else None
        clients, info = pipeline.build_clients(
            mcfg, cdata, ds.ranges, ds.n_classes, r["seed"], fl_cfg, schedule=schedule,
            cache_dir=cache_dir, transform=scaler)
        vectors = info.vectors()
        if vectors:
            np.savez_compressed(out_dir / f"{run_id}_vectors.npz", **{k: v.astype(np.float32) for k, v in vectors.items()})
        prov["build_time_s"] = time.perf_counter() - t_build
        fl.seed_everything(r["seed"])
        model = build_model(r["model"], info.input_dim or ds.d, ds.n_classes)
        X_test_np = scaler(X_eval_raw)
        if info.selected_features:
            X_test_np = X_test_np[:, np.array(info.selected_features)]
        if info.input_rescale_test:
            X_test_np = X_test_np / np.asarray(info.input_rescale_test)
        X_test = torch.as_tensor(X_test_np, dtype=torch.float32)
        y_test = torch.as_tensor(y_eval, dtype=torch.long)
        rows = []

        def log(rec):
            rows.append(rec.as_row())

        # Optional server-side observer (exposure study, D48): per participant and
        # round, the column-wise l2 norm and mean of the first-layer weight update
        # (one value per input feature).  Saved next to the run; the run itself is
        # unchanged (the flag is not part of the config hash).
        upd_rows = []
        norm_rows = []     # (round, client, l2 norm of the update BEFORE update clipping); --record-norms

        def update_hook(t, cid, res):
            norm_rows.append((t, cid, res.update_norm))
            if not r.get("record_updates"):
                return
            W = res.delta.get("net.0.weight")
            if W is not None:
                W = W.float()
                upd_rows.append((t, cid, W.norm(dim=0).numpy().astype(np.float32), W.mean(dim=0).numpy().astype(np.float32)))

        model, records = fl.fedavg(model, clients, fl_cfg, X_test, y_test, schedule=schedule, log=log,
                                   update_hook=update_hook if (r.get("record_updates") or r.get("record_norms")) else None)
        if r.get("record_norms"):
            np.savez_compressed(out_dir / f"{run_id}_norms.npz", rounds=np.array([u[0] for u in norm_rows]),
                                cids=np.array([u[1] for u in norm_rows]), norm=np.array([u[2] for u in norm_rows]))
        # confusion matrix of the final model (rows = true class, columns = predicted class)
        with torch.no_grad():
            model.eval()
            pred = torch.cat([model(X_test[i:i + 4096]) for i in range(0, X_test.shape[0], 4096)]).argmax(dim=1).numpy()
        cm = np.zeros((ds.n_classes, ds.n_classes), dtype=int)
        np.add.at(cm, (np.asarray(y_eval, dtype=int), pred), 1)
        prov["confusion_matrix"] = cm.tolist()
        xs = [c.X for c in clients if isinstance(getattr(c, "X", None), torch.Tensor)]
        if xs:      # mean absolute value of the inputs as they enter the network (training side)
            prov["mean_abs_input"] = float(sum(float(x.abs().sum()) for x in xs) / sum(x.numel() for x in xs))
        if upd_rows:
            np.savez_compressed(out_dir / f"{run_id}_updates.npz", rounds=np.array([u[0] for u in upd_rows]),
                                cids=np.array([u[1] for u in upd_rows]), colnorm=np.stack([u[2] for u in upd_rows]),
                                colmean=np.stack([u[3] for u in upd_rows]))
            prov["updates_file"] = f"{run_id}_updates.npz"
        import pandas as pd

        # Divergence flag: a non-finite parameter or a non-finite training
        # loss means the final metrics describe a broken model (its argmax
        # is then a constant class), not a trained one.  Recorded, never
        # repaired.
        params_finite = all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
        losses_finite = all(np.isfinite(rec.train_loss) for rec in records if rec.n_participants > 0)
        prov["diverged"] = not (params_finite and losses_finite)

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        if r.get("save_model"):
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_dir / f"{run_id}_model.pt")
        final = records[-1].as_row()
        if mcfg.method in ("local_dpsgd", "fs_dpsgd", "aniso_dpsgd"):
            spent = [c.epsilon_spent() for c in clients]
            two_phase = mcfg.method in ("fs_dpsgd", "aniso_dpsgd")
            # oracle (non-private) scores spend no eps1: report the DP-SGD part only, as for ldpmic oracle
            info.eps_realized = float(max(spent)) + (info.eps1 if two_phase and info.private else 0.0)
            if two_phase and info.private:
                info.delta_realized = info.delta1 + info.delta2
            info.notes.append(f"eps spent per client: min={min(spent):.6g} max={max(spent):.6g}")
        prov.update({
            "status": "completed", "final": final, "n_params": num_params(model),
            "privacy": info.as_dict(), "dataset_meta": ds.meta, "feature_names": ds.feature_names,
            "client_sizes": [int(len(p)) for p in parts],
            "participation_schedule": [p.tolist() for p in schedule],
            "client_summaries": {str(c.cid): c.run_summary() for c in clients},
            "bytes_phase1_per_client": info.phase1_bytes_per_client,
            "sensitive_indices": list(mcfg.sensitive), "p_max": mcfg.p_max,
        })
    except Exception:
        prov["status"] = "failed"
        prov["traceback"] = traceback.format_exc().replace(str(ROOT), "<repo>")   # no local absolute paths in the record
        print(prov["traceback"], file=sys.stderr)
    prov["end_time"] = dt.datetime.now(dt.timezone.utc).isoformat()
    prov["duration_s"] = time.perf_counter() - t0
    with open(json_path, "w") as f:
        json.dump(prov, f, indent=1, default=str)
    append_runlog(f"| {prov['start_time']} | {run_id} | {prov['status']} | {prov['config_hash'][:12]} | "
                  f"{prov['git_commit'][:12]} | {prov['duration_s']:.1f} | `{command}` |")
    return prov


def already_done(r: dict, out_dir: Path) -> bool:
    json_path = out_dir / f"{run_id_of(r)}.json"
    if not json_path.exists():
        return False
    with open(json_path) as f:
        prov = json.load(f)
    return prov.get("status") == "completed" and prov.get("config_hash") == config_hash(r)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="override the config's seeds")
    ap.add_argument("--eps", nargs="*", type=float, default=None, help="override the config's eps_grid")
    ap.add_argument("--lr-grid", nargs="*", type=float, default=None,
                    help="run the config once per lr (tuning); run ids are tagged with the lr")
    ap.add_argument("--clip-grid", nargs="*", type=float, default=None,
                    help="run the config once per update-clipping norm C (tuning); run ids are tagged with C")
    ap.add_argument("--override", "-o", action="append", default=[], help="key.path=value (YAML value)")
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "raw")
    ap.add_argument("--rerun", action="store_true", help="re-run even if completed")
    ap.add_argument("--record-updates", action="store_true",
                    help="save per-participant first-layer update summaries (<run_id>_updates.npz); hash-neutral")
    ap.add_argument("--record-norms", action="store_true",
                    help="save every participant's pre-clipping update norm (<run_id>_norms.npz); hash-neutral")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    command = "python scripts/run_sweep.py " + " ".join(sys.argv[1:] if argv is None else argv)

    n_done = n_skip = n_fail = 0
    t_all = time.perf_counter()
    for cpath in args.configs:
        cfg = load_config(cpath, args.override)
        if args.record_updates:
            cfg["record_updates"] = True
        if args.record_norms:
            cfg["record_norms"] = True
        seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [0])
        is_private = cfg["method"]["method"] != "nodp"
        eps_list = (args.eps if args.eps is not None else cfg.get("eps_grid", [cfg["method"].get("eps")])) if is_private else [None]
        lr_list = args.lr_grid if args.lr_grid else [None]
        clip_list = args.clip_grid if args.clip_grid else [None]
        for lr in lr_list:
          for clip in clip_list:
            for eps in eps_list:
                for seed in seeds:
                    r = resolve(cfg, eps, seed)
                    if lr is not None:
                        r["fl"]["lr"] = float(lr)
                        r["tag_lr"] = True
                    if clip is not None:
                        r["fl"]["clip_update"] = float(clip)
                        r["tag_clip"] = True
                    rid = run_id_of(r)
                    if not args.rerun and already_done(r, args.results):
                        n_skip += 1
                        print(f"skip (done) {rid}")
                        continue
                    if args.dry_run:
                        print(f"would run {rid}")
                        continue
                    print(f"run {rid}", flush=True)
                    prov = execute(r, args.results, command)
                    if prov["status"] == "completed":
                        n_done += 1
                        fin = prov["final"]
                        print(f"  done in {prov['duration_s']:.1f}s  acc={fin.get('accuracy'):.4f} "
                              f"bal_acc={fin.get('balanced_accuracy'):.4f} auc={fin.get('auc'):.4f} "
                              f"eps_realized={prov['privacy']['eps_realized']}", flush=True)
                    else:
                        n_fail += 1
                        print(f"  FAILED after {prov['duration_s']:.1f}s", flush=True)
    print(f"sweep finished: {n_done} completed, {n_skip} skipped, {n_fail} failed, "
          f"{time.perf_counter() - t_all:.1f}s wall-clock")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
