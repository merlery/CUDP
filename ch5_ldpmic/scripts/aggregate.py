#!/usr/bin/env python
"""Aggregate results/raw/*.json (+ .csv) into results/summary/.

    python scripts/aggregate.py [--raw results/raw] [--out results/summary] [--dataset adult]

Writes, per dataset:
    <out>/<dataset>_final.csv   one row per (config name, method, allocation, eps):
                                mean, std (ddof=1), n_seeds of the FINAL
                                accuracy / balanced_accuracy / auc, plus the
                                realized (eps, delta), mean wall-clock per
                                round and bytes per client per round.
    <out>/<dataset>_curves.csv  per-round mean/std of accuracy over seeds.
    <out>/<dataset>_failed.csv  failed runs (run_id, config hash, first traceback line), if any.
and prints the final table.  Only runs with status == "completed" enter the
means; failures are listed, never filled in.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["accuracy", "balanced_accuracy", "auc"]


def label_of(cfg: dict) -> str:
    m = cfg["method"]
    if m["method"] == "ldpmic":
        if m.get("k", 0):   # selection + release (D37-D39)
            crit = "random" if m.get("selection") == "random" else m.get("score", "fixed_grid")
            if crit == "fixed_grid":
                crit = "mi"
            if m.get("allocation") == "oracle":
                crit += "_oracle"
            elif crit != "random" and m.get("score_sensitivity", "conservative") == "conservative":
                crit += "_cons"
            return f"sel[{crit},k={m['k']},{m.get('share_mode', 'exp')}]"
        return f"ldpmic[{m.get('share_mode', 'exp')},{m.get('allocation', 'per_client')}]"
    if m["method"] == "dimsample":
        return f"dimsample[k={m.get('dimsample_k')}]"
    if m["method"] == "fs_dpsgd":
        return f"fs_dpsgd[k={m.get('k')},{m.get('allocation', 'global')}]"
    if m["method"] == "aniso_dpsgd":
        return f"aniso_dpsgd[{m.get('share_mode', 'exp')},beta={m.get('beta', 2.0)},{m.get('allocation', 'global')}]"
    return m["method"]


def variant_of(cfg: dict) -> str:
    """Hash of the resolved config WITHOUT the seed: runs with the same
    variant are repetitions over seeds and may be averaged; anything else
    (a different lr, participation, ...) is a different row."""
    c = copy.deepcopy(cfg)
    c.pop("seed", None)
    c.pop("record_updates", None)      # observer flag (run_sweep --record-updates): same run, same variant
    return hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:10]


def load_runs(raw: Path):
    rows, failed, curves = [], [], []
    for jp in sorted(raw.glob("*.json")):
        if jp.name.startswith("attack_"):
            continue  # attack provenance is aggregated by aggregate_attacks.py
        with open(jp) as f:
            prov = json.load(f)
        if "attack" in prov or "analysis" in prov:
            continue  # attack and analysis records (recon_*.json, attr_budget_*.json): not training runs
        cfg = prov["config"]
        if "_rt_" in cfg["name"]:
            continue  # short runtime-measurement runs: see runtime.csv (aggregate_attacks.py)
        base = {
            "run_id": prov["run_id"], "dataset": cfg["dataset"], "config": cfg["name"], "label": label_of(cfg),
            "method": cfg["method"]["method"], "allocation": cfg["method"].get("allocation", ""),
            "partition": cfg["partition"], "alpha": cfg.get("alpha"), "eps": cfg["method"].get("eps"),
            "seed": cfg["seed"], "lr": cfg["fl"]["lr"], "lr_tuning": bool(cfg.get("tag_lr") or cfg.get("tag_clip")),   # tuning rows (lr grid or clip grid, D17/D31)
            "config_hash": prov["config_hash"], "commit": prov.get("git_commit"), "variant": variant_of(cfg),
        }
        if prov.get("status") != "completed":
            tb = prov.get("traceback", "")
            failed.append({**base, "status": prov.get("status"), "error": tb.strip().splitlines()[-1] if tb else ""})
            continue
        fin = prov["final"]
        priv = prov.get("privacy", {})
        rows.append({**base, **{m: fin.get(m) for m in METRICS},
                     "eps_realized": priv.get("eps_realized"), "delta_realized": priv.get("delta_realized"),
                     "private": priv.get("private"), "duration_s": prov["duration_s"],
                     "score_l1_mean": priv.get("score_l1_mean", float("nan")),
                     "score_l1_global": priv.get("score_l1_global", float("nan")),
                     "rounds": cfg["fl"]["rounds"], "diverged": bool(prov.get("diverged", False))})
        csv = jp.with_suffix(".csv")
        if csv.exists():
            c = pd.read_csv(csv)
            c["run_id"] = prov["run_id"]
            for k in ("dataset", "config", "label", "eps", "seed", "lr_tuning", "variant"):
                c[k] = base[k]
            curves.append(c)
    return (pd.DataFrame(rows), pd.DataFrame(failed),
            pd.concat(curves, ignore_index=True) if curves else pd.DataFrame())


def summarize_final(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "config", "label", "method", "allocation", "partition", "alpha", "eps", "lr", "lr_tuning", "variant"]
    g = df.groupby(keys, dropna=False)
    out = g.agg(n_seeds=("seed", "nunique"), seeds=("seed", lambda s: ",".join(map(str, sorted(s)))),
                **{f"{m}_mean": (m, "mean") for m in METRICS},
                **{f"{m}_std": (m, lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else float("nan")) for m in METRICS},
                eps_realized=("eps_realized", "max"), delta_realized=("delta_realized", "max"),
                private=("private", "all"), duration_s_mean=("duration_s", "mean"),
                rounds=("rounds", "max"), n_diverged=("diverged", "sum"),
                score_l1_mean=("score_l1_mean", "mean"), score_l1_std=("score_l1_mean", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else float("nan")),
                score_l1_global_mean=("score_l1_global", "mean")).reset_index()
    return out.sort_values(["dataset", "lr_tuning", "config", "eps"], na_position="first")


def summarize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    if curves.empty:
        return curves
    keys = ["dataset", "config", "label", "eps", "lr_tuning", "variant", "round"]
    g = curves.groupby(keys, dropna=False)
    agg = {"accuracy_mean": ("accuracy", "mean"), "accuracy_std": ("accuracy", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else float("nan")),
           "train_loss_mean": ("train_loss", "mean"), "wall_clock_s_mean": ("wall_clock_s", "mean"),
           "bytes_per_client_mean": ("bytes_per_client", "mean"), "n_seeds": ("seed", "nunique")}
    for m in ("balanced_accuracy", "auc"):
        if m in curves:
            agg[f"{m}_mean"] = (m, "mean")
    return g.agg(**agg).reset_index()


def format_table(summ: pd.DataFrame) -> str:
    lines = []
    for ds, sub in summ.groupby("dataset"):
        lines.append(f"\n=== {ds} ===")
        lines.append(f"{'config':<34}{'label':<24}{'eps':>6}{'T':>4}{'n':>3}{'div':>4}  {'accuracy':>17}  {'bal_acc':>17}  {'auc':>17}  {'score_L1':>9}  eps_realized")
        for _, r in sub.iterrows():
            cells = []
            for m in METRICS:
                sd = "" if np.isnan(r[f"{m}_std"]) else f" ± {r[f'{m}_std']:.4f}"
                cells.append(f"{r[f'{m}_mean']:.4f}{sd}".rjust(17))
            eps = "" if pd.isna(r["eps"]) else f"{r['eps']:g}"
            alpha = "" if pd.isna(r["alpha"]) else f"{r['alpha']:g}"
            l1 = "" if pd.isna(r["score_l1_mean"]) else f"{r['score_l1_mean']:.3f}"
            tag = " (lr tuning)" if r["lr_tuning"] else ""
            lines.append(f"{r['config'] + tag:<34}{r['label']:<24}{eps:>6}{int(r['rounds']):>4}{int(r['n_seeds']):>3}{int(r['n_diverged']):>4}  " + "  ".join(cells)
                         + f"  {l1:>9}  {r['eps_realized']}")
    return "\n".join(lines)


def decomposition_table(summ: pd.DataFrame) -> pd.DataFrame:
    """Score-privatization decomposition: per (dataset, partition/alpha, eps),
    the ldpmic oracle / per_client / global rows and iso, with accuracy and
    the mean L1 distance between the private and oracle score vectors
    (NaN for iso and for oracle itself, whose distance is 0 by definition)."""
    keep = summ[(~summ.lr_tuning) & (summ.method.isin(["ldpmic", "iso"])) & (summ.eps.isin([1.0, 5.0]))]
    cols = ["dataset", "partition", "alpha", "eps", "config", "label", "n_seeds", "rounds", "n_diverged", "accuracy_mean", "accuracy_std",
            "balanced_accuracy_mean", "auc_mean", "score_l1_mean", "score_l1_std", "score_l1_global_mean", "private"]
    return keep[cols].sort_values(["dataset", "alpha", "eps", "config"], na_position="first")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "summary")
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args(argv)
    df, failed, curves = load_runs(args.raw)
    if args.dataset:
        df = df[df.dataset == args.dataset]
        failed = failed[failed.dataset == args.dataset] if not failed.empty else failed
        curves = curves[curves.dataset == args.dataset] if not curves.empty else curves
    if df.empty and failed.empty:
        print("no runs found under", args.raw)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    summ = summarize_final(df) if not df.empty else pd.DataFrame()
    for ds in sorted(set(df.dataset) | (set(failed.dataset) if not failed.empty else set())):
        if not summ.empty:
            summ[summ.dataset == ds].to_csv(args.out / f"{ds}_final.csv", index=False)
            cv = summarize_curves(curves[curves.dataset == ds]) if not curves.empty else pd.DataFrame()
            if not cv.empty:
                cv.to_csv(args.out / f"{ds}_curves.csv", index=False)
        if not failed.empty and (failed.dataset == ds).any():
            failed[failed.dataset == ds].to_csv(args.out / f"{ds}_failed.csv", index=False)
    if not summ.empty:
        print(format_table(summ))
        dec = decomposition_table(summ)
        if not dec.empty:
            dec.to_csv(args.out / "decomposition.csv", index=False)
            print("\n=== score-privatization decomposition (eps in {1, 5}) ===")
            print(dec.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not failed.empty:
        print(f"\n{len(failed)} FAILED run(s) (listed in *_failed.csv, NOT included in means):")
        for _, r in failed.iterrows():
            print(f"  {r['run_id']}: {r['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
