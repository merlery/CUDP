#!/usr/bin/env python
"""Aggregate attack provenance (results/raw/attack_*.json) and the
runtime/communication measurements into results/summary/.

Outputs
    attribute_inference.csv  per (dataset, partition, config, eps): mean/std over
                             seeds of attacker accuracy / balanced accuracy,
                             majority baseline, raw-record ceiling, and the
                             mean SNR Delta_j/sigma_j of the attribute's columns
    membership.csv           per (dataset, config, eps): mean/std of AUC and advantage
    runtime.csv              per (dataset, config): mean per-round client wall-clock,
                             bytes sent per client per round, one-time Phase 1/3
                             cost and Phase 1 bytes (from the *_rt_* runs)
Only completed attacks/runs enter the means; failures are listed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from aggregate import label_of, variant_of  # noqa: E402


def _std(s):
    return float(np.std(s, ddof=1)) if len(s) > 1 else float("nan")


def attribute_rows(raw: Path):
    rows, failed = [], []
    for jp in sorted(raw.glob("attack_attribute_*.json")):
        p = json.loads(jp.read_text())
        c = p["config"]
        base = {"dataset": c["dataset"], "partition": c["partition"], "alpha": c.get("alpha"), "config": c["name"],
                "label": label_of(c) + ("+cap" if c["method"].get("sensitive_prefixes") else ""),
                "eps": c["method"].get("eps"), "seed": c["seed"], "variant": variant_of(c)}
        if p["status"] != "completed":
            failed.append({**base, "error": p.get("traceback", "").strip().splitlines()[-1:]})
            continue
        for attr, r in p["results"].items():
            rows.append({**base, "attribute": attr, "accuracy": r["accuracy"], "balanced_accuracy": r["balanced_accuracy"],
                         "majority_accuracy": r["majority_accuracy"], "raw_accuracy": r["raw_accuracy"],
                         "raw_balanced_accuracy": r["raw_balanced_accuracy"], "n_eval": r["n_eval"],
                         "snr_mean": (float(np.mean([v for v in r["snr"].values() if v is not None]))
                                      if any(v is not None for v in r["snr"].values()) else float("nan")),
                         "snr": json.dumps(r["snr"]), "attribute_released": r.get("attribute_released", True),
                         "direct_balanced_accuracy": r.get("direct_balanced_accuracy"), "all_balanced_accuracy": r.get("all_balanced_accuracy"),
                         "direct_accuracy": r.get("direct_accuracy"), "all_accuracy": r.get("all_accuracy")})
    return pd.DataFrame(rows), failed


def membership_rows(raw: Path):
    rows, failed = [], []
    for jp in sorted(raw.glob("attack_membership_*.json")):
        p = json.loads(jp.read_text())
        c = p["config"]
        base = {"dataset": c["dataset"], "partition": c["partition"], "config": c["name"], "label": label_of(c),
                "eps": c["method"].get("eps"), "seed": c["seed"], "variant": variant_of(c)}
        if p["status"] != "completed":
            failed.append({**base, "error": p.get("traceback", "").strip().splitlines()[-1:]})
            continue
        r = p["results"]
        rows.append({**base, "auc": r["auc"], "advantage": r["advantage"], "member_loss_mean": r["member_loss_mean"],
                     "nonmember_loss_mean": r["nonmember_loss_mean"], "n_members": r["n_members"],
                     "final_accuracy": p.get("final_accuracy"), "diverged": p.get("diverged")})
    return pd.DataFrame(rows), failed


def runtime_rows(raw: Path):
    rows = []
    for jp in sorted(raw.glob("*_rt_*.json")):
        if jp.name.startswith("attack_"):
            continue
        p = json.loads(jp.read_text())
        if p["status"] != "completed":
            continue
        c = p["config"]
        df = pd.read_csv(jp.with_suffix(".csv"))
        pv = p["privacy"]
        rows.append({"dataset": c["dataset"], "config": c["name"], "label": label_of(c), "eps": c["method"].get("eps"),
                     "seed": c["seed"], "rounds": len(df), "n_params": p.get("n_params"),
                     "client_time_mean_s": float(df["client_time_mean_s"].mean()),
                     "client_time_max_s": float(df["client_time_max_s"].mean()),
                     "round_wall_clock_s": float(df["wall_clock_s"].mean()),
                     "eval_time_s": float(df["eval_time_s"].mean()),
                     "bytes_per_client_per_round": float(df["bytes_per_client"].mean()),
                     "phase1_time_s": pv.get("phase1_time_s", 0.0), "phase3_time_s": pv.get("phase3_time_s", 0.0),
                     "phase1_bytes_per_client": pv.get("phase1_bytes_per_client", 0),
                     "build_time_s": p.get("build_time_s"), "threads": p["host"].get("torch_threads")})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "summary")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    fmt = lambda v: f"{v:.4f}"  # noqa: E731

    att, att_failed = attribute_rows(args.raw)
    if not att.empty:
        keys = ["dataset", "partition", "alpha", "config", "label", "eps", "variant", "attribute"]
        g = att.groupby(keys, dropna=False).agg(
            n_seeds=("seed", "nunique"), accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", _std),
            balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_std=("balanced_accuracy", _std),
            majority_accuracy_mean=("majority_accuracy", "mean"), raw_accuracy_mean=("raw_accuracy", "mean"),
            raw_balanced_accuracy_mean=("raw_balanced_accuracy", "mean"), snr_mean=("snr_mean", "mean"),
            direct_balanced_accuracy_mean=("direct_balanced_accuracy", "mean"), direct_balanced_accuracy_std=("direct_balanced_accuracy", "std"),
            all_balanced_accuracy_mean=("all_balanced_accuracy", "mean"), all_balanced_accuracy_std=("all_balanced_accuracy", "std"),
            n_eval_mean=("n_eval", "mean")).reset_index().sort_values(["dataset", "partition", "attribute", "label", "eps"])
        g.to_csv(args.out / "attribute_inference.csv", index=False)
        print("\n=== attribute inference (attacker on sanitized records; majority = majority-class accuracy; raw = same attacker on raw records) ===")
        cols = ["dataset", "partition", "attribute", "label", "eps", "n_seeds", "accuracy_mean", "accuracy_std",
                "balanced_accuracy_mean", "balanced_accuracy_std", "majority_accuracy_mean", "raw_accuracy_mean", "raw_balanced_accuracy_mean", "snr_mean",
                "direct_balanced_accuracy_mean", "direct_balanced_accuracy_std", "all_balanced_accuracy_mean", "all_balanced_accuracy_std"]
        print(g[cols].to_string(index=False, float_format=fmt))
    if att_failed:
        print(f"\n{len(att_failed)} FAILED attribute attacks:", att_failed)

    mem, mem_failed = membership_rows(args.raw)
    if not mem.empty:
        keys = ["dataset", "partition", "config", "label", "eps", "variant"]
        g = mem.groupby(keys, dropna=False).agg(
            n_seeds=("seed", "nunique"), auc_mean=("auc", "mean"), auc_std=("auc", _std),
            advantage_mean=("advantage", "mean"), advantage_std=("advantage", _std),
            member_loss_mean=("member_loss_mean", "mean"), nonmember_loss_mean=("nonmember_loss_mean", "mean"),
            final_accuracy_mean=("final_accuracy", "mean"), n_diverged=("diverged", "sum")).reset_index().sort_values(["dataset", "label", "eps"])
        g.to_csv(args.out / "membership.csv", index=False)
        print("\n=== membership inference (loss threshold; advantage = max TPR-FPR) ===")
        print(g.drop(columns=["variant", "config"]).to_string(index=False, float_format=fmt))
    if mem_failed:
        print(f"\n{len(mem_failed)} FAILED membership attacks:", mem_failed)

    rt = runtime_rows(args.raw)
    if not rt.empty:
        keys = ["dataset", "config", "label", "eps", "n_params", "threads"]
        g = rt.groupby(keys, dropna=False).agg(
            n_seeds=("seed", "nunique"), client_time_mean_s=("client_time_mean_s", "mean"), client_time_std_s=("client_time_mean_s", _std),
            client_time_max_s=("client_time_max_s", "mean"), round_wall_clock_s=("round_wall_clock_s", "mean"),
            eval_time_s=("eval_time_s", "mean"), bytes_per_client_per_round=("bytes_per_client_per_round", "mean"),
            phase1_time_s=("phase1_time_s", "mean"), phase3_time_s=("phase3_time_s", "mean"),
            phase1_bytes_per_client=("phase1_bytes_per_client", "mean"), build_time_s=("build_time_s", "mean")).reset_index()
        g.to_csv(args.out / "runtime.csv", index=False)
        print("\n=== runtime / communication (per round: mean client local-update wall-clock and bytes sent; one-time: Phase 1 / Phase 3 seconds for the whole federation, Phase 1 bytes per client) ===")
        print(g.drop(columns=["config"]).to_string(index=False, float_format=fmt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
