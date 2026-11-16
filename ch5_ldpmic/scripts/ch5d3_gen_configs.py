#!/usr/bin/env python
"""Generate the configs of the Chapter 5 follow-up runs (results/ch5/draft3/config.yaml, DECISIONS.md D65-D70).

Every config is a copy of a stored config with the stated change only; the
stored configs and results are not touched.

    python scripts/ch5d3_gen_configs.py            # writes configs/ch5d3_*.yaml and results/ch5/draft3/jobs.txt
    python scripts/ch5d3_gen_configs.py --lr10     # additionally the Part B repeat with lr / 10 (only if triggered)
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = "results/ch5/draft3/runs"
S_PUBLIC = 0.25
EXP1 = [(ds, m) for ds in ("adult_occ", "bank_occ", "mnist_a0.5", "fmnist_a0.5") for m in ("iso", "ldpmic_global", "ldpmic_oracle")]
C_EPS = [0.5, 2, 5, 10]          # eps = 1: the rule gives the stored C, the stored *_v8_local_dpsgd runs are reused


def load(name):
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def dump(cfg, comment):
    p = ROOT / "configs" / f"{cfg['name']}.yaml"
    with open(p, "w") as f:
        f.write(f"# {cfg['name']} — {comment}\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    return p


def clip_rule(ds, eps):
    """D32 rule applied per budget: median over rounds of the per-round mean update norm of the stored
    local_dpsgd run WITHOUT update clipping at that eps, seed 0; rounded to 3 decimals (as 18.357 in D32)."""
    f = glob.glob(str(ROOT / "results" / "raw" / f"{ds}_a0.5_local_dpsgd_eps{eps:g}_s0_*.csv"))
    assert len(f) == 1, f
    return round(float(pd.read_csv(f[0]).mean_update_norm.median()), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr10", action="store_true")
    args = ap.parse_args()
    jobs = []

    def add(cfg, part):
        for e in cfg["eps_grid"]:
            for s in cfg["seeds"]:
                jobs.append((part, f"scripts/run_sweep.py configs/{cfg['name']}.yaml --eps {e:g} --seeds {s} --results {OUT} --record-norms"))

    if args.lr10:
        for ds, m in EXP1:
            c = load(f"{ds}_{m}")
            c["name"] = f"ch5d3_Blr_{ds}_{m}"
            c["method"]["input_rescale"] = S_PUBLIC
            c["fl"]["lr"] = c["fl"]["lr"] / 10.0
            c.pop("lr_grid", None)
            dump(c, f"Part B repeat: {ds}_{m} + input_rescale {S_PUBLIC}, learning rate divided by 10")
            add(c, "Blr")
        Path(ROOT / "results/ch5/draft3/jobs_lr10.txt").write_text("".join(f"{j}\n" for _, j in jobs))
        return 0

    for k in (10, 20):                                   # Part A
        c = load(f"adult_occ_v7_sel_mi_k{k}")
        c["name"] = f"ch5d3_A_adult_occ_sel_local_k{k}"
        c["method"]["score_release"] = "local"
        c["eps_grid"] = [100, 200]
        dump(c, f"Part A: adult_occ_v7_sel_mi_k{k} with score_release local (every client adds the full noise)")
        add(c, "A")
    for ds, m in EXP1:                                   # Part B
        c = load(f"{ds}_{m}")
        c["name"] = f"ch5d3_B_{ds}_{m}"
        c["method"]["input_rescale"] = S_PUBLIC
        c.pop("lr_grid", None)
        dump(c, f"Part B: {ds}_{m} + input_rescale {S_PUBLIC} (released coordinate / sqrt(1 + sigma_j^2 / s_j^2))")
        add(c, "B")
    for ds in ("mnist", "fmnist"):                       # Part C
        for e in C_EPS:
            c = load(f"{ds}_a0.5_v8_local_dpsgd")
            c["name"] = f"ch5d3_C_{ds}_a0.5_local_dpsgd_e{e:g}"
            c["fl"]["clip_update"] = clip_rule(ds, e)
            c["eps_grid"] = [e]
            c.pop("clip_grid", None)
            dump(c, f"Part C: {ds}_a0.5_v8_local_dpsgd with the update clipping bound chosen at eps = {e:g} by the D32 rule")
            add(c, "C")
    c = load("adult_occ_v7_iso")                         # Part F
    c["name"] = "ch5d3_F_adult_occ_iso_group"
    c["method"]["sensitivity"] = "group"
    c["eps_grid"] = [10, 50, 100, 200]
    dump(c, "Part F: adult_occ_v7_iso with group-wise l2 sensitivity sqrt(n_numeric + 2 n_categorical)")
    add(c, "F")
    c = load("adult_occ_v7_iso")
    c["name"] = "ch5d3_F_adult_occ_iso_ref10"
    c["eps_grid"] = [10]
    dump(c, "Part F reference: adult_occ_v7_iso at eps = 10 (the stored v7 grid starts at 20); same protocol")
    add(c, "F")
    order = {"A": 0, "B": 1, "C": 2, "F": 3}
    jobs.sort(key=lambda j: order[j[0]])
    Path(ROOT / "results/ch5/draft3/jobs.txt").write_text("".join(f"{j}\n" for _, j in jobs))
    print(len(jobs), "jobs", {p: sum(1 for q, _ in jobs if q == p) for p in order})
    return 0


if __name__ == "__main__":
    sys.exit(main())
