#!/usr/bin/env python
"""Diagnostics and report of the Chapter 5 follow-up runs (DECISIONS.md D65-D70).

    python scripts/ch5d3_report.py diag      # diagnostics that need no new run (Parts A noise table, B.1, C.1 stored, D, E)
    python scripts/ch5d3_report.py report    # CSVs per part, config_used.yaml, summary.md (fails if a required run is missing)

Every number in summary.md is read from a recorded run (results/raw or
results/ch5/draft3/runs) or computed here from the calibration code; nothing
is typed by hand.  Stored results are only read.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_sweep as rs  # noqa: E402
from ldpmic import data as ldata  # noqa: E402
from ldpmic import fl, pipeline, scores  # noqa: E402
from ldpmic.privacy import sigma0, split_budget  # noqa: E402

RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "ch5" / "draft3"
NEW = OUT / "runs"
DIAG = OUT / "diag"
DELTA = 1e-5
TAB_EPS = [0.5, 1, 2, 5, 10]
EXP1_DS = ["adult_occ", "bank_occ", "mnist_a0.5", "fmnist_a0.5"]
EXP1_M = ["iso", "ldpmic_global", "ldpmic_oracle"]
M_LABEL = {"iso": "uniform release", "ldpmic_global": "LDP-MIC", "ldpmic_oracle": "oracle allocation"}


class Missing(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# run lookup
# --------------------------------------------------------------------------- #

def cfg_of(name):
    return rs.load_config(ROOT / "configs" / f"{name}.yaml", None)


def prov_of(name, eps, seed, where=RAW, required=True):
    """Provenance of the completed run of config `name` at (eps, seed).  Prefers the run whose hash matches the
    current config file; otherwise the unique completed run of that name (stored runs that predate a config edit)."""
    cfg = cfg_of(name)
    rid = rs.run_id_of(rs.resolve(cfg, eps, seed))
    p = where / f"{rid}.json"
    cands = [p] if p.exists() else [Path(f) for f in sorted(glob.glob(str(where / f"{name}_eps{eps:g}_s{seed}_????????.json")))]
    done = []
    for c in cands:
        with open(c) as f:
            j = json.load(f)
        if j.get("status") == "completed":
            done.append(j)
    if len(done) != 1:
        if required:
            raise Missing(f"{name} eps={eps} seed={seed}: {len(done)} completed runs in {where.name}")
        return None
    return done[0]


def rounds_of(prov, where=RAW):
    return pd.read_csv(where / f"{prov['run_id']}.csv")


def ms(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def fmt(m, s=None, nd=3):
    return f"{m:.{nd}f}" if s is None or (isinstance(s, float) and math.isnan(s)) else f"{m:.{nd}f} ± {s:.{nd}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def delta_fg(m):
    """The task's formula Delta_FG(m) = 3 (1 + ln m) / (m ln 2)."""
    return 3.0 * (1.0 + math.log(m)) / (m * math.log(2.0))


_DS = {}


def dataset(name):
    if name not in _DS:
        _DS[name] = ldata.load(name)
    return _DS[name]


def client_sizes(cfg, seed):
    ds = dataset(cfg["dataset"])
    parts = ldata.make_partition(ds, cfg["partition"], int(cfg["n_clients"]), seed, cfg.get("alpha"))
    return ds, parts


# --------------------------------------------------------------------------- #
# score-noise quantities (Parts A and D)
# --------------------------------------------------------------------------- #

def score_noise(sizes, d, n_classes, G, eps1, delta1):
    """Std of the noise on the n-weighted average score vector in both release modes, for the code's analytic
    sensitivity and for the task's Delta_FG."""
    sizes = np.asarray(sizes, dtype=float)
    M = sizes.sum()
    c = sigma0(eps1, delta1, "analytic")
    out = {"d": d, "n_clients": len(sizes), "n_min": int(sizes.min()), "n_max": int(sizes.max()), "n_sum": int(M),
           "eps1": eps1, "delta1": delta1, "c": c, "normalizer_ln_min_G_C": math.log(min(G, n_classes))}
    for tag, fn in (("code", lambda m: scores.score_sensitivity(G, int(m), n_classes, "analytic")), ("task", delta_fg)):
        ds_ = np.array([min(1.0, fn(m)) if tag == "task" else fn(m) for m in sizes])
        w = sizes / M
        out[f"secagg_{tag}"] = math.sqrt(d) * float((sizes * ds_).max()) / M * c
        out[f"local_{tag}"] = float(np.sqrt(np.sum((w * math.sqrt(d) * ds_ * c) ** 2)))
        out[f"local_perclient_max_{tag}"] = float(math.sqrt(d) * ds_.max() * c)
        out[f"delta_s_at_nmax_{tag}"] = float(ds_[np.argmax(sizes)])
        out[f"delta_s_max_{tag}"] = float(ds_.max())
    return out


# --------------------------------------------------------------------------- #
# Part A
# --------------------------------------------------------------------------- #

def part_a():
    rows, noise_rows = [], []
    for k in (10, 20):
        names = {"private, local": (f"ch5d3_A_adult_occ_sel_local_k{k}", NEW), "private, secure aggregation": (f"adult_occ_v7_sel_mi_k{k}", RAW),
                 "random k-subset": (f"adult_occ_v7_sel_random_k{k}", RAW), "oracle": (f"adult_occ_v7_sel_mi_oracle_k{k}", RAW)}
        for eps in (100, 200):
            for seed in range(5):
                rnd = prov_of(names["random k-subset"][0], eps, seed)["final"]["accuracy"]
                for label, (name, where) in names.items():
                    p = prov_of(name, eps, seed, where)
                    pr = p["privacy"]
                    raw = np.array(pr.get("score_raw_global") or [])
                    sel = set(pr["selected_features"])
                    row = {"part": "A", "method": label, "config": name, "run_id": p["run_id"], "eps": eps, "k": k, "seed": seed,
                           "accuracy": p["final"]["accuracy"], "balanced_accuracy": p["final"]["balanced_accuracy"],
                           "auc": p["final"].get("auc"), "diff_to_random": p["final"]["accuracy"] - rnd,
                           "git_commit": p["git_commit"], "stored": where == RAW}
                    if len(raw):
                        exact = set(pipeline.select_top_k(raw, k))
                        srt = np.sort(raw)[::-1]
                        row.update(overlap=len(sel & exact), exact_max=float(srt[0]), exact_kth=float(srt[k - 1]),
                                   exact_k_plus_1=float(srt[k]))
                    if label.startswith("private"):
                        cfg = cfg_of(name)
                        sizes = p["client_sizes"]
                        d = len(raw)
                        ds = dataset("adult")
                        sn = score_noise(sizes, d, ds.n_classes, int(cfg["method"]["G"]), pr["eps1"], pr["delta1"])
                        mode = "local" if "local" in label else "secagg"
                        row.update(noise_std_avg=sn[f"{mode}_code"], noise_std_avg_task_formula=sn[f"{mode}_task"],
                                   recorded_sigma_s=pr["sigma_s"], eps1=pr["eps1"])
                        if mode == "local":      # what the average would select without the per-client clipping (diagnostic)
                            row["overlap_unclipped_average"] = _local_unclipped_overlap(cfg, seed, pr, k, exact)
                    rows.append(row)
    df = pd.DataFrame(rows)
    # MNIST noise table (no training)
    cfg = cfg_of("mnist_a0.5_v8_fs_mi_k400")
    ds, parts = client_sizes(cfg, 0)
    sizes = [len(p) for p in parts]
    ref = prov_of("mnist_a0.5_v8_fs_mi_k400", 1, 0)
    raw = np.sort(np.array(ref["privacy"]["score_raw_global"]))[::-1]
    for eps in (1, 5):
        (e1, d1), _ = split_budget(float(eps), DELTA, 0.2)
        sn = score_noise(sizes, ds.d, ds.n_classes, 10, e1, d1)
        noise_rows.append({"dataset": "mnist (alpha 0.5, seed 0)", "eps": eps, "eps1": e1, "d": ds.d, "n_clients": len(sizes),
                           "n_min": sn["n_min"], "n_max": sn["n_max"],
                           "noise_std_avg_local": sn["local_code"], "noise_std_avg_secagg": sn["secagg_code"],
                           "noise_std_per_client_local": sn["local_perclient_max_code"],
                           "noise_std_avg_local_task_formula": sn["local_task"], "noise_std_avg_secagg_task_formula": sn["secagg_task"],
                           "exact_max": float(raw[0]), "exact_100th": float(raw[99]), "exact_400th": float(raw[399])})
    if ref["privacy"]["eps1"] == 0.2:
        assert abs(ref["privacy"]["sigma_s"] - noise_rows[0]["noise_std_avg_secagg"]) < 1e-9, "recorded MNIST sigma_s not reproduced"
    return df, pd.DataFrame(noise_rows)


def _local_unclipped_overlap(cfg, seed, pr, k, exact):
    ds, parts = client_sizes(cfg, seed)
    cdata = ldata.client_data(ds, parts)
    G = int(cfg["method"]["G"])
    c = sigma0(pr["eps1"], pr["delta1"], "analytic")
    d = ds.d
    acc = np.zeros(d)
    M = float(sum(X.shape[0] for X, _ in cdata))
    chk = np.zeros(d)
    for i, (X, y) in enumerate(cdata):
        raw = scores.fixed_grid_scores(X, y, ds.ranges, G, ds.n_classes)
        sig = math.sqrt(d) * scores.score_sensitivity(G, X.shape[0], ds.n_classes, "analytic") * c
        noisy = raw + sig * np.random.default_rng(fl.derive_seed(seed, "scores", i)).standard_normal(d)
        acc += X.shape[0] / M * noisy
        chk += X.shape[0] / M * np.clip(noisy, 0.0, 1.0)
    assert np.allclose(chk, pr["score_private_global"], atol=1e-9), "local release not reproduced"
    return len(set(pipeline.select_top_k(acc, k)) & exact)


def summarize_a(df):
    rows = []
    for (eps, k, method), g in df.groupby(["eps", "k", "method"], sort=False):
        g = g.sort_values("seed")
        r = {"eps": eps, "k": k, "method": method, "accuracy": fmt(*ms(g.accuracy)), "balanced accuracy": fmt(*ms(g.balanced_accuracy)),
             "overlap with exact top-k, seeds 0-4": " ".join(str(int(v)) for v in g.overlap) if g.overlap.notna().all() else "-",
             "noise std on averaged scores": f"{g.noise_std_avg.iloc[0]:.4g}" if "noise_std_avg" in g and g.noise_std_avg.notna().all() else "-",
             "exact score: largest / k-th / (k+1)-th": (f"{g.exact_max.mean():.3f} / {g.exact_kth.mean():.4f} / {g.exact_k_plus_1.mean():.4f}"
                                                        if g.exact_max.notna().all() else "-"),
             "paired diff. to random, seeds 0-4": " ".join(f"{v:+.3f}" for v in g.diff_to_random) if method != "random k-subset" else "-",
             "mean diff.": f"{g.diff_to_random.mean():+.3f}" if method != "random k-subset" else "-"}
        rows.append(r)
    out = pd.DataFrame(rows).sort_values(["eps", "k"], kind="stable")
    loc = df[df.method == "private, local"].sort_values(["eps", "k", "seed"])
    extra = loc.groupby(["eps", "k"]).overlap_unclipped_average.apply(lambda s: " ".join(str(int(v)) for v in s)).reset_index()
    extra.columns = ["eps", "k", "overlap if the average were taken without per-client clipping, seeds 0-4"]
    return out, extra


# --------------------------------------------------------------------------- #
# Part B
# --------------------------------------------------------------------------- #

def eps_grid(ds, m):
    return [1, 5] if (m == "ldpmic_oracle" and ds.startswith(("mnist", "fmnist"))) else TAB_EPS


def seeds_of(ds):
    return range(3) if ds.startswith(("mnist", "fmnist")) else range(5)


def stored_diverged(p):
    if p.get("diverged") is not None:
        return bool(p["diverged"])
    d = rounds_of(p)
    return bool((~np.isfinite(d.train_loss[d.n_participants > 0])).any())


def binary_cm(acc, bal, n0, n1):
    """Confusion matrix [[TN, FP], [FN, TP]] of a binary classifier from accuracy and balanced accuracy."""
    n = n0 + n1
    tn = (2.0 * bal - acc * n / n1) / (1.0 / n0 - 1.0 / n1)
    tp = acc * n - tn
    tn_i, tp_i = int(round(tn)), int(round(tp))
    assert abs(tn - tn_i) < 1e-3 and abs(tp - tp_i) < 1e-3, (acc, bal, tn, tp)
    return [[tn_i, n0 - tn_i], [n1 - tp_i, tp_i]]


def part_b_diag():
    """B.1: input scale of the stored Experiment 1 runs (Phases 1-3 re-run without training, seed 0) and the Adult
    confusion matrices recovered from the stored metrics."""
    rows = []
    for dsn in EXP1_DS:
        for m in EXP1_M:
            cfg = cfg_of(f"{dsn}_{m}")
            ds, parts = client_sizes(cfg, 0)
            cdata = ldata.client_data(ds, parts)
            scaler = ldata.public_scaler(ds.ranges)
            for eps in eps_grid(dsn, m):
                r = rs.resolve(cfg, eps, 0)
                mcfg = pipeline.MethodConfig(**r["method"])
                fl_cfg = fl.FLConfig(rounds=1, participation=1.0, local_epochs=1, batch_size=64, lr=0.1, clip_update=None, seed=0)
                clients, info = pipeline.build_clients(mcfg, cdata, ds.ranges, ds.n_classes, 0, fl_cfg, cache_dir=None, transform=scaler)
                tot = sum(float(c.X.abs().sum()) for c in clients)
                cnt = sum(c.X.numel() for c in clients)
                sig = np.concatenate([(c.sig / (ds.ranges[:, 1] - ds.ranges[:, 0])) for c in clients])
                stored = prov_of(f"{dsn}_{m}", eps, 0, required=False)
                sha_ok = None
                if stored is not None and stored["privacy"].get("sanitized_sha256"):
                    sha_ok = stored["privacy"]["sanitized_sha256"] == info.sanitized_sha256
                clean = float(np.mean([np.abs(scaler(X)).mean() for X, _ in cdata]))
                rows.append({"dataset": dsn, "method": M_LABEL[m], "eps": eps, "mean_abs_input": tot / cnt,
                             "scaled_noise_std_min": float(sig.min()), "scaled_noise_std_max": float(sig.max()),
                             "mean_abs_clean_input": clean, "sha256_equals_stored_run": sha_ok})
    scale = pd.DataFrame(rows)
    ds = dataset("adult")
    n1 = int((ds.y_test == 1).sum()); n0 = int((ds.y_test == 0).sum())
    cms = []
    for m in EXP1_M:
        for eps in (1, 2, 5):
            for seed in range(5):
                p = prov_of(f"adult_occ_{m}", eps, seed)
                cm = binary_cm(p["final"]["accuracy"], p["final"]["balanced_accuracy"], n0, n1)
                d = rounds_of(p)
                cms.append({"method": M_LABEL[m], "eps": eps, "seed": seed, "accuracy": p["final"]["accuracy"],
                            "TN": cm[0][0], "FP": cm[0][1], "FN": cm[1][0], "TP": cm[1][1],
                            "predicted_class_0": cm[0][0] + cm[1][0], "predicted_class_1": cm[0][1] + cm[1][1],
                            "non_finite_loss_rounds": int((~np.isfinite(d.train_loss)).sum()),
                            "max_round_mean_update_norm": float(d.mean_update_norm.max())})
    return scale, pd.DataFrame(cms), (n0, n1)


def part_b(prefix="ch5d3_B_"):
    rows = []
    for dsn in EXP1_DS:
        ds = dataset(cfg_of(f"{dsn}_iso")["dataset"])
        binary = ds.n_classes == 2
        n1 = int((ds.y_test == 1).sum()); n0 = int((ds.y_test == 0).sum())
        for m in EXP1_M:
            for eps in eps_grid(dsn, m):
                for seed in seeds_of(dsn):
                    old = prov_of(f"{dsn}_{m}", eps, seed)
                    if not (ROOT / "configs" / f"{prefix}{dsn}_{m}.yaml").exists():
                        continue
                    new = prov_of(f"{prefix}{dsn}_{m}", eps, seed, NEW, required=prefix == "ch5d3_B_")
                    if new is None:
                        continue
                    cm = np.array(new["confusion_matrix"])
                    old_single = None
                    if binary:
                        ocm = np.array(binary_cm(old["final"]["accuracy"], old["final"]["balanced_accuracy"], n0, n1))
                        old_single = bool((ocm.sum(axis=0) > 0).sum() == 1)
                    sha = None
                    if old["privacy"].get("sanitized_sha256"):
                        sha = old["privacy"]["sanitized_sha256"] == new["privacy"]["sanitized_sha256"]
                    rows.append({"part": "B", "dataset": dsn, "method": M_LABEL[m], "config": new["config"]["name"], "run_id": new["run_id"],
                                 "eps": eps, "seed": seed, "lr": new["config"]["fl"]["lr"],
                                 "accuracy": new["final"]["accuracy"], "balanced_accuracy": new["final"]["balanced_accuracy"],
                                 "auc": new["final"].get("auc"), "diverged": bool(new["diverged"]),
                                 "single_class": bool((cm.sum(axis=0) > 0).sum() == 1), "mean_abs_input": new.get("mean_abs_input"),
                                 "rescale_min": new["privacy"]["input_rescale_range"][0], "rescale_max": new["privacy"]["input_rescale_range"][1],
                                 "stored_run_id": old["run_id"], "stored_accuracy": old["final"]["accuracy"],
                                 "stored_balanced_accuracy": old["final"]["balanced_accuracy"], "stored_diverged": stored_diverged(old),
                                 "stored_single_class": old_single, "sha256_equals_stored_run": sha,
                                 "git_commit": new["git_commit"], "duration_s": new["duration_s"]})
    return pd.DataFrame(rows)


def summarize_b(df):
    out = {}
    for dsn, g in df.groupby("dataset", sort=False):
        rows = []
        for (m, eps), c in g.groupby(["method", "eps"], sort=False):
            n = len(c)
            sc_old = "-" if c.stored_single_class.isna().all() else f"{int(c.stored_single_class.sum())}/{n}"
            rows.append({"method": m, "eps": eps,
                         "stored: accuracy": fmt(*ms(c.stored_accuracy)), "stored: failed": f"{int(c.stored_diverged.sum())}/{n}",
                         "stored: single-class": sc_old,
                         "rescaled: accuracy": fmt(*ms(c.accuracy)), "rescaled: balanced acc.": fmt(*ms(c.balanced_accuracy)),
                         "rescaled: failed": f"{int(c.diverged.sum())}/{n}", "rescaled: single-class": f"{int(c.single_class.sum())}/{n}",
                         "mean abs. input": f"{c.mean_abs_input.mean():.3g}"})
        out[dsn] = pd.DataFrame(rows)
    return out


# --------------------------------------------------------------------------- #
# Part C
# --------------------------------------------------------------------------- #

def _nm(p):
    v = np.array(list(p["privacy"]["dpsgd_noise_multiplier"].values()), dtype=float)
    return float(v.min()), float(np.median(v)), float(v.max())


def part_c():
    rules = yaml.safe_load(open(OUT / "config.yaml"))["part_C"]["clip_update"]
    rows = []
    for ds in ("mnist", "fmnist"):
        for eps in TAB_EPS:
            for seed in range(3):
                versions = [("stored, no update clipping", f"{ds}_a0.5_local_dpsgd", RAW)]
                if eps in (1, 5):
                    versions.append(("stored, fixed C (chosen at eps = 1)", f"{ds}_a0.5_v8_local_dpsgd", RAW))
                if eps == 1:
                    versions.append(("C chosen per budget", f"{ds}_a0.5_v8_local_dpsgd", RAW))
                else:
                    versions.append(("C chosen per budget", f"ch5d3_C_{ds}_a0.5_local_dpsgd_e{eps:g}", NEW))
                for label, name, where in versions:
                    p = prov_of(name, eps, seed, where)
                    d = rounds_of(p, where)
                    C = p["config"]["fl"].get("clip_update")
                    if label == "C chosen per budget":
                        assert abs(C - rules[ds][eps]) < 1e-9, (ds, eps, C)
                    lo, med, hi = _nm(p)
                    row = {"part": "C", "dataset": ds, "version": label, "config": name, "run_id": p["run_id"], "eps": eps, "seed": seed,
                           "accuracy": p["final"]["accuracy"], "clip_update": C, "calibration": p["config"]["method"].get("dpsgd_calibration", "(code default at the run's commit)"),
                           "noise_multiplier_min": lo, "noise_multiplier_median": med, "noise_multiplier_max": hi,
                           "median_round_mean_update_norm": float(d.mean_update_norm.median()),
                           "share_rounds_mean_norm_above_C": (float((d.mean_update_norm > C).mean()) if C else 0.0),
                           "eps_realized": p["privacy"]["eps_realized"], "git_commit": p["git_commit"], "stored": where == RAW,
                           "duration_s": p["duration_s"]}
                    nf = where / f"{p['run_id']}_norms.npz"
                    if nf.exists():
                        z = np.load(nf)
                        row["median_update_norm"] = float(np.median(z["norm"]))
                        row["share_updates_clipped"] = float((z["norm"] > C).mean())
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize_c(df):
    out = {}
    for ds, g in df.groupby("dataset", sort=False):
        acc = g.groupby(["version", "eps"], sort=False).accuracy.agg(["mean", "std", "count"]).reset_index()
        acc["cell"] = [fmt(m, s) for m, s in zip(acc["mean"], acc["std"])]
        t = acc.pivot(index="version", columns="eps", values="cell").reindex(
            ["stored, no update clipping", "stored, fixed C (chosen at eps = 1)", "C chosen per budget"]).fillna("-")
        t.columns = [f"eps = {c:g}" for c in t.columns]
        diag = []
        for (v, eps), c in g.groupby(["version", "eps"], sort=False):
            diag.append({"version": v, "eps": eps,
                         "noise multiplier (min / median / max over clients)": f"{c.noise_multiplier_min.iloc[0]:.3f} / {c.noise_multiplier_median.iloc[0]:.3f} / {c.noise_multiplier_max.iloc[0]:.3f}",
                         "update clipping bound C": "none" if pd.isna(c.clip_update.iloc[0]) else f"{c.clip_update.iloc[0]:g}",
                         "median round-mean update norm before clipping, seeds 0-2": " ".join(f"{x:.2f}" for x in c.sort_values("seed").median_round_mean_update_norm),
                         "share of rounds with mean norm > C": "-" if pd.isna(c.clip_update.iloc[0]) else f"{c.share_rounds_mean_norm_above_C.mean():.2f}",
                         "median update norm (per update)": f"{c.median_update_norm.mean():.2f}" if "median_update_norm" in c and c.median_update_norm.notna().all() else "not stored",
                         "share of updates clipped": f"{c.share_updates_clipped.mean():.2f}" if "share_updates_clipped" in c and c.share_updates_clipped.notna().all() else "not stored"})
        out[ds] = (t.reset_index(), pd.DataFrame(diag))
    return out


def part_c3(df):
    lines = []
    for ds, name in (("mnist", "MNIST"), ("fmnist", "Fashion-MNIST")):
        for eps in (1, 5):
            a = df[(df.dataset == ds) & (df.eps == eps) & (df.version == "stored, no update clipping")].sort_values("seed")
            b = df[(df.dataset == ds) & (df.eps == eps) & (df.version == "stored, fixed C (chosen at eps = 1)")].sort_values("seed")
            lines.append(f"- {name}, eps = {eps:g}: `{a.config.iloc[0]}` (update clipping {'none' if pd.isna(a.clip_update.iloc[0]) else a.clip_update.iloc[0]}) "
                         f"mean {a.accuracy.mean():.4f}, per seed {' '.join(f'{x:.4f}' for x in a.accuracy)}; "
                         f"`{b.config.iloc[0]}` (update clipping {b.clip_update.iloc[0]:g}) mean {b.accuracy.mean():.4f}, per seed {' '.join(f'{x:.4f}' for x in b.accuracy)}; "
                         f"share of rounds whose mean update norm exceeds {b.clip_update.iloc[0]:g} in the clipped runs: {b.share_rounds_mean_norm_above_C.mean():.2f}; "
                         f"noise multipliers identical: {bool(np.allclose(a.noise_multiplier_median, b.noise_multiplier_median) and np.allclose(a.noise_multiplier_max, b.noise_multiplier_max))}.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Part D
# --------------------------------------------------------------------------- #

def part_d():
    rows = []
    for name, label in (("adult_occ_v3_fs_dpsgd_k10", "Adult (occupation)"), ("mnist_a0.5_v3_fs_dpsgd_k100", "MNIST (alpha 0.5)")):
        p = prov_of(name, 1, 0)
        cfg = p["config"]
        ds = dataset(cfg["dataset"])
        pr = p["privacy"]
        G = int(cfg["method"].get("G", 10))
        sn = score_noise(p["client_sizes"], ds.d, ds.n_classes, G, pr["eps1"], pr["delta1"])
        sizes = np.array(p["client_sizes"], dtype=float)
        rows.append({"pilot": label, "config": name, "d": ds.d, "n_clients": len(sizes), "n_min": sn["n_min"], "n_max": sn["n_max"], "n_sum": sn["n_sum"],
                     "eps1": pr["eps1"], "delta1": pr["delta1"], "c(eps1, delta1)": sn["c"], "G": G, "n_classes": ds.n_classes,
                     "score normalizer ln(min(G, classes))": sn["normalizer_ln_min_G_C"],
                     "Delta_FG(n_max), task formula": delta_fg(sizes.max()), "Delta_s(n_max), code": sn["delta_s_at_nmax_code"],
                     "max_i n_i Delta / sum n, task formula": float(max(m * min(1.0, delta_fg(m)) for m in sizes) / sizes.sum()),
                     "max_i n_i Delta / sum n, code": float(max(m * scores.score_sensitivity(G, int(m), ds.n_classes, 'analytic') for m in sizes) / sizes.sum()),
                     "sigma, task formula": sn["secagg_task"], "sigma, code path recomputed": sn["secagg_code"],
                     "sigma, recorded by the run (sigma_s)": pr["sigma_s"],
                     "sum sensitivity, recorded": pr["score_sensitivity_value"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Part E
# --------------------------------------------------------------------------- #

def part_e():
    rows = []
    for eps in (100, 200):
        for k in (10, 20):
            for seed in range(5):
                a = prov_of(f"adult_occ_v7_sel_mi_k{k}", eps, seed)
                b = prov_of(f"adult_occ_v7_sel_random_k{k}", eps, seed)
                rows.append({"part": "E", "eps": eps, "k": k, "seed": seed, "private_selection_run": a["run_id"], "random_subset_run": b["run_id"],
                             "accuracy_private_selection": a["final"]["accuracy"], "accuracy_random_subset": b["final"]["accuracy"],
                             "difference": a["final"]["accuracy"] - b["final"]["accuracy"]})
    df = pd.DataFrame(rows)
    s = []
    for (eps, k), g in df.groupby(["eps", "k"], sort=False):
        s.append({"eps": eps, "k": k, "private selection, seeds 0-4": " ".join(f"{v:.4f}" for v in g.accuracy_private_selection),
                  "random k-subset, seeds 0-4": " ".join(f"{v:.4f}" for v in g.accuracy_random_subset),
                  "paired differences": " ".join(f"{v:+.4f}" for v in g.difference), "mean": f"{g.difference.mean():+.4f}",
                  "range": f"{g.difference.min():+.4f} to {g.difference.max():+.4f}", "seeds ahead": f"{int((g.difference > 0).sum())}/5"})
    return df, pd.DataFrame(s)


# --------------------------------------------------------------------------- #
# Part F
# --------------------------------------------------------------------------- #

def part_f():
    rows = []
    for eps in (10, 50, 100, 200):
        for seed in range(5):
            arms = [("group-wise sensitivity", "ch5d3_F_adult_occ_iso_group", NEW),
                    ("uniform, sqrt(d)", "ch5d3_F_adult_occ_iso_ref10" if eps == 10 else "adult_occ_v7_iso", NEW if eps == 10 else RAW)]
            if eps == 10:
                arms.append(("uniform, sqrt(d), stored without update clipping", "adult_occ_iso", RAW))
            for label, name, where in arms:
                p = prov_of(name, eps, seed, where)
                sig = np.array(list(p["privacy"]["sigmas"].values())[0])
                rng_ = np.array(dataset("adult").ranges); dl = rng_[:, 1] - rng_[:, 0]
                rows.append({"part": "F", "arm": label, "config": name, "run_id": p["run_id"], "eps": eps, "seed": seed,
                             "accuracy": p["final"]["accuracy"], "balanced_accuracy": p["final"]["balanced_accuracy"], "auc": p["final"].get("auc"),
                             "scaled_noise_std": float((sig / dl)[0]), "clip_update": p["config"]["fl"].get("clip_update"),
                             "l2_sens_sq": (p.get("group_sensitivity") or {}).get("l2_sens_sq", len(sig)),
                             "n_numeric": (p.get("group_sensitivity") or {}).get("n_numeric"), "n_categorical": (p.get("group_sensitivity") or {}).get("n_categorical"),
                             "git_commit": p["git_commit"], "stored": where == RAW})
    df = pd.DataFrame(rows)
    s = []
    for (eps, arm), g in df.groupby(["eps", "arm"], sort=False):
        s.append({"eps": eps, "arm": arm, "squared l2 sensitivity (units of Delta_j)": f"{g.l2_sens_sq.iloc[0]:g}",
                  "noise std / Delta_j": f"{g.scaled_noise_std.iloc[0]:.3f}", "update clipping": "none" if pd.isna(g.clip_update.iloc[0]) else f"{g.clip_update.iloc[0]:g}",
                  "accuracy": fmt(*ms(g.accuracy)), "balanced accuracy": fmt(*ms(g.balanced_accuracy)), "AUC": fmt(*ms(g.auc))})
    return df, pd.DataFrame(s)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def decision_entries():
    """The follow-up's entries of DECISIONS.md, verbatim (D65 onwards)."""
    txt = (ROOT / "DECISIONS.md").read_text()
    i = txt.index("## D65 ")
    return txt[i:].replace("\n## ", "\n### ").replace("## D65 ", "### D65 ", 1).strip()


def diag():
    DIAG.mkdir(parents=True, exist_ok=True)
    d = part_d(); d.to_csv(DIAG / "partD.csv", index=False); print(d.T.to_string())
    e, es = part_e(); e.to_csv(DIAG / "partE.csv", index=False); print(es.to_string())
    scale, cms, n = part_b_diag()
    scale.to_csv(DIAG / "partB1_input_scale.csv", index=False); cms.to_csv(DIAG / "partB1_adult_confusion.csv", index=False)
    print(scale.to_string()); print(cms.to_string()); print("adult test class counts", n)


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    S = ["# Chapter 5 follow-up runs for the Draft 2 feedback", ""]
    notes_path = OUT / "summary_text.yaml"       # the prose of the summary (no numbers); the tables are generated below
    T = yaml.safe_load(open(notes_path))
    S += [T["header"], ""]

    a, a_noise = part_a(); a.to_csv(OUT / "partA.csv", index=False); a_noise.to_csv(OUT / "partA_mnist_score_noise.csv", index=False)
    a_sum, a_extra = summarize_a(a)
    S += ["## Part A. Selection in the local score-release mode", "", T["part_A"], "", md_table(a_sum), "", md_table(a_extra), "",
          "Score noise on MNIST (no training; eps1 = 0.2 eps, delta1 = 5e-6):", "", md_table(a_noise.round(5)), ""]

    scale = pd.read_csv(DIAG / "partB1_input_scale.csv"); cms = pd.read_csv(DIAG / "partB1_adult_confusion.csv")
    b = part_b(); b.to_csv(OUT / "partB.csv", index=False)
    S += ["## Part B. Experiment 1 with stable training", "", "### B.1 Diagnosis (stored runs, no new training)", "", T["part_B1"], "",
          "Input scale of the stored runs (seed 0; Phases 1-3 re-run without training):", "",
          md_table(scale.assign(**{c: scale[c].map(lambda v: f"{v:.4g}") for c in ("mean_abs_input", "scaled_noise_std_min", "scaled_noise_std_max", "mean_abs_clean_input")})), "",
          "Adult, stored Experiment 1, confusion matrix of the final model per seed (rows of the matrix: true class; class 1 is the positive, minority class):", "",
          md_table(cms.assign(accuracy=cms.accuracy.map(lambda v: f"{v:.4f}"), max_round_mean_update_norm=cms.max_round_mean_update_norm.map(lambda v: f"{v:.3g}"))), "",
          "### B.2 / B.3 Rescaled rerun next to the stored table", "", T["part_B2"], ""]
    for dsn, t in summarize_b(b).items():
        S += [f"**{dsn}**", "", md_table(t), ""]
    sha = b.sha256_equals_stored_run.dropna()
    S += [f"Sanity check: released arrays identical (SHA-256, all clients) to the stored run of the same config, eps and seed in "
          f"{int(sha.sum())} of {len(sha)} runs where the stored run recorded the hashes ({int(b.sha256_equals_stored_run.isna().sum())} image runs: hashes not stored).", ""]
    n_div = int(b.diverged.sum())
    S += [f"Rescaled runs flagged as failed numerically: {n_div} of {len(b)}.", ""]
    blr = part_b("ch5d3_Blr_")
    if len(blr):
        blr.to_csv(OUT / "partB_lr10.csv", index=False)
        S += ["Repeat with the learning rate divided by 10 (datasets with at least one failed rescaled run):", ""]
        for dsn, t in summarize_b(blr).items():
            S += [f"**{dsn}, lr / 10**", "", md_table(t.drop(columns=[c for c in t.columns if c.startswith("stored")])), ""]
    else:
        S += ["The repeat with lr / 10 was not triggered." if n_div == 0 else "REPEAT WITH lr / 10 PENDING.", ""]

    try:
        c = part_c(); c.to_csv(OUT / "partC.csv", index=False)
        S += ["## Part C. Local DP-SGD baseline", "", T["part_C"], ""]
        for ds, (t, dg) in summarize_c(c).items():
            S += [f"**{ds}: accuracy (mean ± std, 3 seeds)**", "", md_table(t), "", f"**{ds}: diagnostics (C.1)**", "", md_table(dg), ""]
        S += ["### C.3 Reconciliation of the eps = 1 values", "", T["part_C3"], "", part_c3(c), ""]
    except Missing as ex:
        S += ["## Part C. Local DP-SGD baseline", "", f"NOT COMPLETE (runs in progress): {ex}", ""]

    d = part_d(); d.to_csv(OUT / "partD.csv", index=False)
    dd = d.set_index("pilot").T.reset_index().rename(columns={"index": "quantity"})
    for col in dd.columns[1:]:
        dd[col] = dd[col].map(lambda v: f"{v:.6g}" if isinstance(v, float) else v)
    S += ["## Part D. Aggregated score noise", "", T["part_D"], "", md_table(dd), ""]

    e, es = part_e(); e.to_csv(OUT / "partE.csv", index=False)
    S += ["## Part E. Paired comparison for the selection result", "", T["part_E"], "", md_table(es), ""]

    try:
        f, fs_ = part_f(); f.to_csv(OUT / "partF.csv", index=False)
        S += ["## Part F. Group-wise sensitivity (optional)", "", T["part_F"], "", md_table(fs_), ""]
    except Missing as ex:
        S += ["## Part F. Group-wise sensitivity (optional)", "", f"Not completed: {ex}", ""]

    # runtime
    new_runs = [json.load(open(p)) for p in sorted(NEW.glob("*.json"))]
    tot = sum(j["duration_s"] for j in new_runs)
    by = {}
    for j in new_runs:
        k = j["config"]["name"].split("_")[1]
        by[k] = by.get(k, 0.0) + j["duration_s"]
    starts = [pd.Timestamp(j["start_time"]) for j in new_runs]; ends = [pd.Timestamp(j["end_time"]) for j in new_runs]
    S += ["## Runtime", "", f"{len(new_runs)} new runs, {sum(1 for j in new_runs if j['status'] != 'completed')} failed to complete; "
          f"summed run time {tot / 3600:.1f} h (" + ", ".join(f"Part {k}: {v / 3600:.2f} h" for k, v in sorted(by.items())) + "); "
          f"first start {min(starts):%Y-%m-%d %H:%M} UTC, last end {max(ends):%Y-%m-%d %H:%M} UTC; 4 parallel single-thread workers on a 4-core CPU container.", "",
          "Code commits of the new runs: " + ", ".join(sorted({j["git_commit"][:12] for j in new_runs})) + ".", ""]
    S += ["## Decision log", "", T["decision_log"], "", decision_entries(), "", "## Deviations", "", T["deviations"], ""]
    (OUT / "summary.md").write_text("\n".join(S))
    used = {"config": yaml.safe_load(open(OUT / "config.yaml")),
            "resolved_configs": {p.stem: yaml.safe_load(open(p)) for p in sorted((ROOT / "configs").glob("ch5d3_*.yaml"))},
            "git_commits_of_new_runs": sorted({j["git_commit"] for j in new_runs})}
    with open(OUT / "config_used.yaml", "w") as fh:
        yaml.safe_dump(used, fh, sort_keys=False)
    print("wrote", OUT / "summary.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diag", "report"])
    a_ = ap.parse_args()
    diag() if a_.cmd == "diag" else report()
