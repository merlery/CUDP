"""Paired tests, Holm families, bootstrap CIs, and the pre-registered hypothesis decisions.

Utility U is always "higher is better": U = ROC-AUC (classification) or U = -MAE (regression).
A paired difference is U(method A) - U(method B) for the same trial seed.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from . import data as D

MAIN_BASELINES = ["uniform", "pearson_alloc", "pearson_adj", "partialcorr_adj"]


# ----------------------------------------------------------------------------- primitives

def wilcoxon_p(diff: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    if np.all(diff == 0):
        return float("nan"), 1.0
    res = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def bootstrap_mean_ci(x: np.ndarray, rng: np.random.Generator, B: int = 10000,
                      level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, x.size, size=(B, x.size))
    m = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def load_trials(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(D.ROOT / cfg["outputs"]["dir"] / "raw_trials.parquet")


def utility_matrix(raw: pd.DataFrame, model: str = "primary", experiment: str = "main") -> pd.DataFrame:
    """Rows (dataset, eps, method, trial) -> U, primary_value, distortion."""
    sub = raw[(raw.experiment == experiment) & (raw.model == model)]
    return sub.set_index(["dataset", "eps", "method", "trial"])[["U", "primary_value", "distortion"]]


def paired_U(raw: pd.DataFrame, dataset: str, eps: float, a: str, b: str,
             model: str = "primary") -> tuple[np.ndarray, np.ndarray]:
    sub = raw[(raw.experiment == "main") & (raw.model == model) & (raw.dataset == dataset)]
    A = sub[(sub.method == a) & (sub.eps == eps)].set_index("trial").U.sort_index()
    if b == "nonprivate":
        Bs = sub[sub.method == "nonprivate"].set_index("trial").U.sort_index()
    else:
        Bs = sub[(sub.method == b) & (sub.eps == eps)].set_index("trial").U.sort_index()
    assert A.index.equals(Bs.index) and len(A) > 0, (dataset, eps, a, b)
    return A.to_numpy(), Bs.to_numpy()


# ----------------------------------------------------------------------------- families

def family_specs(cfg: dict) -> dict[str, list[tuple[str, float, str, str]]]:
    real = D.real_dataset_names(cfg)
    eps = cfg["privacy"]["eps_grid"]
    fam = {
        "F1": [(d, e, "mic_dp", b) for d in real for e in eps for b in MAIN_BASELINES],
        "F2": [(d, e, "mic_dp", "pearson_alloc") for d in ("syn_lin", "syn_nl_mid") for e in eps],
        "F3": [(d, e, "mic_dp", "random_alloc") for d in real for e in eps],
        "F4": [(d, e, "mic_public", "mic_dp") for d in real for e in eps],
        "F5": [("adult", e, "mic_dp_override", "mic_dp") for e in eps],
    }
    return fam


FAMILY_HYP = {"F1": "H1", "F2": "H2", "F3": "H3", "F4": "H5", "F5": "H6", "F1_secondary": "-"}


def run_family(raw: pd.DataFrame, name: str, tests: list, cfg: dict, model: str = "primary",
               fam_index: int = 0) -> pd.DataFrame:
    rows = []
    B = cfg["stats"]["bootstrap_resamples"]
    for k, (d, e, a, b) in enumerate(tests):
        UA, UB = paired_U(raw, d, e, a, b, model)
        diff = UA - UB
        W, p = wilcoxon_p(diff)
        rng = np.random.default_rng([cfg["seeds"]["bootstrap"], fam_index, k])
        lo, hi = bootstrap_mean_ci(diff, rng, B, cfg["stats"]["ci"])
        rows.append({"family": name, "hypothesis": FAMILY_HYP[name], "dataset": d, "eps": e,
                     "method_a": a, "method_b": b, "model": model, "n": diff.size,
                     "mean_U_a": UA.mean(), "mean_U_b": UB.mean(), "mean_diff_U": diff.mean(),
                     "ci_lo": lo, "ci_hi": hi, "wilcoxon_W": W, "p_raw": p,
                     "n_zero_diffs": int(np.sum(diff == 0))})
    df = pd.DataFrame(rows)
    df["p_holm"] = multipletests(df.p_raw.to_numpy(), alpha=cfg["stats"]["alpha_level"], method="holm")[1]
    sig = df.p_holm < cfg["stats"]["alpha_level"]
    df["outcome"] = np.where(sig & (df.mean_diff_U > 0), "A better",
                             np.where(sig & (df.mean_diff_U < 0), "B better", "n.s."))
    return df


def all_tests(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fams = family_specs(cfg)
    out = [run_family(raw, n, t, cfg, "primary", i) for i, (n, t) in enumerate(fams.items())]
    out.append(run_family(raw, "F1_secondary", fams["F1"], cfg, "secondary", 99))
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- UR / summaries

def summary_table(raw: pd.DataFrame, model: str = "primary") -> pd.DataFrame:
    """Mean / std of primary metric, U, distortion per (dataset, eps, method)."""
    sub = raw[(raw.experiment == "main") & (raw.model == model)]
    g = sub.groupby(["dataset", "eps", "method"], dropna=False)
    s = g.agg(metric_mean=("primary_value", "mean"), metric_std=("primary_value", "std"),
              U_mean=("U", "mean"), U_std=("U", "std"), dist_mean=("distortion", "mean"),
              dist_std=("distortion", "std"), n=("trial", "count"),
              const_pred_frac=("n_unique_pred", lambda v: float(np.mean(np.asarray(v) <= 1)))).reset_index()
    return s


def ur_table(raw: pd.DataFrame, model: str = "primary") -> pd.DataFrame:
    """UR = (U_m - U_uni) / (U_np - U_uni) from trial means, plus per-trial UR mean/std and
    relative improvement over Uniform in the primary metric (positive = better)."""
    sub = raw[(raw.experiment == "main") & (raw.model == model)]
    rows = []
    for d, gd in sub.groupby("dataset"):
        np_ = gd[gd.method == "nonprivate"].set_index("trial").U.sort_index()
        for e, ge in gd[gd.method != "nonprivate"].groupby("eps"):
            uni = ge[ge.method == "uniform"].set_index("trial").U.sort_index()
            for m, gm in ge.groupby("method"):
                um = gm.set_index("trial").U.sort_index()
                den = np_.mean() - uni.mean()
                ur = (um.mean() - uni.mean()) / den if den != 0 else np.nan
                den_t = (np_ - uni).to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    ur_t = (um.to_numpy() - uni.to_numpy()) / den_t
                rel = (um.mean() - uni.mean()) / abs(uni.mean())
                rows.append({"dataset": d, "eps": e, "method": m, "UR": ur,
                             "UR_trial_mean": np.nanmean(ur_t), "UR_trial_std": np.nanstd(ur_t, ddof=1),
                             "rel_improvement_vs_uniform": rel, "U_mean": um.mean(),
                             "U_uniform": uni.mean(), "U_nonprivate": np_.mean()})
    return pd.DataFrame(rows)


def _syn_U(raw: pd.DataFrame, cfg: dict, e: float, method: str = "mic_dp"):
    """(variants, log lambda, U arrays [variant x trial] for method, uniform, non-private)."""
    variants = [k for k, v in cfg["synthetic"]["variants"].items() if v["signal"] == "nonlinear"]
    lam = np.log([cfg["synthetic"]["variants"][k]["lambda"] for k in variants])
    sub = raw[(raw.experiment == "main") & (raw.model == "primary")]
    M, Un, Np = [], [], []
    for v in variants:
        g = sub[sub.dataset == v]
        Np.append(g[g.method == "nonprivate"].set_index("trial").U.sort_index().to_numpy())
        ge = g[g.eps == e]
        Un.append(ge[ge.method == "uniform"].set_index("trial").U.sort_index().to_numpy())
        M.append(ge[ge.method == method].set_index("trial").U.sort_index().to_numpy())
    return variants, lam, np.vstack(M), np.vstack(Un), np.vstack(Np)


def ur_of_means(M, Un, Np):
    """UR per row (variant) from trial means; arrays are [variant x trial]."""
    return (M.mean(axis=1) - Un.mean(axis=1)) / (Np.mean(axis=1) - Un.mean(axis=1))


def h4_slopes(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per eps: OLS slope of UR(MIC-DP) (ratio of trial means) on log(lambda) over the three
    nonlinear variants; percentile bootstrap CI resampling trial seeds jointly across variants."""
    rows = []
    B = cfg["stats"]["bootstrap_resamples"]
    for ei, e in enumerate(cfg["privacy"]["eps_grid"]):
        variants, lam, M, Un, Np = _syn_U(raw, cfg, e)
        xc = lam - lam.mean()

        def slope(ur):
            return float((xc * (ur - ur.mean())).sum() / (xc ** 2).sum())

        ur = ur_of_means(M, Un, Np)
        s = slope(ur)
        n = M.shape[1]
        rng = np.random.default_rng([cfg["seeds"]["bootstrap"], 400, ei])
        bs, bur = np.empty(B), np.empty((B, len(variants)))
        for b in range(B):
            i = rng.integers(0, n, n)
            bur[b] = ur_of_means(M[:, i], Un[:, i], Np[:, i])
            bs[b] = slope(bur[b])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        row = {"eps": e, "slope": s, "ci_lo": float(lo), "ci_hi": float(hi)}
        for k, v in enumerate(variants):
            row[f"UR_{v}"] = float(ur[k])
            row[f"UR_{v}_ci_lo"], row[f"UR_{v}_ci_hi"] = (float(x) for x in np.quantile(bur[:, k], [0.025, 0.975]))
        rows.append(row)
    return pd.DataFrame(rows)


def ur_bootstrap_se(raw: pd.DataFrame, cfg: dict, dataset: str, eps: float, method: str,
                    B: int | None = None) -> float:
    """Bootstrap standard error of UR (ratio of trial means), resampling trial seeds."""
    B = B or cfg["stats"]["bootstrap_resamples"]
    sub = raw[(raw.experiment == "main") & (raw.model == "primary") & (raw.dataset == dataset)]
    Np = sub[sub.method == "nonprivate"].set_index("trial").U.sort_index().to_numpy()
    ge = sub[sub.eps == eps]
    Un = ge[ge.method == "uniform"].set_index("trial").U.sort_index().to_numpy()
    M = ge[ge.method == method].set_index("trial").U.sort_index().to_numpy()
    rng = np.random.default_rng([cfg["seeds"]["bootstrap"], 500, int(round(eps * 100))])
    n = M.size
    I = rng.integers(0, n, size=(B, n))
    ur = (M[I].mean(1) - Un[I].mean(1)) / (Np[I].mean(1) - Un[I].mean(1))
    return float(np.std(ur, ddof=1))


def h6_equivalence(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Loss = AUC(MIC-DP) - AUC(MIC-DP+Override), Adult, primary model."""
    rows = []
    B = cfg["stats"]["bootstrap_resamples"]
    for ei, e in enumerate(cfg["privacy"]["eps_grid"]):
        a, b = paired_U(raw, "adult", e, "mic_dp", "mic_dp_override")
        loss = a - b
        lo, hi = bootstrap_mean_ci(loss, np.random.default_rng([cfg["seeds"]["bootstrap"], 600, ei]), B)
        rows.append({"eps": e, "mean_loss": loss.mean(), "ci_lo": lo, "ci_hi": hi,
                     "within_margin": hi <= cfg["hypotheses"]["H6_margin"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- hypotheses

def decide(tests: pd.DataFrame, raw: pd.DataFrame, cfg: dict) -> dict:
    hy = cfg["hypotheses"]
    real = D.real_dataset_names(cfg)
    out = {}

    # H1
    f1 = tests[(tests.family == "F1") & (tests.method_b == "uniform")]
    per_ds = {}
    for d in real:
        cells = f1[(f1.dataset == d) & (f1.eps.isin(hy["H1_eps"]))]
        per_ds[d] = bool((cells.outcome == "A better").all() and len(cells) == len(hy["H1_eps"]))
    n_ok = sum(per_ds.values())
    any_sig = int((f1.outcome == "A better").sum())
    verdict = "PASS" if n_ok >= 3 else ("PARTIAL" if (n_ok >= 1 or any_sig > 0) else "FAIL")
    out["H1"] = {"verdict": verdict, "datasets_qualifying": per_ds, "n_qualifying": n_ok,
                 "sig_wins_vs_uniform_all_eps": any_sig,
                 "sig_losses_vs_uniform_all_eps": int((f1.outcome == "B better").sum())}

    # H2
    f2 = tests[tests.family == "F2"]
    nl = f2[f2.dataset == "syn_nl_mid"]
    lin = f2[f2.dataset == "syn_lin"]
    part_a = int((nl.outcome == "A better").sum()) >= hy["H2_min_eps"]
    part_b = int((lin.outcome != "n.s.").sum()) == 0
    verdict = "PASS" if (part_a and part_b) else ("PARTIAL" if (part_a or part_b) else "FAIL")
    out["H2"] = {"verdict": verdict, "nl_mid_sig_wins": int((nl.outcome == "A better").sum()),
                 "nl_mid_sig_losses": int((nl.outcome == "B better").sum()),
                 "lin_sig_differences": int((lin.outcome != "n.s.").sum()),
                 "part_nonlinear": part_a, "part_linear_no_difference": part_b}

    # H3
    f3 = tests[tests.family == "F3"]
    wins, losses = int((f3.outcome == "A better").sum()), int((f3.outcome == "B better").sum())
    verdict = "PASS" if (wins >= hy["H3_min_cells"] and losses == 0) else ("PARTIAL" if wins >= 1 else "FAIL")
    out["H3"] = {"verdict": verdict, "sig_wins": wins, "sig_losses": losses, "cells": len(f3)}

    # H4
    h4 = h4_slopes(raw, cfg)
    n_pos = int((h4.ci_lo > 0).sum())
    verdict = "PASS" if n_pos >= hy["H4_min_eps"] else ("PARTIAL" if n_pos >= 1 else "FAIL")
    out["H4"] = {"verdict": verdict, "eps_with_positive_slope_CI": n_pos,
                 "eps_with_negative_slope_CI": int((h4.ci_hi < 0).sum()), "table": h4}

    # H5
    f4 = tests[tests.family == "F4"]
    part_a = int((f4.outcome == "B better").sum()) == 0
    gaps = {}
    for d in real:
        g = f4[f4.dataset == d].set_index("eps").mean_diff_U
        gaps[d] = (float(g.loc[hy["H5_small_eps"]].mean()), float(g.loc[hy["H5_large_eps"]].mean()))
    n_b = sum(1 for s, l in gaps.values() if s > l)
    part_b = n_b >= 3
    verdict = "PASS" if (part_a and part_b) else ("PARTIAL" if (part_a or part_b) else "FAIL")
    out["H5"] = {"verdict": verdict, "public_sig_better": int((f4.outcome == "A better").sum()),
                 "public_sig_worse": int((f4.outcome == "B better").sum()), "part_never_worse": part_a,
                 "gap_small_vs_large_eps": gaps, "datasets_gap_larger_at_small_eps": n_b,
                 "part_gap_largest_small_eps": part_b}

    # H6
    h6 = h6_equivalence(raw, cfg)
    n_in = int(h6.within_margin.sum())
    verdict = "PASS" if n_in == len(h6) else ("PARTIAL" if n_in >= 3 else "FAIL")
    f5 = tests[tests.family == "F5"]
    out["H6"] = {"verdict": verdict, "eps_within_margin": n_in, "table": h6,
                 "sig_override_worse": int((f5.outcome == "B better").sum()),
                 "sig_override_better": int((f5.outcome == "A better").sum())}
    return out


def main() -> None:
    cfg = D.load_config()
    raw = load_trials(cfg)
    tests = all_tests(raw, cfg)
    out = D.ROOT / cfg["outputs"]["dir"]
    tests.to_csv(out / "stats.csv", index=False)
    print(tests.groupby(["family", "outcome"]).size())


if __name__ == "__main__":
    main()
