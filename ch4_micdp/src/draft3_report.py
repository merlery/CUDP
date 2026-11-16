"""Part A analyses, statistics, LaTeX table rows, figure and summary.md for the Draft 3 follow-up runs.

    python -m src.draft3_report
"""
from __future__ import annotations

import subprocess
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from . import data as D  # noqa: E402
from . import stats as ST  # noqa: E402

BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
WIDTH = 6.3
DS = ["adult", "cdc_diabetes"]
DS_TXT = {"adult": "Adult", "cdc_diabetes": "CDC-Diabetes"}
M_TXT = {"uniform": "Uniform", "mic_dp": "MIC-DP", "mic_public": "MIC-Public (15%)", "nonprivate": "Non-private",
         "ft_uniform": "Public-Pretrain + Uniform-FT", "ft_mic_dp": "Public-Pretrain + MIC-DP-FT",
         "public_only": "Public-Only", "randomk_per_record": "Random-k", "randomk_fixed": "Random-k-fixed"}
SIZE_TXT = {"p15": "15%", "p5": "5%"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def pm(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f} $\\pm$ {x.std(ddof=1):.3f}"


def pmd(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"


def md(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if not np.isfinite(v) else (f"{v:.3g}" if 0 < abs(v) < 1e-3 else floatfmt.format(v)))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def paired_family(tests: list[dict], fam: str, fam_index: int, cfg) -> pd.DataFrame:
    """tests: dicts with 'a', 'b' (aligned arrays over trials) and descriptive keys."""
    c3 = cfg["draft3"]
    rows = []
    for k, t in enumerate(tests):
        A, B = np.asarray(t.pop("a"), float), np.asarray(t.pop("b"), float)
        diff = A - B
        _, p = ST.wilcoxon_p(diff)
        lo, hi = ST.bootstrap_mean_ci(diff, np.random.default_rng([c3["bootstrap_seed"], fam_index, k]),
                                      c3["bootstrap_resamples"], c3["ci"])
        rows.append({"family": fam, **t, "mean_a": A.mean(), "mean_b": B.mean(), "diff": diff.mean(), "ci_lo": lo,
                     "ci_hi": hi, "n_a_wins": int((diff > 0).sum()), "n_b_wins": int((diff < 0).sum()), "p_raw": p})
    df = pd.DataFrame(rows)
    df["p_holm"] = multipletests(df.p_raw.to_numpy(), alpha=0.05, method="holm")[1]
    sig = df.p_holm < 0.05
    df["outcome"] = np.where(sig & (df["diff"] > 0), "A better", np.where(sig & (df["diff"] < 0), "B better", "n.s."))
    return df


def arr(df, **kw):
    s = df
    for k, v in kw.items():
        s = s[np.isclose(s[k], v)] if isinstance(v, float) else s[s[k] == v]
    s = s.sort_values("trial")
    assert len(s) == 30 and (s.trial.to_numpy() == np.arange(30)).all(), kw
    return s


# ----------------------------------------------------------------------------- main

def main():
    t0 = time.time()
    cfg = D.load_config()
    c3 = cfg["draft3"]
    out = D.ROOT / c3["out_dir"]
    eps_grid = c3["eps_grid"]
    feats = {d: D.prepare(d, cfg).features for d in DS}
    stored_all = pd.read_parquet(D.ROOT / cfg["outputs"]["dir"] / "raw_trials.parquet")
    stored_all = stored_all[stored_all.experiment == "main"]
    S = stored_all[stored_all.dataset.isin(DS) & (stored_all.model == "primary")]
    stats_csv = pd.read_csv(D.ROOT / cfg["outputs"]["dir"] / "stats.csv")
    pub = pd.read_parquet(D.ROOT / cfg["public_baselines"]["out_dir"] / "raw_trials_public.parquet")
    pub = pub[pub.model == "primary"]
    stats_pub = pd.read_csv(D.ROOT / cfg["public_baselines"]["out_dir"] / "stats_public.csv")
    R = pd.read_parquet(out / "raw_trials_draft3.parquet")
    ref = R[R.part == "refit"]
    sec, tex, fams = {}, {}, []

    # ============================================================== A.1 per-trial ROC-AUC at eps <= 0.5
    lo_b, hi_b = c3["part_a"]["a1_bins"]
    rows = []
    for d in DS:
        for e in c3["part_a"]["a1_eps"]:
            for m in c3["refit"]["methods"]:
                a = arr(S, dataset=d, eps=float(e), method=m).roc_auc.to_numpy()
                rows.append({"dataset": DS_TXT[d], "eps": e, "method": M_TXT[m], "n_below_0.45": int((a < lo_b).sum()),
                             "n_0.45_to_0.55": int(((a >= lo_b) & (a <= hi_b)).sum()), "n_above_0.55": int((a > hi_b).sum()),
                             "min": a.min(), "max": a.max()})
    a1 = pd.DataFrame(rows)
    sec["a1"] = md(a1, "{:.3f}")
    tex["tab_a1_trials.tex"] = ["% A.1: per-trial ROC-AUC (stored), primary model, 30 trials; columns: dataset & eps & method & "
                                "n < 0.45 & n in [0.45, 0.55] & n > 0.55 & min & max"]
    for i, r in a1.iterrows():
        tex["tab_a1_trials.tex"].append(f"{r.dataset if i % 9 == 0 else ''} & {r.eps:g} & {r.method} & {r['n_below_0.45']} & "
                                        f"{r['n_0.45_to_0.55']} & {r['n_above_0.55']} & {r['min']:.3f} & {r['max']:.3f} \\\\")
    # low trials: features and association signs
    low = ref[(ref.eps <= 0.5) & (ref.na_auc < c3["refit"]["low_trials"]["threshold"])].copy()
    lrows = []
    for r in low.sort_values(["dataset", "eps", "method", "trial"]).itertuples():
        g = np.asarray(r.split_gain)
        cnt = np.asarray(r.split_count)
        top = np.argsort(-g)[:3]
        share = g / g.sum() if g.sum() > 0 else g
        lrows.append({"dataset": DS_TXT[r.dataset], "eps": r.eps, "method": M_TXT[r.method], "trial": r.trial,
                      "AUC_test": r.na_auc,
                      "top features by gain (share, splits)": "; ".join(f"{feats[r.dataset][j]} ({share[j]:.2f}, {cnt[j]})" for j in top),
                      "sign test": "+" if r.na_auc > 0.5 else "-",
                      "AUC public (sign)": f"{r.auc_public_na:.3f} ({'+' if r.auc_public_na > 0.5 else '-'})",
                      "AUC train release (sign)": f"{r.auc_train_direct:.3f} ({'+' if r.auc_train_direct > 0.5 else '-'})"})
    lowdf = pd.DataFrame(lrows)
    # aggregate: how often each feature is the top-1 feature among the low trials
    agg = []
    for (d, m), g in low.groupby(["dataset", "method"]):
        top1 = [feats[d][int(np.argmax(np.asarray(x)))] for x in g.split_gain]
        vc = pd.Series(top1).value_counts()
        agg.append({"dataset": DS_TXT[d], "method": M_TXT[m], "low trials": len(g),
                    "top-1 feature (count)": "; ".join(f"{k} ({v})" for k, v in vc.items()),
                    "public AUC < 0.5": int((g.auc_public_na < 0.5).sum()),
                    "train-release AUC < 0.5": int((g.auc_train_direct < 0.5).sum())})
    sec["a1_low"] = md(lowdf, "{:.3f}") if len(lowdf) else "No trial below 0.45."
    sec["a1_agg"] = md(pd.DataFrame(agg)) if agg else ""
    # reference: same diagnostics for all trials at eps <= 0.5 (share of trials with public AUC < 0.5)
    allr = ref[ref.eps <= 0.5]
    sec["a1_ref"] = md(allr.groupby(["dataset", "eps", "method"]).apply(
        lambda g: pd.Series({"trials": len(g), "test AUC < 0.45": int((g.na_auc < 0.45).sum()),
                             "public AUC < 0.5": int((g.auc_public_na < 0.5).sum()),
                             "train-release AUC < 0.5": int((g.auc_train_direct < 0.5).sum()),
                             "corr(test AUC, public AUC)": float(np.corrcoef(g.na_auc, g.auc_public_na)[0, 1])}),
        include_groups=False).reset_index().assign(dataset=lambda x: x.dataset.map(DS_TXT), method=lambda x: x.method.map(M_TXT)), "{:.3f}")
    # figure
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.6), sharex=True, sharey=True)
    bw = c3["part_a"]["a1_hist_bin_width"]
    bins = np.arange(0.2, 1.0 + bw / 2, bw)
    for i, d in enumerate(DS):
        for j, e in enumerate(c3["part_a"]["a1_eps"]):
            ax = axes[i, j]
            ax.axvspan(lo_b, hi_b, color=GRAY, alpha=0.10, lw=0)
            ax.axvline(0.5, color=GRAY, ls="--", lw=0.8)
            for m, col in zip(c3["refit"]["methods"], (BLUE, ORANGE, GREEN)):
                a = arr(S, dataset=d, eps=float(e), method=m).roc_auc.to_numpy()
                ax.hist(np.clip(a, bins[0], bins[-1] - 1e-9), bins=bins, histtype="step", color=col, lw=1.3, label=M_TXT[m])
            ax.set_title(f"{DS_TXT[d]}, $\\varepsilon = {e:g}$")
            if i == 1:
                ax.set_xlabel("ROC-AUC (test)")
            if j == 0:
                ax.set_ylabel("trials")
    from matplotlib.ticker import MaxNLocator
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "auc_trials_ch4.pdf")
    fig.savefig(out / "auc_trials_ch4.png", dpi=200)
    plt.close(fig)

    # ============================================================== A.2 thresholds
    rows = []
    for d in DS:
        for e in eps_grid:
            for m in c3["refit"]["methods"]:
                g = arr(ref, dataset=d, eps=float(e), method=m)
                rows.append({"dataset": DS_TXT[d], "eps": e, "method": M_TXT[m], "acc@0.5": pmd(g.na_acc),
                             "BA@0.5": pmd(g.na_ba), "t* (median)": f"{np.median(g.t_star):.3f}",
                             "acc@t*": pmd(g.acc_at_t), "BA@t*": pmd(g.ba_at_t),
                             "BA public@t*": pmd(g.ba_public_at_t), "_g": g})
    a2 = pd.DataFrame(rows)
    sec["a2"] = md(a2.drop(columns="_g"))
    tex["tab_a2_threshold.tex"] = ["% A.2: primary model, noise-aware prediction, clean test set, mean $\\pm$ std over 30 trials (refits); "
                                   "columns: dataset & eps & method & accuracy at 0.5 & balanced accuracy at 0.5 & accuracy at t* & "
                                   "balanced accuracy at t*; t* maximizes balanced accuracy on the clean 15% public split"]
    for i, r in a2.iterrows():
        g = r._g
        tex["tab_a2_threshold.tex"].append(f"{r.dataset if i % 18 == 0 else ''} & {r.eps:g} & {r.method} & {pm(g.na_acc)} & "
                                           f"{pm(g.na_ba)} & {pm(g.acc_at_t)} & {pm(g.ba_at_t)} \\\\")

    # ============================================================== A.3 95% intervals
    rows = []
    for k, (d, e) in enumerate(c3["part_a"]["a3_cells"]):
        A = arr(S, dataset=d, eps=float(e), method="mic_dp").roc_auc.to_numpy()
        B = arr(S, dataset=d, eps=float(e), method="uniform").roc_auc.to_numpy()
        diff = A - B
        rng = np.random.default_rng([c3["bootstrap_seed"], 3, k])
        idx = rng.integers(0, 30, size=(c3["bootstrap_resamples"], 30))
        mb = diff[idx].mean(axis=1)
        st = stats_csv[(stats_csv.family == "F1") & (stats_csv.dataset == d) & np.isclose(stats_csv.eps, e)
                       & (stats_csv.method_b == "uniform")].iloc[0]
        _, p = ST.wilcoxon_p(diff)
        rows.append({"dataset": DS_TXT[d], "eps": e, "mean diff": diff.mean(),
                     "95% CI (new)": f"[{np.quantile(mb, 0.025):+.4f}, {np.quantile(mb, 0.975):+.4f}]",
                     "90% CI (same resamples)": f"[{np.quantile(mb, 0.05):+.4f}, {np.quantile(mb, 0.95):+.4f}]",
                     "stored F1 95% CI (stats.csv)": f"[{st.ci_lo:+.4f}, {st.ci_hi:+.4f}]",
                     "Wilcoxon p (raw)": p, "p Holm (F1)": st.p_holm,
                     "_lo": np.quantile(mb, 0.025), "_hi": np.quantile(mb, 0.975)})
    a3 = pd.DataFrame(rows)
    sec["a3"] = md(a3.drop(columns=["_lo", "_hi"]))
    tex["tab_a3_ci.tex"] = ["% A.3: MIC-DP - Uniform, ROC-AUC, primary model, 30 paired trials; columns: dataset & eps & mean "
                            "difference & 95% percentile bootstrap interval (10,000 resamples)"]
    for _, r in a3.iterrows():
        tex["tab_a3_ci.tex"].append(f"{r.dataset} & {r.eps:g} & {r['mean diff']:+.3f} & [{r._lo:+.3f}, {r._hi:+.3f}] \\\\")

    # ============================================================== A.4 F1 vs F1_secondary
    f1 = stats_csv[stats_csv.family == "F1"].reset_index(drop=True)
    f1s = stats_csv[stats_csv.family == "F1_secondary"].reset_index(drop=True)
    fams_spec = ST.family_specs(cfg)
    recomputed = ST.run_family(stored_all, "F1_secondary", fams_spec["F1"], cfg, "secondary", 99)
    same_cells = bool((f1[["dataset", "eps", "method_b"]].values == f1s[["dataset", "eps", "method_b"]].values).all())
    eq_out = int((f1.outcome.values == f1s.outcome.values).sum())
    eq_diff = int(np.isclose(f1.mean_diff_U.values, f1s.mean_diff_U.values, rtol=0, atol=1e-12).sum())
    rec_match = bool(np.allclose(recomputed.mean_diff_U.values, f1s.mean_diff_U.values, rtol=0, atol=1e-12)
                     and np.allclose(recomputed.p_holm.values, f1s.p_holm.values, rtol=0, atol=1e-12)
                     and (recomputed.outcome.values == f1s.outcome.values).all())
    sec_vals = stored_all[(stored_all.model == "secondary") & (stored_all.method == "mic_dp")]
    a4 = pd.DataFrame({"dataset": f1.dataset, "eps": f1.eps, "baseline": f1.method_b,
                       "diff primary": f1.mean_diff_U, "p_holm primary": f1.p_holm, "outcome F1": f1.outcome,
                       "diff secondary": f1s.mean_diff_U, "p_holm secondary": f1s.p_holm,
                       "outcome F1_secondary": f1s.outcome})
    sec["a4"] = md(a4)
    sec["a4_counts"] = (f"F1: {dict((k, int(v)) for k, v in f1.outcome.value_counts().items())}; F1_secondary: {dict((k, int(v)) for k, v in f1s.outcome.value_counts().items())}. "
                        f"Same 96 cells in the same order: {same_cells}. Cells with the same outcome: {eq_out} of 96. "
                        f"Cells with the same mean difference (to 1e-12): {eq_diff} of 96. F1_secondary recomputed from the "
                        f"secondary-model rows of raw_trials.parquet (src/stats.run_family, model = 'secondary') equals the stored "
                        f"F1_secondary in every mean difference, Holm p-value and outcome: {rec_match}. Secondary-model rows of "
                        f"MIC-DP in raw_trials.parquet: {len(sec_vals)}; model names: {sorted(sec_vals.model_name.unique().tolist())}.")
    a4["dataset_txt"] = a4.dataset.map(lambda x: cfg["datasets"][x]["display"])
    tex["tab_a4_f1.tex"] = ["% A.4: family F1 (MIC-DP vs baseline, Holm over 96) with the primary model and with the secondary "
                            "(linear) model; columns: dataset & eps & baseline & mean difference (primary) & outcome & "
                            "mean difference (secondary) & outcome; difference in U (ROC-AUC, or -MAE for regression)"]
    BN = {"uniform": "Uniform", "pearson_alloc": "Pearson-Alloc", "pearson_adj": "Pearson-Adj", "partialcorr_adj": "PartialCorr-Adj"}
    for _, r in a4.iterrows():
        tex["tab_a4_f1.tex"].append(f"{r.dataset_txt} & {r.eps:g} & {BN[r.baseline]} & {r['diff primary']:+.3f} & "
                                    f"{r['outcome F1']} & {r['diff secondary']:+.3f} & {r['outcome F1_secondary']} \\\\")

    # ============================================================== B. noise-aware vs direct
    tests, brows = [], []
    for d in DS:
        for m in c3["part_b"]["methods"]:
            for e in eps_grid:
                g = arr(ref, dataset=d, eps=float(e), method=m)
                tests.append({"dataset": d, "method": m, "eps": e, "comparison": "noise-aware - direct",
                              "a": g.na_auc.to_numpy(), "b": g.direct_auc.to_numpy()})
                brows.append({"dataset": d, "method": m, "eps": e, "na": pmd(g.na_auc), "direct": pmd(g.direct_auc),
                              "const na": int((g.na_n_unique == 1).sum()), "const direct": int((g.direct_n_unique == 1).sum()),
                              "_na": g.na_auc.to_numpy(), "_dir": g.direct_auc.to_numpy()})
    fb = paired_family(tests, "B_noise_aware", c3["part_b"]["family_index"], cfg)
    fams.append(fb)
    bt = pd.DataFrame(brows).merge(fb[["dataset", "method", "eps", "diff", "ci_lo", "ci_hi", "p_raw", "p_holm", "outcome"]],
                                   on=["dataset", "method", "eps"])
    uni = fb[fb.method == "uniform"]
    claim = bool((uni.outcome == "A better").all() and len(uni) == 12)
    sec["b"] = md(bt.drop(columns=["_na", "_dir"]).assign(dataset=lambda x: x.dataset.map(DS_TXT), method=lambda x: x.method.map(M_TXT))
                  .rename(columns={"na": "AUC noise-aware", "direct": "AUC direct", "diff": "mean diff (na - direct)"}))
    sec["b_claim"] = (f"Claim \"noise-aware prediction improves ROC-AUC for Uniform at every budget on both datasets\": "
                      f"{'holds' if claim else 'does not hold'}. Uniform cells with noise-aware significantly better: "
                      f"{int((uni.outcome == 'A better').sum())} of 12; significantly worse: {int((uni.outcome == 'B better').sum())}; "
                      f"n.s.: {int((uni.outcome == 'n.s.').sum())}. MIC-DP: {int((fb[fb.method == 'mic_dp'].outcome == 'A better').sum())} "
                      f"better, {int((fb[fb.method == 'mic_dp'].outcome == 'B better').sum())} worse, "
                      f"{int((fb[fb.method == 'mic_dp'].outcome == 'n.s.').sum())} n.s. of 12.")
    tex["tab_b_noiseaware.tex"] = ["% B: primary model, ROC-AUC on the clean test set, mean $\\pm$ std over 30 trials; columns: dataset & "
                                   "method & eps & direct & noise-aware & mean difference (noise-aware - direct) & 95% bootstrap "
                                   "interval & Holm p (24 tests) & constant predictions direct & constant predictions noise-aware"]
    for i, r in bt.iterrows():
        tex["tab_b_noiseaware.tex"].append(
            f"{DS_TXT[r.dataset] if i % 12 == 0 else ''} & {M_TXT[r.method] if i % 6 == 0 else ''} & {r.eps:g} & {pm(r._dir)} & "
            f"{pm(r._na)} & {r['diff']:+.3f} & [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {r.p_holm:.3g} & {r['const direct']} & {r['const na']} \\\\")

    # ============================================================== C. repaired fine-tuning
    ft = R[R.part == "ft_repaired"]
    st_f7 = stats_pub[stats_pub.family == "F7"]
    crow, cfam = [], {}
    for d in DS:
        tests = []
        for m in c3["part_c"]["methods"]:
            parent = cfg["public_baselines"]["methods"][m]["release"]
            for size in c3["part_c"]["sizes"]:
                for e in eps_grid:
                    g = arr(ft, dataset=d, eps=float(e), method=m, size=size)
                    old = arr(pub, dataset=d, eps=float(e), method=m, public_size=size)
                    po = arr(pub[pub.method == "public_only"], dataset=d, public_size=size)
                    par = arr(S, dataset=d, eps=float(e), method=parent)
                    tests.append({"dataset": d, "method": m, "size": size, "eps": e, "vs": "Public-Only",
                                  "a": g.na_auc.to_numpy(), "b": po.roc_auc.to_numpy()})
                    tests.append({"dataset": d, "method": m, "size": size, "eps": e, "vs": "parent",
                                  "a": g.na_auc.to_numpy(), "b": par.roc_auc.to_numpy()})
                    so = st_f7[(st_f7.dataset == d) & (st_f7.method == m) & (st_f7["size"] == size) & np.isclose(st_f7.eps, e)]
                    crow.append({"dataset": d, "method": m, "size": size, "eps": e,
                                 "AUC repaired": pmd(g.na_auc), "const repaired": int((g.na_n_unique == 1).sum()),
                                 "max leaf (max over trials)": g.max_abs_leaf_continued.max(),
                                 "max leaf (median)": g.max_abs_leaf_continued.median(),
                                 "AUC stored": pmd(old.roc_auc), "const stored": int((old.n_unique_pred == 1).sum()),
                                 "stored F7 vs Public-Only": so[so.parent == "Public-Only"].outcome.iloc[0],
                                 "stored F7 vs parent": so[so.parent == "private"].outcome.iloc[0],
                                 "AUC direct repaired": pmd(g.direct_auc), "frac start clipped": g.start_frac_clipped.mean(),
                                 "_g": g, "_old": old})
        fam = paired_family(tests, f"F7r_{d}", c3["part_c"]["family_index"][d], cfg)
        cfam[d] = fam
        fams.append(fam)
    ct = pd.DataFrame(crow)
    for d in DS:
        f = cfam[d]
        for vs, col in (("Public-Only", "repaired F7r vs Public-Only"), ("parent", "repaired F7r vs parent")):
            sub = f[f.vs == vs].set_index(["method", "size", "eps"]).outcome
            ct.loc[ct.dataset == d, col] = [sub.loc[(r.method, r.size, r.eps)] for r in ct[ct.dataset == d].itertuples()]
    for d in DS:
        sec[f"c_{d}"] = md(ct[ct.dataset == d].drop(columns=["_g", "_old", "dataset"])
                           .assign(method=lambda x: x.method.map(M_TXT), size=lambda x: x["size"].map(SIZE_TXT)), "{:.3f}")
        sec[f"c_tests_{d}"] = md(cfam[d].assign(method=lambda x: x.method.map(M_TXT), size=lambda x: x["size"].map(SIZE_TXT))
                                 .drop(columns=["family", "dataset"]))
        tex[f"tab_c_ft_{d}.tex"] = [f"% C ({DS_TXT[d]}): Public-Pretrain + FT, repaired (start clipped to [-3, 3], l2 = 1 for the continued "
                                    "trees) and stored, primary model, noise-aware ROC-AUC, mean $\\pm$ std over 30 trials; columns: method & "
                                    "public size & eps & repaired & constant (repaired) & max |leaf| (continued trees, max over trials) & "
                                    "F7r vs Public-Only & F7r vs parent & stored & constant (stored)"]
        sub = ct[ct.dataset == d].reset_index(drop=True)
        for i, r in sub.iterrows():
            tex[f"tab_c_ft_{d}.tex"].append(
                f"{M_TXT[r.method] if i % 12 == 0 else ''} & {SIZE_TXT[r['size']] if i % 6 == 0 else ''} & {r.eps:g} & "
                f"{pm(r._g.na_auc)} & {r['const repaired']} & {r['max leaf (max over trials)']:.3g} & "
                f"{r['repaired F7r vs Public-Only']} & {r['repaired F7r vs parent']} & {pm(r._old.roc_auc)} & {r['const stored']} \\\\")

    dpath = out / "diagnostic_ft_clip.csv"
    if dpath.exists():
        dg = pd.read_csv(dpath)
        sec["c_diag"] = md(dg.assign(dataset=lambda x: x.dataset.map(DS_TXT), method=lambda x: x.method.map(M_TXT),
                                     size=lambda x: x["size"].map(SIZE_TXT)), "{:.3f}")

    # ============================================================== D. Random-k
    rk = R[R.part == "randomk"]
    tests, drow = [], []
    for d in DS:
        for e in eps_grid:
            ref_s = {m: arr(S, dataset=d, eps=float(e), method=m).roc_auc.to_numpy() for m in ("uniform", "mic_dp", "mic_public")}
            row = {"dataset": d, "eps": e, **{M_TXT[m]: pmd(v) for m, v in ref_s.items()}, "_ref": ref_s}
            for mode in ("per_record", "fixed"):
                for k in c3["part_d"]["k"][d]:
                    g = arr(rk, dataset=d, eps=float(e), method=f"randomk_{mode}", k=float(k))
                    lab = f"{M_TXT['randomk_' + mode]} k={k}"
                    row[lab] = pmd(g.na_auc)
                    row[f"_{mode}_{k}"] = g.na_auc.to_numpy()
                    row[f"const {lab}"] = str(int((g.na_n_unique == 1).sum()))
                    for vs in ("uniform", "mic_dp"):
                        tests.append({"dataset": d, "variant": mode, "k": k, "eps": e, "vs": M_TXT[vs],
                                      "a": g.na_auc.to_numpy(), "b": ref_s[vs]})
            drow.append(row)
    f9 = paired_family(tests, "F9", c3["part_d"]["family_index"], cfg)
    fams.append(f9)
    dt = pd.DataFrame(drow)
    for d in DS:
        sub = dt[dt.dataset == d]
        sec[f"d_{d}"] = md(sub[[c for c in sub.columns if not c.startswith("_") and c != "dataset" and sub[c].notna().all()]])
    sec["d_tests"] = md(f9.assign(dataset=lambda x: x.dataset.map(DS_TXT)).drop(columns="family"))
    sec["d_counts"] = "; ".join(f"vs {a}: {b} {int(n)}" for (a, b), n in f9.groupby(["vs", "outcome"]).size().items())
    for d in DS:
        ks = c3["part_d"]["k"][d]
        tex[f"tab_d_randomk_{d}.tex"] = [
            f"% D ({DS_TXT[d]}): primary model, noise-aware ROC-AUC, mean $\\pm$ std over 30 trials; columns: eps & Uniform & MIC-DP & "
            f"MIC-Public (15%) & Random-k k={ks[0]} & Random-k k={ks[1]} & Random-k-fixed k={ks[0]} & Random-k-fixed k={ks[1]}; "
            "F9 outcomes (vs Uniform, vs MIC-DP) in tab_d_f9.tex"]
        for _, r in dt[dt.dataset == d].iterrows():
            cells = [pm(r["_ref"][m]) for m in ("uniform", "mic_dp", "mic_public")]
            cells += [pm(r[f"_{mode}_{k}"]) for mode in ("per_record", "fixed") for k in ks]
            tex[f"tab_d_randomk_{d}.tex"].append(f"{r.eps:g} & " + " & ".join(cells) + " \\\\")
    tex["tab_d_f9.tex"] = ["% D: family F9 (96 tests, Holm): Random-k variants minus Uniform and minus MIC-DP, ROC-AUC, primary model; "
                           "columns: dataset & variant & k & eps & vs & mean difference & 95% bootstrap interval & Holm p & outcome"]
    for i, r in f9.iterrows():
        tex["tab_d_f9.tex"].append(f"{DS_TXT[r.dataset]} & {M_TXT['randomk_' + r.variant]} & {r.k} & {r.eps:g} & {r.vs} & "
                                   f"{r['diff']:+.3f} & [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {r.p_holm:.3g} & {r.outcome} \\\\")

    # ============================================================== outputs
    allf = pd.concat(fams, ignore_index=True)
    a3o = a3.drop(columns=["_lo", "_hi"]).assign(family="A3_intervals")
    pd.concat([allf, a3o], ignore_index=True).to_csv(out / "stats_draft3.csv", index=False)
    for fn, lines in tex.items():
        (out / fn).write_text("\n".join(lines) + "\n")
    write_summary(cfg, out, sec, allf, time.time() - t0)


def git(*a):
    return subprocess.run(["git", *a], cwd=D.ROOT, capture_output=True, text=True).stdout.strip()


def write_summary(cfg, out, sec, allf, t_report):
    notes = (D.ROOT / "draft3_notes.md").read_text()
    sanity = (out / "sanity_draft3.log").read_text()
    sanity = "\n".join(l for l in sanity.splitlines() if l.startswith("[") or l.startswith("=="))
    runtime = (out / "runtime_draft3.txt").read_text().strip()
    c1, c2 = git("rev-parse", "8b2e018"), git("rev-parse", "6209f42")
    d1, d2 = git("show", "-s", "--format=%ci", c1), git("show", "-s", "--format=%ci", c2)
    fam_counts = allf.groupby(["family", "outcome"]).size().unstack(fill_value=0)
    L = [
        "# Chapter 4 (MIC-DP): Draft 3 follow-up runs",
        "",
        "Produced by `python -m src.draft3 && python -m src.draft3_report`. Numbers are reported as they came out. Datasets: Adult "
        "and CDC-Diabetes; eps in {0.1, 0.3, 0.5, 1, 5, 10}; delta = 1e-5; 30 paired trials (seed 1000 + t); primary model "
        "HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=seed), as in the main run. No stored "
        "result, split, seed or hyperparameter was changed.",
        "",
        "## Commits",
        "",
        f"- Configuration and notes of Parts A to D, before any part ran: `{c1}` ({d1}).",
        f"- Code and tests, and the amendment of the Part C noise-aware prediction (after a one-unit smoke run, before the main run): `{c2}` ({d2}).",
        "- The main run started after the second commit (`results/ch4/draft3/run.log`).",
        "",
        "## Notes fixed before each part (draft3_notes.md, verbatim)",
        "",
        notes.replace("\n# ", "\n### ").replace("\n## ", "\n### "),
        "",
        "## Sanity checks",
        "",
        "```", sanity.strip(), "```",
        "",
        "## Part A. Analyses of stored results",
        "",
        "### A.1 Per-trial ROC-AUC at eps <= 0.5 (stored values; figure auc_trials_ch4.pdf)",
        "",
        sec["a1"], "",
        "Trials below 0.45 (refit models, which reproduce the stored ROC-AUC exactly): top 3 features by total split gain "
        "(gain share, number of splits), sign of the association of the noise-aware score with the label on the test set, "
        "ROC-AUC (sign) on the clean public split, and of the direct prediction on the privatized training rows:", "",
        sec["a1_low"], "",
        "Top-1 feature among the trials below 0.45:", "", sec["a1_agg"], "",
        "Same diagnostics over all 30 trials of each cell (for reference):", "", sec["a1_ref"], "",
        "### A.2 Accuracy at a sensible threshold",
        "",
        "Predictions are not stored in raw_trials.parquet (it holds ROC-AUC, accuracy at 0.5 and the number of distinct predictions), "
        "so the primary models were refit on the stored releases with the stored seeds and streams; every refit reproduces the "
        "stored noise-aware ROC-AUC, accuracy and number of distinct predictions exactly (sanity check refit_reproduces_stored). "
        "acc@0.5 and BA@0.5: noise-aware prediction at threshold 0.5 on the clean test set; t*: the threshold that maximizes "
        "balanced accuracy of the noise-aware prediction on the clean 15% public split; acc@t* and BA@t*: on the test set.", "",
        sec["a2"], "",
        "### A.3 95% intervals of MIC-DP minus Uniform (ROC-AUC, primary model)",
        "",
        sec["a3"], "",
        "### A.4 Family F1 with the primary and with the linear model",
        "",
        sec["a4_counts"], "",
        sec["a4"], "",
        "## Part B. Noise-aware prediction, with and without",
        "",
        "Refit models of the stored Uniform and MIC-DP releases; direct = predict_proba on the clean test inputs; difference = "
        "noise-aware minus direct per trial; 95% percentile bootstrap interval (10,000 resamples); paired two-sided Wilcoxon, Holm "
        "over the 24 tests. const = number of trials (of 30) with a constant test prediction.", "",
        sec["b"], "",
        sec["b_claim"], "",
        "## Part C. Fine-tuning, repaired",
        "",
        "Repaired = start of the continued stage clipped to [-3, 3] and l2_regularization = 1.0 for trees 101-200; noise-aware "
        "prediction clip(E[F_pre], -3, 3) + E[F_cont] (draft3_notes.md, amendment). Stored = Public-Pretrain + FT of the public-"
        "baselines run (raw_trials_public.parquet), with its F7 outcomes (Holm over 144 tests). F7r = the new tests of the repaired "
        "models (Holm within the 48 tests of each dataset); A = repaired model, B = Public-Only at the same size or the private "
        "parent (stored Uniform for Uniform-FT, stored MIC-DP for MIC-DP-FT). Max leaf = largest |leaf value| of the continued trees "
        "(101-200). frac start clipped = share of privatized training rows whose pretrained raw score lay outside [-3, 3] (mean over "
        "trials). AUC direct repaired: clip(F_pre(x), -3, 3) + F_cont(x) on the clean test inputs, for reference.", "",
        "### Adult", "", sec["c_adult"], "", "F7r tests (Adult):", "", sec["c_tests_adult"], "",
        "### CDC-Diabetes (check)", "", sec["c_cdc_diabetes"], "", "F7r tests (CDC-Diabetes):", "", sec["c_tests_cdc_diabetes"], "",
        "### Diagnostic of the noise-aware prediction (exploratory, trial 0; diagnostics/draft3_ft_clip.py)", "",
        "Range of the exact E[F_pre(x + eta)] (pretrained trees with the baseline) over the test records, share of test records "
        "with |E[F_pre]| > 3 (where the clip of the noise-aware prediction is active), and ROC-AUC of E[F_pre], of E[F_cont] "
        "(continued trees), of the reported noise-aware prediction clip(E[F_pre], -3, 3) + E[F_cont], and of E[F_pre] + E[F_cont] "
        "without the clip. Same code, seeds and models as the run; nothing was changed.", "",
        sec.get("c_diag", "(not run)"), "",
        "## Part D. Attribute-sampling baseline",
        "",
        "Random-k: per record, a uniform random k-subset released with noise sqrt(k) c(eps, delta) per coordinate, the other "
        "coordinates set to the public-split mean; Random-k-fixed: one random k-subset for all records. Noise-aware prediction as "
        "in draft3_notes.md. Uniform, MIC-DP and MIC-Public: stored values. const = trials with a constant test prediction.", "",
        "Why Random-k satisfies (eps, delta)-DP for the record: the subset of each record is drawn independently of the data, and "
        "given the subset the release of that record is a Gaussian mechanism on k coordinates of range one (L2 sensitivity sqrt(k)) "
        "with noise sqrt(k) c(eps, delta), which is (eps, delta)-DP by the analytic Gaussian calibration; the mean fill uses public "
        "data only, the other records do not depend on the changed record, and a mixture over a data-independent subset of "
        "(eps, delta)-DP mechanisms is (eps, delta)-DP. Random-k-fixed is the case of one subset for all records.", "",
        "### Adult", "", sec["d_adult"], "", "### CDC-Diabetes", "", sec["d_cdc_diabetes"], "",
        f"Family F9 (96 tests, Holm), outcome counts by comparison: {sec['d_counts']}", "",
        sec["d_tests"], "",
        "## Outcome counts of the new families", "",
        md(fam_counts.reset_index()), "",
        "## Runtime", "",
        runtime + f"; python -m src.draft3_report: {t_report/60:.1f} min. One smoke unit before the main run (Adult, trial 0): 5.0 min.",
        "",
        "## Deviations", "",
        "D1 (Part C, before the main run). The noise-aware prediction of the repaired fine-tuned model fixed in commit 8b2e018 "
        "estimated E[clip(F_pre(x + eta))] with a Monte-Carlo term of 256 draws per test record. On the smoke unit (Adult, trial 0, "
        "Uniform release, eps = 0.3, 15% split) the spread of the exact E[F_pre] across test records was a standard deviation of "
        "0.012 in logit, below the Monte-Carlo error per record, and the estimate gave ROC-AUC 0.503 against 0.833 for "
        "clip(E[F_pre]) + E[F_cont]. The noise-aware prediction was changed to clip(E[F_pre(x + eta)], -3, 3) + E[F_cont(x + eta)], "
        "both exact, and recorded in draft3_notes.md and config.yaml (commit 6209f42) before the main run. The repair itself and "
        "every other setting are as fixed in 8b2e018. The smoke outputs are not part of the results.",
        "",
        "D2 (presentation only, after the run). diagnostics/draft3_ft_clip.py was added and run on trial 0 (output "
        "diagnostic_ft_clip.csv, shown under Part C) to show the range of E[F_pre] and where the clip of the noise-aware prediction "
        "is active. statsmodels (used by src/stats.py for Holm) was missing from the restored environment and was installed "
        "(0.15.0) before the report ran. No reported number of a method changed.",
        "",
        "Files in this folder: raw_trials_draft3.parquet (one row per refit, repaired FT model and Random-k model), stats_draft3.csv "
        "(all new tests: B_noise_aware, F7r_adult, F7r_cdc_diabetes, F9, and the A.3 intervals), config_used.yaml, tab_*.tex (rows "
        "only), auc_trials_ch4.pdf/png, sanity_checks_draft3.csv, sanity_draft3.log, runtime_draft3.txt, run.log.",
    ]
    (out / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
