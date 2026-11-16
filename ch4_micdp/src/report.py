"""LaTeX tables, figures and summary.md for Chapter 4.

    python -m src.report
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from . import data as D
from . import dp
from . import stats as S

# ----------------------------------------------------------------------------- style
# Categorical palette (validated: adjacent CVD dE >= 9.1, normal-vision dE >= 19.6 on white).
# Every method keeps the same colour, marker and dash pattern in every figure; markers and
# dashes are the secondary encoding so identity never rests on colour alone.
STYLE = {
    "nonprivate":      dict(color="#0b0b0b", marker=None, ls=(0, (5, 3))),
    "uniform":         dict(color="#6b6a66", marker="o", ls="-"),
    "mic_dp":          dict(color="#2a78d6", marker="s", ls="-"),
    "mic_public":      dict(color="#eb6834", marker="^", ls="--"),
    "pearson_alloc":   dict(color="#1baf7a", marker="D", ls="-."),
    "pearson_adj":     dict(color="#eda100", marker="v", ls=(0, (1, 1.5))),
    "partialcorr_adj": dict(color="#e87ba4", marker="P", ls=(0, (3, 1, 1, 1, 1, 1))),
    "random_alloc":    dict(color="#008300", marker="X", ls=":"),
    "mic_oracle":      dict(color="#4a3aa7", marker="*", ls="--"),
    "mic_dp_override": dict(color="#e34948", marker="h", ls="-"),
}
EPS_RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # sequential blue
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

MAIN_COLS = ["uniform", "mic_dp", "mic_public", "pearson_alloc", "pearson_adj", "partialcorr_adj"]
ABL_COLS = ["mic_dp", "random_alloc", "mic_oracle", "mic_public"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "lines.markersize": 4.5, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def disp(cfg, m):
    return cfg["methods"]["list"][m]["display"]


def dname(cfg, d):
    return cfg["datasets"][d]["display"]


def metric_name(task):
    return "ROC-AUC" if task == "classification" else "MAE"


def better(task):
    return np.argmax if task == "classification" else np.argmin


def fmt(m, s, nd=4):
    return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"


def eps_tex(e):
    return f"{e:g}"


def savefig(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- tables

def main_table(summ, tests, cfg, ds, model, family, out, captions):
    task = cfg["datasets"][ds]["task"]
    sub = summ[summ.dataset == ds]
    npv = sub[sub.method == "nonprivate"].iloc[0]
    for m in MAIN_COLS:
        assert cfg["methods"]["list"][m]["dp"], "non-DP method in a DP comparison table"
    lines = [f"% {dname(cfg, ds)} | {'primary (HistGradientBoosting)' if model == 'primary' else 'secondary (linear)'} model | "
             f"{metric_name(task)} ({'higher' if task == 'classification' else 'lower'} is better), mean $\\pm$ std over "
             f"{cfg['protocol']['n_trials']} trials",
             f"% columns: eps & " + " & ".join(disp(cfg, m) for m in MAIN_COLS),
             f"% caption note: Non-private {metric_name(task)} = {fmt(npv.metric_mean, npv.metric_std)}",
             "% bold = best DP method in the row; $^\\dagger$ = MIC-DP significantly better than all four "
             "baselines (Uniform, Pearson-Alloc, Pearson-Adj, PartialCorr-Adj), paired Wilcoxon, Holm-adjusted p < 0.05"
             + ("" if family else " [no daggers: this dataset is not in the F1 family]")]
    for e in cfg["privacy"]["eps_grid"]:
        row = sub[sub.eps == e].set_index("method")
        means = np.array([row.loc[m, "metric_mean"] for m in MAIN_COLS])
        stds = np.array([row.loc[m, "metric_std"] for m in MAIN_COLS])
        best = means[better(task)(means)]
        cells = []
        dagger = False
        if family is not None:
            t = tests[(tests.family == family) & (tests.dataset == ds) & (tests.eps == e)]
            dagger = len(t) == 4 and bool((t.outcome == "A better").all())
        for m, mu, sd in zip(MAIN_COLS, means, stds):
            c = fmt(mu, sd)
            if round(mu, 4) == round(best, 4):
                c = f"\\textbf{{{mu:.4f}}} $\\pm$ {sd:.4f}"
            if m == "mic_dp" and dagger:
                c += "$^\\dagger$"
            cells.append(c)
        lines.append(f"{eps_tex(e)} & " + " & ".join(cells) + " \\\\")
    stem = f"tab_{'main' if model == 'primary' else 'secondary'}_{ds}"
    (out / f"{stem}.tex").write_text("\n".join(lines) + "\n")
    key = "".join(ch for ch in dname(cfg, ds).title() if ch.isalpha())
    captions.append(f"\\newcommand{{\\NonPrivate{'Main' if model == 'primary' else 'Secondary'}{key}}}"
                    f"{{{npv.metric_mean:.4f} $\\pm$ {npv.metric_std:.4f}}}  % {dname(cfg, ds)}, {metric_name(task)}")


def distortion_table(summ, cfg, out):
    lines = ["% Feature distortion: mean |X_tilde - X| over all private-train entries (features scaled to [0,1]); "
             "mean $\\pm$ std over trials",
             "% columns: dataset & eps & " + " & ".join(disp(cfg, m) for m in MAIN_COLS)]
    for ds in D.all_dataset_names(cfg):
        for i, e in enumerate(cfg["privacy"]["eps_grid"]):
            row = summ[(summ.dataset == ds) & (summ.eps == e)].set_index("method")
            cells = [fmt(row.loc[m, "dist_mean"], row.loc[m, "dist_std"]) for m in MAIN_COLS]
            lab = dname(cfg, ds) if i == 0 else ""
            lines.append(f"{lab} & {eps_tex(e)} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    (out / "tab_distortion.tex").write_text("\n".join(lines) + "\n")


def ablation_table(summ, cfg, out):
    lines = ["% Ablation (primary model, primary metric, mean $\\pm$ std). MIC-Oracle uses raw private-train MIC "
             "WITHOUT privatization: it is NOT differentially private and appears only in this table.",
             "% columns: dataset & eps & " + " & ".join(disp(cfg, m) for m in ABL_COLS)]
    for ds in D.all_dataset_names(cfg):
        task = cfg["datasets"][ds]["task"]
        for i, e in enumerate(cfg["privacy"]["eps_grid"]):
            row = summ[(summ.dataset == ds) & (summ.eps == e)].set_index("method")
            cells = [fmt(row.loc[m, "metric_mean"], row.loc[m, "metric_std"]) for m in ABL_COLS]
            lab = f"{dname(cfg, ds)} ({metric_name(task)})" if i == 0 else ""
            lines.append(f"{lab} & {eps_tex(e)} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    (out / "tab_ablation.tex").write_text("\n".join(lines[:-1]) + "\n")


def synthetic_table(raw, ur, cfg, out):
    variants = [k for k in cfg["synthetic"]["variants"]]
    meths = ["mic_dp", "mic_public", "pearson_alloc", "pearson_adj", "partialcorr_adj", "random_alloc"]
    lines = ["% Utility recovered UR = (U_m - U_Uniform) / (U_NonPrivate - U_Uniform) from trial means, primary model, "
             "U = -MAE. Value $\\pm$ bootstrap standard error over trial seeds (UR is a ratio of means, so there is "
             "no per-trial std). UR > 0: better than Uniform; UR = 1: non-private utility",
             "% columns: method & eps & " + " & ".join(dname(cfg, v) for v in variants)]
    for m in meths:
        for i, e in enumerate(cfg["privacy"]["eps_grid"]):
            cells = []
            for v in variants:
                r = ur[(ur.dataset == v) & (ur.eps == e) & (ur.method == m)].iloc[0]
                cells.append(fmt(r.UR, S.ur_bootstrap_se(raw, cfg, v, e, m)))
            lines.append(f"{disp(cfg, m) if i == 0 else ''} & {eps_tex(e)} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    (out / "tab_synthetic.tex").write_text("\n".join(lines[:-1]) + "\n")


def override_table(raw, cfg, out):
    sub = raw[(raw.experiment == "main") & (raw.model == "primary") & (raw.dataset == "adult")]
    feats = list(cfg["datasets"]["adult"]["features"])
    prot = cfg["methods"]["list"]["mic_dp_override"]["protected"]
    idx = [feats.index(f) for f in prot]
    lines = ["% Adult, demographic override (race, sex, age capped at the uniform share 1/d). Primary model ROC-AUC and "
             "per-feature noise std sigma_j (features scaled to [0,1]); mean $\\pm$ std over trials",
             "% columns: eps & method & ROC-AUC & " + " & ".join(f"$\\sigma_{{\\mathrm{{{f}}}}}$" for f in prot)]
    for e in cfg["privacy"]["eps_grid"]:
        for i, m in enumerate(["uniform", "mic_dp", "mic_dp_override"]):
            g = sub[(sub.method == m) & (sub.eps == e)]
            sig = np.vstack(g.sigma.to_numpy())[:, idx]
            cells = [fmt(g.primary_value.mean(), g.primary_value.std())]
            for k in range(len(idx)):
                col = sig[:, k]
                fin = col[np.isfinite(col)]
                c = fmt(fin.mean(), fin.std(ddof=1) if fin.size > 1 else 0.0, 2) if fin.size else "not released"
                if 0 < fin.size < col.size:
                    c += f" ({100 * (1 - fin.size / col.size):.0f}\\% not released)"
                cells.append(c)
            lines.append(f"{eps_tex(e) if i == 0 else ''} & {disp(cfg, m)} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    (out / "tab_override.tex").write_text("\n".join(lines[:-1]) + "\n")


def sweep_table(raw, summ, cfg, exp, out):
    sc = cfg["protocol"]["sensitivity"]
    grid = sc["alpha_grid"] if exp == "alpha" else sc["rho_grid"]
    fixed = (f"rho = {cfg['methods']['defaults']['rho']}" if exp == "alpha"
             else f"alpha = {cfg['methods']['defaults']['alpha']:g}")
    sub = raw[raw.experiment == exp]
    sym = "\\alpha" if exp == "alpha" else "\\rho"
    lines = [f"% Sensitivity of MIC-DP to {exp} ({fixed}); primary model, primary metric, mean $\\pm$ std over trials. "
             "Uniform (main run, same trial seeds) shown for reference",
             "% columns: dataset & eps & " + " & ".join(f"${sym}={g:g}$" for g in grid) + " & Uniform"]
    for ds in sc["datasets"]:
        task = cfg["datasets"][ds]["task"]
        for i, e in enumerate(sc["eps"]):
            cells = []
            for g in grid:
                v = sub[(sub.dataset == ds) & (sub.eps == e) & (np.isclose(sub.param, g))].primary_value
                cells.append(fmt(v.mean(), v.std()))
            u = summ[(summ.dataset == ds) & (summ.eps == e) & (summ.method == "uniform")].iloc[0]
            cells.append(fmt(u.metric_mean, u.metric_std))
            lab = f"{dname(cfg, ds)} ({metric_name(task)})" if i == 0 else ""
            lines.append(f"{lab} & {eps_tex(e)} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    (out / f"tab_{exp}.tex").write_text("\n".join(lines[:-1]) + "\n")


# ----------------------------------------------------------------------------- figures

def dual_figure(summ, cfg, ds, out):
    task = cfg["datasets"][ds]["task"]
    eps = np.array(cfg["privacy"]["eps_grid"])
    sub = summ[summ.dataset == ds]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for m in MAIN_COLS:
        st = STYLE[m]
        g = sub[sub.method == m].set_index("eps").loc[eps]
        a1.plot(eps, g.dist_mean, color=st["color"], marker=st["marker"], ls=st["ls"], label=disp(cfg, m))
        a2.plot(eps, g.metric_mean, color=st["color"], marker=st["marker"], ls=st["ls"], label=disp(cfg, m))
        a2.fill_between(eps, g.metric_mean - g.metric_std, g.metric_mean + g.metric_std, color=st["color"],
                        alpha=0.12, lw=0)
    npv = sub[sub.method == "nonprivate"].iloc[0]
    a2.axhline(npv.metric_mean, color=STYLE["nonprivate"]["color"], ls=STYLE["nonprivate"]["ls"], lw=1.2,
               label="Non-private")
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xticks(eps)
        a.set_xticklabels([f"{e:g}" for e in eps])
        a.minorticks_off()
        a.set_xlabel(r"privacy budget $\varepsilon$ ($\delta=10^{-5}$)")
    a1.set_yscale("log")
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a1.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if str(f"{v:g}")[0] in "25" else ""))
    a1.set_ylabel(r"feature distortion  mean $|\tilde X - X|$")
    a2.set_ylabel(f"test {metric_name(task)} ({'higher' if task == 'classification' else 'lower'} is better)")
    a1.set_title("Feature distortion", loc="left")
    a2.set_title(f"Downstream utility (HistGradientBoosting)", loc="left")
    h, l = a2.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=7.5,
               handlelength=2.6, columnspacing=1.0)
    fig.suptitle(dname(cfg, ds), x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, out, f"dual_{ds}")


def dose_figure(raw, cfg, out, h4):
    variants = [k for k, v in cfg["synthetic"]["variants"].items() if v["signal"] == "nonlinear"]
    lam = np.array([cfg["synthetic"]["variants"][k]["lambda"] for k in variants])
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    offs = np.linspace(-0.05, 0.05, len(cfg["privacy"]["eps_grid"]))
    for ei, e in enumerate(cfg["privacy"]["eps_grid"]):
        row = h4[h4.eps == e].iloc[0]
        mu = np.array([row[f"UR_{v}"] for v in variants])
        lo = np.array([row[f"UR_{v}_ci_lo"] for v in variants])
        hi = np.array([row[f"UR_{v}_ci_hi"] for v in variants])
        x = lam * np.exp(offs[ei])
        ax.errorbar(x, mu, yerr=[mu - lo, hi - mu], color=EPS_RAMP[ei], marker="os^DvP"[ei], capsize=2,
                    label=fr"$\varepsilon$={e:g} (slope {row.slope:+.3f})")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(lam)
    ax.set_xticklabels([f"{x:g}" for x in lam])
    ax.minorticks_off()
    ax.set_xlabel(r"signal strength $\lambda$ (nl-low, nl-mid, nl-high)")
    ax.set_ylabel("UR of MIC-DP (0 = Uniform, 1 = non-private)")
    ax.set_title("Synthetic dose-response (UR, 95% bootstrap CI)", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    savefig(fig, out, "synthetic_dose")


def sweep_figure(raw, summ, cfg, exp, out):
    sc = cfg["protocol"]["sensitivity"]
    grid = np.array(sc["alpha_grid"] if exp == "alpha" else sc["rho_grid"], dtype=float)
    sub = raw[raw.experiment == exp]
    n = len(sc["datasets"])
    fig, axes = plt.subplots(1, n, figsize=(2.15 * n, 2.6))
    eps_cols = {sc["eps"][0]: EPS_RAMP[2], sc["eps"][1]: EPS_RAMP[5]}
    for ax, ds in zip(axes, sc["datasets"]):
        task = cfg["datasets"][ds]["task"]
        for k, e in enumerate(sc["eps"]):
            mu = [sub[(sub.dataset == ds) & (sub.eps == e) & np.isclose(sub.param, g)].primary_value.mean() for g in grid]
            sd = [sub[(sub.dataset == ds) & (sub.eps == e) & np.isclose(sub.param, g)].primary_value.std() for g in grid]
            ax.errorbar(grid, mu, yerr=sd, color=eps_cols[e], marker="os"[k], capsize=2, lw=1.4,
                        label=fr"MIC-DP, $\varepsilon$={e:g}")
            u = summ[(summ.dataset == ds) & (summ.eps == e) & (summ.method == "uniform")].iloc[0].metric_mean
            ax.axhline(u, color=eps_cols[e], ls=":", lw=1.1, label=fr"Uniform, $\varepsilon$={e:g}")
        ax.set_title(dname(cfg, ds), loc="left")
        ax.set_xlabel(r"$\alpha$ (softmax temperature)" if exp == "alpha" else r"$\rho$ (score budget share)")
        ax.set_ylabel(metric_name(task))
        if exp == "rho":
            ax.set_xscale("log")
            ax.minorticks_off()
        ax.set_xticks(grid)
        ax.set_xticklabels([f"{g:g}" for g in grid])
        ax.tick_params(axis="x", labelsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, out, exp)


def weights_figure(w_df, scores_raw, cfg, ds, out):
    feats = list(cfg["datasets"][ds]["features"])
    d = len(feats)
    g = w_df[(w_df.experiment == "main") & (w_df.dataset == ds) & (w_df.eps == 1.0) & (w_df.method == "mic_dp")]
    agg = g.groupby("feature_index").w.agg(["mean", "std"]).reindex(range(d))
    gp = w_df[(w_df.experiment == "main") & (w_df.dataset == ds) & (w_df.eps == 1.0) & (w_df.method == "mic_public")]
    wp = gp.groupby("feature_index").w.mean().reindex(range(d))
    mic = scores_raw[scores_raw.dataset == ds].set_index("feature").loc[feats].fgmic_public.to_numpy()
    yy = np.arange(d)[::-1]
    h = max(2.4, 0.19 * d + 1.0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, h), sharey=True)
    bh = 0.38
    xerr = np.vstack([np.minimum(agg["std"], agg["mean"]), agg["std"]])
    a1.barh(yy + bh / 2, agg["mean"], height=bh, xerr=xerr, color=STYLE["mic_dp"]["color"],
            error_kw=dict(ecolor=INK2, lw=0.7, capsize=1.5), label=r"MIC-DP (privatized scores), $\varepsilon$=1")
    a1.barh(yy - bh / 2, wp, height=bh, color=STYLE["mic_public"]["color"], label="MIC-Public (public scores)")
    a1.axvline(1 / d, color=INK, ls=(0, (4, 2)), lw=1, label=f"Uniform 1/d = {1/d:.3f}")
    a1.set_xlabel(r"mean allocation weight $w_j$ (±1 std over trials)")
    a1.set_title("Budget allocation", loc="left")
    a2.barh(yy, mic, height=0.6, color="#86b6ef")
    a2.set_xlabel(r"FG-MIC(feature, target) on public split")
    a2.set_title("Public-split dependence", loc="left")
    a1.set_yticks(yy)
    a1.set_yticklabels(feats, fontsize=7 if d > 15 else 8)
    h_, l_ = a1.get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, fontsize=7.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(dname(cfg, ds), x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05 if d > 15 else 0.08, 1, 1))
    savefig(fig, out, f"weights_{ds}")


# ----------------------------------------------------------------------------- summary.md

def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_summary(cfg, raw, tests, summ, ur, dec, ds_df, out, t_report):
    real = D.real_dataset_names(cfg)
    eps_grid = cfg["privacy"]["eps_grid"]
    delta = cfg["privacy"]["delta"]
    L = []
    L.append("# Chapter 4 (MIC-DP) experiment: summary\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python -m src.report`. {cfg['protocol']['n_trials']} paired "
             f"trials, eps in {eps_grid}, delta = {delta:g}. Primary model: HistGradientBoosting(max_iter=200, "
             "early_stopping=False); secondary: LogisticRegression / Ridge. Primary metric: ROC-AUC "
             "(classification) or MAE (regression). Utility U = AUC or -MAE (higher is better). "
             "\"Significant\" means a paired two-sided Wilcoxon test with Holm-adjusted p < 0.05 within its family.\n")

    # ---- datasets
    L.append("## Datasets\n")
    t = ds_df.copy()
    t["target"] = [f"pos. rate {r.positive_rate:.3f}" if r.task == "classification" else f"std {r.target_std:.3f}"
                   for r in t.itertuples()]
    t = t[["name", "task", "n_public", "n_train", "n_test", "d", "target", "max_MIC_public", "mean_MIC_public"]]
    t["max_MIC_public"] = t.max_MIC_public.map(lambda v: f"{v:.3f}")
    t["mean_MIC_public"] = t.mean_MIC_public.map(lambda v: f"{v:.3f}")
    L.append(md_table(t))
    L.append("\nMIC statistics are FG-MIC (fixed-grid MIC, src/fgmic.py) on the full public split. "
             "CDC-Diabetes was subsampled from 253,680 to 60,000 rows before splitting.\n")

    # ---- hypotheses
    L.append("## Hypotheses (pre-registered decision rules in `config.yaml`)\n")
    h = dec["H1"]
    f1u = tests[(tests.family == "F1") & (tests.method_b == "uniform")]
    L.append(f"### H1 - MIC-DP beats Uniform at eps <= 1 on >= 3 of 4 real datasets: **{h['verdict']}**\n")
    L.append(f"Datasets where MIC-DP is significantly better than Uniform at every eps in {cfg['hypotheses']['H1_eps']}: "
             f"{h['n_qualifying']} of 4 ({', '.join(f'{dname(cfg, k)}: {v}' for k, v in h['datasets_qualifying'].items())}). "
             f"Across all 24 (dataset, eps) cells: {h['sig_wins_vs_uniform_all_eps']} significant wins, "
             f"{h['sig_losses_vs_uniform_all_eps']} significant losses vs Uniform.\n")
    t = f1u[["dataset", "eps", "mean_U_a", "mean_U_b", "mean_diff_U", "ci_lo", "ci_hi", "p_holm", "outcome"]].copy()
    t["dataset"] = t.dataset.map(lambda d: dname(cfg, d))
    t = t.rename(columns={"mean_U_a": "U MIC-DP", "mean_U_b": "U Uniform", "mean_diff_U": "diff",
                          "ci_lo": "CI lo", "ci_hi": "CI hi"})
    for c in ["U MIC-DP", "U Uniform", "diff", "CI lo", "CI hi"]:
        t[c] = t[c].map(lambda v: f"{v:.4f}")
    t["p_holm"] = t.p_holm.map(lambda v: f"{v:.3g}")
    L.append(md_table(t) + "\n")

    h = dec["H2"]
    L.append(f"### H2 - MIC-DP beats Pearson-Alloc on nonlinear data, not on linear: **{h['verdict']}**\n")
    L.append(f"Synthetic-nl-mid: {h['nl_mid_sig_wins']} of 6 eps significantly better, {h['nl_mid_sig_losses']} "
             f"significantly worse (needed >= {cfg['hypotheses']['H2_min_eps']} wins). Synthetic-lin: "
             f"{h['lin_sig_differences']} of 6 eps with a significant difference (needed 0).\n")
    t = tests[tests.family == "F2"][["dataset", "eps", "mean_diff_U", "ci_lo", "ci_hi", "p_holm", "outcome"]].copy()
    t["dataset"] = t.dataset.map(lambda d: dname(cfg, d))
    for c in ["mean_diff_U", "ci_lo", "ci_hi"]:
        t[c] = t[c].map(lambda v: f"{v:.4f}")
    t["p_holm"] = t.p_holm.map(lambda v: f"{v:.3g}")
    L.append(md_table(t.rename(columns={"mean_diff_U": "diff U (MIC-DP - Pearson-Alloc)"})) + "\n")

    h = dec["H3"]
    L.append(f"### H3 - MIC-DP beats Random-Alloc: **{h['verdict']}**\n")
    L.append(f"{h['sig_wins']} significant wins and {h['sig_losses']} significant losses out of {h['cells']} "
             f"(dataset, eps) cells (needed >= {cfg['hypotheses']['H3_min_cells']} wins, 0 losses).\n")
    t = tests[tests.family == "F3"].pivot(index="dataset", columns="eps", values="mean_diff_U")
    t.index = [dname(cfg, d) for d in t.index]
    L.append("Mean paired difference U(MIC-DP) - U(Random-Alloc):\n")
    L.append(md_table(t.map(lambda v: f"{v:+.4f}").reset_index().rename(columns={"index": "dataset"})) + "\n")

    h = dec["H4"]
    L.append(f"### H4 - gain of MIC-DP over Uniform grows with signal strength: **{h['verdict']}**\n")
    L.append(f"Slope of UR(MIC-DP) (ratio of trial means) on log(lambda) over nl-low/mid/high, 95% bootstrap CI "
             f"resampling trial seeds. "
             f"CI above 0 at {h['eps_with_positive_slope_CI']} of 6 eps (needed >= {cfg['hypotheses']['H4_min_eps']}); "
             f"CI below 0 at {h['eps_with_negative_slope_CI']}.\n")
    t = h["table"][["eps", "slope", "ci_lo", "ci_hi"] + [c for c in h["table"].columns
                                                         if c.startswith("UR_") and not c.endswith(("lo", "hi"))]].copy()
    L.append(md_table(t.map(lambda v: f"{v:.4f}" if isinstance(v, float) else v)) + "\n")

    h = dec["H5"]
    L.append(f"### H5 - MIC-Public >= MIC-DP, gap largest at small eps: **{h['verdict']}**\n")
    L.append(f"(a) MIC-Public significantly better in {h['public_sig_better']} and significantly worse in "
             f"{h['public_sig_worse']} of 24 cells -> never worse: {h['part_never_worse']}. "
             f"(b) Mean gap U(MIC-Public) - U(MIC-DP), eps in {cfg['hypotheses']['H5_small_eps']} vs "
             f"{cfg['hypotheses']['H5_large_eps']}: " +
             "; ".join(f"{dname(cfg, d)} {s:+.4f} vs {l:+.4f}" for d, (s, l) in h["gap_small_vs_large_eps"].items()) +
             f" -> larger at small eps for {h['datasets_gap_larger_at_small_eps']} of 4 datasets "
             f"(needed >= 3).\n")

    h = dec["H6"]
    L.append(f"### H6 - demographic override costs little utility on Adult: **{h['verdict']}**\n")
    L.append(f"Loss = AUC(MIC-DP) - AUC(MIC-DP+Override); equivalence margin {cfg['hypotheses']['H6_margin']} AUC. "
             f"Upper 95% CI bound within the margin at {h['eps_within_margin']} of 6 eps. Holm-adjusted Wilcoxon (F5): "
             f"override significantly worse at {h['sig_override_worse']} eps, significantly better at "
             f"{h['sig_override_better']} eps.\n")
    L.append(md_table(h["table"].map(lambda v: f"{v:.4f}" if isinstance(v, float) else v)) + "\n")

    # ---- per-dataset UR
    L.append("## Utility recovered and relative improvement over Uniform (primary model)\n")
    L.append("UR = (U_m - U_Uniform) / (U_NonPrivate - U_Uniform) from trial means (0 = Uniform, 1 = non-private). "
             "Rel. = (U_m - U_Uniform) / |U_Uniform| in %, i.e. AUC gain or MAE reduction relative to Uniform.\n")
    for ds in D.all_dataset_names(cfg):
        g = ur[(ur.dataset == ds) & (ur.method.isin(MAIN_COLS[1:] + ["random_alloc"]))]
        L.append(f"**{dname(cfg, ds)}** (non-private {metric_name(cfg['datasets'][ds]['task'])} = "
                 f"{summ[(summ.dataset == ds) & (summ.method == 'nonprivate')].iloc[0].metric_mean:.4f})\n")
        p1 = g.pivot(index="method", columns="eps", values="UR")
        p2 = g.pivot(index="method", columns="eps", values="rel_improvement_vs_uniform")
        rows = []
        for m in MAIN_COLS[1:] + ["random_alloc"]:
            rows.append({"method": disp(cfg, m), **{f"eps={e:g}": f"{p1.loc[m, e]:+.3f} ({100*p2.loc[m, e]:+.2f}%)"
                                                     for e in eps_grid}})
        L.append(md_table(pd.DataFrame(rows)) + "\n")

    # ---- counts
    L.append("## Significant wins per family (Holm within family)\n")
    rows = []
    for fam in ["F1", "F2", "F3", "F4", "F5", "F1_secondary"]:
        t = tests[tests.family == fam]
        rows.append({"family": fam, "comparison": f"{disp(cfg, t.method_a.iloc[0])} vs " +
                     ", ".join(sorted({disp(cfg, b) for b in t.method_b})),
                     "tests": len(t), "A sig. better": int((t.outcome == "A better").sum()),
                     "A sig. worse": int((t.outcome == "B better").sum()), "n.s.": int((t.outcome == "n.s.").sum())})
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    f1 = tests[tests.family == "F1"]
    rows = []
    for b in S.MAIN_BASELINES:
        t = f1[f1.method_b == b]
        rows.append({"baseline": disp(cfg, b), "MIC-DP sig. better": int((t.outcome == "A better").sum()),
                     "MIC-DP sig. worse": int((t.outcome == "B better").sum()), "n.s.": int((t.outcome == "n.s.").sum())})
    L.append("F1 broken down by baseline (24 cells each):\n")
    L.append(md_table(pd.DataFrame(rows)) + "\n")

    # ---- std range
    L.append("## Across-trial standard deviation of the primary metric: MIC-DP vs Uniform\n")
    rows = []
    for ds in D.all_dataset_names(cfg):
        g = summ[summ.dataset == ds]
        a = g[g.method == "mic_dp"].metric_std
        b = g[g.method == "uniform"].metric_std
        rows.append({"dataset": dname(cfg, ds), "MIC-DP std range": f"{a.min():.4f} - {a.max():.4f}",
                     "Uniform std range": f"{b.min():.4f} - {b.max():.4f}",
                     "median ratio MIC-DP/Uniform": f"{np.median(a.to_numpy() / np.maximum(b.to_numpy(), 1e-12)):.2f}"})
    L.append(md_table(pd.DataFrame(rows)) + "\n")

    # ---- sigma_s
    dsk = ds_df.set_index("key")
    L.append("## Score-privatization noise sigma_s (rho = 0.2)\n")
    L.append("sigma_s = sqrt(d) * 3(1 + ln n)/(n ln 2) * c(rho*eps, rho*delta), with n the private-train size. "
             "For scale, the last column gives the standard deviation of the exact FG-MIC scores across features "
             "(private-train split): a ranking is only informative where sigma_s is not much larger than this.\n")
    sraw = pd.read_csv(out / "scores_raw.csv")
    rows = []
    for ds in D.all_dataset_names(cfg):
        d = int(dsk.loc[ds, "d"])
        s1 = math.sqrt(d) * float(dsk.loc[ds, "fgmic_sensitivity"])
        rows.append({"dataset": dname(cfg, ds), "d": d,
                     **{f"eps={e:g}": f"{s1 * dp.noise_multiplier(0.2 * e, 0.2 * delta):.4f}" for e in eps_grid},
                     "sd of FG-MIC": f"{sraw[sraw.dataset == ds].fgmic_train.std():.4f}"})
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    sub = raw[(raw.experiment == "main") & (raw.model == "primary") & (raw.method == "mic_dp")]
    rows = []
    for e in eps_grid:
        g = sub[(sub.eps == e) & sub.dataset.isin(real)]
        rows.append({"eps": f"{e:g}", "mean shrinkage factor B": f"{g.shrink_B.mean():.3f}",
                     "share of trials with pruning (B >= 0.5)": f"{np.mean(g.shrink_B >= cfg['methods']['defaults']['prune_min_B']):.2f}",
                     "mean share of columns released": f"{np.mean(g.n_released / g.dataset.map(lambda k: dsk.loc[k, 'd'])):.2f}"})
    L.append("MIC-DP on the four real datasets (B = 0: weights fall back to uniform; B = 1: exact scores):\n")
    L.append(md_table(pd.DataFrame(rows)) + "\n")

    # ---- collapse diagnostic
    L.append("## Diagnostic: constant predictions on the clean test set\n")
    L.append("Before the noise-aware prediction, HistGradientBoosting returned one constant prediction for the whole "
             "clean test set in many fits at eps <= 1 (all clean test values fall into one histogram bin). Share of "
             "primary-model fits with a single distinct test prediction in this run:\n")
    sub = raw[(raw.experiment == "main") & (raw.model == "primary") & (raw.method.isin(MAIN_COLS))]
    t = sub.assign(const=sub.n_unique_pred <= 1).groupby(["dataset", "eps"]).const.mean().unstack()
    t.index = [dname(cfg, d) for d in t.index]
    L.append(md_table(t.map(lambda v: f"{v:.2f}").reset_index().rename(columns={"index": "dataset"})) + "\n")

    # ---- reading the results (all numbers computed here)
    L.append("## Reading the results\n")
    def ratio(e):
        out_ = []
        for ds in real:
            g = summ[(summ.dataset == ds) & (summ.eps == e)].set_index("method")
            out_.append(g.loc["mic_dp", "dist_mean"] / g.loc["uniform", "dist_mean"])
        return min(out_), max(out_)
    f1s_w = tests[(tests.family == "F1_secondary") & (tests.method_b == "uniform") & (tests.outcome == "A better")]
    f1s_l = tests[(tests.family == "F1_secondary") & (tests.method_b == "uniform") & (tests.outcome == "B better")]
    f1u = tests[(tests.family == "F1") & (tests.method_b == "uniform")]
    lo1, hi1 = ratio(1.0)
    lo10, hi10 = ratio(10.0)
    L.append(f"1. **MIC-DP vs Uniform, primary model:** {int((f1u.outcome == 'A better').sum())} significant wins and "
             f"{int((f1u.outcome == 'B better').sum())} significant losses in 24 (dataset, eps) cells. Wins: " +
             (", ".join(f"{dname(cfg, r.dataset)} eps={r.eps:g} ({r.mean_diff_U:+.4f})"
                        for r in f1u[f1u.outcome == 'A better'].itertuples()) or "none") + ".")
    L.append(f"2. **MIC-DP vs Uniform, linear model (F1_secondary):** {len(f1s_w)} significant wins, {len(f1s_l)} "
             f"significant losses.")
    L.append(f"3. **Distortion of the released columns.** MIC-DP's mean |X_tilde - X| over released entries is "
             f"{lo1:.2f}-{hi1:.2f} times Uniform's at eps = 1 and {lo10:.2f}-{hi10:.2f} times at eps = 10 on the real "
             f"datasets. Below 1 means the released columns are less noisy than under Uniform (fewer columns share the budget).")
    n_sig = int((tests[tests.family.isin(['F1', 'F2', 'F3', 'F4', 'F5'])].outcome != "n.s.").sum())
    n_all = int(tests.family.isin(['F1', 'F2', 'F3', 'F4', 'F5']).sum())
    L.append(f"4. **Power.** {n_sig} of {n_all} pre-registered comparisons are significant after Holm correction. "
             "Where a hypothesis is PARTIAL because a 'no difference' or 'never worse' part holds, that part rests on "
             "non-significant tests and is weak evidence on its own.\n")

    # ---- deviations
    L.append("## Deviations from the specification, and why\n")
    L.append("""1. **Enhanced MIC-DP (v2), at the author's request after the first run.** The first run showed that the specified score release (sensitivity sqrt(d), sigma_s = 6-796 on scores in [0, 1]) leaves no usable ranking. This run changes the method; every setting below was fixed before it ran and is in `config.yaml`:
   - **Scores (Phase 1):** fixed-grid MIC (FG-MIC, `src/fgmic.py`) on the full private-train split instead of MIC on a 3,000-row subsample. Grid cut points are quantiles of the public split (32 candidate bins per feature; target rows = classes, or 2-16 public-quantile rows for regression), so the grid family does not depend on the private data and each score has sensitivity 3(1 + ln n)/(n ln 2) (proof sketch in `src/fgmic.py`, empirical check in `tests/test_fgmic.py`) instead of 1.
   - **Pearson / partial-correlation baselines** get the same treatment: one Gaussian release of the first and second moments of [x, y] (sensitivity sqrt((d+1) + (d+1)(d+2)/2)/n), correlations by post-processing. Regression targets are clipped to the public-split 0.5-99.5 percentiles for this release only.
   - **Allocation (Phase 2):** noisy scores are shrunk toward their mean (B = min(positive-part James-Stein factor, tau0^2/(tau0^2 + sigma^2)), tau0 = 0.1); weights w_j ~ (s_j / max s)^alpha with alpha = 2; columns with relative score below theta = 0.2 are not released once B >= 0.5. The specification required w_j > 0; not releasing a column is allowed by DP and keeps sum over released columns of 1/sigma_j^2 = 1/c^2. Pearson-Adj and PartialCorr-Adj keep their (1 + s)^-2 rule on shrunken scores.
   - **Random-Alloc** is MIC-DP's weight vector of the same trial, randomly permuted (same shape, signal removed), instead of softmax of U(0,1) scores.
   - **Prediction:** for every private method, the primary model predicts E[f(x + eta)] under the public release noise, computed exactly from the trees (`src/noiseaware.py`). This is post-processing; non-private and linear models are unchanged.
   - **Sweeps:** alpha in {0, 1, 2, 5, 10} is now the exponent of the relevance rule (alpha = 0 is uniform over released columns).
2. **MIC code.** `src/mic.py` (own MIC_e implementation, cross-checked against minepy in `mic_crosscheck.csv`) is kept and tested but no longer used by the pipeline; FG-MIC has its own exact dynamic programme (`tests/test_fgmic.py` checks it against brute force).
3. **Public split usage.** Beyond MIC-Public, the public split now supplies the grid cut points (feature and target quantiles) and the target range for the moment release. MIC-Public uses FG-MIC on the public split.
4. **Data source.** `ucimlrepo` (archive.ics.uci.edu) and sklearn's `fetch_california_housing` (figshare) are blocked here. The same files were downloaded from GitHub mirrors and pinned by SHA-256 in `config.yaml`: UCI Adult `adult.data` + `adult.test` (48,842 rows, byte-identical across two mirrors), the CDC BRFSS 2015 binary file behind UCI id 891 (253,680 rows, byte-identical across two mirrors), the original authors' Appliances repository (19,735 rows), and the StatLib `cal_housing.data` file that sklearn downloads. CalHousing is rebuilt with sklearn's exact transformation.
5. **Appliances temperature ranges.** T1-T9 use the DHT-22 operating range [-40, 80] C, recalled from Candanedo et al. (2017) and not re-checked from this sandbox; please confirm the citation.
6. **Adult ranges.** capital-gain uses [0, 99999] (the Census top-code); capital-loss uses the public-split fallback. Categorical codes follow `adult.names`; '?' is an extra category for workclass, occupation and native-country.
7. **MIC-DP+Override and Random-Alloc draws.** Both reuse the MIC-DP trial's privatized scores (stream id 2) and draw their own mechanism noise (ids 9 and 7).
8. **Extra outputs.** Tables for the synthetic datasets (no daggers), the `F1_secondary` family, `scores_raw.csv`, `sanity.log`, `runtime.txt`, `captions.tex`, and the columns `n_unique_pred`, `n_released`, `shrink_B`, `s_tilde` in `raw_trials.parquet`.
""")
    L.append(f"\n_Report generation time: {t_report:.0f} s._\n")
    (out / "summary.md").write_text("\n".join(L))


# ----------------------------------------------------------------------------- main

def main() -> None:
    t0 = time.time()
    cfg = D.load_config()
    out = D.ROOT / cfg["outputs"]["dir"]
    raw = S.load_trials(cfg)
    w_df = pd.read_parquet(out / "weights.parquet")
    ds_df = pd.read_csv(out / "datasets.csv")
    scores_raw = pd.read_csv(out / "scores_raw.csv")

    tests = S.all_tests(raw, cfg)
    dec = S.decide(tests, raw, cfg)
    h4 = dec["H4"]["table"].assign(family="H4_slope", hypothesis="H4", dataset="syn_nl_low/mid/high",
                                   method_a="mic_dp", method_b="uniform", model="primary",
                                   mean_diff_U=lambda t: t.slope)
    h6 = dec["H6"]["table"].assign(family="H6_equivalence", hypothesis="H6", dataset="adult",
                                   method_a="mic_dp", method_b="mic_dp_override", model="primary",
                                   mean_diff_U=lambda t: t.mean_loss)
    pd.concat([tests, h4[["family", "hypothesis", "dataset", "eps", "method_a", "method_b", "model", "mean_diff_U",
                          "ci_lo", "ci_hi"]],
               h6[["family", "hypothesis", "dataset", "eps", "method_a", "method_b", "model", "mean_diff_U",
                   "ci_lo", "ci_hi"]]], ignore_index=True).to_csv(out / "stats.csv", index=False)

    summ = S.summary_table(raw, "primary")
    summ2 = S.summary_table(raw, "secondary")
    ur = S.ur_table(raw, "primary")
    ur.to_csv(out / "ur_primary.csv", index=False)

    captions = ["% Non-private reference values for table captions (mean $\\pm$ std over trials)"]
    real = D.real_dataset_names(cfg)
    for ds in D.all_dataset_names(cfg):
        main_table(summ, tests, cfg, ds, "primary", "F1" if ds in real else None, out, captions)
        main_table(summ2, tests, cfg, ds, "secondary", "F1_secondary" if ds in real else None, out, captions)
        dual_figure(summ, cfg, ds, out)
        weights_figure(w_df, scores_raw, cfg, ds, out)
    (out / "captions.tex").write_text("\n".join(captions) + "\n")
    distortion_table(summ, cfg, out)
    ablation_table(summ, cfg, out)
    synthetic_table(raw, ur, cfg, out)
    override_table(raw, cfg, out)
    for exp in ("alpha", "rho"):
        sweep_table(raw, summ, cfg, exp, out)
        sweep_figure(raw, summ, cfg, exp, out)
    dose_figure(raw, cfg, out, dec["H4"]["table"])

    # MIC-Oracle must not appear in any DP table except the ablation
    for f in out.glob("tab_*.tex"):
        if f.name != "tab_ablation.tex":
            assert "MIC-Oracle" not in f.read_text(), f"MIC-Oracle leaked into {f.name}"
    write_summary(cfg, raw, tests, summ, ur, dec, ds_df, out, time.time() - t0)
    print(f"report written to {out} in {time.time() - t0:.0f} s")
    for k in ["H1", "H2", "H3", "H4", "H5", "H6"]:
        print(k, dec[k]["verdict"])


if __name__ == "__main__":
    main()
