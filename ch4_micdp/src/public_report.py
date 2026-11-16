"""Statistics, LaTeX table bodies, figure and summary.md for the public-data baselines.

    python -m src.public_report
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from . import data as D  # noqa: E402

BLUE, ORANGE, GREEN, PURPLE, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#7c5cc4", "#52514e"
SIZES = ["p15", "p5", "p1"]
SIZE_TXT = {"p15": "15%", "p5": "5%", "p1": "1%", "shift5": "shifted 5%"}
DS_TXT = {"adult": "Adult", "cdc_diabetes": "CDC-Diabetes"}
WIDTH = 6.3

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def name_of(key: str) -> str:
    m, _, s = key.partition("@")
    base = {"nonprivate": "Non-private", "uniform": "Uniform", "mic_dp": "MIC-DP", "mic_public": "MIC-Public",
            "public_only": "Public-Only", "ft_uniform": "Public-Pretrain + Uniform-FT",
            "ft_mic_dp": "Public-Pretrain + MIC-DP-FT", "pooled_uniform": "Public + Uniform (pooled)"}[m]
    return f"{base} ({SIZE_TXT[s]})" if s else base


# ----------------------------------------------------------------------------- data

def load(cfg):
    pb = cfg["public_baselines"]
    out = D.ROOT / pb["out_dir"]
    new = pd.read_parquet(out / "raw_trials_public.parquet")
    new["key"] = new.method + "@" + new.public_size
    old = pd.read_parquet(D.ROOT / cfg["outputs"]["dir"] / "raw_trials.parquet")
    old = old[(old.experiment == "main") & old.dataset.isin(pb["datasets"])
              & old.method.isin(["nonprivate", "uniform", "mic_dp", "mic_public"])].copy()
    old["key"] = old.method.map(lambda m: "mic_public@p15" if m == "mic_public" else m)
    cols = ["dataset", "trial", "model", "eps", "key", "roc_auc", "accuracy"]
    L = pd.concat([old[cols], new[cols]], ignore_index=True)
    return L, new, out


class Store:
    def __init__(self, L, n_trials):
        self.n = n_trials
        self.v = {}
        for (ds, mo, key), g in L.groupby(["dataset", "model", "key"]):
            for e, h in g.groupby(g.eps.fillna(-1.0)):
                h = h.sort_values("trial")
                assert h.trial.tolist() == list(range(n_trials)), (ds, mo, key, e)
                self.v[(ds, mo, key, float(e))] = (h.roc_auc.to_numpy(), h.accuracy.to_numpy())

    def auc(self, ds, key, eps, model="primary"):
        k = (ds, model, key, float(eps))
        if k not in self.v:
            k = (ds, model, key, -1.0)           # eps-independent (Non-private, Public-Only)
        return self.v[k][0]

    def acc(self, ds, key, eps, model="primary"):
        k = (ds, model, key, float(eps))
        if k not in self.v:
            k = (ds, model, key, -1.0)
        return self.v[k][1]


# ----------------------------------------------------------------------------- statistics

def wilcoxon_p(d):
    d = np.asarray(d, float)
    if np.all(d == 0):
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def boot(x, cfg, *key):
    x = np.asarray(x, float)
    rng = np.random.default_rng([cfg["public_baselines"]["bootstrap_seed"], zlib.crc32(repr(key).encode())])
    idx = rng.integers(0, x.size, size=(cfg["stats"]["bootstrap_resamples"], x.size))
    m = x[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def test_rows(S, cfg, family, cells, correct=True):
    rows = []
    for c in cells:
        a = S.auc(c["dataset"], c["a"], c["eps"])
        b = S.auc(c["dataset"], c["b"], c["eps"])
        d = a - b
        lo, hi = boot(d, cfg, family, c["dataset"], c["a"], c["b"], c["eps"])
        rows.append({"family": family, **c, "mean_a": a.mean(), "mean_b": b.mean(), "diff": d.mean(),
                     "ci_lo": lo, "ci_hi": hi, "n_a_wins": int((d > 0).sum()), "n_b_wins": int((d < 0).sum()),
                     "p_raw": wilcoxon_p(d)})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p_raw.to_numpy()) if correct else np.nan
    p = t.p_holm if correct else t.p_raw
    sig = p < cfg["stats"]["alpha_level"]
    t["outcome"] = np.where(sig & (t["diff"] > 0), "A better", np.where(sig & (t["diff"] < 0), "B better", "n.s."))
    if not correct:
        t["outcome"] = t["outcome"] + " (unadjusted)"
    return t


def families(S, cfg):
    pb = cfg["public_baselines"]
    eps_grid = pb["eps_grid"]
    f6 = [dict(dataset=ds, size=s, eps=e, a=("mic_public@p15" if pb["best_private"][ds][e] == "mic_public"
                                             else pb["best_private"][ds][e]), b=f"public_only@{s}")
          for ds in pb["datasets"] for s in SIZES for e in eps_grid]
    f7 = []
    for ds in pb["datasets"]:
        for e in eps_grid:
            for s in ("p15", "p5"):
                for meth, parent in (("ft_uniform", "uniform"), ("ft_mic_dp", "mic_dp"), ("pooled_uniform", "uniform")):
                    for par in (f"public_only@{s}", parent):
                        f7.append(dict(dataset=ds, size=s, eps=e, method=meth, a=f"{meth}@{s}", b=par,
                                       parent="Public-Only" if par.startswith("public_only") else "private"))
    f8 = [dict(dataset=ds, size=s, eps=e, a=f"mic_public@{s}", b="mic_dp")
          for ds in pb["datasets"] for s in SIZES for e in eps_grid]
    sh = []
    for ds in pb["datasets"]:
        for e in eps_grid:
            best = "mic_public@p15" if pb["best_private"][ds][e] == "mic_public" else pb["best_private"][ds][e]
            sh += [dict(dataset=ds, eps=e, comparison="best private - Public-Only (shifted 5%)", a=best, b="public_only@shift5"),
                   dict(dataset=ds, eps=e, comparison="Public-Only (shifted 5%) - Public-Only (5%)", a="public_only@shift5", b="public_only@p5"),
                   dict(dataset=ds, eps=e, comparison="MIC-Public (shifted 5%) - MIC-Public (5%)", a="mic_public@shift5", b="mic_public@p5"),
                   dict(dataset=ds, eps=e, comparison="MIC-Public (shifted 5%) - MIC-DP", a="mic_public@shift5", b="mic_dp")]
    return (test_rows(S, cfg, "F6", f6), test_rows(S, cfg, "F7", f7), test_rows(S, cfg, "F8", f8),
            test_rows(S, cfg, "shift", sh, correct=False))


def gap_table(S, cfg):
    """G = AUC(method) - AUC(Public-Only at the same public size); methods without a public size against
    every size. Mean, 95% bootstrap interval, unadjusted Wilcoxon p (F6 is the corrected family)."""
    pb = cfg["public_baselines"]
    rows = []
    for ds in pb["datasets"]:
        for e in pb["eps_grid"]:
            pairs = [(k, f"public_only@{s}") for k in ("nonprivate", "uniform", "mic_dp") for s in SIZES]
            pairs += [(f"mic_public@{s}", f"public_only@{s}") for s in SIZES]
            pairs += [(f"{m}@{s}", f"public_only@{s}") for m in ("ft_uniform", "ft_mic_dp", "pooled_uniform") for s in ("p15", "p5")]
            pairs += [("mic_public@shift5", "public_only@shift5")]
            for a, b in pairs:
                g = S.auc(ds, a, e) - S.auc(ds, b, e)
                lo, hi = boot(g, cfg, "G", ds, a, b, e)
                rows.append(dict(dataset=ds, eps=e, method=a, public_only=b, G=g.mean(), ci_lo=lo, ci_hi=hi,
                                 p_raw=wilcoxon_p(g)))
    return pd.DataFrame(rows)


def ur_table(S, cfg, keys):
    pb = cfg["public_baselines"]
    rows = []
    for ds in pb["datasets"]:
        for e in pb["eps_grid"]:
            U = S.auc(ds, "uniform", e)
            N = S.auc(ds, "nonprivate", e)
            rng = np.random.default_rng([pb["bootstrap_seed"], 77, zlib.crc32(f"{ds}{e}".encode())])
            idx = rng.integers(0, U.size, size=(cfg["stats"]["bootstrap_resamples"], U.size))
            for k in keys:
                M = S.auc(ds, k, e)
                den = N.mean() - U.mean()
                ur = (M.mean() - U.mean()) / den
                bden = N[idx].mean(1) - U[idx].mean(1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    bur = (M[idx].mean(1) - U[idx].mean(1)) / bden
                bur = bur[np.isfinite(bur)]
                rows.append(dict(dataset=ds, eps=e, method=k, UR=ur, UR_lo=float(np.quantile(bur, 0.025)),
                                 UR_hi=float(np.quantile(bur, 0.975))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- answers to the rules

def answers(f6, f7, f8, cfg):
    pb = cfg["public_baselines"]
    small = [e for e in pb["eps_grid"] if e <= 1]
    A = {}
    for ds in pb["datasets"]:
        a = {}
        t = f6[f6.dataset == ds]
        hold = [s for s in ("p1", "p5", "p15")
                if all((t[(t["size"] == s) & (t.eps == e)].outcome == "B better").all() for e in small)]
        a["rule1"] = dict(smallest=(SIZE_TXT[hold[0]] if hold else "none"), sizes_where_it_holds=[SIZE_TXT[s] for s in hold])
        r2 = {}
        for s in SIZES:
            ee = t[(t["size"] == s) & (t.outcome == "A better")].eps.tolist()
            r2[SIZE_TXT[s]] = dict(largest=(f"{max(ee):g}" if ee else "none"), all_eps=[f"{x:g}" for x in sorted(ee)])
        a["rule2"] = r2
        r3 = []
        t7 = f7[f7.dataset == ds]
        for (e, s, m), g in t7.groupby(["eps", "size", "method"], sort=False):
            o = dict(zip(g.parent, g.outcome))
            r3.append(dict(eps=e, size=SIZE_TXT[s], method=name_of(m + "@" + s).split(" (")[0],
                           vs_public_only=o["Public-Only"], vs_private=o["private"],
                           better_than_both="yes" if (o["Public-Only"] == "A better" and o["private"] == "A better") else "no"))
        a["rule3"] = r3
        r4 = []
        t8 = f8[f8.dataset == ds]
        for e in pb["eps_grid"]:
            g = t8[t8.eps == e].set_index("size")
            helps = {s: g.loc[s].outcome == "A better" for s in SIZES}
            if all(helps.values()):
                stop = "none"
            elif not helps["p15"]:
                stop = "15%"
            elif not helps["p5"]:
                stop = "5%"
            else:
                stop = "1%"
            r4.append(dict(eps=e, mic_dp=g.loc["p15"].mean_b, **{f"mic_public_{SIZE_TXT[s]}": g.loc[s].mean_a for s in SIZES},
                           **{f"outcome_{SIZE_TXT[s]}": g.loc[s].outcome for s in SIZES}, stops_helping_at=stop))
        a["rule4"] = r4
        A[ds] = a
    return A


# ----------------------------------------------------------------------------- tables and figure

TAB_COLS = ["public_only@p15", "public_only@p5", "public_only@p1", "uniform", "mic_dp", "mic_public@p15",
            "mic_public@p5", "mic_public@p1", "ft_uniform@p15", "ft_mic_dp@p15", "pooled_uniform@p15"]
SHIFT_COLS = ["public_only@shift5", "public_only@p5", "mic_public@shift5", "mic_public@p5", "mic_dp", "uniform"]


def pm(x):
    return f"{x.mean():.3f} $\\pm$ {x.std(ddof=1):.3f}"


def pm_md(x):
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"


def write_tables(S, cfg, out):
    pb = cfg["public_baselines"]
    for ds in pb["datasets"]:
        for model, stem in (("primary", f"tab_public_{ds}"), ("secondary", f"tab_public_secondary_{ds}")):
            npv = S.auc(ds, "nonprivate", 0, model)
            lines = [f"% {DS_TXT[ds]}, {'primary model (HistGradientBoosting)' if model == 'primary' else 'secondary model (logistic regression)'}, "
                     f"ROC-AUC on the clean test set, mean $\\pm$ std over {S.n} trials",
                     "% columns: eps & " + " & ".join(name_of(k) for k in TAB_COLS),
                     f"% caption note: Non-private (private-training split, no noise) ROC-AUC = {npv.mean():.3f} $\\pm$ {npv.std(ddof=1):.3f}; "
                     "Public-Only does not depend on eps and touches no private record"]
            for e in pb["eps_grid"]:
                lines.append(f"{e:g} & " + " & ".join(pm(S.auc(ds, k, e, model)) for k in TAB_COLS) + " \\\\")
            (out / f"{stem}.tex").write_text("\n".join(lines) + "\n")
    lines = ["% EXPLORATORY: shifted public split (5% of the dataset drawn from public records with age <= its public "
             "median: Adult age <= 38, CDC-Diabetes Age category <= 8); primary model, ROC-AUC, mean $\\pm$ std over trials",
             "% columns: dataset & eps & " + " & ".join(name_of(k) for k in SHIFT_COLS)]
    for ds in pb["datasets"]:
        for i, e in enumerate(pb["eps_grid"]):
            lines.append(f"{DS_TXT[ds] if i == 0 else ''} & {e:g} & " + " & ".join(pm(S.auc(ds, k, e)) for k in SHIFT_COLS) + " \\\\")
        lines.append("\\midrule")
    (out / "tab_public_shift.tex").write_text("\n".join(lines[:-1]) + "\n")


def figure(S, cfg, out):
    pb = cfg["public_baselines"]
    eps = pb["eps_grid"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharex=True)
    lines = [("uniform", ORANGE, "o"), ("mic_dp", BLUE, "s"), ("mic_public@p15", GREEN, "^"), ("ft_mic_dp@p15", PURPLE, "D")]
    for ax, ds in zip(axes, pb["datasets"]):
        for k, col, mk in lines:
            mu = np.array([S.auc(ds, k, e).mean() for e in eps])
            sd = np.array([S.auc(ds, k, e).std(ddof=1) for e in eps])
            ax.fill_between(eps, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
            ax.plot(eps, mu, color=col, marker=mk, ms=3.5, label=name_of(k))
        labels = []
        for s, ls in (("p15", (0, (8, 2.5))), ("p5", (0, (4.5, 2))), ("p1", (0, (2.2, 1.6)))):
            v = S.auc(ds, f"public_only@{s}", 0).mean()
            ax.axhline(v, color=GRAY, ls=ls, lw=1.1)
            labels.append([v, f"Public-Only {SIZE_TXT[s]}"])
        npv = S.auc(ds, "nonprivate", 0).mean()
        ax.axhline(npv, color=GRAY, ls=(0, (0.6, 2.0)), lw=1.6, dash_capstyle="round")
        labels.append([npv, "Non-private"])
        lo, hi = ax.get_ylim()
        gap = 0.045 * (hi - lo)
        labels.sort(key=lambda t: t[0])
        ys = [t[0] for t in labels]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        for (v, txt), y in zip(labels, ys):
            ax.annotate(txt, xy=(eps[-1], v), xytext=(eps[-1] * 1.25, y), va="center", ha="left", fontsize=6.3,
                        color=GRAY, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5) if abs(y - v) > 1e-9 else None)
        ax.set_ylim(lo, max(hi, ys[-1] + gap / 2))
        ax.set_xscale("log")
        ax.set_xticks(eps)
        ax.set_xticklabels([f"{e:g}" for e in eps])
        ax.minorticks_off()
        ax.set_title(DS_TXT[ds])
        ax.set_xlabel(r"privacy budget $\varepsilon$")
    axes[0].set_ylabel("ROC-AUC (primary model)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=4.5)
    fig.savefig(out / "public_ch4.pdf")
    fig.savefig(out / "public_ch4.png", dpi=220)
    plt.close(fig)


# ----------------------------------------------------------------------------- summary

def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v:.4g}" if abs(v) < 1e-3 and v != 0 else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_tests(t, extra=()):
    k = t.copy()
    k["A"] = k.a.map(name_of)
    k["B"] = k.b.map(name_of)
    k["dataset"] = k.dataset.map(DS_TXT)
    if "size" in k:
        k["size"] = k["size"].map(SIZE_TXT)
    cols = ["dataset"] + [c for c in ("size", "eps", "method", "parent", "comparison") if c in k] + list(extra)
    cols += ["A", "B", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi", "n_a_wins", "n_b_wins", "p_raw", "p_holm", "outcome"]
    if "method" in cols:
        k["method"] = k.method.map(lambda m: name_of(m).split(" (")[0])
    return md(k[[c for c in cols if c in k]])


def write_summary(S, cfg, out, f6, f7, f8, fsh, G, UR, A):
    pb = cfg["public_baselines"]
    L = ["# Chapter 4 (MIC-DP): public-data baselines\n",
         "Produced by `python -m src.public && python -m src.public_report`. Numbers are reported as they came out. "
         "Primary model: HistGradientBoosting (200 iterations, no early stopping, random_state = trial seed) with "
         "noise-aware prediction for every method that trains on a privatized release; secondary model: logistic "
         f"regression. ROC-AUC and accuracy on the clean test set, {S.n} paired trials (seed 1000 + t), delta = 1e-5. "
         "Existing methods (Non-private, Uniform, MIC-DP, MIC-Public 15%) are taken from `results/ch4/raw_trials.parquet` "
         "of the main run; their trial 0 was re-run and reproduced (sanity checks).\n"]
    L.append("## Pre-run commit\n")
    L.append((out / "prerun_commit.txt").read_text().strip() + "\n")
    L.append("## Privacy accounting\n")
    L.append("Public-Only touches no private record and has no (eps, delta). Every method that trains on a privatized "
             "release has exactly the (eps, delta) of that release: Public-Pretrain + Uniform-FT and Public + Uniform "
             "(pooled) use the Uniform release of the main run, Public-Pretrain + MIC-DP-FT uses the MIC-DP release, and "
             "MIC-Public at 5%, 1% and shifted 5% spend no budget on scores (rho = 0, scores from public data) and the "
             "whole (eps, delta) on the data release. Pretraining on public data, pooling with public rows, and "
             "noise-aware prediction are post-processing of the release and add no privacy cost.\n")
    L.append("## Reporting rules (verbatim, committed before the run)\n")
    L.append((D.ROOT / "public_reporting_rules.md").read_text().split("# Public-data baselines: reporting rules (committed before the first run)\n")[-1] + "\n")
    L.append("## Repository\n")
    L.append("The original Chapter 4 repository and its commit 0772eff did not survive a reset of the compute environment. "
             "It was restored from the archived snapshot of the v2 run (ch4_micdp_experiment_v2.zip: code, config, tests, "
             "results) as commit 52748a9; src/dp.py, fgmic.py, methods.py and data.py have the SHA-256 recorded at 0772eff, and "
             "the environment has the package versions of the main run (numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, "
             "scipy 1.18.1; Python 3.13.18 instead of 3.13.15). The re-run of trial 0 reproduces the stored results of the "
             "existing methods to 1e-9 (sanity checks).\n")
    L.append("## Sanity checks\n")
    L.append("```\n" + (out / "sanity_public.log").read_text() + "```\n")
    L.append("## Notes on the outputs\n")
    L.append("Public-Only has no random component (HistGradientBoosting without early stopping or feature subsampling; "
             "deterministic logistic regression), so it returns the same model and the same ROC-AUC in every trial; its "
             "standard deviation over trials is 0.\n")
    raw = pd.read_parquet(out / "raw_trials_public.parquet")
    c = raw[raw.model == "primary"].assign(const=lambda x: x.n_unique_pred <= 1)
    c = c.groupby(["dataset", "method", "public_size", "eps"], dropna=False).const.sum().reset_index()
    c = c[c.const > 0]
    L.append("Runs whose primary-model test predictions are constant (n_unique_pred = 1; ROC-AUC = 0.5), out of 30 trials "
             "(cells without constant predictions are omitted):\n")
    if len(c):
        c["dataset"] = c.dataset.map(DS_TXT)
        c["method"] = (c.method + "@" + c.public_size).map(name_of)
        L.append(md(c[["dataset", "method", "eps", "const"]].rename(columns={"const": "constant runs"})) + "\n")
    else:
        L.append("None.\n")
    dpath = out / "diagnostic_ft.csv"
    if dpath.exists():
        dg = pd.read_csv(dpath)
        dg["dataset"] = dg.dataset.map(DS_TXT)
        dg["method"] = (dg.method + "@p15").map(name_of)
        L.append("Diagnostic of the fine-tuned primary models (exploratory; `diagnostics/ft_leaf_values.py`, trial 0, 15% "
                 "split, the same code path as the run; nothing was changed): mean and maximum |raw output| of the pretrained "
                 "model on the privatized private-training rows, median logistic Hessian p(1 - p) at those values, largest "
                 "|leaf value| among the 100 pretrained trees, median and maximum of the per-tree largest |leaf value| among the "
                 "100 continued trees, range of the noise-aware raw prediction on the test set, and the number of distinct "
                 "test probabilities.\n")
        L.append(md(dg) + "\n")
    # tables
    for model in ("primary", "secondary"):
        L.append(f"## ROC-AUC, {model} model (mean ± std over trials)\n")
        for ds in pb["datasets"]:
            npv = S.auc(ds, "nonprivate", 0, model)
            keys = ["public_only@p15", "public_only@p5", "public_only@p1", "uniform", "mic_dp", "mic_public@p15",
                    "mic_public@p5", "mic_public@p1", "ft_uniform@p15", "ft_uniform@p5", "ft_mic_dp@p15", "ft_mic_dp@p5",
                    "pooled_uniform@p15", "pooled_uniform@p5"]
            rows = []
            for k in keys:
                r = {"method": name_of(k)}
                for e in pb["eps_grid"]:
                    r[f"eps={e:g}"] = pm_md(S.auc(ds, k, e, model))
                rows.append(r)
            L.append(f"**{DS_TXT[ds]}** (Non-private {pm_md(npv)}):\n")
            L.append(md(pd.DataFrame(rows)) + "\n")
    L.append("## Accuracy, primary model (mean over trials)\n")
    for ds in pb["datasets"]:
        rows = []
        for k in ["public_only@p15", "public_only@p5", "public_only@p1", "uniform", "mic_dp", "mic_public@p15",
                  "mic_public@p5", "mic_public@p1", "ft_uniform@p15", "ft_mic_dp@p15", "pooled_uniform@p15"]:
            rows.append({"method": name_of(k), **{f"eps={e:g}": f"{S.acc(ds, k, e).mean():.4f}" for e in pb["eps_grid"]}})
        L.append(f"**{DS_TXT[ds]}** (Non-private {S.acc(ds, 'nonprivate', 0).mean():.4f}):\n")
        L.append(md(pd.DataFrame(rows)) + "\n")
    L.append("## Gap to Public-Only, G = AUC(method) - AUC(Public-Only at the same public size), primary model\n")
    L.append("Mean over trials, 95% bootstrap interval, unadjusted paired Wilcoxon p (the Holm-corrected comparisons "
             "are F6, F7 and F8 below). Methods that use no public data are compared with Public-Only at every size.\n")
    g = G.copy()
    g["dataset"] = g.dataset.map(DS_TXT)
    g["method"] = g.method.map(name_of)
    g["public_only"] = g.public_only.map(name_of)
    L.append(md(g) + "\n")
    L.append("## Utility recovered UR (anchors Uniform and Non-private), primary model\n")
    u = UR.copy()
    u["dataset"] = u.dataset.map(DS_TXT)
    u["method"] = u.method.map(name_of)
    L.append(md(u) + "\n")
    L.append("## Family F6: best private method vs Public-Only (Holm over 36 tests)\n")
    L.append(fmt_tests(f6) + "\n")
    L.append("## Answers to the reporting rules\n")
    for ds in pb["datasets"]:
        a = A[ds]
        L.append(f"### {DS_TXT[ds]}\n")
        L.append(f"1. Smallest public size at which Public-Only is significantly better than the best private method at "
                 f"every eps <= 1: **{a['rule1']['smallest']}** (sizes at which this holds: "
                 f"{', '.join(a['rule1']['sizes_where_it_holds']) or 'none'}).\n")
        L.append("2. Largest eps at which the best private method is significantly better than Public-Only: "
                 + "; ".join(f"{s}: {v['largest']} (all such eps: {', '.join(v['all_eps']) or 'none'})" for s, v in a["rule2"].items())
                 + ".\n")
        L.append("3. Fine-tuned and pooled methods against both parents (F7, Holm over 144 tests):\n")
        L.append(md(pd.DataFrame(a["rule3"])) + "\n")
        L.append("4. MIC-Public at 15%, 5% and 1% against MIC-DP (F8, Holm over 36 tests):\n")
        L.append(md(pd.DataFrame(a["rule4"])) + "\n")
    L.append("5. Shifted public split (EXPLORATORY; unadjusted Wilcoxon p, no family):\n")
    rows = []
    for ds in pb["datasets"]:
        for e in pb["eps_grid"]:
            rows.append({"dataset": DS_TXT[ds], "eps": e, **{name_of(k): pm_md(S.auc(ds, k, e)) for k in SHIFT_COLS}})
    L.append(md(pd.DataFrame(rows)) + "\n")
    L.append(fmt_tests(fsh) + "\n")
    L.append("## Family F7 (all tests)\n")
    L.append(fmt_tests(f7) + "\n")
    L.append("## Family F8 (all tests)\n")
    L.append(fmt_tests(f8) + "\n")
    L.append("## Runtime\n")
    L.append((out / "runtime_public.txt").read_text() + "\n")
    L.append("## Deviations\n")
    dev = out / "deviations_public.md"
    L.append(dev.read_text() if dev.exists() else "None.\n")
    (out / "summary.md").write_text("\n".join(L))


def main():
    cfg = D.load_config()
    L, new, out = load(cfg)
    S = Store(L, cfg["public_baselines"]["n_trials"])
    f6, f7, f8, fsh = families(S, cfg)
    G = gap_table(S, cfg)
    keys = ["public_only@p15", "public_only@p5", "public_only@p1", "public_only@shift5", "mic_dp", "mic_public@p15",
            "mic_public@p5", "mic_public@p1", "mic_public@shift5", "ft_uniform@p15", "ft_uniform@p5", "ft_mic_dp@p15",
            "ft_mic_dp@p5", "pooled_uniform@p15", "pooled_uniform@p5"]
    UR = ur_table(S, cfg, keys)
    stats = pd.concat([f6, f7, f8, fsh], ignore_index=True)
    stats.to_csv(out / "stats_public.csv", index=False)
    G.to_csv(out / "gap_public.csv", index=False)
    UR.to_csv(out / "ur_public.csv", index=False)
    A = answers(f6, f7, f8, cfg)
    with open(out / "answers_public.json", "w") as f:
        json.dump(A, f, indent=1, default=str)
    write_tables(S, cfg, out)
    figure(S, cfg, out)
    write_summary(S, cfg, out, f6, f7, f8, fsh, G, UR, A)
    print("report written to", out)


if __name__ == "__main__":
    main()
