#!/usr/bin/env python
"""Generate every paper figure from results/summary/*.csv.

    python scripts/make_figures.py [--out figures]

Vector PDF plus a 300-dpi PNG fallback per figure.  One colour/marker per
method everywhere (paper_common.METHODS).  Log-scale eps axes, shaded +-1
std bands, axis labels with units, no titles inside the figures (panel
letters only; captions live in LaTeX).  Any missing required cell raises
``MissingCell`` naming (dataset, method, eps).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import (DATASETS, DELTA, EPS_GRID, METHODS, PU_CONFIG_SUFFIX, PU_EPS, PU_METHODS, SEL_DATASETS,  # noqa: E402
                          SEL_EPS_GRID, MissingCell, cell, load_csv, load_final, method_of)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 100, "savefig.dpi": 300, "lines.linewidth": 1.2, "lines.markersize": 4,
})
EPS_LABEL = r"privacy budget $\varepsilon$  ($\delta = 10^{-5}$)"
ACC_LABEL = "test accuracy (fraction correct)"
PANEL = "abcdefgh"


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def panel_label(ax, i: int, text: str) -> None:
    # panel letter + dataset tag just above the axes (never over the data; not a title)
    ax.text(0.0, 1.02, f"({PANEL[i]}) {text}", transform=ax.transAxes, ha="left", va="bottom", fontsize=8)


def style(m: str):
    s = METHODS[m]
    return dict(color=s["color"], marker=s["marker"] or None, ls=s["ls"], label=s["name"])


# --------------------------------------------------------------------------- #
# privacy-utility curves
# --------------------------------------------------------------------------- #

def majority_rate(ds: str) -> float:
    """Majority-class rate of the test labels (the accuracy of a constant predictor)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_sweep as rs  # noqa: E402
    y = np.asarray(rs.get_dataset(ds).y_test)
    return float(np.bincount(y).max() / len(y))


def privacy_utility(datasets, out: Path, name: str, methods=PU_METHODS, eps_grid=tuple(EPS_GRID)):
    """Accuracy against eps for every one-shot release and local DP-SGD.  Releases that coincide (all at the
    majority rate) are drawn with hollow markers of decreasing size so that every one stays visible."""
    EPS_GRID = list(eps_grid)  # noqa: N806  (a subset is used by `make reproduce-small`)
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 2.7), sharey=False)
    sizes = np.linspace(8, 3, len(methods))
    for i, (ax, ds) in enumerate(zip(np.atleast_1d(axes), datasets)):
        df = load_final(ds)
        pre = DATASETS[ds]["prefix"]
        nodp = cell(df, ds, f"{pre}_nodp", None)
        ax.axhline(nodp.accuracy_mean, **style("nodp"))
        ax.axhspan(nodp.accuracy_mean - nodp.accuracy_std, nodp.accuracy_mean + nodp.accuracy_std,
                   color=METHODS["nodp"]["color"], alpha=0.12, lw=0)
        ax.axhline(majority_rate(ds), color="#999999", ls=":", lw=1.0)
        for j, m in enumerate(methods):
            grid = [e for e in PU_EPS.get((ds, m), EPS_GRID) if e in EPS_GRID]
            rows = [cell(df, ds, f"{pre}_{PU_CONFIG_SUFFIX.get(m, m)}", e) for e in grid]
            mean = np.array([r.accuracy_mean for r in rows]); std = np.array([r.accuracy_std for r in rows])
            st = style(m)
            ax.plot(grid, mean, color=st["color"], ls=st["ls"] if len(grid) > 2 else "none", marker=st["marker"],
                    markersize=sizes[j], markerfacecolor="none", markeredgewidth=1.0, zorder=3 + j)
            if len(grid) > 2:
                ax.fill_between(grid, mean - std, mean + std, color=METHODS[m]["color"], alpha=0.10, lw=0)
        ax.set_xscale("log"); ax.set_xticks(EPS_GRID); ax.set_xticklabels([f"{e:g}" for e in EPS_GRID])
        ax.set_xlabel(EPS_LABEL); ax.set_ylabel(ACC_LABEL)
        panel_label(ax, i, DATASETS[ds]["name"])
    handles = [Line2D([], [], **style("nodp")), Line2D([], [], color="#999999", ls=":", label="majority class")]
    for j, m in enumerate(methods):
        st = style(m)
        handles.append(Line2D([], [], color=st["color"], ls=st["ls"], marker=st["marker"], markersize=sizes[j],
                              markerfacecolor="none", label=st["label"]))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    save(fig, out, name)


# --------------------------------------------------------------------------- #
# convergence
# --------------------------------------------------------------------------- #

def convergence(out: Path, datasets=("adult", "mnist"), eps_list=(1.0, 5.0),
                methods=("iso", "ldpmic_global", "ldpmic_per_client", "local_dpsgd")):
    fig, axes = plt.subplots(len(datasets), len(eps_list), figsize=(3.3 * len(eps_list), 2.4 * len(datasets)), squeeze=False)
    k = 0
    for r, ds in enumerate(datasets):
        curves = load_csv(f"{ds}_curves.csv")
        curves = curves[~curves.lr_tuning]
        pre = DATASETS[ds]["prefix"]
        for c, eps in enumerate(eps_list):
            ax = axes[r][c]
            nd = curves[(curves.config == f"{pre}_nodp")]
            if nd.empty:
                raise MissingCell(f"missing curve: dataset={ds}, method=nodp")
            nd = nd.sort_values("round")
            ax.plot(nd["round"] + 1, nd.accuracy_mean, **style("nodp"))
            for m in methods:
                cv = curves[(curves.config == f"{pre}_{m}") & ((curves.eps - eps).abs() < 1e-9)].sort_values("round")
                if cv.empty:
                    raise MissingCell(f"missing curve: dataset={ds}, method={m}, eps={eps}")
                ax.plot(cv["round"] + 1, cv.accuracy_mean, **{**style(m), "marker": None})
                ax.fill_between(cv["round"] + 1, cv.accuracy_mean - cv.accuracy_std.fillna(0),
                                cv.accuracy_mean + cv.accuracy_std.fillna(0), color=METHODS[m]["color"], alpha=0.15, lw=0)
            ax.set_xlabel("communication round"); ax.set_ylabel(ACC_LABEL)
            panel_label(ax, k, f"{DATASETS[ds]['name']}, $\\varepsilon={eps:g}$"); k += 1
    handles = [Line2D([], [], **{**style(m), "marker": None}) for m in ("nodp",) + tuple(methods)]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_convergence")


# --------------------------------------------------------------------------- #
# heterogeneity
# --------------------------------------------------------------------------- #

def heterogeneity(out: Path, datasets=("mnist", "fmnist"), alphas=(0.1, 0.5, 1.0), eps_list=(1.0, 5.0),
                  methods=("iso", "ldpmic_global")):
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets), 2.6))
    width = 0.8 / (len(methods) * len(eps_list))
    hatches = {eps_list[0]: "", eps_list[1]: "//"}
    for i, (ax, ds) in enumerate(zip(np.atleast_1d(axes), datasets)):
        df = load_final(ds)
        x = np.arange(len(alphas))
        for mi, m in enumerate(methods):
            for ei, eps in enumerate(eps_list):
                means, stds = [], []
                for a in alphas:
                    cfg = f"{ds}_a{a}_{m}"   # configs are named with the literal alpha string (0.1, 0.5, 1.0)
                    r = cell(df, ds, cfg, eps)
                    means.append(r.accuracy_mean); stds.append(r.accuracy_std)
                off = (mi * len(eps_list) + ei - (len(methods) * len(eps_list) - 1) / 2) * width
                ax.bar(x + off, means, width, yerr=stds, capsize=2, color=METHODS[m]["color"], hatch=hatches[eps],
                       edgecolor="black", linewidth=0.4, alpha=0.9 if ei == 0 else 0.6)
        ax.set_xticks(x); ax.set_xticklabels([f"{a:g}" for a in alphas])
        ax.set_xlabel(r"label-skew concentration $\alpha$ (Dirichlet)"); ax.set_ylabel(ACC_LABEL)
        panel_label(ax, i, DATASETS[ds]["name"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHODS[m]["color"], label=METHODS[m]["name"]) for m in methods]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=hatches[e],
                              label=f"$\\varepsilon={e:g}$") for e in eps_list]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_heterogeneity")


# --------------------------------------------------------------------------- #
# score-privatization decomposition
# --------------------------------------------------------------------------- #

def decomposition(out: Path, datasets=("adult", "bank", "mnist", "fmnist"), eps_list=(1.0, 5.0),
                  methods=("oracle", "ldpmic_global", "ldpmic_per_client", "iso")):
    fig, axes = plt.subplots(1, len(datasets), figsize=(2.6 * len(datasets), 2.6))
    width = 0.8 / len(methods)
    for i, (ax, ds) in enumerate(zip(axes, datasets)):
        df = load_final(ds)
        pre = DATASETS[ds]["prefix"]
        x = np.arange(len(eps_list))
        for mi, m in enumerate(methods):
            cfg = f"{pre}_ldpmic_oracle" if m == "oracle" else f"{pre}_{m}"
            rows = [cell(df, ds, cfg, e) for e in eps_list]
            off = (mi - (len(methods) - 1) / 2) * width
            ax.bar(x + off, [r.accuracy_mean for r in rows], width, yerr=[r.accuracy_std for r in rows], capsize=2,
                   color=METHODS[m]["color"], edgecolor="black", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels([f"{e:g}" for e in eps_list])
        ax.set_xlabel(EPS_LABEL); ax.set_ylabel(ACC_LABEL if i == 0 else "")
        panel_label(ax, i, DATASETS[ds]["name"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHODS[m]["color"], label=METHODS[m]["name"]) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_decomposition")


# --------------------------------------------------------------------------- #
# attribute exposure
# --------------------------------------------------------------------------- #

def exposure(out: Path, partition: str = "group", attributes=("sex", "race"),
             configs=(("adult_occ_iso", "iso"), ("adult_occ_ldpmic_global", "ldpmic_global"), ("adult_occ_cap_sexrace", "cap"))):
    att = load_csv("attribute_inference.csv")
    att = att[(att.dataset == "adult") & (att.partition == partition)]
    fig, axes = plt.subplots(1, len(attributes), figsize=(3.3 * len(attributes), 2.6))
    for i, (ax, attr) in enumerate(zip(np.atleast_1d(axes), attributes)):
        sub = att[att.attribute == attr]
        majority = None
        for cfg, m in configs:
            rows = []
            for e in EPS_GRID:
                r = sub[(sub.config == cfg) & ((sub.eps - e).abs() < 1e-9)]
                if len(r) != 1:
                    raise MissingCell(f"missing attack cell: dataset=adult, method={m}, attribute={attr}, eps={e}")
                rows.append(r.iloc[0])
            mean = np.array([r.accuracy_mean for r in rows]); std = np.array([r.accuracy_std for r in rows])
            ax.plot(EPS_GRID, mean, **style(m))
            ax.fill_between(EPS_GRID, mean - std, mean + std, color=METHODS[m]["color"], alpha=0.15, lw=0)
            majority = rows[0].majority_accuracy_mean
            raw = rows[0].raw_accuracy_mean
        ax.axhline(majority, color="black", ls="--", lw=1, label="majority class")
        ax.axhline(raw, color="black", ls=":", lw=1, label="raw records (no noise)")
        ax.set_xscale("log"); ax.set_xticks(EPS_GRID); ax.set_xticklabels([f"{e:g}" for e in EPS_GRID])
        ax.set_xlabel(EPS_LABEL); ax.set_ylabel("attacker accuracy (fraction correct)")
        panel_label(ax, i, f"attribute: {attr}")
    handles = [Line2D([], [], **style(m)) for _, m in configs]
    handles += [Line2D([], [], color="black", ls="--", label="majority class"),
                Line2D([], [], color="black", ls=":", label="attacker on raw records")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_exposure")


# --------------------------------------------------------------------------- #
# per-feature allocation heatmap (Adult)
# --------------------------------------------------------------------------- #

ALLOC_NAMES = {"uniform": "uniform (iso)", "exp": "exp", "linear": "linear", "cap": "exp + cap"}


def allocation_adult(out: Path, eps: float = 1.0, allocations=("uniform", "exp", "linear", "cap")):
    al = load_csv("adult_allocation.csv")
    al = al[(al.eps - eps).abs() < 1e-9]
    feats = al[al.allocation == allocations[0]].sort_values("feature_index")
    if feats.empty:
        raise MissingCell(f"missing allocation rows: dataset=adult, method={allocations[0]}, eps={eps}")
    names = feats.feature.tolist()
    d = len(names)
    P = np.full((len(allocations), d), np.nan); S = np.full((len(allocations), d), np.nan)
    for i, a in enumerate(allocations):
        sub = al[al.allocation == a].sort_values("feature_index")
        if len(sub) != d:
            raise MissingCell(f"missing allocation rows: dataset=adult, method={a}, eps={eps} ({len(sub)} of {d})")
        P[i] = sub.p_mean.to_numpy(); S[i] = sub.sigma_scaled_mean.to_numpy()
    sensitive = feats.sensitive.to_numpy().astype(bool)
    # attribute group boundaries from "attr=value" names
    groups = [n.split("=")[0] for n in names]
    bounds, labels, start = [], [], 0
    for j in range(1, d + 1):
        if j == d or groups[j] != groups[start]:
            bounds.append((start, j)); labels.append(groups[start]); start = j
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True)
    for ax, M, lab in ((axes[0], P, r"share $p_j$"), (axes[1], S, r"noise std $\sigma_j / \Delta_j$ (scaled units)")):
        im = ax.imshow(np.log10(M), aspect="auto", cmap="viridis", interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
        cb.set_label(r"$\log_{10}$ " + lab)
        ax.set_yticks(range(len(allocations)))
        ax.set_yticklabels([ALLOC_NAMES[a] for a in allocations])
        for s, e in bounds[1:]:
            ax.axvline(s - 0.5, color="white", lw=0.4)
        for j in np.where(sensitive)[0]:
            ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(allocations), fill=False, edgecolor="red", lw=1.2))
    axes[1].set_xticks([(s + e - 1) / 2 for s, e in bounds]); axes[1].set_xticklabels(labels, rotation=90, fontsize=6)
    axes[1].set_xlabel(f"feature $j$ (grouped by attribute; red boxes = sensitive columns); $\\varepsilon = {eps:g}$")
    fig.tight_layout()
    save(fig, out, "fig_allocation_adult")


# --------------------------------------------------------------------------- #
# selection experiment (D38/D39): private top-k selection + uniform one-shot release
# --------------------------------------------------------------------------- #

def selection(out: Path, name: str = "fig_selection", keys=("adult_occ", "bank_occ", "adult_dir0.5"),
              methods=("iso", "sel_random_k10", "sel_mi_k10", "sel_mi_k20", "sel_mi_oracle_k20", "local_dpsgd")):
    """Accuracy vs eps at the budgets where one-shot record perturbation acts
    (eps 20-200), for the uniform all-feature release (iso), the random-k
    control, the private MI selection (k = 10, 20), its non-private oracle
    and local DP-SGD.  No-DP accuracy (recorded main sweep) as a reference."""
    fig, axes = plt.subplots(1, len(keys), figsize=(3.3 * len(keys), 2.6), sharey=False)
    for i, (ax, key) in enumerate(zip(np.atleast_1d(axes), keys)):
        meta = SEL_DATASETS[key]
        ds, pre = meta["dataset"], meta["prefix"]
        df = load_final(ds)
        nodp = cell(df, ds, meta["nodp"], None)
        ax.axhline(nodp.accuracy_mean, **style("nodp"))
        for m in methods:
            if m not in meta["rows"]:       # not recorded for this dataset by design (D39)
                continue
            rows = [cell(df, ds, f"{pre}_{m}", e) for e in SEL_EPS_GRID]
            mean = np.array([r.accuracy_mean for r in rows]); std = np.array([r.accuracy_std for r in rows])
            ax.plot(SEL_EPS_GRID, mean, **style(m))
            ax.fill_between(SEL_EPS_GRID, mean - std, mean + std, color=METHODS[m]["color"], alpha=0.15, lw=0)
        ax.set_xscale("log"); ax.set_xticks(SEL_EPS_GRID); ax.set_xticklabels([f"{e:g}" for e in SEL_EPS_GRID])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(EPS_LABEL); ax.set_ylabel(ACC_LABEL)
        panel_label(ax, i, meta["name"])
    handles = [Line2D([], [], **style(m)) for m in ("nodp",) + tuple(methods)]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.tight_layout()
    save(fig, out, name)


# --------------------------------------------------------------------------- #
# exposure study (D46-D49): attribute exposure against utility per release
# --------------------------------------------------------------------------- #

EXPOSURE_ARMS = (("adult_occ_v7_iso", "uniform, all features", "#1f77b4", "o"),
                 ("adult_occ_v9_alloc_b2", r"exp $\beta=2$", "#aec7e8", "s"), ("adult_occ_v9_alloc_b5", r"exp $\beta=5$", "#ff9896", "s"),
                 ("adult_occ_v9_alloc_b10", r"exp $\beta=10$", "#d62728", "s"), ("adult_occ_v9_alloc_b20", r"exp $\beta=20$", "#8b0000", "s"),
                 ("adult_occ_v9_alloc_b20_cap", r"exp $\beta=20$ + cap", "#8b0000", "P"),
                 ("adult_occ_v7_sel_mi_k10", "top-10 selection", "#9467bd", "^"), ("adult_occ_v9_sel_k10_cap", "top-10 selection + cap", "#9467bd", "P"),
                 ("adult_occ_v9_sel_k10_excl", "top-10, sex/race excluded", "#9467bd", "X"))


def exposure_channels(out: Path, name: str = "fig_exposure_channels", eps_list=(50.0, 100.0, 200.0), attr="sex"):
    """Attacker balanced accuracy on all released coordinates (y) against the
    model's test accuracy (x), one marker per release, one panel per eps."""
    att = load_csv("attribute_inference.csv")
    att = att[(att.dataset == "adult") & (att.partition == "group") & (att.attribute == attr)]
    fin = load_final("adult")
    fig, axes = plt.subplots(1, len(eps_list), figsize=(3.3 * len(eps_list), 2.8), sharey=True)
    for i, (ax, e) in enumerate(zip(np.atleast_1d(axes), eps_list)):
        for cfg, lab, col, mk in EXPOSURE_ARMS:
            r = cell(fin, "adult", cfg, e)
            a = att[(att.config == cfg) & ((att.eps - e).abs() < 1e-9)]
            if len(a) != 1:
                raise MissingCell(f"missing attack cell: dataset=adult, config={cfg}, attribute={attr}, eps={e}")
            a = a.iloc[0]
            ax.errorbar(r.accuracy_mean, a.all_balanced_accuracy_mean, xerr=r.accuracy_std, yerr=a.all_balanced_accuracy_std,
                        fmt=mk, color=col, ms=6, capsize=2, label=lab, lw=0.8)
        maj = att[(att.eps - e).abs() < 1e-9].majority_accuracy_mean.iloc[0]
        ax.axhline(0.5, color="#999999", lw=0.8, ls=":")
        ax.set_xlabel(ACC_LABEL); panel_label(ax, i, f"$\\varepsilon = {e:g}$")
        if i == 0:
            ax.set_ylabel(f"{attr} inference, balanced accuracy")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.25), frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, out, name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--summary", type=Path, default=None, help="summary directory (default results/summary)")
    ap.add_argument("--methods", nargs="*", default=None, help="methods for the privacy-utility figures")
    ap.add_argument("--eps", nargs="*", type=float, default=None, help="eps grid for the privacy-utility figures")
    ap.add_argument("--datasets-tabular", nargs="*", default=("adult", "bank"))
    ap.add_argument("--datasets-images", nargs="*", default=("mnist", "fmnist"))
    args = ap.parse_args(argv)
    if args.summary is not None:
        import paper_common
        paper_common.SUMMARY = args.summary
    pu_kw = {}
    if args.methods:
        pu_kw["methods"] = tuple(args.methods)
    if args.eps:
        pu_kw["eps_grid"] = tuple(args.eps)
    jobs = {  # Chapter 5: the selection figure; the privacy-utility figure is drawn by scripts/gen_ch5.py
        "fig_selection": lambda: selection(args.out),
    }
    for name, job in jobs.items():
        if args.only and name not in args.only:
            continue
        job()
    return 0


if __name__ == "__main__":
    sys.exit(main())
