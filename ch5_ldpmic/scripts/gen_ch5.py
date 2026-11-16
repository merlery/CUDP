#!/usr/bin/env python
"""Generate the numbers, tables and figures of Chapter 5 (LDP-MIC) from the recorded runs.

    python scripts/gen_ch5.py          (after `make aggregate figures tables`)

Outputs (in thesis_tex/):
  ch5_numbers.tex        \\newcommand macros for the numbers of the chapter
  ch5_tables.tex         all tables (booktabs), labels tab:ch5:*
  figures/pu_ch5.pdf     privacy-utility curves of Experiment 1 (rerun with rescaled inputs)
  figures/selection_ch5.pdf, figures/mask_ch5.pdf
Sources: results/summary/*.csv, results/raw/*.json (pilots), results/ch5/draft3/*.csv
(follow-up runs), results/ch5/central/central_runs.csv (Experiment 5), ldpmic.privacy.
Every required value is checked; a missing one raises an error.
"""
from __future__ import annotations
import csv, glob, json, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from ldpmic.privacy import sigma0_analytic  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

OUT = ROOT / "thesis_tex"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
S = ROOT / "results" / "summary"
R = ROOT / "results" / "raw"
macros: dict[str, str] = {}


_DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def M(name: str, value) -> None:
    """Register a macro (LaTeX command names: letters only; digits are spelled out)."""
    name = "".join(_DIGITS.get(c, c) for c in name).replace("_", "").replace(".", "")
    assert re.fullmatch(r"[A-Za-z]+", name), name
    macros[name] = value if isinstance(value, str) else f"{value}"


def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"
def pm(m, s): return f"{m:.3f} $\\pm$ {s:.3f}"


def final(ds):
    df = pd.read_csv(S / f"{ds}_final.csv")
    return df[~df.lr_tuning]


def cell(df, config, eps):
    sub = df[df.config == config]
    sub = sub[sub.eps.isna()] if eps is None else sub[(sub.eps - eps).abs() < 1e-9]
    if len(sub) != 1:
        raise SystemExit(f"missing cell {config} eps={eps} ({len(sub)} rows)")
    return sub.iloc[0]


tables: list[str] = []


def table(cols, header, rows, caption, label):
    body = [r"\begin{table}[!htbp]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\small",
            r"\begin{tabular}{" + cols + "}", r"\toprule", header + r" \\", r"\midrule"]
    body += [r + r" \\" for r in rows]
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    tables.append("\n".join(body))


# --------------------------------------------------------------------------- #
# 1. noise analysis: eps at which sigma0(eps2) sqrt(d) reaches 1 and 0.5
# --------------------------------------------------------------------------- #
dims = {"adult": 105, "bank": 65, "mnist": 784, "fmnist": 784}
rows = []
for name, d in [("Adult", 105), ("Bank", 65), ("MNIST / F-MNIST", 784), ("$k = 20$ selected features", 20), ("$k = 10$ selected features", 10)]:
    e1 = brentq(lambda e: sigma0_analytic(0.8 * e, 5e-6) * math.sqrt(d) - 1.0, 0.05, 5000)
    e5 = brentq(lambda e: sigma0_analytic(0.8 * e, 5e-6) * math.sqrt(d) - 0.5, 0.05, 5000)
    r1 = sigma0_analytic(0.8, 5e-6) * math.sqrt(d); r10 = sigma0_analytic(8.0, 5e-6) * math.sqrt(d)
    rows.append(f"{name} & {d} & {r1:.1f} & {r10:.2f} & {e1:.0f} & {e5:.0f}")
    key = {"Adult": "Adult", "Bank": "Bank", "MNIST / F-MNIST": "Mnist", "$k = 20$ selected features": "Ktwenty", "$k = 10$ selected features": "Kten"}[name]
    M("epsOne" + key, f"{e1:.0f}"); M("epsHalf" + key, f"{e5:.0f}"); M("ratioAtOne" + key, f"{r1:.1f}"); M("ratioAtTen" + key, f"{r10:.2f}")
table("lrrrrr", r"release & $d$ & $\sigma_j/\Delta_j$ at $\eps=1$ & at $\eps=10$ & $\eps$ for ratio $1$ & $\eps$ for ratio $0.5$", rows,
      r"Per-feature noise of the one-shot Gaussian release with uniform shares, $\sigma_j/\Delta_j = \sigma_0(0.8\eps, \delta/2)\sqrt{d}$ "
      r"($\delta = 10^{-5}$, analytic Gaussian mechanism), and the total budget $\eps$ at which the noise standard deviation equals one "
      r"feature range and half a feature range. Computed from the calibration code of the artifact.", "tab:ch5:law")

# --------------------------------------------------------------------------- #
# 2. main sweeps at eps <= 10 (recorded multi-seed sweeps)
# --------------------------------------------------------------------------- #
pre = {"adult": "adult_occ", "bank": "bank_occ", "mnist": "mnist_a0.5", "fmnist": "fmnist_a0.5"}
names = {"adult": "Adult (occupation)", "bank": "Bank (job)", "mnist": "MNIST (Dir.\\ 0.5)", "fmnist": "F-MNIST (Dir.\\ 0.5)"}
methods = [("iso", "Uniform release (iso)"), ("ldpmic_global", "\\LDPMIC{} (global)"), ("ldpmic_oracle", "Oracle scores (non-private)"),
           ("local_dpsgd", "Local DP-SGD")]
eps_main = [0.5, 1.0, 2.0, 5.0, 10.0]
rows = []
for ds in ("adult", "bank", "mnist", "fmnist"):
    df = final(ds); p = pre[ds]
    nodp = cell(df, f"{p}_nodp", None)
    M(f"nodp{ds}", pm(nodp.accuracy_mean, nodp.accuracy_std)); M(f"nodpBal{ds}", f3(nodp.balanced_accuracy_mean))
    M(f"seeds{ds}", str(int(nodp.n_seeds)))
    rows.append((r"\midrule" + "\n" if rows else "") + r"\multicolumn{6}{l}{\textit{" + names[ds] + f"; no DP: {pm(nodp.accuracy_mean, nodp.accuracy_std)}; {int(nodp.n_seeds)} seeds" + "}}")
    for suf, title in methods:
        cells = []
        for e in eps_main:
            sub = df[(df.config == f"{p}_{suf}") & ((df.eps - e).abs() < 1e-9)]
            if len(sub) == 0 and suf == "ldpmic_oracle":      # the oracle control was recorded at eps 1 and 5 only on images
                cells.append("--"); continue
            r = cell(df, f"{p}_{suf}", e)
            cells.append(pm(r.accuracy_mean, r.accuracy_std) + (f" ({int(r.n_diverged)}d)" if r.n_diverged > 0 else ""))
            if e in (1.0, 5.0, 10.0):
                M(f"{suf.replace('_', '')}{ds}Eps{ {1.0: 'One', 5.0: 'Five', 10.0: 'Ten'}[e] }", pm(r.accuracy_mean, r.accuracy_std))
                M(f"{suf.replace('_', '')}{ds}Eps{ {1.0: 'One', 5.0: 'Five', 10.0: 'Ten'}[e] }Bal", f3(r.balanced_accuracy_mean))
        rows.append(title + " & " + " & ".join(cells))
table("lrrrrr", r"method & " + " & ".join(f"$\\eps={e:g}$" for e in eps_main), rows,
      r"Main sweeps of the independent implementation at $\eps \le 10$: final test accuracy (mean $\pm$ std over seeds; ``$k$d'': $k$ seeds diverged). "
      r"Uniform release, \LDPMIC{} with the global allocation, the same allocation from non-private oracle scores, and local DP-SGD, all under the "
      r"same record-level $(\eps, 10^{-5})$ guarantee against the server. Tabular data: 50 clients, $q = 0.2$, $T = 200$, 5 seeds; images: 100 clients, $q = 0.3$, $T = 100$, 3 seeds; ``--'': not recorded (the oracle control was run at $\eps \in \{1, 5\}$ on the image datasets).",
      "tab:ch5:main-first-runs")
# majority rates of the test sets (results/summary/majority_rates.csv, written by scripts/majority_rates.py)
MAJ = pd.read_csv(S / "majority_rates.csv").set_index("dataset").majority_rate
for ds in ("adult", "bank", "mnist", "fmnist"):
    M(f"majority{ds}", f3(MAJ[ds]))

# --------------------------------------------------------------------------- #
# 3. pilots v3 / v4 (seed 0): allocation in the gradient domain, exploratory
# --------------------------------------------------------------------------- #
def prov(pattern):
    fs = sorted(glob.glob(str(R / pattern)))
    return [json.load(open(f)) for f in fs]


rows = []
for ds, label, ks in (("adult_occ", "Adult", (10, 20, 40)), ("mnist_a0.5", "MNIST", (100, 200, 400))):
    for e in (1, 5):
        base = prov(f"{ds}_v3_local_dpsgd_eps{e}_s0_*.json")[0]["final"]["accuracy"]
        cells = [f3(base)]
        for k in ks:
            j = prov(f"{ds}_v3_fs_dpsgd_k{k}_eps{e}_s0_*.json")[0]
            ov = [n for n in j["privacy"]["notes"] if "overlap" in n][0].split(": ")[1]
            cells.append(f"{j['final']['accuracy']:.3f} ({ov})")
            M(f"vthree{label}K{k}Eps{'One' if e == 1 else 'Five'}", f3(j["final"]["accuracy"]))
        M(f"vthree{label}FullEps{'One' if e == 1 else 'Five'}", f3(base))
        if e == 1:
            for k in ks:
                o = prov(f"{ds}_v3_fs_oracle_k{k}_eps1_s0_*.json")
                if o:
                    M(f"vthree{label}OracleK{k}", f3(o[0]["final"]["accuracy"]))
        rows.append(f"{label} & {e} & " + " & ".join(cells))
    j = prov(f"{ds}_v3_fs_dpsgd_k{ks[0]}_eps1_s0_*.json")[0]["privacy"]
    M(f"scoreNoise{label}", f"{j['sigma_s']:.3f}"); M(f"scoreMax{label}", f"{max(j['score_raw_global']):.3f}")
table("llrrrr", r"dataset & $\eps$ & full-input DP-SGD & $k$ small & $k$ medium & $k$ large", rows,
      r"Exploratory pilot (seed 0): local DP-SGD on all features versus private top-$k$ selection followed by local DP-SGD on the $k$ selected features "
      r"(Adult: $k \in \{10, 20, 40\}$; MNIST: $k \in \{100, 200, 400\}$); in parentheses the overlap of the private top-$k$ set with the non-private top-$k$ set. "
      r"Selection spends $0.2\eps$, DP-SGD $0.8\eps$.", "tab:ch5:pilotA")

rows = []
for ds, label in (("adult_occ", "Adult"), ("mnist_a0.5", "MNIST")):
    base = prov(f"{ds}_v3_local_dpsgd_eps1_s0_*.json")[0]["final"]["accuracy"]
    cells = [f3(base)]
    for arm in ("b2", "b5", "b20", "b50", "b2_pub"):
        j = prov(f"{ds}_v4_aniso_oracle_{arm}_eps1_s0_*.json")[0]
        cells.append(f3(j["final"]["accuracy"]))
        M(f"vfour{label}{arm.replace('_', '')}", f3(j["final"]["accuracy"]))
    rows.append(f"{label} & " + " & ".join(cells))
table("lrrrrrr", r"dataset & isotropic DP-SGD & $\beta=2$ & $\beta=5$ & $\beta=20$ & $\beta=50$ & $\beta=2$, public scores", rows,
      r"Exploratory pilot (seed 0, $\eps = 1$): score-weighted anisotropic DP-SGD with \emph{non-private} scores (the first-layer gradient of input $j$ "
      r"scaled by $\sqrt{d\,p_j}$ before clipping, isotropic noise, unscaled afterwards). All arms except the last spend $0.2\eps$ on the (unused) score release; "
      r"the last gives DP-SGD the full budget.", "tab:ch5:pilotB")

# --------------------------------------------------------------------------- #
# 4. selection sweep (5 seeds): copy the generated tables with chapter labels
# --------------------------------------------------------------------------- #
for name, newlabel in (("tab_selection", "tab:ch5:selection"), ("tab_selection_bal", "tab:ch5:selbal"),
                       ("tab_selection_ablation", "tab:ch5:ablation"), ("tab_selection_exposure", "tab:ch5:exposure")):
    t = (ROOT / "tables" / f"{name}.tex").read_text().split("\n", 1)[1]
    t = re.sub(r"\\label\{[^}]*\}", r"\\label{" + newlabel + "}", t)
    t = t.replace(r"\begin{table*}", r"\begin{table}").replace(r"\end{table*}", r"\end{table}")
    t = t.replace(r"\begin{tabular}", r"\resizebox{\linewidth}{!}{\begin{tabular}").replace(r"\end{tabular}", r"\end{tabular}}")
    tables.append(t)
# macros for the prose (Adult occupation, selection sweep)
da = final("adult"); dbk = final("bank")
for suf, key in (("iso", "Iso"), ("sel_random_k10", "RandTen"), ("sel_random_k20", "RandTwenty"), ("sel_mi_k10", "MiTen"), ("sel_mi_k20", "MiTwenty"),
                 ("sel_mi_oracle_k10", "OracleTen"), ("sel_mi_oracle_k20", "OracleTwenty"), ("local_dpsgd", "Dpsgd")):
    for e, ek in ((50.0, "Fifty"), (100.0, "Hundred"), (200.0, "TwoHundred")):
        r = cell(da, f"adult_occ_v7_{suf}", e); M(f"sel{key}{ek}", pm(r.accuracy_mean, r.accuracy_std)); M(f"sel{key}{ek}Bal", pm(r.balanced_accuracy_mean, r.balanced_accuracy_std))
        r = cell(dbk, f"bank_occ_v7_{suf}", e); M(f"selBank{key}{ek}", pm(r.accuracy_mean, r.accuracy_std))
        r = cell(da, f"adult_dir0.5_v7_{suf}", e) if suf != "local_dpsgd" else None
        if r is not None: M(f"selDir{key}{ek}", pm(r.accuracy_mean, r.accuracy_std))
for suf, key in (("sel_mi_cons", "Cons"), ("sel_pearson", "Pearson"), ("sel_mi_exp", "Exp")):
    for k in (10, 20):
        r = cell(da, f"adult_occ_v7_{suf}_k{k}", 200.0); M(f"sel{key}K{k}", pm(r.accuracy_mean, r.accuracy_std))
M("selBankMajority", macros["majoritybank"])
# overlaps of the private selection with the oracle set (v7, Adult, eps 200, 5 seeds)
for k in (10, 20):
    ovs = []
    for j in prov(f"adult_occ_v7_sel_mi_k{k}_eps200_s*_*.json"):
        ovs.append(int([n for n in j["privacy"]["notes"] if "overlap" in n][0].split(": ")[1].split("/")[0]))
    M(f"overlapK{k}", f"{min(ovs)}--{max(ovs)}")
# attack numbers (Adult, eps 200, sex)
att = pd.read_csv(S / "attribute_inference.csv")
for cfg, key in (("adult_occ_v7_iso", "Iso"), ("adult_occ_v7_sel_random_k10", "Rand"), ("adult_occ_v7_sel_mi_k10", "MiTen"), ("adult_occ_v7_sel_mi_k20", "MiTwenty")):
    r = att[(att.config == cfg) & (att.attribute == "sex") & ((att.eps - 200.0).abs() < 1e-9)].iloc[0]
    M(f"attSex{key}", f3(r.balanced_accuracy_mean)); M(f"attSexRaw{key}", f3(r.raw_balanced_accuracy_mean))
M("attSexMajority", f3(r.majority_accuracy_mean))

# --------------------------------------------------------------------------- #
# 5. input masking for local DP-SGD (v8, 3 seeds): table, macros, figure
# --------------------------------------------------------------------------- #
arms = [("local_dpsgd", "Full input (784 pixels)", "Full"), ("fs_mi_k400", "MI-private mask, $k=400$", "Mi"),
        ("fs_random_k400", "Random mask, $k=400$", "Rand"), ("fs_variance_k400", "Variance mask, $k=400$ (public)", "Var")]
rows = []; fig = {}
for ds, dl in (("mnist", "MNIST"), ("fmnist", "F-MNIST")):
    for e in (1, 5):
        rows.append((r"\midrule" + "\n" if rows else "") + r"\multicolumn{7}{l}{\textit{" + f"{dl}, $\\eps = {e}$" + "}}")
        for tag, title, key in arms:
            accs, bals, kb, ct, du, ov = [], [], [], [], [], []
            for f in sorted(glob.glob(str(R / f"{ds}_a0.5_v8_{tag}_eps{e}_s*_*.json"))):
                j = json.load(open(f)); rr = list(csv.DictReader(open(f[:-5] + ".csv")))
                accs.append(j["final"]["accuracy"]); bals.append(j["final"]["balanced_accuracy"]); du.append(j["duration_s"])
                kb.append(np.mean([float(x["bytes_per_client"]) for x in rr]) / 1000); ct.append(np.mean([float(x["client_time_mean_s"]) for x in rr]))
                o = [n for n in j["privacy"]["notes"] if "overlap" in n]
                if o: ov.append(int(o[0].split(": ")[1].split("/")[0]))
            if len(accs) != 3: raise SystemExit(f"expected 3 seeds for {ds} {tag} eps {e}, got {len(accs)}")
            m, s = np.mean(accs), np.std(accs, ddof=1)
            rows.append(f"{title} & {pm(m, s)} & {np.mean(bals):.3f} & {np.mean(kb):.0f} & {np.mean(ct):.2f} & {np.mean(du):.0f} & " + (f"{min(ov)}--{max(ov)}" if ov else "--"))
            K = f"mask{dl.replace('-', '')}Eps{'One' if e == 1 else 'Five'}{key}"
            M(K, pm(m, s)); M(K + "Mean", f3(m)); M(K + "Std", f3(s)); M(K + "Kb", f"{np.mean(kb):.0f}"); M(K + "Ct", f2(np.mean(ct))); M(K + "Dur", f"{np.mean(du):.0f}")
            if ov: M(K + "Ov", f"{min(ov)}--{max(ov)}")
            fig[(dl, e, key)] = (m, s, np.mean(kb), np.mean(ct))
table("lrrrrrr", r"input & accuracy & bal.\ acc. & kB / client / round & client s / round & run s & overlap", rows,
      r"Input masking for local DP-SGD ($k = 400$ of 784 pixels, 3 seeds, Dirichlet(0.5), 100 clients, $q = 0.3$, $T = 100$): final test accuracy "
      r"(mean $\pm$ std), balanced accuracy, mean bytes sent per client per round, mean client computation per round, run duration, and the overlap of the "
      r"mask with the non-private top-400 MI set. The MI mask spends $0.2\eps$ on the score release and $0.8\eps$ on DP-SGD; the random and variance masks "
      r"spend no score budget and use the same $0.8\eps$ for DP-SGD; the full-input model uses $\eps$. The variance mask is computed on the training "
      r"records as a stand-in for a public statistic and is not differentially private.", "tab:ch5:mask")
# derived macros
M("maskBytesSaving", f"{100 * (1 - fig[('MNIST', 1, 'Mi')][2] / fig[('MNIST', 1, 'Full')][2]):.0f}")
cts = [100 * (1 - fig[(d, e, 'Mi')][3] / fig[(d, e, 'Full')][3]) for d in ("MNIST", "F-MNIST") for e in (1, 5)]
M("maskComputeSavingMin", f"{min(cts):.0f}"); M("maskComputeSavingMax", f"{max(cts):.0f}")
gaps = {(d, e): fig[(d, e, 'Full')][0] - fig[(d, e, 'Mi')][0] for d in ("MNIST", "F-MNIST") for e in (1, 5)}
M("maskGapMnistFive", f3(gaps[("MNIST", 5)])); M("maskStdMnistFive", f3(fig[("MNIST", 5, "Full")][1]))
# figure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
fg, axes = plt.subplots(1, 4, figsize=(11, 2.6), sharey=False)
cols = {"Full": "#2ca02c", "Mi": "#d62728", "Rand": "#7f7f7f", "Var": "#1f77b4"}
labels = {"Full": "full input", "Mi": "MI-private mask", "Rand": "random mask", "Var": "variance mask (public)"}
for ax, (dl, e) in zip(axes, [("MNIST", 1), ("MNIST", 5), ("F-MNIST", 1), ("F-MNIST", 5)]):
    for i, key in enumerate(("Full", "Mi", "Rand", "Var")):
        m, s, kb, ct = fig[(dl, e, key)]
        jitter = {"Full": 0, "Mi": -25, "Rand": 0, "Var": 25}[key]     # masks share the same x: offset for legibility
        ax.errorbar(kb + jitter, m, yerr=s, fmt="o", color=cols[key], capsize=3, label=labels[key])
    ax.set_xlabel("kB per client per round"); ax.set_ylabel("test accuracy"); ax.set_title(f"{dl}, $\\varepsilon = {e}$", fontsize=9)
    ax.set_xlim(400, 1100)
axes[0].legend(fontsize=7, frameon=False, loc="lower right")
fg.tight_layout(); fg.savefig(OUT / "figures" / "mask_ch5.pdf", bbox_inches="tight"); fg.savefig(OUT / "figures" / "mask_ch5.png", dpi=200, bbox_inches="tight")

# --------------------------------------------------------------------------- #
# counts and provenance
# --------------------------------------------------------------------------- #
allj = glob.glob(str(R / "*.json"))
M("nRuns", str(len(allj))); M("nAttackRuns", str(len(glob.glob(str(R / "attack_*.json")))))
M("nSelectionRuns", str(len([f for f in glob.glob(str(R / "*_v7_*.json")) if "attack_" not in f]))); M("nMaskRuns", str(len(glob.glob(str(R / "*_v8_*.json")))))

import shutil
shutil.copy(ROOT / "figures" / "fig_selection.pdf", OUT / "figures" / "selection_ch5.pdf")
exec(open(ROOT / "scripts" / "gen_ch5_followup.py").read())
(OUT / "ch5_numbers.tex").write_text("% generated by scripts/gen_ch5.py from the recorded runs -- do not edit\n" +
                                     "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
(OUT / "ch5_tables.tex").write_text("% generated by scripts/gen_ch5.py from the recorded runs -- do not edit\n\n" + "\n".join(tables))
print(f"{len(macros)} macros, {len(tables)} tables written to {OUT}")
