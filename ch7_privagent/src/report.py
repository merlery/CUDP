"""Statistics, LaTeX table bodies, figures and summary.md for Chapter 7.

    python -m src.report            # reads results/ch7/
    python -m src.report --smoke    # reads results/ch7_smoke/
"""
from __future__ import annotations

import json
import subprocess
import sys
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import stats as S  # noqa: E402
from . import uldp as U  # noqa: E402
from .run import ROOT, load_config  # noqa: E402

BLUE, ORANGE, GREEN, PURPLE, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#7c5cc4", "#52514e"
STYLE = {   # same color per method in every figure
    "pa6":             dict(color=BLUE, hatch=None),
    "pa6_bayes":       dict(color=PURPLE, hatch=None),
    "lpla":            dict(color=GREEN, hatch=None),
    "lpla_bayes":      dict(color=GREEN, hatch="///"),
    "slider":          dict(color=ORANGE, hatch=None),
    "expert":          dict(color=GRAY, hatch=None),
    "uniform_default": dict(color=GRAY, hatch="///"),
    "oracle":          dict(color="#000000", hatch=None),
}
METHODS = ["oracle", "pa6", "pa6_bayes", "lpla", "lpla_bayes", "expert", "uniform_default", "slider"]
PLOT_METHODS = METHODS[1:]
SETTING_NAMES = {("adult", "linear"): "Adult, linear", ("adult", "exponential"): "Adult, exponential",
                 ("cdc", "linear"): "CDC-Diabetes, linear", ("cdc", "exponential"): "CDC-Diabetes, exponential"}
PROFILE_NAMES = {"privacy_focused": "Privacy-focused", "accuracy_focused": "Accuracy-focused",
                 "balanced": "Balanced", "resource_constrained": "Resource-constrained"}
WIDTH = 6.3

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def disp(cfg, m):
    return cfg["methods"][m]["display"]


def seed_of(cfg, *key):
    return [cfg["stats"]["bootstrap_seed"], zlib.crc32(repr(key).encode())]


def ci(cfg, values, reps, *key):
    return S.cluster_ci(values, reps, cfg["stats"]["bootstrap_resamples"], seed_of(cfg, *key), cfg["stats"]["ci"])


def f3(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.3f}"


def fci(t):
    return f"{f3(t[0])} [{f3(t[1])}, {f3(t[2])}]"


def savefig(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=220)
    plt.close(fig)


# ----------------------------------------------------------------------------- summaries

def weight_summary(cfg, users, population="uniform", analysis="main", methods=METHODS, extra=None):
    u = users[(users.population == population) & (users.analysis == analysis)]
    if extra is not None:
        u = extra(u)
    rows = []
    for m in methods:
        g = u[u.method == m]
        if not len(g):
            continue
        r = dict(method=m)
        for metric in ("cosine", "l1", "share_error"):
            r[metric] = ci(cfg, g[metric].to_numpy(), g.rep.to_numpy(), population, analysis, m, metric)
        r["n_prompts"] = float(g.n_prompts.mean())
        r["repair_rate"] = float(g.repair.mean())
        rows.append(r)
    return rows


def decision_summary(cfg, dec, population="uniform", analysis="main", methods=METHODS, settings=None, extra=None):
    d = dec[(dec.population == population) & (dec.analysis == analysis)]
    if extra is not None:
        d = extra(d)
    settings = settings or [tuple(s) for s in cfg["uldp"]["settings"]]
    rows = []
    for m in methods:
        for ds, lp in settings:
            g = d[(d.method == m) & (d.dataset == ds) & (d.lp_map == lp)]
            if not len(g):
                continue
            rows.append(dict(method=m, dataset=ds, lp_map=lp,
                             agree=ci(cfg, g.agree.to_numpy(float), g.rep.to_numpy(), population, analysis, m, ds, lp, "agree"),
                             loss=ci(cfg, g.loss.to_numpy(float), g.rep.to_numpy(), population, analysis, m, ds, lp, "loss"),
                             undefined=float(g.undefined.mean()), n=len(g)))
    return rows


# ----------------------------------------------------------------------------- tables

def write_tex(out: Path, stem: str, header: list[str], lines: list[str]):
    (out / f"{stem}.tex").write_text("\n".join([f"% {h}" for h in header] + lines) + "\n")


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------------- figures

def rounds_fig(cfg, users, dec, out):
    ks = cfg["prompts"]["rounds"]
    u = users[(users.population == "uniform")]
    d = dec[(dec.population == "uniform") & (dec.dataset == "adult") & (dec.lp_map == "linear")]
    cos, agr = [], []
    for k in ks:
        if k < 6:
            gu = u[(u.method == "pa6") & (u.analysis == "rounds") & (u.k == k)]
            gd = d[(d.method == "pa6") & (d.analysis == "rounds") & (d.k == k)]
        else:
            gu = u[(u.method == "pa6") & (u.analysis == "main")]
            gd = d[(d.method == "pa6") & (d.analysis == "main")]
        cos.append(ci(cfg, gu.cosine.to_numpy(), gu.rep.to_numpy(), "rounds", k, "cos"))
        agr.append(ci(cfg, gd.agree.to_numpy(float), gd.rep.to_numpy(), "rounds", k, "agr"))
    eu = u[(u.method == "expert") & (u.analysis == "main")]
    ed = d[(d.method == "expert") & (d.analysis == "main")]
    e_cos = eu.cosine.mean()
    e_agr = ed.agree.mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.5), sharex=True)
    for ax, vals, ref, lab in ((a1, cos, e_cos, "cosine agreement"), (a2, agr, e_agr, "budget agreement (Adult, linear)")):
        v = np.array(vals)
        ax.fill_between(ks, v[:, 1], v[:, 2], color=BLUE, alpha=0.2, lw=0)
        ax.plot(ks, v[:, 0], color=BLUE, marker="o", ms=3.5, label=disp(cfg, "pa6"))
        ax.axhline(ref, color=GRAY, ls="--", lw=1.1, label=disp(cfg, "expert"))
        ax.set_xlabel("prompts")
        ax.set_ylabel(lab)
        ax.set_xticks(ks)
    a1.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    savefig(fig, out, "rounds_ch7")
    return cos, agr, e_cos, e_agr


def share_error_fig(cfg, users, out):
    u = users[(users.population == "uniform") & (users.analysis == "main")]
    data = [u[u.method == m].share_error.to_numpy() for m in PLOT_METHODS]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    parts = ax.violinplot(data, positions=range(len(PLOT_METHODS)), showextrema=False, widths=0.8)
    for body, m in zip(parts["bodies"], PLOT_METHODS):
        body.set_facecolor(STYLE[m]["color"])
        body.set_edgecolor(STYLE[m]["color"])
        body.set_alpha(0.45)
        if STYLE[m]["hatch"]:
            body.set_hatch(STYLE[m]["hatch"])
    med = [np.median(x) for x in data]
    ax.scatter(range(len(PLOT_METHODS)), med, color="white", edgecolor="black", zorder=3, s=22, label="median")
    ax.set_xticks(range(len(PLOT_METHODS)))
    ax.set_xticklabels([disp(cfg, m) for m in PLOT_METHODS], rotation=20, ha="right")
    ax.set_ylabel(r"$|\hat\pi - \pi^*|$ (privacy-share error)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    savefig(fig, out, "share_error_ch7")


def budget_fig(cfg, dsum, out):
    settings = [tuple(s) for s in cfg["uldp"]["settings"]]
    fig, ax = plt.subplots(figsize=(WIDTH, 4.4))
    nm = len(PLOT_METHODS)
    h = 0.8 / nm
    for gi, st in enumerate(settings):
        for mi, m in enumerate(PLOT_METHODS):
            r = [x for x in dsum if x["method"] == m and (x["dataset"], x["lp_map"]) == st][0]
            y = gi + (mi - (nm - 1) / 2) * h
            val, lo, hi = r["agree"]
            ax.barh(y, val, height=h * 0.92, color=STYLE[m]["color"], alpha=0.85 if not STYLE[m]["hatch"] else 0.45,
                    hatch=STYLE[m]["hatch"], edgecolor=STYLE[m]["color"], lw=0.6,
                    label=disp(cfg, m) if gi == 0 else None)
            ax.errorbar(val, y, xerr=[[val - lo], [hi - val]], color="black", lw=0.8, capsize=1.5)
    ax.set_yticks(range(len(settings)))
    ax.set_yticklabels([SETTING_NAMES[s] for s in settings])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"budget agreement (share of users with the same $\varepsilon^*$ as the true weights)")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    savefig(fig, out, "budget_agreement_ch7")


def _tern(W):
    W = np.asarray(W, float)
    return W[:, 1] + W[:, 2] / 2.0, W[:, 2] * np.sqrt(3) / 2.0


def weights_fig(cfg, users, out):
    spec = cfg["outputs"]["weights_figure"]
    u = users[(users.population == "uniform") & (users.analysis == "main") & (users.rep == spec["replication"])
              & (users.user < spec["users"])]
    panels = [("true", None), ("pa6", "pa6"), ("pa6_bayes", "pa6_bayes")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.35))
    M = U.matrix("adult", "linear")
    g = np.round(np.arange(0, 1.0001, 0.0001), 4)
    idx, _ = U.select(M, np.column_stack([g, 1 - g, np.zeros_like(g)]))
    switches = [float((g[i] + g[i + 1]) / 2) for i in np.nonzero(np.diff(idx))[0]]
    for ax, (key, m) in zip(axes, panels):
        tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
        ax.plot(tri[:, 0], tri[:, 1], color="black", lw=0.8)
        for pi in switches:                       # rays of constant privacy share from the CR vertex
            ax.plot([0.5, 1 - pi], [np.sqrt(3) / 2, 0], color="#9a9a9a", lw=0.6, ls="--")
        if m is None:
            W = u[u.method == "oracle"][["w_lp_true", "w_la_true", "w_cr_true"]].to_numpy()
            color, title = "#000000", "True weights"
        else:
            W = u[u.method == m][["w_lp", "w_la", "w_cr"]].to_numpy()
            color, title = STYLE[m]["color"], disp(cfg, m)
        vals, counts = np.unique(np.round(W, 6), axis=0, return_counts=True)
        x, y = _tern(vals)
        ax.scatter(x, y, s=4 + 6 * np.sqrt(counts), color=color, alpha=0.45, lw=0)
        ax.set_title(f"{title} ({len(vals)} distinct)" if m is not None else f"{title} (300 users)", fontsize=8)
        ax.text(-0.04, -0.06, "LP", ha="center", va="top", fontsize=7.5)
        ax.text(1.04, -0.06, "LA", ha="center", va="top", fontsize=7.5)
        ax.text(0.5, np.sqrt(3) / 2 + 0.04, "CR", ha="center", va="bottom", fontsize=7.5)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.text(0.5, -0.02, "Dashed rays: privacy shares at which the Adult/linear selection changes "
             + ", ".join(f"{s:.3f}" for s in switches), ha="center", fontsize=6.8, color="#555555")
    fig.tight_layout()
    savefig(fig, out, "weights_ch7")


# ----------------------------------------------------------------------------- main

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    smoke = "--smoke" in argv
    cfg = load_config()
    if smoke:
        cfg["stats"]["bootstrap_resamples"] = 1000
    res = ROOT / (cfg["outputs"]["dir"] + ("_smoke" if smoke else ""))
    tabs, figs = res / "tables", res / "figures"
    tabs.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    users = pd.read_parquet(res / "users.parquet")
    dec = pd.read_parquet(res / "decisions.parquet")
    repair = pd.read_csv(res / "repair.csv")

    # ---------------------------------------------------------------- tests
    tests = S.family_tests(users, dec, cfg)
    decisions = S.decide(tests)
    tests.to_csv(res / "stats.csv", index=False)
    with open(res / "hypotheses_outcomes.json", "w") as f:
        json.dump(decisions, f, indent=1)

    # ---------------------------------------------------------------- summaries
    wsum = weight_summary(cfg, users)
    dsum = decision_summary(cfg, dec)
    prof_rows = []
    for p in PROFILE_NAMES:
        ws = {r["method"]: r for r in weight_summary(cfg, users, population=p)}
        prof_rows.append((p, ws, None))
    # rounds
    rounds_cos, rounds_agr, e_cos, e_agr = rounds_fig(cfg, users, dec, figs)
    rounds_err = []
    for k in cfg["prompts"]["rounds"]:
        if k < 6:
            g = users[(users.population == "uniform") & (users.method == "pa6") & (users.analysis == "rounds") & (users.k == k)]
        else:
            g = users[(users.population == "uniform") & (users.method == "pa6") & (users.analysis == "main")]
        rounds_err.append(ci(cfg, g.share_error.to_numpy(), g.rep.to_numpy(), "rounds", k, "err"))
    # tau sensitivity
    tau_rows = []
    for tau in cfg["response"]["tau_grid"]:
        an = "main" if tau == cfg["response"]["tau"] else "tau"
        sel = lambda x, t=tau: x[np.isclose(x.tau, t)]  # noqa: E731
        ws = {r["method"]: r for r in weight_summary(cfg, users, analysis=an, methods=["pa6", "pa6_bayes", "lpla", "lpla_bayes"], extra=sel)}
        ds_ = {(r["method"], r["dataset"]): r for r in decision_summary(cfg, dec, analysis=an, methods=["pa6", "pa6_bayes", "lpla", "lpla_bayes"],
                                                                           settings=[("adult", "linear"), ("cdc", "linear")], extra=sel)}
        rr = float(repair[(repair.population == "uniform") & np.isclose(repair.tau, tau)].repair_rate.mean())
        tau_rows.append((tau, ws, ds_, rr))
    # anchor sensitivity
    lam_rows = []
    for lam in cfg["slider"]["lambda_grid"]:
        an = "main" if lam == cfg["slider"]["lambda"] else "anchor"
        sel = lambda x, l_=lam: x[np.isclose(x.lam, l_)]  # noqa: E731
        ws = weight_summary(cfg, users, analysis=an, methods=["slider"], extra=sel)[0]
        ds_ = {r["dataset"]: r for r in decision_summary(cfg, dec, analysis=an, methods=["slider"],
                                                         settings=[("adult", "linear"), ("cdc", "linear")], extra=sel)}
        lam_rows.append((lam, ws, ds_))

    # ---------------------------------------------------------------- LaTeX tables
    lines = [f"{disp(cfg, r['method'])} & {fci(r['cosine'])} & {fci(r['l1'])} & {fci(r['share_error'])} \\\\" for r in wsum]
    write_tex(tabs, "tab_weight_metrics", [
        "Weight metrics, uniform population (Dirichlet(1,1,1)), primary tau and lambda; pooled mean over all users of "
        f"{cfg['replications']} replications [95% cluster-bootstrap interval]",
        "columns: method & cosine similarity & L1 distance & privacy-share error |pi_hat - pi*|"], lines)
    settings = [tuple(s) for s in cfg["uldp"]["settings"]]
    lines = []
    for m in METHODS:
        cells = []
        for st in settings:
            r = [x for x in dsum if x["method"] == m and (x["dataset"], x["lp_map"]) == st][0]
            cells += [fci(r["agree"]), fci(r["loss"])]
        lines.append(f"{disp(cfg, m)} & " + " & ".join(cells) + " \\\\")
    write_tex(tabs, "tab_budget_agreement", [
        "Budget agreement (share of users whose weights give the same eps* as the true weights) and mean closeness loss "
        "C_true(eps*_true) - C_true(eps*_method); uniform population; [95% cluster-bootstrap interval]",
        "columns: method & " + " & ".join(f"{SETTING_NAMES[s]} agreement & {SETTING_NAMES[s]} loss" for s in settings)], lines)
    lines = []
    for p, ws, ds_ in prof_rows:
        drows = decision_summary(cfg, dec, population=p, methods=["pa6", "lpla", "expert"], settings=[("adult", "linear")])
        dmap = {r["method"]: r for r in drows}
        for i, m in enumerate(["pa6", "lpla", "expert"]):
            lines.append(f"{PROFILE_NAMES[p] if i == 0 else ''} & {disp(cfg, m)} & {fci(ws[m]['cosine'])} & {fci(dmap[m]['agree'])} \\\\")
        lines.append("\\midrule")
    write_tex(tabs, "tab_profiles", [
        "Profile populations w* ~ Dirichlet(20 c), 250 users per profile and replication; cosine similarity and budget "
        "agreement (Adult, linear LP) [95% cluster-bootstrap interval]",
        "columns: profile & method & cosine & budget agreement (Adult, linear)"], lines[:-1])
    lines = [f"{k} & {fci(c)} & {fci(e)} & {fci(a)} \\\\" for k, c, e, a in zip(cfg["prompts"]["rounds"], rounds_cos, rounds_err, rounds_agr)]
    write_tex(tabs, "tab_rounds", [
        "PrivAgent-6 (vote inference) with the first k prompts of the presented order; k = 6 includes the consistency "
        "check and repair; uniform population [95% cluster-bootstrap interval]",
        f"Expert default: cosine {e_cos:.3f}, budget agreement (Adult, linear) {e_agr:.3f}",
        "columns: prompts & cosine & privacy-share error & budget agreement (Adult, linear)"], lines)
    lines = []
    for tau, ws, ds_, rr in tau_rows:
        for i, m in enumerate(["pa6", "pa6_bayes", "lpla", "lpla_bayes"]):
            lines.append(f"{tau:g}" * (i == 0) + f" & {disp(cfg, m)} & {fci(ws[m]['cosine'])} & {fci(ws[m]['share_error'])} & "
                         f"{fci(ds_[(m, 'adult')]['agree'])} & {fci(ds_[(m, 'cdc')]['agree'])} \\\\")
        lines.append("\\midrule")
    write_tex(tabs, "tab_tau", [
        "Sensitivity to the user temperature tau (uniform population); the analyst's tau of the Bayesian inference stays "
        f"{cfg['inference']['bayes']['analyst_tau']:g}; answers are coupled across tau (common random numbers)",
        "columns: tau & method & cosine & privacy-share error & agreement Adult linear & agreement CDC-Diabetes linear"], lines[:-1])
    lines = [f"{lam:g} & {fci(ws['cosine'])} & {fci(ws['share_error'])} & {fci(ds_['adult']['agree'])} & {fci(ds_['cdc']['agree'])} \\\\"
             for lam, ws, ds_ in lam_rows]
    pa6w = [r for r in wsum if r["method"] == "pa6"][0]
    pa6d = {r["dataset"]: r for r in dsum if r["method"] == "pa6" and r["lp_map"] == "linear"}
    write_tex(tabs, "tab_anchor", [
        "Sensitivity of the anchored slider to the anchoring strength lambda (anchor (1/3,1/3,1/3), report noise sd "
        f"{cfg['slider']['noise_sd']:g}; the same noise draws for every lambda); uniform population",
        f"Reference PrivAgent-6: cosine {fci(pa6w['cosine'])}, share error {fci(pa6w['share_error'])}, agreement Adult "
        f"{fci(pa6d['adult']['agree'])}, CDC-Diabetes {fci(pa6d['cdc']['agree'])}",
        "columns: lambda & cosine & privacy-share error & agreement Adult linear & agreement CDC-Diabetes linear"], lines)

    # ---------------------------------------------------------------- figures
    share_error_fig(cfg, users, figs)
    budget_fig(cfg, dsum, figs)
    weights_fig(cfg, users, figs)

    # ---------------------------------------------------------------- metrics csv
    mrows = []
    for r in wsum:
        for metric in ("cosine", "l1", "share_error"):
            mrows.append(dict(population="uniform", method=r["method"], metric=metric, setting="", value=r[metric][0],
                              lo=r[metric][1], hi=r[metric][2]))
    for r in dsum:
        for metric in ("agree", "loss"):
            mrows.append(dict(population="uniform", method=r["method"], metric=metric, setting=f"{r['dataset']}/{r['lp_map']}",
                              value=r[metric][0], lo=r[metric][1], hi=r[metric][2]))
        mrows.append(dict(population="uniform", method=r["method"], metric="undefined", setting=f"{r['dataset']}/{r['lp_map']}",
                          value=r["undefined"], lo=np.nan, hi=np.nan))
    pd.DataFrame(mrows).to_csv(res / "metrics.csv", index=False)

    if not smoke:
        write_summary(res, cfg, tests, decisions, wsum, dsum, prof_rows, rounds_cos, rounds_err, rounds_agr, e_cos,
                      e_agr, tau_rows, lam_rows, repair, users, dec)
    print("report written to", res)


def write_summary(res, cfg, tests, decisions, wsum, dsum, prof_rows, rounds_cos, rounds_err, rounds_agr, e_cos, e_agr,
                  tau_rows, lam_rows, repair, users, dec):
    L = ["# Chapter 7 (PrivAgent) simulation: summary\n",
         "All numbers are produced by `python -m src.run && python -m src.report` from the committed configuration "
         "and are reported as they came out. No human participants are involved: all users are simulated. The "
         "simulated users answer according to fixed true weights with logistic response noise; they have no "
         "anchoring in the dialogue, no present bias and no context-dependence. The anchored-slider baseline uses an "
         "assumed anchoring model (Section 3 of the specification).\n"]
    pre = (res / "preregistration.txt").read_text().strip() if (res / "preregistration.txt").exists() else "(not recorded)"
    L += ["## Pre-registration\n", pre + "\n"]
    L += ["## Sanity checks\n", "```\n" + (res / "sanity.log").read_text() + "```\n"]
    # hypotheses
    L.append("## Hypotheses\n")
    L.append("Primary population (uniform, 20,000 users), primary tau = 0.1, lambda = 0.4. Pooled means with 95% "
             "cluster-bootstrap intervals over replications; Holm within each family; a test counts if Holm-adjusted "
             "p < 0.05 and the pooled difference A - B has the stated sign. n_a_only / n_b_only: users for whom only A "
             "(only B) agrees (McNemar) or A > B (A < B) (Wilcoxon).\n")
    for h in ["H1", "H2", "H3", "H4", "H5"]:
        d = decisions[h]
        t = tests[tests.hypothesis == h]
        L.append(f"### {h}: {d['outcome'].upper()} ({d['in_direction']} of {d['tests']} tests in the stated direction)\n")
        rows = []
        for r in t.itertuples():
            rows.append(dict(test=r.test, setting=(f"{r.dataset}/{r.lp_map}" if r.dataset else "-"),
                             A=disp(cfg, r.a), B=disp(cfg, r.b), metric=r.metric,
                             mean_A=fci((r.mean_a, r.a_lo, r.a_hi)), mean_B=fci((r.mean_b, r.b_lo, r.b_hi)),
                             diff_A_minus_B=fci((r.diff, r.diff_lo, r.diff_hi)), n_a_only=r.n_a_only,
                             n_b_only=r.n_b_only, p_raw=f"{r.p_raw:.3g}", p_holm=f"{r.p_holm:.3g}",
                             stated=r.direction, counts=r.in_direction))
        L.append(md_table(pd.DataFrame(rows)) + "\n")
        if h == "H4":
            L.append("This result depends on the assumed anchoring model (anchor (1/3, 1/3, 1/3), strength lambda = 0.4, "
                     "Gaussian report noise sd 0.1); see the anchoring sensitivity below.\n")
    # tables
    L.append("## Weight metrics (uniform population)\n")
    L.append(md_table(pd.DataFrame([dict(method=disp(cfg, r["method"]), cosine=fci(r["cosine"]), L1=fci(r["l1"]),
                                         privacy_share_error=fci(r["share_error"]), prompts=f"{r['n_prompts']:.3f}")
                                    for r in wsum])) + "\n")
    L.append("## Budget agreement and closeness loss (uniform population)\n")
    settings = [tuple(s) for s in cfg["uldp"]["settings"]]
    rows = []
    for m in METHODS:
        r = dict(method=disp(cfg, m))
        for st in settings:
            x = [y for y in dsum if y["method"] == m and (y["dataset"], y["lp_map"]) == st][0]
            r[f"{SETTING_NAMES[st]} agreement"] = fci(x["agree"])
        rows.append(r)
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    rows = []
    for m in METHODS:
        r = dict(method=disp(cfg, m))
        for st in settings:
            x = [y for y in dsum if y["method"] == m and (y["dataset"], y["lp_map"]) == st][0]
            r[f"{SETTING_NAMES[st]} loss"] = fci(x["loss"])
            r[f"{SETTING_NAMES[st]} undefined"] = f"{x['undefined']:.4f}"
        rows.append(r)
    L.append("Mean closeness loss (undefined selections excluded) and share of undefined selections:\n")
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    L.append("## Profile populations (Dirichlet(20 c), 5,000 users per profile)\n")
    rows = []
    for p, ws, _ in prof_rows:
        drows = {r["method"]: r for r in decision_summary(cfg, dec, population=p, methods=METHODS)
                 if (r["dataset"], r["lp_map"]) == ("adult", "linear")}
        for m in METHODS:
            rows.append(dict(profile=PROFILE_NAMES[p], method=disp(cfg, m), cosine=fci(ws[m]["cosine"]),
                             share_error=fci(ws[m]["share_error"]), agreement_adult_linear=fci(drows[m]["agree"]),
                             repair_rate=f"{ws[m]['repair_rate']:.4f}"))
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    L.append("## Rounds analysis (PrivAgent-6, vote inference, uniform population)\n")
    L.append(md_table(pd.DataFrame([dict(prompts=k, cosine=fci(c), privacy_share_error=fci(e), agreement_adult_linear=fci(a))
                                    for k, c, e, a in zip(cfg["prompts"]["rounds"], rounds_cos, rounds_err, rounds_agr)])) + "\n")
    L.append(f"k < 6: first k prompts, no repair; k = 6: PrivAgent-6 including the consistency check and repair. "
             f"Expert default: cosine {e_cos:.3f}, budget agreement (Adult, linear) {e_agr:.3f}.\n")
    L.append("## Repair rate (PrivAgent-6)\n")
    rr = repair.groupby(["population", "tau"]).repair_rate.mean().reset_index()
    rr["repair_rate"] = rr["repair_rate"].map(lambda v: f"{v:.4f}")
    L.append(md_table(rr) + "\n")
    L.append("Repair is triggered when the three pair results form a cycle; every repair prompt goes to the LP-LA "
             "pair (config.yaml, prompts.repair).\n")
    L.append("## Sensitivity to the user temperature tau (uniform population)\n")
    rows = []
    for tau, ws, ds_, rr_ in tau_rows:
        for m in ["pa6", "pa6_bayes", "lpla", "lpla_bayes"]:
            rows.append(dict(tau=f"{tau:g}", method=disp(cfg, m), cosine=fci(ws[m]["cosine"]),
                             share_error=fci(ws[m]["share_error"]), agreement_adult_linear=fci(ds_[(m, "adult")]["agree"]),
                             agreement_cdc_linear=fci(ds_[(m, "cdc")]["agree"]), repair_rate_pa6=f"{rr_:.4f}"))
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    L.append(f"The analyst's tau of the Bayesian inference stays {cfg['inference']['bayes']['analyst_tau']:g} for every "
             "user tau. Answers are coupled across tau (the same uniform draws).\n")
    L.append("## Sensitivity to the anchoring strength lambda (anchored slider, uniform population)\n")
    L.append(md_table(pd.DataFrame([dict(lam=f"{lam:g}", cosine=fci(ws["cosine"]), share_error=fci(ws["share_error"]),
                                         agreement_adult_linear=fci(ds_["adult"]["agree"]),
                                         agreement_cdc_linear=fci(ds_["cdc"]["agree"])) for lam, ws, ds_ in lam_rows])) + "\n")
    pa6w = [r for r in wsum if r["method"] == "pa6"][0]
    pa6d = {r["dataset"]: r for r in dsum if r["method"] == "pa6" and r["lp_map"] == "linear"}
    L.append(f"PrivAgent-6 for reference: cosine {fci(pa6w['cosine'])}, share error {fci(pa6w['share_error'])}, agreement "
             f"Adult {fci(pa6d['adult']['agree'])}, CDC-Diabetes {fci(pa6d['cdc']['agree'])}.\n")
    L.append("## Runtime\n")
    L.append((res / "runtime.txt").read_text() + "\n")
    if (res / "runtime_report.txt").exists():
        L.append((res / "runtime_report.txt").read_text() + "\n")
    L.append("## Notes on the specification (fixed before the main run)\n")
    L.append("- Switch points of Chapter 6: on a grid of step 0.001 of the privacy share (w_CR = 0), the first grid value "
             "at which Adult/linear selects a new budget is 0.013, 0.472, 0.887 and 0.946, as in Chapter 6; the exact "
             "switch points are 0.0123, 0.4719, 0.8863 and 0.9455.\n"
             "- Repair: a cycle requires all three pairs to be decided, so no pair is split when a cycle is detected; the "
             "repair prompt goes to the first pair in the stated order (LP-LA).\n"
             "- The Bayesian unit test uses 200 answers drawn from exactly the analyst's response model (tau = 0.1); "
             "deterministic answers to the fixed templates reveal only the ranking of the weights.\n"
             "- Bayesian inference: one set of 20,000 prior draws per replication and prompt set (fixed stream), shared by "
             "the users of that replication.\n")
    L.append("## Deviations\n")
    dev = res / "deviations.md"
    L.append(dev.read_text() if dev.exists() else "None.\n")
    (res / "summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
