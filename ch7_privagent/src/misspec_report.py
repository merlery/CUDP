"""Tables, figures and summary.md of the EXPLORATORY Draft 3 misspecification run, and the three thesis figures
regenerated from the stored results/ch7/ data with the current method names.

    python -m src.misspec_report
"""
from __future__ import annotations

import subprocess
import time
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import stats as S  # noqa: E402
from .misspec import METHODS, load_d3  # noqa: E402
from .report import BLUE, GRAY, GREEN, ORANGE, PURPLE, STYLE, WIDTH  # noqa: E402,F401  (also sets rcParams)
from .run import ROOT, load_config  # noqa: E402

POP_TXT = {"baseline": "Baseline (logistic, tau = 0.1)", "lapse_0.1": "Lapse p = 0.1", "lapse_0.2": "Lapse p = 0.2",
           "lapse_0.33": "Lapse p = 0.33", "privacy_0.1": "Privacy-always f = 0.1", "privacy_0.25": "Privacy-always f = 0.25",
           "side_fixed_0.5": "Side bias b = 0.5 (fixed)", "side_fixed_1.0": "Side bias b = 1.0 (fixed)",
           "side_alt_0.5": "Side bias b = 0.5 (alternating)", "side_alt_1.0": "Side bias b = 1.0 (alternating)",
           "lexicographic": "Lexicographic"}
METRICS = [("share_error", "Share error"), ("cosine", "Cosine"), ("agree_adult", "Agreement, Adult"),
           ("loss_adult", "Loss, Adult"), ("agree_cdc", "Agreement, CDC-Diabetes"), ("loss_cdc", "Loss, CDC-Diabetes")]
EXPL = "EXPLORATORY (not pre-registered; Draft 3 follow-up)"


def ci(d3, values, reps, *key):
    return S.cluster_ci(np.asarray(values, float), np.asarray(reps), d3["metrics"]["bootstrap_resamples"],
                        [d3["metrics"]["bootstrap_seed"], zlib.crc32(repr(key).encode())], 0.95)


def f3(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.3f}"


def md(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def savefig(fig, out, stem):
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=220)
    plt.close(fig)


def main():
    t0 = time.time()
    cfg = load_config()
    d3 = load_d3()
    out = ROOT / d3["out_dir"]
    NM = d3["methods"]
    df = pd.read_parquet(out / "misspec_users.parquet")

    # ------------------------------------------------------------------ metrics
    rows = []
    for pname in d3["populations"]:
        g0 = df[df.population == pname]
        groups = [("all", g0)]
        if pname.startswith("privacy"):
            groups += [("affected", g0[g0.affected]), ("unaffected", g0[~g0.affected])]
        for grp, g in groups:
            for m in METHODS:
                gm = g[g.method == m]
                r = {"population": pname, "users": grp, "method": NM[m], "n": len(gm)}
                for col, _ in METRICS:
                    v, lo, hi = ci(d3, gm[col].to_numpy(float), gm.rep.to_numpy(), pname, grp, m, col)
                    r[col], r[col + "_lo"], r[col + "_hi"] = v, lo, hi
                r["undefined_adult"] = float(gm.eps_adult.isna().mean())
                r["undefined_cdc"] = float(gm.eps_cdc.isna().mean())
                r["repair_rate"] = float(gm.repair.mean())
                rows.append(r)
    R = pd.DataFrame(rows)
    R.insert(0, "label", "exploratory")
    R.to_csv(out / "misspec_runs.csv", index=False)

    # ------------------------------------------------------------------ baseline check against stored metrics.csv
    st = pd.read_csv(ROOT / "results/ch7/metrics.csv")
    st = st[st.population == "uniform"]
    inv = {v: k for k, v in NM.items()}
    chk = []
    for r in R[(R.population == "baseline")].itertuples():
        m = inv[r.method]
        for col, stm, setting in (("share_error", "share_error", None), ("cosine", "cosine", None),
                                  ("agree_adult", "agree", "adult/linear"), ("loss_adult", "loss", "adult/linear"),
                                  ("agree_cdc", "agree", "cdc/linear"), ("loss_cdc", "loss", "cdc/linear")):
            s = st[(st.method == m) & (st.metric == stm) & ((st.setting == setting) if setting else st.setting.isna())].iloc[0]
            chk.append({"method": r.method, "metric": col, "stored": s.value, "this run": getattr(r, col),
                        "abs difference": abs(s.value - getattr(r, col))})
    C = pd.DataFrame(chk)
    max_diff = float(C["abs difference"].max())

    # ------------------------------------------------------------------ tables
    def cell(r, col, with_ci=True):
        return f"{f3(r[col])} [{f3(r[col + '_lo'])}, {f3(r[col + '_hi'])}]" if with_ci else f3(r[col])

    tex = [f"% {EXPL}. Misspecified simulated users, uniform true weights, 20 replications x 1,000 users; pooled means "
           "[95% cluster-bootstrap interval over replications]",
           "% columns: population & users & method & share error & cosine & agreement Adult & loss Adult & agreement "
           "CDC-Diabetes & loss CDC-Diabetes (linear LP map)"]
    mdrows = []
    prev = None
    for _, r in R.iterrows():
        pl = POP_TXT[r.population] if (r.population, r.users) != prev else ""
        ul = r.users if (r.population, r.users) != prev else ""
        prev = (r.population, r.users)
        tex.append(f"{pl} & {ul} & {r.method} & " + " & ".join(cell(r, c) for c, _ in METRICS) + " \\\\")
        mdrows.append({"population": POP_TXT[r.population], "users": r.users, "method": r.method,
                       **{lab: cell(r, c) for c, lab in METRICS}})
    (out / "tab_misspec.tex").write_text("\n".join(tex) + "\n")
    M = pd.DataFrame(mdrows)

    # ------------------------------------------------------------------ misspec figure
    pops = list(d3["populations"])
    meths = [NM[m] for m in METHODS]
    fig, ax = plt.subplots(figsize=(WIDTH, 6.2))
    nm = len(METHODS)
    h = 0.8 / nm
    for gi, p in enumerate(pops):
        for mi, m in enumerate(METHODS):
            r = R[(R.population == p) & (R.users == "all") & (R.method == NM[m])].iloc[0]
            y = gi + (mi - (nm - 1) / 2) * h
            stl = STYLE[m]
            ax.barh(y, r.agree_adult, height=h * 0.92, color=stl["color"], alpha=0.85 if not stl["hatch"] else 0.45,
                    hatch=stl["hatch"], edgecolor=stl["color"], lw=0.6, label=NM[m] if gi == 0 else None)
            ax.errorbar(r.agree_adult, y, xerr=[[r.agree_adult - r.agree_adult_lo], [r.agree_adult_hi - r.agree_adult]],
                        color="black", lw=0.8, capsize=1.5)
    ax.set_yticks(range(len(pops)))
    ax.set_yticklabels([POP_TXT[p] for p in pops])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"budget agreement, Adult, linear LP map (exploratory)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    savefig(fig, out, "misspec_ch7")

    # ------------------------------------------------------------------ regenerated thesis figures (stored data)
    users = pd.read_parquet(ROOT / "results/ch7/users.parquet")
    dec = pd.read_parquet(ROOT / "results/ch7/decisions.parquet")
    scfg = dict(cfg)

    def sci(values, reps, *key):
        return S.cluster_ci(values, reps, cfg["stats"]["bootstrap_resamples"],
                            [cfg["stats"]["bootstrap_seed"], zlib.crc32(repr(key).encode())], cfg["stats"]["ci"])

    # budget_agreement_ch7: layout of the stored figure, the six methods with the current names
    settings = [tuple(s) for s in cfg["uldp"]["settings"]]
    SN = {("adult", "linear"): "Adult, linear", ("adult", "exponential"): "Adult, exponential",
          ("cdc", "linear"): "CDC-Diabetes, linear", ("cdc", "exponential"): "CDC-Diabetes, exponential"}
    dm = dec[(dec.population == "uniform") & (dec.analysis == "main")]
    fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
    h = 0.8 / nm
    for gi, stt in enumerate(settings):
        for mi, m in enumerate(METHODS):
            g = dm[(dm.method == m) & (dm.dataset == stt[0]) & (dm.lp_map == stt[1])]
            val, lo, hi = sci(g.agree.to_numpy(float), g.rep.to_numpy(), "uniform", "main", m, stt[0], stt[1], "agree")
            y = gi + (mi - (nm - 1) / 2) * h
            stl = STYLE[m]
            ax.barh(y, val, height=h * 0.92, color=stl["color"], alpha=0.85 if not stl["hatch"] else 0.45,
                    hatch=stl["hatch"], edgecolor=stl["color"], lw=0.6, label=NM[m] if gi == 0 else None)
            ax.errorbar(val, y, xerr=[[val - lo], [hi - val]], color="black", lw=0.8, capsize=1.5)
    ax.set_yticks(range(len(settings)))
    ax.set_yticklabels([SN[s] for s in settings])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"budget agreement (share of users with the same $\varepsilon^*$ as the true weights)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    savefig(fig, out, "budget_agreement_ch7")

    # rounds_ch7: Vote-6 after the first k prompts (three panels, thesis caption)
    ks = cfg["prompts"]["rounds"]
    uu = users[users.population == "uniform"]
    dd = dec[(dec.population == "uniform") & (dec.dataset == "adult") & (dec.lp_map == "linear")]
    vals = {"cos": [], "err": [], "agr": []}
    for k in ks:
        if k < 6:
            gu = uu[(uu.method == "pa6") & (uu.analysis == "rounds") & (uu.k == k)]
            gd = dd[(dd.method == "pa6") & (dd.analysis == "rounds") & (dd.k == k)]
        else:
            gu = uu[(uu.method == "pa6") & (uu.analysis == "main")]
            gd = dd[(dd.method == "pa6") & (dd.analysis == "main")]
        vals["cos"].append(sci(gu.cosine.to_numpy(), gu.rep.to_numpy(), "rounds", k, "cos"))
        vals["err"].append(sci(gu.share_error.to_numpy(), gu.rep.to_numpy(), "rounds", k, "err"))
        vals["agr"].append(sci(gd.agree.to_numpy(float), gd.rep.to_numpy(), "rounds", k, "agr"))
    eu = uu[(uu.method == "expert") & (uu.analysis == "main")]
    ed = dd[(dd.method == "expert") & (dd.analysis == "main")]
    refs = {"cos": eu.cosine.mean(), "err": eu.share_error.mean(), "agr": ed.agree.mean()}
    labs = {"cos": "(a) cosine agreement", "err": "(b) privacy-share error", "agr": "(c) budget agreement (Adult, linear)"}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3), sharex=True)
    for ax, key in zip(axes, ("cos", "err", "agr")):
        v = np.array(vals[key])
        ax.fill_between(ks, v[:, 1], v[:, 2], color=BLUE, alpha=0.2, lw=0)
        ax.plot(ks, v[:, 0], color=BLUE, marker="o", ms=3.2, label=NM["pa6"])
        ax.axhline(refs[key], color=GRAY, ls="--", lw=1.1, label=NM["expert"])
        ax.set_xlabel("prompts")
        ax.set_title(labs[key], fontsize=8)
        ax.set_xticks(ks)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    savefig(fig, out, "rounds_ch7")

    # tau_ch7: the four dialogue variants against the user tau (two panels, thesis caption)
    taus = cfg["response"]["tau_grid"]
    dvars = ["pa6", "pa6_bayes", "lpla", "lpla_bayes"]
    mk = {"pa6": "o", "pa6_bayes": "s", "lpla": "^", "lpla_bayes": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharex=True)
    tau_tab = []
    for m in dvars:
        cs, ag = [], []
        for t in taus:
            an = "main" if t == cfg["response"]["tau"] else "tau"
            gu = uu[(uu.method == m) & (uu.analysis == an) & np.isclose(uu.tau, t)]
            gd = dd[(dd.method == m) & (dd.analysis == an) & np.isclose(dd.tau, t)]
            cs.append(sci(gu.cosine.to_numpy(), gu.rep.to_numpy(), "tau", m, t, "cos"))
            ag.append(sci(gd.agree.to_numpy(float), gd.rep.to_numpy(), "tau", m, t, "agr"))
            tau_tab.append({"tau": t, "method": NM[m], "cosine": f3(cs[-1][0]), "agreement Adult": f3(ag[-1][0])})
        for ax, v in zip(axes, (np.array(cs), np.array(ag))):
            stl = STYLE[m]
            ax.errorbar(taus, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=stl["color"], marker=mk[m],
                        ms=3.5, lw=1.3, capsize=1.5, ls="--" if stl["hatch"] else "-", label=NM[m],
                        mfc="white" if stl["hatch"] else stl["color"])
    axes[0].axhline(refs["cos"], color=GRAY, ls="--", lw=1.1, label=NM["expert"])
    axes[1].axhline(refs["agr"], color=GRAY, ls="--", lw=1.1, label=NM["expert"])
    for ax, lab in zip(axes, ("(a) cosine agreement", "(b) budget agreement (Adult, linear)")):
        ax.set_xscale("log")
        ax.set_xticks(taus)
        ax.set_xticklabels([f"{t:g}" for t in taus])
        ax.minorticks_off()
        ax.set_xlabel(r"user temperature $\tau$")
        ax.set_title(lab, fontsize=8)
    fig.tight_layout()
    hd, lb = axes[1].get_legend_handles_labels()
    fig.legend(hd, lb, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=5)
    savefig(fig, out, "tau_ch7")

    write_summary(cfg, d3, out, M, R, C, max_diff, pd.DataFrame(tau_tab), vals, refs, time.time() - t0)


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def write_summary(cfg, d3, out, M, R, C, max_diff, tau_tab, vals, refs, t_rep):
    c_set, c_code = git("rev-parse", "4002e0e"), git("rev-parse", "61af0cd")
    c_fix = git("rev-parse", "2a112d5")
    sanity = (out / "sanity_misspec.log").read_text().strip()
    runtime = (out / "runtime_misspec.txt").read_text().strip()
    keyM = M[M.users == "all"][["population", "method", "Share error", "Cosine", "Agreement, Adult", "Agreement, CDC-Diabetes"]]
    pa = M[M.users != "all"]
    L = [
        "# Chapter 7: misspecified simulated users (EXPLORATORY, Draft 3 follow-up)",
        "",
        "Every result in this file and in results/ch7/draft3/ is exploratory. It is not part of the pre-registered run "
        "(commit 7bffabb), whose configuration, code and outputs (results/ch7/) are unchanged. Produced by "
        "`python -m src.misspec && python -m src.misspec_report`.",
        "",
        "## Commits",
        "",
        "- The archived snapshot of the pre-registered repository had no git history; it was restored as commit "
        f"`{git('rev-parse', 'dc16b0e')}` (files as archived; results/ch7/summary.md records the pre-registration commit 7bffabb).",
        f"- Settings of this run, committed before the run: `{c_set}` (config_draft3.yaml).",
        f"- Simulation code and tests, before the run: `{c_code}`; `{c_fix}` changes only how the worker function is "
        "passed to joblib (the first launch stopped before any replication ran).",
        "",
        "## Design (config_draft3.yaml)",
        "",
        "Uniform true weights (Dirichlet(1, 1, 1)), 1,000 users x 20 replications per population. Every population uses "
        "the streams of the stored uniform population for the true weights, prompt orders, answer draws, repair draws, "
        "slider noise and prior draws, so the populations share the same users and draws and differ only in the "
        "response rule; the Bayesian inference keeps the analyst temperature 0.1. Response rules: lapse, each answer "
        "replaced by a fair coin with probability p; privacy-always, exactly round(f x 1,000) users per replication "
        "choose the LP option in every LP-LA and LP-CR prompt and follow the logistic rule otherwise; side bias, +b on "
        "the log-odds of the option shown first; lexicographic, the option better on the highest-weighted criterion "
        "(in these prompts the third criterion is equal in both options, so the rule chooses A iff w_j > w_k, which is "
        "the tau -> 0 limit of the logistic rule).",
        "",
        "Side assignment: the stored design (src/dialogue.py) has no screen position and does not alternate the "
        "privacy option. Option A always favors the first criterion of the pair (LP in LP-LA and LP-CR, LA in LA-CR). "
        "Two assignments were therefore run: fixed (A shown first in every prompt, the stored design read literally: "
        "the bias always favors LP, or LA in LA-CR) and alternating (A shown first at presented positions 1, 3, 5 and "
        "the repair prompt, second at 2, 4, 6, in each user's random order).",
        "",
        "The expert default and the anchored slider do not use the dialogue and are the same in every population.",
        "",
        "## Baseline check",
        "",
        "```", sanity, "```",
        "",
        f"Pooled metrics of the baseline population against results/ch7/metrics.csv (uniform, main; largest absolute "
        f"difference {max_diff:.1e}):",
        "",
        md(C.assign(stored=C.stored.map(lambda x: f"{x:.4f}"), **{"this run": C["this run"].map(lambda x: f"{x:.4f}"),
                                                                 "abs difference": C["abs difference"].map(lambda x: f"{x:.1e}")})),
        "",
        "## Results (exploratory)",
        "",
        "Pooled means over all users of all replications with 95% cluster-bootstrap intervals over replications. Linear "
        "LP map. Loss = closeness loss C_true(eps*_true) - C_true(eps*_method).",
        "",
        "### All metrics",
        "",
        md(M[M.users == "all"].drop(columns="users")),
        "",
        "### Privacy-always populations: affected and unaffected users",
        "",
        "The metrics of the affected users compare the estimates with the weights drawn for them, which their answers to "
        "the LP prompts ignore.",
        "",
        md(pa),
        "",
        "### Point estimates only (compact view)",
        "",
        md(keyM.assign(**{c: keyM[c].str.split(" ").str[0] for c in keyM.columns[2:]})),
        "",
        "Shares of undefined selections and repair rates per population and method: misspec_runs.csv.",
        "",
        "## Regenerated thesis figures (stored data, current names)",
        "",
        "budget_agreement_ch7.pdf (layout of the stored figure; the six methods named Vote-6, Bayes-6, Vote-LPLA, "
        "PrivAgent, Expert default, Anchored slider), rounds_ch7.pdf and tau_ch7.pdf, all from results/ch7/users.parquet "
        "and decisions.parquet of the pre-registered run, with its bootstrap seed scheme. Values plotted in tau_ch7:",
        "",
        md(tau_tab),
        "",
        f"rounds_ch7 (Vote-6, k = 1..6): cosine {', '.join(f3(v[0]) for v in vals['cos'])}; share error "
        f"{', '.join(f3(v[0]) for v in vals['err'])}; agreement Adult {', '.join(f3(v[0]) for v in vals['agr'])}; "
        f"expert default {f3(refs['cos'])}, {f3(refs['err'])}, {f3(refs['agr'])}.",
        "",
        "## Runtime",
        "",
        runtime + f"; python -m src.misspec_report: {t_rep:.1f} s.",
        "",
        "## Deviations",
        "",
        "D1. Side-biased users: the task asks for the privacy option to alternate sides \"as in the stored design\", but the "
        "stored design has no side assignment (option A always favors the first criterion of the pair). Both a fixed and an "
        "alternating assignment were run (config_draft3.yaml); neither is the stored design in the sense of an existing "
        "screen layout.",
        "",
        "D2. Regenerated figures: the archive of the main run holds budget_agreement_ch7 and a two-panel rounds_ch7, but no "
        "tau_ch7. rounds_ch7 and tau_ch7 were drawn to the layouts described by the thesis captions (fig:ch7:rounds, three "
        "panels; fig:ch7:tau, two panels) from the stored data. The Uniform default (same eps* as the expert default for "
        "every user) is not among the six names asked for and is not in the regenerated budget_agreement_ch7. The stored "
        "figures in results/ch7/figures/ are not overwritten.",
        "",
        "D3. The worker function is passed to joblib by module reference (commit 2a112d5) after the first launch stopped "
        "with a pickling error before any replication ran; the simulation is unchanged.",
        "",
        "D4. Environment: Python 3.13 with the installed numpy, scipy, pandas and statsmodels instead of the recorded "
        "Python 3.10; the baseline check reproduces the stored estimates to 1.8e-15 and every stored decision.",
    ]
    (out / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
