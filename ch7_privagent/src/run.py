"""Main loop of the Chapter 7 simulation.

    python -m src.run            # unit tests, static checks, all replications, sanity checks, raw outputs
    python -m src.run --smoke    # 2 replications, 100 users -> results/ch7_smoke/

The run stops (non-zero exit) if any sanity check fails.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from . import dialogue as DLG  # noqa: E402
from . import infer as INF  # noqa: E402
from . import uldp as U  # noqa: E402
from . import users as USR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None) -> dict:
    path = Path(path) if path else Path(os.environ.get("CH7_CONFIG", ROOT / "config.yaml"))
    with open(path) as f:
        return yaml.safe_load(f)


def rng_for(cfg, rep, pop_code, name):
    return np.random.default_rng([cfg["seed"], rep, 100 * pop_code + cfg["stream_offsets"][name]])


def populations(cfg, smoke=False):
    """[(name, code, alpha, n, full)]; full = run the sensitivity analyses and the rounds analysis."""
    u = cfg["populations"]["uniform"]
    out = [("uniform", u["code"], u["alpha"], 100 if smoke else u["n"], True)]
    pr = cfg["populations"]["profiles"]
    for name, spec in pr["centers"].items():
        alpha = [pr["concentration"] * c for c in spec["c"]]
        out.append((name, spec["code"], alpha, 25 if smoke else pr["n"], False))
    return out


# ----------------------------------------------------------------------------- one population

def simulate_population(cfg, rep, name, code, alpha, n, full):
    tau0, lam0 = cfg["response"]["tau"], cfg["slider"]["lambda"]
    h_tab, l_tab, k_tab = DLG.contrast_table(cfg)
    bay = cfg["inference"]["bayes"]
    W = USR.draw_weights(rng_for(cfg, rep, code, "true_weights"), alpha, n)
    rows = []

    def add(method, est, analysis="main", tau=np.nan, lam=np.nan, k=np.nan, n_prompts=0, repair=None):
        rows.append(dict(method=method, analysis=analysis, tau=tau, lam=lam, k=k, W=est,
                         n_prompts=np.broadcast_to(np.asarray(n_prompts), (n,)).astype(int),
                         repair=np.zeros(n, bool) if repair is None else repair))

    add("oracle", W.copy())
    for m in ("expert", "uniform_default"):
        add(m, np.tile(np.asarray(cfg["methods"][m]["w"], float), (n, 1)))
    eta = rng_for(cfg, rep, code, "slider_noise").normal(0.0, cfg["slider"]["noise_sd"], size=(n, 3))
    for lam in (cfg["slider"]["lambda_grid"] if full else [lam0]):
        add("slider", USR.anchored_slider(W, lam, cfg["slider"]["anchor"], eta),
            analysis="main" if lam == lam0 else "anchor", lam=lam)
    draws6 = rng_for(cfg, rep, code, "bayes_draws_pa6").dirichlet(bay["prior"], size=bay["draws"])
    drawsL = rng_for(cfg, rep, code, "bayes_draws_lpla").dirichlet(bay["prior"], size=bay["draws"])
    repair_rates = {}
    for tau in (cfg["response"]["tau_grid"] if full else [tau0]):
        analysis = "main" if tau == tau0 else "tau"
        d6 = DLG.run(W, cfg, "pa6", tau, rng_for(cfg, rep, code, "dialogue_pa6"), repair=True)
        dl = DLG.run(W, cfg, "lpla", tau, rng_for(cfg, rep, code, "dialogue_lpla"), repair=False)
        np6 = d6["used"].sum(axis=1)
        npl = dl["used"].sum(axis=1)
        add("pa6", INF.vote(d6, k_tab), analysis, tau=tau, n_prompts=np6, repair=d6["repair"])
        add("pa6_bayes", INF.bayes(d6, h_tab, l_tab, draws6, bay["analyst_tau"]), analysis, tau=tau,
            n_prompts=np6, repair=d6["repair"])
        add("lpla", INF.vote(dl, k_tab), analysis, tau=tau, n_prompts=npl)
        add("lpla_bayes", INF.bayes(dl, h_tab, l_tab, drawsL, bay["analyst_tau"]), analysis, tau=tau,
            n_prompts=npl)
        repair_rates[tau] = float(d6["repair"].mean())
        if full and tau == tau0:
            for k in cfg["prompts"]["rounds"]:
                if k >= d6["n_base"]:
                    continue                      # k = 6 is PrivAgent-6 itself
                dk = DLG.truncate(d6, k)
                add("pa6", INF.vote(dk, k_tab), "rounds", tau=tau, k=k, n_prompts=k)
    sanity = {}
    if full:
        ts = cfg["response"]["tau_sanity"]
        dl0 = DLG.run(W, cfg, "lpla", ts, rng_for(cfg, rep, code, "dialogue_lpla"), repair=False)
        pi = U.privacy_share(W)
        far = np.abs(pi - 0.5) > 0.05
        for nm, est in (("vote", INF.vote(dl0, k_tab)),
                        ("bayes", INF.bayes(dl0, h_tab, l_tab, drawsL, bay["analyst_tau"]))):
            pe = U.privacy_share(est)
            sanity[f"tau0_{nm}_wrong_side"] = int(((pe > 0.5) != (pi > 0.5))[far].sum())
        sanity["tau0_users_checked"] = int(far.sum())
    return W, rows, repair_rates, sanity


# ----------------------------------------------------------------------------- metrics and decisions

def metrics_for(cfg, rep, name, W, rows):
    n = W.shape[0]
    pi_true = U.privacy_share(W)
    settings = [tuple(s) for s in cfg["uldp"]["settings"]]
    truth = {}
    for ds, lp in settings:
        M = U.matrix(ds, lp)
        idx, C = U.select(M, W)
        truth[(ds, lp)] = (M, idx, C)
    urows, drows = [], []
    for r in rows:
        E = r["W"]
        cos = (E * W).sum(1) / (np.linalg.norm(E, axis=1) * np.linalg.norm(W, axis=1))
        pi_e = U.privacy_share(E)
        base = dict(rep=np.full(n, rep), population=np.full(n, name), user=np.arange(n),
                    method=np.full(n, r["method"]), analysis=np.full(n, r["analysis"]),
                    tau=np.full(n, r["tau"]), lam=np.full(n, r["lam"]), k=np.full(n, r["k"]))
        urows.append(pd.DataFrame(dict(
            base, w_lp_true=W[:, 0], w_la_true=W[:, 1], w_cr_true=W[:, 2],
            w_lp=E[:, 0], w_la=E[:, 1], w_cr=E[:, 2], pi_true=pi_true, pi=pi_e,
            cosine=cos, l1=np.abs(E - W).sum(1), share_error=np.abs(pi_e - pi_true),
            n_prompts=r["n_prompts"], repair=r["repair"])))
        for (ds, lp), (M, idx_t, C_t) in truth.items():
            idx_m, _ = U.select(M, E)
            undef = idx_m == U.UNDEFINED
            rows_ = np.arange(n)
            c_best = C_t[rows_, idx_t]
            c_m = np.where(undef, np.nan, C_t[rows_, np.where(undef, 0, idx_m)])
            drows.append(pd.DataFrame(dict(
                base, dataset=np.full(n, ds), lp_map=np.full(n, lp),
                eps_true=U.CANDIDATES[idx_t], eps=np.where(undef, np.nan, U.CANDIDATES[np.where(undef, 0, idx_m)]),
                agree=(idx_m == idx_t), loss=c_best - c_m, undefined=undef)))
    return urows, drows


def run_replication(cfg, rep, smoke=False):
    urows, drows, rr, san = [], [], {}, {}
    for name, code, alpha, n, full in populations(cfg, smoke):
        W, rows, repair_rates, sanity = simulate_population(cfg, rep, name, code, alpha, n, full)
        a, b = metrics_for(cfg, rep, name, W, rows)
        urows += a
        drows += b
        rr[name] = repair_rates
        if sanity:
            san[name] = sanity
    return pd.concat(urows, ignore_index=True), pd.concat(drows, ignore_index=True), rr, san


# ----------------------------------------------------------------------------- static checks

def static_checks(cfg) -> tuple[bool, list[str]]:
    lines, ok = [], True
    exp = np.asarray(cfg["methods"]["expert"]["w"])
    uni = np.asarray(cfg["methods"]["uniform_default"]["w"])
    for ds, lp in cfg["uldp"]["settings"]:
        M = U.matrix(ds, lp)
        a, _ = U.select(M, exp)
        b, _ = U.select(M, uni)
        good = a[0] == b[0]
        ok &= bool(good)
        lines.append(f"{'PASS' if good else 'FAIL'} expert and uniform defaults select the same eps* "
                     f"({ds}, {lp}): {U.CANDIDATES[a[0]]:g} and {U.CANDIDATES[b[0]]:g}")
    # Chapter 7, Table 7.3
    table73 = {("adult", "linear"): ([1, 5, 1, 1], 1), ("adult", "exponential"): ([0.1, 10, 0.1, 0.1], 0.1),
               ("cdc", "linear"): ([1, 5, 1, 1], 1), ("cdc", "exponential"): ([0.1, 10, 0.3, 0.3], 0.3)}
    centers = [spec["c"] for spec in cfg["populations"]["profiles"]["centers"].values()]
    for (ds, lp), (want_prof, want_def) in table73.items():
        M = U.matrix(ds, lp)
        got = [float(U.CANDIDATES[U.select(M, c)[0][0]]) for c in centers]
        got_def = float(U.CANDIDATES[U.select(M, exp)[0][0]])
        good = np.allclose(got, want_prof) and np.isclose(got_def, want_def)
        ok &= bool(good)
        lines.append(f"{'PASS' if good else 'FAIL'} Table 7.3 ({ds}, {lp}): profiles {got} (want {want_prof}), "
                     f"default {got_def:g} (want {want_def:g})")
    return ok, lines


# ----------------------------------------------------------------------------- main

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    smoke = "--smoke" in argv
    t0 = time.time()
    cfg = load_config()
    out = ROOT / (cfg["outputs"]["dir"] + ("_smoke" if smoke else ""))
    out.mkdir(parents=True, exist_ok=True)

    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr
    print(last)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit("unit tests failed: stopping")
    lines = [f"PASS unit tests ({last})"]

    ok, sl = static_checks(cfg)
    lines += sl
    if not ok:
        print("\n".join(lines))
        sys.exit("a static sanity check failed: stopping")

    reps = 2 if smoke else cfg["replications"]
    res = Parallel(n_jobs=cfg["n_jobs"], verbose=5)(delayed(run_replication)(cfg, rep, smoke) for rep in range(reps))
    users = pd.concat([x[0] for x in res], ignore_index=True)
    dec = pd.concat([x[1] for x in res], ignore_index=True)
    repair = pd.DataFrame([dict(rep=i, population=p, tau=t, repair_rate=v)
                           for i, x in enumerate(res) for p, d in x[2].items() for t, v in d.items()])
    tau0 = pd.DataFrame([dict(rep=i, population=p, **s) for i, x in enumerate(res) for p, s in x[3].items()])

    # dynamic sanity checks
    orc = dec[dec.method == "oracle"]
    for (ds, lp), g in orc.groupby(["dataset", "lp_map"]):
        good = bool(g.agree.all()) and float(np.nanmax(np.abs(g.loss))) == 0.0
        ok &= good
        lines.append(f"{'PASS' if good else 'FAIL'} Oracle ({ds}, {lp}): agreement {g.agree.mean():.4f}, "
                     f"max |closeness loss| {np.nanmax(np.abs(g.loss)):.2e}, users {len(g)}")
    ed = dec[(dec.method.isin(["expert", "uniform_default"]))]
    piv = ed.pivot_table(index=["rep", "population", "user", "dataset", "lp_map"], columns="method", values="eps")
    good = bool((piv["expert"] == piv["uniform_default"]).all())
    ok &= good
    lines.append(f"{'PASS' if good else 'FAIL'} expert and uniform defaults give the same eps* for every user "
                 f"and setting ({len(piv)} cases)")
    wv, wb, nchk = int(tau0.tau0_vote_wrong_side.sum()), int(tau0.tau0_bayes_wrong_side.sum()), int(tau0.tau0_users_checked.sum())
    good = wv == 0 and wb == 0
    ok &= good
    lines.append(f"{'PASS' if good else 'FAIL'} tau = {cfg['response']['tau_sanity']:g}, LP-LA prompts: privacy share on "
                 f"the wrong side of 0.5 for {wv} (vote) and {wb} (Bayes) of {nchk} users with |pi - 0.5| > 0.05")
    rr = repair[(repair.population == "uniform")].groupby("tau").repair_rate.mean()
    lines.append("INFO repair rate (PrivAgent-6, uniform population) by tau: "
                 + ", ".join(f"{t:g}: {v:.4f}" for t, v in rr.items()))
    und = dec[dec.analysis == "main"].groupby("method").undefined.mean()
    lines.append("INFO share of undefined selections (main analysis, all populations and settings): "
                 + ", ".join(f"{m}: {v:.4f}" for m, v in und.items()))
    text = "\n".join(lines)
    print(text)
    (out / "sanity.log").write_text(text + "\n")
    if not ok:
        sys.exit("a sanity check failed: stopping (outputs not written)")

    users.to_parquet(out / "users.parquet", index=False)
    dec.to_parquet(out / "decisions.parquet", index=False)
    repair.to_csv(out / "repair.csv", index=False)
    tau0.to_csv(out / "sanity_tau0.csv", index=False)
    import importlib.metadata as im
    used = dict(cfg)
    used["_versions"] = {m: im.version(m) for m in ["numpy", "scipy", "pandas", "statsmodels", "matplotlib",
                                                    "PyYAML", "pytest", "joblib", "pyarrow"]}
    used["_python"] = sys.version
    with open(out / "config_used.yaml", "w") as f:
        yaml.safe_dump(used, f, sort_keys=False)
    shutil.copy(ROOT / "hypotheses.md", out / "hypotheses.md")
    rt = time.time() - t0
    (out / "runtime.txt").write_text(f"python -m src.run{' --smoke' if smoke else ''}: {rt:.1f} s wall clock "
                                     f"({reps} replications, {len(users)} user-method rows, {len(dec)} decisions)\n")
    print(f"done in {rt:.1f} s")


if __name__ == "__main__":
    main()
