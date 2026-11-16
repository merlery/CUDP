"""Chapter 7, Draft 3 follow-up (EXPLORATORY): simulated users who do not follow the response model of the
inference. Settings: config_draft3.yaml. The pre-registered run (src/run.py, results/ch7/) is not changed.

    python -m src.misspec            # all populations, baseline check -> results/ch7/draft3/misspec_users.parquet
    python -m src.misspec_report     # tables, figures, summary.md
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.special import expit, logit  # noqa: E402

from . import dialogue as DLG  # noqa: E402
from . import infer as INF  # noqa: E402
from . import uldp as U  # noqa: E402
from . import users as USR  # noqa: E402
from .run import ROOT, load_config, rng_for  # noqa: E402

METHODS = ["pa6", "pa6_bayes", "lpla", "lpla_bayes", "expert", "slider"]


def load_d3():
    with open(ROOT / "config_draft3.yaml") as f:
        return yaml.safe_load(f)


def prob_A(W, pair_idx, h, l, tau, spec, position, affected):
    """P(user answers A) for one prompt per user. pair_idx (N,), position (N,) 1-based presented position."""
    N = W.shape[0]
    pj = np.array([p[0] for p in DLG.PAIRS])[pair_idx]
    pk = np.array([p[1] for p in DLG.PAIRS])[pair_idx]
    rows = np.arange(N)
    diff = (W[rows, pj] - W[rows, pk]) * (h - l)
    rule = spec["rule"]
    if rule == "lexicographic":
        return (W[rows, pj] > W[rows, pk]).astype(float)
    p = expit(diff / tau)
    if rule == "logistic":
        return p
    if rule == "lapse":
        return (1 - spec["p"]) * p + spec["p"] / 2
    if rule == "privacy_always":
        involves_lp = pj == 0                        # pairs LP-LA and LP-CR (LP is criterion j)
        return np.where(affected & involves_lp, 1.0, p)
    if rule == "side_bias":
        b = spec["b"]
        if spec["sides"] == "fixed":
            a_first = np.ones(N, bool)
        else:
            a_first = (position % 2 == 1)            # A shown first at positions 1, 3, 5 (and 7 = repair)
        z = diff / tau
        return np.where(a_first, expit(z + b), expit(z - b))
    raise ValueError(rule)


def run_dialogue(W, cfg, set_name, tau, rng, repair, spec, affected):
    """src/dialogue.run with the response rule of the population; identical draw order (prompt order, P uniform
    answer draws, one repair draw), so the logistic rule reproduces the stored dialogue exactly."""
    N = W.shape[0]
    base = DLG.prompt_set(cfg, set_name)
    P = base.shape[0]
    order = rng.permuted(np.tile(np.arange(P), (N, 1)), axis=1)
    u = rng.random((N, P))
    u_rep = rng.random(N)
    h_tab, l_tab, k_tab = DLG.contrast_table(cfg)
    pair = base[order, 0]
    con = base[order, 1]
    ans = np.zeros((N, P), dtype=np.int8)
    for s in range(P):
        pa = prob_A(W, pair[:, s], h_tab[con[:, s]], l_tab[con[:, s]], tau, spec, np.full(N, s + 1), affected)
        ans[:, s] = (u[:, s] < pa).astype(np.int8)
    pair_all = np.concatenate([pair, np.zeros((N, 1), int)], axis=1)
    con_all = np.concatenate([con, np.zeros((N, 1), int)], axis=1)
    ans_all = np.concatenate([ans, np.zeros((N, 1), np.int8)], axis=1)
    used = np.concatenate([np.ones((N, P), bool), np.zeros((N, 1), bool)], axis=1)
    triggered = np.zeros(N, bool)
    if repair:
        res = DLG.pair_results(pair, ans)
        triggered = DLG.is_cycle(res)
        rp = DLG.repair_pair(res)
        strong = DLG.CONTRAST_NAMES.index(cfg["prompts"]["repair"]["contrast"])
        pair_all[:, P] = rp
        con_all[:, P] = strong
        pr = prob_A(W, rp, h_tab[strong], l_tab[strong], tau, spec, np.full(N, P + 1), affected)
        ans_all[:, P] = (u_rep < pr).astype(np.int8)
        used[:, P] = triggered
    return dict(pair=pair_all, contrast=con_all, answer=ans_all, used=used, repair=triggered, n_base=P)


def simulate(cfg, d3, rep, pname, spec):
    n = d3["n_users"]
    tau, tau_a = d3["tau"], d3["analyst_tau"]
    bay = cfg["inference"]["bayes"]
    h_tab, l_tab, k_tab = DLG.contrast_table(cfg)
    code0 = cfg["populations"]["uniform"]["code"]
    W = USR.draw_weights(rng_for(cfg, rep, code0, "true_weights"), cfg["populations"]["uniform"]["alpha"], n)
    affected = np.zeros(n, bool)
    if spec["rule"] == "privacy_always":
        k = int(round(spec["f"] * n))
        idx = np.random.default_rng([cfg["seed"], rep, 100 * spec["code"] + 10]).choice(n, size=k, replace=False)
        affected[idx] = True
    eta = rng_for(cfg, rep, code0, "slider_noise").normal(0.0, cfg["slider"]["noise_sd"], size=(n, 3))
    draws6 = rng_for(cfg, rep, code0, "bayes_draws_pa6").dirichlet(bay["prior"], size=bay["draws"])
    drawsL = rng_for(cfg, rep, code0, "bayes_draws_lpla").dirichlet(bay["prior"], size=bay["draws"])
    d6 = run_dialogue(W, cfg, "pa6", tau, rng_for(cfg, rep, code0, "dialogue_pa6"), True, spec, affected)
    dl = run_dialogue(W, cfg, "lpla", tau, rng_for(cfg, rep, code0, "dialogue_lpla"), False, spec, affected)
    est = {"pa6": INF.vote(d6, k_tab), "pa6_bayes": INF.bayes(d6, h_tab, l_tab, draws6, tau_a),
           "lpla": INF.vote(dl, k_tab), "lpla_bayes": INF.bayes(dl, h_tab, l_tab, drawsL, tau_a),
           "expert": np.tile(np.asarray(cfg["methods"]["expert"]["w"], float), (n, 1)),
           "slider": USR.anchored_slider(W, cfg["slider"]["lambda"], cfg["slider"]["anchor"], eta)}
    pi_true = U.privacy_share(W)
    truth = {(ds, lp): U.select(U.matrix(ds, lp), W) for ds, lp in map(tuple, d3["metrics"]["settings"])}
    frames = []
    for m in METHODS:
        E = est[m]
        cos = (E * W).sum(1) / (np.linalg.norm(E, axis=1) * np.linalg.norm(W, axis=1))
        f = dict(rep=rep, population=pname, user=np.arange(n), method=m, affected=affected,
                 w_lp=E[:, 0], w_la=E[:, 1], w_cr=E[:, 2], w_lp_true=W[:, 0], w_la_true=W[:, 1], w_cr_true=W[:, 2],
                 cosine=cos, share_error=np.abs(U.privacy_share(E) - pi_true),
                 repair=d6["repair"] if m.startswith("pa6") else np.zeros(n, bool))
        for (ds, lp), (idx_t, C_t) in truth.items():
            idx_m, _ = U.select(U.matrix(ds, lp), E)
            undef = idx_m == U.UNDEFINED
            rows = np.arange(n)
            f[f"agree_{ds}"] = idx_m == idx_t
            f[f"loss_{ds}"] = C_t[rows, idx_t] - np.where(undef, np.nan, C_t[rows, np.where(undef, 0, idx_m)])
            f[f"eps_{ds}"] = np.where(undef, np.nan, U.CANDIDATES[np.where(undef, 0, idx_m)])
        frames.append(pd.DataFrame(f))
    return pd.concat(frames, ignore_index=True)


def run_rep(cfg, d3, rep):
    return pd.concat([simulate(cfg, d3, rep, p, s) for p, s in d3["populations"].items()], ignore_index=True)


def baseline_check(cfg, df) -> list[str]:
    st = pd.read_parquet(ROOT / "results/ch7/users.parquet")
    st = st[(st.population == "uniform") & (st.analysis == "main") & st.method.isin(METHODS)]
    sd = pd.read_parquet(ROOT / "results/ch7/decisions.parquet")
    sd = sd[(sd.population == "uniform") & (sd.analysis == "main") & sd.method.isin(METHODS) & (sd.lp_map == "linear")]
    b = df[df.population == "baseline"]
    key = ["rep", "user", "method"]
    j = b.merge(st, on=key, suffixes=("", "_st"))
    lines = []
    ok_w = len(j) == len(b) == len(st) and all(np.max(np.abs(j[c] - j[c + "_st"])) <= 1e-12 for c in ("w_lp", "w_la", "w_cr"))
    lines.append(f"{'PASS' if ok_w else 'FAIL'} baseline: estimated weights equal users.parquet (uniform, main) for "
                 f"{len(j)} of {len(st)} user-method rows (max abs difference "
                 f"{max(float(np.max(np.abs(j[c] - j[c + '_st']))) for c in ('w_lp', 'w_la', 'w_cr')):.1e})")
    ok_r = bool((j.repair == j.repair_st).all())
    lines.append(f"{'PASS' if ok_r else 'FAIL'} baseline: repair flags equal users.parquet")
    ok_d = True
    for ds in ("adult", "cdc"):
        g = sd[sd.dataset == ds].merge(b, on=key)
        same = (g.agree == g[f"agree_{ds}"]) & ((g.eps == g[f"eps_{ds}"]) | (g.eps.isna() & g[f"eps_{ds}"].isna()))
        ok = bool(same.all()) and len(g) == len(b)
        ok_d &= ok
        lines.append(f"{'PASS' if ok else 'FAIL'} baseline: decisions ({ds}, linear) equal decisions.parquet for "
                     f"{int(same.sum())} of {len(g)} rows")
    return lines, ok_w and ok_r and ok_d


def main(argv=None):
    t0 = time.time()
    cfg = load_config()
    d3 = load_d3()
    out = ROOT / d3["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    from src.misspec import run_rep as run_rep_ref        # pickled by reference, not from __main__
    res = Parallel(n_jobs=cfg["n_jobs"], verbose=0)(delayed(run_rep_ref)(cfg, d3, r) for r in range(d3["replications"]))
    df = pd.concat(res, ignore_index=True)
    lines, ok = baseline_check(cfg, df)
    a = df[df.population.str.startswith("privacy")].groupby("population").affected.mean()
    lines.append("INFO share of privacy-always users: " + ", ".join(f"{k}: {v:.3f}" for k, v in a.items()))
    rr = df[df.method == "pa6"].groupby("population").repair.mean()
    lines.append("INFO repair rate (Vote-6 dialogue): " + ", ".join(f"{k}: {v:.4f}" for k, v in rr.items()))
    text = "\n".join(lines)
    print(text)
    (out / "sanity_misspec.log").write_text(text + "\n")
    if not ok:
        sys.exit("baseline check failed: stopping")
    df.to_parquet(out / "misspec_users.parquet", index=False)
    rt = time.time() - t0
    (out / "runtime_misspec.txt").write_text(f"python -m src.misspec: {rt:.1f} s wall clock ({d3['replications']} "
                                             f"replications x {len(d3['populations'])} populations x {d3['n_users']} users)\n")
    print(f"done in {rt:.1f} s")


if __name__ == "__main__":
    main()
