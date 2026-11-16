"""Cluster bootstrap, McNemar, Wilcoxon, Holm, and the H1-H5 decisions (pre-registered)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


# ----------------------------------------------------------------------------- primitives

def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def cluster_ci(values: np.ndarray, reps: np.ndarray, B: int, seed, ci: float = 0.95):
    """Pooled mean over all users and its percentile interval from a bootstrap that resamples whole
    replications (clusters) with replacement. NaN values are dropped (e.g. undefined closeness loss)."""
    values = np.asarray(values, float)
    reps = np.asarray(reps)
    ok = np.isfinite(values)
    values, reps = values[ok], reps[ok]
    ids = np.unique(reps)
    sums = np.array([values[reps == r].sum() for r in ids])
    cnts = np.array([(reps == r).sum() for r in ids], float)
    point = sums.sum() / cnts.sum() if cnts.sum() else np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    a = (1 - ci) / 2
    return float(point), float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def mcnemar_test(a: np.ndarray, b: np.ndarray):
    """Paired binary outcomes. Returns (n_a_only, n_b_only, p)."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    table = [[int((a & b).sum()), n10], [n01, int((~a & ~b).sum())]]
    if n10 + n01 == 0:
        return n10, n01, 1.0
    exact = (n10 + n01) < 25
    res = mcnemar(table, exact=exact, correction=True)
    return n10, n01, float(res.pvalue)


def wilcoxon_test(diff: np.ndarray) -> float:
    diff = np.asarray(diff, float)
    if np.all(diff == 0):
        return 1.0
    return float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)


# ----------------------------------------------------------------------------- data access

def main_rows(users: pd.DataFrame, dec: pd.DataFrame, population: str = "uniform"):
    u = users[(users.population == population) & (users.analysis == "main")]
    d = dec[(dec.population == population) & (dec.analysis == "main")]
    return u, d


def paired(frame: pd.DataFrame, a: str, b: str, value: str, extra=None):
    """Align method a and method b per (rep, user) [and extra filter]; returns rep, va, vb."""
    fa = frame[frame.method == a].set_index(["rep", "user"])[value]
    fb = frame[frame.method == b].set_index(["rep", "user"])[value]
    j = fa.index.intersection(fb.index)
    return np.asarray(j.get_level_values(0)), fa.loc[j].to_numpy(), fb.loc[j].to_numpy()


# ----------------------------------------------------------------------------- families

def family_tests(users: pd.DataFrame, dec: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    st = cfg["stats"]
    u, d = main_rows(users, dec)
    rows = []
    for fi, (fam, spec) in enumerate(st["families"].items()):
        cells = []
        if spec["metric"] == "agreement":
            for ds, lp in spec["settings"]:
                g = d[(d.dataset == ds) & (d.lp_map == lp)]
                reps, va, vb = paired(g, spec["a"], spec["b"], "agree")
                n10, n01, p = mcnemar_test(va, vb)
                cells.append(dict(dataset=ds, lp_map=lp, reps=reps, va=va.astype(float), vb=vb.astype(float),
                                  n_a_only=n10, n_b_only=n01, p_raw=p, test="McNemar"))
        else:
            reps, va, vb = paired(u, spec["a"], spec["b"], spec["metric"])
            p = wilcoxon_test(va - vb)
            cells.append(dict(dataset="", lp_map="", reps=reps, va=va, vb=vb, n_a_only=int((va > vb).sum()),
                              n_b_only=int((va < vb).sum()), p_raw=p, test="Wilcoxon"))
        adj = holm([c["p_raw"] for c in cells])
        for ci, (c, pa) in enumerate(zip(cells, adj)):
            seed = [st["bootstrap_seed"], fi, ci]
            ma = cluster_ci(c["va"], c["reps"], st["bootstrap_resamples"], seed + [0], st["ci"])
            mb = cluster_ci(c["vb"], c["reps"], st["bootstrap_resamples"], seed + [1], st["ci"])
            md = cluster_ci(c["va"] - c["vb"], c["reps"], st["bootstrap_resamples"], seed + [2], st["ci"])
            sig = pa < st["alpha"]
            want = spec["direction"]
            in_dir = sig and ((md[0] > 0) if want == "greater" else (md[0] < 0))
            rows.append(dict(family=fam, hypothesis=spec["hypothesis"], a=spec["a"], b=spec["b"],
                             metric=spec["metric"], dataset=c["dataset"], lp_map=c["lp_map"], test=c["test"],
                             n=len(c["va"]), mean_a=ma[0], a_lo=ma[1], a_hi=ma[2], mean_b=mb[0], b_lo=mb[1],
                             b_hi=mb[2], diff=md[0], diff_lo=md[1], diff_hi=md[2], n_a_only=c["n_a_only"],
                             n_b_only=c["n_b_only"], p_raw=c["p_raw"], p_holm=float(pa),
                             direction=want, significant=bool(sig), in_direction=bool(in_dir)))
    return pd.DataFrame(rows)


def decide(tests: pd.DataFrame) -> dict:
    """pass = every test of the family significant in the stated direction; partial = some; fail = none."""
    out = {}
    for h, g in tests.groupby("hypothesis"):
        k = int(g.in_direction.sum())
        oc = "pass" if k == len(g) else ("partial" if k > 0 else "fail")
        out[h] = dict(outcome=oc, family=g.family.iloc[0], tests=len(g), in_direction=k)
    return out
