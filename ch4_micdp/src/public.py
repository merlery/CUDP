"""Public-data baselines for Chapter 4 (MIC-DP): Public-Only, Public-Pretrain + FT, pooled training, and
MIC-Public with smaller or shifted public splits.

    python -m src.public            # unit tests, sanity checks, all (dataset, trial) units, raw outputs
    python -m src.public_report     # stats_public.csv, tables, figure, summary.md

Nothing in the existing pipeline is changed: the releases of Uniform and MIC-DP are regenerated with the
streams of the main run ([seed, method_id, eps_index]) through the unchanged functions of src/methods.py
and src/dp.py, and the models are those of src/run.fit_eval. The run stops if any sanity check fails.
"""
from __future__ import annotations

import copy
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from . import data as D
from . import dp
from . import fgmic as F
from .methods import allocate
from .noiseaware import predict_noise_aware
from .run import _release_and_eval, fit_eval
from .scores import v2_statistics

warnings.filterwarnings("ignore")


class SanityError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- models

class ContinuedHGBClassifier(HistGradientBoostingClassifier):
    """HistGradientBoostingClassifier whose warm-start continuation on NEW data starts from the
    ensemble's actual raw predictions on those rows.

    sklearn refits the bin mapper in every call to fit and, on a warm start, computes the starting raw
    predictions of the existing trees from the new binned data with the old trees' bin thresholds, which
    is only valid when the data are the same. Here the first call to _raw_predict during a warm-started
    fit (the starting predictions) is computed from the unbinned new rows with the numeric thresholds.
    On identical data both are equal, so the result equals one longer fit."""

    def fit(self, X, y, sample_weight=None):
        cont = self.warm_start and hasattr(self, "_predictors") and len(getattr(self, "_predictors", [])) > 0
        self._ws_X = np.asarray(X, dtype=float) if cont else None
        try:
            return super().fit(X, y, sample_weight)
        finally:
            self._ws_X = None

    def _raw_predict(self, X, n_threads=None):
        Xr = getattr(self, "_ws_X", None)
        if getattr(self, "_in_fit", False) and Xr is not None and X.shape[0] == Xr.shape[0]:
            self._ws_X = None                       # only the starting predictions
            self._in_fit = False
            try:
                return super()._raw_predict(Xr, n_threads)
            finally:
                self._in_fit = True
        return super()._raw_predict(X, n_threads)


def evaluate(model, Xte, yte, sigma=None) -> dict:
    """As run.fit_eval (classification): noise-aware prediction for the primary model if sigma is given."""
    p = predict_noise_aware(model, Xte, sigma, "classification") if sigma is not None else model.predict_proba(Xte)[:, 1]
    return {"model_name": type(model).__name__, "roc_auc": float(roc_auc_score(yte, p)),
            "accuracy": float(accuracy_score(yte, (p >= 0.5).astype(int))), "n_unique_pred": int(np.unique(p).size)}


def primary_model(cfg, seed, max_iter=None, warm=False):
    mc = cfg["models"]["primary"]
    cls = ContinuedHGBClassifier if warm else HistGradientBoostingClassifier
    return cls(max_iter=max_iter or mc["max_iter"], early_stopping=mc["early_stopping"], random_state=seed,
               warm_start=warm)


def secondary_model(cfg, warm=False):
    s = cfg["models"]["secondary"]["classification"]
    return LogisticRegression(C=s["C"], max_iter=s["max_iter"], warm_start=warm)


# ----------------------------------------------------------------------------- public splits

def public_subsets(p, cfg, name) -> dict:
    """Positions (into the public split) of p15, p5, p1 and shift5."""
    pb = cfg["public_baselines"]
    n = p.n_total
    pos15 = np.arange(len(p.idx_public))
    n5, n1 = int(round(pb["sizes"]["p5"] * n)), int(round(pb["sizes"]["p1"] * n))
    s = pb["public_subsample_seed"]
    pos5 = np.sort(train_test_split(pos15, train_size=n5, stratify=p.y_public, random_state=s)[0])
    pos1 = np.sort(train_test_split(pos5, train_size=n1, stratify=p.y_public[pos5], random_state=s + 1)[0])
    cov = pb["shift"]["covariate"][name]
    j = p.features.index(cov)
    raw = p.a[j] + p.X_public[:, j] * (p.b[j] - p.a[j])
    med = float(np.median(raw))
    pool = pos15[raw <= med + 1e-9]
    nsh = int(round(pb["shift"]["size"] * n))
    posS = np.sort(np.random.default_rng(pb["shift_seed"]).choice(pool, size=nsh, replace=False))
    return {"p15": pos15, "p5": pos5, "p1": pos1, "shift5": posS, "_median": med, "_pool": len(pool),
            "_cov_raw": raw}


def fgmic_on(X, y, task, cfg) -> np.ndarray:
    """FG-MIC of every feature on (X, y), with candidate cut points and B(n) from the same data."""
    fc = cfg["scores"]["fgmic"]
    x_cuts = [F.public_cuts(X[:, j], fc["n_x_bins"]) for j in range(X.shape[1])]
    B = max(len(y) ** fc["alpha_B"], 4.0)
    rows = F.row_partitions(task, y, B, fc["l_max"])
    return F.fgmic_scores(X, y, x_cuts, rows, fc["alpha_B"])


# ----------------------------------------------------------------------------- releases

def release(ctx, name, eps, ei, seed, cfg):
    """The release of an existing method exactly as in run.run_main_trial (stream [seed, id, ei])."""
    m = cfg["methods"]["list"][name]
    rng = np.random.default_rng([seed, m["id"], ei])
    d = ctx["X_train"].shape[1]
    al = allocate(name, m, eps, cfg["privacy"]["delta"], ctx["raw"], d, rng, dict(cfg["methods"]["defaults"]),
                  score_rng=None, protected_idx=[])
    Xt, sigma, keep = dp.weighted_gaussian_subset(ctx["X_train"], al.w, al.eps_m, al.delta_m, rng)
    return al, Xt, sigma, keep, rng


# ----------------------------------------------------------------------------- one unit

def build_context(name, cfg):
    p = D.prepare(name, cfg)
    raw = v2_statistics(p, cfg)
    subs = public_subsets(p, cfg, name)
    fg = {k: fgmic_on(p.X_public[subs[k]], p.y_public[subs[k]], p.task, cfg) for k in ("p5", "p1", "shift5")}
    fg["p15"] = fgmic_on(p.X_public, p.y_public, p.task, cfg)
    return {"name": name, "task": p.task, "features": p.features, "X_train": p.X_train, "y_train": p.y_train,
            "X_test": p.X_test, "y_test": p.y_test, "X_public": p.X_public, "y_public": p.y_public,
            "idx_public": p.idx_public, "idx_train": p.idx_train, "idx_test": p.idx_test, "raw": raw,
            "subs": subs, "fg": fg, "n_total": p.n_total}


def run_unit(ctx, trial, cfg, check_existing=False, stored=None):
    seed = cfg["seeds"]["trial_base"] + trial
    pb = cfg["public_baselines"]
    ds = ctx["name"]
    Xtr, ytr, Xte, yte = ctx["X_train"], ctx["y_train"], ctx["X_test"], ctx["y_test"]
    Xp_all, yp_all = ctx["X_public"], ctx["y_public"]
    d = Xtr.shape[1]
    h_te, h_tr = D.array_hash(Xte), D.array_hash(Xtr)
    train_set = set(ctx["idx_train"].tolist())
    rows, checks = [], []
    base = {"dataset": ds, "trial": trial, "seed": seed}

    def add(method, size, kind, eps, ei, r, **extra):
        rows.append({**base, "method": method, "public_size": size, "model": kind, "eps": eps, "eps_index": ei,
                     **r, **extra})

    # ---- Public-Only
    po_primary, po_secondary = {}, {}
    for size in pb["methods"]["public_only"]["sizes"]:
        pos = ctx["subs"][size]
        Xp, yp = Xp_all[pos], yp_all[pos]
        # never reads a private-training record: the fit matrix is exactly X_public[pos], and the rows are
        # public rows (global indices disjoint from private-train)
        gidx = ctx["idx_public"][pos]
        ok = (D.array_hash(Xp) == D.array_hash(np.ascontiguousarray(Xp_all[pos]))) and not (set(gidx.tolist()) & train_set)
        checks.append(("public_only_reads_no_private_row", ds, trial, size, bool(ok)))
        m1 = primary_model(cfg, seed).fit(Xp, yp)
        po_primary[size] = m1
        add("public_only", size, "primary", np.nan, -1, evaluate(m1, Xte, yte), fit_rows_hash=D.array_hash(Xp))
        m2 = secondary_model(cfg).fit(Xp, yp)
        po_secondary[size] = m2
        add("public_only", size, "secondary", np.nan, -1, evaluate(m2, Xte, yte), fit_rows_hash=D.array_hash(Xp))
    stage100 = {}
    for size, m in po_primary.items():
        for k, dfun in enumerate(m.staged_decision_function(Xte), start=1):
            if k == 100:
                stage100[size] = np.asarray(dfun, dtype=float).ravel()
                break

    # ---- existing methods, re-run (trial 0 only)
    if check_existing:
        st = stored[(stored.dataset == ds) & (stored.trial == trial)]
        for kind in ("primary", "secondary"):
            r = fit_eval(kind, ctx["task"], Xtr, ytr, Xte, yte, seed, cfg)
            s = st[(st.method == "nonprivate") & (st.model == kind)].iloc[0]
            checks.append(("existing_reproduced", ds, trial, f"nonprivate/{kind}",
                           bool(abs(r["roc_auc"] - s.roc_auc) <= 1e-9 and abs(r["accuracy"] - s.accuracy) <= 1e-9)))
        for ei, eps in enumerate(pb["eps_grid"]):
            for name in ("uniform", "mic_dp", "mic_public"):
                al, Xt, sigma, keep, _ = release(ctx, name, eps, ei, seed, cfg)
                for kind in ("primary", "secondary"):
                    r = fit_eval(kind, ctx["task"], Xt, ytr, Xte[:, keep], yte, seed, cfg, sigma=sigma[keep])
                    s = st[(st.method == name) & (st.model == kind) & np.isclose(st.eps, eps)].iloc[0]
                    good = abs(r["roc_auc"] - s.roc_auc) <= 1e-9 and abs(r["accuracy"] - s.accuracy) <= 1e-9
                    checks.append(("existing_reproduced", ds, trial, f"{name}/{kind}/eps={eps:g}", bool(good)))

    # ---- per eps
    for ei, eps in enumerate(pb["eps_grid"]):
        rels = {}
        for name in ("uniform", "mic_dp"):
            al, Xt, sigma, keep, _ = release(ctx, name, eps, ei, seed, cfg)
            rels[name] = (al, Xt, sigma, keep)
        for meth, relname in (("ft_uniform", "uniform"), ("ft_mic_dp", "mic_dp")):
            al, Xt, sigma, keep = rels[relname]
            for size in pb["methods"][meth]["sizes"]:
                pos = ctx["subs"][size]
                Xp, yp = Xp_all[pos][:, keep], yp_all[pos]
                # primary: pretrain 100 iterations on public, continue 100 on the release
                pre = primary_model(cfg, seed, max_iter=100, warm=True).fit(Xp, yp)
                dpre = np.asarray(pre.decision_function(Xte[:, keep]), dtype=float).ravel()
                if keep.all():
                    ref = stage100[size]
                else:
                    refm = primary_model(cfg, seed).fit(Xp, yp)
                    ref = [np.asarray(v, float).ravel() for k, v in enumerate(refm.staged_decision_function(Xte[:, keep]), 1) if k == 100][0]
                checks.append(("pretrained_reproduces_public_only", ds, trial, f"{meth}/{size}/eps={eps:g}/primary",
                               bool(np.max(np.abs(dpre - ref)) <= 1e-9)))
                pre.set_params(max_iter=200)
                pre.fit(Xt, ytr)
                add(meth, size, "primary", eps, ei, evaluate(pre, Xte[:, keep], yte, sigma[keep]),
                    n_released=int(keep.sum()), n_trees=len(pre._predictors))
                # secondary: logistic regression, warm start
                lr = secondary_model(cfg, warm=True).fit(Xp, yp)
                if keep.all():
                    a0 = evaluate(lr, Xte, yte)["roc_auc"]
                    a1 = evaluate(po_secondary[size], Xte, yte)["roc_auc"]
                    same = np.max(np.abs(lr.coef_ - po_secondary[size].coef_)) <= 1e-9 and abs(a0 - a1) <= 1e-9
                    checks.append(("pretrained_reproduces_public_only", ds, trial, f"{meth}/{size}/eps={eps:g}/secondary", bool(same)))
                lr.fit(Xt, ytr)
                add(meth, size, "secondary", eps, ei, evaluate(lr, Xte[:, keep], yte), n_released=int(keep.sum()))
        # pooled: clean public rows + Uniform release
        al, Xt, sigma, keep = rels["uniform"]
        for size in pb["methods"]["pooled_uniform"]["sizes"]:
            pos = ctx["subs"][size]
            Xpool = np.vstack([Xp_all[pos][:, keep], Xt])
            ypool = np.concatenate([yp_all[pos], ytr])
            m1 = primary_model(cfg, seed).fit(Xpool, ypool)
            add("pooled_uniform", size, "primary", eps, ei, evaluate(m1, Xte[:, keep], yte, sigma[keep]),
                n_released=int(keep.sum()))
            m2 = secondary_model(cfg).fit(Xpool, ypool)
            add("pooled_uniform", size, "secondary", eps, ei, evaluate(m2, Xte[:, keep], yte), n_released=int(keep.sum()))
        del rels
        # MIC-Public with scores from the smaller / shifted public split
        mcfg = cfg["methods"]["list"]["mic_public"]
        for size, mid in pb["methods"]["mic_public_small"]["ids"].items():
            raw_s = dict(ctx["raw"], fgmic_public=ctx["fg"][size])
            rng = np.random.default_rng([seed, mid, ei])
            al = allocate("mic_public", mcfg, eps, cfg["privacy"]["delta"], raw_s, d, rng,
                          dict(cfg["methods"]["defaults"]))
            res, sigma, keep, dist, _ = _release_and_eval(ctx, al, rng, seed, cfg, ("primary", "secondary"))
            for kind, r in res.items():
                add("mic_public", size, kind, eps, ei, {k: r[k] for k in ("model_name", "roc_auc", "accuracy", "n_unique_pred")},
                    n_released=int(keep.sum()), w=al.w.tolist(), distortion=dist)
    if D.array_hash(Xte) != h_te or D.array_hash(Xtr) != h_tr:
        raise SanityError(f"{ds}: train or test data changed during trial {trial}")
    return rows, checks


def _task(name, trial, cfg, cache_file, check_existing, stored):
    t0 = time.time()
    ctx = CTX_CACHE.get(name)
    if ctx is None:
        ctx = CTX_CACHE[name] = build_context(name, cfg)
    rows, checks = run_unit(ctx, trial, cfg, check_existing, stored)
    with open(cache_file + ".tmp", "wb") as f:
        pickle.dump((rows, checks, time.time() - t0), f)
    os.replace(cache_file + ".tmp", cache_file)
    return name, trial, time.time() - t0


CTX_CACHE: dict = {}


# ----------------------------------------------------------------------------- main

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    t0 = time.time()
    cfg = D.load_config()
    pb = cfg["public_baselines"]
    out = D.ROOT / pb["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    for f in cache.glob("*.tmp"):
        f.unlink()
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    def check(cond, msg):
        log(("[PASS] " if cond else "[FAIL] ") + msg)
        if not cond:
            (out / "sanity_public.log").write_text("\n".join(log_lines) + "\n")
            raise SanityError(msg)

    import pytest
    rc = pytest.main(["-q", "-p", "no:cacheprovider", "-m", "not crosscheck", str(D.ROOT / "tests")])
    check(rc == 0, f"all unit tests pass (pytest exit code {rc})")
    D_ = {}
    for name in pb["datasets"]:
        p = D.prepare(name, cfg)
        D_[name] = p
        hs = pb["split_hashes"][name]
        for k in ("X_train", "y_train", "X_test", "y_test", "idx_train", "idx_test", "X_public"):
            check(D.array_hash(getattr(p, k)) == hs[k], f"{name}: {k} byte-identical to the split of the main run (SHA-256)")
        subs = public_subsets(p, cfg, name)
        s15, s5, s1, sS = (set(subs[k].tolist()) for k in ("p15", "p5", "p1", "shift5"))
        check(s1 <= s5 <= s15, f"{name}: p1 ({len(s1)}) subset of p5 ({len(s5)}) subset of p15 ({len(s15)})")
        nr = pb["n_records"][name]
        check(len(s15) == nr["p15"] and len(s5) == nr["p5"] and len(s1) == nr["p1"] and len(sS) == nr["shift5"],
              f"{name}: public sizes {len(s15)}, {len(s5)}, {len(s1)}, shifted {len(sS)} as in config.yaml")
        med = pb["shift"]["median"][name]
        check(abs(subs["_median"] - med) < 1e-9 and subs["_pool"] == pb["shift"]["pool_size"][name]
              and sS <= s15 and bool(np.all(subs["_cov_raw"][subs["shift5"]] <= med + 1e-9)),
              f"{name}: shifted split = {len(sS)} public records with {pb['shift']['covariate'][name]} <= "
              f"{med:g} (pool {subs['_pool']})")
        ctx0 = build_context(name, cfg)
        check(np.array_equal(ctx0["fg"]["p15"], ctx0["raw"]["fgmic_public"]),
              f"{name}: FG-MIC of the 15% split computed by the new code equals the scores of the existing MIC-Public")
        del ctx0
        gp = set(p.idx_public.tolist())
        check(not (gp & set(p.idx_train.tolist())) and not (gp & set(p.idx_test.tolist())),
              f"{name}: public split disjoint from private-train and test")

    stored = pd.read_parquet(D.ROOT / cfg["outputs"]["dir"] / "raw_trials.parquet")
    stored = stored[(stored.experiment == "main") & stored.dataset.isin(pb["datasets"]) & (stored.trial == 0)]
    jobs = []
    for name in pb["datasets"]:
        for t in range(pb["n_trials"]):
            jobs.append((name, t, str(cache / f"{name}_{t:02d}.pkl"), t == 0))
    todo = [j for j in jobs if not os.path.exists(j[2])]
    log(f"== running {len(todo)} of {len(jobs)} units (rest cached), n_jobs = {pb['n_jobs']}")
    t_run = time.time()
    if todo:
        par = Parallel(n_jobs=pb["n_jobs"], backend="loky", return_as="generator_unordered")
        done = 0
        for name, t, dt in par(delayed(_task)(n, t, cfg, f, ce, stored) for n, t, f, ce in todo):
            done += 1
            el = time.time() - t_run
            log(f"  {done}/{len(todo)} units | {name} t={t} ({dt:.0f} s) | elapsed {el/60:.1f} min | "
                f"ETA {(len(todo)-done)*el/done/60:.1f} min")
    rows, checks, unit_s = [], [], []
    for _, _, f, _ in jobs:
        with open(f, "rb") as fh:
            r, c, dt = pickle.load(fh)
        rows += r
        checks += c
        unit_s.append(dt)
    ch = pd.DataFrame(checks, columns=["check", "dataset", "trial", "item", "ok"])
    for c, g in ch.groupby("check"):
        bad = g[~g.ok]
        check(len(bad) == 0, f"{c}: {int(g.ok.sum())} of {len(g)} checks pass"
              + ("" if len(bad) == 0 else f"; failing: {bad.head(5).to_dict('records')}"))
    df = pd.DataFrame(rows)
    df.to_parquet(out / "raw_trials_public.parquet", index=False)
    ch.to_csv(out / "sanity_checks_public.csv", index=False)
    import importlib.metadata as im
    used = {**cfg, "_run_public": {"python": sys.version.split()[0],
                                   **{m: im.version(m) for m in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib")}}}
    with open(out / "config_used.yaml", "w") as f:
        yaml.safe_dump(used, f, sort_keys=False)
    total = time.time() - t0
    (out / "runtime_public.txt").write_text(
        f"python -m src.public: {total/60:.1f} min wall clock ({total:.0f} s); units run this time: {len(todo)}; "
        f"sum of unit times {sum(unit_s)/60:.1f} min (2 workers)\n")
    log(f"wrote raw_trials_public.parquet ({len(df)} rows); total {total/60:.1f} min")
    (out / "sanity_public.log").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    try:
        main()
    except SanityError as e:
        print(f"\nRUN STOPPED: sanity check failed: {e}", file=sys.stderr)
        sys.exit(1)
