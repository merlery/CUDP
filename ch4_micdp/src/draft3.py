"""Draft 3 follow-up runs for Chapter 4 (MIC-DP): refits (Parts A.1, A.2, B), repaired fine-tuning (Part C) and
the attribute-sampling baseline (Part D).

    python -m src.draft3            # unit tests, sanity checks, all (dataset, trial) units, raw outputs
    python -m src.draft3_report     # Part A analyses of stored results, stats, tables, figure, summary.md

Nothing in the existing pipeline is changed. Releases of the existing methods are regenerated through
src/public.release (streams [seed, method_id, eps_index], unchanged allocate / weighted_gaussian_subset).
Settings: config.yaml, section draft3; notes fixed before the run: draft3_notes.md.
"""
from __future__ import annotations

import copy
import math
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.special import expit, ndtr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from . import data as D
from . import dp
from .noiseaware import expected_raw, predict_noise_aware
from .public import ContinuedHGBClassifier, SanityError, build_context, primary_model, release

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- helpers

def tree_stats(model, keep: np.ndarray, d: int, start: int = 0, stop: int | None = None) -> dict:
    """Split gain and split count per ORIGINAL feature index, and max |leaf value|, over trees [start, stop)."""
    cols = np.flatnonzero(keep)
    gain, cnt = np.zeros(d), np.zeros(d, dtype=int)
    leaf = 0.0
    for it in model._predictors[start:stop]:
        n = it[0].nodes
        inner = ~n["is_leaf"].astype(bool)
        f = cols[n["feature_idx"][inner]]
        np.add.at(gain, f, n["gain"][inner])
        np.add.at(cnt, f, 1)
        lv = n["value"][n["is_leaf"].astype(bool)]
        if lv.size:
            leaf = max(leaf, float(np.max(np.abs(lv))))
    return {"gain": gain, "count": cnt, "max_abs_leaf": leaf}


def best_threshold(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """t* maximizing balanced accuracy of (p >= t) on (p, y); candidates: sorted distinct p and +inf;
    ties -> smallest candidate. Returns (t*, balanced accuracy at t*)."""
    cand = np.r_[np.unique(p), np.inf]
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    P, N = ys.sum(), (1 - ys).sum()
    # for candidate t, predicted positive = {p >= t}; # of records with p < t:
    below = np.searchsorted(ps, cand, side="left")
    cum_pos = np.r_[0, np.cumsum(ys)]
    fn = cum_pos[below]                      # positives predicted 0
    tn = below - fn                          # negatives predicted 0
    ba = 0.5 * ((P - fn) / P + tn / N)
    i = int(np.argmax(ba))                   # first maximum = smallest candidate
    return float(cand[i]), float(ba[i])


def cls_metrics(p: np.ndarray, y: np.ndarray, prefix: str) -> dict:
    return {f"{prefix}_auc": float(roc_auc_score(y, p)),
            f"{prefix}_acc": float(accuracy_score(y, (p >= 0.5).astype(int))),
            f"{prefix}_ba": float(balanced_accuracy_score(y, (p >= 0.5).astype(int))),
            f"{prefix}_n_unique": int(np.unique(p).size)}


# ----------------------------------------------------------------------------- Part C: repaired fine-tuning

class RepairedContinuedHGB(ContinuedHGBClassifier):
    """ContinuedHGBClassifier whose starting raw score on the new rows is clipped to [-start_clip, start_clip]."""

    start_clip: float | None = None

    def _raw_predict(self, X, n_threads=None):
        Xr = getattr(self, "_ws_X", None)
        starting = getattr(self, "_in_fit", False) and Xr is not None and X.shape[0] == Xr.shape[0]
        out = super()._raw_predict(X, n_threads)
        if starting and self.start_clip is not None:
            out = np.clip(out, -self.start_clip, self.start_clip)
            self._start_raw_range = (float(out.min()), float(out.max()))
        return out


def _part(model, start, stop, baseline: bool):
    m = copy.copy(model)
    m._predictors = model._predictors[start:stop]
    if not baseline:
        m._baseline_prediction = np.zeros_like(model._baseline_prediction)
    return m


def repaired_raw(model, X, n_pre: int, clip: float) -> np.ndarray:
    pre = _part(model, 0, n_pre, True)._raw_predict(X).ravel()
    cont = _part(model, n_pre, None, False)._raw_predict(X).ravel()
    return np.clip(pre, -clip, clip) + cont


def repaired_noise_aware(model, X, sigma, n_pre: int, clip: float) -> np.ndarray:
    """clip(E[F_pre(x + eta)], -clip, clip) + E[F_cont(x + eta)], both expectations exact (src/noiseaware).
    (Replaced the Monte-Carlo estimate of E[clip(F_pre(x + eta))] before the main run; see Deviations.)"""
    pre, cont = _part(model, 0, n_pre, True), _part(model, n_pre, None, False)
    return expit(np.clip(expected_raw(pre, X, sigma), -clip, clip) + expected_raw(cont, X, sigma))


# ----------------------------------------------------------------------------- Part D: Random-k

def random_k_release(X, k, eps, delta, rng, mode: str, m: np.ndarray):
    n, d = X.shape
    s = math.sqrt(k) * dp.noise_multiplier(eps, delta)
    if mode == "per_record":
        U = rng.random((n, d))
        idx = np.argpartition(U, k - 1, axis=1)[:, :k]
        M = np.zeros((n, d), dtype=bool)
        np.put_along_axis(M, idx, True, axis=1)
    elif mode == "fixed":
        S = rng.choice(d, size=k, replace=False)
        M = np.zeros((n, d), dtype=bool)
        M[:, S] = True
    else:
        raise ValueError(mode)
    Z = rng.standard_normal((n, d))
    Xt = np.where(M, X + s * Z, m[None, :])
    return Xt, s, M


def expected_raw_sampled(model, X, s: float, m: np.ndarray, k: int, mode: str,
                         released: np.ndarray | None = None, chunk: int = 4096) -> np.ndarray:
    """Exact expected raw output under the Random-k release of a record with clean features x:
    released coordinate x_j + N(0, s^2), unreleased coordinate m_j; released set a uniform k-subset
    (per_record) or the fixed set `released` (fixed)."""
    if any(len(it) != 1 for it in model._predictors):
        raise ValueError("single-output models only")
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    base = float(np.ravel(model._baseline_prediction)[0])
    trees = []
    for it in model._predictors:
        nd = it[0].nodes
        trees.append((nd["is_leaf"].astype(bool).tolist(), nd["feature_idx"].tolist(), nd["num_threshold"].tolist(),
                      nd["left"].tolist(), nd["right"].tolist(), nd["value"].tolist()))
    total = math.comb(d, k)
    wcache: dict = {}

    def weights(p):
        w = wcache.get(p)
        if w is None:
            w = np.array([math.comb(d - p, k - a) / total if 0 <= k - a <= d - p else 0.0 for a in range(p + 1)])
            wcache[p] = w
        return w

    out = np.full(n, base)
    for st in range(0, n, chunk):
        Xc = X[st:st + chunk]
        nc = Xc.shape[0]
        acc = np.zeros(nc)
        cache: dict = {}

        def cdf(f, t):
            if t == np.inf:
                return 1.0
            if t == -np.inf:
                return 0.0
            v = cache.get((f, t))
            if v is None:
                v = ndtr((t - Xc[:, f]) / s)
                cache[(f, t)] = v
            return v

        for is_leaf, feat, thr, left, right, value in trees:
            stack = [(0, {})]
            while stack:
                i, bounds = stack.pop()
                if is_leaf[i]:
                    if not bounds:
                        acc += value[i]
                        continue
                    if mode == "fixed":
                        pr = np.ones(nc)
                        for f, (lo, hi) in bounds.items():
                            if released[f]:
                                pr = pr * (cdf(f, hi) - cdf(f, lo))
                            elif not (lo < m[f] <= hi):
                                pr = None
                                break
                        if pr is not None:
                            acc += value[i] * pr
                        continue
                    # coef[a] = coefficient of z^a in prod_j (1[m_j in I_j] + z * P(x_j + eta_j in I_j))
                    coef = [np.ones(nc)]
                    for f, (lo, hi) in bounds.items():
                        r = cdf(f, hi) - cdf(f, lo)
                        ind = lo < m[f] <= hi
                        new = [np.zeros(nc) for _ in range(len(coef) + 1)]
                        for a, c in enumerate(coef):
                            if ind:
                                new[a] += c
                            new[a + 1] += c * r
                        coef = new
                    w = weights(len(bounds))
                    pr = np.zeros(nc)
                    for a, c in enumerate(coef):
                        if w[a]:
                            pr += w[a] * c
                    acc += value[i] * pr
                    continue
                f, t_ = feat[i], thr[i]
                lo, hi = bounds.get(f, (-np.inf, np.inf))
                bl = dict(bounds)
                bl[f] = (lo, min(hi, t_))
                br = dict(bounds)
                br[f] = (max(lo, t_), hi)
                stack.append((right[i], br))
                stack.append((left[i], bl))
        out[st:st + nc] += acc
    return out


# ----------------------------------------------------------------------------- one unit

def run_unit(ctx, trial, cfg, stored_main, stored_pub):
    seed = cfg["seeds"]["trial_base"] + trial
    c3 = cfg["draft3"]
    ds = ctx["name"]
    Xtr, ytr, Xte, yte = ctx["X_train"], ctx["y_train"], ctx["X_test"], ctx["y_test"]
    Xp_all, yp_all = ctx["X_public"], ctx["y_public"]
    n_tr, d = Xtr.shape
    delta = cfg["privacy"]["delta"]
    h_te, h_tr = D.array_hash(Xte), D.array_hash(Xtr)
    rows, checks, extra = [], [], []
    base = {"dataset": ds, "trial": trial, "seed": seed}
    st_main = stored_main[(stored_main.dataset == ds) & (stored_main.trial == trial)]
    t_part = {"refit": 0.0, "ft": 0.0, "randk": 0.0}

    for ei, eps in enumerate(c3["eps_grid"]):
        # ---------------- Parts A.1, A.2, B: refits of the stored releases
        t0 = time.time()
        for name in c3["refit"]["methods"]:
            al, Xt, sigma, keep, _ = release(ctx, name, eps, ei, seed, cfg)
            model = primary_model(cfg, seed).fit(Xt, ytr)
            sk = sigma[keep]
            p_na = predict_noise_aware(model, Xte[:, keep], sk, "classification")
            p_dir = model.predict_proba(Xte[:, keep])[:, 1]
            p_pub = predict_noise_aware(model, Xp_all[:, keep], sk, "classification")
            p_tr = model.predict_proba(Xt)[:, 1]
            r = {**cls_metrics(p_na, yte, "na"), **cls_metrics(p_dir, yte, "direct")}
            t_star, ba_pub = best_threshold(p_pub, yp_all)
            yh = (p_na >= t_star).astype(int)
            r.update(t_star=t_star, ba_public_at_t=ba_pub, acc_at_t=float(accuracy_score(yte, yh)),
                     ba_at_t=float(balanced_accuracy_score(yte, yh)),
                     auc_public_na=float(roc_auc_score(yp_all, p_pub)), auc_train_direct=float(roc_auc_score(ytr, p_tr)))
            ts = tree_stats(model, keep, d)
            r.update(split_gain=ts["gain"].tolist(), split_count=ts["count"].tolist(), n_released=int(keep.sum()))
            rows.append({**base, "part": "refit", "method": name, "variant": "", "size": "", "k": np.nan,
                         "eps": eps, "eps_index": ei, **r})
            s = st_main[(st_main.method == name) & (st_main.model == "primary") & np.isclose(st_main.eps, eps)].iloc[0]
            checks.append(("refit_reproduces_stored", ds, trial, f"{name}/eps={eps:g}",
                           bool(abs(r["na_auc"] - s.roc_auc) <= 1e-9 and abs(r["na_acc"] - s.accuracy) <= 1e-9
                                and r["na_n_unique"] == s.n_unique_pred)))
            del Xt
        t_part["refit"] += time.time() - t0

        # ---------------- Part C: repaired fine-tuning
        t0 = time.time()
        pc = c3["part_c"]
        for mi, meth in enumerate(pc["methods"]):
            relname = cfg["public_baselines"]["methods"][meth]["release"]
            al, Xt, sigma, keep, _ = release(ctx, relname, eps, ei, seed, cfg)
            sk = sigma[keep]
            for si, size in enumerate(pc["sizes"]):
                pos = ctx["subs"][size]
                Xp, yp = Xp_all[pos][:, keep], yp_all[pos]
                pre = RepairedContinuedHGB(max_iter=100, early_stopping=cfg["models"]["primary"]["early_stopping"],
                                           random_state=seed, warm_start=True)
                pre.fit(Xp, yp)
                if trial == 0:
                    ref = primary_model(cfg, seed).fit(Xp, yp)
                    r100 = [np.asarray(v, float).ravel() for kk, v in enumerate(ref.staged_decision_function(Xte[:, keep]), 1) if kk == 100][0]
                    dpre = np.asarray(pre.decision_function(Xte[:, keep]), float).ravel()
                    checks.append(("pretrained_reproduces_public_only", ds, trial, f"{meth}/{size}/eps={eps:g}",
                                   bool(np.max(np.abs(dpre - r100)) <= 1e-9)))
                start_raw = pre._raw_predict(Xt).ravel()
                pre.start_clip = pc["start_clip"]
                pre.set_params(max_iter=200, l2_regularization=pc["l2_continued"])
                pre.fit(Xt, ytr)
                lo_hi = getattr(pre, "_start_raw_range", (np.nan, np.nan))
                checks.append(("ft_start_clipped", ds, trial, f"{meth}/{size}/eps={eps:g}",
                               bool(lo_hi[0] >= -pc["start_clip"] - 1e-12 and lo_hi[1] <= pc["start_clip"] + 1e-12
                                    and len(pre._predictors) == 200)))
                p_na = repaired_noise_aware(pre, Xte[:, keep], sk, 100, pc["start_clip"])
                p_dir = expit(repaired_raw(pre, Xte[:, keep], 100, pc["start_clip"]))
                r = {**cls_metrics(p_na, yte, "na"), **cls_metrics(p_dir, yte, "direct")}
                tc = tree_stats(pre, keep, d, 100, None)
                tp = tree_stats(pre, keep, d, 0, 100)
                r.update(max_abs_leaf_continued=tc["max_abs_leaf"], max_abs_leaf_pretrained=tp["max_abs_leaf"],
                         start_abs_raw_mean=float(np.abs(start_raw).mean()), start_abs_raw_max=float(np.abs(start_raw).max()),
                         start_frac_clipped=float(np.mean(np.abs(start_raw) > pc["start_clip"])),
                         n_released=int(keep.sum()))
                rows.append({**base, "part": "ft_repaired", "method": meth, "variant": "repaired", "size": size,
                             "k": np.nan, "eps": eps, "eps_index": ei, **r})
            del Xt
        t_part["ft"] += time.time() - t0

        # ---------------- Part D: Random-k
        t0 = time.time()
        pdd = c3["part_d"]
        mvec = Xp_all.mean(axis=0)
        for mode, tag in (("per_record", "per_record"), ("fixed", "fixed")):
            for kq, k in zip(("q", "h"), pdd["k"][ds]):
                rid = pdd["ids"][f"{tag}_{kq}"]
                rng = np.random.default_rng([seed, rid, ei])
                Xt, s, M = random_k_release(Xtr, k, eps, delta, rng, mode, mvec)
                cnt = M.sum(axis=1)
                c = dp.noise_multiplier(eps, delta)
                inv = k / s ** 2 * c * c
                checks.append(("randomk_release", ds, trial, f"{mode}/k={k}/eps={eps:g}",
                               bool(np.all(cnt == k) and abs(inv - 1.0) < 1e-9
                                    and np.all(Xt[~M] == np.broadcast_to(mvec, M.shape)[~M]))))
                model = primary_model(cfg, seed).fit(Xt, ytr)
                released = M[0] if mode == "fixed" else None
                raw = expected_raw_sampled(model, Xte, s, mvec, k, mode, released)
                p_na = expit(raw)
                p_dir = model.predict_proba(Xte)[:, 1]
                r = {**cls_metrics(p_na, yte, "na"), **cls_metrics(p_dir, yte, "direct")}
                ts = tree_stats(model, np.ones(d, bool), d)
                r.update(noise_s=s, n_released=k,
                         fixed_subset=(np.flatnonzero(M[0]).tolist() if mode == "fixed" else []),
                         split_count=ts["count"].tolist())
                rows.append({**base, "part": "randomk", "method": f"randomk_{mode}", "variant": mode, "size": "",
                             "k": k, "eps": eps, "eps_index": ei, **r})
                del Xt, M
        t_part["randk"] += time.time() - t0

    if D.array_hash(Xte) != h_te or D.array_hash(Xtr) != h_tr:
        raise SanityError(f"{ds}: train or test data changed during trial {trial}")
    return rows, checks, t_part


CTX_CACHE: dict = {}


def _task(name, trial, cfg, cache_file, stored_main, stored_pub):
    t0 = time.time()
    ctx = CTX_CACHE.get(name)
    if ctx is None:
        ctx = CTX_CACHE[name] = build_context(name, cfg)
    rows, checks, tp = run_unit(ctx, trial, cfg, stored_main, stored_pub)
    with open(cache_file + ".tmp", "wb") as f:
        pickle.dump((rows, checks, time.time() - t0, tp), f)
    os.replace(cache_file + ".tmp", cache_file)
    return name, trial, time.time() - t0


# ----------------------------------------------------------------------------- main

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    t0 = time.time()
    cfg = D.load_config()
    c3 = cfg["draft3"]
    pb = cfg["public_baselines"]
    smoke = "--smoke" in argv
    out = D.ROOT / (c3["out_dir"] + ("_smoke" if smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    for f in cache.glob("*.tmp"):
        f.unlink()
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    def check(cond, msg):
        log(("[PASS] " if cond else "[FAIL] ") + msg)
        if not cond:
            (out / "sanity_draft3.log").write_text("\n".join(lines) + "\n")
            raise SanityError(msg)

    import pytest
    rc = pytest.main(["-q", "-p", "no:cacheprovider", "-m", "not crosscheck", str(D.ROOT / "tests")])
    check(rc == 0, f"all unit tests pass (pytest exit code {rc})")
    for name in c3["datasets"]:
        p = D.prepare(name, cfg)
        hs = pb["split_hashes"][name]
        for k in ("X_train", "y_train", "X_test", "y_test", "idx_train", "idx_test", "X_public"):
            check(D.array_hash(getattr(p, k)) == hs[k], f"{name}: {k} byte-identical to the split of the main run (SHA-256)")
        dd = p.d
        kk = [int(math.floor(dd / 4 + 0.5)), int(math.floor(dd / 2 + 0.5))]
        check(kk == c3["part_d"]["k"][name], f"{name}: d = {dd}, k = {kk} (d/4 and d/2 rounded half up) as in config.yaml")

    stored_main = pd.read_parquet(D.ROOT / cfg["outputs"]["dir"] / "raw_trials.parquet")
    stored_main = stored_main[(stored_main.experiment == "main") & stored_main.dataset.isin(c3["datasets"])
                              & (stored_main.model == "primary")]
    stored_main = stored_main[["dataset", "trial", "method", "model", "eps", "roc_auc", "accuracy", "n_unique_pred"]]
    stored_pub = None
    n_trials = 1 if smoke else c3["n_trials"]
    dsets = c3["datasets"][:1] if smoke else c3["datasets"]
    jobs = [(n, t, str(cache / f"{n}_{t:02d}.pkl")) for n in dsets for t in range(n_trials)]
    todo = [j for j in jobs if not os.path.exists(j[2])]
    log(f"== running {len(todo)} of {len(jobs)} units (rest cached), n_jobs = {c3['n_jobs']}")
    t_run = time.time()
    if todo:
        from src.draft3 import _task as task_fn       # pickled by reference, not from __main__
        par = Parallel(n_jobs=c3["n_jobs"], backend="loky", return_as="generator_unordered")
        done = 0
        for name, t, dt in par(delayed(task_fn)(n, t, cfg, f, stored_main, stored_pub) for n, t, f in todo):
            done += 1
            el = time.time() - t_run
            log(f"  {done}/{len(todo)} units | {name} t={t} ({dt:.0f} s) | elapsed {el/60:.1f} min | "
                f"ETA {(len(todo)-done)*el/done/60:.1f} min")
    rows, checks, unit_s, parts = [], [], [], []
    for _, _, f in jobs:
        with open(f, "rb") as fh:
            r, c, dt, tp = pickle.load(fh)
        rows += r
        checks += c
        unit_s.append(dt)
        parts.append(tp)
    ch = pd.DataFrame(checks, columns=["check", "dataset", "trial", "item", "ok"])
    for c, g in ch.groupby("check"):
        bad = g[~g.ok]
        check(len(bad) == 0, f"{c}: {int(g.ok.sum())} of {len(g)} checks pass"
              + ("" if len(bad) == 0 else f"; failing: {bad.head(5).to_dict('records')}"))
    df = pd.DataFrame(rows)
    df.to_parquet(out / "raw_trials_draft3.parquet", index=False)
    ch.to_csv(out / "sanity_checks_draft3.csv", index=False)
    import importlib.metadata as im
    used = {**cfg, "_run_draft3": {"python": sys.version.split()[0],
                                   **{m: im.version(m) for m in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib")}}}
    with open(out / "config_used.yaml", "w") as f:
        yaml.safe_dump(used, f, sort_keys=False)
    total = time.time() - t0
    tp = pd.DataFrame(parts).sum() / 60
    (out / "runtime_draft3.txt").write_text(
        f"python -m src.draft3: {total/60:.1f} min wall clock ({total:.0f} s); units run this time: {len(todo)}; "
        f"sum of unit times {sum(unit_s)/60:.1f} min ({c3['n_jobs']} workers); by part (sum over units, min): "
        f"refits (A.1, A.2, B) {tp.get('refit', 0):.1f}, Part C {tp.get('ft', 0):.1f}, Part D {tp.get('randk', 0):.1f}\n")
    log(f"wrote raw_trials_draft3.parquet ({len(df)} rows); total {total/60:.1f} min")
    (out / "sanity_draft3.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        main()
    except SanityError as e:
        print(f"\nRUN STOPPED: sanity check failed: {e}", file=sys.stderr)
        sys.exit(1)
