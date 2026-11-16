"""Main experiment loop for Chapter 4 (MIC-DP).

    python -m src.run            # resumes from results/ch4/cache/trials if present
    python -m src.run --fresh    # wipes the trial cache first

Steps: unit tests -> data + cached raw scores -> datasets.csv -> pre-flight checks ->
main grid + sensitivity sweeps (joblib, parallel over trials) -> raw_trials.parquet,
weights.parquet -> post-run sanity checks. The run stops if any check fails.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from . import data as D
from . import dp
from .methods import allocate
from .noiseaware import predict_noise_aware
from .scores import v2_statistics

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- logging

class Log:
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "w")

    def __call__(self, *msg):
        s = " ".join(str(m) for m in msg)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


class SanityError(RuntimeError):
    pass


def check(cond: bool, msg: str, log: Log):
    log(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        raise SanityError(msg)


# ----------------------------------------------------------------------------- models

def fit_eval(kind: str, task: str, Xtr, ytr, Xte, yte, seed: int, cfg: dict, sigma=None) -> dict:
    """Fit on (Xtr, ytr), evaluate on the clean test set. sigma (released columns' noise std) turns
    on noise-aware prediction for the primary model (config models.primary.noise_aware)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score

    mc = cfg["models"]
    if kind == "primary":
        cls = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
        model = cls(max_iter=mc["primary"]["max_iter"], early_stopping=mc["primary"]["early_stopping"],
                    random_state=seed)
    elif task == "classification":
        s = mc["secondary"]["classification"]
        model = LogisticRegression(C=s["C"], max_iter=s["max_iter"])
    else:
        model = Ridge(alpha=mc["secondary"]["regression"]["alpha"])
    model.fit(Xtr, ytr)
    noise_aware = (kind == "primary" and sigma is not None and mc["primary"].get("noise_aware", False))
    if task == "classification":
        p = predict_noise_aware(model, Xte, sigma, task) if noise_aware else model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)
        acc = accuracy_score(yte, (p >= 0.5).astype(int))
        return {"model_name": type(model).__name__, "roc_auc": auc, "accuracy": acc,
                "mae": np.nan, "r2": np.nan, "primary_value": auc, "U": auc,
                "n_unique_pred": int(np.unique(p).size)}
    pred = predict_noise_aware(model, Xte, sigma, task) if noise_aware else model.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    return {"model_name": type(model).__name__, "roc_auc": np.nan, "accuracy": np.nan,
            "mae": mae, "r2": r2_score(yte, pred), "primary_value": mae, "U": -mae,
            "n_unique_pred": int(np.unique(pred).size)}


# ----------------------------------------------------------------------------- one trial

def _method_specs(cfg: dict, dataset: str) -> list[tuple[str, dict]]:
    out = []
    for name, m in cfg["methods"]["list"].items():
        if name == "nonprivate":
            continue
        if "only" in m and dataset not in m["only"]:
            continue
        out.append((name, m))
    return out


def _params(cfg: dict, **over) -> dict:
    p = dict(cfg["methods"]["defaults"])
    p.update(over)
    return p


def _release_and_eval(ctx, al, rng, seed, cfg, kinds):
    """Release the columns with w_j > 0, fit the models on them and evaluate on clean test data."""
    Xtr, ytr, Xte, yte, task = ctx["X_train"], ctx["y_train"], ctx["X_test"], ctx["y_test"], ctx["task"]
    Xt, sigma, keep = dp.weighted_gaussian_subset(Xtr, al.w, al.eps_m, al.delta_m, rng)
    absdiff = np.abs(Xt - Xtr[:, keep])
    dist_j = np.full(Xtr.shape[1], np.nan)
    dist_j[keep] = absdiff.mean(axis=0)
    dist = float(absdiff.mean())
    del absdiff
    res = {kind: fit_eval(kind, task, Xt, ytr, Xte[:, keep], yte, seed, cfg, sigma=sigma[keep])
           for kind in kinds}
    del Xt
    return res, sigma, keep, dist, dist_j


def _row(base, eps, ei, name, mid, kind, r, dist, al, sigma, keep):
    return {**base, "eps": eps, "eps_index": ei, "method": name, "method_id": mid, "model": kind, **r,
            "distortion": dist, "sigma_s": al.sigma_s, "eps_m": al.eps_m, "delta_m": al.delta_m,
            "w": al.w.tolist(), "sigma": sigma.tolist(), "n_released": int(keep.sum()),
            "shrink_B": al.B,
            "s_hat": [] if al.s_hat is None else np.asarray(al.s_hat).tolist(),
            "s_tilde": [] if al.s_tilde is None else np.asarray(al.s_tilde).tolist()}


def run_main_trial(ctx: dict, trial: int, cfg: dict) -> tuple[list[dict], list[dict]]:
    seed = cfg["seeds"]["trial_base"] + trial
    ds, task = ctx["name"], ctx["task"]
    Xtr, ytr, Xte, yte = ctx["X_train"], ctx["y_train"], ctx["X_test"], ctx["y_test"]
    h0 = D.array_hash(Xte)
    d = Xtr.shape[1]
    delta = cfg["privacy"]["delta"]
    params = _params(cfg)
    rows, wrows = [], []
    base = {"experiment": "main", "dataset": ds, "trial": trial, "seed": seed, "param": np.nan}

    for kind in ("primary", "secondary"):
        r = fit_eval(kind, task, Xtr, ytr, Xte, yte, seed, cfg)
        rows.append({**base, "eps": np.nan, "eps_index": -1, "method": "nonprivate", "method_id": 0,
                     "model": kind, **r, "distortion": 0.0, "sigma_s": np.nan, "eps_m": np.nan,
                     "delta_m": np.nan, "w": [], "sigma": [], "n_released": d, "shrink_B": np.nan,
                     "s_hat": [], "s_tilde": []})

    ov = cfg["methods"]["list"]["mic_dp_override"]
    prot_idx = [ctx["features"].index(f) for f in ov["protected"]] if ds in ov.get("only", []) else []
    mic_id = cfg["methods"]["list"]["mic_dp"]["id"]

    for ei, eps in enumerate(cfg["privacy"]["eps_grid"]):
        for name, m in _method_specs(cfg, ds):
            rng = np.random.default_rng([seed, m["id"], ei])
            score_rng = np.random.default_rng([seed, mic_id, ei]) \
                if m["weights"] in ("permute", "relevance_override") else None
            al = allocate(name, m, eps, delta, ctx["raw"], d, rng, params, score_rng=score_rng,
                          protected_idx=prot_idx)
            res, sigma, keep, dist, dist_j = _release_and_eval(ctx, al, rng, seed, cfg, ("primary", "secondary"))
            for kind, r in res.items():
                rows.append(_row(base, eps, ei, name, m["id"], kind, r, dist, al, sigma, keep))
            for j in range(d):
                wrows.append({"experiment": "main", "dataset": ds, "eps": eps, "method": name,
                              "param": np.nan, "trial": trial, "feature_index": j,
                              "feature": ctx["features"][j], "w": al.w[j], "sigma": sigma[j],
                              "s_hat": np.nan if al.s_hat is None else al.s_hat[j],
                              "s_tilde": np.nan if al.s_tilde is None else al.s_tilde[j],
                              "released": bool(keep[j]), "distortion": dist_j[j]})
    if D.array_hash(Xte) != h0:
        raise SanityError(f"test set of {ds} changed during trial {trial}")
    return rows, wrows


def run_sens_trial(ctx: dict, trial: int, cfg: dict) -> tuple[list[dict], list[dict]]:
    seed = cfg["seeds"]["trial_base"] + trial
    ds = ctx["name"]
    Xte = ctx["X_test"]
    h0 = D.array_hash(Xte)
    d = ctx["X_train"].shape[1]
    delta = cfg["privacy"]["delta"]
    sc = cfg["protocol"]["sensitivity"]
    dflt = cfg["methods"]["defaults"]
    grid = cfg["privacy"]["eps_grid"]
    mcfg = cfg["methods"]["list"]["mic_dp"]
    rows, wrows = [], []
    for eps in sc["eps"]:
        ei = grid.index(eps)
        sweeps = [("alpha", a, dflt["rho"], sc["alpha_method_id_base"] + i) for i, a in enumerate(sc["alpha_grid"])]
        sweeps += [("rho", dflt["alpha"], r, sc["rho_method_id_base"] + i) for i, r in enumerate(sc["rho_grid"])]
        for exp, a, r, mid in sweeps:
            rng = np.random.default_rng([seed, mid, ei])
            al = allocate("mic_dp", mcfg, eps, delta, ctx["raw"], d, rng, _params(cfg, alpha=float(a), rho=float(r)))
            res, sigma, keep, dist, _ = _release_and_eval(ctx, al, rng, seed, cfg, ("primary",))
            param = a if exp == "alpha" else r
            base = {"experiment": exp, "dataset": ds, "trial": trial, "seed": seed, "param": float(param)}
            rows.append(_row(base, eps, ei, "mic_dp", mid, "primary", res["primary"], dist, al, sigma, keep))
            for j in range(d):
                wrows.append({"experiment": exp, "dataset": ds, "eps": eps, "method": "mic_dp",
                              "param": float(param), "trial": trial, "feature_index": j,
                              "feature": ctx["features"][j], "w": al.w[j], "sigma": sigma[j],
                              "s_hat": al.s_hat[j], "s_tilde": al.s_tilde[j], "released": bool(keep[j]),
                              "distortion": np.nan})
    if D.array_hash(Xte) != h0:
        raise SanityError(f"test set of {ds} changed during sensitivity trial {trial}")
    return rows, wrows


def _task(kind: str, ctx: dict, trial: int, cfg: dict, cache_file: str):
    fn = run_main_trial if kind == "main" else run_sens_trial
    t0 = time.time()
    out = fn(ctx, trial, cfg)
    with open(cache_file + ".tmp", "wb") as f:
        pickle.dump(out, f)
    os.replace(cache_file + ".tmp", cache_file)
    return kind, ctx["name"], trial, time.time() - t0


# ----------------------------------------------------------------------------- main

EXPERIMENT_KEYS = ["seeds", "privacy", "protocol", "methods", "scores", "models", "data", "splits",
                   "ranges", "synthetic", "datasets"]


def config_hash(cfg: dict) -> str:
    """Hash of the settings that determine trial results (not the stats/report/hypothesis text)."""
    sub = {k: cfg[k] for k in EXPERIMENT_KEYS}
    return hashlib.sha256(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:12]


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    t_start = time.time()
    cfg = D.load_config()
    out = D.ROOT / cfg["outputs"]["dir"]
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "sanity.log")
    chash = config_hash(cfg)
    cache = out / "cache" / f"trials_{chash}"
    if "--fresh" in argv and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    log(f"Chapter 4 MIC-DP experiment | config hash {chash} | {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- 1. unit tests
    log("\n== Sanity 1: unit tests")
    import pytest
    # the minepy cross-check (tests/test_mic_vs_minepy.py) is run separately: the experiment
    # pipeline does not depend on minepy at all
    rc = pytest.main(["-q", "-p", "no:cacheprovider", "-m", "not crosscheck", str(D.ROOT / "tests")])
    check(rc == 0, f"all unit tests pass (pytest exit code {rc})", log)

    # ---- 2. data + scores
    log("\n== Data")
    delta = cfg["privacy"]["delta"]
    eps_grid = cfg["privacy"]["eps_grid"]
    ctxs, preps, ds_rows, score_rows = {}, {}, [], []
    for k, name in enumerate(D.all_dataset_names(cfg)):
        p = D.prepare(name, cfg)
        preps[name] = p
        check(delta < 1 / len(p.idx_train), f"{name}: delta={delta} < 1/n_train = {1/len(p.idx_train):.2e}", log)
        log(f"  {name:13s} task={p.task:14s} final d={p.d:2d}  n_raw={p.n_raw} n_used={p.n_total} "
            f"public/train/test={len(p.idx_public)}/{len(p.idx_train)}/{len(p.idx_test)}")
        raw = v2_statistics(p, cfg)
        ctxs[name] = {"name": name, "task": p.task, "features": p.features, "X_train": p.X_train,
                      "y_train": p.y_train, "X_test": p.X_test, "y_test": p.y_test, "raw": raw}
        tgt = {"positive_rate": float(np.mean(np.r_[p.y_public, p.y_train, p.y_test]))} \
            if p.task == "classification" else {"target_std": float(np.std(np.r_[p.y_public, p.y_train, p.y_test]))}
        ds_rows.append({"name": cfg["datasets"][name]["display"], "key": name, "n_public": len(p.idx_public),
                        "n_train": len(p.idx_train), "n_test": len(p.idx_test), "d": p.d, "task": p.task,
                        "positive_rate": tgt.get("positive_rate", np.nan), "target_std": tgt.get("target_std", np.nan),
                        "max_MIC_public": float(raw["fgmic_public"].max()),
                        "mean_MIC_public": float(raw["fgmic_public"].mean()),
                        "n_raw": p.n_raw, "fgmic_sensitivity": raw["fgmic_sens"],
                        "moment_sensitivity": raw["moment_sens"]})
        for j, f in enumerate(p.features):
            score_rows.append({"dataset": name, "feature": f, "a": p.a[j], "b": p.b[j], "range_source": p.range_sources[j],
                               "clip_frac_train": p.clip_frac_train[j], "fgmic_train": raw["fgmic"][j],
                               "fgmic_public": raw["fgmic_public"][j]})
    check(True, "scores use the private-train split (and the public split for MIC-Public); grid cut points and "
                "the target range for the moment release come from the public split only", log)
    pd.DataFrame(ds_rows).to_csv(out / "datasets.csv", index=False)
    pd.DataFrame(score_rows).to_csv(out / "scores_raw.csv", index=False)
    log("  wrote datasets.csv and scores_raw.csv (MIC statistics in datasets.csv: FG-MIC on the PUBLIC split)")

    # ---- 3. sigma_s table
    log("\n== Sanity 6: score-privatization noise sigma_s = sqrt(d) * 3(1+ln n)/(n ln 2) * c(rho*eps, rho*delta), rho = 0.2")
    rho = cfg["methods"]["defaults"]["rho"]
    log("  dataset        d " + "".join(f"  eps={e:<5g}" for e in eps_grid))
    for name, p in preps.items():
        s1 = math.sqrt(p.d) * ctxs[name]["raw"]["fgmic_sens"]
        vals = [s1 * dp.noise_multiplier(rho * e, rho * delta) for e in eps_grid]
        log(f"  {name:13s} {p.d:2d} " + "".join(f"  {v:9.4f}" for v in vals))
    log("  (FG-MIC scores live in [0, 1]; compare with the spread of the scores in scores_raw.csv)")

    # ---- 4. pre-flight: Uniform distortion on one noise draw per (dataset, eps)
    log("\n== Pre-flight: Uniform distortion vs sqrt(d) c sqrt(2/pi) (one draw, trial 0)")
    for name, p in preps.items():
        for ei, e in enumerate(eps_grid):
            rng = np.random.default_rng([cfg["seeds"]["trial_base"], cfg["methods"]["list"]["uniform"]["id"], ei])
            Xt, _ = dp.weighted_gaussian(p.X_train, np.full(p.d, 1 / p.d), e, delta, rng)
            got = np.abs(Xt - p.X_train).mean()
            exp_ = math.sqrt(p.d) * dp.noise_multiplier(e, delta) * math.sqrt(2 / math.pi)
            if abs(got / exp_ - 1) > 0.02:
                check(False, f"pre-flight distortion {name} eps={e}: {got:.4f} vs {exp_:.4f}", log)
    log("  [PASS] all pre-flight Uniform distortions within 2%")

    # ---- 5. run trials
    n_trials = cfg["protocol"]["n_trials"]
    sens_ds = cfg["protocol"]["sensitivity"]["datasets"]
    order = sorted(ctxs, key=lambda n: -preps[n].X_train.size)
    jobs = []
    for name in order:
        for t in range(n_trials):
            jobs.append(("main", name, t, str(cache / f"main_{name}_{t:02d}.pkl")))
    for name in [n for n in order if n in sens_ds]:
        for t in range(n_trials):
            jobs.append(("sens", name, t, str(cache / f"sens_{name}_{t:02d}.pkl")))
    todo = [j for j in jobs if not os.path.exists(j[3])]
    log(f"\n== Running {len(todo)} of {len(jobs)} trial tasks (rest cached) with n_jobs={cfg['protocol']['n_jobs']}")
    t_run = time.time()
    if todo:
        par = Parallel(n_jobs=cfg["protocol"]["n_jobs"], backend="loky", verbose=0, return_as="generator_unordered")
        done = 0
        for kind, name, t, dt in par(delayed(_task)(k, ctxs[n], t, cfg, f) for k, n, t, f in todo):
            done += 1
            if done % 10 == 0 or done == len(todo):
                el = time.time() - t_run
                log(f"  {done}/{len(todo)} tasks done | last: {kind} {name} t={t} ({dt:.0f}s) | "
                    f"elapsed {el/60:.1f} min | ETA {(len(todo)-done)*el/done/60:.1f} min")
    log(f"  trial loop wall time: {(time.time()-t_run)/60:.1f} min")

    rows, wrows = [], []
    for _, _, _, f in jobs:
        with open(f, "rb") as fh:
            r, w = pickle.load(fh)
        rows += r
        wrows += w
    raw_df = pd.DataFrame(rows)
    w_df = pd.DataFrame(wrows)
    raw_df.to_parquet(out / "raw_trials.parquet", index=False)
    w_df.to_parquet(out / "weights.parquet", index=False)
    log(f"  wrote raw_trials.parquet ({len(raw_df)} rows) and weights.parquet ({len(w_df)} rows)")

    # ---- 6. post-run sanity checks
    log("\n== Sanity 2: Uniform mean feature distortion within 2% of sqrt(d) * c(eps, delta) * sqrt(2/pi)")
    main_df = raw_df[(raw_df.experiment == "main") & (raw_df.model == "primary")]
    worst = 0.0
    for name, p in preps.items():
        for e in eps_grid:
            got = main_df[(main_df.dataset == name) & (main_df.method == "uniform") & (main_df.eps == e)].distortion.mean()
            exp_ = math.sqrt(p.d) * dp.noise_multiplier(e, delta) * math.sqrt(2 / math.pi)
            worst = max(worst, abs(got / exp_ - 1))
            if abs(got / exp_ - 1) > 0.02:
                check(False, f"Uniform distortion {name} eps={e}: {got:.4f} vs expected {exp_:.4f}", log)
    check(True, f"Uniform distortion matches theory for all datasets x eps (worst relative error {worst:.2e})", log)

    log("\n== Sanity 3: the test set is never noised; scores, ranges and models never touch test data")
    for name, p in preps.items():
        if D.array_hash(p.X_test) != p.test_hash:
            check(False, f"{name}: test features changed", log)
    check(True, "test-feature SHA-256 unchanged after every trial (checked in each worker and here); "
                "the mechanism is only ever applied to X_train", log)
    check(True, "dependence scores use the private-train split (and the public split for MIC-Public); "
                "public ranges and grid cut points come from config.yaml (documented) or the public split", log)

    log("\n== Sanity 4: privatized scores vary across trials (per-feature std of the noisy FG-MIC over trials, Adult, MIC-DP)")
    for e in eps_grid:
        sub = main_df[(main_df.dataset == "adult") & (main_df.method == "mic_dp") & (main_df.eps == e)]
        S = np.vstack(sub.sort_values("trial").s_hat.to_numpy())
        sd = S.std(axis=0)
        log(f"  eps={e:<5g} std(s_hat_j) = " + " ".join(f"{v:.3f}" for v in sd)
            + f" | mean shrinkage factor B = {sub.shrink_B.mean():.2f} | mean released columns = {sub.n_released.mean():.1f}")
        if not np.all(sd > 0):
            check(False, f"s_hat constant across trials at eps={e}", log)
    check(True, "privatized scores vary across trials for every feature and eps", log)

    log("\n== Sanity 7: privacy invariant on every stored DP release: sum over released columns of 1/sigma_j^2 = 1/c(eps_m, delta_m)^2")
    worst = 0.0
    dp_rows = raw_df[raw_df.method != "nonprivate"]
    for r in dp_rows.itertuples():
        s = np.asarray(r.sigma, dtype=float)
        c = dp.noise_multiplier(r.eps_m, r.delta_m)
        worst = max(worst, abs((1.0 / s[np.isfinite(s)] ** 2).sum() * c * c - 1.0))
    check(worst < 1e-9, f"invariant holds on {len(dp_rows)} stored releases (max relative error {worst:.1e})", log)

    log("\n== Sanity 5: MIC-Oracle is not DP and is excluded from DP comparisons")
    check(cfg["methods"]["list"]["mic_oracle"]["dp"] is False,
          "mic_oracle is flagged dp: false; report.py asserts it appears only in tab_ablation.tex", log)

    with open(out / "config_used.yaml", "w") as f:
        yaml.safe_dump({**cfg, "_run": {"config_hash": chash, "python": sys.version.split()[0],
                                        "numpy": np.__version__, "pandas": pd.__version__,
                                        "sklearn": __import__("sklearn").__version__,
                                        "scipy": __import__("scipy").__version__,
                                        "scores": "v2: FG-MIC and moment-based correlations (src/fgmic.py)"}},
                       f, sort_keys=False)
    total = time.time() - t_start
    (out / "runtime.txt").write_text(f"src.run wall time: {total/60:.1f} min ({total:.0f} s); "
                                     f"trial loop: {(time.time()-t_run)/60:.1f} min; tasks run this time: {len(todo)}\n")
    log(f"\nAll sanity checks passed. Total wall time {total/60:.1f} min.")


if __name__ == "__main__":
    try:
        main()
    except SanityError as e:
        print(f"\nRUN STOPPED: sanity check failed: {e}", file=sys.stderr)
        sys.exit(1)
