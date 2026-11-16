"""Data loading, encoding, public ranges, and splits for the Chapter 4 experiment.

Every dataset goes through the same steps:

1. load + encode (categorical -> documented ordinal codes 0..K-1, drop ID/date columns)
2. fixed-seed subsample to at most ``max_rows`` rows (before splitting)
3. one fixed-seed split: public 15% / private-train 65% / test 20% (stratified for classification)
4. clip every feature to its public range [a_j, b_j] (from config.yaml) and scale to [0, 1]

Ranges come from documentation / domain knowledge, or (fallback) from the 0.5-99.5th
percentiles of the PUBLIC split. They never use private-train or test rows.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    import os
    path = Path(path) if path else Path(os.environ.get("CH4_CONFIG", ROOT / "config.yaml"))
    with open(path) as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------- raw loaders

ADULT_COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
              "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
              "hours-per-week", "native-country", "income"]


def _raw_path(cfg: dict, key: str) -> Path:
    return ROOT / cfg["data"]["raw_dir"] / cfg["data"]["files"][key]["file"]


def _load_adult(cfg: dict, dcfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    kw = dict(header=None, names=ADULT_COLS, skipinitialspace=True, dtype=str,
              na_filter=False, skip_blank_lines=True)
    tr = pd.read_csv(_raw_path(cfg, "adult_data"), **kw)
    te = pd.read_csv(_raw_path(cfg, "adult_test"), skiprows=1, **kw)
    df = pd.concat([tr, te], ignore_index=True)
    df = df[df["age"].str.len() > 0].reset_index(drop=True)
    inc = df["income"].str.strip().str.rstrip(".")
    if not set(inc.unique()) <= {">50K", "<=50K"}:
        raise ValueError(f"unexpected income labels: {set(inc.unique())}")
    y = (inc == ">50K").astype(int).to_numpy()
    df = df.drop(columns=["income"] + list(dcfg["drop"]))
    out = pd.DataFrame(index=df.index)
    for name, spec in dcfg["features"].items():
        col = df[name].str.strip()
        if "categories" in spec:
            cats = list(spec["categories"])
            code = {c: i for i, c in enumerate(cats)}
            unknown = set(col.unique()) - set(code)
            if unknown:
                raise ValueError(f"{name}: undocumented categories {unknown}")
            out[name] = col.map(code).astype(float)
        else:
            out[name] = pd.to_numeric(col).astype(float)
    return out, y


def _load_cdc(cfg: dict, dcfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(_raw_path(cfg, "cdc"))
    y = df[dcfg["target"]].astype(int).to_numpy()
    X = df[list(dcfg["features"])].astype(float)
    return X, y


def _load_calhousing(cfg: dict, dcfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Rebuild sklearn's fetch_california_housing() from the StatLib cal_housing.data file.

    This reproduces sklearn/datasets/_california_housing.py exactly: column reorder
    [8, 7, 2, 3, 4, 5, 6, 1, 0], per-household averages, target in units of $100,000.
    """
    raw = np.loadtxt(_raw_path(cfg, "calhousing"), delimiter=",")
    raw = raw[:, [8, 7, 2, 3, 4, 5, 6, 1, 0]]
    target, data = raw[:, 0].copy(), raw[:, 1:].copy()
    data[:, 2] /= data[:, 5]            # AveRooms  = total rooms / households
    data[:, 3] /= data[:, 5]            # AveBedrms = total bedrooms / households
    data[:, 5] = data[:, 4] / data[:, 5]  # AveOccup = population / households
    target = target / 100000.0
    names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
             "Latitude", "Longitude"]
    X = pd.DataFrame(data, columns=names)
    return X[list(dcfg["features"])], target


def _load_appliances(cfg: dict, dcfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(_raw_path(cfg, "appliances"))
    df.columns = [c.strip() for c in df.columns]
    y = pd.to_numeric(df[dcfg["target"]]).astype(float).to_numpy()
    df = df.drop(columns=[dcfg["target"]] + list(dcfg["drop"]))
    X = df[list(dcfg["features"])].apply(pd.to_numeric).astype(float)
    return X, y


def make_synthetic(cfg: dict) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Generate the synthetic family. All variants share X, the label noise and the splits."""
    s = cfg["synthetic"]
    rng = np.random.default_rng(cfg["seeds"]["synthetic"])
    n, d, bs, rho = s["n"], s["d"], s["block_size"], s["within_block_corr"]
    block = np.full((bs, bs), rho) + (1 - rho) * np.eye(bs)
    Sigma = np.kron(np.eye(d // bs), block)
    X = rng.multivariate_normal(np.zeros(d), Sigma, size=n, method="cholesky")
    e = rng.standard_normal(n)
    x0, x4, x8, x12, x16 = (X[:, j] for j in s["informative"])
    f_nl = (np.sin(np.pi * x0) + (x4 ** 2 - 1) + (np.abs(x8) - math.sqrt(2 / math.pi))
            + 2 * ((x12 > 0).astype(float) - 0.5) + x16)
    f_lin = x0 + x4 + x8 + x12 + x16
    cols = [f"x{j}" for j in range(d)]
    Xdf = pd.DataFrame(X, columns=cols)
    out = {}
    for name, v in s["variants"].items():
        f = f_lin if v["signal"] == "linear" else f_nl
        y = v["lambda"] * f / f.std() + e
        out[name] = (Xdf, y)
    return out


_LOADERS = {"adult": _load_adult, "cdc_diabetes": _load_cdc,
            "calhousing": _load_calhousing, "appliances": _load_appliances}


# ----------------------------------------------------------------------------- splitting

def split_indices(n: int, y: np.ndarray, task: str, cfg: dict) -> tuple[np.ndarray, ...]:
    """Public 15% / private-train 65% / test 20% with a fixed seed (stratified if classification)."""
    sp = cfg["splits"]
    idx = np.arange(n)
    strat = y if task == "classification" else None
    rest, test = train_test_split(idx, test_size=sp["test"], random_state=cfg["seeds"]["split"],
                                  stratify=strat)
    strat_rest = y[rest] if task == "classification" else None
    train, public = train_test_split(rest, test_size=sp["public"] / (1 - sp["test"]),
                                     random_state=cfg["seeds"]["split"] + 1, stratify=strat_rest)
    return np.sort(public), np.sort(train), np.sort(test)


def fallback_range(x_public: np.ndarray, q: tuple[float, float]) -> tuple[float, float]:
    lo, hi = np.percentile(x_public, q)
    return float(lo), float(hi)


def clip_scale(X: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Clip column j to [a_j, b_j] and map it affinely to [0, 1] (so Delta_j = 1)."""
    if np.any(b <= a):
        raise ValueError("every public range must have b_j > a_j")
    Xc = np.clip(X, a[None, :], b[None, :])
    return (Xc - a[None, :]) / (b - a)[None, :]


def array_hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@dataclass
class Prepared:
    name: str
    display: str
    task: str
    features: list[str]
    X_public: np.ndarray
    y_public: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    a: np.ndarray
    b: np.ndarray
    range_sources: list[str]
    idx_public: np.ndarray
    idx_train: np.ndarray
    idx_test: np.ndarray
    n_total: int
    n_raw: int
    test_hash: str = ""
    clip_frac_train: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def d(self) -> int:
        return len(self.features)


def _ranges(dcfg: dict, Xpub: pd.DataFrame, cfg: dict, strict: bool = True):
    a, b, src = [], [], []
    q = tuple(cfg["ranges"]["fallback_percentiles"])
    for name, spec in dcfg["features"].items():
        if "categories" in spec:
            lo, hi = 0.0, float(len(spec["categories"]) - 1)
            s = spec["source"]
        elif spec["source"] == "public-split":
            lo, hi = fallback_range(Xpub[name].to_numpy(), q)
            stored = spec.get("range")
            if stored is not None and strict:
                if not np.allclose([lo, hi], stored, rtol=1e-6, atol=1e-9):
                    raise ValueError(f"{name}: stored fallback range {stored} != recomputed {[lo, hi]}")
                lo, hi = float(stored[0]), float(stored[1])
            s = "public-split"
        else:
            lo, hi = map(float, spec["range"])
            s = spec["source"]
        if hi <= lo:  # degenerate public percentile range: widen minimally (logged)
            raise ValueError(f"{name}: degenerate range [{lo}, {hi}]")
        a.append(lo); b.append(hi); src.append(s)
    return np.array(a), np.array(b), src


def load_frame(name: str, cfg: dict) -> tuple[pd.DataFrame, np.ndarray, str]:
    dcfg = cfg["datasets"][name]
    if dcfg.get("synthetic"):
        X, y = make_synthetic(cfg)[name]
    else:
        X, y = _LOADERS[name](cfg, dcfg)
    return X, y, dcfg["task"]


def prepare(name: str, cfg: dict, strict_ranges: bool = True) -> Prepared:
    dcfg = cfg["datasets"][name]
    X, y, task = load_frame(name, cfg)
    n_raw = len(X)
    if dcfg.get("expected_rows") is not None and n_raw != dcfg["expected_rows"]:
        raise ValueError(f"{name}: expected {dcfg['expected_rows']} rows, got {n_raw}")
    max_rows = cfg["data"]["max_rows"]
    if n_raw > max_rows:
        rng = np.random.default_rng(cfg["seeds"]["subsample"])
        keep = np.sort(rng.choice(n_raw, size=max_rows, replace=False))
        X = X.iloc[keep].reset_index(drop=True)
        y = y[keep]
    n = len(X)
    pub, tr, te = split_indices(n, y, task, cfg)
    assert not (set(pub) & set(tr)) and not (set(pub) & set(te)) and not (set(tr) & set(te))
    a, b, src = _ranges(dcfg, X.iloc[pub], cfg, strict=strict_ranges)
    Xv = X.to_numpy(dtype=float)
    clip_frac = ((Xv[tr] < a) | (Xv[tr] > b)).mean(axis=0)
    p = Prepared(
        name=name, display=dcfg["display"], task=task, features=list(dcfg["features"]),
        X_public=clip_scale(Xv[pub], a, b), y_public=y[pub].astype(float),
        X_train=clip_scale(Xv[tr], a, b), y_train=y[tr].astype(float),
        X_test=clip_scale(Xv[te], a, b), y_test=y[te].astype(float),
        a=a, b=b, range_sources=src, idx_public=pub, idx_train=tr, idx_test=te,
        n_total=n, n_raw=n_raw, clip_frac_train=clip_frac)
    p.X_test.setflags(write=False)
    p.test_hash = array_hash(p.X_test)
    return p


def all_dataset_names(cfg: dict) -> list[str]:
    return list(cfg["datasets"].keys())


def real_dataset_names(cfg: dict) -> list[str]:
    return [k for k, v in cfg["datasets"].items() if not v.get("synthetic")]


def compute_fallback_table(cfg: dict) -> dict:
    """Helper used once (before any experiment) to fill the public-split fallback ranges."""
    out = {}
    q = tuple(cfg["ranges"]["fallback_percentiles"])
    for name in real_dataset_names(cfg):
        dcfg = cfg["datasets"][name]
        X, y, task = load_frame(name, cfg)
        if len(X) > cfg["data"]["max_rows"]:
            rng = np.random.default_rng(cfg["seeds"]["subsample"])
            keep = np.sort(rng.choice(len(X), size=cfg["data"]["max_rows"], replace=False))
            X = X.iloc[keep].reset_index(drop=True); y = y[keep]
        pub, _, _ = split_indices(len(X), y, task, cfg)
        for f, spec in dcfg["features"].items():
            if spec.get("source") == "public-split":
                out[(name, f)] = fallback_range(X.iloc[pub][f].to_numpy(), q)
    return out


if __name__ == "__main__":
    import sys
    cfg = load_config()
    if "--fallback" in sys.argv:
        for (ds, f), (lo, hi) in compute_fallback_table(cfg).items():
            print(f"{ds:14s} {f:16s} [{lo!r}, {hi!r}]")
    else:
        for nm in all_dataset_names(cfg):
            p = prepare(nm, cfg)
            print(nm, p.task, "d =", p.d, "n_pub/train/test =", len(p.idx_public),
                  len(p.idx_train), len(p.idx_test))
