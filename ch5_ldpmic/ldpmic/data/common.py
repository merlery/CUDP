"""Shared dataset utilities: verified downloads, feature schemas with PUBLIC
ranges, and seeded, disk-cached client partitions.

Nothing here looks at the private data to set a range: every ``[a_j, b_j]``
comes from a ``FeatureSchema`` written from the dataset documentation
(``data/RANGES.md``).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("LDPMIC_DATA_DIR", REPO_ROOT / "data" / "cache"))


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, sha256: Optional[str]) -> Path:
    """Download ``url`` to ``dest`` unless present; verify SHA256 when known.

    ``sha256=None`` means the digest has not been recorded yet: the file is
    downloaded and its digest printed so it can be added to
    ``data/MANIFEST.md`` and to the loader; a run never silently proceeds
    with an unrecorded digest (``LDPMIC_ALLOW_UNVERIFIED=1`` overrides for
    first-time recording only).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, dest)
    digest = sha256_file(dest)
    if sha256 is None:
        msg = f"SHA256 of {dest.name} not recorded yet; observed {digest} from {url}"
        if os.environ.get("LDPMIC_ALLOW_UNVERIFIED") != "1":
            raise RuntimeError(msg + " (set LDPMIC_ALLOW_UNVERIFIED=1 to record it)")
        print("WARNING:", msg)
    elif digest != sha256:
        raise RuntimeError(f"SHA256 mismatch for {dest}: expected {sha256}, got {digest}")
    return dest


# --------------------------------------------------------------------------- #
# Feature schema
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class NumericFeature:
    name: str
    low: float
    high: float
    source: str  # where the public range comes from (see data/RANGES.md)


@dataclass(frozen=True)
class CategoricalFeature:
    name: str
    categories: Tuple[str, ...]  # public category list from the documentation
    unknown: Optional[str] = None  # token used for missing values, if any


Feature = object  # NumericFeature | CategoricalFeature


@dataclass
class TabularDataset:
    name: str
    X_train: np.ndarray          # float64, ORIGINAL units (clipped to the public box)
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    ranges: np.ndarray           # (d, 2) public [a_j, b_j]
    feature_names: List[str]
    groups_train: np.ndarray     # group label per training record (for the by-group partition)
    group_name: str
    n_classes: int
    meta: Dict[str, object]

    @property
    def d(self) -> int:
        return self.ranges.shape[0]


def encode(df: pd.DataFrame, features: Sequence[Feature]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Encode a frame into (X, ranges, names): numeric features are clipped to
    their public range; categoricals are one-hot over the DOCUMENTED category
    list (a value outside the list maps to all-zeros) with range [0, 1]."""
    cols, ranges, names = [], [], []
    for f in features:
        if isinstance(f, NumericFeature):
            v = pd.to_numeric(df[f.name], errors="coerce").to_numpy(dtype=np.float64)
            if np.isnan(v).any():
                raise ValueError(f"non-numeric values in {f.name}")
            cols.append(np.clip(v, f.low, f.high)[:, None])
            ranges.append((f.low, f.high))
            names.append(f.name)
        elif isinstance(f, CategoricalFeature):
            s = df[f.name].astype(str).str.strip()
            for c in f.categories:
                cols.append((s == c).to_numpy(dtype=np.float64)[:, None])
                ranges.append((0.0, 1.0))
                names.append(f"{f.name}={c}")
        else:
            raise TypeError(f"unknown feature type {type(f)}")
    return np.concatenate(cols, axis=1), np.asarray(ranges, dtype=float), names


def public_scaler(ranges: np.ndarray):
    """Affine map x -> (x - a) / (b - a) using only PUBLIC ranges; applied to
    sanitized (or raw) data before it reaches the model.  Post-processing."""
    a, b = ranges[:, 0], ranges[:, 1]

    def f(X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - a) / (b - a)

    return f


# --------------------------------------------------------------------------- #
# Partitions
# --------------------------------------------------------------------------- #

def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float, rng: np.random.Generator,
                        min_size: int = 10, max_tries: int = 1000) -> List[np.ndarray]:
    """Label-skew partition with equal client sizes (Hsu, Qi, Brown 2019,
    arXiv:1909.06335): client i draws a class distribution
    ``q_i ~ Dirichlet(alpha * prior)`` (prior = global label frequencies) and
    fills its ``m // n_clients`` slots by sampling labels from ``q_i``,
    taking records from the per-class pools; when a class pool is exhausted
    the remaining slots are filled from the other classes.  Equal sizes keep
    every client trainable even for alpha = 0.1 on a binary label, which the
    per-class-split variant does not."""
    y = np.asarray(y)
    m = len(y)
    classes, prior = np.unique(y, return_counts=True)
    prior = prior / prior.sum()
    pools = [list(rng.permutation(np.where(y == c)[0])) for c in classes]
    sizes = np.full(n_clients, m // n_clients)
    sizes[: m - sizes.sum()] += 1
    if sizes.min() < min_size:
        raise ValueError(f"{m} records over {n_clients} clients gives fewer than min_size={min_size} each")
    parts: List[List[int]] = []
    for i in range(n_clients):
        q = rng.dirichlet(alpha * prior)
        part: List[int] = []
        for _ in range(int(sizes[i])):
            avail = np.array([len(p) > 0 for p in pools])
            w = q * avail
            if w.sum() <= 0:
                w = avail.astype(float)
            k = rng.choice(len(classes), p=w / w.sum())
            part.append(pools[k].pop())
        parts.append(part)
    assert all(len(p) == 0 for p in pools)
    return [np.sort(np.asarray(p, dtype=int)) for p in parts]


def group_partition(groups: np.ndarray, n_clients: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Partition by a categorical attribute (e.g. occupation): every client
    holds records of ONE group.  Each group gets at least one client; the
    remaining clients are assigned to groups proportionally to group size
    (largest remainder), and a group's records are split uniformly at random
    into its clients."""
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    if n_clients < len(uniq):
        raise ValueError(f"need at least {len(uniq)} clients for {len(uniq)} groups")
    quota = np.ones(len(uniq), dtype=int)
    extra = n_clients - len(uniq)
    share = counts / counts.sum() * extra
    quota += share.astype(int)
    rem = extra - int(share.astype(int).sum())
    for k in np.argsort(-(share - share.astype(int)))[:rem]:
        quota[k] += 1
    parts: List[np.ndarray] = []
    for g, q in zip(uniq, quota):
        idx = rng.permutation(np.where(groups == g)[0])
        parts.extend(np.sort(chunk) for chunk in np.array_split(idx, q))
    assert len(parts) == n_clients
    return parts


def partition_cache_path(dataset: str, partition: str, n_clients: int, seed: int, alpha: Optional[float]) -> Path:
    tag = f"{dataset}_{partition}_N{n_clients}_s{seed}" + (f"_a{alpha}" if alpha is not None else "")
    return DATA_DIR / "partitions" / f"{tag}.json"


def make_partition(ds: TabularDataset, partition: str, n_clients: int, seed: int,
                   alpha: Optional[float] = None, cache: bool = True) -> List[np.ndarray]:
    """Return (and cache by seed) the list of training-index arrays per client."""
    path = partition_cache_path(ds.name, partition, n_clients, seed, alpha)
    if cache and path.exists():
        with open(path) as f:
            obj = json.load(f)
        if obj["n_train"] != len(ds.y_train):
            raise RuntimeError(f"stale partition cache {path}")
        return [np.asarray(p, dtype=int) for p in obj["parts"]]
    rng = np.random.default_rng(hash_seed(seed, "partition", ds.name, partition, n_clients, alpha))
    if partition == "group":
        parts = group_partition(ds.groups_train, n_clients, rng)
    elif partition == "dirichlet":
        if alpha is None:
            raise ValueError("dirichlet partition needs alpha")
        parts = dirichlet_partition(ds.y_train, n_clients, alpha, rng)
    else:
        raise ValueError(f"unknown partition {partition!r}")
    all_idx = np.concatenate(parts)
    assert len(all_idx) == len(ds.y_train) and len(np.unique(all_idx)) == len(ds.y_train)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"n_train": len(ds.y_train), "parts": [p.tolist() for p in parts]}, f)
    return parts


def hash_seed(seed: int, *tags) -> int:
    h = hashlib.sha256(("|".join([str(seed)] + [str(t) for t in tags])).encode()).digest()
    return int.from_bytes(h[:4], "little")


def client_data(ds: TabularDataset, parts: Sequence[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [(ds.X_train[p], ds.y_train[p]) for p in parts]
