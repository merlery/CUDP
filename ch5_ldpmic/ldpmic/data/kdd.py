"""UCI Census-Income (KDD) -- the third tabular dataset of the exposure study
(DECISIONS.md D50).  Sensitive attributes: sex, race.

Source: https://archive.ics.uci.edu/dataset/117/census+income+kdd  (zip ->
census.tar.gz -> census-income.data / .test / .names).  The schema (attribute
list and category values) is parsed from census-income.names, i.e. from the
dataset's own documentation; numeric ranges are documented top-codes or
field widths (data/RANGES.md).  Attributes with more than 30 categories or
no predictive meaning for the task are dropped (listed in DROP; D50), which
keeps the one-hot dimension at about 150.  For the CPU budget the training
split is a fixed-seed subsample of 40,000 of the 199,523 records and the
test split of 10,000 of the 99,762 records (D50); both are deterministic
given the seed and recorded in the provenance meta.
"""

from __future__ import annotations

import re
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .common import DATA_DIR, CategoricalFeature, NumericFeature, TabularDataset, download, encode, sha256_file

ZIP_URL = "https://archive.ics.uci.edu/static/public/117/census+income+kdd.zip"
ZIP_SHA = "54fd206becffcfaf099544c3938c681d64a65709800c94c00a4fba0a00df10c9"
INNER = {  # file inside census.tar.gz -> SHA256 (data/MANIFEST.md)
    "census-income.data": "3676a81db7d3528f3f8b9f3c699d0f0aa28db45e6e994fa0b8ed38327539ee86",
    "census-income.test": "98402b1ab879573d0a7f38a699a40258080e25e33d3401e7bf9c96d3fa0fab8c",
    "census-income.names": "d6fea52977f68504ab3149351980458e09bfc3116808da8986646710d254ef98",
}
N_TRAIN_FULL, N_TEST_FULL = 199523, 99762
N_TRAIN, N_TEST, SUBSAMPLE_SEED = 40000, 10000, 0

# numeric attributes: public ranges (data/RANGES.md)
NUMERIC = {
    "age": (0, 90, "CPS top-code 90"),
    "wage per hour": (0, 9999, "declared: 4-digit field (cents), see RANGES.md"),
    "capital gains": (0, 99999, "CPS top-code 99999"),
    "capital losses": (0, 9999, "declared: 4-digit field"),
    "dividends from stocks": (0, 99999, "CPS top-code 99999"),
    "num persons worked for employer": (0, 6, "documented 7 codes 0..6"),
    "weeks worked in year": (0, 52, "weeks in a year"),
}
# dropped attributes (D50): > 30 categories, or sampling weight / survey year
DROP = {"detailed industry recode", "detailed occupation recode", "state of previous residence",
        "detailed household and family stat", "country of birth father", "country of birth mother",
        "country of birth self", "instance weight", "year"}
GROUP_COLUMN = "major occupation code"
LABEL = "label"


def parse_schema(names_text: str) -> Dict[str, Optional[List[str]]]:
    """Attribute -> list of documented category values (None for continuous),
    in file order, from the 'attribute: v1, v2, ...' lines of census-income.names."""
    schema: Dict[str, Optional[List[str]]] = {}
    for line in names_text.splitlines():
        m = re.match(r"^([a-z][a-z0-9' -]+?):\s*(.*?)\.?\s*$", line)
        if not m or line.startswith("|"):
            continue
        name, rest = m.group(1).strip(), m.group(2).strip()
        if rest == "continuous":
            schema[name] = None
        elif rest:
            schema[name] = [v.strip() for v in rest.split(",")]
    return schema


def build_features(schema: Dict[str, Optional[List[str]]]):
    feats = []
    for name, cats in schema.items():
        if name in DROP:
            continue
        if cats is None:
            lo, hi, note = NUMERIC[name]
            feats.append(NumericFeature(name, lo, hi, note))
        else:
            feats.append(CategoricalFeature(name, tuple(cats), unknown="?"))
    return feats


def _extract(root: Path) -> Dict[str, Path]:
    zpath = download(ZIP_URL, root / "census+income+kdd.zip", ZIP_SHA)
    out = root / "extracted"
    out.mkdir(parents=True, exist_ok=True)
    if not all((out / n).exists() for n in INNER):
        with zipfile.ZipFile(zpath) as z:
            z.extract("census.tar.gz", out)
        with tarfile.open(out / "census.tar.gz") as t:
            t.extractall(out, filter="data")
    paths = {}
    for n, sha in INNER.items():
        p = out / n
        if not p.exists():
            raise FileNotFoundError(f"{n} missing after extraction")
        got = sha256_file(p)
        if got != sha:
            raise RuntimeError(f"SHA256 mismatch for {n}: {got}")
        paths[n] = p
    return paths


def _read(path: Path, columns: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=columns, skipinitialspace=True, dtype=str,
                     na_values=[], keep_default_na=False)
    for c, (lo, hi, _) in NUMERIC.items():
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[LABEL] = df[LABEL].str.strip().str.rstrip(".")
    return df.dropna()


def load_kdd(data_dir: Optional[Path] = None) -> TabularDataset:
    root = Path(data_dir) if data_dir is not None else DATA_DIR / "kdd"
    paths = _extract(root)
    schema = parse_schema(paths["census-income.names"].read_text())
    columns = list(schema.keys()) + [LABEL]
    if len(columns) != 42:
        raise RuntimeError(f"expected 41 attributes + label in census-income.names, parsed {len(columns) - 1}")
    train = _read(paths["census-income.data"], columns)
    test = _read(paths["census-income.test"], columns)
    if len(train) != N_TRAIN_FULL or len(test) != N_TEST_FULL:
        raise RuntimeError(f"unexpected sizes train={len(train)} test={len(test)} (expected {N_TRAIN_FULL}/{N_TEST_FULL})")
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    train = train.iloc[np.sort(rng.choice(len(train), N_TRAIN, replace=False))].reset_index(drop=True)
    test = test.iloc[np.sort(rng.choice(len(test), N_TEST, replace=False))].reset_index(drop=True)
    feats = build_features(schema)
    label = lambda df: (df[LABEL] == "50000+").to_numpy(dtype=np.int64)  # noqa: E731
    X_tr, ranges, names = encode(train, feats)
    X_te, _, _ = encode(test, feats)
    groups = train[GROUP_COLUMN].astype(str).str.strip().to_numpy()
    return TabularDataset(
        name="kdd", X_train=X_tr, y_train=label(train), X_test=X_te, y_test=label(test), ranges=ranges,
        feature_names=names, groups_train=groups, group_name=GROUP_COLUMN, n_classes=2,
        meta={"files": {"census+income+kdd.zip": {"url": ZIP_URL, "sha256": ZIP_SHA},
                        **{n: {"url": ZIP_URL + "#census.tar.gz/" + n, "sha256": sha256_file(p)} for n, p in paths.items()}},
              "n_train": int(len(train)), "n_test": int(len(test)), "d": int(X_tr.shape[1]),
              "subsample": {"seed": SUBSAMPLE_SEED, "n_train_full": N_TRAIN_FULL, "n_test_full": N_TEST_FULL},
              "dropped_attributes": sorted(DROP)},
    )
