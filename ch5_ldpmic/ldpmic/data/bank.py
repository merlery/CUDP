"""UCI Bank Marketing, file bank-additional-full.csv (41188 rows, 20 inputs).
No standard train/test split exists: a fixed stratified 80/20 split with its
own constant seed (independent of run seeds) is used.  Public ranges and
category lists: data/RANGES.md.

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .common import (DATA_DIR, CategoricalFeature, NumericFeature, TabularDataset, download, encode,
                     sha256_file)

ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
ZIP_SHA256: Optional[str] = "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4"  # see data/MANIFEST.md
CSV_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"  # bank-additional-full.csv extracted from the zip
CSV_NAME = "bank-additional-full.csv"
SPLIT_SEED = 20240222   # constant; the split is part of the dataset definition, not of a run
TEST_FRACTION = 0.2

MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")

# Category lists copied from bank-additional-names.txt.  ``duration`` is
# excluded on the documentation's own advice ("should be discarded if the
# intention is to have a realistic predictive model").  ``pdays`` (999 = not
# previously contacted) is split into an indicator and a clipped day count.
FEATURES = [
    NumericFeature("age", 17, 100, "declared a priori; see RANGES.md"),
    CategoricalFeature("job", ("admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
                               "self-employed", "services", "student", "technician", "unemployed", "unknown")),
    CategoricalFeature("marital", ("divorced", "married", "single", "unknown")),
    CategoricalFeature("education", ("basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                                     "professional.course", "university.degree", "unknown")),
    CategoricalFeature("default", ("no", "yes", "unknown")),
    CategoricalFeature("housing", ("no", "yes", "unknown")),
    CategoricalFeature("loan", ("no", "yes", "unknown")),
    CategoricalFeature("contact", ("cellular", "telephone")),
    CategoricalFeature("month", MONTHS),
    CategoricalFeature("day_of_week", ("mon", "tue", "wed", "thu", "fri")),
    NumericFeature("campaign", 1, 60, "declared a priori; see RANGES.md"),
    NumericFeature("pdays_contacted", 0, 1, "derived: 1 if pdays != 999 (documented sentinel)"),
    NumericFeature("pdays_days", 0, 30, "derived: pdays if contacted else 0; declared cap 30 days"),
    NumericFeature("previous", 0, 10, "declared a priori; see RANGES.md"),
    CategoricalFeature("poutcome", ("failure", "nonexistent", "success")),
    NumericFeature("emp.var.rate", -4.0, 2.0, "declared a priori (public macro series); see RANGES.md"),
    NumericFeature("cons.price.idx", 90.0, 96.0, "declared a priori (public macro series); see RANGES.md"),
    NumericFeature("cons.conf.idx", -60.0, -20.0, "declared a priori (public macro series); see RANGES.md"),
    NumericFeature("euribor3m", 0.0, 6.0, "declared a priori (public macro series); see RANGES.md"),
    NumericFeature("nr.employed", 4900.0, 5300.0, "declared a priori (public macro series); see RANGES.md"),
]

GROUP_COLUMN = "job"


def _extract_csv(zip_path: Path) -> bytes:
    """Return the bytes of bank-additional-full.csv, searching nested zips."""
    def search(zf: zipfile.ZipFile):
        for info in zf.infolist():
            if info.filename.endswith(CSV_NAME):
                return zf.read(info)
        for info in zf.infolist():
            if info.filename.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(zf.read(info))) as inner:
                    found = search(inner)
                    if found is not None:
                        return found
        return None

    with zipfile.ZipFile(zip_path) as zf:
        data = search(zf)
    if data is None:
        raise RuntimeError(f"{CSV_NAME} not found inside {zip_path}")
    return data


def _stratified_split(y: np.ndarray, test_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    test_idx = []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        test_idx.extend(idx[: int(round(test_fraction * len(idx)))].tolist())
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test_idx] = True
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def load_bank(data_dir: Optional[Path] = None) -> TabularDataset:
    root = Path(data_dir) if data_dir is not None else DATA_DIR / "bank"
    zip_path = download(ZIP_URL, root / "bank+marketing.zip", ZIP_SHA256)
    csv_path = root / CSV_NAME
    if not csv_path.exists():
        csv_path.write_bytes(_extract_csv(zip_path))
    if sha256_file(csv_path) != CSV_SHA256:
        raise RuntimeError(f"SHA256 mismatch for {csv_path}")
    df = pd.read_csv(csv_path, sep=";", na_values=[], keep_default_na=False)
    if len(df) != 41188:
        raise RuntimeError(f"unexpected Bank size {len(df)} (expected 41188)")
    pdays = pd.to_numeric(df["pdays"])
    df["pdays_contacted"] = (pdays != 999).astype(float)
    df["pdays_days"] = np.where(pdays != 999, pdays, 0).astype(float)
    y = (df["y"].astype(str).str.strip() == "yes").to_numpy(dtype=np.int64)
    X, ranges, names = encode(df, FEATURES)
    tr, te = _stratified_split(y, TEST_FRACTION, SPLIT_SEED)
    groups = df[GROUP_COLUMN].astype(str).str.strip().to_numpy()
    return TabularDataset(
        name="bank", X_train=X[tr], y_train=y[tr], X_test=X[te], y_test=y[te], ranges=ranges,
        feature_names=names, groups_train=groups[tr], group_name=GROUP_COLUMN, n_classes=2,
        meta={"files": {"bank+marketing.zip": {"url": ZIP_URL, "sha256": sha256_file(zip_path)},
                        CSV_NAME: {"url": ZIP_URL + "#" + CSV_NAME, "sha256": sha256_file(csv_path)}},
              "split": {"seed": SPLIT_SEED, "test_fraction": TEST_FRACTION, "stratified": True},
              "n_train": int(len(tr)), "n_test": int(len(te)), "d": int(X.shape[1]),
              "dropped": ["duration"]},
    )
