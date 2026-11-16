"""UCI CDC Diabetes Health Indicators (BRFSS 2015) -- the medical dataset of
the exposure study (DECISIONS.md D58).  Sensitive attributes: Age (13-level
age group; label-predictive) and Sex (not label-predictive; a within-dataset
control).

Source: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
(file https://archive.ics.uci.edu/static/public/891/data.csv).  Codes and
ranges are those of the UCI variable descriptions (BRFSS codebook): binary
indicators 0/1; GenHlth 1-5; MentHlth, PhysHlth 0-30 days; Age 1-13 (5-year
groups, 1 = 18-24 ... 13 = 80+); Education 1-6; Income 1-8; Sex 0 = female,
1 = male.  BMI has no documented range and is declared a priori as [10, 100]
(data/RANGES.md).  Age, Income and Sex are one-hot encoded so that the
attribute attacker can target them; the other features are numeric.
There is no official split: a fixed-seed permutation gives 40,000 training
and 10,000 test records (CPU budget).  Clients are formed by Education level
(6 groups -> 50 clients), as Adult and KDD use an occupation attribute.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .common import DATA_DIR, CategoricalFeature, NumericFeature, TabularDataset, download, encode, sha256_file

URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
SHA = "9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964"
N_FULL, N_TRAIN, N_TEST, SPLIT_SEED = 253680, 40000, 10000, 0
LABEL = "Diabetes_binary"
GROUP_COLUMN = "Education"

BINARY = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
          "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk"]
FEATURES = ([NumericFeature(b, 0, 1, "doc: binary indicator") for b in BINARY] + [
    NumericFeature("BMI", 10, 100, "declared a priori (no documented range); see RANGES.md"),
    NumericFeature("GenHlth", 1, 5, "doc: 5-level scale"),
    NumericFeature("MentHlth", 0, 30, "doc: days in the past 30"),
    NumericFeature("PhysHlth", 0, 30, "doc: days in the past 30"),
    NumericFeature("Education", 1, 6, "doc: 6-level scale"),
    CategoricalFeature("Age", tuple(str(i) for i in range(1, 14))),
    CategoricalFeature("Income", tuple(str(i) for i in range(1, 9))),
    CategoricalFeature("Sex", ("Female", "Male")),
])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Map the coded Age / Income levels to category strings and Sex 0/1 to Female/Male."""
    df = df.copy()
    for c in ("Age", "Income"):
        df[c] = df[c].astype(int).astype(str)
    df["Sex"] = df["Sex"].map({0: "Female", 1: "Male"})
    return df


def load_cdc(data_dir: Optional[Path] = None) -> TabularDataset:
    root = Path(data_dir) if data_dir is not None else DATA_DIR / "cdc"
    path = download(URL, root / "data.csv", SHA)
    df = pd.read_csv(path)
    if len(df) != N_FULL:
        raise RuntimeError(f"unexpected CDC size {len(df)} (expected {N_FULL})")
    df = prepare(df)
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(df))
    train = df.iloc[np.sort(perm[:N_TRAIN])].reset_index(drop=True)
    test = df.iloc[np.sort(perm[N_TRAIN:N_TRAIN + N_TEST])].reset_index(drop=True)
    X_tr, ranges, names = encode(train, FEATURES)
    X_te, _, _ = encode(test, FEATURES)
    groups = train[GROUP_COLUMN].astype(int).astype(str).to_numpy()
    return TabularDataset(
        name="cdc", X_train=X_tr, y_train=train[LABEL].to_numpy(dtype=np.int64), X_test=X_te,
        y_test=test[LABEL].to_numpy(dtype=np.int64), ranges=ranges, feature_names=names, groups_train=groups,
        group_name=GROUP_COLUMN, n_classes=2,
        meta={"files": {"data.csv": {"url": URL, "sha256": sha256_file(path)}}, "n_train": int(len(train)),
              "n_test": int(len(test)), "d": int(X_tr.shape[1]),
              "split": {"seed": SPLIT_SEED, "n_full": N_FULL, "rule": "fixed-seed permutation; first 40000 train, next 10000 test"}},
    )
