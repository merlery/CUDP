"""UCI Adult (Census Income).  Standard train/test split (adult.data /
adult.test).  Public ranges and category lists: data/RANGES.md.

Source: https://archive.ics.uci.edu/dataset/2/adult  (files under
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .common import (DATA_DIR, CategoricalFeature, NumericFeature, TabularDataset, download, encode,
                     sha256_file)

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
FILES: Dict[str, Optional[str]] = {
    # file name -> SHA256 (recorded in data/MANIFEST.md)
    "adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
    "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05",
    "adult.names": "c248284c0b5de30c9e1958d6cdd168a34a654758b620e68f46aefa83fc0a576a",
}

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
           "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
           "native-country", "income"]

# Category lists are copied verbatim from adult.names (public documentation).
FEATURES = [
    NumericFeature("age", 17, 90, "adult.names extraction rule AAGE>16; CPS top-code 90"),
    CategoricalFeature("workclass", ("Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
                                     "State-gov", "Without-pay", "Never-worked"), unknown="?"),
    NumericFeature("fnlwgt", 10000, 1500000, "declared a priori (census sampling weight); see RANGES.md"),
    CategoricalFeature("education", ("Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm",
                                     "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th",
                                     "Doctorate", "5th-6th", "Preschool")),
    NumericFeature("education-num", 1, 16, "adult.names: 16 education levels"),
    CategoricalFeature("marital-status", ("Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                                          "Married-spouse-absent", "Married-AF-spouse")),
    CategoricalFeature("occupation", ("Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                                      "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                                      "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                                      "Armed-Forces"), unknown="?"),
    CategoricalFeature("relationship", ("Wife", "Own-child", "Husband", "Not-in-family", "Other-relative",
                                        "Unmarried")),
    CategoricalFeature("race", ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")),
    CategoricalFeature("sex", ("Female", "Male")),
    NumericFeature("capital-gain", 0, 99999, "CPS top-code 99999"),
    NumericFeature("capital-loss", 0, 5000, "declared a priori; see RANGES.md"),
    NumericFeature("hours-per-week", 1, 99, "CPS top-code 99"),
    CategoricalFeature("native-country", (
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)",
        "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland",
        "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador",
        "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
        "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"), unknown="?"),
]

GROUP_COLUMN = "occupation"


def _read(path: Path, test: bool) -> pd.DataFrame:
    text = path.read_text()
    if test:
        # adult.test starts with the line "|1x3 Cross validator" and labels end with "."
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("|")]
        text = "\n".join(lines)
    df = pd.read_csv(io.StringIO(text), header=None, names=COLUMNS, skipinitialspace=True, na_values=[], keep_default_na=False)
    df = df.dropna()
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    return df


SMOKE_DIR = Path(__file__).resolve().parents[2] / "data" / "smoke" / "adult"


def load_adult(data_dir: Optional[Path] = None) -> TabularDataset:
    """Loads from ``data_dir`` (default data/cache/adult), downloading if
    needed.  If the committed smoke-test copy data/smoke/adult/ holds all
    three files, they are used instead (their SHA256 is verified against the
    same recorded digests), so ``make smoke`` and the Docker image need no
    network access."""
    root = Path(data_dir) if data_dir is not None else DATA_DIR / "adult"
    if data_dir is None and all((SMOKE_DIR / n).exists() for n in FILES):
        root = SMOKE_DIR
    paths = {name: download(BASE_URL + name, root / name, sha) for name, sha in FILES.items()}
    train = _read(paths["adult.data"], test=False)
    test = _read(paths["adult.test"], test=True)
    if len(train) != 32561 or len(test) != 16281:
        raise RuntimeError(f"unexpected Adult sizes train={len(train)} test={len(test)} (expected 32561/16281)")
    label = lambda df: (df["income"] == ">50K").to_numpy(dtype=np.int64)  # noqa: E731
    X_tr, ranges, names = encode(train, FEATURES)
    X_te, _, _ = encode(test, FEATURES)
    groups = train[GROUP_COLUMN].astype(str).str.strip().to_numpy()
    return TabularDataset(
        name="adult", X_train=X_tr, y_train=label(train), X_test=X_te, y_test=label(test), ranges=ranges,
        feature_names=names, groups_train=groups, group_name=GROUP_COLUMN, n_classes=2,
        meta={"files": {n: {"url": BASE_URL + n, "sha256": sha256_file(p)} for n, p in paths.items()},
              "n_train": int(len(train)), "n_test": int(len(test)), "d": int(X_tr.shape[1])},
    )
