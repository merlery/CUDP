#!/usr/bin/env python
"""Majority-class rate of each test set -> results/summary/majority_rates.csv (needs `make data`)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_sweep as rs  # noqa: E402

rows = []
for ds in ("adult", "bank", "mnist", "fmnist"):
    y = np.asarray(rs.get_dataset(ds).y_test)
    counts = np.bincount(y)
    rows.append({"dataset": ds, "n_test": len(y), "majority_class": int(counts.argmax()), "majority_rate": counts.max() / len(y)})
out = ROOT / "results" / "summary" / "majority_rates.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(pd.DataFrame(rows).to_string(index=False))
