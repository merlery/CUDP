"""Small CPU check for Chapter 4 (about a minute, no data download needed).

    python check.py

1. Unit tests (the tests that load the four real datasets run only if data/raw/ is filled by fetch_data.py).
2. Recomputes the pre-specified families F1-F5 and F1_secondary from the stored per-trial results
   (results/ch4/raw_trials.parquet) and compares every Holm-adjusted outcome with results/ch4/stats.csv.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import data as D  # noqa: E402
from src import stats as S  # noqa: E402

cfg = D.load_config()
have_data = all((ROOT / cfg["data"]["raw_dir"] / f["file"]).exists() for f in cfg["data"]["files"].values())
args = ["-q", "-p", "no:cacheprovider", "-m", "not crosscheck", str(ROOT / "tests")]
if not have_data:
    args += ["--ignore", str(ROOT / "tests" / "test_data.py")]
    print("data/raw/ is empty: skipping tests/test_data.py (run python fetch_data.py to include it)")
rc = pytest.main(args)
if rc != 0:
    sys.exit("unit tests failed")

raw = S.load_trials(cfg)
new = S.all_tests(raw, cfg)
old = pd.read_csv(ROOT / "results/ch4/stats.csv")
key = ["family", "dataset", "eps", "method_a", "method_b", "model"]
old = old[old.family.isin(new.family.unique())]
m = new.merge(old, on=key, suffixes=("", "_stored"))
same = len(m) == len(new) == len(old) and bool((m.outcome == m.outcome_stored).all())
diff = (m.mean_diff_U - m.mean_diff_U_stored).abs().max()
print(new.groupby(["family", "outcome"]).size().unstack(fill_value=0))
print(f"{'PASS' if same and diff < 1e-12 else 'FAIL'}: {len(new)} family tests recomputed from raw_trials.parquet match "
      f"results/ch4/stats.csv (max |difference| of mean paired difference {diff:.1e})")
sys.exit(0 if same and diff < 1e-12 else 1)
