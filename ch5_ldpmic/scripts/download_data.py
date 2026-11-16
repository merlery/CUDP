#!/usr/bin/env python
"""Download and verify every dataset (`make data`).

Each loader downloads its files from the official source recorded in
data/MANIFEST.md and refuses any file whose SHA256 differs from the recorded
digest.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldpmic import data as ldata  # noqa: E402

names = sys.argv[1:] or ["adult", "bank", "mnist", "fmnist"]
for name in names:
    ds = ldata.load(name)
    print(f"{name}: train {ds.X_train.shape}, test {ds.X_test.shape}, d={ds.d}; files verified:")
    for f, meta in ds.meta["files"].items():
        print(f"   {f}  sha256={meta['sha256']}")
