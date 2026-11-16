"""Download the four public datasets of Chapter 4 into data/raw/ and check their SHA-256 against config.yaml.

    python fetch_data.py

The raw files are not redistributed in this repository. URLs, checksums and notes on each source are in
config.yaml (section data.files) and data/raw/README.md.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    raw = ROOT / cfg["data"]["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    bad = 0
    for key, f in cfg["data"]["files"].items():
        path = raw / f["file"]
        if not path.exists():
            print(f"downloading {f['file']} ...", flush=True)
            urllib.request.urlretrieve(f["url"], path)
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        ok = h == f["sha256"]
        bad += not ok
        print(f"{'OK  ' if ok else 'BAD '} {f['file']}  {h}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
