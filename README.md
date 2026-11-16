# CUDP — dissertation code

Public code snapshot of the dissertation. Each chapter has its own directory with its code, configurations,
recorded runs, the scripts that generate its tables and figures, and one small CPU check command.

| Chapter | Directory | Thesis tables and figures | CPU check (from the chapter directory) | Code commit |
|---|---|---|---|---|
| 5 — LDP-MIC: federated correlation-aware differential privacy enforced by the client | [`ch5_ldpmic/`](ch5_ldpmic/) | see [`ch5_ldpmic/README.md`](ch5_ldpmic/README.md#thesis-tables-and-figures--scripts--runs) | `make setup && . .venv/bin/activate && make check` | `a342959` |

## Setup

Python 3.13, CPU only.

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Datasets are not included; each chapter downloads them from their original sources and verifies their SHA-256
(see the chapter's `data/MANIFEST.md`).

## License

MIT (see `LICENSE`). The datasets keep their own licenses.
