# Chapter 4 experiment: MIC-DP

Reproduces every number, table and figure for Chapter 4 (MIC-DP) from `config.yaml`.

The code implements the enhanced MIC-DP ("v2"): fixed-grid MIC with a provable small sensitivity
computed on the full private-train split (`src/fgmic.py`), shrinkage of noisy scores, relevance
weights that may leave weak columns unreleased (`src/methods.py`), and noise-aware prediction for the
boosting model (`src/noiseaware.py`). The first run of the original design (v1) is not part of this snapshot.

```bash
python -m src.run      # unit tests, data, scores, all trials, sanity checks (stops if any check fails)
python -m src.report   # stats.csv, LaTeX table bodies, figures, summary.md
```

`python -m src.run --fresh` ignores the per-trial cache in `results/ch4/cache/`. The cache is keyed by a
hash of the experiment settings in `config.yaml`, so changing any of them reruns the trials.

## Layout

| Path | Contents |
|---|---|
| `config.yaml` | every setting: seeds, budgets, methods, public ranges (with sources), decision rules for H1-H6 |
| `src/dp.py` | analytic Gaussian calibration c(eps, delta), weighted Gaussian mechanism, score privatization |
| `src/fgmic.py` | fixed-grid MIC, moment release, and their sensitivity bounds |
| `src/noiseaware.py` | exact expected tree prediction under the public release noise |
| `src/mic.py` | own MIC / MIC_e implementation (kept and tested; not used by v2) |
| `src/data.py` | loading, encoding, public ranges, splits, synthetic family |
| `src/scores.py` | MIC_e, \|Pearson\|, \|partial correlation\| |
| `src/methods.py` | weight rules for every method |
| `src/run.py` | main loop (joblib over trials) and sanity checks |
| `src/stats.py` | Wilcoxon + Holm families, bootstrap CIs, hypothesis decisions |
| `src/report.py` | tables, figures, `summary.md` |
| `tests/` | unit tests (`pytest -m "not crosscheck"`); `test_mic_vs_minepy.py` is an optional cross-check |
| `data/raw/` | raw files (SHA-256 pinned in `config.yaml`) |
| `results/ch4/` | all outputs |

## Environment

Python >= 3.10 with numpy, scipy, pandas, scikit-learn, statsmodels, matplotlib, joblib, pyyaml, pyarrow,
pytest. minepy is only needed for the optional cross-check:
`python -m pytest -m crosscheck tests/test_mic_vs_minepy.py` and `python -m tests.test_mic_vs_minepy`
(writes `results/ch4/mic_crosscheck.csv`).
