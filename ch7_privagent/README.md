# Chapter 7 experiment: PrivAgent with simulated users

Evaluates how well the pairwise dialogue of PrivAgent recovers preference weights and whether the weights
lead ULDP (TOPSIS, Chapter 6) to the same privacy budget eps* as the true weights. No human participants.

```bash
python -m src.run --smoke    # 2 replications, 100 users -> results/ch7_smoke/
python -m src.run            # unit tests, static checks, 20 replications, sanity checks, raw outputs
python -m src.report         # stats.csv, LaTeX table bodies, figures, summary.md
```

| Path | Contents |
|---|---|
| `config.yaml` | every setting (pre-registered) |
| `hypotheses.md` | H1-H5 and decision rules (pre-registered) |
| `src/uldp.py` | TOPSIS of Chapter 6 and the decision matrices of Table 6.1 |
| `src/users.py` | true weights, logistic response model, anchored slider |
| `src/dialogue.py` | prompt sets, simulated dialogue, consistency check and repair |
| `src/infer.py` | vote inference (Chapter 7) and Bayesian posterior mean |
| `src/run.py` | main loop (joblib over replications), sanity checks |
| `src/stats.py` | cluster bootstrap, McNemar, Wilcoxon, Holm, H1-H5 decisions |
| `src/report.py` | LaTeX tables, figures, summary.md |
| `tests/` | pytest |
| `results/ch7/` | outputs |

Environment: Python 3.10 with numpy, scipy, pandas, statsmodels, matplotlib, pyyaml, pytest, plus joblib
(parallel replications) and pyarrow (parquet files).
