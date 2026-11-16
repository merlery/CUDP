# Chapter 6: PrefSelect (preference-based budget selection with TOPSIS)

| Path | Contents |
|---|---|
| `src/prefselect.py` | TOPSIS selection of the budget (Eqs. 6.6 to 6.12), the two LP maps, the Adult and CDC-Diabetes decision matrices, privacy share. The same file is `ch7_privagent/src/uldp.py`, which Chapter 7 uses. |
| `make_ch6.py` | Builds the decision matrices from the Chapter 4 MIC-DP curves and produces every Chapter 6 table and both figures; checks the results against the values printed in the thesis and stops if one differs. |
| `inputs/ch6_inputs.csv` | Mean MIC-DP utility per budget (rounded to 4 decimals as in the Chapter 4 tables), non-private utility, and the trivial predictor (ROC-AUC 0.5; for regression the MAE of the training mean on the test split). Rebuild with `python make_ch6.py --from-ch4 ../ch4_micdp/results/ch4/raw_trials.parquet`. |
| `results/` | Output of `make_ch6.py`: tables as CSV, `tradeoff_ch6.pdf/png`, `selection_ch6.pdf/png`, `checks.log`. |
| `tests/` | Closeness coefficients and selected budgets of the profile table. |

`make_ch6.py` was written for this public snapshot from the definitions of Chapter 6; the script used while writing
the chapter is not part of the snapshot. Its output reproduces the numbers of the thesis (see `results/checks.log`).
The figures are redrawn from the same values and may differ in style from the thesis figures.

Check (a few seconds): `python -m pytest -q && python make_ch6.py`
