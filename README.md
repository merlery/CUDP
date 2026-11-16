# CUDP: code for the dissertation

**Correlation-Aware and User-Centric Differential Privacy: Feature-Level Budget Allocation and Preference-Based
Budget Selection in Central and Federated Settings**

Wenjun Yang, University of Washington Tacoma, 2026.

This repository holds the code, configurations, recorded results and check commands for Chapters 4 to 8 of the
dissertation, and the de-identified data of the Chapter 8 user study. Each chapter directory is self-contained and
is run from inside that directory.

| Directory | Chapter |
|---|---|
| [`ch4_micdp/`](ch4_micdp/) | 4: MIC-DP |
| [`ch5_ldpmic/`](ch5_ldpmic/) | 5: LDP-MIC, federated correlation-aware differential privacy enforced by the client |
| [`ch6_prefselect/`](ch6_prefselect/) | 6: PrefSelect, preference-based budget selection with TOPSIS |
| [`ch7_privagent/`](ch7_privagent/) | 7: PrivAgent with simulated users |
| [`ch8_user_study/`](ch8_user_study/) | 8: user study of the preference-based front end |
| [`thesis_figures/`](thesis_figures/) | scripts for individual thesis figures and analyses |

There is no code directory for Chapter 9.

## Setup

Python 3.13 (the runs used 3.13.18), CPU only.

```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

`requirements.txt` pins one version of each package for all chapters; the chapters do not pin conflicting
versions. The extra index provides the CPU-only PyTorch wheel used by Chapter 5. Chapter 8 also keeps its own
two-line `ch8_user_study/code/requirements.txt` (numpy and pandas, same versions).

## Check commands

Each command is run from the chapter directory in the environment above. None of them needs a dataset download.

| Chapter | Command | What it checks |
|---|---|---|
| 4 | `python check.py` | unit tests; recomputes the hypothesis-test families from `results/ch4/raw_trials.parquet` and compares them with `results/ch4/stats.csv` |
| 5 | `make check` | unit tests; regenerates every table and figure of the chapter from the recorded runs into `figures/`, `tables/` and `thesis_tex/` |
| 6 | `python -m pytest -q && python make_ch6.py` | unit tests; rebuilds the Chapter 6 tables and figures and stops if a value differs from the thesis |
| 7 | `python -m src.run --smoke` | the chapter README gives no check command; this is its smoke run (2 replications, 100 users, written to `results/ch7_smoke/`) |
| 8 | `python3 -I code/check_ch8.py data/ch8_user_study_deidentified.csv outputs` | compares 16 reported numbers with the thesis and exits with status 1 if any differ |

The Chapter 5 README also describes `make setup`, which builds a separate `.venv` inside `ch5_ldpmic/` from this
`requirements.txt`; with the environment above it is not needed.

## Thesis tables and figures

### Chapter 4: MIC-DP

The chapter README gives no per-table map. It states that the two commands below reproduce every number, table and
figure of Chapter 4 from `config.yaml`. They need the datasets (`python fetch_data.py`).

| Thesis item | Script | Output |
|---|---|---|
| All trials and sanity checks | `python -m src.run` | `results/ch4/raw_trials.parquet`, `results/ch4/sanity.log` |
| Tables, figures and summary | `python -m src.report` | `results/ch4/`: `stats.csv`, LaTeX table bodies, figures, `summary.md` |

### Chapter 5: LDP-MIC

Taken from [`ch5_ldpmic/README.md`](ch5_ldpmic/README.md#thesis-tables-and-figures--scripts--runs), which also
lists the recorded runs behind each item. Tables are written to `thesis_tex/ch5_tables.tex` and the numbers
quoted in the text to `thesis_tex/ch5_numbers.tex`.

| Thesis item | Label | Script |
|---|---|---|
| Per-feature noise of the one-shot release | `tab:ch5:law` | `scripts/gen_ch5.py` (§1) |
| Uniform release, per-column vs group-wise calibration | `tab:ch5:group` | `scripts/gen_ch5_followup.py` |
| Experiment 1: main sweeps at ε ≤ 10 | `tab:ch5:main` | `scripts/gen_ch5_followup.py` |
| Experiment 1: privacy–utility curves | `fig:ch5:pu` (`pu_ch5.pdf`) | `scripts/gen_ch5_followup.py` |
| Experiment 1, Bank (appendix) | `tab:ch5:bank_main` | `scripts/gen_ch5_followup.py` |
| Experiment 1: first runs (quoted in the text) | `tab:ch5:main-first-runs` | `scripts/gen_ch5.py` (§2) |
| Experiment 1: DP-SGD with the clipping bound chosen per budget (text) | — | `scripts/ch5d3_report.py report` → `results/ch5/draft3/summary.md`, `partC.csv` |
| Experiment 1: input-scale and failure diagnosis (text) | — | `scripts/ch5d3_report.py diag` → `results/ch5/draft3/diag/` |
| Experiment 2: gradient-domain pilots (appendix) | `tab:ch5:pilotA`, `tab:ch5:pilotB` | `scripts/gen_ch5.py` (§3) |
| Experiment 3: selection at large budgets | `tab:ch5:selection`, `fig:ch5:selection` (`selection_ch5.pdf`) | `scripts/make_tables.py`, `scripts/make_figures.py` |
| Experiment 3: balanced accuracy (appendix) | `tab:ch5:selbal` | `scripts/make_tables.py` |
| Experiment 3: score and bound ablation | `tab:ch5:ablation` | `scripts/make_tables.py` |
| Experiment 3: attribute inference | `tab:ch5:exposure` | `scripts/make_tables.py`; attacks by `scripts/run_attacks.py attribute` |
| Experiment 3: local vs secure-aggregation score release | `tab:ch5:local` | `scripts/gen_ch5_followup.py` |
| Experiment 4: input masking for local DP-SGD | `tab:ch5:mask`, `fig:ch5:mask` (`mask_ch5.pdf`) | `scripts/gen_ch5.py` (§5) |
| Experiment 5: central vs federated training | `tab:ch5:central` | `scripts/gen_ch5_followup.py`; runs by `scripts/run_central.py` |

### Chapter 6: PrefSelect

The chapter README gives no per-table map. `make_ch6.py` produces every Chapter 6 table and both figures from
`inputs/ch6_inputs.csv` (the Chapter 4 MIC-DP curves). The README notes that `make_ch6.py` was written for this
snapshot from the chapter's definitions and that the figures are redrawn from the same values.

| Thesis item | Script | Output |
|---|---|---|
| Every Chapter 6 table | `make_ch6.py` | `results/tab_*.csv` |
| Both Chapter 6 figures | `make_ch6.py` | `results/tradeoff_ch6.pdf`, `results/selection_ch6.pdf` |
| Comparison with the values printed in the thesis | `make_ch6.py` | `results/checks.log` |

### Chapter 7: PrivAgent

The chapter README gives no per-table map.

| Thesis item | Script | Output |
|---|---|---|
| Simulation (20 replications) and sanity checks | `python -m src.run` | `results/ch7/` raw outputs, `sanity.log` |
| Tables, figures and summary | `python -m src.report` | `results/ch7/tables/*.tex`, `results/ch7/figures/`, `results/ch7/summary.md` |

### Chapter 8: user study

Taken from [`ch8_user_study/CHAPTER_NOTES.md`](ch8_user_study/CHAPTER_NOTES.md). Figure files keep their `_ch9`
names from an earlier chapter numbering.

| Thesis item | Script | Output |
|---|---|---|
| Table: Participants of the user study (`tab:ch9:participants`) | `analysis.py` (not included) | table in the text |
| Figure: Self-rated understanding before and after the demo (`fig:ch9:comprehension`) | `analysis.py` (not included) | `comprehension_ch9.pdf` |
| Figure: Weights from sliders and from PrivAgent (`fig:ch9:weights`) | `analysis.py` (not included) | `weights_ch9.pdf` |
| Table: Budgets selected from the participants' weights (`tab:ch9:selection`) | `analysis.py` (not included) | table in the text |
| Figure: Recommended, selected and chosen budgets (`fig:ch9:budgets`) | `analysis.py` (not included) | `budgets_ch9.pdf` |
| Figure: Ratings of the tool (`fig:ch9:ratings`) | `analysis.py` (not included) | `ratings_ch9.pdf` |
| Slider movement, paragraph "Slider Weights Stay Near Their Start Position" | `code/ch8_extra_analysis.py` | `outputs/n100/extra_analysis.json` (`anchoring`) |
| ε* = 1 interval and direct choice, paragraph "With Sliders That Started at 40/40/20" | `code/ch8_extra_analysis.py` | `outputs/n100/extra_analysis.json` (`anchoring.left_eps1_interval_*`, `anchoring.C2_dist`) |
| Variance of log10 ε, paragraph "The Chosen Budget Depended on the Kind of Data, but Far More on the Person" | `code/ch8_extra_analysis.py` | `outputs/n100/extra_analysis.json` (`variance_log_eps`) |
| N = 97 check, Sensitivity Analysis | `code/check_ch8.py` | `outputs/n97/extra_analysis.json` |

The main analysis script `analysis.py` and the open-answer coding script `code_open.py` are not part of this
repository. Bootstrap intervals computed from the public data may differ from the thesis in the last digit because
the rows were reordered; point estimates are identical (see the chapter notes).

### `thesis_figures/`

| Thesis item | Script | Status |
|---|---|---|
| Figure 5.1 | `make_fig5_1_pu_ch5.py` | not included in this snapshot; the Chapter 5 README lists `pu_ch5.pdf` as generated by `ch5_ldpmic/scripts/gen_ch5_followup.py` |
| Figure 8.3 | `make_fig8_3_budgets.py` | not included in this snapshot; the Chapter 8 notes list `budgets_ch9.pdf` as generated by `analysis.py`, which is also not included |
| Chapter 8 follow-up analysis | `ch8_extra_analysis.py` | same file as `ch8_user_study/code/ch8_extra_analysis.py`; run it as described in the Chapter 8 notes |

## Datasets

The datasets are not redistributed. Each chapter downloads them from the original sources or public mirrors and
checks their SHA-256.

| Chapter | Datasets | Sources and checksums |
|---|---|---|
| 4 | Adult; CDC Diabetes Health Indicators (BRFSS 2015); California Housing; Appliances Energy Prediction; synthetic data generated from `config.yaml` | `ch4_micdp/data/raw/README.md`, `ch4_micdp/config.yaml` (`data.files`); download with `python fetch_data.py` |
| 5 | UCI Adult, UCI Bank Marketing, MNIST, Fashion-MNIST | `ch5_ldpmic/data/MANIFEST.md`; download with `make data` |
| 6 | none; uses the Chapter 4 results in `inputs/ch6_inputs.csv` | — |
| 7 | none; simulated users | — |
| 8 | the de-identified user study data, included | `ch8_user_study/data/` |

## Development repositories

The reported runs were made in development repositories, which are not public. These directories are snapshots of
that code without its history.

| Chapter | Commits |
|---|---|
| 4 | not stated in the chapter README; the run records name the commits (`results/ch4/public/prerun_commit.txt`, `results/ch4/public/summary.md`, `draft3_notes.md`) |
| 5 | follow-up runs launched at `a342959` (`a3429596b1390fef5c7e28be38114de4b5d306fc`); each run records its own commit in its JSON (`git_commit`). The robustness evaluation cited in the discussion is from commit `65fd7b6` of an earlier code base and is not included. |
| 6 | not stated in the chapter README |
| 7 | not stated in the chapter README; the pre-registration commit is recorded in `results/ch7/preregistration.txt` |
| 8 | none; the code was not developed in a repository (see `CHAPTER_NOTES.md`) |

## User study data

The Chapter 8 data are de-identified. The study was reviewed by the University of Washington Human Subjects
Division and determined to be exempt (IRB STUDY00025867). `ch8_user_study/data/README.md` lists what was removed
and describes each column; `ch8_user_study/CHAPTER_NOTES.md` lists what was excluded and why.

## Citation and license

Citation metadata is in `CITATION.cff`. The code is released under the MIT License (`LICENSE`). The datasets keep
their own licenses.
