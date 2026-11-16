# Chapter 5 (LDP-MIC): run from this directory (ch5_ldpmic/).
PYTHON ?= python

.PHONY: setup data check test outputs aggregate followup central

# Python 3.13 environment with the pinned requirements (CPU-only PyTorch wheel).
setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r ../requirements.txt

# Small CPU check (offline, a few minutes): unit tests, then every table and figure of the chapter
# regenerated from the committed summaries and run records into thesis_tex/.
check: test outputs

test:
	$(PYTHON) -m pytest -q

outputs:
	$(PYTHON) scripts/make_figures.py --out figures
	$(PYTHON) scripts/make_tables.py --out tables
	$(PYTHON) scripts/gen_ch5.py

# Download and SHA-256-verify Adult, Bank Marketing, MNIST, Fashion-MNIST into data/cache/ (from the URLs in data/MANIFEST.md).
data:
	$(PYTHON) scripts/download_data.py

# results/raw -> results/summary (needs no data except majority_rates.py, which reads the test labels).
aggregate:
	$(PYTHON) scripts/aggregate.py
	$(PYTHON) scripts/aggregate_attacks.py
	$(PYTHON) scripts/majority_rates.py

# Follow-up runs (Draft 3): CSVs and summary in results/ch5/draft3/ (needs `make data`).
followup:
	$(PYTHON) scripts/ch5d3_report.py diag
	$(PYTHON) scripts/ch5d3_report.py report

# Experiment 5: central training on the uploaded releases (needs `make data`; the report alone needs no data).
central:
	$(PYTHON) scripts/run_central.py
	$(PYTHON) scripts/run_central.py --report
