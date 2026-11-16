"""Shared definitions for make_figures.py and make_tables.py.

Everything the paper shows is read from results/summary/*.csv (never from
results/raw).  A missing required cell raises ``MissingCell`` naming the
(dataset, method, eps) so that a figure or table can never silently show a
blank or a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary"

EPS_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
DELTA = 1e-5

# ONE dictionary of method names, colours, markers and line styles used by
# every figure and table (keys are the canonical method ids).
METHODS: Dict[str, Dict[str, str]] = {
    "nodp":              {"name": "No DP (FedAvg)",        "color": "#444444", "marker": "",  "ls": "--"},
    "iso":               {"name": "Uniform release",       "color": "#1f77b4", "marker": "o", "ls": "-"},
    "ldpmic_global":     {"name": "Exp allocation",        "color": "#d62728", "marker": "s", "ls": "-"},
    "ldpmic_per_client": {"name": "Exp allocation (per-client scores)", "color": "#ff7f0e", "marker": "^", "ls": "-"},
    "local_dpsgd":       {"name": "Local DP-SGD",          "color": "#2ca02c", "marker": "D", "ls": "-"},
    "oracle":            {"name": "Exp allocation, exact scores (non-private)", "color": "#9467bd", "marker": "v", "ls": ":"},
    "cap":               {"name": "Exp allocation + cap", "color": "#8c564b", "marker": "P", "ls": "-"},
    "linear":            {"name": "Linear shares",          "color": "#e377c2", "marker": "X", "ls": "-"},
    "random":            {"name": "Random (permuted exp)",  "color": "#7f7f7f", "marker": "h", "ls": "-"},
    "pearson":           {"name": "Pearson scores",         "color": "#bcbd22", "marker": "<", "ls": "-"},
    "spearman":          {"name": "Spearman scores",        "color": "#17becf", "marker": ">", "ls": "-"},
    # selection experiment (DECISIONS.md D38/D39): private top-k selection + uniform one-shot release
    "sel_mi_k10":        {"name": "Top-10 selection",               "color": "#d62728", "marker": "s", "ls": "-"},
    "sel_mi_k20":        {"name": "Top-20 selection",               "color": "#e8803a", "marker": "^", "ls": "-"},
    "sel_random_k10":    {"name": "Random $k=10$ (control)",        "color": "#7f7f7f", "marker": "h", "ls": "-"},
    "sel_random_k20":    {"name": "Random $k=20$ (control)",        "color": "#a0a0a0", "marker": "H", "ls": "-"},
    "sel_mi_oracle_k10": {"name": "Top-10, exact scores (non-private)", "color": "#9467bd", "marker": "v", "ls": ":"},
    "sel_mi_oracle_k20": {"name": "Top-20, exact scores (non-private)", "color": "#c5a3e0", "marker": "v", "ls": ":"},
    "sel_mi_cons":       {"name": "Top-$k$, conservative bound",         "color": "#8c564b", "marker": "P", "ls": "-"},
    "sel_pearson":       {"name": "Pearson, private",               "color": "#bcbd22", "marker": "<", "ls": "-"},
    "sel_mi_exp":        {"name": "Top-$k$ + exp shares", "color": "#e377c2", "marker": "X", "ls": "-"},
    # dimsample baseline (spec; DECISIONS.md D60): eps on k coordinates sampled per record
    "dimsample_k1":      {"name": "Dim. sampling, $k=1$",   "color": "#8c564b", "marker": "p", "ls": "--"},
    "dimsample_k4":      {"name": "Dim. sampling, $k=4$",   "color": "#bcbd22", "marker": "*", "ls": "--"},
    "dimsample_k16":     {"name": "Dim. sampling, $k=16$",  "color": "#17becf", "marker": "X", "ls": "--"},
}
# privacy-utility figures: every one-shot release and the record-level reference, and the eps at which each
# is recorded (a missing listed cell is an error).  The oracle allocation was run at eps 1 and 5 on the images.
PU_METHODS = ("iso", "ldpmic_global", "ldpmic_per_client", "oracle", "dimsample_k1", "dimsample_k4", "dimsample_k16", "local_dpsgd")
PU_CONFIG_SUFFIX = {"oracle": "ldpmic_oracle"}
PU_EPS = {("mnist", "oracle"): [1.0, 5.0], ("fmnist", "oracle"): [1.0, 5.0]}

# selection experiment: eps grid and the datasets / partitions it was run on
SEL_EPS_GRID = [20.0, 50.0, 100.0, 200.0]
# "rows": the variants recorded for that dataset (DECISIONS.md D39: the Dirichlet
# heterogeneity check has no DP-SGD reference and no ablations); every listed
# row is required and a missing one is an error.
SEL_FULL = ("iso", "sel_random_k10", "sel_random_k20", "sel_mi_k10", "sel_mi_k20",
            "sel_mi_oracle_k10", "sel_mi_oracle_k20", "local_dpsgd")
SEL_DATASETS = {  # key -> (summary dataset, config prefix, display name, no-DP reference, recorded rows)
    "adult_occ":    {"dataset": "adult", "prefix": "adult_occ_v7",    "name": "Adult (occupation)",    "nodp": "adult_occ_nodp",    "rows": SEL_FULL},
    "bank_occ":     {"dataset": "bank",  "prefix": "bank_occ_v7",     "name": "Bank (job)",            "nodp": "bank_occ_nodp",     "rows": SEL_FULL},
    "adult_dir0.5": {"dataset": "adult", "prefix": "adult_dir0.5_v7", "name": "Adult (Dirichlet 0.5)", "nodp": "adult_dir0.5_nodp",
                     "rows": tuple(r for r in SEL_FULL if r != "local_dpsgd")},
}

DATASETS = {  # dataset id -> (display name, main-sweep config prefix, partition display)
    "adult":  {"name": "Adult",   "prefix": "adult_occ",  "partition": "occupation"},
    "bank":   {"name": "Bank",    "prefix": "bank_occ",   "partition": "job"},
    "mnist":  {"name": "MNIST",   "prefix": "mnist_a0.5", "partition": "Dirichlet(0.5)"},
    "fmnist": {"name": "F-MNIST", "prefix": "fmnist_a0.5", "partition": "Dirichlet(0.5)"},
}

# config-name suffix -> method id (main sweeps and ablations)
_SUFFIX = [
    ("_sel_mi_oracle_k10", "sel_mi_oracle_k10"), ("_sel_mi_oracle_k20", "sel_mi_oracle_k20"),
    ("_sel_random_k10", "sel_random_k10"), ("_sel_random_k20", "sel_random_k20"),
    ("_sel_mi_k10", "sel_mi_k10"), ("_sel_mi_k20", "sel_mi_k20"),
    ("_ldpmic_per_client", "ldpmic_per_client"), ("_ldpmic_global", "ldpmic_global"),
    ("_ldpmic_oracle", "oracle"), ("_alloc_oracle", "oracle"), ("_local_dpsgd", "local_dpsgd"),
    ("_cap_sexrace", "cap"), ("_alloc_linear", "linear"), ("_alloc_random", "random"),
    ("_alloc_pearson", "pearson"), ("_alloc_spearman", "spearman"), ("_nodp", "nodp"), ("_iso", "iso"),
]


class MissingCell(RuntimeError):
    pass


def method_of(config: str) -> Optional[str]:
    for suf, m in _SUFFIX:
        if config.endswith(suf):
            return m
    return None


def load_final(dataset: str) -> pd.DataFrame:
    path = SUMMARY / f"{dataset}_final.csv"
    if not path.exists():
        raise MissingCell(f"summary file {path} not found (dataset={dataset})")
    df = pd.read_csv(path)
    # exclude lr-tuning rows, the membership-inference training runs (`*_occ_mi_*`)
    # and the runtime-measurement runs (`*_rt_*`)
    df = df[(~df.lr_tuning) & (~df.config.str.contains("_occ_mi_")) & (~df.config.str.contains("_rt_"))].copy()
    df["method_id"] = df.config.map(method_of)
    return df


def load_csv(name: str) -> pd.DataFrame:
    path = SUMMARY / name
    if not path.exists():
        raise MissingCell(f"summary file {path} not found")
    return pd.read_csv(path)


def cell(df: pd.DataFrame, dataset: str, config: str, eps: Optional[float], what: str = "accuracy") -> pd.Series:
    """The one summary row of (config, eps); raises MissingCell otherwise."""
    sub = df[df.config == config]
    sub = sub[sub.eps.isna()] if eps is None else sub[(sub.eps - eps).abs() < 1e-9]
    if len(sub) != 1:
        raise MissingCell(f"missing cell: dataset={dataset}, config={config} (method={method_of(config)}), eps={eps}"
                          f" ({len(sub)} rows found)")
    return sub.iloc[0]


def pm(mean: float, std: float, digits: int = 3) -> str:
    if pd.isna(mean):
        raise MissingCell("NaN mean in a required cell")
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def fmt_eps(e: float) -> str:
    return f"{e:g}"
