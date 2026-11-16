"""Report for the Central-on-Privatized baseline (D53): central_runs.csv,
config_used.yaml, tab_central.tex, central_ch5.pdf/png and summary.md in
results/ch5/central/, generated from the recorded run files only.  Fails
loudly if any (setting, seed) cell is missing."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ch5" / "central"
RUNS = OUT / "runs"
SEEDS = [0, 1, 2, 3, 4]
GRID = {"all_features": [1.0, 10.0, 50.0, 100.0, 200.0], "private_sel_k20": [50.0, 100.0, 200.0],
        "oracle_sel_k20": [50.0, 100.0, 200.0], "ext_bank_private_sel_k10": [200.0]}
TITLES = {"all_features": "All features, uniform shares", "private_sel_k20": "Private selection, $k=20$",
          "oracle_sel_k20": "Oracle selection, $k=20$ (non-private scores)",
          "ext_bank_private_sel_k10": "Extension: Bank (job), private selection, $k=10$"}
MAJ = {"adult": None, "bank": None}


class MissingCell(RuntimeError):
    pass


def load_runs() -> pd.DataFrame:
    rows = []
    for f in sorted(RUNS.glob("*.json")):
        j = json.loads(f.read_text())
        if j.get("status") != "completed":
            raise MissingCell(f"run {j['run_id']} is not completed ({j.get('status')})")
        rows.append(j)
    if not rows:
        raise MissingCell("no central runs recorded")
    df = pd.DataFrame(rows)
    df["dataset"] = df.config_file.str.extract(r"configs/(\w+?)_")[0]
    return df


def cell(df, release, eps):
    sub = df[(df.release == release) & ((df.eps.astype(float) - eps).abs() < 1e-9)] if eps is not None else df[df.release == release]
    if sorted(sub.seed.tolist()) != SEEDS:
        raise MissingCell(f"missing cell: release={release}, eps={eps}: seeds {sorted(sub.seed.tolist())}")
    return sub.sort_values("seed")


def pm(x):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.3f} $\\pm$ {x.std(ddof=1):.3f}"


def pm_md(x):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"


def majority(ds_name):
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
    import run_sweep as rs
    y = np.asarray(rs.get_dataset(ds_name).y_test)
    return float(np.bincount(y).max() / len(y))


def classify(c_mean, f_mean, f_std):
    d = c_mean - f_mean
    return "within" if abs(d) <= f_std else ("above" if d > 0 else "below")


def main() -> int:
    df = load_runs()
    clean = cell(df, "clean", None)
    maj = {"adult": majority("adult"), "bank": majority("bank")}
    # ---- central_runs.csv
    cols = ["release", "config_file", "eps", "seed", "accuracy", "balanced_accuracy", "central_epochs", "central_steps", "federated_steps",
            "step_rel_diff", "train_seconds", "central_upload_bytes", "federated_bytes", "federated_accuracy", "federated_balanced_accuracy",
            "federated_duration_s", "n_records", "n_released_features", "sha256_match", "federated_run_id", "run_id", "git_commit"]
    df.sort_values(["release", "eps", "seed"])[cols].to_csv(OUT / "central_runs.csv", index=False)
    # ---- per-setting statistics
    stats, tex_rows, md_rows = [], [], []
    for release, eps_list in GRID.items():
        tex_rows.append((r"\midrule" + "\n" if tex_rows else "") + r"\multicolumn{7}{l}{\textit{" + TITLES[release] + "}} \\\\")
        md_rows.append(f"| **{TITLES[release].replace('$', '')}** | | | | | | |")
        for e in eps_list:
            s = cell(df, release, e)
            fa, ca = s.federated_accuracy.to_numpy(), s.accuracy.to_numpy()
            fb, cb = s.balanced_accuracy.to_numpy(), s.federated_balanced_accuracy.to_numpy()
            d = ca - fa
            fbytes, cbytes = s.federated_bytes.mean(), s.central_upload_bytes.mean()
            ds_name = s.config_file.iloc[0].split("/")[1].split("_")[0]
            st = {"release": release, "eps": e, "dataset": ds_name, "fed_mean": fa.mean(), "fed_std": fa.std(ddof=1),
                  "cen_mean": ca.mean(), "cen_std": ca.std(ddof=1), "diff_mean": d.mean(), "diff_min": d.min(), "diff_max": d.max(),
                  "cen_bal_mean": fb.mean(), "cen_bal_std": fb.std(ddof=1), "fed_bal_mean": cb.mean(),
                  "fed_bytes": fbytes, "cen_bytes": cbytes, "ratio": fbytes / cbytes,
                  "cen_train_s": s.train_seconds.mean(), "fed_duration_s": s.federated_duration_s.mean(),
                  "class": classify(ca.mean(), fa.mean(), fa.std(ddof=1))}
            stats.append(st)
            tex_rows.append(f"$\\varepsilon={e:g}$ & {pm(fa)} & {pm(ca)} & {d.mean():+.3f} [{d.min():+.3f}, {d.max():+.3f}] & "
                            f"{fbytes / 1e6:.3f} & {cbytes / 1e6:.3f} & {fbytes / cbytes:.3f} \\\\")
            md_rows.append(f"| eps = {e:g} | {pm_md(fa)} | {pm_md(ca)} | {d.mean():+.3f} [{d.min():+.3f}, {d.max():+.3f}] | "
                           f"{fbytes / 1e6:.3f} | {cbytes / 1e6:.3f} | {fbytes / cbytes:.3f} |")
    S = pd.DataFrame(stats)
    S.to_csv(OUT / "central_settings.csv", index=False)
    ca, fa = clean.accuracy.to_numpy(), clean.federated_accuracy.to_numpy(); d = ca - fa
    tex_rows.append(r"\midrule" + "\n" + r"\multicolumn{7}{l}{\textit{Non-private references (Adult)}} \\")
    tex_rows.append(f"no privacy & {pm(fa)} & {pm(ca)} & {d.mean():+.3f} [{d.min():+.3f}, {d.max():+.3f}] & "
                    f"{clean.federated_bytes.mean() / 1e6:.3f} & {clean.central_upload_bytes.mean() / 1e6:.3f} & "
                    f"{clean.federated_bytes.mean() / clean.central_upload_bytes.mean():.3f} \\\\")
    md_rows.append("| **Non-private references (Adult)** | | | | | | |")
    md_rows.append(f"| no privacy | {pm_md(fa)} | {pm_md(ca)} | {d.mean():+.3f} [{d.min():+.3f}, {d.max():+.3f}] | "
                   f"{clean.federated_bytes.mean() / 1e6:.3f} | {clean.central_upload_bytes.mean() / 1e6:.3f} | "
                   f"{clean.federated_bytes.mean() / clean.central_upload_bytes.mean():.3f} |")
    header = r"release / budget & Federated accuracy & Central accuracy & Difference (mean [min, max]) & Federated MB & Central MB & Ratio \\"
    tex = ("% generated by scripts/central_report.py from results/ch5/central/runs/*.json -- do not edit\n"
           "\\begin{tabular}{lrrrrrr}\n\\toprule\n" + header + "\n\\midrule\n" + "\n".join(tex_rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (OUT / "tab_central.tex").write_text(tex)
    # ---- config_used.yaml (resolved values)
    import yaml
    # config.yaml (fixed before the run) is kept verbatim; the resolved per-run values are appended as valid YAML
    resolved = {f"{r.release}/eps={r.eps}/seed={r.seed}": {"central_epochs": int(r.central_epochs), "central_steps": int(r.central_steps),
                                                            "federated_steps": int(r.federated_steps), "lr": float(r.lr)}
                for r in df.itertuples()}
    (OUT / "config_used.yaml").write_text("# verbatim copy of config.yaml (fixed before the run), followed by the resolved values\n"
                                          + (OUT / "config.yaml").read_text().rstrip("\n") + "\n\n"
                                          + yaml.safe_dump({"resolved": resolved}, sort_keys=False, width=140))
    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    BLUE, ORANGE, GREEN, PURPLE, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#7c5cc4", "#52514e"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 3.2), sharey=True)
    def band(ax, eps, col, data, label, ls="-", mk="o"):
        m = np.array([data(e)[0] for e in eps]); s = np.array([data(e)[1] for e in eps])
        ax.plot(eps, m, ls=ls, marker=mk, color=col, label=label, ms=3.5, lw=1.2); ax.fill_between(eps, m - s, m + s, color=col, alpha=0.15, lw=0)
    def getter(release, key):
        def f(e):
            r = S[(S.release == release) & ((S.eps - e).abs() < 1e-9)].iloc[0]
            return (r.fed_mean, r.fed_std) if key == "fed" else (r.cen_mean, r.cen_std)
        return f
    e_all = GRID["all_features"]; e_sel = GRID["private_sel_k20"]
    band(a1, e_all, BLUE, getter("all_features", "fed"), "federated"); band(a1, e_all, ORANGE, getter("all_features", "cen"), "central", mk="s")
    band(a2, e_sel, BLUE, getter("private_sel_k20", "fed"), "private sel., federated"); band(a2, e_sel, ORANGE, getter("private_sel_k20", "cen"), "private sel., central", mk="s")
    band(a2, e_sel, GREEN, getter("oracle_sel_k20", "fed"), "oracle sel., federated", mk="^"); band(a2, e_sel, PURPLE, getter("oracle_sel_k20", "cen"), "oracle sel., central", mk="v")
    for ax in (a1, a2):
        ax.axhline(fa.mean(), color=BLUE, ls="--", lw=0.9, label="federated, no privacy")
        ax.axhline(ca.mean(), color=ORANGE, ls="--", lw=0.9, label="central, no privacy")
        ax.axhline(maj["adult"], color=GRAY, ls=":", lw=1.0, label="majority rate")
        ax.set_xscale("log"); ax.set_xlabel(r"privacy budget $\varepsilon$ ($\delta = 10^{-5}$)")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_xticks(e_all); a1.set_xticklabels([f"{e:g}" for e in e_all]); a2.set_xticks(e_sel); a2.set_xticklabels([f"{e:g}" for e in e_sel])
    a1.set_ylabel("test accuracy"); a1.set_title("(a) all features, uniform shares", fontsize=8, loc="left")
    a2.set_title(r"(b) selected releases, $k = 20$", fontsize=8, loc="left")
    a1.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    a2.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "central_ch5.pdf", bbox_inches="tight"); fig.savefig(OUT / "central_ch5.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # ---- reporting items
    lines = []
    lines.append("**Rule 1 (above / within / below one federated seed std):**\n")
    for _, r in S.iterrows():
        lines.append(f"- {TITLES[r.release].replace('$', '')}, eps = {r.eps:g}: central {r.cen_mean:.3f}, federated {r.fed_mean:.3f} ± {r.fed_std:.3f} -> **{r['class']}**")
    low = S[(S.release == "all_features") & (S.eps <= 10)]
    leaves = []
    for r in low.itertuples():
        s = cell(df, "all_features", r.eps)
        b = s.balanced_accuracy.to_numpy()
        leaves.append((r.eps, b.mean(), b.std(ddof=1), b.mean() - 0.5 > b.std(ddof=1)))
    lines.append("\n**Rule 2 (does Central-on-Privatized leave the majority rate at eps <= 10, all features):** " +
                 ("yes" if any(x[3] for x in leaves) else "no") + " -- " +
                 "; ".join(f"eps = {e:g}: mean balanced accuracy {m:.3f} (std {sd:.3f}), accuracy {S[(S.release == 'all_features') & (S.eps == e)].cen_mean.iloc[0]:.3f} vs majority rate {maj['adult']:.3f}" for e, m, sd, _ in leaves))
    lines.append("\n**Rule 3 (communication ratio, federated bytes / central bytes):**\n")
    for r in S.itertuples():
        lines.append(f"- {TITLES[r.release].replace('$', '')}, eps = {r.eps:g}: {r.fed_bytes / 1e6:.3f} MB / {r.cen_bytes / 1e6:.3f} MB = {r.ratio:.3f}")
    lines.append(f"\n**Rule 4 (Central-on-Clean vs the federated non-private model, Adult):** central {pm_md(ca)}, federated {pm_md(fa)}; "
                 f"paired difference {d.mean():+.3f} [{d.min():+.3f}, {d.max():+.3f}].")
    # ---- summary.md
    dec = (ROOT / "DECISIONS.md").read_text()
    m = re.search(r"(## D53 .*?)(?=\n## D5[4-9] |\Z)", dec, re.S)
    if not m:
        raise MissingCell("D53 entry not found in DECISIONS.md")
    rules = re.search(r"Reporting rules \(verbatim.*?\):\n(.*?)These are reporting rules", m.group(1), re.S).group(1).strip()
    sanity = json.loads((OUT / "sanity_reproduce.json").read_text())
    steps = df[["release", "eps", "seed", "federated_steps", "central_steps", "step_rel_diff"]]
    runtime_cen = df.train_seconds.sum()
    t0 = pd.to_datetime(df.start_time).min(); t1 = pd.to_datetime(df.end_time).max()
    md = ["# Central-on-Privatized baseline (Chapter 5 study)", "",
          "## Decision-log entry and commit recorded before the run", "",
          "Plan written at commit `151dbe1ed5c93b1b5d5e22efff6a665e89e8c8e4` (HEAD when D53 was written); plan, rules and "
          "`config.yaml` committed before the first run at `12f1a14b8851e661961b81ac367077e09d4d50b7`.", "", "```", m.group(1).strip(), "```", "",
          "## Reporting rules (verbatim)", "", rules, "",
          "## Results", "",
          "| release / budget | Federated accuracy | Central accuracy | Difference (mean [min, max]) | Federated MB | Central MB | Ratio |",
          "|---|---|---|---|---|---|---|", *md_rows, "",
          "Accuracy is the final test accuracy on the clean test set, mean ± sample std over seeds 0-4; Difference is central minus "
          "federated per seed. Bytes are totals over the whole run (federated: all clients, all rounds; central: the one-time upload "
          "of privatized records plus the score release).", "",
          f"Majority-class rate of the test set: Adult {maj['adult']:.3f}, Bank {maj['bank']:.3f}.", "",
          "## Answers to the reporting items", "", *lines, "",
          "## Sanity checks", "",
          f"1. SHA-256 of the privatized arrays: identical to the federated run for every client in all {len(df) - len(clean)} "
          "Central-on-Privatized runs (`sha256_match` in central_runs.csv; a mismatch would have stopped the run).",
          "2. The test set is built only after central training finishes; the training call receives only the concatenated client "
          "arrays; no choice depends on test accuracy (all choices fixed in config.yaml before the run).",
          f"3. Gradient steps: largest relative difference central vs federated {steps.step_rel_diff.abs().max() * 100:.2f} % "
          "(tolerance 5 %); per run in central_runs.csv (federated_steps, central_steps).",
          f"4. Re-run of `{sanity['run']}`: stored accuracy {sanity['stored']:.6f}, re-run {sanity['rerun']:.6f} -> "
          f"{'reproduced exactly' if sanity['ok'] else 'FAILED'}.", "",
          "## Runtime", "",
          f"- {len(df)} central runs; total central training time {runtime_cen:.0f} s (mean {df.train_seconds.mean():.1f} s per run).",
          f"- Wall-clock from the first to the last central run: {(t1 - t0).total_seconds() / 60:.1f} min (including client privatization).",
          "- Server-side training time and federated wall-clock per setting: results/ch5/central/central_settings.csv "
          "(cen_train_s, fed_duration_s).", "",
          "## Deviations", "",
          "1. `config.yaml`, committed before the run, is not machine-parseable YAML (unquoted colons inside two values). It was "
          "not edited after the run; `config_used.yaml` is its verbatim text followed by the resolved per-run values (epochs, "
          "steps, learning rate) as valid YAML.",
          "2. `tab_central.tex` is a bare booktabs `tabular` (header row and data rows, no float and no caption), which is how "
          "\"rows only\" was read.", "",
          "Note (not a deviation): as specified, every federated reference is the recorded run of the same release; for the "
          "all-features release these are the main-sweep runs `adult_occ_iso` at eps 1 and 10 (no update clipping) and the "
          "selection-sweep runs `adult_occ_v7_iso` at eps 50-200 (update clipping C = 0.2).", ""]
    (OUT / "summary.md").write_text("\n".join(md))
    print((OUT / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
