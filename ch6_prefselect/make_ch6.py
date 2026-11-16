"""Chapter 6 (PrefSelect): decision matrices, profile table, switch points, fixed defaults, fixed-ideal variant,
and the two figures, from the measured MIC-DP curves of Chapter 4.

    python make_ch6.py            # uses inputs/ch6_inputs.csv (no data download needed)
    python make_ch6.py --from-ch4 ../ch4_micdp/results/ch4/raw_trials.parquet   # rebuild the inputs first

Written for this public snapshot from the definitions of Chapter 6 (the original evaluation script is not part of
it); src/prefselect.py is the TOPSIS code used by Chapter 7. The script checks its results against the values
printed in the thesis and exits with an error if one differs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import prefselect as P

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CAND = P.CANDIDATES
DS = ["adult", "cdc", "calhousing", "appliances"]
DS_TXT = {"adult": "Adult", "cdc": "CDC-Diabetes", "calhousing": "CalHousing", "appliances": "Appliances"}
PROFILES = {"Privacy-focused": (0.6, 0.2, 0.2), "Accuracy-focused": (0.2, 0.6, 0.2),
            "Balanced": (1 / 3, 1 / 3, 1 / 3), "Resource-constrained": (0.2, 0.2, 0.6)}


def inputs_from_ch4(path: str) -> pd.DataFrame:
    r = pd.read_parquet(path)
    r = r[(r.experiment == "main") & (r.model == "primary")]
    key = {"adult": "adult", "cdc": "cdc_diabetes", "calhousing": "calhousing", "appliances": "appliances"}
    # constant predictor (training mean) on the test split, computed with ch4_micdp/src/data.py (needs the raw data)
    const_mae = {"calhousing": 0.9248, "appliances": 61.4116}
    rows = []
    for d, k in key.items():
        g = r[r.dataset == k]
        unp = g[g.method == "nonprivate"].primary_value.mean()
        for e in CAND:
            u = g[(g.method == "mic_dp") & np.isclose(g.eps, e)].primary_value.mean()
            # rounded to 4 decimals, as printed in the Chapter 4 tables from which the Chapter 6 matrices were built
            rows.append({"dataset": d, "eps": e, "U": round(float(u), 4), "U_nonprivate": round(float(unp), 4),
                         "U_trivial": 0.5 if d in ("adult", "cdc") else const_mae[d],
                         "metric": "roc_auc" if d in ("adult", "cdc") else "mae"})
    return pd.DataFrame(rows)


def la_column(g: pd.DataFrame) -> np.ndarray:
    sgn = 1.0 if g.metric.iloc[0] == "roc_auc" else -1.0          # U = ROC-AUC or -MAE
    u, u0, unp = sgn * g.U.to_numpy(), sgn * g.U_trivial.iloc[0], sgn * g.U_nonprivate.iloc[0]
    return np.clip((u - u0) / (unp - u0), 0.0, 1.0)


def matrix(la: np.ndarray, lp_map: str) -> np.ndarray:
    return np.column_stack([P.LP[lp_map], la, np.ones(len(CAND))])


def simplex_grid(step=0.05):
    n = int(round(1 / step))
    W = np.array([(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)], float) / n
    return W[~((W[:, 0] == 0) & (W[:, 1] == 0))]          # (0, 0, 1) leaves every C undefined


def fixed_ideal_select(M, W):
    V = W[:, None, :] * M[None, :, :]
    dp = np.sqrt(((V - W[:, None, :]) ** 2).sum(2))
    dn = np.sqrt((V ** 2).sum(2))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(dp + dn > 0, dn / (dp + dn), np.nan)
    und = np.all(np.isnan(C), axis=1)
    return np.where(und, -1, np.argmax(np.where(np.isnan(C), -np.inf, C), axis=1))


def fail(msg):
    sys.exit("CHECK FAILED: " + msg)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    OUT.mkdir(exist_ok=True)
    if "--from-ch4" in argv:
        inp = inputs_from_ch4(argv[argv.index("--from-ch4") + 1])
        inp.to_csv(HERE / "inputs" / "ch6_inputs.csv", index=False)
    inp = pd.read_csv(HERE / "inputs" / "ch6_inputs.csv")
    LA = {d: la_column(inp[inp.dataset == d]) for d in DS}
    lines = []

    # ---- Table 6.x: decision matrices (tab:ch6:matrix)
    rows = []
    for d in DS:
        g = inp[inp.dataset == d]
        rows.append({"dataset": DS_TXT[d], "row": "U", **{f"eps={e:g}": u for e, u in zip(CAND, g.U)}})
        rows.append({"dataset": DS_TXT[d], "row": "LA", **{f"eps={e:g}": v for e, v in zip(CAND, LA[d])}})
    pd.DataFrame(rows).round(4).to_csv(OUT / "tab_matrix.csv", index=False)
    for d in ("adult", "cdc"):
        if np.max(np.abs(LA[d] - P.LA[d])) > 5e-6:
            fail(f"LA of {d} differs from src/prefselect.py: {LA[d]} vs {P.LA[d]}")
    lines.append("PASS LA columns of Adult and CDC-Diabetes equal the matrices hard-coded in src/prefselect.py (5e-6)")
    if not (abs(LA["calhousing"][-1] - 0.016) < 5e-4 and np.all(LA["appliances"] == 0)):
        fail("CalHousing / Appliances LA differ from the thesis")
    lines.append("PASS CalHousing LA at eps = 10 is 0.016 and Appliances LA is 0 at every budget (thesis table)")

    # ---- profiles (tab:ch6:profiles)
    rows = []
    for d in DS:
        for name, w in PROFILES.items():
            idx, C = P.select(matrix(LA[d], "linear"), np.array(w))
            ie, _ = P.select(matrix(LA[d], "exponential"), np.array(w))
            rows.append({"dataset": DS_TXT[d], "profile": name, **{f"C({e:g})": c for e, c in zip(CAND, C[0])},
                         "eps_linear": CAND[idx[0]], "eps_exponential": CAND[ie[0]]})
    prof = pd.DataFrame(rows)
    prof.round(3).to_csv(OUT / "tab_profiles.csv", index=False)
    want = {("Adult", "Privacy-focused"): (0.850, 1, 0.1), ("Adult", "Accuracy-focused"): (0.853, 5, 10),
            ("CDC-Diabetes", "Balanced"): (0.737, 1, 0.3), ("CDC-Diabetes", "Accuracy-focused"): (0.866, 5, 10)}
    for (d, p), (cmax, el, ee) in want.items():
        r = prof[(prof.dataset == d) & (prof.profile == p)].iloc[0]
        cs = [r[f"C({e:g})"] for e in CAND]
        if not (abs(max(cs) - cmax) < 5e-4 and r.eps_linear == el and r.eps_exponential == ee):
            fail(f"profile {d}/{p}")
    lines.append("PASS profile table: closeness coefficients and selected budgets as in the thesis")

    # ---- selection against the privacy share (switch points, figure selection_ch6)
    share = np.round(np.arange(0, 1.0001, 0.001), 3)
    Wsh = np.column_stack([share, 1 - share, np.zeros_like(share)])
    sel = {}
    srows = []
    for lp in ("linear", "exponential"):
        for d in DS:
            idx, _ = P.select(matrix(LA[d], lp), Wsh)
            sel[(lp, d)] = idx
            for e_i in sorted(set(idx[idx >= 0])):
                s = share[idx == e_i]
                srows.append({"lp_map": lp, "dataset": DS_TXT[d], "eps": CAND[e_i], "share_min": s.min(), "share_max": s.max()})
    pd.DataFrame(srows).to_csv(OUT / "tab_switch.csv", index=False)

    # ---- fixed defaults over the simplex grid (tab:ch6:defaults) and selection frequency (tab:ch6:freq)
    G = simplex_grid()
    if len(G) != 230:
        fail(f"simplex grid has {len(G)} vectors, expected 230")
    drows, frows = [], []
    for d in DS:
        for lp in ("linear", "exponential"):
            M = matrix(LA[d], lp)
            idx, C = P.select(M, G)
            rows_ = np.arange(len(G))
            frows.append({"dataset": DS_TXT[d], "lp_map": lp, **{f"eps={e:g}": int(np.sum(idx == k)) for k, e in enumerate(CAND)},
                          "undefined": int(np.sum(idx < 0))})
            if d not in ("adult", "cdc"):
                continue
            for k, e in enumerate(CAND):
                drows.append({"dataset": DS_TXT[d], "lp_map": lp, "eps_fixed": e,
                              "agreement": float(np.mean(idx == k)), "mean_loss": float(np.mean(C[rows_, idx] - C[:, k]))})
    dft = pd.DataFrame(drows)
    dft.round(4).to_csv(OUT / "tab_defaults.csv", index=False)
    freq = pd.DataFrame(frows)
    freq.to_csv(OUT / "tab_freq.csv", index=False)
    want_f = {("Adult", "linear"): [22, 13, 86, 89, 20, 0], ("CDC-Diabetes", "exponential"): [71, 60, 0, 0, 99, 0],
              ("CalHousing", "linear"): [88, 0, 0, 0, 142, 0], ("Appliances", "linear"): [210, 0, 0, 0, 0, 20]}
    for (d, lp), v in want_f.items():
        r = freq[(freq.dataset == d) & (freq.lp_map == lp)].iloc[0]
        if [int(r[f"eps={e:g}"]) for e in CAND] + [int(r.undefined)] != v:
            fail(f"selection frequency {d}/{lp}")
    lines.append("PASS selection frequency over the 230 simplex vectors as in the appendix table (spot-checked rows)")
    chk = dft[(dft.dataset == "Adult") & (dft.lp_map == "linear") & (dft.eps_fixed == 1.0)].iloc[0]
    chk2 = dft[(dft.dataset == "CDC-Diabetes") & (dft.lp_map == "linear") & (dft.eps_fixed == 1.0)].iloc[0]
    if not (abs(chk.agreement - 0.374) < 5e-4 and abs(chk.mean_loss - 0.134) < 5e-4 and abs(chk2.agreement - 0.348) < 5e-4
            and abs(chk2.mean_loss - 0.115) < 5e-4):
        fail(f"fixed default eps = 1: {chk.agreement:.4f}/{chk.mean_loss:.4f}, {chk2.agreement:.4f}/{chk2.mean_loss:.4f}")
    lines.append("PASS fixed default eps = 1 (linear): agreement 37.4% / 34.8%, mean loss 0.134 / 0.115, as in the thesis")

    # ---- fixed-ideal variant (tab:ch6:absolute)
    arows = []
    for d in DS:
        for lp in ("linear", "exponential"):
            M = matrix(LA[d], lp)
            a = [CAND[P.select(M, np.array(w))[0][0]] for w in PROFILES.values()]
            b = [CAND[fixed_ideal_select(M, np.array([w]))[0]] for w in PROFILES.values()]
            i1, _ = P.select(M, G)
            i2 = fixed_ideal_select(M, G)
            ok = (i1 >= 0) & (i2 >= 0)
            arows.append({"dataset": DS_TXT[d], "lp_map": lp, "standard": ", ".join(f"{x:g}" for x in a),
                          "fixed_ideals": ", ".join(f"{x:g}" for x in b), "same_on_grid": float(np.mean(i1[ok] == i2[ok])),
                          "defined": int(ok.sum())})
    pd.DataFrame(arows).round(4).to_csv(OUT / "tab_absolute.csv", index=False)

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
                         "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
    col = {"adult": "#2a78d6", "cdc": "#eb6834", "calhousing": "#1baf7a", "appliances": "#7c5cc4"}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7), sharey=True)
    for ax, lp, lab in zip(axes, ("linear", "exponential"), ("(a) linear LP map", "(b) exponential LP map")):
        for d in DS:
            ax.plot(P.LP[lp], LA[d], marker="o", ms=3.5, color=col[d], label=DS_TXT[d])
        for e, x, y in zip(CAND, P.LP[lp], LA["adult"]):
            ax.annotate(f"{e:g}", (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7)
        ax.set_xlabel("level of privacy (LP)")
        ax.set_title(lab, fontsize=8.5)
    axes[0].set_ylabel("level of accuracy (LA)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "tradeoff_ch6.pdf")
    fig.savefig(OUT / "tradeoff_ch6.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7), sharey=True)
    for ax, lp, lab in zip(axes, ("linear", "exponential"), ("(a) linear LP map", "(b) exponential LP map")):
        for k, d in enumerate(DS):
            idx = sel[(lp, d)]
            y = np.where(idx >= 0, np.arange(len(CAND))[np.maximum(idx, 0)], np.nan) + (k - 1.5) * 0.04
            ax.step(share, y, where="post", color=col[d], lw=1.3, label=DS_TXT[d])
        ax.set_xlabel(r"privacy share $w_{LP}/(w_{LP}+w_{LA})$")
        ax.set_title(lab, fontsize=8.5)
        ax.set_yticks(range(len(CAND)))
        ax.set_yticklabels([f"{e:g}" for e in CAND])
    axes[0].set_ylabel(r"selected $\varepsilon^*$")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "selection_ch6.pdf")
    fig.savefig(OUT / "selection_ch6.png", dpi=200)
    plt.close(fig)
    (OUT / "checks.log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {sorted(p.name for p in OUT.iterdir())}")


if __name__ == "__main__":
    main()
