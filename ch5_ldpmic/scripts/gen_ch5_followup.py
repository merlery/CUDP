# Part of scripts/gen_ch5.py (executed in its namespace): the tables and the figure of Chapter 5 that come from the
# follow-up runs (results/ch5/draft3, settings fixed before the runs at commit a342959) and from Experiment 5
# (results/ch5/central).  Uses: M, pm, f3, final, cell, tables, MAJ, ROOT, S, OUT, eps_main, pd, np.

D3 = ROOT / "results" / "ch5" / "draft3"
CEN = ROOT / "results" / "ch5" / "central"
PB = pd.read_csv(D3 / "partB.csv")       # Experiment 1 rerun with rescaled inputs (one row per run)
PA = pd.read_csv(D3 / "partA.csv")       # selection with the local score release
PF = pd.read_csv(D3 / "partF.csv")       # group-wise sensitivity
RERUN_LABEL = {"uniform release": "Uniform release", "LDP-MIC": "\\LDPMIC{}", "oracle allocation": "Oracle allocation"}
DSETS = {"adult": ("adult_occ", "Adult (occupation)"), "bank": ("bank_occ", "Bank (job)"),
         "mnist": ("mnist_a0.5", "MNIST (Dirichlet 0.5)"), "fmnist": ("fmnist_a0.5", "Fashion-MNIST (Dirichlet 0.5)")}


def _need(df, what):
    if len(df) == 0:
        raise RuntimeError(f"missing: {what}")
    return df


def rerun_runs(prefix, method, e):
    return PB[(PB.dataset == prefix) & (PB.method == method) & ((PB.eps - e).abs() < 1e-9)].sort_values("seed")


def main_rows(keys):
    rows = []
    for ds in keys:
        prefix, name = DSETS[ds]
        df = final(ds)
        nodp = cell(df, f"{prefix}_nodp", None)
        rows.append((r"\midrule" + "\n" if rows else "") + r"\multicolumn{6}{@{}l}{" + f"{name}; non-private {pm(nodp.accuracy_mean, nodp.accuracy_std)}; "
                    f"majority {f3(MAJ[ds])}" + "}")
        for method, label in RERUN_LABEL.items():
            cells = []
            for e in eps_main:
                r = rerun_runs(prefix, method, e)
                cells.append("--" if len(r) == 0 else pm(r.accuracy.mean(), r.accuracy.std(ddof=1)))
            if all(c == "--" for c in cells):
                raise RuntimeError(f"no rerun runs for {prefix} {method}")
            rows.append(label + " & " + " & ".join(cells))
        cells = []
        for e in eps_main:
            r = cell(df, f"{prefix}_local_dpsgd", e)
            cells.append(pm(r.accuracy_mean, r.accuracy_std))
        rows.append("Local DP-SGD & " + " & ".join(cells))
    return rows


HDR = r"Method & " + " & ".join(f"$\\eps = {e:g}$" for e in eps_main)
table("@{}lccccc@{}", HDR, main_rows(("adult", "mnist", "fmnist")),
      r"Final test accuracy (mean $\pm$ standard deviation over seeds) at $\eps \le 10$ under the same record-level $(\eps, 10^{-5})$ "
      r"guarantee against the server. Tabular data: 50 clients, $q = 0.2$, $T = 200$, 5 seeds; images: 100 clients, $q = 0.3$, $T = 100$, "
      r"3 seeds. The one-shot releases are the runs with inputs rescaled by their known noise scale. A dash marks a setting that was not run.",
      "tab:ch5:main")
table("@{}lccccc@{}", HDR, main_rows(("bank",)),
      r"As Table~\ref{tab:ch5:main} for Bank (job), 50 clients, 5 seeds.", "tab:ch5:bank_main")
M("rerunRuns", str(len(PB))); M("rerunDiverged", str(int(PB.diverged.sum())))

# group-wise sensitivity (Adult, occupation)
rows = []
for arm, label in (("uniform, sqrt(d)", "Per column"), ("group-wise sensitivity", "Group-wise")):
    accs, aucs = [], []
    for e in (10, 50, 100, 200):
        r = _need(PF[(PF.arm == arm) & (PF.eps == e)], f"partF {arm} {e}")
        accs.append(pm(r.accuracy.mean(), r.accuracy.std(ddof=1))); aucs.append(f3(r.auc.mean()))
    rows.append(f"{label}, accuracy & " + " & ".join(accs))
    rows.append(f"{label}, AUC & " + " & ".join(aucs))
table("@{}lcccc@{}", r"Calibration & $\eps = 10$ & $\eps = 50$ & $\eps = 100$ & $\eps = 200$", rows[0:1] + rows[2:3] + rows[1:2] + rows[3:4],
      r"Uniform release of all features on Adult (occupation) with the noise calibrated to $\sqrt{d} = \sqrt{105}$ and to the group-wise "
      r"sensitivity $\sqrt{22}$: final test accuracy (mean $\pm$ standard deviation over 5 seeds) and AUC.", "tab:ch5:group")

# selection with the local and the secure-aggregation score release (Adult, occupation)
rows = []
for e in (100, 200):
    for k in (10, 20):
        for i, (meth, label) in enumerate((("private, local", "local"), ("private, secure aggregation", "secure aggregation"))):
            r = _need(PA[(PA.eps == e) & (PA.k == k) & (PA.method == meth)], f"partA {meth} {e} {k}")
            lead = f"{e} & {k}" if i == 0 else " & "
            rows.append(f"{lead} & {label} & {pm(r.accuracy.mean(), r.accuracy.std(ddof=1))} & ${r.diff_to_random.mean():+.3f}$ & "
                        f"{int(r.overlap.min())}--{int(r.overlap.max())}")
        loc = PA[(PA.eps == e) & (PA.k == k) & (PA.method == "private, local")]
        M(f"localNoise{'Hundred' if e == 100 else 'TwoHundred'}", f"{loc.noise_std_avg.iloc[0]:.3f}")
table("@{}rrlccc@{}", r"$\eps$ & $k$ & Score release & Accuracy & Difference to random & Overlap", rows,
      r"Private top-$k$ selection on Adult (occupation) with the local and with the secure-aggregation score release: final test accuracy "
      r"(mean $\pm$ standard deviation over 5 seeds), mean paired difference to the random $k$-subset of the same seed, and overlap of the "
      r"selected set with the exact top-$k$ set (range over seeds).", "tab:ch5:local")

# Experiment 5: central and federated training on the same release (Adult, occupation)
CR = pd.read_csv(CEN / "central_runs.csv")
rows = []
for release, title in (("all_features", "Adult (occupation), all features, uniform shares; majority " + f3(MAJ["adult"])),
                       ("private_sel_k20", "Adult (occupation), private selection, $k = 20$"),
                       ("oracle_sel_k20", "Adult (occupation), oracle selection, $k = 20$ (non-private scores)")):
    rows.append((r"\midrule" + "\n" if rows else "") + r"\multicolumn{6}{@{}l}{" + title + "}")
    for e in sorted(CR[CR.release == release].eps.unique()):
        c = CR[(CR.release == release) & ((CR.eps - e).abs() < 1e-9)].sort_values("seed")
        fed = c.federated_accuracy.to_numpy()
        if release == "all_features" and e <= 10:      # federated values at eps <= 10 from the rerun with rescaled inputs
            rr = rerun_runs("adult_occ", "uniform release", e)
            if len(rr) != len(c) or list(rr.seed) != list(c.seed):
                raise RuntimeError(f"rerun seeds do not match the central runs at eps {e}")
            fed = rr.accuracy.to_numpy()
        diff = c.accuracy.to_numpy() - fed
        rows.append(f"$\\eps = {e:g}$ & {pm(fed.mean(), fed.std(ddof=1))} & {pm(c.accuracy.mean(), c.accuracy.std(ddof=1))} & "
                    f"${diff.mean():+.3f}$ $[{diff.min():+.3f}, {diff.max():+.3f}]$ & {f3(c.balanced_accuracy.mean())} & "
                    f"{c.federated_bytes.iloc[0] / 1e6:.2f} / {c.central_upload_bytes.iloc[0] / 1e6:.2f}")
clean = CR[CR.release == "clean"].sort_values("seed")
if len(clean):
    d = clean.accuracy.to_numpy() - clean.federated_accuracy.to_numpy()
    rows.append(r"\midrule" + "\n" + r"\multicolumn{6}{@{}l}{Adult, non-private references}")
    rows.append(f"Clean records & {pm(clean.federated_accuracy.mean(), clean.federated_accuracy.std(ddof=1))} & "
                f"{pm(clean.accuracy.mean(), clean.accuracy.std(ddof=1))} & ${d.mean():+.3f}$ $[{d.min():+.3f}, {d.max():+.3f}]$ & -- & --")
table("@{}lccccc@{}", r"Release, $\eps$ & Federated & Central & Difference & Central bal.\ acc. & MB (fed.\ / central)", rows,
      r"Final test accuracy (mean $\pm$ standard deviation over 5 seeds) of federated training and of central training on the same "
      r"privatized records, the paired difference (central minus federated; mean and range over seeds), the balanced accuracy of the "
      r"central model, and the total bytes sent by the clients under each option ($\delta = 10^{-5}$). The federated values at "
      r"$\eps \le 10$ are from the rerun with rescaled inputs.", "tab:ch5:central")
M("centralRuns", str(len(CR)))

# Figure: privacy-utility curves of Experiment 1 (rerun with rescaled inputs), four panels
fg, axes = plt.subplots(1, 4, figsize=(12, 2.8))
style = {"uniform release": ("#1f77b4", "o", 7), "LDP-MIC": ("#d62728", "s", 5), "oracle allocation": ("#9467bd", "v", 3)}
for ax, (ds, tag) in zip(axes, (("adult", "(a)"), ("bank", "(b)"), ("mnist", "(c)"), ("fmnist", "(d)"))):
    prefix, name = DSETS[ds]
    df = final(ds)
    for method, label in RERUN_LABEL.items():
        es, ms, ss = [], [], []
        for e in eps_main:
            r = rerun_runs(prefix, method, e)
            if len(r):
                es.append(e); ms.append(r.accuracy.mean()); ss.append(r.accuracy.std(ddof=1))
        col, mk, size = style[method]
        ax.errorbar(es, ms, yerr=ss, color=col, marker=mk, ms=size, mfc="none", lw=1, capsize=2, label=label.replace("\\LDPMIC{}", "LDP-MIC"))
    rs_ = [cell(df, f"{prefix}_local_dpsgd", e) for e in eps_main]
    ax.errorbar(eps_main, [r.accuracy_mean for r in rs_], yerr=[r.accuracy_std for r in rs_], color="#2ca02c", marker="D", ms=4, lw=1,
                capsize=2, label="Local DP-SGD")
    nodp = cell(df, f"{prefix}_nodp", None)
    ax.axhline(nodp.accuracy_mean, color="#444444", ls="--", lw=1)
    ax.axhline(MAJ[ds], color="#999999", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xticks(eps_main); ax.set_xticklabels([f"{e:g}" for e in eps_main])
    ax.set_xlabel(r"privacy budget $\varepsilon$ ($\delta = 10^{-5}$)"); ax.set_title(f"{tag} {name}", fontsize=9)
axes[0].set_ylabel("test accuracy")
axes[0].legend(fontsize=7, frameon=False, loc="lower right")
fg.tight_layout()
fg.savefig(OUT / "figures" / "pu_ch5.pdf", bbox_inches="tight"); fg.savefig(OUT / "figures" / "pu_ch5.png", dpi=200, bbox_inches="tight")
plt.close(fg)
