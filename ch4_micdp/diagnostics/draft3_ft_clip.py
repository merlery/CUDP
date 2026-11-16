"""Diagnostic (exploratory, trial 0) of the Part C noise-aware prediction clip(E[F_pre], -3, 3) + E[F_cont]:
range of the exact E[F_pre(x + eta)] on the test set, share of test records where the clip is active, and the
ROC-AUC of each component. Same code path and seeds as src/draft3 (nothing is changed).

    python -m diagnostics.draft3_ft_clip     # -> results/ch4/draft3/diagnostic_ft_clip.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import data as D
from src import draft3 as T
from src.noiseaware import expected_raw
from src.public import build_context, release


def main():
    cfg = D.load_config()
    c3 = cfg["draft3"]
    pc = c3["part_c"]
    rows = []
    seed = cfg["seeds"]["trial_base"]
    for ds in c3["datasets"]:
        ctx = build_context(ds, cfg)
        for ei, eps in enumerate(c3["eps_grid"]):
            for meth in pc["methods"]:
                al, Xt, sigma, keep, _ = release(ctx, cfg["public_baselines"]["methods"][meth]["release"], eps, ei, seed, cfg)
                sk = sigma[keep]
                for size in pc["sizes"]:
                    pos = ctx["subs"][size]
                    m = T.RepairedContinuedHGB(max_iter=100, early_stopping=False, random_state=seed, warm_start=True)
                    m.fit(ctx["X_public"][pos][:, keep], ctx["y_public"][pos])
                    m.start_clip = pc["start_clip"]
                    m.set_params(max_iter=200, l2_regularization=pc["l2_continued"])
                    m.fit(Xt, ctx["y_train"])
                    Xte, yte = ctx["X_test"][:, keep], ctx["y_test"]
                    ep = expected_raw(T._part(m, 0, 100, True), Xte, sk)
                    ec = expected_raw(T._part(m, 100, None, False), Xte, sk)
                    c = pc["start_clip"]
                    rows.append({"dataset": ds, "method": meth, "size": size, "eps": eps, "E_pre_min": ep.min(),
                                 "E_pre_max": ep.max(), "share_clip_active": float(np.mean(np.abs(ep) > c)),
                                 "auc_E_pre": roc_auc_score(yte, ep), "auc_E_cont": roc_auc_score(yte, ec),
                                 "auc_noise_aware": roc_auc_score(yte, np.clip(ep, -c, c) + ec),
                                 "auc_E_pre_plus_E_cont_unclipped": roc_auc_score(yte, ep + ec)})
    pd.DataFrame(rows).to_csv(D.ROOT / c3["out_dir"] / "diagnostic_ft_clip.csv", index=False)


if __name__ == "__main__":
    main()
