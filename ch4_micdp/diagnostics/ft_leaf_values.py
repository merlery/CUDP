"""Diagnostic (exploratory; no effect on any result): for trial 0, every eps and both datasets, rebuild
Public-Pretrain + Uniform-FT (15%) and Public-Pretrain + MIC-DP-FT (15%) exactly as src/public.py does and
record the pretrained model's raw output on the privatized rows, the leaf values of the 100 continued
trees, and the range of the noise-aware raw prediction on the test set."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as D          # noqa: E402
from src import public as PB       # noqa: E402
from src.noiseaware import expected_raw  # noqa: E402

cfg = D.load_config()
rows = []
for ds in cfg["public_baselines"]["datasets"]:
    ctx = PB.build_context(ds, cfg)
    seed = cfg["seeds"]["trial_base"]
    pos = ctx["subs"]["p15"]
    for ei, eps in enumerate(cfg["public_baselines"]["eps_grid"]):
        for meth, rel in (("ft_uniform", "uniform"), ("ft_mic_dp", "mic_dp")):
            al, Xt, sigma, keep, _ = PB.release(ctx, rel, eps, ei, seed, cfg)
            Xp, yp = ctx["X_public"][pos][:, keep], ctx["y_public"][pos]
            m = PB.primary_model(cfg, seed, max_iter=100, warm=True).fit(Xp, yp)
            start = m.decision_function(Xt)
            p = 1 / (1 + np.exp(-start))
            hess = p * (1 - p)
            m.set_params(max_iter=200)
            m.fit(Xt, ctx["y_train"])
            leaf = np.array([np.abs(t[0].nodes["value"][t[0].nodes["is_leaf"].astype(bool)]).max() for t in m._predictors])
            na = expected_raw(m, ctx["X_test"][:, keep], sigma[keep])
            pr = 1 / (1 + np.exp(-na))
            rows.append(dict(dataset=ds, eps=eps, method=meth, start_abs_raw_mean=np.abs(start).mean(),
                             start_abs_raw_max=np.abs(start).max(), start_hessian_median=np.median(hess),
                             max_abs_leaf_pretrained=leaf[:100].max(), median_max_abs_leaf_continued=np.median(leaf[100:]),
                             max_abs_leaf_continued=leaf[100:].max(), test_raw_min=na.min(), test_raw_max=na.max(),
                             test_unique_probabilities=int(np.unique(pr).size)))
            print(rows[-1], flush=True)
out = D.ROOT / cfg["public_baselines"]["out_dir"] / "diagnostic_ft.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out)
