"""Feature-target dependence scores, each in [0, 1].

* MIC(X_j, y)  - MIC_e (Reshef et al. 2016) from our own implementation in src/mic.py
                 (no minepy / mictools code). alpha from the sample-size rule (0.55 for
                 2,500 <= n < 5,000), c = 5. Author's choice on 2026-10-01.
* |r_j|        - absolute Pearson correlation
* |pc_j|       - absolute partial correlation of X_j and y given all other features,
                 from the precision matrix P of [X, y]: pc_j = -P[j, y] / sqrt(P[j, j] P[y, y])
"""
from __future__ import annotations

import numpy as np

from . import mic as micmod


def subsample_rows(n: int, max_rows: int, seed: int) -> np.ndarray:
    """Fixed-seed subsample (sorted) of at most max_rows row indices."""
    if n <= max_rows:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_rows, replace=False))


def mic_scores(X: np.ndarray, y: np.ndarray, c: float = 5, alpha: float | None = None,
               est: str = "mic_e") -> np.ndarray:
    """MIC_e(X_j, y) for every column. alpha=None -> sample-size rule (0.55 at n = 3,000)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    a = micmod.alpha_for_n(len(y)) if alpha is None else alpha
    out = np.array([micmod.mic(X[:, j], y, alpha=a, c=c, est=est) for j in range(X.shape[1])])
    return np.clip(out, 0.0, 1.0)


def mic_alpha(n: int) -> float:
    return float(micmod.alpha_for_n(n))


def pearson_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc * yc[:, None]).sum(axis=0) / den
    r = np.where(np.isfinite(r), r, 0.0)
    return np.clip(np.abs(r), 0.0, 1.0)


def partialcorr_scores(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    Z = np.column_stack([X, y])
    C = np.cov(Z, rowvar=False)
    P = np.linalg.inv(C + ridge * np.eye(C.shape[0]))
    k = Z.shape[1] - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        pc = -P[:k, k] / np.sqrt(np.diag(P)[:k] * P[k, k])
    pc = np.where(np.isfinite(pc), pc, 0.0)
    return np.clip(np.abs(pc), 0.0, 1.0)


def all_scores(X: np.ndarray, y: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    m = cfg["scores"]["mic"]
    return {
        "mic": mic_scores(X, y, c=m["c"], alpha=m["alpha"], est=m["est"]),
        "pearson": pearson_scores(X, y),
        "partialcorr": partialcorr_scores(X, y, cfg["scores"]["partialcorr_ridge"]),
    }


# ----------------------------------------------------------------------------- v2 statistics

def v2_statistics(p, cfg: dict) -> dict:
    """Exact statistics cached once per dataset for the enhanced pipeline.

    fgmic         FG-MIC on the full private-train split (privatized per trial by DP methods)
    fgmic_public  FG-MIC on the public split (MIC-Public; no noise)
    fgmic_sens    per-score sensitivity 3 (1 + ln n) / (n ln 2), n = private-train size
    moments       first/second moments of [x, y01] on the private-train split
    moment_sens   L2 sensitivity of the moment vector
    Cut points and the target range come from the PUBLIC split only.
    """
    from . import fgmic as F
    fc = cfg["scores"]["fgmic"]
    n = len(p.y_train)
    x_cuts = [F.public_cuts(p.X_public[:, j], fc["n_x_bins"]) for j in range(p.d)]

    def fg(X, y):
        B = max(len(y) ** fc["alpha_B"], 4.0)
        rows = F.row_partitions(p.task, p.y_public, B, fc["l_max"])
        return F.fgmic_scores(X, y, x_cuts, rows, fc["alpha_B"])

    lo, hi = np.percentile(p.y_public, cfg["scores"]["moments"]["target_percentiles"])
    y01 = F.scale_target(p.y_train, p.task, lo, hi)
    return {"fgmic": fg(p.X_train, p.y_train), "fgmic_public": fg(p.X_public, p.y_public),
            "fgmic_sens": F.fgmic_sensitivity(n),
            "moments": F.moment_vector(p.X_train, y01), "moment_sens": F.moment_sensitivity(n, p.d),
            "y_range": (float(lo), float(hi)), "n": n}
