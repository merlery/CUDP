"""Loss-threshold membership inference (Yeom, Giacomelli, Fredrikson, Jha,
CSF 2018, Sec. 4.1) on the final global model.

Members are training records of clients that participated in at least one
round; non-members are held-out records (the test split).  The attacker's
score is the negative per-record cross-entropy loss of the model on the RAW
record (lower loss => "member").  We report the AUC of that score and the
attacker advantage max_t (TPR(t) - FPR(t)) over all thresholds, on a
balanced sample of n members and n non-members.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class MembershipResult:
    n_members: int
    n_nonmembers: int
    auc: float
    advantage: float               # max over thresholds of TPR - FPR
    threshold_loss: float          # loss threshold achieving the advantage
    member_loss_mean: float
    nonmember_loss_mean: float

    def as_dict(self) -> Dict:
        return asdict(self)


@torch.no_grad()
def per_record_loss(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    # loss in float64 from the float32 logits so that confident predictions do
    # not all collapse to exactly 0 (ties would mask the ranking)
    out = [F.cross_entropy(model(Xt[i:i + batch]).double(), yt[i:i + batch], reduction="none") for i in range(0, len(Xt), batch)]
    return torch.cat(out).numpy()


def loss_threshold_attack(model: torch.nn.Module, X_members: np.ndarray, y_members: np.ndarray,
                          X_non: np.ndarray, y_non: np.ndarray, seed: int, n: int | None = None) -> MembershipResult:
    rng = np.random.default_rng(seed)
    n = min(len(y_members), len(y_non)) if n is None else min(n, len(y_members), len(y_non))
    im = rng.choice(len(y_members), n, replace=False)
    inn = rng.choice(len(y_non), n, replace=False)
    lm = per_record_loss(model, X_members[im], y_members[im])
    ln = per_record_loss(model, X_non[inn], y_non[inn])
    score = -np.concatenate([lm, ln])
    truth = np.concatenate([np.ones(n), np.zeros(n)])
    finite = np.isfinite(score)
    if finite.sum() < 2 or len(np.unique(truth[finite])) < 2:
        return MembershipResult(n, n, float("nan"), float("nan"), float("nan"), float(np.nanmean(lm)), float(np.nanmean(ln)))
    score = np.where(finite, score, score[finite].min() - 1.0)   # non-finite loss => "non-member"
    auc = float(roc_auc_score(truth, score))
    fpr, tpr, thr = roc_curve(truth, score)
    k = int(np.argmax(tpr - fpr))
    return MembershipResult(n, n, auc, float(tpr[k] - fpr[k]), float(-thr[k]), float(np.mean(lm)), float(np.mean(ln)))
