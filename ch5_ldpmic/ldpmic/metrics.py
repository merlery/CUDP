"""Evaluation metrics (numpy / scikit-learn only)."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def classification_metrics(logits: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """accuracy, balanced accuracy and AUC (binary: positive-class score;
    multi-class: macro one-vs-rest).  AUC is NaN if a class is absent from y."""
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y).astype(int)
    pred = logits.argmax(axis=1)
    out = {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }
    n_classes = logits.shape[1]
    present = np.unique(y)
    try:
        if n_classes == 2:
            out["auc"] = float(roc_auc_score(y, softmax(logits)[:, 1])) if present.size == 2 else float("nan")
        else:
            out["auc"] = (float(roc_auc_score(y, softmax(logits), multi_class="ovr", average="macro"))
                          if present.size == n_classes else float("nan"))
    except ValueError:
        out["auc"] = float("nan")
    return out
