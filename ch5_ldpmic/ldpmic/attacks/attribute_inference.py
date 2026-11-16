"""Attribute inference on released records (experiment E6).

Threat model: the untrusted server sees every client's sanitized records.
It wants the value of a sensitive categorical attribute (e.g. ``sex``) of a
record from the OTHER sanitized coordinates of that record (the attribute's
own one-hot columns are removed from the attacker's input; their direct
signal-to-noise ratio Delta_j / sigma_j is reported separately).

Protocol (fixed, nothing is tuned on the evaluation clients):
  * clients are split in two halves by a seeded permutation;
  * the attacker (scikit-learn LogisticRegression, default regularisation,
    class_weight=None, max_iter=1000, lbfgs) is trained on the sanitized
    records of the first half with the TRUE attribute (from the raw data)
    as label;
  * it is evaluated on the sanitized records of the second half;
  * reported: accuracy, balanced accuracy, the majority-class accuracy of
    the evaluation records, and the same attacker trained/evaluated on the
    RAW records as a reference ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def attribute_columns(feature_names: Sequence[str], attribute: str) -> List[int]:
    """Indices of the one-hot columns ``attribute=*`` (or a single column named ``attribute``)."""
    idx = [j for j, n in enumerate(feature_names) if n == attribute or n.startswith(attribute + "=")]
    if not idx:
        raise ValueError(f"attribute {attribute!r} not found in feature names")
    return idx


def attribute_labels(X_raw: np.ndarray, cols: Sequence[int]) -> np.ndarray:
    """Label of each record: argmax over the attribute's one-hot columns;
    records with no active column (missing value) get label -1."""
    block = X_raw[:, list(cols)]
    lab = block.argmax(axis=1)
    lab[block.max(axis=1) <= 0] = -1
    return lab.astype(int)


@dataclass
class AttackResult:
    attribute: str
    n_train: int
    n_eval: int
    n_classes: int
    majority_accuracy: float
    accuracy: float
    balanced_accuracy: float
    raw_accuracy: float            # same attacker on raw records (reference ceiling)
    raw_balanced_accuracy: float
    snr: Dict[str, Optional[float]]  # Delta_j / sigma_j of the attribute's own columns (None = not released)
    released_features: Optional[List[int]] = None   # selected release: the feature indices in the sanitized columns
    attribute_released: bool = True                 # whether any of the attribute's own columns is released
    # two further channels (same attacker, same split): "direct" reads only the
    # attribute's own released (noisy) columns; "all" uses every released column
    direct_accuracy: Optional[float] = None
    direct_balanced_accuracy: Optional[float] = None
    all_accuracy: Optional[float] = None
    all_balanced_accuracy: Optional[float] = None

    def as_dict(self):
        return asdict(self)


def _fit_eval(Xtr, ytr, Xev, yev, seed: int) -> Tuple[float, float]:
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xev)
    return float((pred == yev).mean()), float(balanced_accuracy_score(yev, pred))


def attribute_inference(client_raw: Sequence[np.ndarray], client_san: Sequence[np.ndarray],
                        feature_names: Sequence[str], attribute: str, ranges: np.ndarray,
                        sigmas_per_client: Sequence[np.ndarray], seed: int,
                        transform=None, released: Optional[Sequence[int]] = None,
                        san_transform=None) -> AttackResult:
    """Run the E6 attacker for one attribute.

    ``client_raw[i]`` / ``client_san[i]`` are client i's raw and sanitized
    record matrices (same row order).  ``transform`` is the public affine
    scaler (applied to both raw and sanitized inputs so the attacker sees
    the same units the model sees).

    Selected release (ldpmic with k > 0): ``released`` lists the feature
    indices that the sanitized columns correspond to (``client_san[i]`` and
    ``sigmas_per_client[i]`` then have len(released) columns) and
    ``san_transform`` is the public transform restricted to those columns.
    The attacker uses the released columns other than the attribute's own;
    the raw ceiling uses the same feature set un-noised.  Attribute columns
    that are not released get snr None.
    """
    cols = attribute_columns(feature_names, attribute)
    d = len(feature_names)
    released = list(range(d)) if released is None else [int(j) for j in released]
    pos = {j: i for i, j in enumerate(released)}
    keep = [j for j in released if j not in cols]        # feature indices the attacker uses
    keep_san = [pos[j] for j in keep]                     # their columns in the sanitized matrices
    N = len(client_raw)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    train_c, eval_c = perm[: N // 2], perm[N // 2:]

    def stack(ids, mats):
        return np.concatenate([mats[i] for i in ids], axis=0)

    f = transform if transform is not None else (lambda Z: Z)
    g = san_transform if san_transform is not None else f
    y_tr = attribute_labels(stack(train_c, client_raw), cols)
    y_ev = attribute_labels(stack(eval_c, client_raw), cols)
    Xs_tr, Xs_ev = g(stack(train_c, client_san))[:, keep_san], g(stack(eval_c, client_san))[:, keep_san]
    Xr_tr, Xr_ev = f(stack(train_c, client_raw))[:, keep], f(stack(eval_c, client_raw))[:, keep]
    m_tr, m_ev = y_tr >= 0, y_ev >= 0          # drop records with a missing attribute value
    Xs_tr, Xr_tr, y_tr = Xs_tr[m_tr], Xr_tr[m_tr], y_tr[m_tr]
    Xs_ev, Xr_ev, y_ev = Xs_ev[m_ev], Xr_ev[m_ev], y_ev[m_ev]
    if len(np.unique(y_tr)) < 2:
        raise ValueError("attacker training labels have a single class")
    acc, bacc = _fit_eval(Xs_tr, y_tr, Xs_ev, y_ev, seed)
    racc, rbacc = _fit_eval(Xr_tr, y_tr, Xr_ev, y_ev, seed)
    # direct channel: the attribute's own released columns only; all channel: every released column
    own_san = [pos[j] for j in cols if j in pos]
    S_tr_full, S_ev_full = g(stack(train_c, client_san))[m_tr], g(stack(eval_c, client_san))[m_ev]
    dacc = dbacc = None
    if own_san:
        dacc, dbacc = _fit_eval(S_tr_full[:, own_san], y_tr, S_ev_full[:, own_san], y_ev, seed)
    aacc, abacc = _fit_eval(S_tr_full, y_tr, S_ev_full, y_ev, seed)
    majority = float(np.bincount(y_ev).max() / len(y_ev))
    deltas = ranges[:, 1] - ranges[:, 0]
    S = np.stack(sigmas_per_client)
    snr = {feature_names[j]: (float(np.mean(deltas[j] / S[:, pos[j]])) if j in pos else None) for j in cols}
    return AttackResult(attribute, int(len(y_tr)), int(len(y_ev)), int(len(np.unique(y_ev))), majority,
                        acc, bacc, racc, rbacc, snr,
                        released_features=None if len(released) == d else released,
                        attribute_released=any(j in pos for j in cols),
                        direct_accuracy=dacc, direct_balanced_accuracy=dbacc,
                        all_accuracy=aacc, all_balanced_accuracy=abacc)
