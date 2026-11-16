"""Glue: from raw per-client data + a method configuration to FL clients.

This is where the three phases are wired together.  Experiment scripts call
``build_clients``; nothing dataset-specific lives here.

Methods
-------
nodp         PlainClient on raw data.
iso          Phase 3 with uniform shares and NO Phase 1 (eps2 = eps, delta2 = delta).
ldpmic       Phase 1 (one-shot score release) + Phase 2 (shares) + Phase 3.
dimsample    Phase 3 on k sampled coordinates, no Phase 1 (eps2 = eps).
local_dpsgd  DPSGDClient on raw data with a per-client RDP accountant.
fs_dpsgd     mechanism A: Phase 1 scores -> global top-k feature set -> local
             DP-SGD on the k inputs (eps2).
aniso_dpsgd  mechanism B: Phase 1 scores -> global shares p (Phase 2) ->
             local DP-SGD whose first-layer gradient is clipped and noised in
             the coordinates scaled by D_j = sqrt(d p_j) (eps2): per-feature
             noise std sigma C / D_j, i.e. the Phase 3 precision allocation
             moved from the records to the gradient.

Allocation modes for ``ldpmic`` (``--allocation``):
per_client   each client uses its own noisy scores.
global       the server averages the clients' noisy score vectors (weighted
             by m_i) and broadcasts ONE share vector; every client
             re-validates it (``allocation.validate_shares``) before use.
oracle       non-private scores (no Phase 1 noise).  A ceiling, never a
             private method: ``RunInfo.private`` is False and the realized
             budget is reported as (eps2, delta2) with a warning flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import allocation, perturb, scores
from .fl import Client, FLConfig, PlainClient, SanitizedClient, derive_seed
from .privacy import sigma0, split_budget

METHODS = ("nodp", "iso", "ldpmic", "dimsample", "local_dpsgd", "fs_dpsgd", "aniso_dpsgd")
INLINE_VECTOR_LIMIT = 20000   # N*d above which per-client vectors leave the JSON
ALLOCATIONS = ("per_client", "global", "oracle")
SCORES = ("fixed_grid", "mic", "pearson", "spearman")


@dataclass
class MethodConfig:
    method: str = "ldpmic"
    eps: float = 1.0
    delta: float = 1e-5
    eps1_frac: float = 0.2            # eps1 = eps1_frac * eps, eps2 = rest; delta split in half
    allocation: str = "per_client"    # per_client | global | oracle
    share_mode: str = "exp"           # uniform | exp | linear | random | pearson | spearman
    beta: float = 2.0
    floor: float = 0.05
    score: str = "fixed_grid"         # fixed_grid | mic | pearson | spearman
    G: int = 10
    score_sensitivity: str = "conservative"   # conservative | analytic
    sensitive: Tuple[int, ...] = ()
    p_max: Optional[float] = None
    dimsample_k: int = 4
    sigma0_method: str = "analytic"   # analytic | classical
    clip_grad: float = 1.0            # C_g for local_dpsgd
    dpsgd_conversion: str = "balle"   # RDP -> (eps, delta) conversion for local_dpsgd
    # local_dpsgd noise calibration: "worst_case" = every client calibrates to
    # participating in all T rounds (author decision, QUESTIONS.md Q8);
    # "schedule" = to its own planned participations from the seeded schedule
    # (the setting of the recorded first sweeps, DECISIONS.md D3).
    dpsgd_calibration: str = "worst_case"
    k: int = 0                        # fs_dpsgd / ldpmic: number of features kept (global top-k of the averaged scores); 0 = all
    # Phase 1 release: "local" = every client releases its own noisy vector
    # (local DP, l2 sensitivity sqrt(d) Delta_s); "secagg" = the server sees
    # only the secure-aggregation sum of the vectors, each client adds its share
    # of ONE Gaussian calibrated to a record's effect on the m-weighted average
    # (distributed DP; requires non-colluding clients for the masking step).
    score_release: str = "local"
    # ldpmic with k > 0: how S is chosen.  "top_k" = the k largest averaged
    # scores (private or oracle); "random" = a seeded random k-subset, no
    # scores and no Phase 1 release (control for the selection criterion;
    # keeps the same eps split so that Phase 3 gets the same eps2).
    selection: str = "top_k"
    # selection with k > 0: drop the `sensitive` columns from the candidate set
    # before the top-k, so that a sensitive attribute is never released
    # (exposure study, CLAUDE.md addendum 2026-11-07b)
    exclude_sensitive: bool = False
    # Record-perturbation methods: divide every released coordinate (after the
    # public affine scaling to [0,1]) by sqrt(1 + (sigma_j/Delta_j)^2 / s^2),
    # s = input_rescale = a PUBLIC scale of a [0,1] feature.  sigma_j is known
    # to the client and is data-independent given the released shares, so this
    # is post-processing of the release.  None = off (the recorded sweeps).
    input_rescale: Optional[float] = None
    # iso only: squared l2 sensitivity of a record in units of Delta_j
    # (None = d, every coordinate may move by its full range).  The group-wise
    # value n_numeric + 2 n_categorical assumes VALID one-hot records (a
    # replacement changes at most two columns of each one-hot group).
    l2_sens_sq: Optional[float] = None


@dataclass
class RunInfo:
    """Everything about the privacy side of a run that goes into provenance."""
    method: str
    private: bool
    eps_target: float
    delta_target: float
    eps1: float = 0.0
    delta1: float = 0.0
    eps2: float = 0.0
    delta2: float = 0.0
    eps_realized: float = float("nan")
    delta_realized: float = float("nan")
    sigma0: float = float("nan")
    sigma_s: float = float("nan")
    score_sensitivity_value: float = float("nan")
    shares: Dict[str, List[float]] = field(default_factory=dict)      # per client id (or "global")
    sigmas: Dict[str, List[float]] = field(default_factory=dict)
    sanitized_sha256: Dict[str, str] = field(default_factory=dict)
    dpsgd_noise_multiplier: Dict[str, float] = field(default_factory=dict)
    # Phase 1 diagnostics (ldpmic only): L1 distance between the score vector a
    # client actually used (private, clipped to [0,1]) and its oracle
    # (non-private) score vector, per client and averaged; for the global
    # allocation also the distance between the m-weighted averages that the
    # server would form from private vs. oracle scores.
    score_l1_per_client: List[float] = field(default_factory=list)
    score_l1_mean: float = float("nan")
    score_l1_global: float = float("nan")
    score_raw_global: List[float] = field(default_factory=list)      # m-weighted mean of oracle scores
    score_private_global: List[float] = field(default_factory=list)  # m-weighted mean of private scores
    # one-time costs (seconds, whole federation): Phase 1 = scores + release
    # + shares; Phase 3 = clipping + perturbation (+ cache I/O)
    phase1_time_s: float = 0.0
    phase3_time_s: float = 0.0
    phase1_bytes_per_client: int = 0   # score vector upload under the global allocation
    selected_features: List[int] = field(default_factory=list)   # fs_dpsgd: global top-k index set
    gradient_scale: List[float] = field(default_factory=list)    # aniso_dpsgd: D_j = sqrt(d p_j), public
    input_dim: int = 0                 # model input dimension (d, or k for fs_dpsgd)
    # input_rescale: the public divisor applied to the TEST features (computed from
    # the m-weighted mean over clients of the scaled noise variance per feature)
    input_rescale_test: List[float] = field(default_factory=list)
    input_rescale_range: List[float] = field(default_factory=list)   # [min, max] of the per-client divisors
    notes: List[str] = field(default_factory=list)

    def as_dict(self):
        d = asdict(self)
        d.pop("_shares_matrix", None)
        d.pop("_sigmas_matrix", None)
        return d

    def vectors(self) -> Dict[str, np.ndarray]:
        """Per-client share / sigma matrices (N x d) for compact storage."""
        out = {}
        if self._shares_matrix is not None:
            out["shares"] = self._shares_matrix
        if self._sigmas_matrix is not None:
            out["sigmas"] = self._sigmas_matrix
        return out

    _shares_matrix: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _sigmas_matrix: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


def _raw_scores(mcfg: MethodConfig, X: np.ndarray, y: np.ndarray, ranges: np.ndarray, n_classes: int) -> np.ndarray:
    if mcfg.score == "fixed_grid":
        return scores.fixed_grid_scores(X, y, ranges, mcfg.G, n_classes)
    if mcfg.score == "mic":
        try:
            return scores.mic_scores(X, y)
        except ImportError:
            return scores.fixed_grid_scores(X, y, ranges, mcfg.G, n_classes)
    if mcfg.score in ("pearson", "spearman"):
        return scores.correlation_scores(X, y, mcfg.score)
    raise ValueError(f"unknown score {mcfg.score!r}; choose from {SCORES}")


def _share_mode(mcfg: MethodConfig) -> str:
    # pearson/spearman ablations use the exp formula on |corr| scores
    if mcfg.score in ("pearson", "spearman") and mcfg.share_mode == "exp":
        return mcfg.score
    return mcfg.share_mode


def secagg_sum_sensitivity(sizes: Sequence[int], deltas_s: Sequence[float], d: int) -> float:
    """l2 sensitivity of the m-weighted average of the clients' score vectors
    to the replacement of ONE record: a record of client i moves that
    client's vector by at most Delta_s_i per coordinate, hence the average
    by m_i Delta_s_i / M per coordinate."""
    M = float(sum(sizes))
    return math.sqrt(d) * max(m * ds for m, ds in zip(sizes, deltas_s)) / M


def _phase1_scores(mcfg: MethodConfig, client_data, ranges, n_classes: int, seed: int, eps1: float, delta1: float,
                   s0, info: "RunInfo"):
    """Phase 1 for every client: raw dependence scores and their ONE private
    release (or the raw scores under the oracle allocation).  Records the
    L1 diagnostics in ``info``.  Returns (noisy, raws, priv_avg, raw_avg).

    score_release="secagg": each client adds Gaussian noise with variance
    sigma^2 / (N w_i^2) (w_i = m_i / M) to its vector, so that the m-weighted
    average carries exactly N(0, sigma^2 I) with sigma = sens * sigma0(eps1,
    delta1) and sens = secagg_sum_sensitivity(...): the aggregate is
    (eps1, delta1)-DP for every record.  Individual vectors are never seen by
    the server (secure aggregation, simulated here), so ``noisy[i]`` are the
    clients' masked contributions and only ``priv_avg`` is a release."""
    noisy: List[np.ndarray] = []
    raws: List[np.ndarray] = []
    if mcfg.score_release not in ("local", "secagg"):
        raise ValueError(f"unknown score_release {mcfg.score_release!r}")
    sizes = [X.shape[0] for X, _ in client_data]
    if mcfg.score_release == "secagg" and mcfg.allocation != "oracle":
        d = ranges.shape[0]
        sens_mode = mcfg.score_sensitivity if mcfg.score == "fixed_grid" else "conservative"
        deltas_s = [scores.score_sensitivity(mcfg.G, m, n_classes, sens_mode) for m in sizes]
        sens = secagg_sum_sensitivity(sizes, deltas_s, d)
        sigma_agg = sens * s0(eps1, delta1)
        N = len(sizes)
        M = float(sum(sizes))
        info.sigma_s = sigma_agg
        info.score_sensitivity_value = sens
        info.notes.append(f"secagg score release: aggregate noise std {sigma_agg:.4g} (sum sensitivity {sens:.4g}), N={N}")
    for i, (X, y) in enumerate(client_data):
        raw = _raw_scores(mcfg, X, y, ranges, n_classes)
        raws.append(raw)
        if mcfg.allocation == "oracle":
            noisy.append(raw)
        elif mcfg.score_release == "secagg":
            w_i = sizes[i] / M
            tau_i = sigma_agg / (math.sqrt(N) * w_i)
            rng = np.random.default_rng(derive_seed(seed, "scores", i))
            noisy.append(raw + tau_i * rng.standard_normal(raw.shape[0]))   # masked contribution, not clipped
        else:
            # correlation ablations have no histogram sensitivity argument
            sens_mode = mcfg.score_sensitivity if mcfg.score == "fixed_grid" else "conservative"
            Ds = scores.score_sensitivity(mcfg.G, X.shape[0], n_classes, sens_mode)
            rng = np.random.default_rng(derive_seed(seed, "scores", i))
            s_priv, sigma_s = scores.privatize_scores(raw, eps1, delta1, Ds, rng, sigma0_fn=s0)
            info.sigma_s = max(info.sigma_s, sigma_s) if not math.isnan(info.sigma_s) else sigma_s
            info.score_sensitivity_value = Ds
            noisy.append(s_priv)
    weights = np.array([X.shape[0] for X, _ in client_data], dtype=float)
    info.score_l1_per_client = [float(np.abs(a - b).sum()) for a, b in zip(noisy, raws)]
    info.score_l1_mean = float(np.mean(info.score_l1_per_client))
    raw_avg = np.average(np.stack(raws), axis=0, weights=weights)
    priv_avg = np.average(np.stack(noisy), axis=0, weights=weights)
    if mcfg.score_release == "secagg" and mcfg.allocation != "oracle":
        priv_avg = np.clip(priv_avg, 0.0, 1.0)     # post-processing of the aggregate
        info.notes.append("secagg: per-client L1 diagnostics refer to masked contributions, not to releases")
    info.score_l1_global = float(np.abs(priv_avg - raw_avg).sum())
    info.score_raw_global = raw_avg.tolist()
    info.score_private_global = priv_avg.tolist()
    return noisy, raws, priv_avg, raw_avg


def select_top_k(avg_scores: np.ndarray, k: int) -> List[int]:
    """Global top-k feature set from the (m-weighted) averaged private scores;
    ties broken by feature index.  Post-processing of the released scores."""
    d = len(avg_scores)
    if not (1 <= k <= d):
        raise ValueError(f"k must be in [1, d]; got k={k}, d={d}")
    order = np.lexsort((np.arange(d), -np.asarray(avg_scores, dtype=float)))
    return sorted(int(j) for j in order[:k])


def validate_feature_set(sel: Sequence[int], k: int, d: int) -> List[int]:
    """Client-side check of a server-broadcast feature set (untrusted server):
    exactly k distinct in-range indices."""
    sel = [int(j) for j in sel]
    if len(sel) != k or len(set(sel)) != k or any(j < 0 or j >= d for j in sel):
        raise ValueError("invalid broadcast feature set")
    return sorted(sel)


def gradient_scale(p: np.ndarray) -> np.ndarray:
    """Mechanism B: D_j = sqrt(d p_j) from validated shares p (sum p = 1), so
    that sum_j D_j^2 = d and uniform shares give D = 1 (plain DP-SGD).  Noise
    std on first-layer column j is sigma C / D_j: the precision share p_j buys
    precision proportional to p_j, as in Phase 3."""
    p = allocation.validate_shares(np.asarray(p, dtype=np.float64).copy())
    D = np.sqrt(p.shape[0] * p)
    if not np.all(np.isfinite(D)) or np.any(D <= 0):
        raise ValueError("gradient scale must be finite and positive")
    return D


def build_clients(mcfg: MethodConfig, client_data: Sequence[Tuple[np.ndarray, np.ndarray]],
                  ranges: np.ndarray, n_classes: int, seed: int, fl_cfg: FLConfig,
                  schedule: Optional[List[np.ndarray]] = None,
                  cache_dir: Optional[Path] = None, transform=None) -> Tuple[List[Client], RunInfo]:
    """Instantiate the clients of one run.  ``schedule`` (pre-drawn
    participation, see ``fl.participation_schedule``) is needed by
    ``local_dpsgd`` to calibrate its noise multiplier.  ``transform`` is an
    optional PUBLIC map (e.g. ``data.common.public_scaler``) applied to the
    (sanitized or raw) features before training; the test set must get the
    same map."""
    if mcfg.method not in METHODS:
        raise ValueError(f"unknown method {mcfg.method!r}; choose from {METHODS}")
    ranges = np.asarray(ranges, dtype=float)
    d = ranges_full_d = ranges.shape[0]
    deltas = ranges[:, 1] - ranges[:, 0]
    N = len(client_data)
    s0 = lambda e, dl: sigma0(e, dl, mcfg.sigma0_method)  # noqa: E731
    info = RunInfo(method=mcfg.method, private=True, eps_target=mcfg.eps, delta_target=mcfg.delta)

    if mcfg.method == "nodp":
        info.private = False
        info.eps_realized, info.delta_realized = math.inf, 0.0
        clients = [PlainClient(i, X, y, seed, transform) for i, (X, y) in enumerate(client_data)]
        info.input_dim = d
        return clients, info

    if mcfg.method == "aniso_dpsgd":
        # Mechanism B: Phase 1 (one-shot private scores, eps1) -> global shares p
        # (Phase 2, validated by every client) -> local DP-SGD with the
        # first-layer gradient clipped and noised in coordinates scaled by
        # D_j = sqrt(d p_j) (eps2).  D is public post-processing of the score
        # release, so the DP-SGD accountant is unchanged.  Per record:
        # (eps1 + eps2, delta1 + delta2)-DP.
        if schedule is None:
            raise ValueError("aniso_dpsgd needs the participation schedule")
        (eps1, delta1), (eps2, delta2) = split_budget(mcfg.eps, mcfg.delta, mcfg.eps1_frac)
        info.eps1, info.delta1, info.eps2, info.delta2 = eps1, delta1, eps2, delta2
        if mcfg.allocation == "oracle":
            info.private = False
            info.notes.append("oracle gradient allocation: non-private scores; NOT a private method")
        elif mcfg.allocation != "global":
            raise ValueError("aniso_dpsgd supports allocation 'global' (private scores) or 'oracle'")
        if eps1 == 0 and mcfg.allocation != "oracle":
            raise ValueError("eps1_frac = 0 (public scores) is only valid with allocation 'oracle'")
        if mcfg.allocation == "global" and mcfg.score_release == "secagg":
            info.notes.append("threat model: Phase 1 via secure aggregation (server sees the noisy sum only; "
                              "non-colluding clients assumed for the masking); DP-SGD purely local")
        noisy, raws, priv_avg, raw_avg = _phase1_scores(mcfg, client_data, ranges, n_classes, seed, eps1, delta1, s0, info)
        rng = np.random.default_rng(derive_seed(seed, "shares", "global"))
        p_global = allocation.shares(priv_avg, _share_mode(mcfg), mcfg.beta, mcfg.floor, deltas,
                                     mcfg.sensitive, mcfg.p_max, rng)
        p_global = allocation.validate_shares(p_global.copy())      # every client validates the broadcast
        D = gradient_scale(p_global)
        info.shares["global"] = p_global.tolist()
        info.gradient_scale = D.tolist()
        info.phase1_bytes_per_client = d * 8
        info.input_dim = d
        info.notes.append(f"gradient scale D: min={D.min():.4g} max={D.max():.4g} (uniform shares give D=1)")
        dp_eps, dp_delta = eps2, delta2
        input_scale = D
    else:
        input_scale = None
    if mcfg.method == "fs_dpsgd":
        # Mechanism A: Phase 1 (one-shot private scores, eps1) -> global top-k feature
        # set (post-processing; clients validate the broadcast) -> local DP-SGD on the
        # k selected inputs (eps2).  Per record: (eps1 + eps2, delta1 + delta2)-DP.
        if schedule is None:
            raise ValueError("fs_dpsgd needs the participation schedule")
        (eps1, delta1), (eps2, delta2) = split_budget(mcfg.eps, mcfg.delta, mcfg.eps1_frac)
        info.eps1, info.delta1, info.eps2, info.delta2 = eps1, delta1, eps2, delta2
        if mcfg.allocation == "oracle":
            info.private = False
            info.notes.append("oracle feature selection: non-private scores; NOT a private method")
        elif mcfg.allocation != "global":
            raise ValueError("fs_dpsgd supports allocation 'global' (private scores) or 'oracle'")
        if eps1 == 0 and mcfg.allocation != "oracle":
            raise ValueError("eps1_frac = 0 (public scores) is only valid with allocation 'oracle'")
        if mcfg.allocation == "global" and mcfg.score_release == "secagg":
            info.notes.append("threat model: Phase 1 via secure aggregation (server sees the noisy sum only; "
                              "non-colluding clients assumed for the masking); Phase 3 purely local")
        noisy, raws, priv_avg, raw_avg = _phase1_scores(mcfg, client_data, ranges, n_classes, seed, eps1, delta1, s0, info)
        if mcfg.selection == "top_k":
            sel_broadcast = select_top_k(priv_avg, mcfg.k)
            info.phase1_bytes_per_client = d * 8
        elif mcfg.selection in ("random", "variance"):
            # Controls for the selection criterion (pilot v8, CLAUDE.md addendum
            # 2026-10-29c): no score release, eps1 is not spent, DP-SGD keeps the
            # same eps2 so that only the mask differs.  "random": seeded random
            # k-subset.  "variance": top-k by the feature variance of the pooled
            # training records -- a public-proxy mask, simulated here on the
            # training data and therefore flagged NON-PRIVATE.
            if mcfg.selection == "random":
                rng_sel = np.random.default_rng(derive_seed(seed, "random_selection"))
                sel_broadcast = sorted(int(j) for j in rng_sel.choice(d, mcfg.k, replace=False))
                info.notes.append("random k-subset selection (control): data-independent, eps1 not spent")
            else:
                var = np.concatenate([np.asarray(X, dtype=np.float64) for X, _ in client_data], axis=0).var(axis=0)
                sel_broadcast = select_top_k(var, mcfg.k)
                info.private = False
                info.notes.append("variance mask (control): top-k by pooled feature variance, computed on the training "
                                  "records as a stand-in for a public proxy; NOT a private method; eps1 not spent")
            info.eps1, info.delta1 = 0.0, 0.0
        else:
            raise ValueError(f"unknown selection {mcfg.selection!r}")
        sel = validate_feature_set(sel_broadcast, mcfg.k, d)       # every client validates the broadcast
        info.selected_features = sel
        info.input_dim = len(sel)
        oracle_sel = set(select_top_k(raw_avg, mcfg.k))
        info.notes.append(f"top-k overlap with oracle selection: {len(oracle_sel & set(sel))}/{mcfg.k}")
        base_transform = transform
        cols = np.array(sel)

        def transform(X, _f=base_transform, _c=cols):   # noqa: E306
            Z = _f(X) if _f is not None else np.asarray(X, dtype=np.float64)
            return Z[:, _c]

        dp_eps, dp_delta = eps2, delta2
    elif mcfg.method != "aniso_dpsgd":
        dp_eps, dp_delta = mcfg.eps, mcfg.delta

    if mcfg.method in ("local_dpsgd", "fs_dpsgd", "aniso_dpsgd"):
        if schedule is None:
            raise ValueError("local_dpsgd needs the participation schedule")
        counts = np.zeros(N, dtype=int)
        for part in schedule:
            counts[part] += 1
        if mcfg.dpsgd_calibration == "worst_case":
            counts[:] = len(schedule)          # every client budgets for all T rounds
        elif mcfg.dpsgd_calibration != "schedule":
            raise ValueError(f"unknown dpsgd_calibration {mcfg.dpsgd_calibration!r}")
        from .dpsgd import DPSGDClient, DPSGDClientSpec

        clients: List[Client] = []
        for i, (X, y) in enumerate(client_data):
            spec = DPSGDClientSpec(target_eps=dp_eps, target_delta=dp_delta, clip_grad=mcfg.clip_grad,
                                   planned_participations=int(counts[i]), local_epochs=fl_cfg.local_epochs,
                                   batch_size=fl_cfg.batch_size, conversion=mcfg.dpsgd_conversion,
                                   input_scale=input_scale)
            c = DPSGDClient(i, X, y, seed, spec, transform=transform)
            info.dpsgd_noise_multiplier[str(i)] = c.noise_multiplier
            clients.append(c)
        info.eps_realized, info.delta_realized = float("nan"), mcfg.delta  # filled at end of run
        info.notes.append("eps_realized for the DP-SGD methods is eps1 + the max over clients of the DP-SGD accountant at end of run")
        if not info.input_dim:
            info.input_dim = d
        info.notes.append(f"dpsgd_calibration={mcfg.dpsgd_calibration}")
        return clients, info

    # --- Phase 3 budgets -------------------------------------------------
    if mcfg.method in ("iso", "dimsample"):
        eps1, delta1, eps2, delta2 = 0.0, 0.0, mcfg.eps, mcfg.delta
    else:  # ldpmic
        (eps1, delta1), (eps2, delta2) = split_budget(mcfg.eps, mcfg.delta, mcfg.eps1_frac)
    info.eps1, info.delta1, info.eps2, info.delta2 = eps1, delta1, eps2, delta2
    info.sigma0 = s0(eps2, delta2)

    # --- Phase 1 + 2: shares ------------------------------------------------
    import time as _time
    t_p1 = _time.perf_counter()
    share_vectors: List[np.ndarray] = []
    if mcfg.method == "iso":
        share_vectors = [allocation.shares(np.zeros(d), "uniform")] * N
    elif mcfg.method == "dimsample":
        # k coordinates sampled independently PER RECORD inside Phase 3 (perturb.perturb_once_sampled);
        # the recorded share vector is the expected share 1/d and the recorded sigma the std on a
        # sampled coordinate, Delta_j s0 sqrt(k).
        if not (1 <= mcfg.dimsample_k <= d):
            raise ValueError("dimsample_k must be in [1, d]")
        share_vectors = [allocation.shares(np.zeros(d), "uniform")] * N
        info.notes.append(f"dimsample: k={mcfg.dimsample_k} coordinates sampled per record; others set to the public midpoint")
    else:  # ldpmic
        if mcfg.allocation not in ALLOCATIONS:
            raise ValueError(f"unknown allocation {mcfg.allocation!r}")
        noisy, raws, priv_avg, raw_avg = _phase1_scores(mcfg, client_data, ranges, n_classes, seed, eps1, delta1, s0, info)
        mode = _share_mode(mcfg)
        if mcfg.allocation == "oracle":
            info.private = False
            info.notes.append("oracle allocation: non-private scores; NOT a private method")
        if mcfg.k > 0:
            # Selection + perturbation (pilot v5, CLAUDE.md addendum): the
            # server broadcasts the top-k set S of the averaged scores (every
            # client validates it), the shares are computed on S only and
            # Phase 3 releases the k selected coordinates of every record;
            # the other coordinates are never released.  The precision
            # invariant holds on S, so the k-vector release is (eps2, delta2)-DP.
            if mcfg.allocation not in ("global", "oracle"):
                raise ValueError("ldpmic with k > 0 needs allocation 'global' or 'oracle'")
            if mcfg.selection == "random":
                if mode != "uniform":
                    raise ValueError("random selection has no scores: share_mode must be 'uniform'")
                rng_sel = np.random.default_rng(derive_seed(seed, "random_selection"))
                sel = validate_feature_set(sorted(int(j) for j in rng_sel.choice(d, mcfg.k, replace=False)), mcfg.k, d)
                info.notes.append("random k-subset selection (control): data-independent, eps1 not spent; "
                                  "realized budget is the Phase 3 release only")
                info.eps1, info.delta1 = 0.0, 0.0
            elif mcfg.selection == "top_k":
                cand = priv_avg.copy()
                if mcfg.exclude_sensitive and mcfg.sensitive:
                    cand[list(mcfg.sensitive)] = -np.inf      # never selected
                    info.notes.append(f"sensitive columns excluded from selection: {sorted(mcfg.sensitive)}")
                sel = validate_feature_set(select_top_k(cand, mcfg.k), mcfg.k, d)
            else:
                raise ValueError(f"unknown selection {mcfg.selection!r}")
            oracle_sel = set(select_top_k(raw_avg, mcfg.k))
            info.notes.append(f"top-k overlap with oracle selection: {len(oracle_sel & set(sel))}/{mcfg.k}")
            cols = np.array(sel)
            # the cap applies to the sensitive columns that are in S (none under exclusion);
            # without a cap (p_max None) the sensitive list is only used for exclusion
            sens_sel = tuple(int(np.where(cols == j)[0][0]) for j in mcfg.sensitive if j in sel) if mcfg.p_max is not None else ()
            rng = np.random.default_rng(derive_seed(seed, "shares", "global"))
            p_sel = allocation.shares(priv_avg[cols], mode, mcfg.beta, mcfg.floor, deltas[cols], sens_sel, mcfg.p_max, rng)
            share_vectors = [allocation.validate_shares(p_sel.copy()) for _ in range(N)]
            info.shares["global"] = p_sel.tolist()
            info.phase1_bytes_per_client = d * 8
            info.selected_features = sel
            # restrict the data, the ranges and the (coordinate-wise) public transform to S
            client_data = [(np.asarray(X, dtype=np.float64)[:, cols], y) for X, y in client_data]
            ranges = ranges[cols]; deltas = deltas[cols]; d = mcfg.k
            if transform is not None:
                full_transform, full_d = transform, ranges_full_d

                def transform(Z, _f=full_transform, _c=cols, _d=full_d):   # noqa: E306
                    F = np.zeros((np.asarray(Z).shape[0], _d)); F[:, _c] = Z
                    return _f(F)[:, _c]
        elif mcfg.allocation == "global":
            avg = priv_avg
            rng = np.random.default_rng(derive_seed(seed, "shares", "global"))
            p_global = allocation.shares(avg, mode, mcfg.beta, mcfg.floor, deltas, mcfg.sensitive, mcfg.p_max, rng)
            # untrusted server: each client validates the broadcast before use
            share_vectors = [allocation.validate_shares(p_global.copy()) for _ in range(N)]
            info.shares["global"] = p_global.tolist()
            info.phase1_bytes_per_client = d * 8
        else:
            for i, s in enumerate(noisy):
                rng = np.random.default_rng(derive_seed(seed, "shares", i))
                share_vectors.append(allocation.shares(s, mode, mcfg.beta, mcfg.floor, deltas,
                                                       mcfg.sensitive, mcfg.p_max, rng))

    # --- Phase 3: one-shot perturbation --------------------------------------
    # Per-feature p_j and sigma_j are always recorded per client.  They go
    # inline into the provenance JSON when small; for large N*d (images) the
    # JSON keeps only the global vector (if any) and the full N x d matrices
    # are written next to it by run_sweep.py (``RunInfo.vectors``).
    inline = N * d <= INLINE_VECTOR_LIMIT
    info.phase1_time_s = _time.perf_counter() - t_p1
    t_p3 = _time.perf_counter()
    clients = []
    P = np.empty((N, d)); S = np.empty((N, d))
    rescale_var = np.zeros(d); rescale_lo, rescale_hi = math.inf, -math.inf
    if mcfg.input_rescale is not None and mcfg.method == "dimsample":
        raise ValueError("input_rescale is not defined for dimsample")
    for i, (X, y) in enumerate(client_data):
        p = share_vectors[i]
        if mcfg.method == "dimsample":
            sig = deltas * s0(eps2, delta2) * np.sqrt(mcfg.dimsample_k)
            c = SanitizedClient(i, X, y, ranges, sig, seed, cache_dir, transform,
                                sample_k=mcfg.dimsample_k, s0=s0(eps2, delta2))
        else:
            sig = perturb.sigmas(p, deltas, eps2, delta2, sigma0_fn=s0)
            if mcfg.l2_sens_sq is not None:
                if mcfg.method != "iso":
                    raise ValueError("l2_sens_sq is defined for the uniform release (iso) only")
                if not (0 < mcfg.l2_sens_sq <= d):
                    raise ValueError("l2_sens_sq must be in (0, d]")
                sig = deltas * s0(eps2, delta2) * math.sqrt(mcfg.l2_sens_sq)
            tr_i = transform
            if mcfg.input_rescale is not None:
                if transform is None:
                    raise ValueError("input_rescale needs the public [0,1] scaler as transform")
                f_i = np.sqrt(1.0 + (sig / deltas) ** 2 / float(mcfg.input_rescale) ** 2)
                rescale_var += X.shape[0] * (sig / deltas) ** 2
                rescale_lo, rescale_hi = min(rescale_lo, float(f_i.min())), max(rescale_hi, float(f_i.max()))

                def tr_i(Z, _f=transform, _d=f_i):   # noqa: E306
                    return _f(Z) / _d
            c = SanitizedClient(i, X, y, ranges, sig, seed, cache_dir, tr_i)
        clients.append(c)
        P[i], S[i] = p, sig
        if inline:
            if mcfg.allocation != "global" or mcfg.method != "ldpmic":
                info.shares[str(i)] = p.tolist()
            info.sigmas[str(i)] = sig.tolist()
        info.sanitized_sha256[str(i)] = c.sanitized_hash
    info.phase3_time_s = _time.perf_counter() - t_p3
    if mcfg.input_rescale is not None:
        M_tot = float(sum(X.shape[0] for X, _ in client_data))
        info.input_rescale_test = np.sqrt(1.0 + (rescale_var / M_tot) / float(mcfg.input_rescale) ** 2).tolist()
        info.input_rescale_range = [rescale_lo, rescale_hi]
        info.notes.append(f"input_rescale: released coordinates divided by sqrt(1 + (sigma_j/Delta_j)^2 / {mcfg.input_rescale}^2) "
                          "(post-processing); the test features by the same map with the m-weighted mean noise variance")
    if mcfg.l2_sens_sq is not None:
        info.notes.append(f"uniform release calibrated to l2 sensitivity sqrt({mcfg.l2_sens_sq:g}) (in units of Delta_j) "
                          f"instead of sqrt({d}); valid for one-hot-valid records only")
    if not inline:
        info._shares_matrix, info._sigmas_matrix = P, S
        info.notes.append("per-client shares/sigmas stored in <run_id>_vectors.npz (N*d too large for JSON)")
    info.input_dim = d
    if info.private:
        info.eps_realized, info.delta_realized = info.eps1 + info.eps2, info.delta1 + info.delta2
    else:
        info.eps_realized, info.delta_realized = eps2, delta2
    return clients, info
