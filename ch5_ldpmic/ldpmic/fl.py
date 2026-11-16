"""FedAvg loop and the client implementations.

Clients
-------
PlainClient      trains on its raw data (``nodp``).
SanitizedClient  trains on the one-shot sanitized copy of its data
                 (``iso``, ``ldpmic``, ``dimsample``); the sanitized array is
                 produced by ``perturb.perturb_once`` exactly once, cached, and
                 its SHA256 is re-checked after every round.
DPSGDClient      runs DP-SGD on its raw data every round with a persistent
                 per-client RDP accountant (``local_dpsgd``); see dpsgd.py.

Update clipping (``clip_update``, the "C" of the spec) bounds the l2 norm of
the client's model delta before it is sent.  It is a utility/stability
choice only: NOT part of the privacy accounting of any method here (for the
sanitized clients the privacy guarantee is already final before training
starts; for DP-SGD it is given by the per-sample clipping inside dpsgd.py).

Determinism: all randomness derives from ``FLConfig.seed`` through
``derive_seed`` (SHA256 of a tag), so a run is a pure function of
(config, seed).
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import perturb
from .metrics import classification_metrics

StateDict = Dict[str, torch.Tensor]


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #

def derive_seed(seed: int, *tags) -> int:
    """Deterministic 32-bit sub-seed from a master seed and string/int tags."""
    h = hashlib.sha256(("|".join([str(seed)] + [str(t) for t in tags])).encode()).digest()
    return int.from_bytes(h[:4], "little")


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:  # pragma: no cover - older torch
        pass


def torch_gen(seed: int, *tags) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(derive_seed(seed, *tags))
    return g


# --------------------------------------------------------------------------- #
# Parameter utilities
# --------------------------------------------------------------------------- #

def get_state(model: nn.Module) -> StateDict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def set_state(model: nn.Module, state: StateDict) -> None:
    model.load_state_dict({k: v.clone() for k, v in state.items()})


def state_delta(new: StateDict, old: StateDict) -> StateDict:
    return {k: new[k] - old[k] for k in old}


def state_norm(delta: StateDict) -> float:
    return math.sqrt(sum(float((v.double() ** 2).sum()) for v in delta.values()))


def clip_state(delta: StateDict, C: Optional[float]) -> Tuple[StateDict, float]:
    """Scale ``delta`` so that its global l2 norm is at most C.  Returns
    (clipped delta, original norm)."""
    nrm = state_norm(delta)
    if C is None or nrm <= C:
        return delta, nrm
    f = C / nrm
    return {k: v * f for k, v in delta.items()}, nrm


def state_bytes(state: StateDict) -> int:
    return sum(v.numel() * v.element_size() for v in state.values())


# --------------------------------------------------------------------------- #
# Local SGD
# --------------------------------------------------------------------------- #

def local_sgd(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int, batch_size: int,
              lr: float, gen: torch.Generator, momentum: float = 0.0) -> float:
    """Plain minibatch SGD on (X, y) in place.  Returns the mean training loss
    over all steps."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    m = X.shape[0]
    model.train()
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(m, generator=gen)
        for start in range(0, m, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else float("nan")


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #

@dataclass
class LocalResult:
    delta: StateDict
    n_samples: int
    train_loss: float
    update_norm: float
    bytes_sent: int
    time_s: float = float("nan")   # client-side wall-clock of this local update (set by the server loop)


class Client:
    """Interface.  ``local_update`` returns ``None`` if the client declines to
    participate (e.g. its DP-SGD budget is exhausted)."""

    cid: int
    n_samples: int

    def local_update(self, template: nn.Module, global_state: StateDict, round_idx: int,
                     cfg: "FLConfig") -> Optional[LocalResult]:
        raise NotImplementedError

    def training_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def after_round(self, round_idx: int) -> None:
        """Hook called by the server after aggregation (used for integrity checks)."""

    def run_summary(self) -> Dict[str, object]:
        """Per-client fields recorded in the run's provenance/summary."""
        return {}


class _SGDClient(Client):
    """Shared implementation of a client that trains with plain SGD on
    whatever ``training_data`` returns."""

    def __init__(self, cid: int, seed: int):
        self.cid = int(cid)
        self.seed = int(seed)

    def local_update(self, template, global_state, round_idx, cfg):
        X, y = self.training_data()
        model = copy.deepcopy(template)
        set_state(model, global_state)
        gen = torch_gen(self.seed, "local", self.cid, round_idx)
        loss = local_sgd(model, X, y, cfg.local_epochs, cfg.batch_size, cfg.lr, gen, cfg.momentum)
        delta = state_delta(get_state(model), global_state)
        delta, nrm = clip_state(delta, cfg.clip_update)
        return LocalResult(delta, int(X.shape[0]), loss, nrm, state_bytes(delta))


class PlainClient(_SGDClient):
    """``nodp``: trains on raw data."""

    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray, seed: int, transform=None):
        super().__init__(cid, seed)
        X = np.asarray(X, dtype=np.float64)
        self.X = torch.as_tensor(transform(X) if transform is not None else X, dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        self.n_samples = int(self.X.shape[0])

    def training_data(self):
        return self.X, self.y


class SanitizedClient(_SGDClient):
    """``iso`` / ``ldpmic`` / ``dimsample``: Phase 3 happens ONCE in
    ``sanitize`` (called by ``__init__``) and the result is reused for every
    round.  Labels are never perturbed."""

    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray, ranges: np.ndarray, sig: np.ndarray,
                 seed: int, cache_dir: Optional[Path] = None, transform=None, sample_k: Optional[int] = None,
                 s0: Optional[float] = None):
        """``transform`` is an optional PUBLIC (data-independent) map applied to
        the sanitized array before it reaches the model, e.g. the affine
        rescaling by the public ranges; it is post-processing."""
        super().__init__(cid, seed)
        self.transform = transform
        # dimsample: k coordinates sampled per record (perturb.perturb_once_sampled); sig is then the
        # recorded per-coordinate std on a sampled coordinate (Delta_j s0 sqrt(k)).
        self.sample_k, self.s0 = sample_k, s0
        self.ranges = np.asarray(ranges, dtype=float)
        self.sig = np.asarray(sig, dtype=float)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        self.n_samples = int(np.asarray(X).shape[0])
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.sanitized_hash: Optional[str] = None
        self.n_sanitize_calls = 0
        self.X_san = self._sanitize(np.asarray(X, dtype=np.float64))
        self.X = torch.as_tensor(self._to_model(self.X_san), dtype=torch.float32)

    def _to_model(self, Xs: np.ndarray) -> np.ndarray:
        return self.transform(Xs) if self.transform is not None else Xs

    def _cache_key(self, X: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(f"{self.seed}|{self.cid}|".encode())
        h.update(perturb.array_sha256(X).encode())
        h.update(perturb.array_sha256(self.sig).encode())
        h.update(perturb.array_sha256(self.ranges).encode())
        if self.sample_k is not None:
            h.update(f"|sample_k={self.sample_k}".encode())
        return h.hexdigest()[:32]

    def _sanitize(self, X: np.ndarray) -> np.ndarray:
        if self.n_sanitize_calls != 0:
            raise RuntimeError("SanitizedClient.sanitize must run exactly once")
        self.n_sanitize_calls += 1
        cache = None
        if self.cache_dir is not None:
            cache = self.cache_dir / f"sanitized_{self._cache_key(X)}.npz"
            if cache.exists():
                z = np.load(cache)
                Xs, stored = z["X"], str(z["sha256"])
                if perturb.array_sha256(Xs) != stored:
                    raise RuntimeError(f"corrupt sanitized cache {cache}")
                self.sanitized_hash = stored
                self.from_cache = True
                return Xs
        rng = np.random.default_rng(derive_seed(self.seed, "perturb", self.cid))
        if self.sample_k is not None:
            Xs, digest = perturb.perturb_once_sampled(X, self.ranges, self.sample_k, self.s0, rng)
        else:
            Xs, digest = perturb.perturb_once(X, self.ranges, self.sig, rng)
        self.sanitized_hash = digest
        self.from_cache = False
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, X=Xs, sha256=np.array(digest))
        return Xs

    def assert_unchanged(self) -> None:
        if perturb.array_sha256(self.X_san) != self.sanitized_hash:
            raise RuntimeError(f"sanitized data of client {self.cid} changed")
        if perturb.array_sha256(self.X.double().numpy()) != perturb.array_sha256(
                self._to_model(self.X_san).astype(np.float32).astype(np.float64)):
            raise RuntimeError(f"training tensor of client {self.cid} diverged from sanitized data")

    def after_round(self, round_idx: int) -> None:
        self.assert_unchanged()

    def training_data(self):
        return self.X, self.y

    def run_summary(self):
        return {"sanitized_sha256": self.sanitized_hash, "sigma": self.sig.tolist()}


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #

@dataclass
class FLConfig:
    rounds: int
    participation: float
    local_epochs: int
    batch_size: int
    lr: float
    clip_update: Optional[float]
    seed: int
    momentum: float = 0.0
    eval_every: int = 1


def participation_schedule(n_clients: int, rounds: int, q: float, seed: int) -> List[np.ndarray]:
    """Pre-drawn client sample per round: ``max(1, round(q*N))`` distinct
    clients chosen uniformly without replacement (McMahan et al. 2017)."""
    if not (0 < q <= 1):
        raise ValueError("participation rate must be in (0,1]")
    k = max(1, int(round(q * n_clients)))
    rng = np.random.default_rng(derive_seed(seed, "participation"))
    return [np.sort(rng.choice(n_clients, size=k, replace=False)) for _ in range(rounds)]


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 4096) -> Dict[str, float]:
    model.eval()
    logits = torch.cat([model(X[i:i + batch_size]) for i in range(0, X.shape[0], batch_size)])
    out = classification_metrics(logits.numpy(), y.numpy())
    out["test_loss"] = float(F.cross_entropy(logits, y))
    return out


@dataclass
class RoundRecord:
    round: int
    n_participants: int
    train_loss: float
    wall_clock_s: float
    bytes_per_client: float
    mean_update_norm: float
    metrics: Dict[str, float] = field(default_factory=dict)
    client_time_mean_s: float = float("nan")   # mean over participants of local_update wall-clock
    client_time_max_s: float = float("nan")
    eval_time_s: float = float("nan")

    def as_row(self) -> Dict[str, float]:
        row = {
            "round": self.round,
            "n_participants": self.n_participants,
            "train_loss": self.train_loss,
            "wall_clock_s": self.wall_clock_s,
            "bytes_per_client": self.bytes_per_client,
            "mean_update_norm": self.mean_update_norm,
            "client_time_mean_s": self.client_time_mean_s,
            "client_time_max_s": self.client_time_max_s,
            "eval_time_s": self.eval_time_s,
        }
        row.update(self.metrics)
        return row


def fedavg(template: nn.Module, clients: Sequence[Client], cfg: FLConfig, X_test: torch.Tensor,
           y_test: torch.Tensor, schedule: Optional[List[np.ndarray]] = None,
           log=None,
           update_hook=None) -> Tuple[nn.Module, List[RoundRecord]]:
    """Run FedAvg.  Returns the final global model and one record per round.

    Aggregation: the global state moves by the sample-size-weighted mean of
    the participants' (clipped) deltas.  Clients that return ``None`` are
    skipped; a round with no contribution leaves the model unchanged.
    """
    seed_everything(cfg.seed)
    model = copy.deepcopy(template)
    global_state = get_state(model)
    if schedule is None:
        schedule = participation_schedule(len(clients), cfg.rounds, cfg.participation, cfg.seed)
    if len(schedule) < cfg.rounds:
        raise ValueError("schedule shorter than rounds")
    records: List[RoundRecord] = []
    for t in range(cfg.rounds):
        t0 = time.perf_counter()
        results: List[LocalResult] = []
        for cid in schedule[t]:
            tc = time.perf_counter()
            res = clients[int(cid)].local_update(template, global_state, t, cfg)
            if res is not None:
                res.time_s = time.perf_counter() - tc
                if update_hook is not None:      # server-side observer of the (clipped) update, e.g. an attacker
                    update_hook(t, int(cid), res)
                results.append(res)
        if results:
            total = float(sum(r.n_samples for r in results))
            agg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_state.items()}
            for r in results:
                w = r.n_samples / total
                for k in agg:
                    agg[k] += w * r.delta[k].float()
            global_state = {k: global_state[k] + agg[k].to(global_state[k].dtype) for k in global_state}
            train_loss = float(sum(r.train_loss * r.n_samples for r in results) / total)
            bytes_pc = float(np.mean([r.bytes_sent for r in results]))
            upd = float(np.mean([r.update_norm for r in results]))
        else:
            train_loss, bytes_pc, upd = float("nan"), 0.0, float("nan")
        for c in clients:
            c.after_round(t)
        rec = RoundRecord(t, len(results), train_loss, 0.0, bytes_pc, upd)
        if results:
            rec.client_time_mean_s = float(np.mean([r.time_s for r in results]))
            rec.client_time_max_s = float(np.max([r.time_s for r in results]))
        if (t + 1) % cfg.eval_every == 0 or t == cfg.rounds - 1:
            te = time.perf_counter()
            set_state(model, global_state)
            rec.metrics = evaluate(model, X_test, y_test)
            rec.eval_time_s = time.perf_counter() - te
        rec.wall_clock_s = time.perf_counter() - t0
        records.append(rec)
        if log is not None:
            log(rec)
    set_state(model, global_state)
    return model, records
