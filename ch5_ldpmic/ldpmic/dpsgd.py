"""``local_dpsgd`` baseline: client-side DP-SGD with a persistent per-client
RDP accountant.

Each participating round the client runs ``local_epochs`` epochs of
Poisson-subsampled DP-SGD (Abadi et al., CCS 2016) on its RAW data:
per-sample gradients (``torch.func.vmap(grad)``), per-sample l2 clipping to
``clip_grad`` (C_g), Gaussian noise with std ``noise_multiplier * C_g`` on the
summed clipped gradient, division by the expected batch size ``B``.
Every step is composed in ``privacy.RDPAccountant`` across ALL rounds the
client participates in.

Calibration: the noise multiplier is the smallest value for which the
client's *planned* number of steps (planned participations x local epochs x
ceil(m/B), from the pre-drawn participation schedule) spends exactly the
target ``(eps, delta)`` at the end of training.  The client also enforces the
budget itself: it refuses to take a step beyond the planned count, so a
server that over-samples it cannot make it exceed ``(eps, delta)``.

Per-sample gradient computation uses ``torch.func`` rather than Opacus; see
DECISIONS.md for the reason.  The clipping/noising arithmetic mirrors
``opacus.optimizers.DPOptimizer`` (clip factor ``min(1, C/(||g||+1e-6))``,
noise ``N(0, (sigma C)^2)`` on the sum, scaling by the expected batch size).

Anisotropic variant (mechanism B, ``aniso_dpsgd``): ``DPSGDClientSpec.input_scale``
is a vector D of length d (D_j > 0, sum_j D_j^2 = d) applied to the columns
of the FIRST-LAYER weight gradient (column j multiplies input feature j).
Per-sample clipping and the Gaussian noise act on the SCALED gradient, which
is then unscaled: in the original coordinates the noise on column j has std
``sigma C / D_j`` and the per-sample contribution of column j is bounded by
``C / D_j``.  Privacy: the scaled sum has l2 sensitivity exactly C, so this
is the ordinary Gaussian mechanism in a fixed, data-independent change of
coordinates; the same subsampled-Gaussian RDP accountant applies unchanged
(D is public: it is post-processing of the Phase 1 score release).  With
D = 1 (uniform shares) every step is bit-for-bit plain DP-SGD.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .fl import Client, LocalResult, clip_state, get_state, set_state, state_bytes, state_delta, torch_gen
from .privacy import DEFAULT_ORDERS, RDPAccountant, calibrate_noise_multiplier


@dataclass
class DPSGDClientSpec:
    target_eps: float
    target_delta: float
    clip_grad: float
    planned_participations: int
    local_epochs: int
    batch_size: int
    conversion: str = "balle"
    noise_multiplier: Optional[float] = None   # override the calibration (tests)
    # mechanism B: per-input-feature gradient scale D (length d) on the columns
    # of the first-layer weight; None = isotropic DP-SGD
    input_scale: Optional[np.ndarray] = None
    input_weight_param: str = "net.0.weight"   # parameter whose columns are indexed by input features


def per_sample_clipped_grad_sum(model: nn.Module, params: Dict[str, torch.Tensor],
                                buffers: Dict[str, torch.Tensor], X: torch.Tensor, y: torch.Tensor,
                                clip: float, scale: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
    """Sum over the batch of per-sample gradients, each clipped to l2 norm
    ``clip``.  Returns zeros for an empty batch.  ``scale`` (parameter name ->
    tensor broadcastable to that parameter) multiplies the per-sample gradient
    BEFORE clipping; the result is then in the scaled coordinates (the caller
    adds isotropic noise there and divides by the scale afterwards)."""
    if X.shape[0] == 0:
        return {k: torch.zeros_like(v) for k, v in params.items()}

    def loss_fn(p, b, x, t):
        out = functional_call(model, (p, b), (x.unsqueeze(0),))
        return F.cross_entropy(out, t.unsqueeze(0))

    g = vmap(grad(loss_fn), in_dims=(None, None, 0, 0))(params, buffers, X, y)
    if scale:
        g = {k: (v * scale[k] if k in scale else v) for k, v in g.items()}
    sq = sum((v.flatten(1) ** 2).sum(1) for v in g.values())
    factor = torch.clamp(clip / (sq.sqrt() + 1e-6), max=1.0)
    return {k: torch.einsum("i,i...->...", factor, v) for k, v in g.items()}


class DPSGDClient(Client):
    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray, seed: int, spec: DPSGDClientSpec,
                 orders=DEFAULT_ORDERS, transform=None):
        self.cid = int(cid)
        self.seed = int(seed)
        self.spec = spec
        X = np.asarray(X, dtype=np.float64)
        self.X = torch.as_tensor(transform(X) if transform is not None else X, dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        self.n_samples = m = int(self.X.shape[0])
        if spec.batch_size <= 0 or spec.local_epochs <= 0 or spec.clip_grad <= 0:
            raise ValueError("batch_size, local_epochs and clip_grad must be positive")
        self.sample_rate = min(1.0, spec.batch_size / m)
        self.steps_per_epoch = math.ceil(m / spec.batch_size)
        self.steps_per_round = self.steps_per_epoch * spec.local_epochs
        self.max_steps = spec.planned_participations * self.steps_per_round
        self.accountant = RDPAccountant(orders, spec.conversion)
        if spec.noise_multiplier is not None:
            self.noise_multiplier = float(spec.noise_multiplier)
        elif self.max_steps > 0:
            self.noise_multiplier = calibrate_noise_multiplier(
                spec.target_eps, spec.target_delta, self.sample_rate, self.max_steps, orders, spec.conversion)
        else:
            self.noise_multiplier = math.inf  # never trains
        self.exhausted = False
        self.n_rounds_trained = 0
        self.input_scale = None
        if spec.input_scale is not None:
            D = np.asarray(spec.input_scale, dtype=np.float64)
            if D.ndim != 1 or D.shape[0] != self.X.shape[1] or not np.all(np.isfinite(D)) or np.any(D <= 0):
                raise ValueError("input_scale must be a finite positive vector of length d")
            self.input_scale = torch.as_tensor(D, dtype=torch.float32)

    def _scale_dict(self, params) -> Optional[Dict[str, torch.Tensor]]:
        if self.input_scale is None:
            return None
        name = self.spec.input_weight_param
        if name not in params or params[name].ndim != 2 or params[name].shape[1] != self.input_scale.shape[0]:
            raise ValueError(f"input_scale needs a first-layer weight {name!r} of shape (h, d)")
        return {name: self.input_scale.reshape(1, -1)}

    # ------------------------------------------------------------------ #
    def epsilon_spent(self) -> float:
        return self.accountant.get_epsilon(self.spec.target_delta)

    def training_data(self):
        return self.X, self.y

    def local_update(self, template, global_state, round_idx, cfg):
        if self.exhausted or self.accountant.total_steps() >= self.max_steps:
            self.exhausted = True
            return None
        model = copy.deepcopy(template)
        set_state(model, global_state)
        params = {k: v.detach().clone() for k, v in model.named_parameters()}
        buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
        gen = torch_gen(self.seed, "dpsgd", self.cid, round_idx)
        rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,), generator=gen).item())
        sigma_abs = self.noise_multiplier * self.spec.clip_grad
        B = self.spec.batch_size if self.sample_rate < 1.0 else self.n_samples
        scale = self._scale_dict(params)
        losses = []
        steps = 0
        for _ in range(self.steps_per_round):
            if self.accountant.total_steps() >= self.max_steps:
                self.exhausted = True
                break
            mask = rng.random(self.n_samples) < self.sample_rate
            idx = torch.as_tensor(np.nonzero(mask)[0], dtype=torch.long)
            Xb, yb = self.X[idx], self.y[idx]
            if idx.numel() > 0:
                with torch.no_grad():
                    losses.append(float(F.cross_entropy(functional_call(model, (params, buffers), (Xb,)), yb)))
            gsum = per_sample_clipped_grad_sum(model, params, buffers, Xb, yb, self.spec.clip_grad, scale)
            for k in params:
                noise = torch.randn(params[k].shape, generator=gen) * sigma_abs
                step = gsum[k] + noise
                if scale and k in scale:
                    step = step / scale[k]      # back to the original coordinates
                params[k] = params[k] - cfg.lr * step / B
            self.accountant.step(self.noise_multiplier, self.sample_rate, 1)
            steps += 1
        if steps == 0:
            return None
        self.n_rounds_trained += 1
        new_state = dict(global_state)
        new_state.update({k: v.detach() for k, v in params.items()})
        new_state.update(buffers)
        delta = state_delta(new_state, global_state)
        delta, nrm = clip_state(delta, cfg.clip_update)
        loss = float(np.mean(losses)) if losses else float("nan")
        return LocalResult(delta, self.n_samples, loss, nrm, state_bytes(delta))

    def run_summary(self):
        return {
            "noise_multiplier": self.noise_multiplier,
            "steps": self.accountant.total_steps(),
            "max_steps": self.max_steps,
            "eps_spent": self.epsilon_spent(),
            "delta": self.spec.target_delta,
            "anisotropic": self.input_scale is not None,
        }
