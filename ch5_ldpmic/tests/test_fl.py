import numpy as np
import pytest
import torch

from ldpmic import fl, perturb, pipeline
from ldpmic.dpsgd import DPSGDClient, DPSGDClientSpec
from ldpmic.fl import FLConfig, SanitizedClient, fedavg, participation_schedule
from ldpmic.models import MLP
from ldpmic.privacy import dpsgd_epsilon


def _fl_cfg(**kw):
    base = dict(rounds=3, participation=0.5, local_epochs=1, batch_size=32, lr=0.05, clip_update=5.0, seed=0)
    base.update(kw)
    return FLConfig(**base)


def _tensors(X, y):
    return torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)


def test_one_shot(synthetic_federated, monkeypatch, tmp_path):
    """3 rounds of FedAvg leave every sanitized dataset's hash unchanged and
    perturb_once is called exactly once per client."""
    clients_data, (Xt, yt), ranges, C = synthetic_federated
    calls = []
    real = perturb.perturb_once

    def counting(X, r, s, rng):
        calls.append(id(X))
        return real(X, r, s, rng)

    monkeypatch.setattr(perturb, "perturb_once", counting)
    cfg = _fl_cfg(participation=1.0)
    mcfg = pipeline.MethodConfig(method="ldpmic", eps=2.0)
    clients, info = pipeline.build_clients(mcfg, clients_data, ranges, C, seed=0, fl_cfg=cfg, cache_dir=tmp_path)
    assert len(calls) == len(clients_data)
    hashes = [c.sanitized_hash for c in clients]
    assert all(h is not None for h in hashes) and len(set(hashes)) == len(hashes)
    model = MLP(ranges.shape[0], C)
    _, recs = fedavg(model, clients, cfg, *_tensors(Xt, yt))
    assert len(recs) == 3
    assert len(calls) == len(clients_data)  # still once per client
    for c, h in zip(clients, hashes):
        assert c.sanitized_hash == h
        assert perturb.array_sha256(c.X_san) == h
        c.assert_unchanged()
    # the cache is reused on a second build (no new perturbation)
    clients2, _ = pipeline.build_clients(mcfg, clients_data, ranges, C, seed=0, fl_cfg=cfg, cache_dir=tmp_path)
    assert len(calls) == len(clients_data)
    assert [c.sanitized_hash for c in clients2] == hashes
    assert all(c.from_cache for c in clients2)


def test_sanitized_client_detects_tampering(synthetic_federated):
    clients_data, _, ranges, C = synthetic_federated
    X, y = clients_data[0]
    sig = np.full(ranges.shape[0], 0.5)
    c = SanitizedClient(0, X, y, ranges, sig, seed=1)
    c.assert_unchanged()
    c.X_san[0, 0] += 1.0
    with pytest.raises(RuntimeError):
        c.assert_unchanged()


def test_fedavg_deterministic_and_learns(synthetic_federated):
    clients_data, (Xt, yt), ranges, C = synthetic_federated
    cfg = _fl_cfg(rounds=8, participation=0.5, lr=0.1, clip_update=None)
    mcfg = pipeline.MethodConfig(method="nodp")
    outs = []
    for _ in range(2):
        clients, _ = pipeline.build_clients(mcfg, clients_data, ranges, C, seed=3, fl_cfg=cfg)
        torch.manual_seed(3)
        model = MLP(ranges.shape[0], C)
        model, recs = fedavg(model, clients, cfg, *_tensors(Xt, yt))
        outs.append((recs, fl.get_state(model)))
    r0, s0 = outs[0]
    r1, s1 = outs[1]
    assert [r.metrics["accuracy"] for r in r0] == [r.metrics["accuracy"] for r in r1]
    for k in s0:
        assert torch.equal(s0[k], s1[k])
    assert r0[-1].metrics["accuracy"] > r0[0].metrics["accuracy"]
    assert r0[-1].metrics["accuracy"] > 0.5
    assert all(r.n_participants == 2 for r in r0)
    assert all(r.bytes_per_client > 0 and r.wall_clock_s >= 0 for r in r0)
    assert {"accuracy", "balanced_accuracy", "auc", "test_loss"} <= set(r0[-1].metrics)


def test_update_clipping_bounds_norm(synthetic_federated):
    clients_data, _, ranges, C = synthetic_federated
    X, y = clients_data[0]
    cfg = _fl_cfg(clip_update=0.01, lr=1.0, local_epochs=2)
    c = fl.PlainClient(0, X, y, seed=0)
    model = MLP(ranges.shape[0], C)
    res = c.local_update(model, fl.get_state(model), 0, cfg)
    assert res.update_norm > 0.01
    assert fl.state_norm(res.delta) == pytest.approx(0.01, rel=1e-5)


def test_participation_schedule_properties():
    sched = participation_schedule(100, 20, 0.3, seed=0)
    assert len(sched) == 20 and all(len(s) == 30 and len(set(s)) == 30 for s in sched)
    assert [s.tolist() for s in participation_schedule(100, 20, 0.3, seed=0)] == [s.tolist() for s in sched]
    assert [s.tolist() for s in participation_schedule(100, 20, 0.3, seed=1)] != [s.tolist() for s in sched]


def test_dpsgd_client_budget_and_accounting(synthetic_federated):
    """The DP-SGD client composes every local step across rounds, ends at the
    target eps, and refuses to train past its planned step count."""
    clients_data, (Xt, yt), ranges, C = synthetic_federated
    X, y = clients_data[1]  # m = 80
    spec = DPSGDClientSpec(target_eps=4.0, target_delta=1e-5, clip_grad=1.0, planned_participations=2,
                           local_epochs=1, batch_size=20)
    c = DPSGDClient(1, X, y, seed=0, spec=spec)
    assert c.steps_per_round == 4 and c.max_steps == 8
    assert c.sample_rate == pytest.approx(0.25)
    eps_planned = dpsgd_epsilon(c.noise_multiplier, 0.25, 8, 1e-5)
    assert eps_planned <= 4.0 and eps_planned > 4.0 * 0.99
    cfg = _fl_cfg(local_epochs=1, batch_size=20, lr=0.1)
    model = MLP(ranges.shape[0], C)
    state = fl.get_state(model)
    r1 = c.local_update(model, state, 0, cfg)
    assert r1 is not None and c.accountant.total_steps() == 4
    assert 0 < c.epsilon_spent() < 4.0
    r2 = c.local_update(model, state, 1, cfg)
    assert r2 is not None and c.accountant.total_steps() == 8
    assert c.epsilon_spent() == pytest.approx(eps_planned)
    # a third (unplanned) participation is refused: budget is enforced client-side
    assert c.local_update(model, state, 2, cfg) is None
    assert c.accountant.total_steps() == 8 and c.exhausted
    assert fl.state_norm(r1.delta) > 0


def test_dpsgd_per_sample_clipping_bounds_gradient_sum(synthetic_federated):
    from ldpmic.dpsgd import per_sample_clipped_grad_sum

    clients_data, _, ranges, C = synthetic_federated
    X, y = _tensors(*clients_data[0])
    model = MLP(ranges.shape[0], C)
    params = {k: v.detach() for k, v in model.named_parameters()}
    g = per_sample_clipped_grad_sum(model, params, {}, X[:16], y[:16], clip=0.1)
    total = torch.sqrt(sum((v ** 2).sum() for v in g.values()))
    assert float(total) <= 16 * 0.1 + 1e-6
    g0 = per_sample_clipped_grad_sum(model, params, {}, X[:0], y[:0], clip=0.1)
    assert all(torch.all(v == 0) for v in g0.values())


def test_local_dpsgd_pipeline_end_to_end(synthetic_federated):
    clients_data, (Xt, yt), ranges, C = synthetic_federated
    cfg = _fl_cfg(rounds=3, participation=0.5, batch_size=40)
    sched = participation_schedule(len(clients_data), cfg.rounds, cfg.participation, cfg.seed)
    mcfg = pipeline.MethodConfig(method="local_dpsgd", eps=8.0, delta=1e-5, clip_grad=1.0, dpsgd_calibration="schedule")
    clients, info = pipeline.build_clients(mcfg, clients_data, ranges, C, seed=0, fl_cfg=cfg, schedule=sched)
    model = MLP(ranges.shape[0], C)
    _, recs = fedavg(model, clients, cfg, *_tensors(Xt, yt), schedule=sched)
    counts = np.bincount(np.concatenate(sched), minlength=len(clients))
    for c, k in zip(clients, counts):
        assert c.n_rounds_trained == k
        if k > 0:
            assert c.epsilon_spent() <= 8.0 + 1e-9
            assert c.epsilon_spent() > 8.0 * 0.99
        else:
            assert c.epsilon_spent() == 0.0
    assert len(recs) == 3


@pytest.mark.filterwarnings("ignore:Full backward hook")
def test_per_sample_gradients_match_autograd_loop_and_opacus(synthetic_federated):
    """The vmap per-sample gradients equal a per-record autograd loop and
    (when Opacus is importable) Opacus' GradSampleModule."""
    import torch.nn.functional as F
    from ldpmic.dpsgd import per_sample_clipped_grad_sum

    clients_data, _, ranges, C = synthetic_federated
    X, y = _tensors(*clients_data[0])
    X, y = X[:16], y[:16]
    torch.manual_seed(0)
    model = MLP(ranges.shape[0], C)
    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    ours = per_sample_clipped_grad_sum(model, params, {}, X, y, clip=1e9)  # huge clip: plain sum
    loop = {k: torch.zeros_like(v) for k, v in params.items()}
    for i in range(X.shape[0]):
        model.zero_grad()
        F.cross_entropy(model(X[i:i + 1]), y[i:i + 1]).backward()
        for k, p in model.named_parameters():
            loop[k] += p.grad
    for k in loop:
        torch.testing.assert_close(ours[k], loop[k], rtol=1e-5, atol=1e-5)
    opacus = pytest.importorskip("opacus")
    gm = opacus.GradSampleModule(model, loss_reduction="sum")
    gm.zero_grad()
    F.cross_entropy(gm(X), y, reduction="sum").backward()
    for k, p in gm.named_parameters():
        torch.testing.assert_close(p.grad_sample.sum(0), loop[k.replace("_module.", "")], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("calibration", ["schedule", "worst_case"])
def test_dpsgd_calibration_modes(synthetic_federated, calibration):
    """worst_case: every client budgets for all T rounds (smaller noise per
    step is impossible, more noise than schedule-based); schedule: planned
    participations only."""
    clients_data, _, ranges, C = synthetic_federated
    cfg = _fl_cfg(rounds=4, participation=0.5, batch_size=40)
    sched = participation_schedule(len(clients_data), cfg.rounds, cfg.participation, cfg.seed)
    mcfg = pipeline.MethodConfig(method="local_dpsgd", eps=4.0, delta=1e-5, dpsgd_calibration=calibration)
    clients, info = pipeline.build_clients(mcfg, clients_data, ranges, C, seed=0, fl_cfg=cfg, schedule=sched)
    counts = np.bincount(np.concatenate(sched), minlength=len(clients))
    for c, k in zip(clients, counts):
        expected = cfg.rounds if calibration == "worst_case" else int(k)
        assert c.max_steps == expected * c.steps_per_round
    assert f"dpsgd_calibration={calibration}" in info.notes
    with pytest.raises(ValueError):
        pipeline.build_clients(pipeline.MethodConfig(method="local_dpsgd", dpsgd_calibration="bogus"),
                               clients_data, ranges, C, seed=0, fl_cfg=cfg, schedule=sched)


def test_worst_case_calibration_never_exhausts(synthetic_federated):
    clients_data, (Xt, yt), ranges, C = synthetic_federated
    cfg = _fl_cfg(rounds=3, participation=1.0, batch_size=40)
    sched = participation_schedule(len(clients_data), cfg.rounds, cfg.participation, cfg.seed)
    mcfg = pipeline.MethodConfig(method="local_dpsgd", eps=8.0, delta=1e-5, dpsgd_calibration="worst_case")
    clients, _ = pipeline.build_clients(mcfg, clients_data, ranges, C, seed=0, fl_cfg=cfg, schedule=sched)
    _, recs = fedavg(MLP(ranges.shape[0], C), clients, cfg, *_tensors(Xt, yt), schedule=sched)
    assert all(r.n_participants == len(clients) for r in recs)
    assert all(c.epsilon_spent() <= 8.0 + 1e-9 and not c.exhausted for c in clients)


def test_fs_dpsgd_selects_k_features_and_composes_budget(synthetic_federated, monkeypatch):
    """Mechanism A: one private score release per client, global top-k set,
    DP-SGD on k inputs with eps2; realized budget eps1 + eps2."""
    from ldpmic import scores as sc
    clients_data, (Xt, yt), ranges, C = synthetic_federated
    calls = []
    real = sc.privatize_scores
    monkeypatch.setattr(sc, "privatize_scores", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    cfg = _fl_cfg(rounds=2, participation=1.0, batch_size=40)
    sched = participation_schedule(len(clients_data), cfg.rounds, cfg.participation, cfg.seed)
    mcfg = pipeline.MethodConfig(method="fs_dpsgd", eps=4.0, delta=1e-5, k=3, allocation="global",
                                 score_sensitivity="analytic", dpsgd_calibration="worst_case")
    clients, info = pipeline.build_clients(mcfg, clients_data, ranges, C, seed=0, fl_cfg=cfg, schedule=sched)
    assert len(calls) == len(clients_data)
    assert len(info.selected_features) == 3 == info.input_dim and info.selected_features == sorted(info.selected_features)
    assert info.eps1 == pytest.approx(0.8) and info.eps2 == pytest.approx(3.2)
    assert clients[0].X.shape[1] == 3
    assert all(c.spec.target_eps == pytest.approx(3.2) for c in clients)
    assert any("top-k overlap" in n for n in info.notes)
    model = MLP(3, C)
    Xt_sel = torch.as_tensor(Xt[:, info.selected_features], dtype=torch.float32)
    _, recs = fedavg(model, clients, cfg, Xt_sel, torch.as_tensor(yt), schedule=sched)
    assert len(recs) == 2 and all(r.n_participants == len(clients) for r in recs)
    assert max(c.epsilon_spent() for c in clients) <= 3.2 + 1e-9
    # oracle variant: no private release, flagged non-private
    calls.clear()
    mo = pipeline.MethodConfig(method="fs_dpsgd", eps=4.0, k=3, allocation="oracle")
    _, info_o = pipeline.build_clients(mo, clients_data, ranges, C, seed=0, fl_cfg=cfg, schedule=sched)
    assert calls == [] and not info_o.private and len(info_o.selected_features) == 3
    with pytest.raises(ValueError):
        pipeline.build_clients(pipeline.MethodConfig(method="fs_dpsgd", k=99), clients_data, ranges, C, seed=0,
                               fl_cfg=cfg, schedule=sched)


def test_select_top_k_and_validation():
    sel = pipeline.select_top_k(np.array([0.1, 0.9, 0.5, 0.9]), 2)
    assert sel == [1, 3]
    assert pipeline.validate_feature_set([3, 1], 2, 4) == [1, 3]
    for bad in ([1, 1], [1, 2, 3], [5, 1]):
        with pytest.raises(ValueError):
            pipeline.validate_feature_set(bad, 2, 4)


def test_secagg_score_release_noise_and_selection(synthetic_federated):
    """secagg: the m-weighted average of the masked contributions carries
    exactly N(0, sigma^2 I) with sigma = sens * sigma0(eps1, delta1) (checked
    over many seeds on Phase 1 alone), the sensitivity is
    sqrt(d) max_i m_i Delta_i / M, and with that much less noise the selected
    top-k overlaps the oracle set."""
    from ldpmic import scores as sc
    from ldpmic.privacy import sigma0_analytic
    clients_data, _, ranges, C = synthetic_federated
    sizes = [X.shape[0] for X, _ in clients_data]
    d = ranges.shape[0]
    deltas_s = [sc.score_sensitivity(10, m, C, "analytic") for m in sizes]
    sens = pipeline.secagg_sum_sensitivity(sizes, deltas_s, d)
    assert sens == pytest.approx(np.sqrt(d) * max(m * ds for m, ds in zip(sizes, deltas_s)) / sum(sizes))
    s0 = sigma0_analytic
    # (1) noise of the aggregate: Phase 1 alone at a large eps1 so that the
    # [0,1] clip of the aggregate (post-processing) is rarely active
    eps1, delta1 = 50.0, 5e-6
    sigma = sens * s0(eps1, delta1)
    assert sigma < 0.05
    mcfg = pipeline.MethodConfig(method="fs_dpsgd", k=3, allocation="global", score_sensitivity="analytic",
                                 score_release="secagg")
    devs = []
    for seed in range(40):
        info = pipeline.RunInfo(method="fs_dpsgd", private=True, eps_target=100.0, delta_target=1e-5)
        noisy, raws, priv_avg, raw_avg = pipeline._phase1_scores(mcfg, clients_data, ranges, C, seed, eps1, delta1, s0, info)
        inner = (priv_avg > 0.0) & (priv_avg < 1.0)
        devs.extend((priv_avg - raw_avg)[inner].tolist())
        assert info.sigma_s == pytest.approx(sigma, rel=1e-12)
    assert len(devs) >= 150 and np.std(devs) == pytest.approx(sigma, rel=0.2)
    # (2) end to end at a moderate budget: informative features (0,1,2 in the
    # synthetic fixture) are mostly recovered, budget split and notes recorded
    cfg = _fl_cfg(rounds=1, participation=1.0, batch_size=40)
    sched = participation_schedule(len(clients_data), cfg.rounds, cfg.participation, cfg.seed)
    mcfg = pipeline.MethodConfig(method="fs_dpsgd", eps=5.0, delta=1e-5, eps1_frac=0.5, k=3, allocation="global",
                                 score_sensitivity="analytic", score_release="secagg", dpsgd_calibration="schedule")
    overlaps = []
    for seed in range(8):
        _, info = pipeline.build_clients(mcfg, clients_data, ranges, C, seed=seed, fl_cfg=cfg, schedule=sched)
        overlaps.append(len(set(info.selected_features) & {0, 1, 2}))
    assert np.mean(overlaps) >= 2.0   # random selection of 3 of 6 would give 1.5
    assert any("secagg score release" in n for n in info.notes) and any("threat model" in n for n in info.notes)
    assert info.eps1 == pytest.approx(2.5) and info.eps2 == pytest.approx(2.5)
    with pytest.raises(ValueError):
        pipeline.build_clients(pipeline.MethodConfig(method="fs_dpsgd", k=3, score_release="bogus"), clients_data,
                               ranges, C, seed=0, fl_cfg=cfg, schedule=sched)


# --------------------------------------------------------------------------- #
# Mechanism B: anisotropic (score-weighted) DP-SGD
# --------------------------------------------------------------------------- #
def _dp_spec(**kw):
    base = dict(target_eps=2.0, target_delta=1e-5, clip_grad=1.0, planned_participations=2, local_epochs=1,
                batch_size=32, noise_multiplier=1.5)
    base.update(kw)
    return DPSGDClientSpec(**base)


def test_gradient_scale_budget_and_uniform_identity():
    d = 7
    D = pipeline.gradient_scale(np.full(d, 1.0 / d))
    assert np.allclose(D, 1.0)
    p = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02])
    D = pipeline.gradient_scale(p)
    assert D.sum() > 0 and np.isclose((D ** 2).sum(), d)         # fixed scale budget sum D_j^2 = d
    assert np.allclose(D ** 2 / d, p)                            # precision proportional to the share
    with pytest.raises(ValueError):
        pipeline.gradient_scale(np.array([0.5, 0.6]))


def test_aniso_dpsgd_uniform_scale_is_plain_dpsgd(synthetic_federated):
    """With D = 1 the anisotropic client takes exactly the same steps as the
    isotropic one (same seed, same noise), and spends the same budget."""
    clients_data, (Xt, yt), ranges, C = synthetic_federated
    X, y = clients_data[0]
    d = X.shape[1]
    cfg = _fl_cfg(rounds=2, participation=1.0, batch_size=32, lr=0.1)
    plain = DPSGDClient(0, X, y, seed=3, spec=_dp_spec())
    aniso = DPSGDClient(0, X, y, seed=3, spec=_dp_spec(input_scale=np.ones(d)))
    model = MLP(d, C)
    state = fl.get_state(model)
    r1 = plain.local_update(model, state, 0, cfg)
    r2 = aniso.local_update(model, state, 0, cfg)
    for k in r1.delta:
        assert torch.allclose(r1.delta[k], r2.delta[k], atol=1e-6, rtol=1e-5), k
    assert plain.epsilon_spent() == pytest.approx(aniso.epsilon_spent())
    assert plain.noise_multiplier == aniso.noise_multiplier
    assert aniso.run_summary()["anisotropic"] and not plain.run_summary()["anisotropic"]


def test_aniso_dpsgd_noise_scales_inversely_with_D(synthetic_federated, monkeypatch):
    """With the gradient forced to zero the update is pure noise; its std on
    first-layer column j must be sigma C / D_j (D public, noise isotropic in
    the scaled coordinates), and untouched parameters keep std sigma C."""
    from ldpmic import dpsgd as dpm
    clients_data, (Xt, yt), ranges, C = synthetic_federated
    X, y = clients_data[2]
    d = X.shape[1]
    D = np.array([3.0, 1.0 / 3.0, 1.0, 1.0, 1.0, 1.0]); D = D * np.sqrt(d / (D ** 2).sum())
    monkeypatch.setattr(dpm, "per_sample_clipped_grad_sum",
                        lambda model, params, buffers, X, y, clip, scale=None: {k: torch.zeros_like(v) for k, v in params.items()})
    cfg = _fl_cfg(rounds=1, participation=1.0, batch_size=1000, lr=1.0, clip_update=None)
    spec = _dp_spec(batch_size=1000, planned_participations=40, noise_multiplier=1.0, clip_grad=1.0)
    spec.input_scale = D
    client = DPSGDClient(2, X, y, seed=5, spec=spec)
    model = MLP(d, C, hidden=(256, 32))
    state = fl.get_state(model)
    cols = []
    for r in range(40):                       # 40 rounds x 256 rows of pure noise per column
        res = client.local_update(model, state, r, cfg)
        cols.append(res.delta["net.0.weight"] * client.n_samples)   # undo the 1/B scaling
    W = torch.cat(cols, 0).numpy()            # (40*256, d), one step per round
    std = W.std(0)
    assert np.allclose(std, 1.0 / D, rtol=0.08)
    other = torch.cat([c.flatten() for c in [res.delta["net.2.weight"] * client.n_samples]]).numpy()
    assert abs(other.std() - 1.0) < 0.08


def test_aniso_dpsgd_pipeline_global_and_oracle(synthetic_federated):
    clients_data, (Xt, yt), ranges, C = synthetic_federated
    d = ranges.shape[0]
    cfg = _fl_cfg(rounds=2, participation=1.0, batch_size=40)
    sched = participation_schedule(len(clients_data), cfg.rounds, cfg.participation, cfg.seed)
    mcfg = pipeline.MethodConfig(method="aniso_dpsgd", eps=4.0, delta=1e-5, allocation="global", beta=2.0,
                                 score_sensitivity="analytic", score_release="secagg", dpsgd_calibration="schedule")
    clients, info = pipeline.build_clients(mcfg, clients_data, ranges, C, seed=0, fl_cfg=cfg, schedule=sched)
    assert info.private and info.eps1 == pytest.approx(0.8) and info.eps2 == pytest.approx(3.2)
    p = np.array(info.shares["global"]); D = np.array(info.gradient_scale)
    assert p.shape == (d,) and np.isclose(p.sum(), 1.0) and np.all(p > 0)
    assert np.allclose(D, np.sqrt(d * p)) and info.input_dim == d
    assert all(c.input_scale is not None and c.spec.target_eps == pytest.approx(3.2) for c in clients)
    model = MLP(d, C)
    _, recs = fedavg(model, clients, cfg, *_tensors(Xt, yt), schedule=sched)
    assert len(recs) == 2 and all(np.isfinite(r.train_loss) for r in recs)
    assert max(c.epsilon_spent() for c in clients) <= 3.2 + 1e-9
    mo = pipeline.MethodConfig(method="aniso_dpsgd", eps=4.0, allocation="oracle")
    _, info_o = pipeline.build_clients(mo, clients_data, ranges, C, seed=0, fl_cfg=cfg, schedule=sched)
    assert not info_o.private and len(info_o.gradient_scale) == d
    with pytest.raises(ValueError):
        pipeline.build_clients(pipeline.MethodConfig(method="aniso_dpsgd", allocation="per_client"), clients_data,
                               ranges, C, seed=0, fl_cfg=cfg, schedule=sched)
    # public scores: eps1_frac = 0 sends the whole budget to DP-SGD; only with oracle
    mp = pipeline.MethodConfig(method="aniso_dpsgd", eps=4.0, allocation="oracle", eps1_frac=0.0)
    cl_p, info_p = pipeline.build_clients(mp, clients_data, ranges, C, seed=0, fl_cfg=cfg, schedule=sched)
    assert info_p.eps1 == 0 and info_p.delta1 == 0 and info_p.eps2 == 4.0 and info_p.delta2 == 1e-5
    assert all(c.spec.target_eps == 4.0 and c.spec.target_delta == 1e-5 for c in cl_p)
    with pytest.raises(ValueError):
        pipeline.build_clients(pipeline.MethodConfig(method="aniso_dpsgd", allocation="global", eps1_frac=0.0),
                               clients_data, ranges, C, seed=0, fl_cfg=cfg, schedule=sched)


# --------------------------------------------------------------------------- #
# Pilot v5: selection + one-shot perturbation (ldpmic with k > 0)
# --------------------------------------------------------------------------- #
def test_ldpmic_with_selection_releases_k_coordinates(synthetic_federated, tmp_path):
    clients_data, (Xt, yt), ranges, C = synthetic_federated
    d = ranges.shape[0]
    cfg = _fl_cfg(rounds=2, participation=1.0)
    scaler = lambda X: (np.asarray(X) - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0])   # noqa: E731
    mcfg = pipeline.MethodConfig(method="ldpmic", eps=4.0, delta=1e-5, k=3, allocation="global",
                                 score_sensitivity="analytic", score_release="secagg")
    clients, info = pipeline.build_clients(mcfg, clients_data, ranges, C, seed=0, fl_cfg=cfg, cache_dir=tmp_path,
                                           transform=scaler)
    sel = info.selected_features
    assert len(sel) == 3 == info.input_dim and sel == sorted(sel)
    p = np.array(info.shares["global"]); assert p.shape == (3,) and np.isclose(p.sum(), 1.0)
    sig = np.array(info.sigmas["0"]); assert sig.shape == (3,)
    dl = ranges[sel, 1] - ranges[sel, 0]
    assert np.isclose((dl ** 2 / sig ** 2).sum(), 1.0 / info.sigma0 ** 2, rtol=1e-9)   # invariant on S
    assert all(c.X.shape[1] == 3 for c in clients) and info.eps_realized == pytest.approx(4.0)
    # the restricted public transform equals the full one on the selected columns
    X0 = np.asarray(clients_data[0][0]); Z = clients[0].transform(X0[:, sel])
    assert np.allclose(Z, scaler(X0)[:, sel])
    model = MLP(3, C)
    _, recs = fedavg(model, clients, cfg, torch.as_tensor(scaler(Xt)[:, sel], dtype=torch.float32),
                     torch.as_tensor(yt), schedule=None)
    assert len(recs) == 2
    # uniform shares on S = the selected-uniform control; oracle flagged non-private; per_client refused
    mu = pipeline.MethodConfig(method="ldpmic", eps=4.0, k=3, allocation="global", share_mode="uniform")
    _, info_u = pipeline.build_clients(mu, clients_data, ranges, C, seed=0, fl_cfg=cfg)
    assert np.allclose(info_u.shares["global"], 1 / 3)
    mo = pipeline.MethodConfig(method="ldpmic", eps=4.0, k=3, allocation="oracle")
    _, info_o = pipeline.build_clients(mo, clients_data, ranges, C, seed=0, fl_cfg=cfg)
    assert not info_o.private and len(info_o.selected_features) == 3
    with pytest.raises(ValueError):
        pipeline.build_clients(pipeline.MethodConfig(method="ldpmic", k=3, allocation="per_client"), clients_data,
                               ranges, C, seed=0, fl_cfg=cfg)
    # random k-subset control: seeded, no scores, uniform shares, eps1 unspent
    mr = pipeline.MethodConfig(method="ldpmic", eps=4.0, k=3, allocation="global", share_mode="uniform",
                               selection="random")
    cl_r, info_r = pipeline.build_clients(mr, clients_data, ranges, C, seed=0, fl_cfg=cfg)
    cl_r2, info_r2 = pipeline.build_clients(mr, clients_data, ranges, C, seed=0, fl_cfg=cfg)
    assert info_r.selected_features == info_r2.selected_features and len(info_r.selected_features) == 3
    assert info_r.eps1 == 0 and info_r.eps_realized == pytest.approx(3.2) and all(c.X.shape[1] == 3 for c in cl_r)
    with pytest.raises(ValueError):
        pipeline.build_clients(pipeline.MethodConfig(method="ldpmic", k=3, allocation="global", selection="random"),
                               clients_data, ranges, C, seed=0, fl_cfg=cfg)


def test_fs_dpsgd_selection_controls(synthetic_federated):
    """Pilot v8 controls: random k-subset and public variance mask for the
    DP-SGD filter path; no Phase 1 budget is spent, eps2 is unchanged."""
    clients_data, (Xt, yt), ranges, C = synthetic_federated
    d = ranges.shape[0]
    cfg = _fl_cfg(rounds=2, participation=1.0, batch_size=40)
    sched = participation_schedule(len(clients_data), cfg.rounds, cfg.participation, cfg.seed)
    base = dict(method="fs_dpsgd", eps=4.0, delta=1e-5, k=3, allocation="global", score_sensitivity="analytic",
                score_release="secagg", dpsgd_calibration="schedule")
    cl_r, info_r = pipeline.build_clients(pipeline.MethodConfig(selection="random", **base), clients_data, ranges, C,
                                          seed=0, fl_cfg=cfg, schedule=sched)
    cl_r2, info_r2 = pipeline.build_clients(pipeline.MethodConfig(selection="random", **base), clients_data, ranges, C,
                                            seed=0, fl_cfg=cfg, schedule=sched)
    assert info_r.selected_features == info_r2.selected_features and len(info_r.selected_features) == 3
    assert info_r.eps1 == 0 and info_r.eps2 == pytest.approx(3.2) and info_r.private
    assert all(c.spec.target_eps == pytest.approx(3.2) and c.X.shape[1] == 3 for c in cl_r)
    cl_v, info_v = pipeline.build_clients(pipeline.MethodConfig(selection="variance", **base), clients_data, ranges, C,
                                          seed=0, fl_cfg=cfg, schedule=sched)
    var = np.concatenate([X for X, _ in clients_data]).var(axis=0)
    assert info_v.selected_features == sorted(np.argsort(-var)[:3].tolist()) and not info_v.private
    assert info_v.eps1 == 0 and all(c.X.shape[1] == 3 for c in cl_v)
    with pytest.raises(ValueError):
        pipeline.build_clients(pipeline.MethodConfig(selection="bogus", **base), clients_data, ranges, C, seed=0,
                               fl_cfg=cfg, schedule=sched)


def test_selection_excludes_sensitive_columns(synthetic_federated):
    clients_data, (Xt, yt), ranges, C = synthetic_federated
    cfg = _fl_cfg(rounds=1, participation=1.0)
    base = dict(method="ldpmic", eps=4.0, k=3, allocation="oracle", share_mode="uniform", sensitive=(0, 1))
    _, info_a = pipeline.build_clients(pipeline.MethodConfig(**base), clients_data, ranges, C, seed=0, fl_cfg=cfg)
    _, info_b = pipeline.build_clients(pipeline.MethodConfig(exclude_sensitive=True, **base), clients_data, ranges, C, seed=0, fl_cfg=cfg)
    assert len(info_b.selected_features) == 3 and not ({0, 1} & set(info_b.selected_features))
    assert any("excluded from selection" in n for n in info_b.notes)
    if {0, 1} & set(info_a.selected_features):
        assert info_a.selected_features != info_b.selected_features


def test_input_rescale_is_postprocessing_of_the_release(synthetic_federated):
    """input_rescale leaves the released arrays byte-identical and divides the
    model inputs by sqrt(1 + (sigma_j/Delta_j)^2 / s^2); the test divisor is
    the same map with the m-weighted noise variance."""
    from ldpmic.data.common import public_scaler

    clients_data, _, ranges, C = synthetic_federated
    cfg = _fl_cfg()
    deltas = ranges[:, 1] - ranges[:, 0]
    sc = public_scaler(ranges)
    for method, extra in (("iso", {}), ("ldpmic", {"allocation": "global"}), ("ldpmic", {"allocation": "oracle"})):
        base = pipeline.MethodConfig(method=method, eps=5.0, **extra)
        resc = pipeline.MethodConfig(method=method, eps=5.0, input_rescale=0.25, **extra)
        c0, i0 = pipeline.build_clients(base, clients_data, ranges, C, seed=1, fl_cfg=cfg, transform=sc)
        c1, i1 = pipeline.build_clients(resc, clients_data, ranges, C, seed=1, fl_cfg=cfg, transform=sc)
        assert i0.sanitized_sha256 == i1.sanitized_sha256
        assert (i1.eps_realized, i1.delta_realized) == (i0.eps_realized, i0.delta_realized)
        var = np.zeros(len(deltas))
        for a, b in zip(c0, c1):
            f = np.sqrt(1.0 + (a.sig / deltas) ** 2 / 0.25 ** 2)
            np.testing.assert_allclose(b.X.numpy(), a.X.numpy() / f, rtol=1e-5, atol=1e-6)
            var += a.n_samples * (a.sig / deltas) ** 2
        M = sum(c.n_samples for c in c0)
        np.testing.assert_allclose(i1.input_rescale_test, np.sqrt(1.0 + var / M / 0.25 ** 2), rtol=1e-12)
        assert not i0.input_rescale_test


def test_group_sensitivity_calibration(synthetic_federated):
    """l2_sens_sq = S replaces d in the uniform release: sigma_j = Delta_j sigma0 sqrt(S), i.e.
    sum_j-in-support Delta_j^2 / sigma_j^2 over any S changed coordinates equals 1 / sigma0^2."""
    from ldpmic.privacy import sigma0

    clients_data, _, ranges, C = synthetic_federated
    d = ranges.shape[0]
    deltas = ranges[:, 1] - ranges[:, 0]
    S = float(d - 1)
    m = pipeline.MethodConfig(method="iso", eps=3.0, l2_sens_sq=S)
    clients, info = pipeline.build_clients(m, clients_data, ranges, C, seed=0, fl_cfg=_fl_cfg())
    s0 = sigma0(3.0, 1e-5, "analytic")
    for c in clients:
        np.testing.assert_allclose(c.sig, deltas * s0 * np.sqrt(S), rtol=1e-12)
        assert abs(S * (deltas[0] / c.sig[0]) ** 2 - 1.0 / s0 ** 2) < 1e-9
    with pytest.raises(ValueError):
        pipeline.build_clients(pipeline.MethodConfig(method="ldpmic", eps=3.0, l2_sens_sq=S), clients_data, ranges, C,
                               seed=0, fl_cfg=_fl_cfg())
    with pytest.raises(ValueError):
        pipeline.build_clients(pipeline.MethodConfig(method="iso", eps=3.0, l2_sens_sq=d + 1.0), clients_data, ranges, C,
                               seed=0, fl_cfg=_fl_cfg())
