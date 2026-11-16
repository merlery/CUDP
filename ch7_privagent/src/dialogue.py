"""Prompt sets of PrivAgent, the simulated dialogue, and the consistency check with repair.

A prompt is (pair, contrast). pair indexes PAIRS: 0 = LP-LA, 1 = LP-CR, 2 = LA-CR (criteria 0 = LP,
1 = LA, 2 = CR). Option A favors the first criterion of the pair: (m_j, m_j') = (h, l); option B has
(l, h); the third criterion is 0.5 in both. Answer 1 = the user chose A.
"""
from __future__ import annotations

import numpy as np

from .users import choice_prob

PAIRS = [(0, 1), (0, 2), (1, 2)]
PAIR_NAMES = ["LP-LA", "LP-CR", "LA-CR"]
CONTRAST_NAMES = ["strong", "medium", "weak"]


def contrast_table(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = cfg["prompts"]["contrasts"]
    h = np.array([c[k]["h"] for k in CONTRAST_NAMES])
    l = np.array([c[k]["l"] for k in CONTRAST_NAMES])
    kappa = np.array([c[k]["kappa"] for k in CONTRAST_NAMES])
    return h, l, kappa


def prompt_set(cfg: dict, name: str) -> np.ndarray:
    """Array (P, 2) of (pair index, contrast index)."""
    out = []
    for pair, contrast in cfg["prompts"]["sets"][name]:
        out.append((PAIR_NAMES.index(pair), CONTRAST_NAMES.index(contrast)))
    return np.array(out, dtype=int)


def pair_results(pairs: np.ndarray, answers: np.ndarray) -> np.ndarray:
    """Per pair: +1 if both answers favor the first criterion, -1 if both favor the second, 0 if split.
    pairs, answers: (N, P) for the six PrivAgent-6 prompts (two per pair)."""
    N = pairs.shape[0]
    res = np.zeros((N, 3), dtype=int)
    for q in range(3):
        m = pairs == q
        n_a = (answers * m).sum(axis=1)          # answers favoring the first criterion
        n_tot = m.sum(axis=1)
        res[:, q] = np.where(n_a == n_tot, 1, np.where(n_a == 0, -1, 0))
    return res


def is_cycle(res: np.ndarray) -> np.ndarray:
    """Cycle: LP > LA, LA > CR, CR > LP  -> (LP-LA, LP-CR, LA-CR) = (+1, -1, +1), or the reverse
    (-1, +1, -1)."""
    a = (res[:, 0] == 1) & (res[:, 1] == -1) & (res[:, 2] == 1)
    b = (res[:, 0] == -1) & (res[:, 1] == 1) & (res[:, 2] == -1)
    return a | b


def repair_pair(res: np.ndarray) -> np.ndarray:
    """First split pair in the order (LP-LA, LP-CR, LA-CR); if no pair is split, the first pair (LP-LA).
    (A cycle needs three decided pairs, so in a cycle no pair is split; see config.yaml.)"""
    split = res == 0
    first_split = np.argmax(split, axis=1)
    return np.where(split.any(axis=1), first_split, 0)


def run(W: np.ndarray, cfg: dict, set_name: str, tau: float, rng: np.random.Generator,
        repair: bool) -> dict:
    """Simulate the dialogue for all users. Draw order in rng: (1) a random prompt order per user,
    (2) uniform draws for the P answers (in presented order), (3) one uniform draw for the repair answer.
    Returns arrays with P + 1 slots (last slot = repair prompt, used only when triggered)."""
    N = W.shape[0]
    base = prompt_set(cfg, set_name)
    P = base.shape[0]
    order = rng.permuted(np.tile(np.arange(P), (N, 1)), axis=1)
    u = rng.random((N, P))
    u_rep = rng.random(N)
    h_tab, l_tab, k_tab = contrast_table(cfg)
    pair = base[order, 0]
    con = base[order, 1]
    ans = np.zeros((N, P), dtype=np.int8)
    for q in range(3):
        for c in range(3):
            m = (pair == q) & (con == c)
            if not m.any():
                continue
            p = choice_prob(W, PAIRS[q], h_tab[c], l_tab[c], tau)        # (N,)
            ans[m] = (u[m] < np.broadcast_to(p[:, None], (N, P))[m]).astype(np.int8)
    pair_all = np.concatenate([pair, np.zeros((N, 1), int)], axis=1)
    con_all = np.concatenate([con, np.zeros((N, 1), int)], axis=1)
    ans_all = np.concatenate([ans, np.zeros((N, 1), np.int8)], axis=1)
    used = np.concatenate([np.ones((N, P), bool), np.zeros((N, 1), bool)], axis=1)
    triggered = np.zeros(N, bool)
    if repair:
        res = pair_results(pair, ans)
        triggered = is_cycle(res)
        rp = repair_pair(res)
        strong = CONTRAST_NAMES.index(cfg["prompts"]["repair"]["contrast"])
        pair_all[:, P] = rp
        con_all[:, P] = strong
        pr = np.empty(N)
        for q in range(3):
            m = rp == q
            pr[m] = choice_prob(W[m], PAIRS[q], h_tab[strong], l_tab[strong], tau)
        ans_all[:, P] = (u_rep < pr).astype(np.int8)
        used[:, P] = triggered
    return dict(pair=pair_all, contrast=con_all, answer=ans_all, used=used, repair=triggered, n_base=P)


def truncate(d: dict, k: int) -> dict:
    """First k prompts of the presented order, no repair step."""
    used = d["used"].copy()
    used[:, k:] = False
    return dict(d, used=used, repair=np.zeros_like(d["repair"]))
