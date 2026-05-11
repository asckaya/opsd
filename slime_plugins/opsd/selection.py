"""Diversity selection for OPSD (metho.md §5).

k-center greedy over two distance metrics:
  - unigram_jsd: fast, token-count based (method 1 in metho.md)
  - token_jsd:   distribution-based (method 2, recommended)
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def kcenter(scores: list[float], distances: np.ndarray, n_select: int) -> list[int]:
    """k-center greedy given a precomputed symmetric [N, N] distance matrix.

    Seed: highest-quality candidate (argmax scores).
    Expand: argmax over min-distance to already-selected set.
    """
    selected = [int(np.argmax(scores))]
    selected_set = {selected[0]}
    n = len(scores)

    # min_dist[i] = distance from i to its nearest selected center
    min_dist = distances[selected[0]].copy()

    while len(selected) < n_select:
        # Mask already-selected
        min_dist[list(selected_set)] = -1.0
        best = int(np.argmax(min_dist))
        selected.append(best)
        selected_set.add(best)
        # Update min-distances with the new center
        min_dist = np.minimum(min_dist, distances[best])

    return selected


def pairwise_unigram_jsd(tokens: list[list[int]]) -> np.ndarray:
    """[N, N] matrix of unigram-JSD distances between token sequences."""
    n = len(tokens)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = _unigram_jsd(tokens[i], tokens[j])
            dist[i, j] = dist[j, i] = d
    return dist


def pairwise_seq_jsd(
    logits_list: list[torch.Tensor],
    top_k: int,
    device: torch.device,
) -> np.ndarray:
    """[N, N] matrix of mean-token-JSD distances.

    logits_list: list of [T, V] tensors (CPU); pairs are moved to device one at a time
    so peak GPU usage is 2 × [T, V] instead of [N, T, V].
    """
    n = len(logits_list)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        li = logits_list[i].to(device)
        for j in range(i + 1, n):
            lj = logits_list[j].to(device)
            d = _seq_jsd(li, lj, top_k)
            dist[i, j] = dist[j, i] = d
            del lj
        del li
    return dist


# ── per-pair JSD helpers ──────────────────────────────────────────────────────


def _seq_jsd(logits_i: torch.Tensor, logits_j: torch.Tensor, top_k: int) -> float:
    """Mean per-token JSD between two logit sequences [T, V].

    Works in the union of each sequence's top-k vocabulary:
    evaluates p_i at top-k(j) and p_j at top-k(i), then averages
    two half-JSD terms so neither distribution loses mass.
    """
    k = min(top_k, logits_i.size(-1))

    tv_i, ti_i = logits_i.topk(k, dim=-1)  # [T, K]
    tv_j, ti_j = logits_j.topk(k, dim=-1)

    lp_i = F.log_softmax(tv_i, dim=-1)     # [T, K]
    lp_j = F.log_softmax(tv_j, dim=-1)
    pi, pj = lp_i.exp(), lp_j.exp()

    # Cross-evaluate: p_i over top-k(j) and vice-versa
    lv_j_at_i = F.log_softmax(logits_j.gather(-1, ti_i), dim=-1)   # [T, K]
    lv_i_at_j = F.log_softmax(logits_i.gather(-1, ti_j), dim=-1)   # [T, K]
    pj_at_i, pi_at_j = lv_j_at_i.exp(), lv_i_at_j.exp()

    # JSD over top-k(i): p_i vs p_j_at_i
    mix_i = 0.5 * (pi + pj_at_i)
    log_mix_i = mix_i.clamp(min=1e-10).log()
    jsd_i = 0.5 * (
        (pi * (lp_i - log_mix_i)).sum(-1) + (pj_at_i * (lv_j_at_i - log_mix_i)).sum(-1)
    )

    # JSD over top-k(j): p_j vs p_i_at_j
    mix_j = 0.5 * (pj + pi_at_j)
    log_mix_j = mix_j.clamp(min=1e-10).log()
    jsd_j = 0.5 * (
        (pj * (lp_j - log_mix_j)).sum(-1) + (pi_at_j * (lv_i_at_j - log_mix_j)).sum(-1)
    )

    return float((0.5 * (jsd_i + jsd_j)).clamp(min=0).mean().item())


def _unigram_jsd(tokens_i: Iterable[int], tokens_j: Iterable[int]) -> float:
    """Jensen-Shannon divergence between unigram distributions of two token sequences."""
    ti, tj = list(tokens_i), list(tokens_j)
    ci, cj = Counter(ti), Counter(tj)
    ni, nj = len(ti), len(tj)
    jsd = 0.0
    for tok in ci.keys() | cj.keys():
        p, q = ci.get(tok, 0) / ni, cj.get(tok, 0) / nj
        m = 0.5 * (p + q)
        if p > 0:
            jsd += 0.5 * p * math.log(p / m)
        if q > 0:
            jsd += 0.5 * q * math.log(q / m)
    return jsd
