"""Diversity selection for OPSD (ALGO.md Part 1 §5).

k-center greedy over two distance metrics:
  - unigram_jsd: fast, token-count based (method 1 in ALGO.md Part 1)
  - token_jsd:   distribution-based (method 2, recommended)
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


def kcenter(scores: list[float], distances: np.ndarray, n_select: int) -> list[int]:
    """k-center greedy given a precomputed symmetric [N, N] distance matrix.

    Seed: highest-quality candidate (argmax scores).
    Expand: argmax over min-distance to already-selected set.
    """
    selected = [int(np.argmax(scores))]
    selected_set = {selected[0]}

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


def global_topk(
    logits_local: torch.Tensor,
    top_k: int,
    shard_start: int,
    tp_group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k over a vocab-parallel logit shard.

    Each rank contributes its local top-k, then all ranks select the global top-k
    from the concatenated small candidate set. Returned indices are global vocab
    ids and are identical on every TP rank.
    """
    k_local = min(top_k, logits_local.size(-1))
    vals_local, idx_local = logits_local.topk(k_local, dim=-1)
    idx_local = idx_local + shard_start

    tp_size = dist.get_world_size(group=tp_group) if dist.is_available() and dist.is_initialized() else 1
    if tp_size > 1:
        vals_parts = [torch.empty_like(vals_local) for _ in range(tp_size)]
        idx_parts = [torch.empty_like(idx_local) for _ in range(tp_size)]
        dist.all_gather(vals_parts, vals_local.contiguous(), group=tp_group)
        dist.all_gather(idx_parts, idx_local.contiguous(), group=tp_group)
        vals_all = torch.cat(vals_parts, dim=-1)
        idx_all = torch.cat(idx_parts, dim=-1)
    else:
        vals_all, idx_all = vals_local, idx_local

    k = min(top_k, vals_all.size(-1))
    vals, pos = vals_all.topk(k, dim=-1)
    idx = idx_all.gather(-1, pos)
    return vals, idx


def gather_global_logits(
    logits_local: torch.Tensor,
    global_idx: torch.Tensor,
    shard_start: int,
    tp_group,
) -> torch.Tensor:
    """Gather arbitrary global vocab positions from vocab-parallel logits."""
    V_local = logits_local.size(-1)
    local_idx = global_idx - shard_start
    owned = (local_idx >= 0) & (local_idx < V_local)
    safe_idx = local_idx.clamp(0, V_local - 1)
    vals = logits_local.gather(-1, safe_idx)
    vals = vals.masked_fill(~owned, 0)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size(group=tp_group) > 1:
        dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=tp_group)
    return vals


def pairwise_seq_jsd(
    logits_list: list[torch.Tensor],
    top_k: int,
    device: torch.device,
    tp_group,
    shard_start: int,
    chunk_t: int,
) -> np.ndarray:
    """[N, N] matrix of mean-token-JSD distances for TP-local logits.

    logits_list contains [T, V_local] shards. The distance uses global top-k
    vocab ids, but only communicates those top-k ids/logits instead of
    materialising [T, V_full]. ``chunk_t < 0`` disables token-dim chunking.
    """
    n = len(logits_list)
    dist = np.zeros((n, n), dtype=np.float32)
    T_full = logits_list[0].size(0) if n else 0
    eff_chunk = T_full if chunk_t < 0 else chunk_t

    top_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
    for logits in logits_list:
        vals_chunks: list[torch.Tensor] = []
        idx_chunks: list[torch.Tensor] = []
        for t0 in range(0, logits.size(0), eff_chunk):
            t1 = min(t0 + eff_chunk, logits.size(0))
            l_gpu = logits[t0:t1].to(device, non_blocking=True)
            vals, idx = global_topk(l_gpu, top_k, shard_start, tp_group)
            vals_chunks.append(vals.detach().to("cpu"))
            idx_chunks.append(idx.detach().to("cpu"))
            del l_gpu, vals, idx
        top_cache.append((torch.cat(vals_chunks, dim=0), torch.cat(idx_chunks, dim=0)))

    for i in range(n):
        vi, ti = top_cache[i]
        for j in range(i + 1, n):
            vj, tj = top_cache[j]
            d = _seq_jsd(
                logits_list[i],
                logits_list[j],
                vi,
                ti,
                vj,
                tj,
                device,
                shard_start,
                tp_group,
                eff_chunk,
            )
            dist[i, j] = dist[j, i] = d
    return dist


# ── per-pair JSD helpers ──────────────────────────────────────────────────────


def _seq_jsd(
    logits_i: torch.Tensor,
    logits_j: torch.Tensor,
    top_vals_i: torch.Tensor,
    top_idx_i: torch.Tensor,
    top_vals_j: torch.Tensor,
    top_idx_j: torch.Tensor,
    device: torch.device,
    shard_start: int,
    tp_group,
    chunk_t: int,
) -> float:
    """Mean per-token JSD over the union of each sequence's global top-k ids."""
    total = 0.0
    count = 0
    T = logits_i.size(0)
    for t0 in range(0, T, chunk_t):
        t1 = min(t0 + chunk_t, T)
        li = logits_i[t0:t1].to(device, non_blocking=True)
        lj = logits_j[t0:t1].to(device, non_blocking=True)
        vi = top_vals_i[t0:t1].to(device, non_blocking=True)
        ti = top_idx_i[t0:t1].to(device, non_blocking=True)
        vj = top_vals_j[t0:t1].to(device, non_blocking=True)
        tj = top_idx_j[t0:t1].to(device, non_blocking=True)

        lp_i = F.log_softmax(vi.float(), dim=-1)  # [chunk, K]
        lp_j = F.log_softmax(vj.float(), dim=-1)
        pi, pj = lp_i.exp(), lp_j.exp()

        lv_j_at_i = F.log_softmax(gather_global_logits(lj, ti, shard_start, tp_group).float(), dim=-1)
        lv_i_at_j = F.log_softmax(gather_global_logits(li, tj, shard_start, tp_group).float(), dim=-1)
        pj_at_i, pi_at_j = lv_j_at_i.exp(), lv_i_at_j.exp()

        mix_i = 0.5 * (pi + pj_at_i)
        log_mix_i = mix_i.clamp(min=1e-10).log()
        jsd_i = 0.5 * ((pi * (lp_i - log_mix_i)).sum(-1) + (pj_at_i * (lv_j_at_i - log_mix_i)).sum(-1))

        mix_j = 0.5 * (pj + pi_at_j)
        log_mix_j = mix_j.clamp(min=1e-10).log()
        jsd_j = 0.5 * ((pj * (lp_j - log_mix_j)).sum(-1) + (pi_at_j * (lv_i_at_j - log_mix_j)).sum(-1))

        chunk_jsd = (0.5 * (jsd_i + jsd_j)).clamp(min=0)
        total += float(chunk_jsd.sum().item())
        count += int(chunk_jsd.numel())
        del li, lj, vi, ti, vj, tj, lp_i, lp_j, pi, pj, lv_j_at_i, lv_i_at_j, pj_at_i, pi_at_j

    return total / max(count, 1)


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
