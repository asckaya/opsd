"""Training-side OPSD logic (ALGO.md Part 1 §4-9).

Steps executed inside the loss function:
  §4  Add conf term to quality scores; apply TopK_b filter.
  §5  Select N diverse traces (k-center greedy).
  §6  Teacher forward: π_θ(·| x, τ_k, y_{<t}).
  §7  Compute mixture weights w_k^t.
  §8  Mixture teacher q_mix^t = Σ_k w_k^t q_k^t.
  §9  KL distillation loss.  Returns per-token KL [T_i] per sample;
      reduction (sample-mean → batch-sum) is handled by slime's
      `sum_of_sample_mean` in plugin.py to integrate cleanly with
      Megatron's gradient accumulation.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _is_rank0() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


from slime.backends.megatron_utils.loss import get_responses

from .selection import gather_global_logits, global_topk, kcenter, pairwise_seq_jsd, pairwise_unigram_jsd

_TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step -- do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)
_BOXED_INSTRUCTION = "Please reason step by step and put your final answer within \\boxed{}."


# ── teacher input construction ────────────────────────────────────────────────


def build_teacher_inputs(
    batch: dict,
    metadata_list: list[dict],
    unconcat_tokens: list[torch.Tensor],
    tokenizer,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[int],
    list[list[float]],
    list[list[list[int]]],
]:
    """Build teacher input sequences for every privileged candidate.

    Two parallel sequences are constructed per trace, sharing the same flat index:
      • q_inputs[i*n+k]:    [chat(privileged_prompt(x, τ_k)) + y] — for q_k^t (§6).
      • conf_inputs[i*n+k]: [chat(x) + τ_k]                       — for Conf(τ_k) (§4).

    Returns:
        q_inputs:          flat list of [priv_prompt + response] tensors (for §6).
        conf_inputs:       flat list of [chat(problem) + τ_k] tensors (for §4 Conf).
        trace_tokens_flat: per-trace τ_k as long-tensors on device (for §4 Conf).
        counts:            number of candidates per sample (0 if no candidates).
        cand_scores:       structural quality scores per sample.
        cand_tokens:       privileged trace token lists per sample.
    """
    q_inputs: list[torch.Tensor] = []
    conf_inputs: list[torch.Tensor] = []
    trace_tokens_flat: list[torch.Tensor] = []
    counts: list[int] = []
    cand_scores: list[list[float]] = []
    cand_tokens: list[list[list[int]]] = []
    device = unconcat_tokens[0].device

    # Same rollout group shares the privileged-candidates list (identical object
    # across n_samples_per_prompt samples — see rollout.py). Encoding the chat
    # template per (sample, trace) wastes 16× the work; cache by list identity.
    bare_cache: dict[int, list[int]] = {}
    priv_cache: dict[int, list[list[int]]] = {}

    for i, meta in enumerate(metadata_list):
        traces: list[list[int]] = meta["privileged_candidates"]
        scores: list[float] = meta["privileged_candidate_scores"]
        problem: str = meta["problem_text"]

        if not traces:
            counts.append(0)
            cand_scores.append([])
            cand_tokens.append([])
            continue

        resp_tok_list = unconcat_tokens[i][-batch["response_lengths"][i] :].tolist()
        group_id = id(traces)

        bare_prompt_ids = bare_cache.get(group_id)
        if bare_prompt_ids is None:
            bare_prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=True,
                add_generation_prompt=True,
            )
            bare_cache[group_id] = bare_prompt_ids

        priv_ids_list = priv_cache.get(group_id)
        if priv_ids_list is None:
            priv_ids_list = []
            for trace in traces:
                priv_prompt = _build_privileged_prompt(problem, trace, tokenizer)
                priv_ids_list.append(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": priv_prompt}],
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )
            priv_cache[group_id] = priv_ids_list

        for k, trace in enumerate(traces):
            q_inputs.append(torch.tensor(priv_ids_list[k] + resp_tok_list, device=device))
            conf_inputs.append(torch.tensor(bare_prompt_ids + list(trace), device=device))
            trace_tokens_flat.append(torch.tensor(trace, dtype=torch.long, device=device))

        counts.append(len(traces))
        cand_scores.append(scores)
        cand_tokens.append(traces)

    return q_inputs, conf_inputs, trace_tokens_flat, counts, cand_scores, cand_tokens


def _build_privileged_prompt(problem: str, trace_tokens: list[int], tokenizer) -> str:
    reference = tokenizer.decode(trace_tokens, skip_special_tokens=True)
    return f"Problem: {problem}\n\nHere is a reference solution:\n=== Begin ===\n{reference}\n=== End ==={_TRANSITION_PROMPT}{_BOXED_INSTRUCTION}"


# ── teacher forward (§6) ─────────────────────────────────────────────────────


def teacher_forward(
    model: torch.nn.Module,
    inputs: list[torch.Tensor],
    resp_len: int,
) -> list[torch.Tensor]:
    """Run π_θ(·| x, τ_k, y_{<t}) one trace at a time; return TP-local logits.

    Returns vocab-parallel teacher logits [resp_len, V_local] per input, on the
    same device as the model. q_k^t at position t is taken from logit position
    prompt_len + t - 1 (causal-LM convention: output[i] predicts input[i+1]),
    matching the slicing used by slime's own `get_responses`
    (`logits[start-1 : end-1]`).
    """
    from megatron.core.parallel_state import get_tensor_model_parallel_world_size

    tp_size = get_tensor_model_parallel_world_size()
    teacher_logits: list[torch.Tensor] = []

    for inp in inputs:
        orig_len = inp.size(0)
        # Sequence parallel requires seq_len % tp_size == 0.  Right-pad with zeros;
        # causal attention means real positions are unaffected by the tail padding.
        pad_len = (-orig_len) % tp_size
        inp_model = F.pad(inp, (0, pad_len)) if pad_len else inp

        with torch.no_grad():
            # fp32_output=False keeps model output in bf16/fp16; upcasting the full
            # [1, T, V_local] tensor would double peak memory and OOMs on long T.
            out = model(
                input_ids=inp_model.unsqueeze(0),
                position_ids=None,
                attention_mask=None,
                labels=None,
                fp32_output=False,
            )
        del inp_model
        raw = out[0] if isinstance(out, tuple) else out
        # output[i] predicts input[i+1]; response positions [P, P+T) <- logits [P-1, P+T-1).
        # P = orig_len - resp_len, so the slice is [orig_len - resp_len - 1, orig_len - 1).
        # Keep vocab-parallel logits. Gathering [T, V_full] is the dominant OPSD
        # OOM source at long response lengths, and downstream code is TP-aware.
        logit_t = raw[0, orig_len - resp_len - 1 : orig_len - 1].contiguous()  # [T, V_local]
        del out, raw

        teacher_logits.append(logit_t)

    return teacher_logits


# ── Conf(τ) forward (§4) ─────────────────────────────────────────────────────


def compute_trace_confs(
    model: torch.nn.Module,
    conf_inputs: list[torch.Tensor],
    trace_tokens_flat: list[torch.Tensor],
) -> list[float]:
    """Conf(τ_k) = (1/|τ_k|) Σ_t log π_T(τ_k[t] | x, τ_k[<t]) — ALGO.md Part 1 §4.

    One forward pass per trace over `chat(x) + τ_k`. Reads logits at positions
    [P-1, P+|τ|-1) where P = len(chat(x)), then evaluates the log-prob of
    τ_k[t] at each position.
    """
    from megatron.core.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    tp_size = get_tensor_model_parallel_world_size()
    tp_group = get_tensor_model_parallel_group()
    tp_rank = get_tensor_model_parallel_rank()
    confs: list[float] = []

    for inp, trace_t in zip(conf_inputs, trace_tokens_flat, strict=True):
        orig_len = inp.size(0)
        trace_len = trace_t.size(0)
        pad_len = (-orig_len) % tp_size
        inp_model = F.pad(inp, (0, pad_len)) if pad_len else inp

        with torch.no_grad():
            out = model(
                input_ids=inp_model.unsqueeze(0),
                position_ids=None,
                attention_mask=None,
                labels=None,
                fp32_output=False,
            )
        del inp_model
        raw = out[0] if isinstance(out, tuple) else out
        logit_t = raw[0, orig_len - trace_len - 1 : orig_len - 1].float()  # [trace_len, V_local]
        del out, raw

        lse = vocab_parallel_lse_chunk(logit_t, tp_group).squeeze(-1)
        V_local = logit_t.size(-1)
        shard_start = tp_rank * V_local
        at_trace = gather_global_logits(logit_t, trace_t.unsqueeze(-1), shard_start, tp_group).squeeze(-1)
        confs.append((at_trace - lse).mean().item())
        del logit_t, lse, at_trace

    return confs


# ── quality: conf term (§4) ──────────────────────────────────────────────────


def add_conf(
    args,
    base_scores: list[float],
    confs: list[float],
) -> list[float]:
    """Add η_c·Conf(τ) to structural quality scores.

    Conf(τ) = (1/|τ|) Σ_t log π_T(τ_t | x, τ_<t) — the teacher's mean log-prob
    on the trace itself, pre-computed by `compute_trace_confs` per ALGO.md Part 1 §4.

    Raw Conf is in log-prob units (typically -5..-1 nats), incommensurable with
    the structural terms which live in [0, 1]. We normalize Conf across the
    candidates of a single sample so η_c becomes dimensionless and easier to
    tune. Modes:

      - raw    : no normalization (legacy behavior).
      - zscore : (c - mean) / std         — centered, unit variance.
      - minmax : (c - min) / (max - min)  — mapped to [0, 1].
      - rank   : argsort-rank / (n - 1)   — robust to outliers, in [0, 1].
    """
    if args.opsd_quality_conf_weight == 0.0:
        return base_scores
    w = args.opsd_quality_conf_weight
    mode = args.opsd_quality_conf_norm
    n = len(confs)
    if mode == "raw" or n <= 1:
        norm = list(confs)
    elif mode == "zscore":
        arr = np.asarray(confs, dtype=np.float64)
        std = float(arr.std())
        norm = ((arr - arr.mean()) / (std if std > 1e-8 else 1.0)).tolist()
    elif mode == "minmax":
        arr = np.asarray(confs, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        norm = ((arr - lo) / (hi - lo)).tolist() if hi - lo > 1e-8 else [0.5] * n
    elif mode == "rank":
        order = np.argsort(np.asarray(confs))
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        norm = (ranks / (n - 1)).tolist()
    else:
        raise ValueError(f"Unknown opsd_quality_conf_norm: {mode}")
    return [b + w * c for b, c in zip(base_scores, norm, strict=True)]


# ── quality: TopK_b filter (§4) ──────────────────────────────────────────────
#
# The TopK_b filter is applied inline inside distillation_loss (between the
# Conf forward and the q-teacher forward) so we can skip q-forward for the
# discarded candidates.  See `distillation_loss` for the inline implementation.


# ── diversity selection (§5) ─────────────────────────────────────────────────


def select_diverse(
    args,
    scores: list[float],
    tokens: list[list[int]],
    logits_cpu: list[torch.Tensor] | None,
    device: torch.device,
    tp_group,
    shard_start: int,
) -> list[int]:
    """Select N diverse candidates via k-center greedy (ALGO.md Part 1 §5).

    `logits_cpu` is required only for the token_jsd metric; it may be None when
    the metric is unigram_jsd (token-only distance, no q logits needed).
    """
    n_select = min(args.opsd_n, len(scores))
    if len(scores) <= n_select:
        return list(range(len(scores)))

    if args.opsd_diversity_metric == "token_jsd":
        assert logits_cpu is not None, "token_jsd diversity requires teacher logits"
        dist = pairwise_seq_jsd(
            logits_cpu, args.opsd_diversity_top_k, device, tp_group, shard_start, args.opsd_kl_chunk
        )
    else:
        dist = pairwise_unigram_jsd(tokens)

    return kcenter(scores, dist, n_select)


# ── mixture weights (§7) ─────────────────────────────────────────────────────


def mixture_weights(
    args,
    p_gathered: torch.Tensor,
    q_gathered: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token mixture weights w_k^t (ALGO.md Part 1 §7).

    w_k^t ∝ exp(-β·Δ_k^t - γ·h_k^t + ρ·g_k^t)

    where:
      Δ_k^t = KL(q_k^t ‖ p^t)        — KL term, down-weights confusing teachers
      h_k^t = H(q_k^t)                — entropy, down-weights uncertain teachers
      g_k^t = mean_j JSD(q_k^t, q_j^t) — diversity bonus

    All computed on the student's top-K vocabulary for efficiency.

    Args:
        p_gathered:  [T, K] student logits at global top-K positions.
        q_gathered:  [N, T, K] teacher logits at the same global top-K positions.

    Returns:
        w:                [N, T] softmax-normalised mixture weights.
        w_entropy_norm:   [T] per-token normalised entropy H(w_·^t) / log N in
                          [0, 1].  1 ⇒ uniform over teachers (N traces look
                          equivalent — diversity selection contributed nothing);
                          0 ⇒ a single teacher dominates.  Aggregating this over
                          the batch makes a useful health metric.
    """
    N = q_gathered.size(0)

    p_log = F.log_softmax(p_gathered / args.opsd_temperature, dim=-1)  # [T, K]
    q_log = F.log_softmax(q_gathered / args.opsd_temperature, dim=-1)  # [N, T, K]
    q_prob = q_log.exp()

    delta = (q_prob * (q_log - p_log.unsqueeze(0))).sum(-1).clamp(min=0)  # [N, T]
    entropy = -(q_prob * q_log).sum(-1)  # [N, T]
    diversity = _pairwise_jsd(q_prob, q_log) if N > 1 else torch.zeros(N, p_log.size(0), device=p_gathered.device)

    logits = -args.opsd_kl_weight * delta - args.opsd_entropy_weight * entropy + args.opsd_diversity_weight * diversity
    w = F.softmax(logits, dim=0)  # [N, T]

    if N > 1:
        log_w = F.log_softmax(logits, dim=0)  # numerically stable
        # Parenthesize the negation explicitly: Python's unary `-` has lower
        # precedence than `.sum().clamp()`, so `-(w * log_w).sum(0).clamp(min=0)`
        # parses as `-((w*log_w).sum(0).clamp(min=0))`. Since `w*log_w ≤ 0`,
        # the .clamp(min=0) before the negation collapses to 0, then `-0 = -0.0`.
        # The shape we want is `H(w) = -Σ w log w ≥ 0`, with clamp as a numerical
        # guard AFTER the negation.
        w_entropy = (-(w * log_w).sum(0)).clamp(min=0)  # [T] in nats
        w_entropy_norm = w_entropy / math.log(N)
    else:
        w_entropy_norm = torch.zeros(p_log.size(0), device=p_gathered.device)

    if _is_rank0():
        wn = w_entropy_norm.detach()
        T = wn.numel()
        nz = (wn > 1e-8).float().mean().item()

        # Per-term stats across N teachers, then averaged over T tokens. The
        # "spread" lines tell us which term dominates the softmax: if any of
        # them has spread >> 1 across teachers, softmax saturates to one-hot.
        d = delta.detach()
        e = entropy.detach()
        g = diversity.detach() if N > 1 else torch.zeros_like(d)
        lg = logits.detach()
        spread_d = (d.max(0).values - d.min(0).values).mean().item()
        spread_e = (e.max(0).values - e.min(0).values).mean().item()
        spread_g = (g.max(0).values - g.min(0).values).mean().item()
        spread_lg = (lg.max(0).values - lg.min(0).values).mean().item()
        logger.info(
            "[opsd_dbg/mixture] N=%d T=%d temp=%.2f "
            "weights(kl=%.2f,ent=%.2f,div=%.2f) "
            "delta[mean=%.3f,max=%.3f,spread/t=%.3f] "
            "entropy[mean=%.3f,max=%.3f,spread/t=%.3f] "
            "diversity[mean=%.3f,max=%.3f,spread/t=%.3f] "
            "logits_spread/t=%.3f "
            "w_entropy_norm[mean=%.4f,max=%.4f,frac_nonzero=%.3f]",
            N,
            T,
            args.opsd_temperature,
            args.opsd_kl_weight,
            args.opsd_entropy_weight,
            args.opsd_diversity_weight,
            d.mean().item(),
            d.max().item(),
            spread_d,
            e.mean().item(),
            e.max().item(),
            spread_e,
            g.mean().item(),
            g.max().item(),
            spread_g,
            spread_lg,
            wn.mean().item(),
            wn.max().item(),
            nz,
        )

    return w, w_entropy_norm


def _pairwise_jsd(q_prob: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    """Per-token mean pairwise JSD: g_k^t = (1/(N-1)) Σ_{j≠k} JSD(q_k^t, q_j^t).

    Args:
        q_prob, q_log: [N, T, K]

    Returns:
        [N, T]

    Iterates over the N(N-1)/2 unique pairs instead of broadcasting to
    [N, N, T, K] — the broadcast form is ~N² × [T, K] in peak memory which
    blows past 1 GB at N=4, T=16k, K=512.
    """
    N, T, _ = q_prob.shape
    sum_jsd = torch.zeros(N, T, device=q_prob.device, dtype=q_prob.dtype)
    for i in range(N):
        pi = q_prob[i]
        lpi = q_log[i]
        for j in range(i + 1, N):
            pj = q_prob[j]
            lpj = q_log[j]
            log_mix = (0.5 * (pi + pj)).clamp(min=1e-10).log()
            jsd_ij = 0.5 * ((pi * (lpi - log_mix)).sum(-1) + (pj * (lpj - log_mix)).sum(-1)).clamp(min=0)  # [T]
            sum_jsd[i].add_(jsd_ij)
            sum_jsd[j].add_(jsd_ij)
    return sum_jsd / (N - 1)  # [N, T]


# ── vocab-parallel KL divergence (§9) ───────────────────────────────────────


def vocab_parallel_lse_chunk(c: torch.Tensor, tp_group) -> torch.Tensor:
    """Numerically-stable distributed log-sum-exp over the last dim of ``c``.

    Returns ``[c.shape[0], 1]``.  When ``tp_size == 1`` the all-reduces are
    no-ops (skipped) so this collapses to a plain local LSE.
    """
    tp_size = dist.get_world_size(group=tp_group)
    gmax = c.max(dim=-1, keepdim=True).values
    if tp_size > 1:
        dist.all_reduce(gmax, op=dist.ReduceOp.MAX, group=tp_group)
    gsum = (c - gmax).exp().sum(dim=-1, keepdim=True)
    if tp_size > 1:
        dist.all_reduce(gsum, op=dist.ReduceOp.SUM, group=tp_group)
    return gmax + gsum.log()


def vocab_parallel_softmax_chunk(c: torch.Tensor, tp_group) -> torch.Tensor:
    """Softmax over a vocab-parallel logit shard."""
    return (c - vocab_parallel_lse_chunk(c.float(), tp_group)).exp()


def _accumulate_q_mix_chunk(
    teacher_logits: tuple[torch.Tensor, ...],
    w: torch.Tensor,
    t0: int,
    t1: int,
    V_local: int,
    device: torch.device,
    tp_group,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Build q_mix^{t0:t1} = Σ_k w_k · softmax(q_k) on the fly.

    Streams the N teacher chunks CPU→GPU one at a time and folds each into
    a single fp32 accumulator via ``addcmul_`` — no intermediate ``s_k * w_k``
    tensor. ``vocab_parallel_softmax_chunk`` already upcasts inside its LSE,
    so bf16 input avoids an extra .float() temporary on the caller side.
    """
    N = len(teacher_logits)
    q_chunk = torch.zeros(t1 - t0, V_local, dtype=torch.float32, device=device)
    for k in range(N):
        tk = teacher_logits[k][t0:t1].to(device, non_blocking=True)
        s_k = vocab_parallel_softmax_chunk(tk, tp_group)  # fp32 [c, V_local]
        q_chunk.addcmul_(s_k, w[k, t0:t1].unsqueeze(-1).float())
        del tk, s_k
    return q_chunk.to(out_dtype)


class VocabParallelMixtureKLDiv(torch.autograd.Function):
    """Forward-KL(q_mix ‖ p_θ) with q_mix fused into the autograd op.

    ``q_mix^t_v = Σ_k w_k^t · softmax(q_k^t)_v`` is **never** materialised as
    a [T, V_local] persistent buffer. Both forward and backward stream the N
    teacher logit shards once and recompute the chunk-sized q_mix on the fly.
    This eliminates the dominant persistent OPSD GPU buffer at long response
    lengths (~5–10 GB at T=16k, V=152k fp32, depending on TP).

    Saved between forward and backward:
      • ``logit_local``        — student, already an autograd input
      • ``lse``                — [T]
      • ``s_or_empty``         — [T] when clipping; empty sentinel otherwise
      • ``w``                  — [N, T] mixture weights
      • ``*teacher_logits``    — N shards [T, V_local], may live on CPU

    Teacher tensors are passed as positional args so save_for_backward retains
    references without copying (the caller's ``sample_teacher_outputs`` keeps
    them alive on CPU; chunks are pulled to GPU once per pass).

    Per-(position, vocab-entry) KL clipping (paper §3.2 / Figure 4):
      ℓ_{n,v} = q_v · (log q_v − log p_v) clipped to ``clip``. Backward becomes
      ``s · softmax_u − mask_u · q_u`` with ``mask = (ℓ < clip)`` and
      ``s = Σ_v mask_v · q_v`` (TP all-reduced).
    """

    @staticmethod
    def forward(
        ctx,
        logit_local: torch.Tensor,  # [T, V_local], requires_grad
        w: torch.Tensor,  # [N, T], detached mixture weights
        tp_group,
        chunk_t: int,
        clip: float | None,
        *teacher_logits: torch.Tensor,  # N shards of [T, V_local]; CPU or GPU
    ) -> torch.Tensor:  # [T]
        T, V_local = logit_local.shape
        N = len(teacher_logits)
        assert w.shape == (N, T), f"w shape {tuple(w.shape)} != ({N}, {T})"
        if chunk_t < 0:
            chunk_t = T
        tp_size = dist.get_world_size(group=tp_group)
        clip_enabled = clip is not None
        device = logit_local.device
        out_dtype = logit_local.dtype

        lse = logit_local.new_empty(T)
        kl_local = logit_local.new_zeros(T)
        s_local = logit_local.new_zeros(T) if clip_enabled else None

        for t0 in range(0, T, chunk_t):
            t1 = min(t0 + chunk_t, T)
            qc = _accumulate_q_mix_chunk(teacher_logits, w, t0, t1, V_local, device, tp_group, out_dtype)
            lc = logit_local[t0:t1]
            lse_chunk = vocab_parallel_lse_chunk(lc, tp_group)  # [c, 1]
            lse[t0:t1] = lse_chunk.squeeze(-1)
            log_p = lc - lse_chunk  # [c, V_local]
            del lse_chunk
            # Build per_entry = qc * (log qc - log p) with in-place chaining so
            # only one [c, V_local] tensor (`buf`, aliasing the clamp result) is
            # alive on top of qc + log_p.  At chunk_t=T this triples headroom
            # compared to the naive 3-op compositional form.
            buf = qc.clamp(min=1e-10).log_().sub_(log_p).mul_(qc)
            del log_p
            if clip_enabled:
                assert s_local is not None
                mask = buf < clip
                # s = Σ_v mask_v · q_v (TP-summed later)
                s_local[t0:t1] = (mask.to(qc.dtype) * qc).sum(-1)
                buf = torch.where(mask, buf, buf.new_full((), clip))
                del mask
            kl_local[t0:t1] = buf.sum(-1)
            del qc, buf

        if tp_size > 1:
            dist.all_reduce(kl_local, op=dist.ReduceOp.SUM, group=tp_group)
            if clip_enabled:
                assert s_local is not None
                dist.all_reduce(s_local, op=dist.ReduceOp.SUM, group=tp_group)

        ctx.save_for_backward(
            logit_local,
            lse,
            s_local if clip_enabled else logit_local.new_empty(0),
            w,
            *teacher_logits,
        )
        ctx.tp_group = tp_group
        ctx.chunk_t = chunk_t
        ctx.clip_enabled = clip_enabled
        ctx.clip = clip
        ctx.n_teachers = N
        return kl_local

    @staticmethod
    def backward(ctx, grad_kl: torch.Tensor):
        saved = ctx.saved_tensors
        logit_local, lse, s_local = saved[0], saved[1], saved[2]
        w = saved[3]
        teacher_logits = saved[4:]
        tp_group = ctx.tp_group
        chunk_t = ctx.chunk_t
        clip_enabled = ctx.clip_enabled
        T, V_local = logit_local.shape
        device = logit_local.device
        out_dtype = logit_local.dtype

        grad = torch.empty_like(logit_local)
        for t0 in range(0, T, chunk_t):
            t1 = min(t0 + chunk_t, T)
            qc = _accumulate_q_mix_chunk(teacher_logits, w, t0, t1, V_local, device, tp_group, out_dtype)
            log_p = logit_local[t0:t1] - lse[t0:t1].unsqueeze(-1)  # [c, V_local]
            scale = grad_kl[t0:t1].unsqueeze(-1)
            if clip_enabled:
                # mask depends on per-entry pre-clip value; recompute it from
                # (qc, log_p) to avoid persisting the [c, V_local] mask between
                # forward and backward.
                per_entry = qc.clamp(min=1e-10).log().sub_(log_p).mul_(qc)
                mask_q = (per_entry < ctx.clip).to(qc.dtype).mul_(qc)
                sm_c = log_p.exp_()
                grad[t0:t1] = sm_c.mul_(s_local[t0:t1].unsqueeze(-1)).sub_(mask_q).mul_(scale)
                del per_entry, mask_q, sm_c
            else:
                # grad = scale * (softmax(p) - q_mix); fused in-place on log_p
                sm_c = log_p.exp_()
                grad[t0:t1] = sm_c.sub_(qc).mul_(scale)
                del sm_c
            del qc, log_p

        return (grad, None, None, None, None, *([None] * ctx.n_teachers))


class VocabParallelMixtureRKLDiv(torch.autograd.Function):
    """Reverse-KL: KL(p_θ ‖ q_mix), q_mix fused into the autograd op.

    Same fusing strategy as ``VocabParallelMixtureKLDiv``: q_mix never persists
    between forward and backward; both passes recompute it per chunk.

    Per position n:
        L_n = Σ_v p_v · (log p_v − log q_v)
        d L_n / d(logit_u) = p_u · ((log p_u − log q_u) − L_n)

    Used as ALGO.md Part 1 §9's optional auxiliary term ``L_RKL`` (α_RKL ≪ 1).
    """

    @staticmethod
    def forward(
        ctx,
        logit_local: torch.Tensor,
        w: torch.Tensor,
        tp_group,
        chunk_t: int,
        *teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        T, V_local = logit_local.shape
        N = len(teacher_logits)
        assert w.shape == (N, T), f"w shape {tuple(w.shape)} != ({N}, {T})"
        if chunk_t < 0:
            chunk_t = T
        tp_size = dist.get_world_size(group=tp_group)
        device = logit_local.device
        out_dtype = logit_local.dtype

        lse = logit_local.new_empty(T)
        kl_local = logit_local.new_zeros(T)

        for t0 in range(0, T, chunk_t):
            t1 = min(t0 + chunk_t, T)
            qc = _accumulate_q_mix_chunk(teacher_logits, w, t0, t1, V_local, device, tp_group, out_dtype)
            lc = logit_local[t0:t1]
            lse_chunk = vocab_parallel_lse_chunk(lc, tp_group)
            lse[t0:t1] = lse_chunk.squeeze(-1)
            log_p = lc - lse_chunk  # [c, V_local]
            del lse_chunk
            p = log_p.exp()  # [c, V_local]
            # diff = log_p - log_qc; reuse log_p buffer in-place.
            log_p.sub_(qc.clamp(min=1e-10).log_())
            # kl_n = Σ_v p · diff; reuse p buffer in-place.
            kl_local[t0:t1] = p.mul_(log_p).sum(-1)
            del qc, log_p, p

        if tp_size > 1:
            dist.all_reduce(kl_local, op=dist.ReduceOp.SUM, group=tp_group)

        ctx.save_for_backward(logit_local, lse, kl_local, w, *teacher_logits)
        ctx.tp_group = tp_group
        ctx.chunk_t = chunk_t
        ctx.n_teachers = N
        return kl_local

    @staticmethod
    def backward(ctx, grad_kl: torch.Tensor):
        saved = ctx.saved_tensors
        logit_local, lse, kl = saved[0], saved[1], saved[2]
        w = saved[3]
        teacher_logits = saved[4:]
        tp_group = ctx.tp_group
        chunk_t = ctx.chunk_t
        T, V_local = logit_local.shape
        device = logit_local.device
        out_dtype = logit_local.dtype

        grad = torch.empty_like(logit_local)
        for t0 in range(0, T, chunk_t):
            t1 = min(t0 + chunk_t, T)
            qc = _accumulate_q_mix_chunk(teacher_logits, w, t0, t1, V_local, device, tp_group, out_dtype)
            log_p = logit_local[t0:t1] - lse[t0:t1].unsqueeze(-1)  # [c, V_local]
            p = log_p.exp()  # [c, V_local]
            # diff = log_p − log_qc − kl_n; mutate log_p in place to avoid an
            # extra [c, V_local] alloc.
            log_p.sub_(qc.clamp(min=1e-10).log_()).sub_(kl[t0:t1].unsqueeze(-1))
            del qc
            # grad = scale · p · diff; in-place on log_p.
            grad[t0:t1] = log_p.mul_(p).mul_(grad_kl[t0:t1].unsqueeze(-1))
            del log_p, p

        return (grad, None, None, None, *([None] * ctx.n_teachers))


# ── KL distillation loss (§8-9) ───────────────────────────────────────────────


# ── teacher-side prep: Conf + TopK_b + diversity + teacher_forward ────────────


def prepare_teacher_outputs(
    args,
    batch: dict,
    metadata_list: list[dict],
    unconcat_tokens: list[torch.Tensor],
    tokenizer,
    model: torch.nn.Module,
    conf_cache: dict[int, list[float]],
    offload_to_cpu: bool,
) -> list[dict | None]:
    """Run everything that depends on the teacher's parameters and return the
    selected teacher logits per sample.

    Designed to be called from `before_train_step_hook` with the frozen
    `"teacher"` weights restored, so the resulting logits come from the
    initial policy (paper §4.1). Tensors are detached and (optionally)
    offloaded to CPU so they can survive the swap-back to actor weights and
    the subsequent autograd-tracked student forward.

    When called inline from `loss_function` against the current student
    weights (legacy "teacher = student" mode), pass `offload_to_cpu=False`
    so the GPU tensors are consumed immediately by `distillation_loss`.

    For each sample the returned dict carries:
      * `sel_logits`: list of N teacher-logit tensors, each [T, V_local],
        on CPU when offloaded.
    """
    from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_rank

    tp_group = get_tensor_model_parallel_group()
    tp_rank = get_tensor_model_parallel_rank()

    q_inputs, conf_inputs, trace_tokens_flat, counts, cand_scores, cand_tokens = build_teacher_inputs(
        batch, metadata_list, unconcat_tokens, tokenizer
    )

    outputs: list[dict | None] = []
    offset = 0
    for i, n in enumerate(counts):
        if n == 0:
            outputs.append(None)
            continue

        resp_len = int(batch["response_lengths"][i])
        device = q_inputs[offset].device

        # §4 — Conf(τ_k) over the trace itself: π_T(τ_k[t] | x, τ_k[<t]).
        # Frozen teacher ⇒ Conf is invariant; cache by `id(cand_tokens[i])`
        # which is stable for the lifetime of a rollout.
        group_id = id(cand_tokens[i])
        confs = conf_cache.get(group_id)
        if confs is None:
            confs = compute_trace_confs(
                model,
                conf_inputs[offset : offset + n],
                trace_tokens_flat[offset : offset + n],
            )
            conf_cache[group_id] = confs
        full_scores = add_conf(args, cand_scores[i], confs)

        # §4 — TopK_b filter BEFORE the expensive q_forward.
        k_b = args.opsd_kb if args.opsd_kb is not None else n
        if k_b <= 0 or n <= k_b:
            keep = list(range(n))
        else:
            keep = sorted(np.argsort(full_scores)[-k_b:].tolist())
        q_in_sub = [q_inputs[offset + j] for j in keep]
        tokens_i = [cand_tokens[i][j] for j in keep]
        scores = [full_scores[j] for j in keep]
        offset += n

        # §5 + §6 — diversity selection + teacher_forward, order metric-dependent.
        if args.opsd_diversity_metric == "unigram_jsd":
            sel = select_diverse(args, scores, tokens_i, None, device, tp_group, 0)
            q_in_sel = [q_in_sub[k] for k in sel]
            sel_logits = teacher_forward(model, q_in_sel, resp_len)
        else:
            all_logits = teacher_forward(model, q_in_sub, resp_len)
            V_local = all_logits[0].size(-1)
            shard_start = tp_rank * V_local
            sel = select_diverse(args, scores, tokens_i, all_logits, device, tp_group, shard_start)
            sel_logits = [all_logits[k] for k in sel]
            del all_logits

        if offload_to_cpu:
            # Detach + move to CPU so subsequent weight swap / student forward
            # can run without interfering with these tensors.  They are TP-local
            # [T, V_local] shards, so CPU and GPU peak scale with V/tp instead
            # of V_full.
            sel_logits = [t.detach().to("cpu") for t in sel_logits]

        outputs.append({"sel_logits": sel_logits})

    return outputs


# ── KL distillation loss (§7-9) ───────────────────────────────────────────────


def distillation_loss(
    args,
    batch: dict,
    student_logits: list[torch.Tensor],
    sample_teacher_outputs: list[dict | None],
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None, dict[str, torch.Tensor]]:
    """Consume precomputed teacher outputs (one per sample) and produce per-token
    OPSD KL against student logits.

    `sample_teacher_outputs[i]` is either `None` (sample had no privileged
    candidates) or a dict containing `sel_logits` — the N selected teacher
    logit tensors for that sample. When they were precomputed in
    `before_train_step_hook` they live on CPU and are moved back to the
    student's device here; when computed inline from `loss_function` they
    are already on the student device.

    Returns:
        per_token_kl:   list of [T_i] tensors, one per sample, in the SAME
                        order as `student_logits` / `batch["response_lengths"]`.
                        Samples without privileged candidates get a zero
                        tensor that still flows gradient through `student_logits`
                        (so autograd hooks fire on every rank).  The caller
                        concatenates these and feeds to `sum_of_sample_mean`
                        for Megatron-friendly reduction (ALGO.md Part 1 §9 token-
                        mean → sample-mean → batch-sum).
        per_token_rkl:  None when ``--opsd-rkl-weight == 0``; otherwise a list
                        of [T_i] tensors carrying the reverse-KL contribution
                        per sample (ALGO.md Part 1 §9 auxiliary).
        metrics:        dict of detached scalar tensors for logging
                        (currently just ``opsd_w_entropy``).
    """
    from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_rank

    chunk_t = args.opsd_kl_chunk
    tp_rank = get_tensor_model_parallel_rank()
    tp_group = get_tensor_model_parallel_group()

    device = student_logits[0].device
    per_token_kl: list[torch.Tensor] = []
    rkl_enabled = args.opsd_rkl_weight > 0.0
    per_token_rkl: list[torch.Tensor] | None = [] if rkl_enabled else None
    w_entropy_sum = torch.tensor(0.0, device=device)
    w_entropy_count = 0

    for i, sample_out in enumerate(sample_teacher_outputs):
        p_student = student_logits[i]  # [T, V_local] vocab-parallel
        T = p_student.size(0)
        eff_chunk = T if chunk_t < 0 else chunk_t

        if sample_out is None:
            # No privileged candidates → contribute zero loss but keep the
            # student logits in the autograd graph (CP / DDP would otherwise
            # deadlock if some ranks never wire grad through their forward).
            per_token_kl.append(0.0 * p_student.sum(dim=-1))
            if per_token_rkl is not None:
                per_token_rkl.append(0.0 * p_student.sum(dim=-1))
            if _is_rank0():
                logger.info("[opsd_dbg/sample] i=%d sample_out=None (no privileged)", i)
            continue

        V_local = p_student.size(1)
        shard_start = tp_rank * V_local  # global vocab offset for this TP rank

        # Teacher logits are TP-local [T, V_local] shards held by `sample_out`
        # (CPU when offloaded). The fused KL autograd streams them per chunk in
        # both forward and backward, so there is no persistent [T, V_local]
        # q_mix buffer at OPSD scope — only chunk-sized fp32 accumulators.
        sel_src: list[torch.Tensor] = sample_out["sel_logits"]

        # §7 — mixture weights use a global top-K over the vocab-parallel student
        # logits, so every TP rank computes the same w_k^t. The student top-K
        # gather runs on the live [T, V_local] student logits; teacher gathers
        # stream chunk-wise from CPU so we never hold a full teacher shard on
        # GPU just to pick K columns out of it.
        top_k = min(args.opsd_weight_top_k, V_local * dist.get_world_size(group=tp_group))
        # Temperature is a positive scalar; it doesn't change top-K indices, so
        # skip the [T, V_local] copy that `p_student / temp` would allocate.
        # `mixture_weights` re-applies temperature inside its softmax.
        _, weight_idx = global_topk(p_student, top_k, shard_start, tp_group)  # [T, K]
        p_gathered = gather_global_logits(p_student, weight_idx, shard_start, tp_group)

        K_eff = weight_idx.size(-1)
        q_gathered = p_student.new_empty(len(sel_src), T, K_eff)
        for k, ltsr in enumerate(sel_src):
            for t0 in range(0, T, eff_chunk):
                t1 = min(t0 + eff_chunk, T)
                l_gpu = ltsr[t0:t1].to(device, non_blocking=True)
                q_gathered[k, t0:t1] = gather_global_logits(l_gpu, weight_idx[t0:t1], shard_start, tp_group)
                del l_gpu
        w, w_entropy_norm = mixture_weights(args, p_gathered, q_gathered)
        del q_gathered, p_gathered, weight_idx
        w_detached = w.detach()
        del w

        # §8 + §9 — KL(q_mix ‖ p_θ) via fused vocab-parallel autograd.
        # q_mix is reconstructed per chunk inside the Function; pass the teacher
        # shards as positional args so save_for_backward retains references.
        kl = VocabParallelMixtureKLDiv.apply(
            p_student, w_detached, tp_group, eff_chunk, args.opsd_pointwise_kl_clip, *sel_src
        )

        if args.opsd_jsd_token_clip is not None:
            kl = kl.clamp(max=args.opsd_jsd_token_clip)

        per_token_kl.append(kl)

        # Optional ALGO.md Part 1 §9 reverse-KL aux: KL(p_θ ‖ q_mix), full vocab.
        # The *main* gradient signal still comes from the forward-KL above
        # (α << 1 in ALGO.md Part 1 §9), so this is composed in plugin.py rather
        # than added here.
        if per_token_rkl is not None:
            rkl = VocabParallelMixtureRKLDiv.apply(p_student, w_detached, tp_group, eff_chunk, *sel_src)
            per_token_rkl.append(rkl)

        del sel_src, w_detached

        loss_masks = batch.get("loss_masks")
        if loss_masks is not None:
            mask = loss_masks[i]
            sample_we_sum = (w_entropy_norm * mask).sum()
            sample_we_count = int(mask.sum().clamp(min=1).item())
            w_entropy_sum = w_entropy_sum + sample_we_sum
            w_entropy_count += sample_we_count
            if _is_rank0():
                # Per-sample tally — pairs with the mixture/N log to localize
                # whether 0s come from N=1 (zeros tensor), mask=0 (zero rows),
                # or live values diluted by big T (per-token mean ≈ 0).
                logger.info(
                    "[opsd_dbg/sample] i=%d resp_T=%d mask_sum=%.0f "
                    "we_norm_mean=%.4f contrib_sum=%.4f contrib_count=%d",
                    i,
                    int(w_entropy_norm.numel()),
                    float(mask.sum().item()),
                    w_entropy_norm.detach().mean().item(),
                    float(sample_we_sum.item()),
                    sample_we_count,
                )
        else:
            w_entropy_sum = w_entropy_sum + w_entropy_norm.sum()
            w_entropy_count += int(w_entropy_norm.numel())

    metrics: dict[str, torch.Tensor] = {}
    if w_entropy_count > 0:
        metrics["opsd_w_entropy"] = (w_entropy_sum / w_entropy_count).detach()
    return per_token_kl, per_token_rkl, metrics


# ── student response extraction ───────────────────────────────────────────────


def extract_student_responses(logits: torch.Tensor, args, batch: dict) -> list[torch.Tensor]:
    # Student logits stay vocab-parallel [T, V_local]; distillation_loss and the
    # fused mixture autograd ops operate on them without gathering.
    return [
        chunk
        for chunk, _ in get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
        )
    ]
