"""Training-side OPSD logic (metho.md §4-9).

Steps executed inside the loss function:
  §4  Add conf term to quality scores; apply TopK_b filter.
  §5  Select N diverse traces (k-center greedy).
  §6  Teacher forward: π_θ(·| x, τ_k, y_{<t}).
  §7  Compute mixture weights w_k^t.
  §8  Mixture teacher q_mix^t = Σ_k w_k^t q_k^t.
  §9  KL distillation loss.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from slime.backends.megatron_utils.loss import get_responses

from .selection import kcenter, pairwise_seq_jsd, pairwise_unigram_jsd

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
) -> tuple[list[torch.Tensor], list[int], list[list[float]], list[list[list[int]]]]:
    """Build teacher input sequences for every privileged candidate in the batch.

    Returns:
        teacher_inputs: flat list of [prompt_with_priv + response] token tensors.
        counts:         number of candidates per sample (0 if no candidates).
        cand_scores:    structural quality scores per sample.
        cand_tokens:    privileged trace token lists per sample.
    """
    teacher_inputs: list[torch.Tensor] = []
    counts: list[int] = []
    cand_scores: list[list[float]] = []
    cand_tokens: list[list[list[int]]] = []
    device = unconcat_tokens[0].device

    for i, meta in enumerate(metadata_list):
        traces: list[list[int]] = meta["privileged_candidates"]
        scores: list[float] = meta["privileged_candidate_scores"]
        problem: str = meta["problem_text"]

        if not traces:
            counts.append(0)
            cand_scores.append([])
            cand_tokens.append([])
            continue

        resp_tok = unconcat_tokens[i][-batch["response_lengths"][i] :]
        for trace in traces:
            prompt = _build_privileged_prompt(problem, trace, tokenizer)
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            teacher_inputs.append(torch.tensor(prompt_ids + resp_tok.tolist(), device=device))

        counts.append(len(traces))
        cand_scores.append(scores)
        cand_tokens.append(traces)

    return teacher_inputs, counts, cand_scores, cand_tokens


def _build_privileged_prompt(problem: str, trace_tokens: list[int], tokenizer) -> str:
    reference = tokenizer.decode(trace_tokens, skip_special_tokens=True)
    return f"Problem: {problem}\n\nHere is a reference solution:\n=== Begin ===\n{reference}\n=== End ==={_TRANSITION_PROMPT}{_BOXED_INSTRUCTION}"


# ── teacher forward (§6) ─────────────────────────────────────────────────────


def teacher_forward(
    model: torch.nn.Module,
    inputs: list[torch.Tensor],
    resp_len: int,
    resp_tokens: torch.Tensor,
) -> tuple[list[torch.Tensor], list[float]]:
    """Run π_θ(·| x, τ_k, y_{<t}) one trace at a time; offload logits to CPU.

    Processing one trace at a time avoids materialising [N, T, V] on GPU.
    Megatron's float16_to_fp32 produces a full [1, seq, vocab] fp32 tensor per
    forward pass (~1.4 GB for Qwen3-1.7B at T≈2400); it is sliced to [T, V],
    conf is computed on GPU, then the tensor moves to CPU before the next pass.

    Returns:
        logits_cpu: per-trace [T, V] float32 tensors on CPU.
        confs:      per-trace mean token log-prob at resp_tokens positions.
    """
    from megatron.core.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_world_size,
    )

    tp_size = get_tensor_model_parallel_world_size()

    device = inputs[0].device
    resp_t = resp_tokens.to(device=device, dtype=torch.long)  # [T]
    logits_cpu: list[torch.Tensor] = []
    confs: list[float] = []

    for inp in inputs:
        orig_len = inp.size(0)
        # Sequence parallel requires seq_len % tp_size == 0.  Right-pad with zeros;
        # causal attention means real positions are unaffected by the tail padding.
        pad_len = (-orig_len) % tp_size
        inp_model = F.pad(inp, (0, pad_len)) if pad_len else inp

        with torch.no_grad():
            out = model(input_ids=inp_model.unsqueeze(0), position_ids=None, attention_mask=None, labels=None)
        del inp_model
        raw = out[0] if isinstance(out, tuple) else out
        # Slice original (non-padded) response positions; padding is at the tail.
        # With TP > 1 the model returns vocab-parallel logits [T, V/tp]; all-gather
        # to full vocab [T, V] so that resp_t indices (range 0..V-1) are in-bounds
        # and logsumexp covers the full distribution.
        logit_t = raw[0, orig_len - resp_len : orig_len].float()  # [T, V_local]
        del out, raw

        if tp_size > 1:
            tp_group = get_tensor_model_parallel_group()
            shards = [torch.empty_like(logit_t) for _ in range(tp_size)]
            torch.distributed.all_gather(shards, logit_t.contiguous(), group=tp_group)
            logit_t = torch.cat(shards, dim=-1)  # [T, V_full]

        lse = torch.logsumexp(logit_t, dim=-1)  # [T]
        at_resp = logit_t[torch.arange(resp_len, device=device), resp_t]  # [T]
        confs.append((at_resp - lse).mean().item())
        del lse, at_resp

        logits_cpu.append(logit_t.cpu())
        del logit_t
        torch.cuda.empty_cache()

    return logits_cpu, confs


# ── quality: conf term (§4) ──────────────────────────────────────────────────


def add_conf(
    args,
    base_scores: list[float],
    confs: list[float],
) -> list[float]:
    """Add η_c·Conf(τ) to structural quality scores.

    Conf(τ) ≈ (1/T) Σ_t log π_θ(y_t | x, τ, y_{<t}) — pre-computed per trace.
    """
    if args.opsd_quality_conf_weight == 0.0:
        return base_scores
    w = args.opsd_quality_conf_weight
    return [b + w * c for b, c in zip(base_scores, confs, strict=True)]


# ── quality: TopK_b filter (§4) ──────────────────────────────────────────────


def topk_filter(
    k_b: int,
    logits_cpu: list[torch.Tensor],
    tokens: list[list[int]],
    scores: list[float],
) -> tuple[list[torch.Tensor], list[list[int]], list[float]]:
    """Keep top-K_b candidates by full quality score."""
    if k_b <= 0 or len(scores) <= k_b:
        return logits_cpu, tokens, scores
    top_idx = np.argsort(scores)[-k_b:].tolist()
    return [logits_cpu[j] for j in top_idx], [tokens[j] for j in top_idx], [scores[j] for j in top_idx]


# ── diversity selection (§5) ─────────────────────────────────────────────────


def select_diverse(
    args,
    scores: list[float],
    tokens: list[list[int]],
    logits_cpu: list[torch.Tensor],
    device: torch.device,
) -> list[int]:
    """Select N diverse candidates via k-center greedy (metho.md §5)."""
    n_select = min(args.opsd_n, len(scores))
    if len(scores) <= n_select:
        return list(range(len(scores)))

    if args.opsd_diversity_metric == "token_jsd":
        dist = pairwise_seq_jsd(logits_cpu, args.opsd_diversity_top_k, device)
    else:
        dist = pairwise_unigram_jsd(tokens)

    return kcenter(scores, dist, n_select)


# ── mixture weights (§7) ─────────────────────────────────────────────────────


def mixture_weights(
    args,
    p_student: torch.Tensor,
    q_gathered: torch.Tensor,
    weight_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token mixture weights w_k^t (metho.md §7).

    w_k^t ∝ exp(-β·Δ_k^t - γ·h_k^t + ρ·g_k^t)

    where:
      Δ_k^t = KL(q_k^t ‖ p^t)        — KL term, down-weights confusing teachers
      h_k^t = H(q_k^t)                — entropy, down-weights uncertain teachers
      g_k^t = mean_j JSD(q_k^t, q_j^t) — diversity bonus

    All computed on the student's top-K vocabulary for efficiency.

    Args:
        p_student:   [T, V] student logits.
        q_gathered:  [N, T, K] teacher logits pre-gathered at weight_idx positions.
        weight_idx:  [T, K] student top-K token indices used for gathering.
    """
    N, T, K = q_gathered.shape

    p_log = F.log_softmax(p_student.gather(-1, weight_idx) / args.opsd_temperature, dim=-1)  # [T, K]
    q_log = F.log_softmax(q_gathered / args.opsd_temperature, dim=-1)  # [N, T, K]
    q_prob = q_log.exp()

    delta = (q_prob * (q_log - p_log.unsqueeze(0))).sum(-1).clamp(min=0)  # [N, T]
    entropy = -(q_prob * q_log).sum(-1)  # [N, T]
    diversity = _pairwise_jsd(q_prob, q_log) if N > 1 else torch.zeros(N, T, device=p_student.device)

    logits = -args.opsd_kl_weight * delta - args.opsd_entropy_weight * entropy + args.opsd_diversity_weight * diversity
    return F.softmax(logits, dim=0)  # [N, T]


def _pairwise_jsd(q_prob: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    """Per-token mean pairwise JSD: g_k^t = (1/(N-1)) Σ_{j≠k} JSD(q_k^t, q_j^t).

    Args:
        q_prob, q_log: [N, T, K]

    Returns:
        [N, T]
    """
    N = q_prob.size(0)
    pi = q_prob.unsqueeze(1)  # [N, 1, T, K]
    pj = q_prob.unsqueeze(0)  # [1, N, T, K]
    lpi = q_log.unsqueeze(1)
    lpj = q_log.unsqueeze(0)

    mix = 0.5 * (pi + pj)
    log_mix = mix.clamp(min=1e-10).log()
    jsd = 0.5 * ((pi * (lpi - log_mix)).sum(-1) + (pj * (lpj - log_mix)).sum(-1)).clamp(min=0)  # [N, N, T]

    eye = torch.eye(N, dtype=torch.bool, device=q_prob.device)
    jsd = jsd.masked_fill(eye.unsqueeze(-1), 0.0)
    return jsd.sum(dim=1) / (N - 1)  # [N, T]


# ── KL distillation loss (§8-9) ───────────────────────────────────────────────


def distillation_loss(
    args,
    batch: dict,
    student_logits: list[torch.Tensor],
    teacher_inputs: list[torch.Tensor],
    counts: list[int],
    cand_scores: list[list[float]],
    cand_tokens: list[list[list[int]]],
    model: torch.nn.Module,
) -> torch.Tensor:
    """Compute OPSD KL loss over the batch (metho.md §6-9).

    Teacher logits are never assembled as [N, T, V] on GPU.  Each trace is
    processed one at a time; [T, V] tensors are kept on CPU and loaded in
    _CHUNK_T-row slices for q_mix and KL so peak GPU delta ≈ q_mix + 2×chunk.
    """
    _CHUNK_T = 256  # token-position chunk; each slice ≈ 147 MB for Qwen3-1.7B
    device = student_logits[0].device
    total_kl = torch.tensor(0.0, device=device)
    valid = 0
    offset = 0

    for i, n in enumerate(counts):
        if n == 0:
            continue

        resp_len = int(batch["response_lengths"][i])
        p_student = student_logits[i]  # [T, V]
        resp_tokens = batch["unconcat_tokens"][i][-resp_len:]  # [T]
        T, V = p_student.shape

        # §6 — teacher forward: one trace at a time, returns CPU tensors + confs
        logits_cpu, confs = teacher_forward(model, teacher_inputs[offset : offset + n], resp_len, resp_tokens)
        offset += n

        # §4 — full quality score + TopK_b
        scores = add_conf(args, cand_scores[i], confs)
        logits_cpu, tokens_i, scores = topk_filter(args.opsd_kb, logits_cpu, cand_tokens[i], scores)

        # §5 — k-center greedy diversity selection (moves pairs to GPU)
        sel = select_diverse(args, scores, tokens_i, logits_cpu, device)
        sel_logits = [logits_cpu[k] for k in sel]  # N CPU tensors [T, V]
        del logits_cpu

        N_sel = len(sel_logits)

        # §7 — mixture weights over student's top-K gathered teacher logits [N, T, K]
        top_k = min(args.opsd_weight_top_k, V)
        _, weight_idx = torch.topk(p_student / args.opsd_temperature, k=top_k, dim=-1)  # [T, K]
        gathered_list: list[torch.Tensor] = []
        for l in sel_logits:
            l_dev = l.to(device)
            gathered_list.append(l_dev.gather(-1, weight_idx))  # [T, K]
            del l_dev
        q_gathered = torch.stack(gathered_list, dim=0)  # [N, T, K]
        del gathered_list
        w = mixture_weights(args, p_student, q_gathered, weight_idx)  # [N, T]
        del q_gathered, weight_idx

        # §8 — q_mix = Σ_k w_k * softmax(q_k); chunked over T, peak ≈ q_mix + 2×[chunk,V]
        q_mix = torch.zeros(T, V, dtype=torch.float32, device=device)
        for k, l_cpu in enumerate(sel_logits):
            w_k = w[k]  # [T]
            for t0 in range(0, T, _CHUNK_T):
                t1 = min(t0 + _CHUNK_T, T)
                l_c = l_cpu[t0:t1].to(device)  # [chunk, V]
                s_c = F.softmax(l_c, dim=-1)  # [chunk, V]
                del l_c
                s_c.mul_(w_k[t0:t1].unsqueeze(-1))  # in-place weight
                q_mix[t0:t1].add_(s_c)  # in-place accumulate
                del s_c
        del sel_logits, w

        # §9 — KL(q_mix ‖ p_θ); chunked over T, peak ≈ q_mix + 2×[chunk,V]
        kl = torch.zeros(T, device=device)
        for t0 in range(0, T, _CHUNK_T):
            t1 = min(t0 + _CHUNK_T, T)
            qc = q_mix[t0:t1]  # view [chunk, V]
            log_q = qc.clamp(min=1e-10).log()  # [chunk, V]
            log_p = F.log_softmax(p_student[t0:t1], dim=-1)  # [chunk, V]
            log_q.sub_(log_p)  # in-place: log_q -= log_p
            del log_p
            kl[t0:t1] = (qc * log_q).sum(-1)
            del log_q
        del q_mix

        if args.opsd_jsd_token_clip is not None:
            kl = kl.clamp(max=args.opsd_jsd_token_clip)

        loss_masks = batch.get("loss_masks")
        if loss_masks is not None:
            mask = loss_masks[i]
            total_kl += (kl * mask).sum() / mask.sum().clamp(min=1)
        else:
            total_kl += kl.mean()
        valid += 1

    if valid == 0:
        return total_kl
    return total_kl / valid


# ── student response extraction ───────────────────────────────────────────────


def extract_student_responses(logits: torch.Tensor, args, batch: dict) -> list[torch.Tensor]:
    from megatron.core import mpu, tensor_parallel

    # Megatron returns vocab-parallel logits [1, T, V/tp] when TP > 1.
    # Gather to full-vocab [1, T, V] so topk, softmax and KL operate over
    # the complete distribution.  The backward of gather_from_tensor_model_
    # parallel_region is a reduce-scatter that correctly distributes the KL
    # gradient back to each TP rank's shard.
    if mpu.get_tensor_model_parallel_world_size() > 1:
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)

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
