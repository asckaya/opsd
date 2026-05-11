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
    return (
        f"Problem: {problem}\n\n"
        f"Here is a reference solution:\n=== Begin ===\n{reference}\n=== End ==="
        f"{_TRANSITION_PROMPT}{_BOXED_INSTRUCTION}"
    )


# ── teacher forward (§6) ─────────────────────────────────────────────────────


def teacher_forward(
    model: torch.nn.Module,
    inputs: list[torch.Tensor],
    resp_len: int,
    pad_id: int,
    chunk_size: int = 1,
) -> torch.Tensor:
    """Run π_θ(·| x, τ_k, y_{<t}) in no_grad using the current training model.

    Processes inputs in chunks to bound peak memory.  The full logit tensor
    [N, max_len, vocab] is produced by Megatron in fp32 — for Qwen3-1.7B with
    N=16 and max_len≈2400 this is ~23 GB, which OOMs a training GPU.
    By chunking to chunk_size=1 we cap the allocation at ~2.4 GB per chunk.

    Args:
        model:      Training model (DDP-wrapped Megatron GPTModel).
        inputs:     List of [prompt_with_priv_context + response] token tensors.
        resp_len:   Response length; only the last resp_len logit positions are kept.
        pad_id:     Token id used for right-padding within each chunk.
        chunk_size: Traces per forward pass.  Lower = less peak memory.

    Returns:
        [N, resp_len, V] float32 logits.
    """
    device = inputs[0].device
    chunks: list[torch.Tensor] = []

    for start in range(0, len(inputs), chunk_size):
        batch = inputs[start : start + chunk_size]
        max_len = max(t.size(0) for t in batch)
        padded = torch.stack([
            F.pad(t, (0, max_len - t.size(0)), value=pad_id) for t in batch
        ]).to(device)

        with torch.no_grad():
            out = model(input_ids=padded, position_ids=None, attention_mask=None, labels=None)
        del padded

        raw = out[0] if isinstance(out, tuple) else out
        chunks.append(raw[:, -resp_len:].float())  # slice immediately; free the rest
        del out, raw
        torch.cuda.empty_cache()

    return torch.cat(chunks, dim=0)  # [N, resp_len, V]


# ── quality: conf term (§4) ──────────────────────────────────────────────────


def add_conf(
    args,
    base_scores: list[float],
    q_teachers: torch.Tensor,
    resp_tokens: torch.Tensor,
) -> list[float]:
    """Add η_c·Conf(τ) to structural quality scores.

    Conf is approximated as the teacher's mean token log-probability over the
    student response y, conditioned on privileged trace τ:
      Conf(τ) ≈ (1/T) Σ_t log π_θ(y_t | x, τ, y_{<t})
    """
    if args.opsd_quality_conf_weight == 0.0:
        return base_scores

    # q_teachers: [N, T, V],  resp_tokens: [T]
    target = resp_tokens.to(device=q_teachers.device, dtype=torch.long)
    gathered = q_teachers.gather(-1, target[None, :, None].expand(len(base_scores), -1, 1)).squeeze(-1)
    log_probs = gathered - torch.logsumexp(q_teachers, dim=-1)  # [N, T] log-softmax
    conf = log_probs.mean(dim=-1).tolist()  # [N]

    w = args.opsd_quality_conf_weight
    return [b + w * c for b, c in zip(base_scores, conf, strict=True)]


# ── quality: TopK_b filter (§4) ──────────────────────────────────────────────


def topk_filter(
    k_b: int,
    q_teachers: torch.Tensor,
    tokens: list[list[int]],
    scores: list[float],
) -> tuple[torch.Tensor, list[list[int]], list[float]]:
    """Keep top-K_b candidates by full quality score."""
    if k_b <= 0 or len(scores) <= k_b:
        return q_teachers, tokens, scores
    top_idx = np.argsort(scores)[-k_b:].tolist()
    return q_teachers[top_idx], [tokens[j] for j in top_idx], [scores[j] for j in top_idx]


# ── diversity selection (§5) ─────────────────────────────────────────────────


def select_diverse(
    args,
    scores: list[float],
    tokens: list[list[int]],
    q_teachers: torch.Tensor,
) -> list[int]:
    """Select N diverse candidates via k-center greedy (metho.md §5)."""
    n_select = min(args.opsd_n, len(scores))
    if len(scores) <= n_select:
        return list(range(len(scores)))

    if args.opsd_diversity_metric == "token_jsd":
        dist = pairwise_seq_jsd(q_teachers, args.opsd_diversity_top_k)
    else:
        dist = pairwise_unigram_jsd(tokens)

    return kcenter(scores, dist, n_select)


# ── mixture weights (§7) ─────────────────────────────────────────────────────


def mixture_weights(args, p_student: torch.Tensor, q_teachers: torch.Tensor) -> torch.Tensor:
    """Compute per-token mixture weights w_k^t (metho.md §7).

    w_k^t ∝ exp(-β·Δ_k^t - γ·h_k^t + ρ·g_k^t)

    where:
      Δ_k^t = KL(q_k^t ‖ p^t)        — KL term, down-weights confusing teachers
      h_k^t = H(q_k^t)                — entropy, down-weights uncertain teachers
      g_k^t = mean_j JSD(q_k^t, q_j^t) — diversity bonus

    All computed on the student's top-K vocabulary for efficiency.
    """
    N, T, V = q_teachers.shape
    top_k = min(args.opsd_weight_top_k, V)

    # Student top-k tokens as shared vocabulary
    _, idx = torch.topk(p_student / args.opsd_temperature, k=top_k, dim=-1)   # [T, K]

    p_log = F.log_softmax(p_student.gather(-1, idx) / args.opsd_temperature, dim=-1)  # [T, K]
    q_gathered = q_teachers.gather(-1, idx.unsqueeze(0).expand(N, -1, -1))             # [N, T, K]
    q_log = F.log_softmax(q_gathered / args.opsd_temperature, dim=-1)                  # [N, T, K]
    q_prob = q_log.exp()

    delta = (q_prob * (q_log - p_log.unsqueeze(0))).sum(-1).clamp(min=0)   # [N, T]
    entropy = -(q_prob * q_log).sum(-1)                                      # [N, T]
    diversity = _pairwise_jsd(q_prob, q_log) if N > 1 else torch.zeros(N, T, device=p_student.device)

    logits = (
        -args.opsd_kl_weight * delta
        - args.opsd_entropy_weight * entropy
        + args.opsd_diversity_weight * diversity
    )
    return F.softmax(logits, dim=0)  # [N, T]


def _pairwise_jsd(q_prob: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    """Per-token mean pairwise JSD: g_k^t = (1/(N-1)) Σ_{j≠k} JSD(q_k^t, q_j^t).

    Args:
        q_prob, q_log: [N, T, K]

    Returns:
        [N, T]
    """
    N = q_prob.size(0)
    pi  = q_prob.unsqueeze(1)   # [N, 1, T, K]
    pj  = q_prob.unsqueeze(0)   # [1, N, T, K]
    lpi = q_log.unsqueeze(1)
    lpj = q_log.unsqueeze(0)

    mix = 0.5 * (pi + pj)
    log_mix = mix.clamp(min=1e-10).log()
    jsd = 0.5 * (
        (pi * (lpi - log_mix)).sum(-1) + (pj * (lpj - log_mix)).sum(-1)
    ).clamp(min=0)  # [N, N, T]

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
    pad_id: int,
) -> torch.Tensor:
    """Compute OPSD KL loss over the batch (metho.md §6-9)."""
    device = student_logits[0].device
    total_kl = torch.tensor(0.0, device=device)
    valid = 0
    offset = 0

    for i, n in enumerate(counts):
        if n == 0:
            continue

        resp_len = int(batch["response_lengths"][i])
        p_student = student_logits[i]                           # [T, V]
        resp_tokens = batch["unconcat_tokens"][i][-resp_len:]   # [T]

        # §6 — teacher logits for all n candidates
        q_t = teacher_forward(model, teacher_inputs[offset : offset + n], resp_len, pad_id)
        offset += n  # [n, T, V]

        # §4 — full quality score + TopK_b
        scores = add_conf(args, cand_scores[i], q_t, resp_tokens)
        q_t, tokens_i, scores = topk_filter(args.opsd_kb, q_t, cand_tokens[i], scores)

        # §5 — k-center greedy diversity selection
        sel = select_diverse(args, scores, tokens_i, q_t)
        q_t = q_t[sel]  # [N, T, V]

        # §7 — mixture weights
        w = mixture_weights(args, p_student, q_t)   # [N, T]

        # §8 — mixture teacher
        q_mix = (w.unsqueeze(-1) * F.softmax(q_t, dim=-1)).sum(0)  # [T, V]

        # §9 — KL(q_mix ‖ p_θ)
        kl = (q_mix * (q_mix.clamp(min=1e-10).log() - F.log_softmax(p_student, dim=-1))).sum(-1)

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
