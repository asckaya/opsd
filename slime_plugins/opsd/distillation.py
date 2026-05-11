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
import torch.distributed as dist
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
    """Run π_θ(·| x, τ_k, y_{<t}) one trace at a time; return GPU logits.

    Returns full-vocab teacher logits [resp_len, V_full] per input, on the same
    device as the model. q_k^t at position t is taken from logit position
    prompt_len + t - 1 (causal-LM convention: output[i] predicts input[i+1]),
    matching the slicing used by slime's own `get_responses`
    (`logits[start-1 : end-1]`).

    Memory note: with N=4 selected traces and Qwen3 (V≈152k), retained GPU
    memory is N × [T, V] × 4 B ≈ 4.8 GB at T=2048 — fits comfortably alongside
    the 1.7B model. For very large N or very long responses, switch to
    `.cpu()` after the slice if memory becomes tight.
    """
    from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size

    tp_size = get_tensor_model_parallel_world_size()
    teacher_logits: list[torch.Tensor] = []

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
        # output[i] predicts input[i+1]; response positions [P, P+T) ← logits [P-1, P+T-1).
        # P = orig_len - resp_len, so the slice is [orig_len - resp_len - 1, orig_len - 1).
        # With TP > 1 the model returns vocab-parallel logits [T, V/tp]; all-gather
        # to full vocab [T, V] so downstream gathers / q_mix shards are consistent.
        logit_t = raw[0, orig_len - resp_len - 1 : orig_len - 1].float()  # [T, V_local]
        del out, raw

        if tp_size > 1:
            tp_group = get_tensor_model_parallel_group()
            shards = [torch.empty_like(logit_t) for _ in range(tp_size)]
            torch.distributed.all_gather(shards, logit_t.contiguous(), group=tp_group)
            logit_t = torch.cat(shards, dim=-1)  # [T, V_full]

        teacher_logits.append(logit_t)

    return teacher_logits


# ── Conf(τ) forward (§4) ─────────────────────────────────────────────────────


def compute_trace_confs(
    model: torch.nn.Module,
    conf_inputs: list[torch.Tensor],
    trace_tokens_flat: list[torch.Tensor],
) -> list[float]:
    """Conf(τ_k) = (1/|τ_k|) Σ_t log π_T(τ_k[t] | x, τ_k[<t]) — metho.md §4.

    One forward pass per trace over `chat(x) + τ_k`. Reads logits at positions
    [P-1, P+|τ|-1) where P = len(chat(x)), then evaluates the log-prob of
    τ_k[t] at each position.
    """
    from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size

    tp_size = get_tensor_model_parallel_world_size()
    confs: list[float] = []

    for inp, trace_t in zip(conf_inputs, trace_tokens_flat, strict=True):
        orig_len = inp.size(0)
        trace_len = trace_t.size(0)
        pad_len = (-orig_len) % tp_size
        inp_model = F.pad(inp, (0, pad_len)) if pad_len else inp
        device = inp.device

        with torch.no_grad():
            out = model(input_ids=inp_model.unsqueeze(0), position_ids=None, attention_mask=None, labels=None)
        del inp_model
        raw = out[0] if isinstance(out, tuple) else out
        logit_t = raw[0, orig_len - trace_len - 1 : orig_len - 1].float()  # [trace_len, V_local]
        del out, raw

        if tp_size > 1:
            tp_group = get_tensor_model_parallel_group()
            shards = [torch.empty_like(logit_t) for _ in range(tp_size)]
            torch.distributed.all_gather(shards, logit_t.contiguous(), group=tp_group)
            logit_t = torch.cat(shards, dim=-1)  # [trace_len, V_full]

        lse = torch.logsumexp(logit_t, dim=-1)
        at_trace = logit_t[torch.arange(trace_len, device=device), trace_t]
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
    on the trace itself, pre-computed by `compute_trace_confs` per metho.md §4.
    """
    if args.opsd_quality_conf_weight == 0.0:
        return base_scores
    w = args.opsd_quality_conf_weight
    return [b + w * c for b, c in zip(base_scores, confs, strict=True)]


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
) -> list[int]:
    """Select N diverse candidates via k-center greedy (metho.md §5).

    `logits_cpu` is required only for the token_jsd metric; it may be None when
    the metric is unigram_jsd (token-only distance, no q logits needed).
    """
    n_select = min(args.opsd_n, len(scores))
    if len(scores) <= n_select:
        return list(range(len(scores)))

    if args.opsd_diversity_metric == "token_jsd":
        assert logits_cpu is not None, "token_jsd diversity requires teacher logits"
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


# ── vocab-parallel KL divergence (§9) ───────────────────────────────────────


class _VocabParallelKLDiv(torch.autograd.Function):
    """Forward-KL(q_mix ‖ p_θ) for vocab-parallel student logits.

    Each TP rank holds [T, V_local].  A two-phase chunked algorithm avoids
    ever materialising [T, V_full] on any GPU:

    Phase 1 — distributed log-sum-exp:
      Per chunk of token positions, each rank computes its local max and
      exp-sum, all-reduces both, and writes the global lse into a [T]
      buffer.  Peak extra GPU memory = [chunk, V_local] ≈ 38 MB (TP=4).

    Phase 2 — KL contribution + backward buffer:
      Per chunk, log_softmax = logit_local - lse; KL local term is summed
      and the difference (softmax_local - q_local) is stored in a pre-
      allocated [T, V_local] buffer for the backward.  One all-reduce of
      [T] (tiny) gives the global KL at the end.

    Backward: d(KL)/d(logit_local) = softmax_local - q_local.
      Saved tensor is [T, V_local] ≈ 360 MB (TP=4), not [T, V_full].
    """

    @staticmethod
    def forward(
        ctx,
        logit_local: torch.Tensor,  # [T, V_local], requires_grad
        q_local: torch.Tensor,  # [T, V_local], detached q_mix shard
        tp_group,
        chunk_t: int,
    ) -> torch.Tensor:  # [T]
        T, V_local = logit_local.shape
        tp_size = dist.get_world_size(group=tp_group)

        # ── TP=1 fast path: no cross-rank reductions, single softmax per chunk ──
        if tp_size == 1:
            sm_minus_q = logit_local.new_empty(T, V_local)
            kl = logit_local.new_zeros(T)
            for t0 in range(0, T, chunk_t):
                t1 = min(t0 + chunk_t, T)
                lc = logit_local[t0:t1]  # view [c, V]
                qc = q_local[t0:t1]
                log_p = F.log_softmax(lc, dim=-1)  # [c, V]
                kl[t0:t1] = (qc * (qc.clamp(min=1e-10).log() - log_p)).sum(-1)
                sm_minus_q[t0:t1] = log_p.exp() - qc
                del lc, qc, log_p
            ctx.save_for_backward(sm_minus_q)
            ctx.tp_group = tp_group
            return kl

        # ── TP > 1: chunked distributed log-sum-exp + KL ──
        # Phase 1: distributed log-sum-exp
        lse = logit_local.new_empty(T)  # [T]
        for t0 in range(0, T, chunk_t):
            t1 = min(t0 + chunk_t, T)
            c = logit_local[t0:t1]  # view [c, V_local]
            gmax = c.max(dim=-1, keepdim=True).values  # [c, 1] new tensor
            dist.all_reduce(gmax, op=dist.ReduceOp.MAX, group=tp_group)
            gsum = (c - gmax).exp().sum(dim=-1, keepdim=True)  # [c, 1]
            dist.all_reduce(gsum, op=dist.ReduceOp.SUM, group=tp_group)
            lse[t0:t1] = (gmax + gsum.log()).squeeze(-1)
            del gmax, gsum

        # Phase 2: KL + (softmax - q) buffer
        sm_minus_q = logit_local.new_empty(T, V_local)  # [T, V_local]
        kl_local = logit_local.new_zeros(T)  # [T]
        for t0 in range(0, T, chunk_t):
            t1 = min(t0 + chunk_t, T)
            lc = logit_local[t0:t1]  # view [c, V_local]
            qc = q_local[t0:t1]  # view [c, V_local]
            log_p = lc - lse[t0:t1].unsqueeze(-1)  # log-softmax [c, V_local]
            sm_c = log_p.exp()  # softmax [c, V_local]
            kl_local[t0:t1] = (qc * (qc.clamp(min=1e-10).log() - log_p)).sum(-1)
            sm_minus_q[t0:t1] = sm_c - qc
            del log_p, sm_c
        del lse

        # Single all-reduce to sum KL shards across TP ranks
        kl = kl_local.clone()
        dist.all_reduce(kl, op=dist.ReduceOp.SUM, group=tp_group)

        ctx.save_for_backward(sm_minus_q)
        ctx.tp_group = tp_group
        return kl

    @staticmethod
    def backward(ctx, grad_kl: torch.Tensor):
        (sm_minus_q,) = ctx.saved_tensors
        # d(KL)/d(logit_local) = softmax_local - q_local
        grad = grad_kl.unsqueeze(-1) * sm_minus_q  # [T, V_local]
        return grad, None, None, None


# ── KL distillation loss (§8-9) ───────────────────────────────────────────────


def distillation_loss(
    args,
    batch: dict,
    student_logits: list[torch.Tensor],
    q_inputs: list[torch.Tensor],
    conf_inputs: list[torch.Tensor],
    trace_tokens_flat: list[torch.Tensor],
    counts: list[int],
    cand_scores: list[list[float]],
    cand_tokens: list[list[list[int]]],
    model: torch.nn.Module,
) -> torch.Tensor:
    """Compute OPSD KL loss over the batch (metho.md §4, §6-9).

    Student logits stay vocab-parallel [T, V_local] throughout; no full-vocab
    gather is ever performed on the student side.  teacher_forward returns
    full-vocab [T, V] GPU tensors for the N selected traces; with N=4 and
    V≈152k this peaks at ≈4.8 GB per sample alongside the model.

    §4 Conf(τ_k) is computed by compute_trace_confs via a dedicated forward over
    chat(x) + τ_k — separate from the q_k^t forward, per metho.md §4. Results
    are cached per rollout group (same x, same τ set across samples).

    §8 q_mix is built per-TP-shard: teacher softmax is computed on GPU over the
    full vocab and only the local shard accumulates into q_mix_local.

    §9 KL is computed by _VocabParallelKLDiv, a custom autograd Function whose
    backward gradient d(KL)/d(logit_local) = softmax_local - q_local uses only
    the [T, V_local] saved buffer, not [T, V_full].
    """
    from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_rank

    _CHUNK_T = 256
    tp_rank = get_tensor_model_parallel_rank()
    tp_group = get_tensor_model_parallel_group()

    device = student_logits[0].device
    total_kl = torch.tensor(0.0, device=device)
    valid = 0
    offset = 0
    # Same rollout group shares privileged candidates (same problem, same τ set),
    # so Conf(τ_k) is identical across the n_samples_per_prompt samples of that
    # group. Cache by the `cand_tokens[i]` list identity: rollout.py shares the
    # same Python list across same-group samples, and build_teacher_inputs
    # propagates that identity through `cand_tokens.append(traces)`. If identity
    # is broken (e.g. deep copy in some data path), the cache simply misses and
    # we fall back to recomputing — still correct.
    conf_cache: dict[int, list[float]] = {}

    for i, n in enumerate(counts):
        if n == 0:
            continue

        resp_len = int(batch["response_lengths"][i])
        p_student = student_logits[i]  # [T, V_local] vocab-parallel
        T, V_local = p_student.shape
        shard_start = tp_rank * V_local  # global vocab offset for this TP rank

        # §4 — Conf(τ_k) over the trace itself: π_T(τ_k[t] | x, τ_k[<t]).
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

        # §4 — TopK_b filter BEFORE the expensive q_forward.  When K_b < n this
        # skips (n - K_b) full-length forwards over (priv_prompt + response).
        k_b = args.opsd_kb if args.opsd_kb is not None else n
        if k_b <= 0 or n <= k_b:
            keep = list(range(n))
        else:
            keep = sorted(np.argsort(full_scores)[-k_b:].tolist())
        q_in_sub = [q_inputs[offset + j] for j in keep]
        tokens_i = [cand_tokens[i][j] for j in keep]
        scores = [full_scores[j] for j in keep]
        offset += n

        # §5 + §6 — order depends on the diversity metric:
        #   • unigram_jsd needs only token sequences → diversity first, q_forward
        #     only on the N selected traces (saves K_b - N forwards).
        #   • token_jsd needs q logits for distance → q_forward on K_b, then
        #     diversity on the q logits (current path).
        if args.opsd_diversity_metric == "unigram_jsd":
            sel = select_diverse(args, scores, tokens_i, None, device)
            q_in_sel = [q_in_sub[k] for k in sel]
            sel_logits = teacher_forward(model, q_in_sel, resp_len)
        else:
            all_logits = teacher_forward(model, q_in_sub, resp_len)
            sel = select_diverse(args, scores, tokens_i, all_logits, device)
            sel_logits = [all_logits[k] for k in sel]  # N GPU tensors [T, V_full]
            del all_logits

        # §7 — mixture weights using this TP rank's local top-K student tokens.
        # weight_idx is in [0, V_local); teacher logits are fetched from the
        # matching vocab shard so all shapes are consistent.
        top_k = min(args.opsd_weight_top_k, V_local)
        _, weight_idx = torch.topk(p_student / args.opsd_temperature, k=top_k, dim=-1)  # [T, K]
        gathered_list: list[torch.Tensor] = []
        for ltsr in sel_logits:
            # ltsr is [T, V_full] on GPU; take the local shard for this TP rank.
            l_shard = ltsr[:, shard_start : shard_start + V_local]  # [T, V_local]
            gathered_list.append(l_shard.gather(-1, weight_idx))  # [T, K]
        q_gathered = torch.stack(gathered_list, dim=0)  # [N, T, K]
        del gathered_list
        w = mixture_weights(args, p_student, q_gathered, weight_idx)  # [N, T]
        del q_gathered, weight_idx

        # §8 — q_mix for this TP rank's vocab shard [T, V_local].
        # Teacher softmax is computed over the full vocab on GPU (probabilities
        # need to be normalised against the full vocab); only the local shard
        # contributes to q_mix.  Chunked along the time axis to bound the
        # softmax workspace.
        q_mix_local = torch.zeros(T, V_local, dtype=torch.float32, device=device)
        for k, l_gpu in enumerate(sel_logits):
            w_k = w[k]  # [T]
            for t0 in range(0, T, _CHUNK_T):
                t1 = min(t0 + _CHUNK_T, T)
                s_full = F.softmax(l_gpu[t0:t1], dim=-1)  # [chunk, V_full] GPU
                s_local = s_full[:, shard_start : shard_start + V_local]  # [chunk, V_local]
                s_local = s_local * w_k[t0:t1].unsqueeze(-1)
                q_mix_local[t0:t1].add_(s_local)
                del s_full, s_local
        del sel_logits, w

        # §9 — KL(q_mix ‖ p_θ) via vocab-parallel custom autograd.
        # Backward memory = [T, V_local] saved buffer ≈ 360 MB (TP=4).
        kl = _VocabParallelKLDiv.apply(p_student, q_mix_local.detach(), tp_group, _CHUNK_T)
        del q_mix_local

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
    # Student logits stay vocab-parallel [T, V_local]; distillation_loss and
    # _VocabParallelKLDiv are designed to operate on them without gathering.
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
