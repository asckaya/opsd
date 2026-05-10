"""Diverse Self-Privileged OPSD plugin implementation."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from slime.backends.megatron_utils.loss import get_responses, policy_loss_function
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import generate_rollout as base_generate_rollout
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer

from .teacher import OPSDTeacherManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OPSDConstants:
    transition_prompt: str
    boxed_answer_instruction: str


@dataclass(frozen=True)
class OPSDTrace:
    tokens: list[int]
    score: float


class OPSDPlugin(metaclass=SingletonMeta):
    """Diverse Self-Privileged OPSD plugin core."""

    _constants = OPSDConstants(
        transition_prompt=(
            "\n\nAfter reading the reference solution above, make sure you truly understand "
            "the reasoning behind each step -- do not copy or paraphrase it. Now, using your "
            "own words and independent reasoning, derive the same final answer to the problem above. "
            "Think step by step, explore different approaches, and don't be afraid to backtrack "
            "or reconsider if something doesn't work out:\n"
        ),
        boxed_answer_instruction="Please reason step by step and put your final answer within \\boxed{}.",
    )

    def __init__(self) -> None:
        self._model = None
        self._model_chunks = None
        self._tokenizer = None
        self._args = None
        self._teachers = OPSDTeacherManager()

    def _ensure_tokenizer(self, args) -> None:
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    def _set_model(self, model) -> None:
        self._model = model[0] if isinstance(model, list) else model
        self._model_chunks = model if isinstance(model, list) else [model]

    def init_hook(self, args) -> None:
        self._ensure_tokenizer(args)

    def before_train_step_hook(self, args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None:
        self._set_model(model)
        self._teachers.ensure_training_teacher(args, reference_model=self._model)
        if args.opsd_teacher_mode == "ema":
            self._teachers.update_ema(args, self._model_chunks)

    def generate_rollout(self, args, rollout_id, data_source, evaluation=False):
        self._ensure_tokenizer(args)

        if evaluation:
            return base_generate_rollout(args, rollout_id, data_source, evaluation=True)

        opsd_k = args.opsd_k
        opsd_n = args.opsd_n
        opsd_kb = args.opsd_kb

        original_n = args.n_samples_per_prompt
        args.n_samples_per_prompt = opsd_k
        try:
            output = base_generate_rollout(args, rollout_id, data_source, evaluation=False)
        finally:
            args.n_samples_per_prompt = original_n

        if not isinstance(output, RolloutFnTrainOutput):
            return output

        for group in output.samples:
            privileged, scores = self._select_privileged_candidates(args, group, opsd_n, opsd_kb)
            if not privileged and args.opsd_fallback_to_gt:
                privileged = self._fallback_to_ground_truth(group)
                scores = [1.0] if privileged else []
            for sample in group:
                if sample.train_metadata is None:
                    sample.train_metadata = {}
                sample.train_metadata["privileged_candidates"] = privileged
                sample.train_metadata["privileged_candidate_scores"] = scores
                sample.train_metadata["problem_text"] = self._get_problem_text(sample.prompt)

        return output

    def loss_function(self, args, batch, logits, sum_of_sample_mean):
        self._args = args
        self._ensure_tokenizer(args)
        self._teachers.ensure_training_teacher(args, reference_model=self._model)

        base_loss, metrics = policy_loss_function(args, batch, logits, sum_of_sample_mean)

        if not self._model:
            return base_loss, metrics
        if not self._teachers.training_teacher:
            return base_loss, metrics

        metadata_list = batch.get("metadata")
        if metadata_list is None:
            return base_loss, metrics

        unconcat_tokens = batch.get("unconcat_tokens")
        if unconcat_tokens is None:
            logger.warning("OPSD plugin requires unconcat_tokens; skipping OPSD loss.")
            return base_loss, metrics

        response_logits = self._extract_student_responses(logits, batch)
        teacher_inputs, trace_counts, candidate_scores, candidate_tokens = self._build_teacher_inputs(
            batch, metadata_list, unconcat_tokens
        )
        if not teacher_inputs:
            return base_loss, metrics

        opsd_kl = self._compute_opsd_kl(
            args,
            batch,
            response_logits,
            teacher_inputs,
            trace_counts,
            candidate_scores,
            candidate_tokens,
        )
        alpha = args.opsd_alpha
        total_loss = base_loss + alpha * opsd_kl

        metrics.update({"opsd_kl": opsd_kl.detach(), "opsd_total": total_loss.detach()})
        return total_loss, metrics

    def _select_privileged_candidates(
        self,
        args,
        group,
        opsd_n: int,
        opsd_kb: int | None,
    ) -> tuple[list[list[int]], list[float]]:
        if opsd_n <= 0:
            return [], []
        correct = [sample for sample in group if sample.get_reward_value(args) == 1.0]
        if not correct:
            return [], []

        # Rollout keeps only the lightweight structural score; confidence is
        # added later on the training side with the teacher already in memory.
        scored = [
            OPSDTrace(tokens=self._get_response_tokens(sample), score=self._score_trace(args, sample))
            for sample in correct
        ]
        return [trace.tokens for trace in scored], [trace.score for trace in scored]

    def _score_trace(self, args, sample) -> float:
        score = 1.0
        max_len = max(args.rollout_max_response_len, 1)
        resp_len = sample.response_length if sample.response_length else len(sample.tokens)
        score -= args.opsd_quality_len_weight * (resp_len / max_len)
        if "\\boxed{" not in sample.response:
            score -= args.opsd_quality_format_weight
        return score

    def _top_kb(self, traces: list[OPSDTrace], k_b: int) -> list[OPSDTrace]:
        if k_b <= 0 or len(traces) <= k_b:
            return traces
        scores = np.array([trace.score for trace in traces])
        top_idx = np.argsort(scores)[-k_b:]
        return [traces[i] for i in top_idx]

    def _select_diverse_indices(
        self,
        args,
        scores: list[float],
        candidate_tokens: list[list[int]],
        teacher_slice: torch.Tensor,
    ) -> list[int]:
        if not candidate_tokens:
            return []

        max_select = min(args.opsd_n, len(candidate_tokens))
        if max_select <= 0:
            return []

        if args.opsd_diversity_metric == "token_jsd":
            return self._select_diverse_token_jsd(scores, teacher_slice, max_select)

        return self._select_diverse_text(scores, candidate_tokens, max_select)

    def _select_diverse_text(
        self,
        scores: list[float],
        candidate_tokens: list[list[int]],
        max_select: int,
    ) -> list[int]:
        if len(candidate_tokens) <= max_select:
            return list(range(len(candidate_tokens)))

        selected = [int(np.argmax(np.array(scores)))]
        for _ in range(max_select - 1):
            min_jsds = []
            for i, tokens in enumerate(candidate_tokens):
                if i in selected:
                    min_jsds.append(-1.0)
                    continue
                distances = [self._unigram_jsd(tokens, candidate_tokens[j]) for j in selected]
                min_jsds.append(min(distances))
            selected.append(int(np.argmax(min_jsds)))
        return selected

    def _select_diverse_token_jsd(
        self,
        scores: list[float],
        teacher_slice: torch.Tensor,
        max_select: int,
    ) -> list[int]:
        if teacher_slice.size(0) <= max_select:
            return list(range(teacher_slice.size(0)))

        selected = [int(np.argmax(np.array(scores)))]
        for _ in range(max_select - 1):
            min_jsds = []
            for i in range(teacher_slice.size(0)):
                if i in selected:
                    min_jsds.append(-1.0)
                    continue
                distances = [
                    self._token_jsd_distance_from_logits(teacher_slice[i], teacher_slice[j]) for j in selected
                ]
                min_jsds.append(min(distances))
            selected.append(int(np.argmax(min_jsds)))
        return selected

    def _fallback_to_ground_truth(self, group) -> list[list[int]]:
        label = self._get_group_label(group)
        if not label:
            return []
        tokens = self._tokenizer.encode(label, add_special_tokens=False)
        return [tokens] if tokens else []

    def _build_teacher_inputs(
        self, batch, metadata_list, unconcat_tokens
    ) -> tuple[list[torch.Tensor], list[int], list[list[float]], list[list[list[int]]]]:
        teacher_inputs: list[torch.Tensor] = []
        trace_counts: list[int] = []
        candidate_scores: list[list[float]] = []
        candidate_tokens: list[list[list[int]]] = []
        device = batch["tokens"][0].device

        for i, metadata in enumerate(metadata_list):
            traces = metadata.get("privileged_candidates")
            if traces is None:
                traces = []
            problem_text = metadata.get("problem_text", "")
            scores = metadata.get("privileged_candidate_scores")
            if scores is None:
                scores = [1.0] * len(traces)
            if not traces or not problem_text:
                trace_counts.append(0)
                candidate_scores.append([])
                candidate_tokens.append([])
                continue

            response_tokens = unconcat_tokens[i][-batch["response_lengths"][i] :]
            for trace_tokens in traces:
                prompt = self._build_teacher_prompt(problem_text, trace_tokens)
                prompt_ids = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                teacher_inputs.append(torch.tensor(prompt_ids + response_tokens.tolist(), device=device))
            trace_counts.append(len(traces))
            candidate_scores.append(scores)
            candidate_tokens.append(traces)

        return teacher_inputs, trace_counts, candidate_scores, candidate_tokens

    def _build_teacher_prompt(self, problem_text: str, trace_tokens: list[int]) -> str:
        reference = self._tokenizer.decode(trace_tokens, skip_special_tokens=True)
        return (
            f"Problem: {problem_text}\n\n"
            "Here is a reference solution:\n=== Begin ===\n"
            f"{reference}\n=== End ==="
            f"{self._constants.transition_prompt}"
            f"{self._constants.boxed_answer_instruction}"
        )

    def _forward_teacher(
        self,
        teacher_inputs: list[torch.Tensor],
        num_logits_to_keep: int = 0,
    ) -> torch.Tensor:
        max_len = max(tensor.size(0) for tensor in teacher_inputs)
        pad_id = self._tokenizer.pad_token_id or 0
        padded = torch.stack([F.pad(t, (0, max_len - t.size(0)), value=pad_id) for t in teacher_inputs])
        chunk_size = self._args.opsd_teacher_chunk_size
        kwargs = {"logits_to_keep": num_logits_to_keep} if num_logits_to_keep > 0 else {}
        torch.cuda.empty_cache()
        if chunk_size == 0 or chunk_size >= padded.size(0):
            with torch.no_grad():
                return self._teachers.training_teacher(padded, **kwargs).logits

        result = None
        row = 0
        with torch.no_grad():
            for start in range(0, padded.size(0), chunk_size):
                end = min(start + chunk_size, padded.size(0))
                chunk = self._teachers.training_teacher(padded[start:end], **kwargs).logits
                if result is None:
                    # Allocate on CPU to save GPU memory
                    result = torch.empty(padded.size(0), *chunk.shape[1:], dtype=chunk.dtype, device='cpu')
                result[row : row + chunk.size(0)].copy_(chunk, non_blocking=True)
                del chunk
                row += end - start
        return result

    def _compute_opsd_kl(
        self,
        args,
        batch,
        response_logits: list[torch.Tensor],
        teacher_inputs: list[torch.Tensor],
        trace_counts: list[int],
        candidate_scores: list[list[float]],
        candidate_tokens: list[list[list[int]]],
    ) -> torch.Tensor:
        device = response_logits[0].device
        total_kl = torch.tensor(0.0, device=device)
        valid = 0
        offset = 0

        for i, count in enumerate(trace_counts):
            if count == 0:
                continue

            resp_len = int(batch["response_lengths"][i])
            student_logits = response_logits[i]

            # Forward teacher for this sample's candidates only, then discard logits.
            sample_inputs = teacher_inputs[offset : offset + count]
            offset += count
            teacher_logits_i = self._forward_teacher(sample_inputs, num_logits_to_keep=resp_len)
            teacher_slice = teacher_logits_i[:, -resp_len:]

            # Add confidence on the training side so rollout does not need to
            # load a separate teacher instance.
            candidate_scores_i = self._apply_confidence_scores(
                args,
                candidate_scores[i],
                teacher_slice,
                batch["unconcat_tokens"][i][-resp_len:],
            )

            k_b = args.opsd_kb
            if k_b is not None and k_b > 0 and len(candidate_scores_i) > k_b:
                top_idx = np.argsort(np.array(candidate_scores_i))[-k_b:].tolist()
                candidate_scores_i = [candidate_scores_i[j] for j in top_idx]
                candidate_tokens_i = [candidate_tokens[i][j] for j in top_idx]
                teacher_slice = teacher_slice[top_idx]
            else:
                candidate_tokens_i = candidate_tokens[i]

            selected = self._select_diverse_indices(
                args,
                candidate_scores_i,
                candidate_tokens_i,
                teacher_slice,
            )
            if not selected:
                continue
            teacher_slice = teacher_slice[selected].to(student_logits.device)

            weights = self._get_mixture_weights(student_logits, teacher_slice)
            q_mix = 0
            for j in range(teacher_slice.size(0)):
                q_mix = q_mix + weights[j].unsqueeze(-1) * F.softmax(teacher_slice[j], dim=-1)
            kl = (q_mix * (q_mix.clamp(min=1e-10).log() - F.log_softmax(student_logits, dim=-1))).sum(-1)

            if args.opsd_jsd_token_clip is not None:
                kl = kl.clamp(max=args.opsd_jsd_token_clip)
            if "loss_masks" in batch and batch["loss_masks"] is not None:
                mask = batch["loss_masks"][i]
                total_kl += (kl * mask).sum() / mask.sum().clamp(min=1)
            else:
                total_kl += kl.mean()
            valid += 1

        if valid == 0:
            return total_kl
        return total_kl / valid

    def _apply_confidence_scores(
        self,
        args,
        base_scores: list[float],
        teacher_slice: torch.Tensor,
        response_tokens: torch.Tensor,
    ) -> list[float]:
        if not base_scores:
            return []

        target = response_tokens.to(device=teacher_slice.device, dtype=torch.long).unsqueeze(-1)
        conf_scores = []
        for i in range(teacher_slice.size(0)):
            target_logits_i = teacher_slice[i].gather(-1, target).squeeze(-1)
            log_Z_i = torch.logsumexp(teacher_slice[i], dim=-1)
            token_log_probs_i = target_logits_i - log_Z_i
            conf_scores.append(token_log_probs_i.mean(dim=-1))

        conf_weight = args.opsd_quality_conf_weight
        return [base + conf_weight * float(conf.item()) for base, conf in zip(base_scores, conf_scores, strict=False)]

    def _extract_student_responses(self, logits, batch) -> list[torch.Tensor]:
        return [
            chunk
            for chunk, _ in get_responses(
                logits,
                args=self._args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=batch["total_lengths"],
                response_lengths=batch["response_lengths"],
            )
        ]

    def _get_mixture_weights(self, student_logits, teacher_logits) -> torch.Tensor:
        num_refs, num_tokens, vocab = teacher_logits.shape
        temp = self._args.opsd_temperature
        top_k = min(self._args.opsd_weight_top_k, vocab)

        student_scaled = student_logits / temp
        _, idx = torch.topk(student_scaled, k=top_k, dim=-1)

        student_log = F.log_softmax(torch.gather(student_logits, -1, idx) / temp, dim=-1)
        teacher_gathered = torch.gather(teacher_logits, -1, idx.unsqueeze(0).expand(num_refs, -1, -1))
        teacher_log = F.log_softmax(teacher_gathered / temp, dim=-1)
        teacher_prob = teacher_log.exp()

        delta = (teacher_prob * (teacher_log - student_log.unsqueeze(0))).sum(-1).clamp(min=0)
        entropy = -(teacher_prob * teacher_log).sum(-1)

        diversity = torch.zeros(num_refs, num_tokens, device=student_logits.device)
        if num_refs > 1:
            for i in range(num_refs):
                for j in range(i + 1, num_refs):
                    mix = 0.5 * (teacher_prob[i] + teacher_prob[j])
                    mix_log = mix.clamp(min=1e-10).log()
                    jsd = 0.5 * (
                        (teacher_prob[i] * (teacher_log[i] - mix_log)).sum(-1)
                        + (teacher_prob[j] * (teacher_log[j] - mix_log)).sum(-1)
                    ).clamp(min=0)
                    diversity[i] += jsd
                    diversity[j] += jsd
            diversity /= num_refs - 1

        return F.softmax(
            -self._args.opsd_kl_weight * delta
            - self._args.opsd_entropy_weight * entropy
            + self._args.opsd_diversity_weight * diversity,
            dim=0,
        )

    def _get_problem_text(self, prompt) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list) and prompt:
            for message in reversed(prompt):
                if message.get("role") == "user" and message.get("content"):
                    return message["content"]
            return prompt[0].get("content", "")
        return ""

    @staticmethod
    def _get_response_tokens(sample) -> list[int]:
        if sample.response_length:
            return sample.tokens[-sample.response_length :]
        return sample.tokens

    @staticmethod
    def _get_group_label(group) -> str:
        for sample in group:
            label = sample.label
            if isinstance(label, str) and label.strip():
                return label.strip()
        return ""

    @staticmethod
    def _unigram_jsd(tokens_i: Iterable[int], tokens_j: Iterable[int]) -> float:
        tokens_i = list(tokens_i)
        tokens_j = list(tokens_j)
        if not tokens_i or not tokens_j:
            return 1.0

        cnt_i = Counter(tokens_i)
        cnt_j = Counter(tokens_j)
        n_i = len(tokens_i)
        n_j = len(tokens_j)
        jsd = 0.0
        for tok in set(cnt_i) | set(cnt_j):
            p = cnt_i.get(tok, 0) / n_i
            q = cnt_j.get(tok, 0) / n_j
            m = 0.5 * (p + q)
            if m > 0:
                if p > 0:
                    jsd += 0.5 * p * math.log(p / m)
                if q > 0:
                    jsd += 0.5 * q * math.log(q / m)
        return jsd

    def _token_jsd_distance_from_logits(self, logits_i: torch.Tensor, logits_j: torch.Tensor) -> float:
        top_k = min(self._args.opsd_diversity_top_k, logits_i.size(-1))
        top_log_probs_i, top_idx_i = torch.topk(logits_i, k=top_k, dim=-1)
        top_log_probs_j, top_idx_j = torch.topk(logits_j, k=top_k, dim=-1)

        top_log_probs_i = F.log_softmax(top_log_probs_i, dim=-1)
        top_log_probs_j = F.log_softmax(top_log_probs_j, dim=-1)
        p_probs = top_log_probs_i.exp()
        q_probs = top_log_probs_j.exp()

        total = 0.0
        length = min(p_probs.size(0), q_probs.size(0))
        for t in range(length):
            p_map, q_map = self._merge_topk_distributions(p_probs[t], top_idx_i[t], q_probs[t], top_idx_j[t])
            m = 0.5 * (p_map + q_map)
            total += (
                0.5
                * (
                    (p_map * (p_map.clamp(min=1e-10).log() - m.clamp(min=1e-10).log())).sum()
                    + (q_map * (q_map.clamp(min=1e-10).log() - m.clamp(min=1e-10).log())).sum()
                ).item()
            )
        return total / max(length, 1)

    @staticmethod
    def _merge_topk_distributions(
        p_probs: torch.Tensor,
        p_idx: torch.Tensor,
        q_probs: torch.Tensor,
        q_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.cat([p_idx, q_idx]).unique()
        p_map = torch.zeros(tokens.size(0), device=p_probs.device)
        q_map = torch.zeros(tokens.size(0), device=q_probs.device)
        p_pos = {int(tok): idx for idx, tok in enumerate(p_idx.tolist())}
        q_pos = {int(tok): idx for idx, tok in enumerate(q_idx.tolist())}
        for t_i, tok in enumerate(tokens.tolist()):
            if tok in p_pos:
                p_map[t_i] = p_probs[p_pos[tok]]
            if tok in q_pos:
                q_map[t_i] = q_probs[q_pos[tok]]
        p_map = p_map / p_map.sum().clamp(min=1e-8)
        q_map = q_map / q_map.sum().clamp(min=1e-8)
        return p_map, q_map
