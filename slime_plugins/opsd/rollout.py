"""Rollout phase for OPSD (method.md §2-3).

Runs base rollout with K samples per prompt, filters to correct traces,
selects one student y per group (method.md / paper Algorithm 1 line 5:
one student rollout per prompt) and attaches privileged-candidate metadata
for the training step.  The other K-1 traces stay only in the metadata as
the privileged-candidate pool (used downstream for Conf, TopK_b, diversity
and q_k^t teacher targets) — they are NOT trained on directly.
"""

from __future__ import annotations

import logging
import random

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import generate_rollout as base_generate_rollout
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def generate_rollout(args, rollout_id, data_source, tokenizer, evaluation: bool = False):
    if evaluation:
        return base_generate_rollout(args, rollout_id, data_source, evaluation=True)

    original_n = args.n_samples_per_prompt
    args.n_samples_per_prompt = args.opsd_k
    try:
        output = base_generate_rollout(args, rollout_id, data_source, evaluation=False)
    finally:
        args.n_samples_per_prompt = original_n

    if not isinstance(output, RolloutFnTrainOutput):
        return output

    # Deterministic-but-per-rollout-different seeding: derive from rollout_id so
    # the K→1 pick is reproducible across ranks (same rollout_id ⇒ same pick on
    # every DP/TP rank, since base_generate_rollout returns the same group ordering).
    rng = random.Random(rollout_id)

    new_samples: list = []
    for group in output.samples:
        privileged, scores = _collect_privileged(args, group)
        if not privileged and args.opsd_fallback_to_gt:
            privileged = _gt_fallback(group, tokenizer)
            scores = [1.0] * len(privileged)

        meta = {
            "privileged_candidates": privileged,
            "privileged_candidate_scores": scores,
            "problem_text": _extract_problem(group[0].prompt) if group else "",
        }

        student = _pick_student(args, group, privileged, rng)
        if student.train_metadata is None:
            student.train_metadata = {}
        student.train_metadata.update(meta)
        new_samples.append([student])

    output.samples = new_samples
    return output


def _pick_student(args, group, privileged, rng) -> Sample:
    """Choose one trace from the group as the student-y for training.

    See `--opsd-student-pick` arg docs for the rationale of each mode.
    """
    if args.opsd_student_pick == "non_privileged":
        priv_set = {tuple(tr) for tr in privileged} if privileged else set()
        candidates = [s for s in group if tuple(_response_tokens(s)) not in priv_set]
        if candidates:
            return rng.choice(candidates)
        # All K traces were selected into the privileged set (rare: N == K).
        # Fall through to uniform-from-K so we still produce a training sample.
    return rng.choice(group)


def _collect_privileged(args, group) -> tuple[list[list[int]], list[float]]:
    """§2-3: Filter to correct traces and compute structural quality scores."""
    if args.opsd_n <= 0:
        return [], []
    correct = [s for s in group if s.get_reward_value(args) == 1.0]
    if not correct:
        return [], []
    return [_response_tokens(s) for s in correct], [_structural_score(args, s) for s in correct]


def _gt_fallback(group, tokenizer) -> list[list[int]]:
    label = next(
        (s.label for s in group if isinstance(s.label, str) and s.label.strip()),
        None,
    )
    if label is None:
        return []
    ids = tokenizer.encode(label, add_special_tokens=False)
    return [ids] if ids else []


def _structural_score(args, sample) -> float:
    """s(τ) without conf: 1 - η_l·len/L_max - η_f·format_penalty (method.md §4)."""
    score = 1.0
    max_len = max(args.rollout_max_response_len, 1)
    resp_len = sample.response_length if sample.response_length else len(sample.tokens)
    score -= args.opsd_quality_len_weight * (resp_len / max_len)
    if "\\boxed{" not in (sample.response or ""):
        score -= args.opsd_quality_format_weight
    return score


def _response_tokens(sample) -> list[int]:
    return sample.tokens[-sample.response_length :] if sample.response_length else sample.tokens


def _extract_problem(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if msg.get("role") == "user" and msg.get("content"):
                return msg["content"]
        return prompt[0].get("content", "") if prompt else ""
    return ""
