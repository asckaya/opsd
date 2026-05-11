"""OPSDPlugin — slime hook wiring for on-policy self-distillation."""

from __future__ import annotations

import logging

import torch

from slime.backends.megatron_utils.loss import policy_loss_function
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer

from . import rollout as rollout_module
from .distillation import build_teacher_inputs, distillation_loss, extract_student_responses

logger = logging.getLogger(__name__)


class OPSDPlugin(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._tokenizer = None
        self._args = None

    # ── slime hooks ───────────────────────────────────────────────────────────

    def init_hook(self, args) -> None:
        self._tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    def before_train_step_hook(self, args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None:
        from megatron.core.parallel_state import get_pipeline_model_parallel_world_size

        if get_pipeline_model_parallel_world_size() > 1:
            raise NotImplementedError(
                "OPSD plugin requires pipeline-model-parallel-size=1; teacher_forward and compute_trace_confs call model(...) directly, bypassing pipeline scheduling."
            )
        self._model = model[0] if isinstance(model, list) else model

    # ── rollout ───────────────────────────────────────────────────────────────

    def generate_rollout(self, args, rollout_id, data_source, evaluation=False):
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        return rollout_module.generate_rollout(args, rollout_id, data_source, self._tokenizer, evaluation=evaluation)

    # ── loss ──────────────────────────────────────────────────────────────────

    def loss_function(self, args, batch, logits, sum_of_sample_mean):
        self._args = args

        base_loss, metrics = policy_loss_function(args, batch, logits, sum_of_sample_mean)

        assert self._model is not None, "OPSD: model not set — before_train_step_hook not called"
        assert self._tokenizer is not None, "OPSD: tokenizer not set — init_hook not called"

        metadata_list = batch["metadata"]
        unconcat_tokens = batch["unconcat_tokens"]

        student_logits = extract_student_responses(logits, args, batch)
        q_inputs, conf_inputs, trace_tokens_flat, counts, cand_scores, cand_tokens = build_teacher_inputs(
            batch, metadata_list, unconcat_tokens, self._tokenizer
        )
        if not any(n > 0 for n in counts):
            return base_loss, metrics

        opsd_kl, opsd_metrics = distillation_loss(
            args,
            batch,
            student_logits,
            q_inputs,
            conf_inputs,
            trace_tokens_flat,
            counts,
            cand_scores,
            cand_tokens,
            self._model,
        )
        total_loss = base_loss + args.opsd_alpha * opsd_kl
        metrics.update({"opsd_kl": opsd_kl.detach(), "opsd_total": total_loss.detach()})
        metrics.update(opsd_metrics)
        return total_loss, metrics
