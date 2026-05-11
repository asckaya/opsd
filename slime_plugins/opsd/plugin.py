"""OPSDPlugin — slime hook wiring for on-policy self-distillation."""

from __future__ import annotations

import logging
from typing import Any

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
        # MegatronTrainRayActor — typed as Any to avoid a circular import with
        # the slime backend; we only ever call `.role`, `.weights_backuper`, and
        # `.switch_model(tag)` on it.
        self._actor: Any = None
        # Frozen teacher ⇒ Conf(τ_k) is invariant for a given trace, so we can
        # cache it across train steps within a rollout. Keyed by `id(cand_tokens_list)`
        # since rollout.py shares the same Python list across the n_samples_per_prompt
        # samples of a group and that list survives for the rollout's lifetime.
        self._conf_cache: dict[int, list[float]] = {}

    # ── slime hooks ───────────────────────────────────────────────────────────

    def init_hook(self, args) -> None:
        self._tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    def before_train_step_hook(
        self,
        args,
        rollout_id,
        step_id,
        model,
        optimizer,
        opt_param_scheduler,
        data_iterator,
        num_microbatches,
        slime_actor,
    ) -> None:
        from megatron.core.parallel_state import get_pipeline_model_parallel_world_size

        # The same hook path may be invoked from the critic's train loop; OPSD
        # is actor-only, so silently skip non-actor roles.
        if slime_actor.role != "actor":
            return

        if get_pipeline_model_parallel_world_size() > 1:
            raise NotImplementedError(
                "OPSD plugin requires pipeline-model-parallel-size=1; teacher_forward and compute_trace_confs call model(...) directly, bypassing pipeline scheduling."
            )
        self._model = model[0] if isinstance(model, list) else model
        self._actor = slime_actor
        # data_iterator / num_microbatches are accepted for the upcoming path-B
        # implementation (precompute teacher outputs in this hook with the
        # frozen teacher loaded). The in-loss swap that was tried first bumps
        # parameter version counters inside Megatron's autograd window and
        # trips the SavedVariable version check during backward, so it has
        # been removed; the frozen-teacher path is currently inert.
        del data_iterator, num_microbatches
        if step_id == 0:
            self._conf_cache.clear()

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

        # NOTE: the frozen-teacher swap (path A: swap weights inside this loss
        # function) was removed because `weights_backuper.restore` does an
        # in-place `param.copy_()`, which increments the storage version
        # counter. Autograd captured the student weight at version V during
        # the outer forward; backward then sees version V+2 and aborts with
        # "one of the variables needed for gradient computation has been
        # modified by an inplace operation". Path B (precompute teacher
        # outputs in `before_train_step_hook`, fully outside the autograd
        # window) will replace this once wired. For now teacher forwards run
        # against the current student weights — set `--no-opsd-freeze-teacher`
        # in scripts to match this behavior and skip the unused snapshot.
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
            conf_cache=self._conf_cache,
        )

        total_loss = base_loss + args.opsd_alpha * opsd_kl
        metrics.update({"opsd_kl": opsd_kl.detach(), "opsd_total": total_loss.detach()})
        metrics.update(opsd_metrics)
        return total_loss, metrics
