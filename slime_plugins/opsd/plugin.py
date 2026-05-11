"""OPSDPlugin — slime hook wiring for on-policy self-distillation."""

from __future__ import annotations

import logging
from typing import Any

import torch

from slime.backends.megatron_utils.loss import policy_loss_function
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer

from . import rollout as rollout_module
from .distillation import distillation_loss, extract_student_responses, prepare_teacher_outputs

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
        # Per-train-step cache of precomputed teacher outputs, one entry per
        # microbatch.  Populated by `before_train_step_hook` when freeze is on,
        # consumed in order by `loss_function`.
        self._teacher_cache: list[list[dict | None]] = []
        self._mb_counter = 0

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

        # Reset per-step state.
        self._teacher_cache = []
        self._mb_counter = 0
        # New rollout ⇒ new privileged traces ⇒ old Conf cache is dead.
        if step_id == 0:
            self._conf_cache.clear()

        if not args.opsd_freeze_teacher:
            # Legacy "teacher = current student" path: teacher forwards are
            # computed inline from `loss_function`, against the live student
            # weights. Nothing to precompute here.
            return

        # Path B — precompute all teacher outputs for this train step BEFORE
        # the autograd-tracked student forward starts.  In-place weight swaps
        # bump parameter version counters; doing them here is safe because no
        # SavedTensor exists yet for the upcoming forward.
        assert "teacher" in slime_actor.weights_backuper.backup_tags, (
            "OPSD: --opsd-freeze-teacher set but no 'teacher' tag was snapshotted; "
            "the actor init should have produced one."
        )
        assert isinstance(data_iterator, list) and len(data_iterator) == 1, (
            "OPSD requires pipeline-parallel-size=1 / VPP=1; data_iterator must be a " "single-element list."
        )
        iterator = data_iterator[0]

        # Save the iterator's current offset so Megatron's subsequent
        # forward_backward_func sees the same microbatches we precomputed.
        # `train()` only resets iterators once per rollout, so if
        # `num_steps_per_rollout > 1` each step starts where the previous left
        # off — a naive `reset()` here would replay step 0's data forever.
        saved_offset = iterator.offset

        slime_actor.switch_model("teacher")
        try:
            for _ in range(num_microbatches):
                mb = iterator.get_next(["tokens", "response_lengths", "metadata"])
                mb_batch = {
                    "response_lengths": mb["response_lengths"],
                    "unconcat_tokens": mb["tokens"],
                    "metadata": mb["metadata"],
                }
                outputs = prepare_teacher_outputs(
                    args,
                    mb_batch,
                    mb["metadata"],
                    mb["tokens"],
                    self._tokenizer,
                    self._model,
                    self._conf_cache,
                    offload_to_cpu=True,
                )
                self._teacher_cache.append(outputs)
            iterator.offset = saved_offset
        finally:
            slime_actor.switch_model("actor")

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

        student_logits = extract_student_responses(logits, args, batch)

        if args.opsd_freeze_teacher:
            # Pop the precomputed (frozen-teacher) outputs for this microbatch.
            sample_teacher_outputs = self._teacher_cache[self._mb_counter]
            self._mb_counter += 1
        else:
            # Legacy: compute teacher outputs inline against the current student
            # weights, on the same device (no CPU offload).
            sample_teacher_outputs = prepare_teacher_outputs(
                args,
                batch,
                batch["metadata"],
                batch["unconcat_tokens"],
                self._tokenizer,
                self._model,
                self._conf_cache,
                offload_to_cpu=False,
            )

        if not any(o is not None for o in sample_teacher_outputs):
            return base_loss, metrics

        opsd_kl, opsd_metrics = distillation_loss(args, batch, student_logits, sample_teacher_outputs)

        total_loss = base_loss + args.opsd_alpha * opsd_kl
        metrics.update({"opsd_kl": opsd_kl.detach(), "opsd_total": total_loss.detach()})
        metrics.update(opsd_metrics)
        return total_loss, metrics
