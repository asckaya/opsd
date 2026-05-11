"""Diverse Self-Privileged OPSD plugin entrypoints."""

from .plugin import OPSDPlugin


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    """Rollout function entrypoint."""
    return OPSDPlugin().generate_rollout(args, rollout_id, data_source, evaluation=evaluation)


def init_hook(args):
    """Megatron initialization hook entrypoint."""
    OPSDPlugin().init_hook(args)


def before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler, slime_actor):
    """Megatron before-train-step hook entrypoint (registers the training model)."""
    OPSDPlugin().before_train_step_hook(
        args, rollout_id, step_id, model, optimizer, opt_param_scheduler, slime_actor=slime_actor
    )


def loss_function(args, batch, logits, sum_of_sample_mean):
    """Custom loss function entrypoint."""
    return OPSDPlugin().loss_function(args, batch, logits, sum_of_sample_mean)
