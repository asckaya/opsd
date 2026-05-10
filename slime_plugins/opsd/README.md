# Diverse Self-Privileged OPSD Plugin for slime

This plugin implements the **Diverse Self-Privileged OPSD** algorithm, which leverages a student model's own successful reasoning traces as privileged information for on-policy self-distillation.

## Features

- **On-policy sampling**: Generates $K$ samples per prompt.
- **Diversity selection**: Selects $N$ diverse correct traces using k-center greedy (Unigram JSD).
- **Mixture Teacher**: Constructs a token-level mixture teacher from privileged traces.
- **Adaptive Weighting**: Computes per-token mixture weights based on KL, Entropy, and Diversity.
- **Megatron Integration**: Plugs into slime's Megatron backend via custom hooks and loss functions.

## Components

- `plugin.py`: Rollout selection, hook wiring, and mixture-teacher loss.
- `teacher.py`: Lazy training-teacher loading and optional EMA updates.

## Usage

To use this plugin, you can refer to the example script: `scripts/run_qwen3_1.7B_opsd.sh`.

Key arguments:

```bash
# Rollout
--rollout-function-path slime_plugins.opsd.generate_rollout

# Hooks
--custom-megatron-init-path slime_plugins.opsd.init_hook
--custom-megatron-before-train-step-hook-path slime_plugins.opsd.before_train_step_hook

# Loss
--loss-type custom_loss
--custom-loss-function-path slime_plugins.opsd.loss_function

# OPSD Parameters
--opsd-k 16                    # Number of samples to generate (K)
--opsd-n 8                     # Max privileged traces to select (N)
--opsd-kb 16                   # Pre-filter top K_b by quality
--opsd-alpha 1.0               # OPSD loss weight (alpha)
--opsd-kl-weight 1.0           # Mixture weight: KL coefficient (beta)
--opsd-entropy-weight 0.5      # Mixture weight: Entropy coefficient (gamma)
--opsd-diversity-weight 0.5    # Mixture weight: Diversity coefficient (rho)
--opsd-temperature 1.0         # Temperature for mixture weight computation
--opsd-weight-top-k 512        # Vocab truncation for mixture weights
--opsd-jsd-token-clip 10.0     # Clip JSD per token
--opsd-fallback-to-gt          # Use GT traces if no correct self-generated traces
--opsd-quality-len-weight 0.1  # Quality length penalty weight (eta_l)
--opsd-quality-format-weight 0.2 # Quality format penalty weight (eta_f)
--opsd-quality-conf-weight 0.5 # Confidence weight; computed on the training side
--opsd-diversity-metric token_jsd # Diversity distance for k-center selection
--opsd-diversity-top-k 128      # Top-K vocab truncation for token-level JSD
--opsd-teacher-mode ema         # Teacher mode: ema or frozen
--opsd-ema-decay 0.999          # EMA decay when teacher mode is ema
--opsd-teacher-chunk-size 8     # Chunk size for teacher forward (0 disables)
```

## Algorithm Summary

1.  **Sampling**: For each prompt $x$, sample $K$ trajectories $\tau_j \sim \pi_{\theta_{old}}$.
2.  **Filtering**: Keep correct trajectories $B_x = \{ \tau_j | R(x, \tau_j) = 1 \}$.
3.  **Selection**: Select $N$ diverse trajectories $P_x = \text{TopK}(B_x, \text{diverse\_k\_center})$.
4.  **Teacher**: For each $\tau \in P_x$, the teacher is $q_\tau^t = \pi_{\theta_T}(\cdot | x, \tau, y_{<t})$.
5.  **Distillation**: Minimize $\mathbb{E}_{y \sim \pi_\theta} [ KL(\sum w_\tau^t q_\tau^t || \pi_\theta) ]$.
