# OPSD Plugin for slime

On-policy self-distillation (OPD): the student's own successful reasoning traces
serve as privileged information for token-level mixture-teacher distillation.
No separate teacher model — the training model itself is used for teacher inference.

## Algorithm

1. **Rollout**: Sample K trajectories per prompt from the current policy.
2. **Filter**: Keep correct traces `B_x = {τ | R(x, τ) = 1}`.
3. **Quality score**: `s(τ) = 1 - η_l·len/L - η_f·format + η_c·conf`.
   `conf = (1/|τ|) Σ_t log π_θ(τ_t | x, τ_<t)` from a dedicated forward over
   `chat(x) + τ_k` (metho.md §4).
4. **TopK_b**: Retain top-K_b candidates by full quality score, applied *before*
   the q-teacher forward so discarded candidates are not run through the
   expensive q-forward.
5. **Diversity**: Select N traces `P_x` via k-center greedy (unigram-JSD or token-JSD).
6. **Teacher**: `q_k^t = π_θ(·| x, τ_k, y_{<t})` — same weights, privileged context.
7. **Weights**: `w_k^t ∝ exp(-β·Δ_k^t - γ·h_k^t + ρ·g_k^t)`.
8. **Distill**: `L = KL(Σ_k w_k^t q_k^t ‖ p_θ^t)`.

## Constraints

- Requires `--pipeline-model-parallel-size 1`. The teacher and conf forwards
  call `model(...)` directly, bypassing pipeline scheduling.

## Performance notes

- Per privileged candidate the plugin runs **two** teacher forwards: one over
  `chat(x) + τ_k` for Conf (§4), one over `chat(privileged(x, τ_k)) + y` for
  `q_k^t` (§6). Conf results are cached per rollout group, so within a group
  of `n_samples_per_prompt` samples Conf is computed once and reused.
- Setting `--opsd-kb < --opsd-k` cuts q-forward cost proportionally: the
  TopK_b filter runs between the (cheap) Conf forward and the (expensive)
  q-forward, so discarded candidates never reach the q-forward.
- With `--opsd-diversity-metric unigram_jsd` (token-only distance), the
  diversity selection happens BEFORE the q-forward and q-forward runs only on
  the N selected traces — saving (K_b - N) full q-forwards per sample.
  metho.md §5 recommends token_jsd, but the unigram approximation is much
  cheaper and is the default in the provided scripts.
- q_mix is built with GPU-side softmax over the full vocab (peak GPU delta
  ≈ [chunk, V_full] float32 ≈ 150 MB at chunk=256, V=152k).

## Files

| File | Responsibility |
|---|---|
| `__init__.py` | Thin slime entry-point wrappers |
| `plugin.py` | `OPSDPlugin` singleton — hook wiring and loss orchestration |
| `rollout.py` | Rollout phase: K-sample generation, correct-trace filtering, metadata attachment |
| `distillation.py` | Training phase: teacher forward, quality scoring, diversity selection, KL loss |
| `selection.py` | k-center greedy + pairwise JSD distance utilities |

## Usage

See `scripts/run_qwen3_1.7B_opsd.sh` for a complete example.

```bash
--rollout-function-path slime_plugins.opsd.generate_rollout
--custom-megatron-init-path slime_plugins.opsd.init_hook
--custom-megatron-before-train-step-hook-path slime_plugins.opsd.before_train_step_hook
--loss-type custom_loss
--custom-loss-function-path slime_plugins.opsd.loss_function
```

## Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--opsd-k` | — | Rollout samples per prompt (K) |
| `--opsd-n` | — | Diverse traces to select (N) |
| `--opsd-kb` | — | Pre-filter top-K_b by quality |
| `--opsd-alpha` | 1.0 | Distillation loss weight |
| `--opsd-kl-weight` | 1.0 | Mixture weight β (KL term) |
| `--opsd-entropy-weight` | 0.5 | Mixture weight γ (entropy term) |
| `--opsd-diversity-weight` | 0.5 | Mixture weight ρ (diversity term) |
| `--opsd-temperature` | 1.0 | Temperature for weight computation |
| `--opsd-weight-top-k` | 512 | Vocab truncation for weight computation |
| `--opsd-jsd-token-clip` | — | Per-token KL clip value |
| `--opsd-fallback-to-gt` | false | Use GT trace when B_x is empty |
| `--opsd-quality-len-weight` | 0.1 | η_l: length penalty |
| `--opsd-quality-format-weight` | 0.2 | η_f: format penalty |
| `--opsd-quality-conf-weight` | 0.5 | η_c: confidence weight |
| `--opsd-diversity-metric` | `token_jsd` | `token_jsd` or `unigram_jsd` |
| `--opsd-diversity-top-k` | 128 | Vocab truncation for token-JSD diversity |
