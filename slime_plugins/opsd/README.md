# OPSD Plugin for slime

On-policy self-distillation (OPSD): the student's own successful reasoning traces
serve as privileged information for token-level mixture-teacher distillation.
By default the teacher is a **frozen snapshot of the initial actor weights**
(paper §4.1), so no separate teacher checkpoint is required.

The training loss is **pure full-vocabulary forward-KL distillation**
(paper Eq. 8 / method.md §9). No PPO/GRPO policy-gradient signal is mixed in
by default — pass `--opsd-mix-with-policy-loss` to ablate the legacy hybrid.

## Algorithm

1. **Rollout**: Sample K trajectories per prompt from the current policy.
2. **Filter**: Keep correct traces `B_x = {τ | R(x, τ) = 1}`.
3. **Quality score**: `s(τ) = 1 - η_l·len/L - η_f·format + η_c·conf`.
   `conf = (1/|τ|) Σ_t log π_T(τ_t | x, τ_<t)` from a dedicated forward over
   `chat(x) + τ_k` (method.md §4).
4. **TopK_b**: Retain top-K_b candidates by full quality score, applied *before*
   the q-teacher forward so discarded candidates are not run through the
   expensive q-forward.
5. **Diversity**: Select N traces `P_x` via k-center greedy (token-JSD by
   default; unigram-JSD as a faster opt-in approximation).
6. **Teacher**: `q_k^t = π_T(·| x, τ_k, y_{<t})` — frozen initial weights, privileged context.
7. **Weights**: `w_k^t ∝ exp(-β·Δ_k^t - γ·h_k^t + ρ·g_k^t)`.
8. **Distill**: `L = α · KL(Σ_k w_k^t q_k^t ‖ p_θ^t)`. With
   `--opsd-rkl-weight α_RKL > 0`, an aux `α_RKL · KL(p_θ ‖ q_mix)` is added
   (method.md §9; the paper recommends `α_RKL ≪ 1`).

## Frozen-teacher mechanism

`--opsd-freeze-teacher` (default `true`) snapshots the initial actor weights at
init time into the `"teacher"` slot of slime's `weights_backuper`.  Implementation
follows **path B**: all teacher-dependent work (Conf, TopK_b, diversity selection,
and `teacher_forward`) runs inside `before_train_step_hook`, **before** the
autograd-tracked student forward starts.

  1. Swap the live module to teacher weights (in-place `param.copy_()`; safe at
     this point because no `SavedTensor` exists yet for the upcoming forward).
  2. Iterate the train step's `data_iterator` over all microbatches; for each
     microbatch run Conf + selection + `teacher_forward`, then detach + offload
     the selected teacher logits to CPU.
  3. Reset the iterator and swap back to actor weights.

`loss_function` then pops the precomputed teacher outputs per microbatch and
runs only the student-dependent parts (mixture weights, `q_mix`, KL).

Memory at peak (Qwen3, N=4, T=2048, V≈152k, bf16): ~600 MB per trace × N × num
microbatches on host RAM; GPU only ever holds one microbatch's teacher logits at
a time, matching the legacy in-loss path.

A previous attempt (**path A**) swapped weights inside `loss_function`, but
`weights_backuper.restore` bumps the parameter's storage version counter via
`param.copy_()`, and Megatron's `RowParallelLinear.backward` raises a version
mismatch on the saved weight tensor.  Path B sidesteps the issue by ensuring
the swap completes before any `SavedTensor` is created.

Conf(τ) is invariant for a frozen teacher, so the plugin caches Conf per
`id(cand_tokens_list)` for the lifetime of a rollout and reuses it across
train steps.

Requires `--enable-weights-backuper` (multi-tag mode).  Mutually exclusive with
`--opd-type=megatron`, which uses the same `"teacher"` tag for a separately-
loaded teacher checkpoint.  Pass `--no-opsd-freeze-teacher` to revert to the
legacy behavior of running teacher forwards against the current student weights.

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
  method.md §5 recommends token_jsd (the default); switch to unigram_jsd if
  q-forward cost dominates wall-clock and you can tolerate the approximation.
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

Defaults align with **method.md** (the project's algorithm spec).  Where
method.md is silent, defaults fall back to **paper.md** (Table 6: OPSD
column).  Project-specific extensions (rank-normalized Conf, unigram-JSD
diversity, hybrid GRPO+OPSD loss, RKL aux) are opt-in.

| Argument | Default | Description |
|---|---|---|
| `--opsd-k` | `8` | Rollout samples per prompt (K) — method.md §13 range 8–32 |
| `--opsd-n` | `2` | Diverse traces to select (N) — method.md §13 range 2–4 |
| `--opsd-kb` | — | Pre-filter top-K_b by quality (method.md §13 range 8–16) |
| `--opsd-alpha` | `1.0` | Scale on L_distill (method.md §9). Default loss is α·L_distill |
| `--opsd-rkl-weight` | `0.0` | Optional reverse-KL aux weight (method.md §9: α_RKL ≪ 1). 0 disables |
| `--opsd-mix-with-policy-loss` | `False` | Opt-in ablation: add the GRPO PG-loss on top of L_distill |
| `--opsd-kl-weight` | `1.0` | Mixture weight β (KL term) |
| `--opsd-entropy-weight` | `0.5` | Mixture weight γ (entropy term) |
| `--opsd-diversity-weight` | `0.5` | Mixture weight ρ (diversity term) |
| `--opsd-temperature` | `1.0` | Softening for mixture-weight computation only; does NOT affect the final KL |
| `--opsd-weight-top-k` | `512` | Vocab truncation for weight computation |
| `--opsd-jsd-token-clip` | — | Per-position KL clamp (post sum-over-vocab); rarely needed |
| `--opsd-pointwise-kl-clip` | `0.05` | Per-(position, vocab-entry) ℓ_{n,v} cap (paper §3.2 / Figure 4 — matches official OPSD scripts). Pass `<= 0` to disable |
| `--opsd-fallback-to-gt` | `True` | Use GT trace when B_x is empty |
| `--opsd-quality-len-weight` | `0.1` | η_l: length penalty |
| `--opsd-quality-format-weight` | `0.2` | η_f: format penalty |
| `--opsd-quality-conf-weight` | `0.5` | η_c: confidence weight |
| `--opsd-quality-conf-norm` | `rank` | Normalize Conf across a sample's candidates so η_c lives on the same [0,1] axis as the structural terms (method.md §4 is silent on Conf's scale; default `η_c=0.5` only makes sense with normalization). Modes: `rank` / `zscore` / `minmax` / `raw` (literal mean log-prob) |
| `--opsd-diversity-metric` | `token_jsd` | `token_jsd` (method.md §5 recommended) or `unigram_jsd` (cheap approximation) |
| `--opsd-diversity-top-k` | `128` | Vocab truncation for token-JSD diversity |
| `--opsd-freeze-teacher` | `True` | Use frozen initial-policy snapshot as the teacher (paper §4.1). Disable with `--no-opsd-freeze-teacher` |

## External teacher (not OPSD, but supported)

If you want to use an *external stronger* model as the teacher (regular
on-policy distillation, not OPSD self-distillation), combine
`--no-opsd-freeze-teacher` with slime's built-in `--use-opd --opd-type
megatron --opd-teacher-load <path>`. The plugin's `switch_model("teacher")`
will then swap to the externally-loaded teacher checkpoint instead of the
frozen self-snapshot.  Note: this is no longer paper.md's OPSD — it's
vanilla on-policy distillation with the OPSD plugin's multi-teacher /
quality / diversity scaffolding bolted on top.

## Logged metrics

In addition to base policy-loss metrics (only when
`--opsd-mix-with-policy-loss` is set), the OPSD plugin emits:

| Metric | Meaning |
|---|---|
| `opsd_kl` | Per-sample mean of KL(q_mix ‖ p_θ) (the raw distillation signal, before α scaling) |
| `opsd_rkl_loss` | Per-sample mean of α_RKL · KL(p_θ ‖ q_mix), when `--opsd-rkl-weight > 0` |
| `opsd_base_loss` | GRPO PG-loss, only when `--opsd-mix-with-policy-loss` is set |
| `opsd_total` | Final loss (α·L_distill + α_RKL·L_RKL + optional base_loss) |
| `opsd_w_entropy` | Mean over (sample, token) of H(w_·^t) / log N ∈ [0, 1]. Near 1 ⇒ the N selected teachers are near-equivalent and the mixture is ~uniform (diversity selection is providing little distinct signal — consider lowering N, raising `--opsd-diversity-weight`, or switching the diversity metric). Near 0 ⇒ a single teacher dominates each position |
