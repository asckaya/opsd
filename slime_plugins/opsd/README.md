# OPSD Plugin for slime

On-policy self-distillation (OPSD): the student's own successful reasoning traces
serve as privileged information for token-level mixture-teacher distillation.
By default the teacher is a **frozen snapshot of the initial actor weights**
so no separate teacher checkpoint is required.

The training loss is **pure full-vocabulary forward-KL distillation**
(`ALGO.md` Part 1 §1.1 step 8). No PPO/GRPO policy-gradient signal is
mixed in by default — pass `--opsd-mix-with-policy-loss` to ablate the legacy
hybrid.

## Algorithm

1. **Rollout**: Sample K trajectories per prompt from the current policy.
2. **Filter**: Keep correct traces `B_x = {τ | R(x, τ) = 1}`.
3. **Quality score**: `s(τ) = 1 - η_l·len/L - η_f·format + η_c·conf`.
   `conf = (1/|τ|) Σ_t log π_T(τ_t | x, τ_<t)` from a dedicated forward over
   `chat(x) + τ_k`.
4. **TopK_b**: Retain top-K_b candidates by full quality score, applied *before*
   the q-teacher forward so discarded candidates are not run through the
   expensive q-forward.
5. **Diversity**: Select N traces `P_x` via k-center greedy (token-JSD by
   default; unigram-JSD as a faster opt-in approximation).
6. **Teacher**: `q_k^t = π_T(·| x, τ_k, y_{<t})` — frozen initial weights, privileged context.
7. **Weights**: `w_k^t ∝ exp(-β·Δ_k^t - γ·h_k^t + ρ·g_k^t)`.
8. **Distill**: `L = α · KL(Σ_k w_k^t q_k^t ‖ p_θ^t)`. With
   `--opsd-rkl-weight α_RKL > 0`, an aux `α_RKL · KL(p_θ ‖ q_mix)` is added
   (paper recommends `α_RKL ≪ 1`).

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

Memory at peak (Qwen3, N=4, T=2048, V≈152k, bf16): teacher shards are TP-local
[T, V/tp], so host RAM scales as ~600/tp MB per trace × N × num microbatches.
GPU only ever holds one chunk worth of teacher logits at a time during the
fused KL forward+backward, matching the streaming behavior of the legacy
in-loss path.

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
  Token-JSD is the default and the recommended diversity metric; switch to
  unigram-JSD if q-forward cost dominates wall-clock and you can tolerate the
  approximation.
- q_mix is **never** materialised as a persistent [T, V_local] buffer.
  Both forward-KL and reverse-KL run through fused vocab-parallel custom
  autograd ops (`VocabParallelMixture{KL,RKL}Div`) that stream the N teacher
  shards once per pass and reconstruct q_mix at chunk granularity; only a
  [chunk_t, V_local] fp32 accumulator is alive at peak. At T=16384, V=152k
  this saves several GB of resident GPU memory vs. accumulating q_mix first.
  `--opsd-kl-chunk=-1` disables token-dim chunking entirely (chunk = T) when
  memory budget allows; positive values keep the chunked path for OOM
  headroom on tight setups.

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

## Project conventions (where the algorithm spec is silent)

| Convention | Where | Why |
|---|---|---|
| Conf rank-normalized across a sample's candidates | `distillation.py:add_conf` | The spec only gives `Conf = mean log-prob`, no numerical scaling. Raw Conf in [-5,-1] nats overwhelms Len/Format ([0,1] axis), making the default `η_c=0.5` meaningless. `rank → [0,1]` puts the three terms on the same scale. `--opsd-quality-conf-norm raw` restores literal behavior. |
| Top-K=512 vocab approximation for mixture weights | `distillation.py:mixture_weights` | Full-vocab at `N=4 / T=8k` blows GPU memory. The spec doesn't constrain this; it's a pure engineering knob (`--opsd-weight-top-k`). |
| GT-fallback when all candidates are wrong | `rollout.py:_collect_privileged` | Spec says "fallback to GT" but is silent on format; we encode the label as a single trace and proceed normally. |

## Opt-in extensions vs paper defaults

All non-paper behavior is opt-in; default values match paper Table 6 where it
specifies, and `ALGO.md` Part 1 elsewhere.

| Extension | Trigger | Default |
|---|---|---|
| Hybrid distillation + GRPO PG | `--opsd-mix-with-policy-loss` | False (paper is replacement, not addition) |
| Reverse-KL aux term | `--opsd-rkl-weight <α_RKL>` | 0.0 (off) |
| Mixture-weight temperature softening | `--opsd-temperature <T>` | 1.0 (raw) |
| Per-(pos, vocab) KL clip (one-sided; sum-over-vocab can go negative) | `--opsd-pointwise-kl-clip <τ>` | None (off) |
| Cheaper diversity via `unigram_jsd` (selects before q-forward) | `--opsd-diversity-metric unigram_jsd` | `token_jsd` |
| Conf normalization mode | `--opsd-quality-conf-norm rank/zscore/minmax/raw` | `rank` |
| Top-K vocab approximation (mixture / JSD) | `--opsd-weight-top-k` / `--opsd-diversity-top-k` | 512 / 128 |
| Frozen-teacher off (student doubles as teacher) | `--no-opsd-freeze-teacher` | True |
| External teacher (= regular KD, no longer OPSD) | `--no-opsd-freeze-teacher --use-opd --opd-type megatron --opd-teacher-load <path>` | off |

### Relation to slime's built-in `--use-opd`

slime's `--use-opd / --opd-kl-coef` implements paper's alternative objective
(Eq. 9) — a sampled-token PG correction where `reverse_kl = log π_S − log π_T`
plays the role of an advantage modifier in `apply_opd_kl_to_advantages`. That
is a *different* objective from this plugin's main loss:

- slime `--use-opd`: single teacher, sampled-token PG, one scalar signal per token.
- this plugin: N-teacher mixture, **full-vocab** forward-KL, dense per-token signal.

Teacher-loading infrastructure is shared (same `"teacher"` weight tag); the
semantic boundary is `--opsd-freeze-teacher`.

## Hyperparameters

Defaults align with **`ALGO.md` Part 1** (the project's algorithm spec).
Where it is silent, defaults fall back to paper Table 6 (OPSD column, also
in `ALGO.md` §1.2 / Part 2). Project-specific extensions (rank-normalized
Conf, unigram-JSD diversity, hybrid GRPO+OPSD loss, RKL aux) are opt-in.

| Argument | Default | Description |
|---|---|---|
| `--opsd-k` | `8` | Privileged-pool size per prompt (K) — `ALGO.md` §1.3 range 8–32. Total rollouts per prompt = K+1 (1 student + K candidates) |
| `--opsd-n` | `2` | Diverse traces to select (N) — `ALGO.md` §1.3 range 2–4 |
| `--opsd-kb` | — | Pre-filter top-K_b by quality (`ALGO.md` §1.3 range 8–16) |
| `--opsd-alpha` | `1.0` | Scale on L_distill. Default loss is α·L_distill |
| `--opsd-rkl-weight` | `0.0` | Optional reverse-KL aux weight (α_RKL ≪ 1). 0 disables |
| `--opsd-mix-with-policy-loss` | `False` | Opt-in ablation: add the GRPO PG-loss on top of L_distill |
| `--opsd-kl-weight` | `1.0` | Mixture weight β (KL term) |
| `--opsd-entropy-weight` | `0.5` | Mixture weight γ (entropy term) |
| `--opsd-diversity-weight` | `0.5` | Mixture weight ρ (diversity term) |
| `--opsd-temperature` | `1.0` | Softening for mixture-weight computation only; does NOT affect the final KL |
| `--opsd-weight-top-k` | `512` | Vocab truncation for weight computation |
| `--opsd-jsd-token-clip` | `0.05` | Per-position KL clamp post sum-over-vocab (matches official OPSD scripts' `--jsd_token_clip 0.05`). Pass `<= 0` to disable |
| `--opsd-pointwise-kl-clip` | — | Per-(position, vocab-entry) ℓ_{n,v} one-sided cap. NOTE: one-sided clip can drive per-token KL negative when student diverges; prefer `--opsd-jsd-token-clip`. Pass `<= 0` to disable |
| `--opsd-fallback-to-gt` | `True` | Use GT trace when B_x is empty |
| `--opsd-quality-len-weight` | `0.1` | η_l: length penalty |
| `--opsd-quality-format-weight` | `0.2` | η_f: format penalty |
| `--opsd-quality-conf-weight` | `0.5` | η_c: confidence weight |
| `--opsd-quality-conf-norm` | `rank` | Normalize Conf across a sample's candidates so η_c lives on the same [0,1] axis as the structural terms. Modes: `rank` / `zscore` / `minmax` / `raw` (literal mean log-prob) |
| `--opsd-diversity-metric` | `token_jsd` | `token_jsd` (recommended) or `unigram_jsd` (cheap approximation) |
| `--opsd-diversity-top-k` | `128` | Vocab truncation for token-JSD diversity |
| `--opsd-kl-chunk` | `256` | Token-dim chunk for the fused vocab-parallel KL/RKL autograd (q_mix is recomputed per chunk; no persistent [T, V_local] buffer). Pass `-1` to disable chunking (chunk = T) when memory allows; lower the value if you see OOMs |
| `--opsd-freeze-teacher` | `True` | Use frozen initial-policy snapshot as the teacher. Disable with `--no-opsd-freeze-teacher` |

## External teacher (not OPSD, but supported)

If you want to use an *external stronger* model as the teacher (regular
on-policy distillation, not OPSD self-distillation), combine
`--no-opsd-freeze-teacher` with slime's built-in `--use-opd --opd-type
megatron --opd-teacher-load <path>`. The plugin's `switch_model("teacher")`
will then swap to the externally-loaded teacher checkpoint instead of the
frozen self-snapshot.  Note: this is no longer paper's OPSD — it's
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
