# OPSD Implementation Notes

Source-of-truth files (read in this order):

1. `paper.md` — On-Policy Self-Distillation paper. Authoritative for Algorithm 1, Eq.6/8, Table 6 hyperparams, paper §3.2 per-position KL clip `τ=0.05`, paper §4.1 frozen initial-policy teacher.
2. `method.md` — N-teacher mixture extension on top of paper. Authoritative for §4 quality scoring, §5 diversity selection, §6 teacher forward, §7 mixture weights, §8 q_mix, §9 forward-KL + optional RKL aux.
3. `ALGO.md` — Implementation status, BUG.md fixes, opt-in knobs, monitoring.
4. `slime_plugins/opsd/README.md` — User-facing argument table; must stay in sync with `slime/utils/arguments.py`.
5. `BUG.md` — Diagnosed-and-fixed issue log. Each entry must remain solved by the current code; if you reintroduce a regression, update both this file and BUG.md.
6. `CLAUDE.md` — Top-level project instructions for Claude/agent assistants.

`method.md` ⊇ `paper.md`: paper is `method.md` at `N=1` with `τ_k = y*` (ground truth). When in doubt, prefer `method.md` semantics over conveniences; prefer `paper.md` numerics for hyperparameters.

---

## Implementation guidelines

### Algorithmic contract

- Default loss = `α · L_distill = α · (1/T) Σ KL(q_mix^t ‖ p^t)` (forward, full-vocab). No GRPO PG signal mixed in unless `--opsd-mix-with-policy-loss` is explicitly set.
- Teacher = frozen initial-policy snapshot (`--opsd-freeze-teacher=True` default). Teacher forwards run against the snapshot, not the live student weights.
- Rollout produces **K+1** samples per prompt: 1 random student `y`, the other K go to the privileged candidate pool. The student is by construction never one of the τ_k seen by any teacher.
- Quality scoring uses all three method.md §4 terms: length penalty, format penalty, Conf. Conf is rank-normalized across a sample's candidates by default (`--opsd-quality-conf-norm rank`); change with explicit knob if you want raw log-prob behavior.
- Diversity selection = k-center greedy on the configured distance metric. Default `token_jsd` (method.md §5 recommended); `unigram_jsd` is opt-in and changes selection ordering relative to teacher forwards.
- Mixture-weight softmax is on **raw** distributions (no temperature). Method.md §7 has no temperature term; the `--opsd-temperature` knob is a project extension defaulting to 1.0 (identity).
- Per-position KL clamp `τ=0.05` (`--opsd-jsd-token-clip`) applied AFTER sum-over-vocab, preserving non-negativity. The legacy per-(position, vocab-entry) clip `--opsd-pointwise-kl-clip` is one-sided and can drive per-token KL negative — keep it default-off.

### Numerical / framework invariants

- KL must be non-negative on every step. Monitor `train/opsd_kl` and `train/opsd_kl_clamped`; they should stay equal. If they diverge, investigate clip config.
- `train/opsd_w_entropy` ∈ [0, 1]. Health band: 0.3–0.8. If it pins at 0 or 1, mixture is degenerate — check `[opsd_dbg/mixture]` logs for which logit term saturates (Δ / h / g).
- `train/lr-pg_0/1` reads `param_group["lr"]` directly; should equal `--lr` after the first scheduler step. Do NOT revert to `opt_param_scheduler.get_lr(...)` (BUG.md #5).
- Pipeline-parallel size **must be 1** for OPSD: `plugin.py:before_train_step_hook` calls `model(...)` directly, bypassing pipeline scheduling. TP / DP / CP are fine.
- `--opsd-freeze-teacher=True` requires `--enable-weights-backuper` (validated in `arguments.py`).

### Where each invariant lives

| Invariant | File:Symbol |
|---|---|
| K+1 rollout split | `slime_plugins/opsd/rollout.py:generate_rollout` |
| Privileged pool filtering | `slime_plugins/opsd/rollout.py:_collect_privileged` |
| Quality scoring (Len + Format + Conf) | `slime_plugins/opsd/rollout.py:_structural_score` + `distillation.py:add_conf` |
| Diversity selection | `slime_plugins/opsd/selection.py:kcenter` |
| Mixture weights w_k^t | `slime_plugins/opsd/distillation.py:mixture_weights` |
| Forward-KL custom autograd | `slime_plugins/opsd/distillation.py:_VocabParallelKLDiv` |
| RKL custom autograd | `slime_plugins/opsd/distillation.py:_VocabParallelRKLDiv` |
| Frozen-teacher swap | `slime_plugins/opsd/plugin.py:before_train_step_hook` |
| Loss + metrics composition | `slime_plugins/opsd/plugin.py:loss_function` |

---

## When changing OPSD behavior

1. **Read paper.md / method.md first**. Verify your change is consistent with the algorithmic contract; if it isn't, the change must land as an opt-in knob, not as a default change.
2. **Update `ALGO.md`** if the algorithmic contract or opt-in surface changes.
3. **Update `slime_plugins/opsd/README.md`** if any user-facing argument is added / removed / re-defaulted.
4. **Update `slime/utils/arguments.py`** with matching defaults and help text. Keep the `argparse` help short; longer rationale goes into `ALGO.md`.
5. **Update `BUG.md`** if you fix or introduce a behavior listed there.
6. **Run the diagnostic loop**: launch the 1.7B script for a few train steps, check `[opsd_dbg/mixture]` and `[opsd_dbg/sample]` stdout + `train/opsd_*` TB scalars match the health bands in `ALGO.md` Part 4.

### Soft rules

- Prefer method-consistent semantics over convenience shortcuts when the two conflict.
- Default values must be paper-aligned where paper specifies them; opt-in extensions never change defaults.
- The mixture-weight `w_entropy` derivation in `mixture_weights` is sensitive to Python operator precedence — keep the explicit parens around the negation (`(-(w * log_w).sum(0)).clamp(min=0)`). Do NOT collapse to `-(w * log_w).sum(0).clamp(min=0)` (BUG.md #2 root cause).
- Keep `[opsd_dbg/mixture]` and `[opsd_dbg/sample]` rank-0 diagnostic prints until the health-band monitoring is verified stable in production. Once removed, document the removal in this file.

---

## Quick reference: paper Table 6 vs current scripts

| Knob | paper Table 6 (OPSD) | `run_qwen3_*_opsd.sh` | Justification for delta |
|---|---|---|---|
| Learning rate | `5e-6` | `5e-6` | — |
| Effective batch size | `32` | `32` | — |
| Max completion length | `1024` | `8192` | math reasoning needs longer chains; BUG.md #1 |
| Generations per prompt | `1` | `K+1=17` (1 student + K=16 candidates) | method.md mixture extension; paper is N=1 special case |
| Sampling temperature | `1.1` | `1.1` | — |
| Top-p / Top-k | (not stated) | `0.95 / 20` | OPSD official training scripts |
| KL clip τ | `0.05` (paper §3.2 / Fig.4) | `--opsd-jsd-token-clip 0.05` | — |
| LoRA r/α | `64 / 128` | full fine-tune | slime Megatron path doesn't run LoRA |
| Training steps | `100` | `--num-rollout 1000` (longer schedule) | longer-horizon experiments |
