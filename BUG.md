# Known memory-vs-compute trade-offs in OPSD

Places where we deliberately picked a less-efficient algorithm to keep peak GPU
memory low.  Listed so it's easy to find candidates to revisit when (a) GPU
memory grows, (b) compute becomes the new bottleneck, or (c) we switch to a
larger vocab / longer T that changes the trade-off.

## `slime_plugins/opsd/distillation.py`

### 1. Token-chunked KL / RKL forward and backward

`VocabParallelKLDiv` and `VocabParallelRKLDiv` loop over the token dim in
`chunk_t`-sized slices (default 256, configurable via `--opsd-kl-chunk`) instead
of running the math once on the full `[T, V_local]` tensor.

Cost: extra kernel launches (≈ `ceil(T / chunk_t)` per op).  At `T=2048,
chunk_t=256` that's 8 launches per chunk-op — small relative to the model
fwd/bwd, large relative to a single fused softmax.

If we ever run on a setup where `T × V_local × bf16_size` comfortably fits in
peak memory (small vocab, large GPU), drop the loop and use one-shot ops.

### 2. Recompute-in-backward for KL and RKL

The forwards save only `(logit_local, q_local, lse, …)` — light or already
pinned by autograd — and the backwards recompute `softmax` chunk-by-chunk.

Cost: roughly 2× compute on the loss op itself (LM head loss only — not the
model).  In exchange we avoid keeping `[T, V_local]` saved-for-backward
tensors alive between fwd and bwd:

  * KL:  −1 × `[T, V_local]` bf16 grad buffer  (~600 MB at TP=1, V≈152k, T=2k)
  * RKL: −2 × `[T, V_local]` bf16 buffers      (~1.2 GB at same shape)

If activation memory is no longer the bottleneck, we can flip back to saving
the precomputed gradient tensors and skipping the backward recompute.

### 3. Chunked `q_mix` accumulation

In `distillation_loss`, `q_mix_local` is built by iterating over the N selected
teacher logits and accumulating chunked softmax shards.  Same trade-off as
above: more kernel launches in exchange for not allocating the full
`[c, V_full]` softmax buffer for all chunks at once.

### 4. `q_mix_local` accumulated in fp32, cast to student dtype before KL

Accumulation stays in fp32 for numerical safety (small N weighted softmaxes
summing to 1), but we cast down to `p_student.dtype` (bf16) before passing
into the KL ops so every inner-loop temporary inside the KL chunk loop stays
bf16 instead of being upcast to fp32.

Cost: one extra `[T, V_local]` allocation during the cast (fp32 → bf16) — the
old fp32 buffer goes garbage immediately after.  Negligible vs the chunk-loop
temporaries this saves.
