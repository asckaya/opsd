# OPSD memory notes

`VocabParallelMixtureKLDiv` / `VocabParallelMixtureRKLDiv` run token-chunked
forward+backward over the response dim (default `chunk=256`, knob
`--opsd-kl-chunk`; `-1` disables chunking). q_mix is **fused** into the autograd
op — both passes recompute it on the fly per chunk from the saved teacher
shards + mixture weights, so OPSD never holds a persistent `[T, V_local]`
q_mix buffer (which was several GB at `V≈152k`, `T=16k`, fp32). Backward also
recomputes softmax(p) instead of saving a `[T, V_local]` grad buffer. Combined
with TP-local teacher logits (no `[T, V_full]` all-gather) and per-chunk
streaming gathers, peak GPU memory stays in check at long T; raise the chunk
size or pass `-1` when there is headroom.
