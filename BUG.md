# OPSD memory notes

`VocabParallelKLDiv` / `VocabParallelRKLDiv` run token-chunked forward+backward
over the response dim (default `chunk=256`, knob `--opsd-kl-chunk`) and
recompute softmax in backward instead of saving a `[T, V_local]` grad buffer.
This is what keeps peak GPU memory in check at `V≈152k`, `T=8192`, bf16; with
smaller vocab or shorter T you can raise the chunk size to amortize kernel
launches.
