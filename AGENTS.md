# OPSD Implementation Notes

Source of truth for the OPSD method is:

- `../OPSD/metho.md`

Implementation guidelines for this repo:

- Keep the OPSD plugin aligned with `metho.md` before adding extra behavior.
- Prefer method-consistent semantics over convenience shortcuts when the two conflict.
- Quality scoring should follow the method definition:
  - length penalty
  - format penalty
  - teacher confidence term
- Diversity selection should remain k-center greedy over the configured distance metric.
- Mixture-teacher loss should stay token-level and forward-KL based by default.

When changing OPSD behavior:

- Update this file if the intended algorithmic contract changes.
- Keep `slime_plugins/opsd/README.md` consistent with the code path.
