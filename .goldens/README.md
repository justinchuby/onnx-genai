# Real-model greedy goldens

`capture.sh <out.tsv>` runs `greedy_token_dump` over the locally available real
packages and records the greedy token stream for the prompt `"Hello"`, 24 tokens.

It exists so a refactor of the generation path can be diffed against the
*previous* runtime rather than only against itself: `before.tsv` was captured at
`c58eb5b2` (one runtime type, decode core still driving text generation) and any
later capture must match it token for token.

Coverage is whatever this machine has; entries read `MISSING_DIR` when a package
is absent, which keeps a thin run visibly thin instead of silently green.

## `after_prioritized.tsv`

Captured after the scheduler's prioritized drive was moved onto the canonical
body (`ActiveGenerate` carrying a resolved `CanonicalBody`) and `step_decode_loop`
was deleted. Byte-identical to `after_lowering.tsv` and `after_execution.tsv`,
so the last caller to move produced no token change on any real model.
