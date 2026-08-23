# Real-model greedy goldens

`capture.sh <out.tsv>` runs `greedy_token_dump` over the locally available real
packages and records the greedy token stream for the prompt `"Hello"`, 24 tokens.

It exists so a refactor of the generation path can be diffed against the
*previous* runtime rather than only against itself: `before.tsv` was captured at
`c58eb5b2` (one runtime type, decode core still driving text generation) and any
later capture must match it token for token.

Coverage is whatever this machine has; entries read `MISSING_DIR` when a package
is absent, which keeps a thin run visibly thin instead of silently green.

## `after_review2.tsv`

Captured after the last two migrations: the scheduler's prioritized drive moved
onto the canonical body (`ActiveGenerate` carrying a resolved `CanonicalBody`,
`step_decode_loop` deleted), and the canonical guard hoisted above scheduler
admission on both native paths. Byte-identical to `after_lowering.tsv` and
`after_execution.tsv`, so moving the last callers changed no token on any real
model.
