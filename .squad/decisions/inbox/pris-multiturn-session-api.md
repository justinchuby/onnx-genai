# Decision: Wire multiturn benchmark to native session API

**Date:** 2026-07-29
**Author:** Pris
**Status:** Implemented

## Context

PR #397 added session-persistent KV to the native backend
(`create_native_session` + `generate_native_in_session`), but the `multiturn`
benchmark continued using the stateless `generate()` path. This meant the
headline multi-turn numbers compared ORT with KV reuse against native without
it — unfair and not measuring the feature we shipped.

## Decision

1. Default `run_native_session` to use the session API (KV reuse).
2. Added `--native-stateless` flag for the old re-prefill path.
3. Made the report narrative derive from measured TTFT growth instead of
   hardcoded conclusions about KV persistence.

## Corrected results (Apple M1 Max, load 3.6–5.3)

### Qwen2.5-0.5B-f16

- **Session:** No break-even — native wins at every turn count. 1.13× faster
  over 10 turns. Native TTFT 60 ms vs ORT 150 ms (2.5× faster).
- **Stateless:** Break-even at turn 8. ORT 1.18× faster overall.

### TinyStories-33M (FP32)

- **Session:** Break-even at turn 1–4. ORT 1.5–1.7× faster overall. Native
  TTFT 21 ms vs ORT 27 ms (native wins), but decode is 2× slower.
- **Stateless:** Break-even at turn 3. ORT 2.2× faster overall.

## Flat-TTFT anomaly resolution

The coordinator observed flat native TTFT (195→167 ms) on TinyStories-33M
under stateless mode at load ~2.3. This is explained by: TinyStories-33M's
compute is so small (~33M params, 255 MiB weights) that prefill time is
dominated by fixed overhead (reset, kernel dispatch, cache lookups) rather
than O(context) compute. At lower load the growth is visible: 22→157 ms
(6.9×). For Qwen-0.5B-f16, the growth under stateless is clear: 86→752 ms
(8.7×).

## Impact

Roy's Phase 1 hypothesis that session-persistent KV would make native
competitive is **confirmed for the headline model** (Qwen-0.5B-f16). Native
wins at every turn count. For small FP32 models (TinyStories-33M), the decode
throughput gap (~2×) is now the bottleneck, not prefill.
