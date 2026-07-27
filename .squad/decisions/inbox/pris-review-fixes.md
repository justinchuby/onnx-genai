# Pris — PR #227 reviewer-comment fixes

**Date:** 2026-07-27
**Author:** Pris
**Scope:** `crates/onnx-genai-bench/src/bin/compare.rs`

## Fix 1 — `--decode-skip 0` decode window

Extracted `decode_throughput()` helper and fixed the skip==0 path to subtract
`Duration::ZERO` instead of `token_times[0]`. The old code used
`saturating_sub(token_times[decode_skip.saturating_sub(1)])` which, for skip==0,
still subtracted `token_times[0]` (TTFT), inflating tok/s.

Added `decode_throughput_skip_0_1_2` test with a synthetic 5-token series (500 ms
TTFT + 100 ms cadence). The test asserts exact window and tok/s at skip=0, 1, 2,
and the too-few-tokens boundary. Guard-break verified: reintroducing the old
`saturating_sub` expression causes the test to fail at skip=0.

**Published numbers unaffected:** The profile README uses `--decode-skip 2`; no
committed figure was produced with `--decode-skip 0`.

## Fix 2 — `--profile-json -` invalid JSON in non-direct mode

Mirrored the direct-mode pattern: when `--profile-json -`, send the Markdown
report to stderr and only write JSON to stdout. Previously both went to stdout,
producing output that is not valid JSON.
