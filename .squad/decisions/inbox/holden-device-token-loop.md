# Decision drop — device-resident token-feedback loop (native CUDA int4 base decode)

**Author:** Holden (CUDA/systems, decode-latency)
**Branch:** `squad/device-token-feedback` (off main `6dfc30cd`)
**Date:** 2026-08-15
**Status:** WIN (modest) — opt-in (`ONNX_GENAI_DEVICE_TOKEN_LOOP=k`), byte-identical, fallbacks=0, 🟡 needs-review PR
**Refs:** Gaff PDL spike (PDL NO-GO; identified the 17.4% cross-replay host-feedback gap as the remaining dominant idle axis)

## Why
Under CUDA-graph capture the native int4 decode has ~0 per-node launch gaps
(capture recovers them). The remaining dominant idle axis Gaff measured is a
cross-replay **host token-feedback gap**: between two captured-graph replays the
HOST is on the critical path — it syncs to read the just-sampled token
(`greedy_result.read_bytes_into` = a D2H stream sync) and writes the next
token/position back to device (H2D) before the next replay. Under nsys this
bounded region is ~17.4% of the steady decode timeline.

## What landed (the lever)
A device-resident token-feedback loop that keeps the sampled token on-device
across `k` captured replays so the host leaves the per-step critical path:

- **New NVRTC kernel `device_token_writer`** (`onnx-runtime-ep-cuda/src/kernels/`):
  single-thread, plain device-to-device. Reads the on-device argmax result
  (`greedy_result[0]` = token id, `[1]` = capture-error word), writes the token
  as i64 into the persistent `input_ids` binding, sets the attention-mask `1` at
  the next position (guarded to physical width), OPTIONALLY writes the next
  `position_ids` (skipped for models like GLM-4 that derive position from the
  mask — no position binding), appends the token to a host-drainable log, and ORs
  the capture-error word into an accumulator. sm-agnostic (no arch guard).
- **Chained-replay orchestration** (`decode_cuda_greedy_loop`): prime step 0 on
  the host (one async H2D, no sync), then `k×` `[replay → device_argmax (launch
  only, no read) → device_token_writer]` back-to-back on the stream, then ONE
  D2H drain of the `k` token ids + OR'd capture-error at the end. Capture-error
  latching is preserved (rejected before consumption at drain).
- **NativeLoopAdapter integration** with a lookahead buffer so the shared decode
  loop keeps per-token EOS/stop/callback semantics; the host drains `k` tokens
  per chain instead of per token.
- EP-trait method (default-unsupported) + CUDA impl + `DeviceIoBinding` wrapper +
  env parse (`ONNX_GENAI_DEVICE_TOKEN_LOOP`, default off; clamp `k≤16`).

Armed only when the topology is device-loopable: graph capture engaged, mask
frozen to physical width (`!mask_exposes_logical`), batch 1, i64 input_ids/mask,
and — when a `position_ids` binding exists — a rank-1 i64 one. Otherwise (and on
inline-route, capacity growth, or `k<2` after clamping) it falls back to the
single captured `decode_cuda_greedy` step. Fallbacks were 0 on both models.

## The parity bug that byte-identity caught (and the fix)
First armed run produced a constant garbage token in the *second* (measured) run
while the *first* (warmup) run was correct. Root cause: the token-writer sets
`mask[next_position]=1` on **every** step, including the last, whose bit lands one
position past `current_len`. Because the mask is frozen to physical width, the
model derives sequence length from the mask 1-count, so that single stray `1`
survived `session.reset()` (which only clears up to `current_len`) and inflated
the derived length of the *next* generation's prefill → corruption. Fix:
`clear_trailing_mask_bit(current_len)` at chain end, leaving the mask in exactly
the state the per-token path leaves. After the fix: byte-identical.

## Parity gate (NON-NEGOTIABLE — PASSED)
Greedy token id sequence with the flag ON is **byte-identical** to flag-OFF:
- **glm-4-9b-int4** (no position binding, mask-derived position): identical at
  k=2, k=4, k=8 over 128 tokens (exact `diff`, 0 lines).
- **qwen2.5-14b-instruct-int4** (real rank-1 position binding, write_position=true):
  identical at k=4 over 128 tokens; loop armed (chained_steps=256, fallbacks=0).

## Performance — the honest result
glm-4-9b-int4, GPU1 of a shared 8×H200 host, `--steady --tokens 128 --warmups 1`.
**Interleaved** off/on runs (5-run medians each, 4 iterations) to cancel host drift:

| iter | off tok/s | k=8 tok/s | delta |
|---|---|---|---|
| 1 | 211.93 | 213.71 | +0.84% |
| 2 | 212.42 | 214.27 | +0.87% |
| 3 | 211.75 | 213.85 | +0.99% |
| 4 | 211.41 | 214.29 | +1.36% |

The two distributions **do not overlap** (max off 212.42 < min k8 213.71), so the
**~1% gain is a real, reproducible signal, not shared-host noise.** k=4 ≈ k=8
(diminishing returns past k=4). fallbacks=0 throughout.

### nsys cross-replay gap DID shrink (confirmed), but ~17% → ~1% in wall-clock
`nsys --cuda-graph-trace=node --sample=none`, steady window (skip 35%),
cross-replay gaps = inter-kernel gaps >50µs:

| | active% | idle% | big-gap count | big-gap sum | big-gap frac |
|---|---|---|---|---|---|
| OFF | 80.4% | 19.6% | 208 (median 737µs) | 138.6ms | 14.86% |
| k=8 | 90.9% |  9.1% |  64 (median 53µs)  |  31.4ms |  3.72% |

The loop closes the GPU-idle gap almost exactly as predicted (208→64 gaps ≈ one
host round-trip per 8 replays instead of per replay; idle 19.6%→9.1%). **But the
wall-clock win is only ~1%, not ~11-17%.** Conclusion: **the 17.4% gap is largely
a profiler artifact** — nsys instrumentation slows the *host*, which inflates the
host-on-critical-path region. In un-profiled execution the host token round-trip
is already mostly overlapped with GPU work, so eliminating it recovers only ~1%.
The optimization is correct and the mechanism works exactly as designed; the
opportunity was simply smaller than the profiler suggested.

## Verdict
Ship as **opt-in** (default off, zero risk to the existing byte-identical path):
a consistent, reproducible, parity-identical ~1% base-decode gain with fallbacks=0
and a sound device-resident-feedback mechanism. It is **not** the decisive
ORT-gap closer the profiler-measured 17.4% implied — reviewers should read the
win as "small but real," and the primary durable value is the honest correction
that the cross-replay host gap is ~1% of real (un-profiled) throughput, not ~17%.

## Portability / capture
- Opt-in only; default-off path untouched and byte-identical to main.
- Kernel is plain d2d copy — sm-agnostic, no arch guard, capture-safe (the
  argmax + writer are launched *between* replays, not inside the captured graph;
  they write the same persistent bindings the graph reads, so graph shape is
  unchanged).
- GLM (no position binding) and qwen (position binding) both covered by the
  optional-position-write path.

## Gates status
- [x] Phase 0 diagnosis reproduced (17.45% nsys gap, host round-trip bounded at
      `greedy_argmax_finalize → gather_bytes`)
- [x] glm byte-identical at k=2/4/8 (128 tok)
- [x] qwen2.5-14b byte-identical at k=4 (128 tok), armed, fallbacks=0
- [x] Interleaved benchmark: consistent non-overlapping ~1% win, fallbacks=0
- [x] nsys before/after: cross-replay gap 14.86%→3.72%
- [x] touched-crate build clean; `onnx-runtime-session` lib tests 156/156;
      engine native_decode tests 94 pass (2 failures are PRE-EXISTING on clean
      tree, unrelated: `native_decoder_requires_explicit_ambiguous_io`,
      `native_decoder_auto_derive_skips_dense_ambiguous_decoder`)

## Reviewer asks
- **Perf/capture reviewer:** confirm the ~1% is worth an opt-in surface given the
  honest "gap was profiler-inflated" finding, and that the between-replay argmax+
  writer launches are capture-safe as argued.
- Opinion on default-k (currently 4) and whether to expose the mechanism at all
  vs. archiving it as a documented NO-GO-on-magnitude (the mechanism is correct;
  the question is whether ~1% justifies the maintenance surface).
