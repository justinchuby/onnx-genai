# Batch-N large-model decode scaling on 8 GB: 1/N HtoD is real, wall-clock is limiter-bound

**Date:** 2026-08-18
**Author:** Copilot (multi-request batching slice)
**Owner directive:** "继续推进 … multi-request batching 对大模型的支持，提高速度，实现简洁高效"
— does batch-N deliver the design doc's 1/N weight-stream amortization on a
streaming-bound large model, on real hardware, today?

## Hardware / method (house rule §32.2)

- **Box:** RTX 4060 Laptop 8 GB (driver 591.55, CUDA 13.1), Intel i7-13800H
  (14C/20T). CUDA runtime on PATH via the anaconda `nvidia/cu13` + `cudnn`.
- **EP:** native CUDA (`--features "bench-native,cuda"`), greedy device-argmax
  (`decode_greedy_batch`, no per-step logits D2H).
- **Harness:** `profile_native --native-decode-batch-sweep N1,N2,…`,
  uniform-token fanout (identical token to all N rows), content-invariant
  weight/page/capture counters (#884). Throughput instrument = PR #1291
  (`native_decode_batch_throughput`), median ms/step + min-max range (wall time
  is noisy under streaming pressure #863).
- **Baseline:** batch-1 of the same binary, same run (same-binary A/B).
- **Models:** `qwen14b-zp` (14B int4, ~7–8 GB), `qwen05b-q4` (0.5B int4,
  resident). Every number below names its model.

## Result 1 — `qwen14b-zp`, streaming engaged (`ONNX_GENAI_VRAM_LIMIT=6GiB`)

| N | htod_bytes/token (GB) | htod_bytes/step (GB) | aggregate tok/s | vram_free_ms |
|---|-----------------------|----------------------|-----------------|--------------|
| 1 | 5.11                  | 5.11                 | 0.49            | 9.7 s        |
| 2 | 2.61                  | 5.22                 | 0.87            | —            |
| 4 | 1.34                  | 5.36                 | 0.93            | —            |
| 8 | 0.72                  | 5.79                 | 0.80            | 90 s         |

- **htod_bytes/token tracks 1/N** (within the #866 elastic offset); **per-step
  htod stays ~flat** → the weight is streamed once per step regardless of N.
  **The 1/N amortization mechanism is confirmed by a deterministic,
  contention-invariant counter** (trust over wall-clock, measurement-discipline
  #6).
- **Wall-clock aggregate saturates at N≈2–4 and regresses at N=8.** Cause:
  `vram_free_ms` (VMM unmap churn) explodes 9.7 s → 90 s once
  `mapped_physical_bytes` (8.39 GB @ N=8) exceeds physical VRAM (8.19 GB).
  → **N_max ≈ 4–5 on 8 GB**, far below the doc's projected `N_max ~ 19 @ 2048 ctx`
  (#884/#891). This limiter is VMM map/unmap churn — the offload/VMM slice.

## Result 2 — `qwen05b-q4`, fully resident (htod_bytes = 0, no VMM churn)

| N  | median ms/step | per_row tok/s | aggregate tok/s |
|----|----------------|---------------|-----------------|
| 1  | 2.68           | 373           | 373             |
| 2  | 71–101         | ~14           | ~28             |
| 4  | ~96–100        | ~10           | ~40             |
| 8  | ~96–100        | ~10           | ~84             |
| 16 | ~96–100        | ~10           | ~166            |

- A **~33× per-step cliff at N=1→N=2**, then **flat ~100 ms/step for N=2..16**.
  Batch-16 aggregate (166) is **below** batch-1 (373).
- **Split timing** (forward vs argmax readback) localizes it entirely to the
  forward: N=1 forward 0.2 ms (async graph replay) + readback 2.3 ms; N=2
  forward **50–100 ms (blocking)** + readback 2.7 ms. The readback (device argmax
  + D2H sync) is identical for N=1 and N=2. CUDA-graph counters are identical too
  (captures=1, replays=46, fallbacks=0, invalidations=0) — the cliff is inside
  one replay, not a recapture.

## Root cause (code-level)

`crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs`:
- `m == 1` → specialized decode **GEMV** family (lines 6386 / 6657).
- `m > 1` → **prefill tiled GEMM** (line 6987; header line 2). Batch decode
  (M≥2) takes the prefill path every step.

**Inferred (not nsys-confirmed):** the flat-across-M signature is consistent
with the prefill GEMM's cost being dominated by an N×K grid pass with a tile
height that pads small M, so M=2 costs ~M=16 until M exceeds the tile — a fixed,
occupancy-wasteful full-weight-grid pass per decode step.

## Interpretation

The 1/N *data* mechanism is real; the *wall-clock* win is capped by two
**independent** limiters, which must not be conflated:

- **(A) Streaming regime (8 GB + 14B):** VMM `vram_free` churn (Result 1).
  Offload/VMM slice, not batching.
- **(B) Resident regime (any model that fits):** the M≥2 decode-GEMM cliff
  (Result 2). Batching slice. Compatible-to-fix under streaming (the resident
  weight is read M times from VRAM, not re-streamed → htod 1/N preserved).

On the specific 8 GB + 14B target, (B) is hidden behind ~2000 ms/step streaming;
(A) binds. (B) surfaces once the model is resident.

## Disposition

- Landed: throughput instrument PR #1291 (measurement-only).
- Held for owner sign-off (structural, touches a CUDA kernel dispatch): a
  batched-decode GEMV for small M (loop/broaden the M=1 GEMV, or a true multi-row
  decode GEMV) to close limiter (B). **Not built** — on the 8 GB + 14B target it
  does not move the wall-clock number (VMM-bound), and 简洁高效 forbids a refactor
  that does not move the target's measured number. Owner to pick the regime.
- Flagged for coordination: limiter (A) with the offload/VMM agent.
- Not built: the device-sampling producer (killed earlier, #1282).
