# Decision: M≥2 batch-decode cliff is fused-SwiGLU capture segmentation (resident regime)

**Author:** copilot (multi-request batching slice)
**Date:** 2026-08-19
**Hardware:** i7-13800H (14C/20T), RTX 4060 Laptop 8 GB, CUDA 13.1. Baseline `origin/main` @ `cc6a59ae`.

## What was measured

The ~5.4× M=1→M=2 batch-decode step-cost jump on a **resident** model (`qwen05b-q4`) is **CUDA
graph capture segmentation**, not launch count, sampling, KV bookkeeping, or attention:

- CUDA graph capture is **live** at M=1/2/4/8 (`captures=1 replays=46 fallbacks=0
  invalidations=0`) — the "capture degrades at M≥2" hypothesis is falsified.
- Batch=1 captures as a **single whole-subgraph** (`segments=1`, zero-host-work replay,
  `kernel_host_dispatch=0.00 ms`, step 2.57 ms).
- Batch≥2 fragments into **25 segments** with 24 seam nodes =
  `MatMulNBits[KernelCaptureUnsupported]×24` — the **fused gate/up SwiGLU MatMulNBits, one per
  MLP layer** (qwen0.5B = 24 layers). It has no capture-safe path at M>1 (falls through to the
  capture-unsafe tiled prefill GEMM), so the segmented replay re-runs those seams **eagerly
  through the per-node executor every step**: `kernel_host_dispatch ≈ 22 ms`, flat M=2→M=4 —
  the entire M≥2 fixed cost.

## The fix (proven, byte-identical, NOT merged)

Mirror the merged looped decode GEMV (#1312): for M within the crossover window, loop the
capture-safe M==1 fused SwiGLU GEMV once per row. Measured (same binary, byte-identical rows):
segments **25 → 1** at every batch size; batch-2 **25.1 → 5.15 ms/step (4.9×)**, aggregate
**80 → 388 tok/s**; batch-4 2.5×; batch-8 1.7×. Row- and cross-identity byte-exact at all N.

## Why it is not merged (coordination)

The fix is in `run_f16_gate_up_swiglu` — the swiglu team's in-flight area — and violates their
`last_call_capture_safe == (m == 1)` invariant ("only M=1 capture-safe"). It regresses the
previously-green `fused_gate_up_swiglu_rmsnorm_fp32_gamma_is_bit_exact` test (main: 3 failed/27
passed → with fix: 4 failed/26 passed). Landing needs the swiglu team's Estrin bit-exactness
fix **and** an invariant update to admit the looped small-M capture-safe path. **Do not widen
the guard's allowlist unilaterally.** Preserved on branch `squad/batch-swiglu-capture-fix` for
hand-off.

## Model-size dependence

The cliff is **resident-only**. On a streamed 14B (`qwen14b-zp`, 8 GB over-subscribed) the
graph is already 96 segments / 289 `CaptureRecordingFailed` seams at batch=1 (streamed weights
can't be graph-recorded), the step is streaming-bound (~1250 ms), and `htod_bytes_per_token`
amortises ~1/N (2.56 → 1.32 GB, N=1→2). The resident fix is neither helpful nor harmful there.
The resident regime is the production target for large models (they fit in datacentre VRAM);
the resident 0.5B is the representative instrument on this box.

## Shipped

Diagnostic instrument only: batch-path per-step phase profiler
(`ONNX_GENAI_PROFILE_CUDA_DECODE_STEPS=1` on the ragged batch path) and
`native_decode_batch_cuda_graph_segments` seam reporting. Full write-up:
`docs/benchmarks/2026-08-19-batch-decode-mge2-capture-segmentation.md`.

## Cross-references

- #1312 (looped decode GEMV, the pattern this fix mirrors), #1316 (multi-row ceiling probe).
- Adjacent: the swiglu team's Estrin/capture-safe work in `run_f16_gate_up_swiglu`.

## Input for expert-aware MoE batching (owner request)

The fixed per-step batch cost — the constant a route-aware scheduler must overcome — is
**~22-24 ms (resident qwen05b-q4)** before the SwiGLU-capture fix, essentially independent of
batch size M (marginal ~0.4 ms/row). It is **structural (capture segmentation), not diffuse**,
and the byte-identical fix removes it: after the fix the step is `~1.4 ms fixed + ~1.9 ms/row`.

Disposition for the MoE routing-trace simulation: model per-step batch cost as **~22 ms fixed
(today) vs ~1.4 ms fixed (fix landed)** and treat "does expert-aware batching pay?" as
conditional on the fix. Backend-neutral (shared CUDA EP kernel; ORT path inferred, not
measured). Fix ready on `squad/batch-swiglu-capture-fix`, blocked on the swiglu team's Estrin
bit-exactness fix + a capture-safe invariant update. Raised priority: this now gates a class of
MoE serving techniques, not just batch throughput.
