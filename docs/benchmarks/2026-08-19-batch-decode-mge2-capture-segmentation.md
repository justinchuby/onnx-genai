# Batch decode M≥2 step-cost cliff: fused SwiGLU capture segmentation

**Date:** 2026-08-19
**Hardware:** Intel i7-13800H (14C/20T), RTX 4060 Laptop 8 GB, driver 591.55, CUDA 13.1.
**Thread count:** `--test-threads=1`; benchmark process single-threaded decode loop.
**Models:** `qwen05b-q4` (int4, block-32, **resident** in 8 GB), `qwen14b-zp` (int4, **streamed** via `ONNX_GENAI_WEIGHT_OFFLOAD=1`).
**Binary:** `profile_native` built `--features "bench-native,cuda"`, same binary for every A/B arm.
**Baseline:** `origin/main` @ `cc6a59ae`.

House rule: every number below carries hardware / thread count / model / reference baseline
(see `docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md` §32.2).

---

## Question

After the looped decode GEMV (#1312) removed the MatMulNBits M≥2 *penalty* and the
multi-row-GEMV ceiling probe (#1316) showed weight reads are not the binding cost on a
resident 0.5B, one lead remained: the **~5.4× jump from M=1 (~2.55 ms/step) to M=2
(~14–25 ms/step)** in the *non-matmul* path. Localise where that fixed batch>1 cost goes.

## Instrument

Two additions (shipped as the diagnostic PR; observability only, no behaviour change):

1. **Batch-path per-step phase profiler.** `ONNX_GENAI_PROFILE_CUDA_DECODE_STEPS=1` now
   also fires on the ragged batch decode path (`decode_cuda_greedy_batch_ragged`), emitting
   `[onnx-genai-cuda-step] decode_batch,…` CSV lines that attribute each step to
   `kernel_host_dispatch_ms` (per-node executor `exec_kernel.compute`), `logits_read_sync_ms`,
   `executor_other_ms`, offload phases, etc.
2. **Whole-subgraph vs segmented capture reporting.** The batch sweep now prints
   `native_decode_batch_cuda_graph_segments: batch=N segments=S seam_nodes=…`, folding the
   eager seam nodes that split a segmented capture to `op_type[reason]×count`.

## Measured — localisation (qwen05b-q4, resident, graph capture on)

CUDA graph capture is **live at every batch size** (`captures=1 replays=46 fallbacks=0
invalidations=0` in the measured window) — capture is *not* degrading to eager, and it does
*not* re-capture per step. That hypothesis is **falsified**.

Per-step phase breakdown (`ONNX_GENAI_PROFILE_CUDA_DECODE_STEPS=1`, medians over 48 steps
after warmup):

| batch | total_ms | kernel_host_dispatch_ms | logits_read_sync_ms | executor_other_ms |
|------:|---------:|------------------------:|--------------------:|------------------:|
| 1     | 2.76     | **0.00**                | 2.38                | 0.36              |
| 2     | 25.34    | **22.52**               | 0.82                | 1.68              |
| 4     | 25.56    | **22.24**               | 1.57                | 1.80              |

The M≥2 step is dominated by `kernel_host_dispatch_ms` (the per-node executor
`exec_kernel.compute` phase), which is **0.00 at batch=1** and **~22 ms at batch≥2**, and is
**flat M=2→M=4** (a fixed batch>1 cost, not per-row linear).

Root cause, from the segment reporting:

| batch | segments | seam_nodes                              |
|------:|---------:|-----------------------------------------|
| 1     | 1        | none                                    |
| 2     | 25       | `MatMulNBits[KernelCaptureUnsupported]×24` |
| 4     | 25       | `MatMulNBits[KernelCaptureUnsupported]×24` |
| 8     | 25       | `MatMulNBits[KernelCaptureUnsupported]×24` |

`replay_device_graph` has two paths: a **single-graph** whole-subgraph replay (zero host
work — a bare graph relaunch, batch=1's `kernel_host=0`), and a **segmented** replay that
routes through `run_scoped_mode(RunMode::Replay)`, interleaving segment replays with **eager
seam-node execution through the per-node executor** every step (batch≥2's `kernel_host=22 ms`).

The 24 seams are the **fused gate/up SwiGLU `MatMulNBits` node, one per MLP layer**
(qwen0.5B has 24 layers). At M==1 that node has a capture-safe fused GEMV; at M>1 it has no
capture-safe path and falls through to the tiled prefill GEMM that reports capture-unsafe
(`run_f16_gate_up_swiglu`, `last_call_capture_safe = false`). The merged looped decode GEMV
(#1312) handles the *plain* `run_f16` matmuls at M>1 (they are not seams), but it explicitly
excludes the SwiGLU/decomposed-SiLU epilogues. So at batch≥2 those 24 nodes fragment the
whole-subgraph capture into 25 segments, and the segmented replay re-runs them (and the
scoped-mode overhead) eagerly every step — **that is the entire ~22 ms M≥2 fixed cost.**

## Measured — the fix works (proven, not shipped here)

Fix (preserved on branch `squad/batch-swiglu-capture-fix`, **not merged** — see collision
below): mirror #1312 for the SwiGLU node — for M within the crossover window, loop the
existing capture-safe M==1 fused SwiGLU GEMV once per row. Same binary, back-to-back:

| batch | ms/step before | ms/step after | step speedup | agg tok/s before | agg tok/s after | segments after |
|------:|---------------:|--------------:|-------------:|-----------------:|----------------:|---------------:|
| 1     | 2.57           | 2.99*         | (untouched)  | 390              | 334*            | 1              |
| 2     | 25.1           | **5.15**      | **4.9×**     | 80               | **388**         | **1**          |
| 4     | 24.7           | **9.9**       | **2.5×**     | 162              | **405**         | **1**          |
| 8     | 27.5           | **16.4**      | **1.7×**     | 291              | **489**         | **1**          |

\* batch=1 does not reach the new code (`m > 1` gate); the 2.57↔2.99 difference is run-to-run
variance (batch=1 range already spans 2.5–3.1 ms). Provably untouched.

- **Segments collapse 25 → 1 at every batch size**, seam_nodes → none. The zero-host-work
  single-graph replay is restored for batch decode.
- **Byte-identical:** `native_decode_batch_row_identity: all_rows_equal_row0=true` and
  `native_decode_batch_cross_identity: row0_matches_batch1=true` at N=1,2,4,8 — each batched
  row is byte-for-byte the token stream it would produce run alone as M==1. No ULP slack.
- Aggregate throughput now **scales with N** (334→388→405→489 tok/s) instead of collapsing at
  the M≥2 cliff.

**Predicted vs achieved.** The prediction from the diagnosis — "a single-segment batch replay
costs ≈ batch=1 (2.57 ms) + the extra rows' per-row GEMV work" — put batch=2 at ~5 ms.
Achieved 5.15 ms. The prediction survived contact; the mechanism (segmentation, not weight
reads or launch count) is the right model.

## Measured — model-size dependence (qwen14b-zp, streamed)

On the 14B under weight streaming (`ONNX_GENAI_WEIGHT_OFFLOAD=1`, 8 GB over-subscribed), the
graph is **already fragmented at batch=1**: `segments=96 seam_nodes=MatMulNBits[CaptureRecordingFailed]×289`
— the seam reason is `CaptureRecordingFailed` (streamed weights cannot be recorded into a
CUDA graph), not `KernelCaptureUnsupported`. The resident M≥2 SwiGLU-seam cliff is therefore
**masked** by streaming: batch=1 and batch=2 both show 96 segments / 289 seams, and the step
is streaming-bound (~1300 ms/step at N=1, ~1240 ms/step at N=2). `htod_bytes_per_token`
amortises ~1/N (2.56 GB @ N=1 → 1.32 GB @ N=2), confirming the streaming mechanism is intact.

**Conclusion (model-size dependence):** the M≥2 capture-segmentation cliff is a
**resident-model** phenomenon. It binds every fits-in-VRAM deployment (the production target
for large models on datacentre GPUs) at batch>1, and is directly measurable here on the
resident 0.5B. On this 8 GB box the 14B can only stream, where a different limiter
(`CaptureRecordingFailed` streaming seams + HtoD bandwidth) dominates and the resident fix is
neither helpful nor harmful in a measurable way.

## Disposition — coordination collision (why the fix is not merged here)

The fix lives in `run_f16_gate_up_swiglu`, which is the swiglu team's in-flight area (Estrin
SiLU 1-ULP + capture-safe asserts). It **violates an invariant that team owns and enforces**:
the bit-exact tests assert `last_call_capture_safe == (m == 1)` — *"only M=1 decode may be
advertised capture-safe."* Verified against pristine `origin/main` @ `cc6a59ae`:

- Baseline: **3 failed / 27 passed** (`fp16_gate_up_swiglu_is_bit_exact_to_two_op_path`,
  `fused_gate_up_swiglu_rmsnorm_is_bit_exact_to_two_step_path`,
  `fused_gate_up_swiglu_rmsnorm_zero_points_is_bit_exact_to_two_step_path` — Estrin
  bit-exactness, the swiglu team's in-flight work). `fused_gate_up_swiglu_rmsnorm_fp32_gamma_is_bit_exact_to_two_step_path`
  **passes** on main.
- With the fix: **4 failed / 26 passed** — it **regresses the previously-green
  `…_fp32_gamma…` test** at the capture-safe assertion, and moves the panic site of the other
  three earlier (they were already red).

Landing the fix requires (a) the swiglu team's Estrin bit-exactness fix, and (b) updating the
"only M=1 capture-safe" invariant to admit the looped small-M capture-safe path. Per the house
rule, **do not widen a DRY/capture-safe guard's allowlist unilaterally to get a green test.**
This is a "connect the threads" hand-off to the swiglu team, not a merge.

## Shipped here

Diagnostic instrument only (observability, no behaviour change): the batch-path per-step
phase profiler and the `native_decode_batch_cuda_graph_segments` seam reporting that localised
this cliff. The byte-identical perf fix is preserved on `squad/batch-swiglu-capture-fix`.
