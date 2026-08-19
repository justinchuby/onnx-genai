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

## Fixed per-step batch cost — the input expert-aware MoE batching needs

Route-aware / expert-aware batching (dispatch requests that share MoE experts together, so a
loaded expert serves many requests before eviction) is a **batching** technique: its value is
that a shared-expert batch is cheaper than the same requests run separately. That only pays if
the per-step cost of *being* a batch is small. So the question it must answer first is: **what
does batch-N cost before any clever scheduling, and what would it need to cost to pay?**

Decompose the measured step into a fixed intercept + per-row slope (differencing cancels the
shared work; qwen05b-q4, resident, medians):

| arm | fit `total_ms ≈ fixed + slope·M` | fixed per-step batch cost | marginal per row |
|-----|----------------------------------|--------------------------:|-----------------:|
| **before fix** | 25.1 (M=2), 24.7 (M=4), 27.5 (M=8) | **≈ 22–24 ms** | **≈ 0.4 ms/row** |
| **after fix**  | 5.15 (M=2), 9.9 (M=4), 16.4 (M=8)  | **≈ 1.4 ms**   | **≈ 1.9 ms/row** |

The mechanistic view agrees with the fit: before the fix the fixed cost is the segmented-replay
`kernel_host_dispatch` term (~22 ms, flat M=2→M=4); the marginal per row is tiny only because
the capture-unsafe tiled seam GEMM is itself M-independent for M≤16.

**Interpretation for the scheduler.**

- **Before the fix, batching carries a ~22–24 ms flat penalty that is essentially independent
  of how many requests are in the batch.** A route-aware scheduler optimising the numerator
  (bandwidth saved on expert loads) would be doing so while the denominator is broken: every
  step pays ~22 ms of eager seam replay regardless of how well the batch is packed. On this
  stack, in this state, batching-dependent techniques are gated — the fixed cost would likely
  swamp any expert-load bandwidth saving unless the batch is enormous.
- **The penalty is structural and fixable, not diffuse.** It has a single dominant cause —
  CUDA-graph capture segmentation from the 24 fused-SwiGLU seam nodes — and a byte-identical
  change removes essentially all of it. After the fix the step cost is `≈ 1.4 ms + 1.9 ms/row`:
  a normal linear model where adding a request costs about what its work costs. That is the
  regime in which expert-aware batching can pay.
- **So the ~22 ms fixed batch cost is a precondition for a class of MoE serving techniques, not
  just a throughput number.** Recommendation to whoever owns the routing-trace simulation:
  model the per-step batch cost as **~22 ms fixed (today) vs ~1.4 ms fixed (with this fix
  landed)**, and treat "does expert-aware batching pay?" as conditional on this fix landing.
  Run the offline overlap-vs-FIFO sweep against *both* constants — the answer may invert.

**Backend neutrality (DRY).** *Measured:* native CUDA decode. *Inferred:* the seam and its fix
are in the **shared** CUDA kernel (`MatMulNBits` `capture_support` / `run_f16_gate_up_swiglu` in
`onnx-runtime-ep-cuda`), not in a native-only decode-loop branch. CUDA-graph capture and
`replay_device_graph` live in `onnx-runtime-session`, which both the native and ORT decode
drivers use on the CUDA EP. So the fixed penalty and its removal apply to **both** backends
identically — a route-aware technique built on it would not be left native-only. (The ORT batch
decode path was not separately measured here; that neutrality claim is inferred from the shared
kernel/capture location, not measured.)

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

---

## e2e re-measurement on current `main` (2026-08-19, post-#1404 merge)

**Baseline:** `origin/main` @ `156e1dd8` (#1404 + #1410 landed). Same `profile_native` binary, same box (RTX 4060, CUDA 13.1, driver 591.55). Box was under **heavy external contention** during this run — wall-clock numbers below are indicative and heavily caveated; **segment counts are contention-invariant and definitive**.

**Finding: the penalty is NOT removed by default for resident Qwen 0.5B.** The #1404 fix is correct, but it is **gated off** for these models. Their fused gate/up node runs the **plain** path (`gamma=None`) — confirmed across `qwen05b-q4`, `-fresh`, `-main`, `-symzp`, `-q4-zp` — so the landed gamma-gated per-row GEMV loop never fires and M>1 falls to the capture-unsafe seam GEMM.

Segments (`--native-decode-batch-sweep 1,2,4,8`, contention-invariant):

| config | M=1 | M=2 | M=4 | M=8 |
|---|---|---|---|---|
| **default main** (fold OFF → gate/up plain, `gamma=None`) | 1 | **25** | **25** | **25** |
| `ONNX_GENAI_RMSNORM_MIN_HIDDEN=512` (fold ON → rmsnorm path) | 1 | **1** | **1** | **1** |

Seam at M≥2 default: `MatMulNBits[KernelCaptureUnsupported] ×24`. Forcing the fold on collapses 25→1 at every M and the fix engages; `row0_matches_batch1 = true` (byte-identical per-row).

**Root cause.** `CudaSkipRmsNormMatMulFusion` (optimizer.rs) folds RMSNorm gamma into gate/up only when `norm_size >= RMSNORM_FUSION_MIN_HIDDEN = 1280` (optimizer.rs:1299). qwen05b hidden = **896 < 1280** → fold OFF → `gamma=None`. The floor was calibrated on **M=1** throughput (fold adds a serialized single-warp prologue, ~0.7 ms regression on tiny decoders). At M≥2 the ~20 ms segmentation penalty dwarfs that, so the M=1-calibrated floor makes the wrong call for batch decode. The fold is byte-identical, so a batch-aware gate is numerically safe.

**Cost model when the fix engages** (fold ON, clean quiet window, indicative): M=1=3.5, M=2=5.86, M=4=9.02, M=8=16.98 ms/step → **~1.4 ms fixed + ~1.9 ms/row** (slope M2→M8 = 1.85 ms/row) — reproduces the `after fix` fit above. Default (fold off) stays on the ~20 ms flat penalty.

**Phase breakdown re-check** (per-step CSV). Under contention, only the whole-graph-replay steps sampled cleanly: fold-off's clean steps are the M=1 ones (`kernel_host_dispatch=0`, whole-graph replay); fold-on has clean steps at every M with `kernel_host_dispatch≈0`. Consistent with the mechanistic view: the ~22 ms fixed cost is the segmented-replay `kernel_host_dispatch` term, removed wherever the fix engages. The remaining steady-state per-step cost is dominated by `logits_read_sync_ms` (~2.3–2.9 ms, the lm_head read-back), which becomes the largest single phase once dispatch is gone — the next target if batch decode is pushed further, but it is per-step shared work, not a batch penalty.

**Disposition.** #1404 fix retained (correct). Follow-up to make the fold gate batch-aware tracked in **#1421**; MoE 61%-bytes/token lever remains gated for small resident RMSNorm models until then.

---

## Blast radius — which of our models fall below the floor (2026-08-19)

The floor bites only models whose decode hidden size is `< RMSNORM_FUSION_MIN_HIDDEN = 1280`. Inventory of the local model set (`C:\Users\justinchu\dev\models\`), hidden read from `genai_config.json` where present, otherwise inferred and labelled as such:

| model(s) | decode hidden | source | affected? |
|---|---|---|---|
| qwen05b-q4, -fresh, -main, -q4-acc4, -q4-zp, -symzp, -verify | **896** | genai_config.json (measured) | **YES — below 1280** |
| qwen2.5-0.5b-q4_0-mobius (+ -seqmajor) | **896** | Qwen2.5-0.5B arch (inferred; mobius yaml carries no hidden) | **YES** |
| granite-1b-a400m-f16-mobius (GraniteMoE 1B, **MoE**) | **1024** | 16 heads × 64 head_dim (inferred; no explicit hidden) | **YES** |
| qwen15-moe-dense/-qmoe (f32/oracle/mobius) — **MoE** | 2048 | config.json (measured) | no — folds already |
| qwen14b-*, qwen2.5-14b-* | 5120 | genai_config.json (measured) | no — folds |
| gemma-3-27b-onnx | 5376 (decode) | genai_config.json (measured) | no — folds (the 1152 in config.json is the SigLIP vision tower, not the decode path) |
| qwen15-moe-a27b-gptq-int4-mobius | — | **directory empty on this box; could not measure** | unknown |

Test fixtures under `tests/fixtures/` are synthetic ONNX graphs with no HF `hidden_size`; all are tiny and not deployment-relevant.

**Reading of the impact.** The floor affects the **0.5B dense class (896)** and **granite-1B MoE (1024)** — small resident / edge decode models. Every deployment-sized dense model here (14B at 5120, 27B at 5376) and the **Qwen1.5-MoE fixtures (2048)** are `≥ 1280` and already fold, so they inherit #1404's capture-safe fix for free. The larger-MoE concern is therefore largely moot on this set (qwen15-moe = 2048); the one MoE that is gated off is granite-1B (1024) — and MoE decode is exactly where batching pays, so it is worth noting. Net: #1421 matters for **small resident and small-MoE batch decode**, not fleet-wide.

## Operator tuning knob: `ONNX_GENAI_RMSNORM_MIN_HIDDEN`

Until #1421 resolves the default, an operator serving **batched** decode on a small resident RMSNorm model (hidden `< 1280`: the 0.5B class, granite-1B MoE) can lower this floor to make the RMSNorm→gate/up fold engage, which collapses the batch-decode CUDA-graph segmentation (25 → 1 segments at M≥2) and unlocks #1404's capture-safe path:

```
ONNX_GENAI_RMSNORM_MIN_HIDDEN=512    # any value <= the model's hidden folds it
```

The fold is **byte-identical** (`row0_matches_batch1=true`; parity test `fused_gate_up_swiglu_rmsnorm_is_bit_exact_to_two_step_path`), so tokens are unchanged. **This is not free, and the M=1 cost is hardware-dependent** — set it only when you actually batch:

| regime | M=1 fold on vs off | source |
|---|---|---|
| **RTX 4060** (this box), qwen05b, hidden 896 | **neutral-to-faster** — 2.85 ms folded vs 3.00 ms standalone (clean-floor, contended box → range-min) | measured, this doc |
| **H200**, 0.5B, hidden 896 | **−2.7% regression** at M=1 (814.9 vs 816.1 tok/s), which is *why* the floor exists | commit `05e1fd10` |

So on a large GPU folding a small decoder can regress single-stream M=1; on this small GPU it does not. At M≥2 the ~20 ms/step segmentation saving dwarfs the M=1 cost on either GPU. Do not set this for pure single-stream serving on a large GPU without measuring; do set it (or wait for #1421) for batched decode on small models.
