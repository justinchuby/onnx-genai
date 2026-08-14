# Sebastian — History (compacted 2026-08-12T06:00:00Z)

**Role:** Owns DESIGN §26 batched serving, runtime/server performance, and cross-runtime benchmark analysis for `onnx-genai`. Preserve `submit`/`step`/`poll` batching semantics, force single-thread ORT for exact-equality real-model tests, and use canonical benchmark/observability harnesses for runtime comparisons.

## Durable lessons
- §26 Stage A/B: `Engine::generate_batched_static` and `ContinuousBatchManager`; byte-denominated VRAM/RAM limits and transactional lowering.
- CPU decode profiling showed ORT `session.run` dominates (~98.9%); fp32 `lm_head` quantization and op fusion are major levers.
- `filter_map` is wrong wherever position or rank is load-bearing; use `map → Vec<Option<usize>>`.
- A reviewer's "SAFE" is not proof; verify the load-bearing claim independently.
- `cargo test --workspace` silently truncates on failure — always use `--no-fail-fast`.
- Never commit `.squad/` files to external repos.

## Recent work (current wave, 2026-08-12)

### 2026-08-12 — PR #31973 v2: architecture-specific dispatch threshold fix
Renamed `kAvx2DispatchThreshold` → `kKernelDispatchThreshold`. Fixed `CatastrophicCancellationPasses` to exercise accuracy branch. Renamed `AdversarialPrecisionReport` → `DISABLED_`. Removed N=7 benchmark. Head `72e02cd92c`.

### 2026-08-12 — PR #762 S1/S2/S3 resolution (commit a5448fa36)
S1: `production_scratch_alloc(numel, dtype)` helper + 2 new canary tests (`scratch_buffer_wider_write_absorbed_by_padding`, `scratch_buffer_detects_oversized_write`).
S2: `TensorMut::validate_write_dtype()` — exact match for present, byte-size gate for absent. `mark_absent()` invariant documented.
S3: `NodeOutputSink::Absent` variant — `build_subgraph_routing` no longer allocates phantom slots.
Nits: removed 4 no-op identity transmutes.
280 passed / 0 failed. Clippy clean. fmt clean. Miri: 4/4 canary tests clean.

### 2026-08-12 — PR #832 H200 CUDA validation build fix (MERGED `2b62c620`)
Added the missing `bf16_scratch` field (`Mutex<Bf16Scratch>`, `Mutex::new(Bf16Scratch::new(runtime.clone()))`) to 11 `MatMulNBitsKernel` test initializers in `crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs`. Verified `cargo test --no-run -p onnx-runtime-ep-cuda --features cuda` green. Merged as part of the H200 (Muse-Glimmer-30B) CUDA EP validation wave.

Full pre-compaction history in `history-archive.md`.

### 2026-08-12 — CUDA-graph capture escalation (background, agent sebastian-3)
Redirected post-#840 to investigate why CUDA-graph capture does not engage for
Muse-Glimmer native decode. Delivered a cross-domain escalation: **3 stacked
blockers** — (1) LOAD (engine native pipeline can't load the model), (2) CLASSIFY
(vestigial SWA mis-classification on decode path), (3) CAPTURE (infra proven, gated
behind 1+2). No perf PR (model can't load on engine native path yet). Coordinator
dispatched Batty (LOAD) + Deckard (CLASSIFY); I pair on CAPTURE + re-measure once
unblocked. Shared team goal: **beat ORT 40 tok/s via CUDA-graph capture**. Prior
#840 (629fbf90) merged: real cudaMemGetInfo device-capacity + CudaFoldConstantCast,
native decode 10.2→11.4 tok/s (+11.8%).

## 2026-08-12/13 — CUDA capture arc COMPLETE (shared: 11.4 → 23.13 tok/s)
Owned diagnosis + escalation + the two CUDA-EP kernel blockers. **#855** (`1022b912`)
`gqa_decode_bf16` capture-safe kernel (fp32 accumulation; Chew-gated, max_abs 1.953e-3):
segments 54 → 2, 22.52 tok/s. **#854** (`f85a82f0`) skip-norm capture-safety (persistent
`NormBf16Scratch`, demote on `grew` only when `is_capturing()`): segments 2 → 1, 0 seams,
23.13 tok/s (+33% vs capture OFF). Built on #848 (Deckard) / #850 (Batty) / #852 (Leon).
**Corrected my own diagnosis:** with the step captured, decode is now **kernel-bound**
(Cast 40%, MatMulNBits 21%, GQA 14%), not pure dispatch-bound. Next lever = Cast
round-trip elimination to reach ORT's ~40 tok/s.

## 2026-08-12/13 — PR #860 MERGED: 23→40 tok/s, CUDA goal MET (11.4 → 40.21 tok/s)
Closed the final lever. Generalized ep-cuda `CudaDropNormalizationCasts` to fold **bf16**
casts around `RMSNormalization` (op-swap `RMSNormalization`→`SimplifiedLayerNormalization`
for re-inference stability — both map to the same `RmsNormFactory→RmsNormKernel`), and
rewrote `rmsnorm_bf16` with a **parallel f32 tree reduction** (fp32 accumulation; only
summation order changes). Native CUDA decode **23.16 → 40.21 tok/s** (H200, 3-run median),
1 segment / 0 seams, first-16 greedy ids match reference. Cast removal is ~free under
capture; the parallel reduction is the real lever (serial `fadd` chain ≈40% of captured
decode; serial floor ≈33 tok/s). Chew-gated 🟢. Escape hatch
`ONNX_GENAI_CUDA_DISABLE_NORM_CAST_FOLD=1`. **Multi-session CUDA goal MET: matches ORT ~40 tok/s.**

## 2026-08-13 — PR #867 MERGED: 40→47 tok/s, native CUDA now BEATS ORT
Cached the Float16-staged constant int4 scales. `MatMulNBitsKernel::run_bf16` was re-casting
the immutable int4 scale slots bf16→f16 into an ephemeral arena **every** decode step
(~3.3 GB/token, ~25% of int4 weight traffic, 417 redundant cast launches). Added a persistent
per-kernel `Bf16ConstCache` that stages the constant scale slots once (general path input 2;
SwiGLU-fusion inputs 2 and 4) and reuses them; dynamic slots (activation input 0, bias
residual) stay on the ephemeral arena. Capture-safe (alloc+cast on pre-capture warmup; replays
hit lookups only). Native decode **40.21 → 47.25 tok/s** (H200, 3-run median), MatMulNBits eager
share ~44%→~31%, 1 segment / 0 seams, full 128-token sequence byte-identical. **Byte-exact by
construction — no Chew gate**; test `bf16_scale_cache_is_bit_exact_to_inline_staging`.
**MILESTONE: native CUDA EP now clearly beats ORT — 47.25 vs ~40 tok/s, +18%.** Next lever
(deferred): GQA is now the largest eager share (~41%); full bf16-native GEMV / accuracy_level=4
DP4A int8 remain open, both numerics-gated.

## 2026-08-13 — bf16 SwiGLU kernels (#871) + GQA null finding (#870); 47.25 is the CEILING
Shipped **#871** (bf16 decomposed SiLU/SiLU-Mul kernels `decomposed_silu_mul_bf16`/
`decomposed_silu_bf16` in `elementwise.rs`) — fixes a real portability **hard crash**
(bf16 decomposed SiLU previously errored `"requires float16"`); byte-exact **0 ulp** vs f64
oracle, **Chew 🟢**, 5/5 silu tests on H200. Its graph SwiGLU-Mul fold is FLAT (−104 cheapest
nodes). **#870** (doc-only): decode is 2568 nodes/token × ~8.17 µs/node; cheapening any single
kernel inner loop (GQA seq-loop, GEMV depth-loops) is flat → GQA not a viable lever. Full gate_up
bf16 fold went non-deterministic (f16-staging); safe version needs a bf16-native fused
`gate_up_swiglu` kernel + Chew — deferred. **CONCLUSION (with Batty #872/#873): native int4 decode
of Muse-Glimmer-30B is weight-bandwidth/compute-floor bound at ~47.25 tok/s (H200), NOT
dispatch-bound. 47.25 is the architectural ceiling; beat it via fewer weight bytes/token or a
megakernel, NOT node fusion.**

## 2026-08-13 — Lower-bit quant NO-GO + mechanism correction (#885, docs-only)
Researched whether lower-bit quant could beat ~47 tok/s. Byte budget: decoder reads **15.325
GB weights/token** = 724 GB/s at 47 tok/s = only **~15% of H200 HBM roofline**. Ran a controlled
weight-DRAM byte-fold probe (`ONNX_GENAI_WEIGHT_FOLD=D`, throwaway/reverted, node-count
byte-identical): full **47.29** → half **47.98** (+1.5%) → quarter **48.62** (+2.8%). Weight-DRAM-
bound fraction ≈ **3–4%**; int2-everywhere (−45% bytes) projects ≈+1.6%. **VERDICT: lower-bit quant
(int3/int2/mixed/2:4/NF4) is a MEASURED 🟥 NO-GO.** **KEY CORRECTION:** this REFUTES the earlier
"weight-bandwidth/compute-floor bound" attribution (#870/#872/#873) — decode is **LATENCY-bound
on the ~2568-node serial chain (~8.2 µs/node)**, not bandwidth-bound. Ceiling VALUE (~47) and
"marginal fusion isn't a lever" still stand; only the WHY changes. **Megakernel / drastic per-layer
node-collapse REOPENED as the true next lever.** Brief: `docs/research/lowbit-quant-feasibility.md`.

## 2026-08-13 — Dense-decode megakernel: multi-CTA GEMV megakernel MEASURED 🟥 NO-GO (#898)
Built the persistent multi-CTA cooperative one-layer megakernel (MLP triple-GEMV, 1056 co-resident
CTAs = 8/SM × 132 SMs, grid.sync seams, L2-resident scratch, production int4 GEMV math) and measured
vs identical-math per-op baseline (H200, median 200 iters × 3). **Per-op 0.656 → megakernel
0.676–0.680 ms/layer-MLP = −3.2% recovered (~3% SLOWER), byte-exact 0-ulp; grid.sync = 2.23
µs/barrier.** Projected whole-model gain ≈ 0% (stays ~47 tok/s). **This CLOSES the megakernel lever
the #885 correction reopened** — mechanism: (1) CUDA-graph replay already banks the per-launch
overhead, (2) multi-CTA must PAY a grid.sync tax that cancels savings, (3) GEMVs are genuine
full-device weight reads already fanned across 132 SMs. **Redirect: the only remaining
recoverable-overhead lever is graph-side glue node-collapse (Batty, `optimizer.rs`);** my role is
the already-landed fused epilogues (#867, #854) that enable node deletion. Merged PR #898
(main @ 0790849c). Staged arc (superseded): Phase A/B feasibility GO → P1.5 (single-CTA 926× slower,
grid.sync capturable) → P2 multi-CTA NO-GO. Doc: `docs/research/dense-decode-megakernel-feasibility.md` §7.

## 2026-08-13 — bf16 skip-RMSNorm KERNEL byte-exact SHIP; standalone FOLD −1.5% NO-SHIP (#903)
Closed #900's blocker: built the missing byte-exact **bf16 skip-RMSNorm kernel** for Gemma3
sandwich-norm (`skip_rmsnorm_bf16` NVRTC, `crates/onnx-runtime-ep-cuda/src/kernels/normalization.rs`).
`sum = __float2bfloat16_rn(f32(residual)+f32(x))` (bit-for-bit a standalone bf16 `Add`) then the
identical `rmsnorm_bf16` block-tree reduction. **BYTE-EXACT 0-ulp** (GPU unit tests bit-identical;
real-model fold OFF vs ON bit-identical, 128/128 tokens). **KERNEL: SHIP.** But the standalone
`CudaSkipRmsNormFusion` fold **REGRESSES −1.5%** under graph replay (fold OFF 47.77 → ON 47.06
tok/s, 104 seams): at M=1 the single-CTA RMS reduction serializes the residual `Add` that the
standalone spread across 132 SMs; replay already banks the launch saving. **FOLD: NO-SHIP as
default** — retained opt-in behind `ONNX_GENAI_CUDA_ENABLE_SKIP_RMSNORM_FUSION` (default OFF).
This is the THIRD independent confirmation of the batch-1 decode LATENCY FLOOR (with #898 megakernel
NO-GO and #899/#900 glue-collapse +0.9% ceiling). **Next lever (NOT funded):** fold the bf16 norm
into the neighbouring multi-CTA int4 GEMV prologue/epilogue (bf16 analogue of
`CudaSkipRmsNormMatMulFusion`) — keeps the reduction distributed. Do NOT self-merge (Chew gates
numerics). Doc §8.6.

- **2026-08-14 (#916, MERGED):** bf16 norm-into-GEMV-prologue fusion measured NO-GO — −4.6% regression AND numeric divergence (≈token 38) under CUDA-graph replay; fp16 prologue reduction is single-warp-serial on the critical GEMV path. Finding-only (docs §8.7), nothing landed. **Fourth** independent confirmation of the batch-1 decode latency floor; norm→GEMV-prologue kill-gate CLOSED.
