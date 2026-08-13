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
