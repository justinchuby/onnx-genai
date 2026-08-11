# Resch — History (compacted 2026-07-29)

**Role:** Intel CPU Optimization Engineer for x86-64 CPU-EP performance, AVX2/AVX-512/VNNI, MLAS interplay, and int8 DP4A GEMV. Optimizations must be portable beyond one machine, benchmark-backed, and numerically matched to scalar/f64 references with regression tests.

## Durable lessons
- SIMD/NPU paths must be opt-in early returns over an always-present scalar baseline; avoid architecture forks and keep every target compilable.
- Runtime feature detection/dispatch is expected for SIMD paths; tune for consumer/edge hardware, not only flagship machines.
- Untested SIMD paths are as risky as placeholders: every AVX/NEON/SVE/QNN path needs guard-break tests and paired scalar/reference checks.
- Apple Silicon FP16 NEON can widen f16 loads directly while ORT CPU widens before GEMM; do not assume the same architecture distinction on other platforms.
- Cross-platform compile failures can hide in cfg-gated code that only builds on one OS/arch; `x86_64-apple-darwin` alone is insufficient because it changes arch but not enough OS surface.
- Kernel layout should group by role in dispatch, not pure platform, to avoid recreating the banned architecture fork.
- Platform-naming lint catches files with single-arch cfg and no platform marker, but does not catch within-file missing implementations.
- Cross-compile check catches cfg-gating errors in `--all-targets`; known gaps: ep-cpu from macOS, runtime dispatch, and Windows cfgs outside portable matrix.
- Dispatch manifest is CI-only and cross-EP ready; every optimization counter should have a manifest row unless deliberately excluded.
- Inverse manifest check closes the "human must remember to add a row" gap; AtomicUsize/AtomicU64 and `pub static` counters must be recognized.
- BatchNorm fusion elimination does not fit dispatch-tier claim schema; it is an optimizer/opset-registration failure mode needing optimizer-level counters.

## Recent work (current wave, ~2026-07-28/29)

### 2026-07-27T10:40:00-07:00 — Platform-naming lint + x86 GEMM renames (PR #278)
Renamed `simd_gemm.rs` → `x86_sgemm.rs` and `bf16_gemm.rs` → `x86_bf16.rs` with zero behavior change. Added `scripts/check_platform_naming.py`, a CI lint for single-arch cfg files lacking platform markers; it uses a portable-item check to avoid false positives, but cannot catch within-file missing implementations like the sdpa `dot_f32` case. Guard-break restored old names and failed; 945 tests passed, clippy green on aarch64/x86_64.

### 2026-07-27 — Cross-Target Compilation Check (PR #319)
Added `scripts/check_cross_compile.sh` for Roy's structural fix plan. It targets `x86_64-unknown-linux-gnu`, uses `--all-targets`, runs full offline crates on Ubuntu and an FFI-free subset on macOS, and teaches why `x86_64-apple-darwin` is insufficient. Guard-break synthetic `is_undilated` in onnx-runtime-ir failed under the new script. Known gaps: ep-cpu from macOS, runtime dispatch, Windows cfg via portable matrix.

### 2026-07-27T22:05:00-07:00 — Dispatch manifest lint
Created `dispatch_manifest.toml` plus `scripts/check_dispatch_manifest.py`: declarative (op, variant, platform) → minimum tier + proving counter, CI-only and cross-EP ready. Seeded MatMul/Conv/SDPA/KernelDispatch claims, documented depthwise Conv and bf16 M≥2 exclusions, and proved guard-break by renaming `CONV_BNNS_TEST_HITS`. It would have caught 7/9 historical instances; misses compilation errors and unclaimed ops.

### 2026-07-27T23:13:00-07:00 — Manifest backfill + inverse check
Backfilled PR #324 claims (MaxPool→BNNS, Add→vDSP, MatMul f16 colmaj→NEON GEMV), added dilated MaxPool and BatchNorm fusion exclusions, and implemented inverse checking so any optimization counter without a manifest row fails CI. Fixed the AtomicU64 blind spot in both reachability and manifest lints. BatchNorm was documented as graph-level fusion elimination, not a dispatch-tier claim. Guard-breaks failed as intended; 973 CPU EP tests, 4 lints, and fmt were green.

Full pre-compaction history in `history-archive.md`.

### 2026-08-11T03:27:00+00:00 — Upstream CPU pilot: AVX2 LayerNorm kernel

**Phase 1:** Traced x86 fp16 MatMul path end-to-end. On x86, `MlasFp16AccelerationSupported()=false`, `MlasHalfGemmNativePackBSize()=0`, so MatMul<MLFloat16> falls to Eigen fp32-accumulate GEMM (math_cpu.cc:220-226). Since AVX2 has only F16C conversion (no fp16 arithmetic), an AVX2 "half GEMM" would replicate Eigen's path — **not viable**.

**Correction:** GatherBlockQuantized CPU kernel exists at `contrib_ops/cpu/quantization/gather_block_quantized.cc`. Prior gap analysis missed `contrib_ops/cpu/`. Candidate #2 from ranked plan is dead.

**Phase 2:** Chose AVX2 LayerNorm/RMSNorm as alternative. `MlasLayerNormF32` dispatch existed (layernorm.cpp) with only a RVV kernel — no x86. Float LayerNorm fell back to scalar Welford; MLFloat16 path is element-by-element scalar.

**Phase 3:** Implemented `layernorm_kernel_avx2.cpp` — two-pass vectorized kernel (8-wide reduce + normalize) with FMA3 fused multiply-add for bias case. Wired dispatch in `platform.cpp` AVX2 block, declared in `mlasi.h`, added to both Windows and Linux CMake lists. 5 files changed.

**Phase 4:** Full `onnxruntime_mlas` library compiled successfully (zero warnings). No runtime benchmark — stated honestly. Entry points documented for Pris.

## 2026-08-11: AVX2 LayerNorm Hardening (PR #31973)

**Task 1 — Small-N threshold:** Measured crossover on AMD EPYC 9V74. Added `NormSize < 8` guard in `layernorm.cpp` dispatch. Below 8, zero SIMD iterations execute — pure scalar tail with overhead. RMSNorm regresses 3-22% for N≤7; threshold returns `false` so caller uses scalar `ComputeJob`. All 30 tests at N≥8 pass; 9 tests at N∈{1,7} need assertion updates (Chew).

**Task 2 — Welford SIMD (option a):** Replaced two-pass variance with Welford's online algorithm using 8 parallel AVX2 accumulators + pairwise merge. 5-7× faster than scalar Welford at typical sizes (128-4096). Confirmed two-pass suffers catastrophic cancellation on adversarial inputs (mean~1e6: output error 4.18 vs Welford SIMD 0.049). RMSNorm unchanged (sum-of-squares has no cancellation risk).

**Needs from Chew:** (1) Update test assertions for N<8, (2) fix `worst_welford` unused-variable build error, (3) adversarial precision tests at various mean offsets.

## 2026-08-11 — BFloat16 CPU LayerNorm/RMSNorm Registration

**Task:** Register BFloat16 on CPU for LayerNormalization, SimplifiedLayerNormalization, SkipLayerNormalization, SkipSimplifiedLayerNormalization — closing the CPU/CUDA asymmetry.

**Approach:** Shared fp16 path via `is_narrow_float_v<T>` trait. All arithmetic is f32; BFloat16 is storage only. No MLAS kernel, no AVX512-BF16. Round-to-nearest-even via upstream `BFloat16(float)` constructor.

**Files:** 6 files modified in `/workspace/upstream/ort-bf16` (branch `nxrt/mlas-bf16-layernorm`). No MLAS, no CMake changes. Zero overlap with PR #31973.

**Status:** Code complete, not build-verified. Welford semantics preserved.
