# Iran — History

## Project Context
- Mac CPU Optimization Engineer for Apple Silicon CPU-EP perf: Apple Silicon NEON, Accelerate/AMX/BNNS, aarch64-apple-darwin GEMV/GEMM hot paths.
- Joined 2026-07-26. Full pre-summary detail archived at `.squad/agents/iran/history-archive-2026-07-27T02-00-00Z.md`, `.squad/agents/iran/history-archive-2026-07-27T02-00-00Z-rebase.md`, and earlier archive(s).

## Summary through 2026-07-27T02:00:00Z
- PR #227 roofline campaign established access-pattern-specific rooflines, removed dead Accelerate SGEMV paths, fixed decode dispatch, and made direct SIMD reachability/parity tests mandatory for performance claims.
- CPU decode persistent pool default became deterministic (`On`), with adaptive load probing opt-in; unconditional library stderr was replaced by queryable/tracing diagnostics.
- Mac f16 prefill campaign added BNNS/AMX M>=2 dispatch, filter caching, non-contiguous/column-major handling, and guarded M=1 NEON GEMV dispatch; TTFT improved from ~989ms to ~167ms while decode stayed faster than ORT.
- First-decode spike root cause was shape-keyed cold caches and lm_head column-major densification; global transpose cache plus column-major GEMV/BNNS paths removed the spike.
- SiblingProjectionMerge reduced op count but regressed TTFT on BNNS, so it stayed opt-in; wider GEMMs are not automatically faster on Apple Silicon.
- Prefill overhead attribution showed low-load TTFT ~78–80ms, not 160ms; prior overhead was load contention. Non-GEMM cost is concentrated in SDPA, SiLU, and Mul, dominated by fp16↔f32 widen/narrow. Recommended levers: Accelerate SGEMM for SDPA, native fp16 elementwise ops, and fused SiLU·Mul.

## Durable lessons
- Always state benchmark metric and system load; quiet and loaded runs can choose different winners.
- Streaming bandwidth is not GEMV bandwidth; use access-pattern-specific ceilings.
- Production-path probes are required because unit-test dispatch often differs from real `[1,M,K]` and strided-weight shapes.
- Every SIMD/fast dispatch path needs reachability and parity guards before claims ship.
- Future non-contiguous weights should check algebraic layout identities such as column-major-as-transpose before copying.
- Do not chase dispatch count alone; measure where time is actually concentrated under controlled load.

## 2026-07-27: Conv Three-Tier Dispatch (#317)
- Diagnosed 643× ResNet-18 gap: `conv_ref.rs` scalar loop was only path on macOS due to `mlas` feature gate being x86-64-Linux-only.
- Assessed BNNSGraph: requires `.mlmodelc`, cannot do per-op dispatch. No migration target exists.
- Implemented Tier 1 (BNNS Filter Conv, AMX, 877–1458 GFLOPS), Tier 2 (im2col + cblas_sgemm, ~300 GFLOPS), Tier 3 (scalar ref).
- Result: ResNet-18 8792ms → 93ms (94× faster), now 0.15× ORT (from 0.0016×). Whisper-tiny unchanged (MatMul-bound). Decode unregressed.
- Remaining ResNet-18 gap (6.7×) is non-Conv ops (BatchNorm, Pool, Add on scalar paths).
- BNNS Filter API deprecated but no replacement for per-op use. `cblas_sgemm` is durable fallback.
- 2026-07-28: Small-shape GEMV investigation produced a valid negative result: existing inline paths and cblas already cover the remaining cases. SDPA decode PR #349 merged after attribution and after correcting the headline from 1.9x to 1.37x by naming the model (TinyStories-1M vs -33M). Always state which model each ratio refers to.

## 2026-08-11: AVX2 LayerNorm kernel revision (PR #31973 blockers B1/B2/N2/N3/N4)

- **B1 reproduced:** AVX2 lane-parallel Welford hits 28.2% rel err at base=1e5/spread=1e-2/N=512 vs fp64 oracle (scalar Welford: 0.47%). Root cause: per-lane fp32 mean accumulation rounds before merge.
- **B2 evaluated:** Centered two-pass with double-precision first-pass sum is 1.8× faster than AVX2 Welford (427ns vs 751ns, N=1024) AND more accurate (worst 5.95e-3 vs 2.82e-1). fp32 sum is catastrophic at base=1e7 (100% error); double sum is essential.
- **Algorithm replaced:** Welford → centered two-pass with `_mm256_cvtps_pd`+`_mm256_add_pd` for mean, fp32 for centered variance.
- **N2 fixed:** NormSize<8 gate moved to `#if x86` so RVV is unaffected.
- **N3 verified:** RMSNorm mean-skip for null MeanOut was already correct; upgraded RMSNorm MeanOut path to double-sum too.
- **N4 fixed:** Added explicit `set_source_files_properties(/arch:AVX2)` for layernorm_kernel_avx2.cpp on MSVC.
- **Tests:** 39/41 pass. 2 Pris-owned precision tests fail because they check parity vs scalar Welford (wrong reference now). All 32 functional tests pass.
- Lesson: centered two-pass ≠ uncentered two-pass. The prior team rejection of "two-pass" conflated E[x²]-mean² (catastrophic) with sum((x-mean)²) (numerically standard). Always specify the formulation.

### 2026-08-11 — B4: CUDA Plugin Fail-Closed (PR #762 Reviewer Rejection)
- **Owned B4:** CUDA cdylib advertised GPU EP it could not honour.
- **Root cause:** Implementation-blocked, NOT hardware-blocked. Four defects: (1) separate CUDA runtime/context per EP/allocator/stream, (2) non-functional data transfer (no OrtApi, no shared stream), (3) NULL stream handle, (4) `device_free` passes `size=0`.
- **Fix:** `CreateEpFactories` returns 0 factories + actionable status in BOTH feature configs. Crate docs specify all 4 defects as a roadmap for future implementation.
- **CanCopy fail-closed:** Both `transfer_can_copy` and `transfer_full_can_copy` now return `false` for device EPs (were returning `true` unconditionally — fail-open).
- **device_free defect #4:** Documented the `size=0` contract violation with fix specification (allocation size tracking side-table).
- **Validation:** `cargo check --workspace` ✓, `cargo check -p onnx-runtime-ep-cuda-plugin --features cuda` ✓, ep-plugin 154+9 tests ✓, clippy clean ✓, fmt clean ✓.
- **Key correction:** CUDA is implementation-blocked, not hardware-blocked. Even with a GPU, this code cannot work as written.

### 2026-08-11 — ReleaseEpFactory ABI fix (follow-up to B4, reported by Sapper)
- **Bug:** CUDA shim's hand-written `ReleaseEpFactory` returned `void`; correct ABI is `OrtStatus*` per `onnxruntime_ep_c_api.h:2669`.
- **Fix:** Updated return type to `*mut OrtStatus`. On normal release returns null (success); on panic catches unwind and returns actionable status via `panic_to_fail_status`.
- **Macro not used:** `export_ep_factories!` emits both symbols as one expansion; CUDA shim needs custom `CreateEpFactories`, so macro would conflict. Commented in file explaining why it is hand-written and must stay in sync with the macro's `ReleaseEpFactory` arm.
- **CreateEpFactories drift check:** Signature matches `CreateEpApiFactoriesFn` at header line 2654 — no drift.
- **Fail-closed unchanged:** Both configs still return 0 factories + error status.
- **Validation:** `cargo check --workspace` ✓, `cargo check -p onnx-runtime-ep-cuda-plugin --features cuda` ✓, ep-plugin 9 tests ✓, clippy clean ✓, fmt clean ✓.
