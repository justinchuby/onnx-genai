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
