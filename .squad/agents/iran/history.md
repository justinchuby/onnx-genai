# Iran — History

## Project Context
- Mac CPU Optimization Engineer for Apple Silicon CPU-EP perf: Apple Silicon NEON, Accelerate/AMX/BNNS, aarch64-apple-darwin GEMV/GEMM hot paths.
- Joined 2026-07-26. Full pre-summary detail archived at `.squad/agents/iran/history-archive-2026-07-27T02-00-00Z.md`, `.squad/agents/iran/history-archive-2026-07-27T02-00-00Z-rebase.md`, and earlier archive(s).
- Entries through 2026-08-11 (upstream CI correction wave) archived to `.squad/agents/iran/history-archive-2026-08-12T10-15-00Z.md`.

## Summary through 2026-07-27T02:00:00Z
- PR #227 roofline campaign established access-pattern-specific rooflines, removed dead Accelerate SGEMV paths, fixed decode dispatch, and made direct SIMD reachability/parity tests mandatory for performance claims.
- CPU decode persistent pool default became deterministic (`On`), with adaptive load probing opt-in; unconditional library stderr was replaced by queryable/tracing diagnostics.
- Mac f16 prefill campaign added BNNS/AMX M>=2 dispatch, filter caching, non-contiguous/column-major handling, and guarded M=1 NEON GEMV dispatch; TTFT improved from ~989ms to ~167ms while decode stayed faster than ORT.
- First-decode spike root cause was shape-keyed cold caches and lm_head column-major densification; global transpose cache plus column-major GEMV/BNNS paths removed the spike.
- SiblingProjectionMerge reduced op count but regressed TTFT on BNNS, so it stayed opt-in; wider GEMMs are not automatically faster on Apple Silicon.
- Prefill overhead attribution showed low-load TTFT ~78–80ms, not 160ms; prior overhead was load contention. Non-GEMM cost is concentrated in SDPA, SiLU, and Mul, dominated by fp16↔f32 widen/narrow. Recommended levers: Accelerate SGEMM for SDPA, native fp16 elementwise ops, and fused SiLU·Mul.

## Summary through 2026-08-11 (upstream CI correction wave)
- PR #317: Three-tier conv dispatch (BNNS/AMX, im2col+cblas, scalar). ResNet-18: 8792ms → 93ms (94×).
- PR #31973: AVX2 LayerNorm kernel — Welford → centered two-pass + double-precision sum. 42/42 MLAS tests pass. HEAD `6ef1f61f88`.
- PR #762: B4 CUDA fail-closed (0 factories, CanCopy=false). ReleaseEpFactory ABI fixed (`void` → `*mut OrtStatus`). Clippy fix (collapsed duplicate `if/else if` in loader.rs).
- PR #31974: B5 stat precision (WriteStat<U=float>), NarrowToFloat dedup (narrow_float_utils.h).

## Durable lessons
- Always state benchmark metric and system load; quiet and loaded runs can choose different winners.
- Streaming bandwidth is not GEMV bandwidth; use access-pattern-specific ceilings.
- Production-path probes are required because unit-test dispatch often differs from real `[1,M,K]` and strided-weight shapes.
- Every SIMD/fast dispatch path needs reachability and parity guards before claims ship.
- Future non-contiguous weights should check algebraic layout identities such as column-major-as-transpose before copying.
- Do not chase dispatch count alone; measure where time is actually concentrated under controlled load.
- Centered two-pass ≠ uncentered two-pass. Always specify the formulation when discussing variance algorithms.
- CUDA is implementation-blocked, not hardware-blocked; document the roadmap so future hardware-gated work can resume.
- vcvt_f32_f16/vcvt_f16_f32 are baseline ARMv8-A (FCVTL/FCVTN); FEAT_FP16 governs fp16 arithmetic, not conversions.

## 2026-08-12 — PR #762 clippy fix

- Collapsed identical `if`/`else if` blocks in `loader.rs:263` into single `||` condition.
- Preserved short-circuit order: struct_size check before field access.
- Added comment documenting the ordering invariant.
- Tests: 280 passed, 0 failed. Clippy clean. Fmt clean.
- Pushed `08a9105f8`.

## 2026-08-12 — PR #31993 NaN assertion fix + runtime evidence

**B1 (NaN payload not portable):** Relaxed NaN assertions from "payload modulo quiet bit" to "is NaN + sign matches". Proven on AArch64 under QEMU: sNaN 0x7FA00000 → NEON 0x7F00, scalar 0x7E00; qNaN 0x7FC12300 → NEON 0x7E09, scalar 0x7E00. Old assertion would have failed on both.

**B2 (runtime evidence):** Cross-compiled standalone NEON kernel harness with aarch64-linux-gnu-g++ 13.2, ran under qemu-aarch64-static. All tests passed: ±0, ±Inf, NaN (qNaN, sNaN, payload variants), denormals, RNE tie (1+2^-11), bulk at 17 lengths (1–1024). This is QEMU emulation, not native Apple Silicon.

**FEAT_FP16 correction:** Commit narrative updated. vcvt_f32_f16/vcvt_f16_f32 are baseline ARMv8-A (FCVTL/FCVTN); FEAT_FP16 is fp16 arithmetic, not conversions.

**New HEAD:** `5ba2500`. PR stays draft.


## Summary through 2026-07-27T02:00:00Z
- PR #227 roofline campaign established access-pattern-specific rooflines, removed dead Accelerate SGEMV paths, fixed decode dispatch, and made direct SIMD reachability/parity tests mandatory for performance claims.
- CPU decode persistent pool default became deterministic (`On`), with adaptive load probing opt-in; unconditional library stderr was replaced by queryable/tracing diagnostics.
- Mac f16 prefill campaign added BNNS/AMX M>=2 dispatch, filter caching, non-contiguous/column-major handling, and guarded M=1 NEON GEMV dispatch; TTFT improved from ~989ms to ~167ms while decode stayed faster than ORT.
- First-decode spike root cause was shape-keyed cold caches and lm_head column-major densification; global transpose cache plus column-major GEMV/BNNS paths removed the spike.

## 2026-09-01 — PR #2082 feature-unification authority seal

- Removed the additive `runtime-session-authority` Cargo feature from ep-api and every consumer.
- Replaced feature-gated issuance with an opaque zero-sized authority token whose field and constructor are private; the session executor owns the sole production token and never exposes it.
- Executor identity allocation, template binding, and finalization-proof issuance now all require borrowing that token. CUDA integration internals use an explicit test-binary-only raw token; public session integration covers the valid Required lifecycle.
- Added a hostile external Cargo fixture covering five attack shapes across default, session, CUDA, gpu-tests, and workspace-unified graphs (25 intended compile failures), plus manifest/source inventory checks.
- Validation: ep-api/session host suites green; strict affected Clippy and build feature matrix green; idle-A100 route suites 4+5+9+6 and Disabled zero-work 1 green; CUDA honesty 706 tests across 92 targets in both inventories with four suite locks.
