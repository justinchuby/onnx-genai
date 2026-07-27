# Iran — History

## Project Context (joined day)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Role:** Mac CPU Optimization Engineer — Apple Silicon CPU-EP perf (NEON, Accelerate/AMX), aarch64-apple-darwin GEMV/GEMM hot paths.
- **Requested by:** Justin Chu
- **Joined:** 2026-07-26

## History summary through 2026-07-27T04:35:00-07:00

Full pre-summary detail archived at `.squad/agents/iran/history-archive-2026-07-27T04-35-00-07-00.md`.

### 2026-07-26 — Joined CPU & Edge pod
Standing directive: portable optimizations, benchmark-backed claims, and SIMD/NPU correctness against scalar/f64 reference within justified tolerance.

### 2026-07-27 — PR #227 Mac CPU EP roofline campaign
- Implemented Apple Silicon CPU EP acceleration on `squad/mac-cpu-ep-roofline`: pre-transposed column-parallel NEON GEMV, SPMD worker dispatch, NEON SDPA, vectorized SiLU, FMB/output-copy reductions, and direct-from-mmap FP16 GEMV.
- Established that Accelerate/AMX is useful for prefill/SGEMM but collapses or loses to NEON for DRAM-bound decode GEMV because of access pattern and dispatch overhead; removed dead Accelerate SGEMV paths.
- Fixed the `batch_shape` dispatch bug that sent `[1,1,K]` decode through the non-transposed GEMV path.
- Demonstrated FP32 native reached near-ORT but was constrained by GEMV bandwidth plus graph/dispatch overhead; FP16 became the decisive architectural lever because Apple Silicon NEON can read f16 weights directly and widen in-register.
- Added NEON bulk f16↔f32 conversion because scalar conversion erased the FP16 bandwidth win in non-GEMV ops.
- Resolved Fact Checker's FP16 discrepancy: agent load changed the auto-calibrator path and produced asymmetric benchmark corruption; quiet-machine runs verified native FP16 over ORT with low spread.
- Froze the CPU decode auto-calibrator decision after initial commitment to avoid mid-run path switching and token nondeterminism; documented that forced pool can be worse under load even when it wins on a quiet machine.

### Durable notes for future Mac CPU work
- Keep Apple Silicon portability explicit: use runtime feature detection and avoid tuning only for M1 Max.
- State benchmark metrics exactly; compare `tokens/total_time`, p50-derived tok/s, and init-inclusive means separately.
- Use access-pattern-specific rooflines: streaming bandwidth is not GEMV bandwidth.
- Every SIMD path needs direct tests plus guard-break coverage before performance claims ship.

### 2026-07-27 — Load-adaptive path selection made opt-in (Chu directive)
- Changed `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL` default from `Auto` (calibrator) to `On` (deterministic pool).
- Added `=auto` value for explicit opt-in to load-adaptive calibration.
- Renamed enum variants: `Auto`→`Adaptive`, `Forced`→`On`; `Off` unchanged.
- Updated module docs, `report_pool_built` observability message, tests (3 new: `default_selects_pool_without_probing`, `adaptive_flag_enables_calibration`, `on_and_adaptive_build_the_pool`), README.
- Guard-break proof: broke `persistence_mode_from_raw` default→Off, test caught it, restored.
- Quiet: 43.75 tok/s (pool default). Under 8×load: 3.09 tok/s (accepted tradeoff for predictability; old adaptive would have chosen flat ~13 tok/s).
- x86_64-apple-darwin cross-compilation confirmed clean with `cargo clippy -D warnings`.
- Per-generation freeze from `177e8a73` preserved (orthogonal, not touched).

### 2026-07-27 — Coordinator review fixes (eprintln removal + GEMV dispatch)
- **Commit `69f00b83`**: Replaced unconditional `eprintln!` in `report_pool_built()` and `report_spmd_fallback()` with queryable `decode_path_label()` API (`DECODE_PATH_LABEL` OnceLock) + `NXRT_CALIB_DEBUG` gated diagnostics. A library must not print to streams the caller owns.
- **Commit `ed7a65e3`**: Fixed M=1 decode dispatch regression — moved NEON GEMV check *before* `try_matmul_half` in `MatMulKernel::execute_with_backend`. Sebastian's `half_gemm.rs` (50184994) intercepted f16×f16 at all M, causing 14.5→53.4 tok/s recovery (4× regression). Deckard's `fp16_m1_decode_reaches_neon_gemv_not_half_gemm` test now passes.
- **Measurement matrix** (load avg 1-min stated, machine shared with Copilot ~251% CPU baseline):
  - Quiet (load ~4-5): pool 53.35 [46.03,57.61], adaptive 56.10 [50.43,58.57], flat 42.84 [42.01,43.16], ORT 42.19 [41.79,42.61]
  - Under 4×`yes` load (~10): pool 18.96 [18.18,20.48], adaptive 31.95 [31.77,33.30], flat 31.57 [31.16,31.75], ORT 37.76
- Verified: clean stderr (no unconditional prints), all 33 decode_spmd tests pass, Deckard's dispatch test passes, x86_64 clippy clean, `check_profile_table.py` passes.

### 2026-07-27 — Tracing + half_gemm overlap analysis (post main merge)
- **Commit `281481a6`**: Switched from `NXRT_CALIB_DEBUG` gated `eprintln!` to `tracing::debug!` (per `docs/ERROR_AND_LOGGING_CONVENTIONS.md`). Added `tracing = "0.1"` as optional dep behind existing `tracing` feature. Without feature, `NXRT_CALIB_DEBUG` fallback preserved.
- **half_gemm.rs overlap**: Complementary, not duplicated. GEMV (M=1 bandwidth-optimal, inline asm fcvtl ARMv8 base) vs GEMM (M>1 compute-optimal, vcvt_f32_f16 intrinsic requiring FEAT_FP16). Dispatch collision fixed in `ed7a65e3`. Consolidation deferred to separate PR.

### 2026-07-27 — BNNS prefill campaign (PR #275)
- **Commits `f0cbd786`, `aa219b4b`**: BNNS fp16→f32 GEMM at M≥2 on macOS, FilterCache, contiguous_b_f16 cache. TTFT 989→348 ms.
- **Commit `9f1e7684`**: Column-major zero-copy for both BNNS (trans_b) and GEMV. Eliminated ~1s first-decode spike (lm_head 544MB f32 densification). TTFT→167ms, end-to-end 1.50× ORT.
- **Commit `3ab6999a`**: cfg-gated `Arc` import to fix CI on non-macOS targets.
- **Commit `17be7087`**: Correctness fixes from rubber-duck review:
  - Blocking #1: Added `constant_inputs[1]` guard to rescue block — non-constant non-contiguous B was producing all zeros.
  - Blocking #2: Added `clear_weight_transpose_caches()` in `Executor::Drop` — pointer-keyed cache had no lifetime management; address reuse could serve stale data.
  - Poison recovery on all cache lock sites.
  - `debug_assert_eq` for buffer density in precompute.
  - Documented M≥2 threshold rationale (categorical, not tuned).
  - Two new tests: non-constant non-contiguous B (must NOT enter rescue), constant non-contiguous B (must enter rescue).
- **Result:** TTFT 167ms (was 989), decode 1.10–1.67× ORT (load-dependent), end-to-end 1.50× ORT. All guard tests green. x86_64 + aarch64 clippy clean. Platform naming lint passes.
