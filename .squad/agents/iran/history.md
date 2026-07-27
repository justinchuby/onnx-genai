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

### 2026-07-27 — PR #275 BNNS fp16→f32 prefill via AMX
- **Commit `a855f826`** on `squad/mac-prefill-bnns`: Implemented BNNS-based fp16→f32 MatMul for M≥2 prefill/batch-decode on Apple Silicon, reaching AMX at 2451 GFLOPS (vs 52 GFLOPS portable NEON).
- **Three-regime dispatch**: M=1 → NEON GEMV (decode), M≥2 macOS → BNNS BNNSMatMul fp16→f32 (prefill/AMX), M≥2 non-Mac → half_gemm.rs (portable).
- **BNNS FFI**: Raw binding to `BNNSFilterCreateLayerBroadcastMatMul`/`BNNSFilterApplyTwoInput`/`BNNSFilterDestroy` with correct 176-byte NDArrayDescriptor and 544-byte params struct layouts (verified against C). Critical: `b_is_weights=false` (both operands passed at apply time).
- **Threading safety**: BNNS calls from dispatch level only, never inside Rayon parallel regions (avoids 4× GCD oversubscription).
- **Tests**: dispatch reachability (atomic counter), bf16 exclusion (output parity), numerics vs f64 reference at model-scale 128×896×4864, edge values (fp16 max/denorm/NaN/zero), bitwise determinism, guard-break proof.
- **Verification**: `cargo fmt` clean, clippy clean on aarch64 + x86_64 `--all-targets -D warnings`, all 140 matmul tests pass, full CPU EP suite green. Decode guard `fp16_m1_decode_reaches_neon_gemv_not_half_gemm` passes (unregressed).
- **Initial TTFT measurement: null result** — Justin measured 989 ms (unchanged from baseline). BNNS dispatch was reaching the unit test but two production bottlenecks masked the gain.

### 2026-07-27 — Diagnosed and fixed null-result TTFT (filter cache + contiguous B rescue)
- **Commit `58bafd0d`** on `squad/mac-prefill-bnns`: Two fixes that reduced TTFT from 989 ms to **347 ms** (2.8× improvement).
- **Root cause 1 — BNNS filter cold-start**: `BNNSFilterCreateLayerBroadcastMatMul` costs 3–19 ms cold per unique (M,K,N) shape (GCD dispatch setup / AMX micro-code compilation). With ~20 unique shapes, first prefill paid ~60–380 ms. **Fix**: Thread-local `FilterCache` — `HashMap<(usize,usize,usize), BNNSFilter>` + Drop cleanup. Filter created once per shape, reused forever. Subsequent calls: ~0.3 ms → cached: 0 ms.
- **Root cause 2 — Non-contiguous vocab weight**: lm_head weight (896×151936, 272 MB) stored column-major in ONNX model. `try_matmul_half` requires contiguous inputs, so vocab bypassed BNNS entirely and fell through to element-by-element `to_dense_f32_widen` (1066 ms for 136M elements). **Fix**: `MatMulPrepack::contiguous_b_f16` — parallel strided copy via Rayon, cached per session in `OnceLock`. Rescue dispatch in `execute_with_backend` routes non-contiguous f16 B to cached copy → BNNS.
- **Measurements** (M1 Max, load 25–32, qwen2.5-0.5b-f16, 40-token prompt, 5 runs median):
  - TTFT: **347.4 ms** [346.8, 351.1] vs ORT 108.5 ms — 3.2× ratio (down from 9.1×)
  - decode: **58.36 tok/s** [57.32, 59.76] vs ORT 41.98 — 1.390× (unregressed)
  - end-to-end: 22.94 [22.24, 23.34] vs ORT 38.50 — 0.596× (up from 0.464×)
- **BNNS call profile**: 168 calls at M=40, 260–346 GFLOPS per call. Total BNNS GEMM time ~150 ms. Remaining ~200 ms is non-GEMM overhead (LayerNorm, SoftMax, RoPE, embedding, graph dispatch).
- **All guards green**: `fp16_m1_decode_reaches_neon_gemv_not_half_gemm`, `fp16_m_ge2_prefill_reaches_bnns_not_half_gemm`, `bf16_m_ge2_does_not_reach_bnns`, `bnns_f16_prefill_matches_f64_reference_via_matmul_kernel`. Full suite: 936 passed, 0 failed.
- **Verification**: cargo fmt clean, clippy clean aarch64 + x86_64 `--all-targets -D warnings`.

#### Durable lessons
- **BNNS filter creation is expensive cold** — always cache filters. Thread-local with Drop is the safe pattern (BNNSFilter is `*mut c_void`, not Send).
- **Non-contiguous weights are invisible performance cliffs** — `is_contiguous()` gates throughout the codebase silently fall through to element-by-element conversion. Any new fast path must handle or explicitly diagnose non-contiguous weights.
- **Microbenchmark ≠ production performance** — BNNS reached 2451 GFLOPS in microbenchmark but production TTFT includes filter creation, weight materialization, and non-GEMM ops. Always verify with the real model path.
- **compare benchmark creates new Engine per run** — OnceLock caches (e.g. contiguous_b_f16) are NOT shared across measured runs. Thread-local BNNS filter cache IS shared. Production (persistent Engine) TTFT is ~30 ms better than compare reports.
