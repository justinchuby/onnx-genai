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
