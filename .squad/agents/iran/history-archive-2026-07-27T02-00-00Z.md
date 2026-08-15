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
- **Commit `281481a6`**: Switched from `NXRT_CALIB_DEBUG` gated `eprintln!` to `tracing::debug!` (per `docs/architecture/ERROR_AND_LOGGING_CONVENTIONS.md`). Added `tracing = "0.1"` as optional dep behind existing `tracing` feature. Without feature, `NXRT_CALIB_DEBUG` fallback preserved.
- **half_gemm.rs overlap**: Complementary, not duplicated. GEMV (M=1 bandwidth-optimal, inline asm fcvtl ARMv8 base) vs GEMM (M>1 compute-optimal, vcvt_f32_f16 intrinsic requiring FEAT_FP16). Dispatch collision fixed in `ed7a65e3`. Consolidation deferred to separate PR.

### 2026-07-27 — PR #275 Mac prefill campaign (BNNS/AMX + first-decode spike elimination)

#### Phase 1: BNNS prefill (commits `f0cbd786`, `aa219b4b`)
- Implemented three-regime dispatch: M=1→NEON GEMV, M≥2+macOS→BNNS `BNNSMatMul` f16→f32 (AMX), M≥2+other→`half_gemm.rs`
- Initial null result (TTFT unchanged at 989ms) — diagnosed non-contiguous f16 weights bypassing BNNS. Fixed via `FilterCache` + `contiguous_b_f16` OnceLock.
- Second null for lm_head (column-major B) — fixed via `bnns_matmul_f16_trans_b()` zero-copy column-major path and `trivial_batch` fix for engine's `[1,M,K]` A shape.
- **TTFT: 989→348→167 ms** (5.9× prefill speedup). Production BNNS at 260–346 GFLOPS across 168 calls.

#### Phase 2: First-decode spike elimination (commit `9f1e7684`)
- Justin identified ~967ms unaccounted end-to-end gap: expected 1013ms (167+846), measured 1980ms.
- Diagnosed root cause: shape-keyed kernel cache → prefill M=40 → decode M=1 creates new kernel instances with cold OnceLock caches → 169 kernels re-transpose ~776 MB.
- Critically, lm_head (K=896, N=151936, column-major) fell through ALL fast paths to f32 densification (544 MB alloc) → ~960ms alone.
- **Fix 1:** Global weight-transpose cache (`LazyLock<Mutex<HashMap<usize, Arc<Vec<u16>>>>>`). Key = data pointer, value = `Arc<Vec>` shared across kernel instances. Eager pre-transpose during model load (+7ms load, saves 30ms on first decode).
- **Fix 2:** Column-major GEMV — recognized B[K,N] with strides [1,K] = B^T[N,K] in row-major memory. Route directly to `neon_gemv_f16_col_parallel` at M=1. Zero-copy, no transpose needed.
- **End-to-end arithmetic reconciles:** TTFT(170) + 49×14.2 = 865ms ≈ 865ms measured. Spike gone.

#### Final results (measured at load ~12, 5 runs with 1 warmup)
| metric | before campaign | after | ORT | vs ORT |
|---|---:|---:|---:|---:|
| TTFT | 989 ms | 170 ms | 109 ms | 1.56× |
| decode | 57.6 tok/s | 70.6 tok/s | 42.2 tok/s | **1.67×** |
| end-to-end | 17.7 tok/s | 57.8 tok/s | 38.7 tok/s | **1.50×** |
| model load | 105 ms | 114 ms | 1671 ms | 0.068× |

Guard tests green: `fp16_m1_decode_reaches_neon_gemv_not_half_gemm` ✓, `fp16_m_ge2_prefill_reaches_bnns_not_half_gemm` ✓. 959 CPU EP tests pass. Both aarch64 and x86_64 clippy clean.

#### Remaining leads
1. TTFT gap: 170ms vs ORT 109ms (1.56×). BNNS production at 260–346 GFLOPS vs 2451 microbenchmark — per-call M may be too small for AMX to reach steady state, plus ~200ms non-GEMM overhead.
2. Further end-to-end: currently 1.50× ORT; decode dominates at 1.67× but TTFT holds it back.

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

### Session 2026-07-27-c: trans_b zero-copy for column-major vocab weights

**Commit:** `f0cbd786`
**Branch:** `squad/mac-prefill-bnns`

#### Key finding: trans_b eliminates the 180 ms contiguous copy

Column-major B[K,N] in memory is row-major B^T[N,K]. BNNS supports `trans_b: true`, allowing C = A @ (B^T)^T = A @ B without any data copy. The lm_head vocab projection (896×151936, 272 MB column-major) now passes directly to BNNS — zero copy.

Two bugs blocked the path initially:
1. **FilterCache key**: needed `(M,K,N,trans_b)` not just `(M,K,N)` to avoid shape collisions
2. **batch_shape check**: engine emits A as [1,M,K] producing trivial batch [1]. Changed from `batch_shape.is_empty()` to `trivial_batch` (all dims == 1).

Probe technique: added diagnostic `eprintln!` behind env var `NXRT_BNNS_TRANS_B_PROBE`, discovered `batch_empty=false` was blocking the path. This is the 6th instance of "works in unit test, not in production" in this crate.

#### Lead 1 results (BNNS GFLOPS gap)

Production BNNS at 260–346 GFLOPS vs microbenchmark 966–2451 GFLOPS. The gap decomposes as:
- ~2× from M=40 vs M=128 (AMX tile utilization)
- ~3.7× from production environment (mmap'd weights, TLB pressure, GCD scheduling)
- NOT explained by: cold cache, NEON interleaving, Rayon pool, or SPMD pool

#### Lead 2 results (non-GEMM attribution)

Profiled at M=11: non-GEMM ops are only **2.1% (8 ms)** of prefill time. The "~200 ms non-GEMM" hypothesis was wrong — the lm_head contiguous copy (287 ms at M=11 with cold mmap pages) was the real cost, and trans_b eliminates it.

#### Measurements (load 24–27)

| | native | ORT | ratio |
|---|---:|---:|---|
| TTFT | **167.4 ms** [166.8, 168.6] | 108.4 ms | 1.54× |
| decode | 55.31 tok/s | 41.88 | 1.32× ✅ |
| end-to-end | 24.54 | 38.42 | 0.639× |

**Overall: 989 → 167 ms = 5.9× speedup.**
**Decode unregressed**: guard test green, 1.32× ORT.

#### Verification
- `cargo fmt --all -- --check`: clean
- clippy aarch64 `--all-targets -D warnings`: clean
- clippy x86_64 `--all-targets -D warnings`: clean
- Full CPU EP suite: 953 passed, 0 failed
- All BNNS tests green including new `bnns_matmul_f16_trans_b_matches_normal`
- Decode guard `fp16_m1_decode_reaches_neon_gemv_not_half_gemm`: green

#### Durable lessons
- **Column-major = row-major transpose** — this algebraic identity eliminates copies for any column-major weight; future non-contiguous weights should check for this pattern.
- **Trivial batch dims block dispatch** — the engine promotes 2-D to 3-D with batch [1]. Any dispatch gate using `batch_shape.is_empty()` silently blocks the fast path. Use `trivial_batch` (all dims == 1) instead.
- **Env-var probes are essential** — the `NXRT_BNNS_TRANS_B_PROBE` technique found the bug in seconds after the benchmark showed no improvement. Always probe the production path, never trust the unit test alone.

---

### Session 2026-07-27T15:20 — Prefill Fusion Instrumentation

**Task:** Instrument optimizer passes on Qwen2.5-0.5b-f16 prefill graph, diagnose which fusions fire, count actual ops, and test highest-value missing fusion.

#### Findings
1. **SDPA fusion correctly not firing** — model has 24 pre-fused `Attention` contrib ops from exporter; no decomposed pattern to match.
2. **Fusions that fire:** MatMul+Bias (120), CpuSiluFusion (24). Fusions that correctly don't: SDPA, LayerNorm, GELU.
3. **Actual ops dispatched during prefill:** 350 (after full optimization from 590 raw).
4. **SiblingProjectionMerge implemented** — merges Q/K/V and gate/up siblings into wider GEMMs, 350→326 ops. Fully correct (bit-for-bit parity + dispatch reachability tests).
5. **Measurement: fusion REGRESSES TTFT** — 91→153ms (+68%). BNNS prefers individual smaller GEMMs over one wider merged GEMM. Gated behind `ONNX_RT_SIBLING_MERGE=1`.

#### Performance (Apple M1 Max, load 6–8)
- TTFT: 90ms (40 tokens) — baseline unchanged (fusion disabled by default)
- Decode: 70 tok/s (55–61% roofline) — no regression
- End-to-end: 55 tok/s

#### Durable lessons
- **Wider GEMMs ≠ faster on BNNS** — Apple's BNNS internal tiling handles individual [M,K]×[K,N] calls very efficiently. Merging siblings into one wider call hits worse tile utilization at non-power-of-2 N dimensions.
- **Dispatch overhead is not the prefill bottleneck** — at 90ms for 350 ops, per-op average is 0.26ms (compute-dominated). The "dispatch overhead" theory holds only under high system load.
- **Always measure after implementing** — the op-count-reduction argument was logically sound but empirically wrong for this hardware/backend combination.
- **Pre-fused contrib ops from the exporter** — when the model exporter already fuses SDPA/RMSNorm/RotaryEmbedding into contrib ops, runtime fusion passes have nothing to match. This is expected for optimized HF exports.
