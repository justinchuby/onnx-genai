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

## 2026-07-27T10:25:00-07:00 — Kernel layout reorganization plan

- Produced planning document for reorganizing `kernels/` GEMM family into a `kernels/gemm/` subdirectory, requested by Justin Chu.
- Key recommendation: group by role-in-dispatch, not per-platform. Disagreed with pure per-platform split — it encourages the architecture fork the crate has banned.
- Proposed layout: `gemm/mod.rs` (dispatch), `gemm/half.rs` (cross-platform), `gemm/x86_sgemm.rs`, `gemm/x86_bf16.rs`, `gemm/accelerate.rs`, `gemm/portable.rs`.
- Filed to `.squad/decisions/inbox/resch-kernel-layout-plan.md`.

## 2026-07-27T10:40:00-07:00 — Platform-naming lint + x86 GEMM renames (PR #278)

- Renamed `simd_gemm.rs` → `x86_sgemm.rs` and `bf16_gemm.rs` → `x86_bf16.rs`. Pure rename, zero behaviour change.
- Created `scripts/check_platform_naming.py` — CI lint that catches files with single-arch cfg and no platform marker in the name.
- Lint uses a portable-item check (top-level items not preceded by platform cfg) to avoid false positives on files like `simd_normalize.rs` that have portable entry points.
- Known gap documented: doesn't catch within-file missing implementations (the sdpa dot_f32 case).
- Guard-break proven: restoring old names triggers lint failure with actionable error.
- All 945 tests pass; clippy green on both aarch64 and x86_64; dispatch-reachability tests unchanged.
- Filed to `.squad/decisions/inbox/resch-platform-naming-lint.md`.

## 2026-07-27 — Cross-Target Compilation Check (PR #319)

Phase 0 of Roy's structural fix plan.  Added `scripts/check_cross_compile.sh` —
catches `cfg(target_os)` gating errors that the `x86_64-apple-darwin` recipe misses.

- Script targets `x86_64-unknown-linux-gnu` (changes both arch AND os).
- On CI (ubuntu-latest): full offline crate set, native target — no overhead.
- On macOS (local dev): FFI-free subset (ort-sys/cpuinfo excluded due to missing Linux headers).
- Uses `--all-targets` (is_undilated and #227 were in lib-test builds).
- Teaching failure message explains WHY x86_64-apple-darwin is insufficient, cites PR #317.
- Guard-break proof: synthetic `is_undilated` in onnx-runtime-ir — old recipe passes, new script fails.
- Wired into `.github/workflows/ci.yml` quality job.
- Known gaps: can't check ep-cpu from macOS; can't catch runtime dispatch; Windows cfg via portable matrix.
- Filed to `.squad/decisions/inbox/resch-cross-compile-check.md`.

## 2026-07-27T22:05:00-07:00 — Dispatch manifest lint (Phase 1–2 of structural fix)

- Created `dispatch_manifest.toml` — declarative table of (op, variant, platform) → minimum tier + proving counter.
- Created `scripts/check_dispatch_manifest.py` — CI lint that validates every manifest claim has its counter in the declared file.
- Seeded 6 claims: MatMul f16 M=1 (GEMV), MatMul f16 M≥2 (BNNS), Conv standard (BNNS), Conv fallback (im2col+GEMM), SDPA (NEON), KernelDispatch (prebind).
- Documented 2 deliberate exclusions: depthwise Conv, bf16 M≥2.
- Guard-break proof: renamed CONV_BNNS_TEST_HITS → lint correctly failed with targeted error message naming op, platform, tier, counter.
- Zero runtime cost: manifest is CI-only; counters remain #[cfg(test)].
- Cross-EP ready: file field can point to any crate; no CPU-specific logic.
- Would have caught 7/9 historical instances; misses compilation errors (cross-compile script) and un-claimed ops (human judgment).
- Filed to `.squad/decisions/inbox/resch-dispatch-manifest.md`.

## 2026-07-27T23:13:00-07:00 — Manifest backfill + inverse check

- Backfilled 3 new claims from PR #324: MaxPool→BNNS (tier1), Add→vDSP (tier2), MatMul f16 colmaj→NEON GEMV (tier2).
- Added 2 new exclusions: dilated MaxPool, BatchNorm fusion elimination.
- **Inverse check implemented**: any optimization counter (name does NOT contain SCALAR/FALLBACK/RESCUE/REF) without a manifest row now fails CI. This closes the "human must remember to add a row" gap that PR #324 exploited within 1 hour of the manifest shipping.
- Fixed AtomicU64 blind spot: both `check_dispatch_reachability.py` and `check_dispatch_manifest.py` now match `Atomic{Usize,U64}` and `pub static`. Counter count went from 8→12.
- BatchNorm judgement: does NOT fit [[claim]] schema — its optimization is graph-level fusion elimination, not dispatch tier. Documented honestly as [[exclusion]] pending optimizer-level counters. This is the tenth instance and a new failure mode (opset registration, not cfg gate).
- Guard-break proofs: (1) renamed POOL_BNNS_TEST_HITS → lint failed naming MaxPool/aarch64/tier1; (2) added fake counter → inverse check failed naming file and counter.
- 973 CPU EP tests pass; all 4 lints green; cargo fmt clean.
- Filed to `.squad/decisions/inbox/resch-manifest-backfill.md`.


## 2026-07-29T18:35:00-07:00 — Qwen3 native CPU ORT peak-parity wave

- On `qwen3-perf-followups` / PR #398, consolidated the profile-driven path from ~69 tok/s native CPU (about 66% of ORT) to peak/p90 parity around 110 tok/s.
- Key Resch contributions: KAI packed-SDOT trajectory, MLAS QNBit SPMD sharding, residual GQA/norm/Silu fusion, and kernel preselection (`348c39a6`) that cached MLAS packed-B plus reusable SQNBit workspace and improved the best MatMulNBits bucket ~8.6 -> 7.3 ms.
- Negative result is binding: naive per-op work stealing (`fe54dd9d`) regressed (best/median ~95/82 vs fixed SPMD ~105/100), so fixed SPMD remains default. Remaining median lever is a lower-overhead Eigen-parity/whole-step work-stealing pool.

## 2026-07-29T21:00:00-07:00 — Work-stealing decode verdict
- Integrated work-stealing decode behind `ONNX_GENAI_CPU_DECODE_SCHEDULE=steal` (`542f2ebd`) and proved it should stay opt-in: real decode regressed to best/median 97/90 tok/s vs fixed-SPMD 106/99, while ORT was 109/100.
- Final PR #398 result is ORT parity, not a clean ORT beat: fixed-SPMD 106.0/105.2/99.4 best/p90/median vs ORT 108.9/107.3/99.8. Dispatch microbench wins did not predict token throughput; fixed-SPMD locality and lower coordination dominate.

## 2026-07-29T22:00:00-07:00 — Full-width MLAS negative result
- Tested ORT-style full-width MLAS QNBit on the work-stealing backend; it regressed (8.07 ms MatMulNBits, ~93 tok/s) versus static-SPMD (7.45 ms, ~106 tok/s). Keep static-SPMD default and `ONNX_GENAI_CPU_MM_MLAS_NO_SHARD=1` diagnostic-only.

## 2026-07-30T08:20:00-07:00 — ORT-costmodel tuning verdict

- Native static-SPMD CPU EP now matches/slightly beats ORT on Qwen3 best-case and p90 throughput (110.3/109.8 tok/s vs ORT 106.2/106.0), but still trails median on the contended host because variance is higher.
- ORT-style dynamic block claiming helps the isolated full-width QNBit kernel, but loses end-to-end: full-width dynamic 91.72 tok/s best vs static-SPMD 110.32 and ORT 106.16, with stalls from pool park/wake variance across many small ops.
- Full-width path is abandoned as a live toggle. If pushed further, the next candidate lever is vendoring Eigen `NonBlockingThreadPool` for lower wakeup variance, with uncertain payoff.
Full pre-compaction history in `history-archive.md`.
