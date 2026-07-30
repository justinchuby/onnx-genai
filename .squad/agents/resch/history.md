# Resch — History

## Project Context (joined day)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **State when joined:** Native CUDA EP beats/parity ORT on several Foundry models; correctness suite green (int8/block32 f64-adjudicated in #190). Team reorganized into pods; CPU & Edge pod formed to broaden hardware coverage beyond CUDA/Metal.
- **Role:** Intel CPU Optimization Engineer — x86-64 CPU-EP perf (AVX2 baseline, AVX-512/VNNI), MLAS interplay, int8 DP4A GEMV.
- **Requested by:** Justin Chu
- **Joined:** 2026-07-26

## 2026-07-26 — Joined the team
Cast into the CPU & Edge pod. Standing directive: optimizations must be portable (consumer/edge hardware, not just H200); every perf claim backed by a benchmark; SIMD/NPU paths must match the scalar/f64 reference within a justified tolerance and be locked with regression tests.

## 2026-07-26T20:00:00Z — Scribe update

- 2026-07-26T20:00:05Z — Fixed pre-existing main CI red by rustfmt-formatting `decode_spmd.rs`; direct main commit `1bf119af` unblocked dependent PRs.
## 2026-07-27T04:35:00-07:00 — Scribe update: Mac CPU EP PR #227

- Native Mac CPU EP now has Apple-Silicon-general NEON paths for multi-thread GEMV, SDPA, SiLU, and direct-from-mmap FP16 GEMV; runtime feature detection/dispatch is expected for SIMD paths instead of machine-specific tuning.
- FP16 works because Apple Silicon NEON can widen f16 loads directly while ORT CPU widens before GEMM; keep this architectural distinction in mind for CPU EP work on other platforms.
- The campaign learned that untested SIMD paths are as risky as placeholders; new AVX/NEON/SVE/QNN paths need guard-break tests and paired scalar/reference checks.

## 2026-07-27T04:45:00-07:00 — Cross-platform compilation fix (commit 41da3d6b)

- Fixed 5 defects in `crates/onnx-runtime-ep-cpu` that broke CI on every non-Apple platform.
- Root cause: cfg-gated code that compiled only on aarch64-macOS — the mirror image of the original problem this campaign solved.
- Key fix: `dot_f32` scalar fallback made always-reachable (matching `axpy_f32` pattern); imports and parameters scoped to their cfg contexts.
- Enforced the "one implementation, no arch fork" mandate: SIMD paths are opt-in early-returns; the scalar baseline is always present and compilable on every target.
- All 922 local tests green; NEON dispatch test confirms fast path still selected on Apple Silicon.

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
