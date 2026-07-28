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
