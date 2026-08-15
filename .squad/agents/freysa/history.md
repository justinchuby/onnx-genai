# Freysa — History

## Role
MPS Perf & Testing engineer. Owns Apple Metal EP correctness (vs ORT CPU reference), benchmarking, Metal profiling, and E2E testing through onnx-genai. Pairs with Sebastian. Joined 2026-07-12.

## Historical context

Joined during Apple Metal EP bringup. Rejected Batty's onnx-rs binding for lossy paths; cleared Deckard's revision. Handled WP-B raw-protobuf admission rejection. July–August: `disable_cpu_ep_fallback=1` added to conformance setup for PR #762 to prove EP assignment non-vacuously. Corrected false deferral on `Session_GetEpGraphAssignmentInfo` (present since ORT 1.24).

Pre-2026-08-11 entries archived in `history-archive.md`.

## 2026-08-11 — PR #31993 test revision (f16 cast dispatch, lockout)

**Task:** Fix S1 and S2 flagged by Holden's review of MLAS Apple f16 cast PR. Luba (author) and Holden (reviewer) both barred under lockout.

**S1:** Replaced `Convert(1.0) == 1.0f` assertion (passes on scalar fallback too) with direct pointer checks: `GetMlasPlatform().CastF16ToF32Kernel != nullptr` and `CastF32ToF16Kernel != nullptr`. Only honest assertion — NEON and scalar paths are bit-exact by design.

**S2:** Added signalling NaN (`0x7C01`), mid-range denormal (`0x0200`), negative denormal (`0x8001`) to f16→f32. Added `signaling_NaN()` to f32→f16.

**Head:** `54f2fc8`. Tests not run (Linux host); Apple CI will validate.

**Lesson:** A dispatch test that uses values producible by both paths is vacuous. Test the dispatch mechanism itself (pointer, flag) rather than a value that happens to be correct.

## Archive pointer

Older entries in `history-archive.md`.

## 2026-08-12 — PR #31973 comment accuracy fix

- Rewrote 3 stale Welford comments → centered two-pass description
- Renamed 2 scenario names removing obsolete "two-pass=NaN/100%err" suffixes
- Added cross-reference comments at both threshold sites
- Made benchmark comment architecture-neutral
- 41+2 disabled / 43 with disabled; clang-format clean; leak check clean
- Head: `697189f2ae`

## 2026-08-12 — PR #31973 lockout revision: stale Welford comments

- Under reviewer lockout (Luv barred from revision), rewrote three stale Welford comment sites in `test_layernorm.cpp:275,646,1055` to describe centered two-pass.
- Renamed scenario names: `"catastrophic_1e6 (two-pass=NaN)"` → `"catastrophic_1e6"`, `"catastrophic_1e7 (two-pass=100%err)"` → `"catastrophic_1e7"`.
- Added "Keep in sync" cross-references at both threshold literal sites.
- Changed benchmark comment "AVX2 kernel" → "SIMD kernel" for architecture neutrality.
- Fresh build: 41 passed + 2 disabled; clang-format clean; no leaks. Head `697189f2ae`.

## 2026-08-12 — PR #762 final items (ort_discovery + validate_write_dtype)

- Consolidated `find_ort_lib_dir` into `tests/common/ort_discovery.rs`; all three integration tests use `#[path]` include.
- Documented `validate_write_dtype` as test-only contract helper; named `scratch_alloc_bytes` as actual guard.
- 283 passed, 0 failed; clippy clean; fmt clean.
- Commit `5258e0281`.

## 2026-08-12 — PR #762 final items (ort_discovery + validate_write_dtype docs)

Consolidated `find_ort_lib_dir` into `tests/common/ort_discovery.rs`; all three integration test files use `#[path]` include. `validate_write_dtype` documented as test-only contract helper. 283 passed / 0 failed; clippy clean; fmt clean. Closed both substantive items Gaff flagged. PR #762 marked ready for review.
