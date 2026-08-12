# Luba — History Archive

## 2026-07-26 — Joined the team
Cast into the CPU & Edge pod. Standing directive: optimizations must be portable (consumer/edge hardware, not just H200); every perf claim backed by a benchmark; SIMD/NPU paths must match the scalar/f64 reference within a justified tolerance and be locked with regression tests.

## 2026-07-27T04:35:00-07:00 — Scribe update: Mac CPU EP PR #227

- Native Mac CPU EP now has Apple-Silicon-general NEON paths for multi-thread GEMV, SDPA, SiLU, and direct-from-mmap FP16 GEMV; runtime feature detection/dispatch is expected for SIMD paths instead of machine-specific tuning.
- FP16 works because Apple Silicon NEON can widen f16 loads directly while ORT CPU widens before GEMM; keep this architectural distinction in mind for CPU EP work on other platforms.
- The campaign learned that untested SIMD paths are as risky as placeholders; new AVX/NEON/SVE/QNN paths need guard-break tests and paired scalar/reference checks.

## 2026-07-27T19:35:00Z — Roadmap wave update
- Fixed PR #294 aarch64 build by cfg-gating the x86-only perf probe after Drake lockout.

## 2026-08-11 — B3: NxrtStatus cross-module allocator fix (PR #762 rejection)

**Problem:** `NxrtStatus.message` was heap-allocated in the plugin (`CString::into_raw`) and freed in the host (`CString::from_raw`/`Drop`). Across a `cdylib` boundary with different CRTs this is UB (Windows heap corruption).

**Fix:** Replaced `*mut c_char` with inline `[u8; 256]` buffer + `message_len: u32`. `NxrtStatus` is now a pure value type — no heap, no pointers, no `Drop`, no cross-module free. `message_str()` is no longer `unsafe`.

**Also fixed:** Two `as *const i8` casts in `loader.rs` and `provider_adapter.rs` that fail on aarch64 (where `c_char = u8`). Changed to `as *const c_char`.

**Tests:** 32 nxrt-abi unit tests pass, 4 nxrt-host unit + 10 roundtrip tests pass. Clippy + fmt clean.

**Note for Chew:** Two `as *const i8` casts remain in `tests/nxrt_abi_roundtrip.rs:173,187` — need the same `c_char` fix.

## 2026-08-11 — Apple/ARM CI failure triage for #31973 and #31974

**Task:** Investigate three CI failures on upstream PRs to determine if caused by our code.

**Findings:** All three failures are **infra flakes**, not caused by our code:
1. PR #31973 `coreml / build-and-test` — FetchContent download of cpuinfo.zip failed (CDN timeout)
2. PR #31974 `iphone_simulator (arm64)` — FetchContent download of XNNPACK.zip failed
3. PR #31974 `Build Linux arm64 Debug` — Job timeout at link step 1453/1459, no compile error

**Control:** PR #31985 (docs-only, same main) passed all these jobs at the same time, confirming transient infra issues.

**Action:** No code changes. Recommend re-running failed jobs.

## 2026-08-11 (upstream CI correction wave) — Apple/arm64 CI triage

Pulled real logs for all Apple/arm64 failures on #31973 and #31974. All three failures occur before compilation: (1) cpuinfo archive download failure (#31973 coreml), (2) XNNPACK archive download failure (#31974 iphone_simulator), (3) job timeout at step 1453/1459 during `onnxruntime_mlas_test` link (#31974 arm64 Debug). All confirmed infra flakes via control PR #31985. `gh run rerun` refuses fork-PR jobs; only retrigger is a push.

## 2026-08-11 — Apple MLAS FP16 cast kernel exclusion audit

**Task:** Verify claim that `cast_kernel_neon.cpp` is excluded on Apple ARM64 and that enabling it is a ~15-line fix using baseline instructions.

**Findings:**
1. **Gap is REAL.** Apple ARM64 excluded by explicit `!defined(__APPLE__)` in mlas.h:100 and `if (NOT APPLE)` in cmake:608. Fallback is scalar loop.
2. **"Baseline instructions" claim is WRONG.** `vcvt_f32_f16`/`vcvt_f16_f32` and `float16x4_t` require `-march=armv8.2-a+fp16`. However, all Apple Silicon has FEAT_FP16, so this is a build-system issue, not a hardware issue.
3. **"~15-line fix" is misleading.** Enabling just the cast kernel (without the full fp16 family) requires a new compile-time macro, changes in mlas.h, mlasi.h, cmake, and platform.cpp — roughly 15-20 lines but with non-trivial design choices.
4. **No ARM64 LayerNormF32 kernel exists** — confirmed gap.
5. **No prior attempt** found to fix the Apple exclusion.

**Action:** Decision written to `.squad/decisions/inbox/luba-apple-mlas-audit.md`. No code changes.

## 2026-08-11 — Apple f16↔f32 cast kernel implementation

Implemented and opened draft PR https://github.com/microsoft/onnxruntime/pull/31993
against microsoft/onnxruntime. Added `MLAS_CAST_F16_NEON_SUPPORTED` macro gated
on `__APPLE__ && MLAS_TARGET_ARM64` in mlas.h, updated mlasi.h/platform.cpp dispatch,
added cast_kernel_neon.cpp to CMake for Apple with `-march=armv8.2-a+fp16` scoped
to the single file, and enhanced test_cast_fp16.cpp with special-value coverage and
non-aligned lengths. Tests not run (Linux x86-64 host). No performance claims.
Branch: `nxrt/mlas-apple-f16-cast` on `justinchuby/onnxruntime`.
