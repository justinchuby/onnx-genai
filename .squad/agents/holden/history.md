# Holden — History (compacted 2026-08-11T23-30-00Z)

**Role:** Security engineer for onnx-genai. Focuses on unsafe/resource/supply-chain, path confinement, FFI, allocation bounds, and adversarial tests.

## Durable lessons
- Enforce reviewer lockout end-to-end: no author revises their own rejected artifact.
- Require `catch_unwind` on every `extern "C"` callback — panic across FFI is UB.
- `OrtGraph*`/`OrtNode*` must NOT be stored beyond callback return.
- `OrtMemoryInfo` passed to `EpDevice_AddAllocatorInfo` must outlive the `OrtEpDevice` — ORT stores the raw pointer; do not call `ReleaseMemoryInfo` on success.
- Leak scans must grep source content in the diff, not only `.squad/` paths and commit messages.
- "Not caused by us" is not the same as "safe to mark ready" — draft until CI board is green.

## Historical context

Pre-2026-08-11 entries archived in `history-archive.md`. Covers: Wave 3 path confinement, EP plugin FFI audit (CRITICAL catch_unwind, HIGH static mut race), EP plugin re-audit (C1 partial — compute_execute unguarded), EP plugin final ship verdict (🟡 YELLOW), EP milestone 2 audit (M2-1 stream_release leak), #31973/#31974 initial re-reviews (all blockers fixed; detected .squad/ leakage), PR #762 final sign-off (APPROVE).

## 2026-08-11 — PR #31974 re-review + PR #762 final sign-off

**PR #31974 re-review** (after B1-B6 fixes on `nxrt/mlas-bf16-layernorm`):
- Verdict: READY FOR REVIEW. B5 stat-narrowing fixed (WriteStat<U=float>); B4 test file deletion correct (zero MLAS calls); B6 17/17 BFloat16 tests pass, all 5 op families covered. Anti-fallback `ConfigEp(DefaultCpuExecutionProvider())` confirmed.
- Flagged `.squad/` leakage in git history of both upstream branches (content reachable after delete commit).

**PR #762 final sign-off** at `fb9d757b3`:
- Verdict: APPROVE. CPU EP E2E: 23/23 ORT conformance tests. CUDA: zero factories + actionable status, `catch_unwind` at 18+ sites. nxrt ABI: 10/10 roundtrip. Honesty sweep: clean.
- 211+ tests, 0 failed, 7 ignored. Clippy clean. Cross-platform c_char verified.

## 2026-08-11 — Re-review PR #31973 (AVX2 LayerNorm)

Fresh adversarial pass after rebase onto upstream/main. All 6 original blockers remain fixed. Found 2 substantive issues: internal name leaks ("Iran" at test_layernorm.cpp:1191, "Pris" at :492). Kernel is reachable, tests are non-vacuous, no upstream conflict. Verdict: NOT ready to leave draft until name leaks are scrubbed; otherwise clear. Triggered lockout-clean fix by Chew.

## 2026-08-11 (upstream CI correction wave) — Lessons enforced

Persona name leaks found only after two prior diff sweeps. Rule confirmed: grep the actual source content in the diff, not just `.squad/` paths. Two PRs converted back to draft — "infra flake" reasoning does not justify marking ready while CI is red.

---

## 2026-08-11 — Review of onnxruntime PR #31993 (MLAS Apple f16 cast)

**Task:** Adversarial review of `nxrt/mlas-apple-f16-cast` @ `df162d9`. Verified ARM64-only gating across preprocessor, CMake, and universal2 paths. Confirmed kernel is wired up (not dead code). Found two substantive issues: (S1) dispatch reachability test is a no-op — cannot distinguish scalar from vectorised path; (S2) missing sNaN test cases. No blocking issues. No performance claims, no persona leaks. Recommended: push to CI as draft, fix S1/S2 before marking ready.

**Lockout:** Holden barred from revision after authoring the review; Freysa revised.

**Output:** `.squad/decisions/inbox/holden-review-31993.md` (now merged)

## 2026-08-12 — Delta review PR #31974 (commit a12c7ddde3)

Focused delta re-review of three new BF16 tests and comment hygiene. All three tests verified non-vacuous: PrePack tests compare against independent f32 reference, GenericBroadcast confirmed to trigger `outer_dep=true` via scale_dims padding. Tolerance comment now exactly matches `checkers.cc` implementation — numpy.isclose semantics with correct defaults. Stat tests retain 36× margin to detect pre-fix bf16 round-trip bug. No internal vocabulary, no weakened assertions. 20 BF16, 106 LayerNorm, 7 SkipLayerNormPrePack tests all green. No blockers — ready to leave draft.

## 2026-08-12 — Delta re-review PR #31974 (commit a12c7ddde3)

- **Task:** Focused adversarial delta review of three new BF16 tests and comment hygiene.
- **Verdict:** No blockers. All three tests verified non-vacuous (traced PrePack path at `layer_norm_impl.cc:678-693, 720-755`; confirmed `outer_dep=true` via `layer_norm_helper.h:132-142`).
- **Tolerance:** Verified `checkers.cc:117-120` — `absolute + relative × |expected|`. Stat tests retain 36× margin over pre-fix bug (0.00011 effective vs 0.004 pre-fix error).
- **Hygiene:** Clean sweep — no internal vocabulary in branch diff or commit messages.
- **Test counts:** 20/20 BF16, 106/106 LayerNorm suite, 7/7 SkipLayerNormPrePack.
- **Nit (non-blocking):** `GenericBroadcast` test reference comment slightly imprecise — scale's dim-0 is broadcast-expanded, not "outer dim broadcast."

### 2026-08-12 — Focused review of PR #32001 blocker fix (ec73d63a0e)
- **Scope:** Validation that `--use_apple_accelerate` rejects all non-macOS-arm64 targets.
- **Verdict:** No blockers. Rejection set is complete. Tests are non-vacuous (each guard independently necessary). Body matches code. Lint clean. No persona leaks.
- **Substantive (non-blocking):** S1: duplicate validation in `build.py` and `build_args.py` — `build.py` copy is dead code. S2: accept-path tests don't verify cmake arg emission.
- **Decision:** Ready to leave draft. macOS CI lane deferral is reasonable.
- **Output:** `.squad/decisions/inbox/holden-32001-focused.md`

### 2026-08-12 — Focused review of PR #32001 after Zhora deduplication (3a0bd75aa3)

- Confirmed S1 resolved: single validation site now in `build_args.py`; `build.py` dead-code copy removed.
- Error messages now name the actual cause (non-macOS wording distinguishes wrong OS from wrong arch).
- 13/13 tests pass; ruff clean; no persona leaks; body matches code.
- PR #32001 confirmed **ready for review**.

### 2026-08-12 — Final gate reviews: PR #32001 (184f76a00e) and PR #31993 (a0a9d98)

**PR #32001 (Apple Accelerate infrastructure):**
- Rejection set now complete: 17 tests covering non-macOS, iOS, tvOS, visionOS, Catalyst, Android, build_wasm, build_wasm_static_lib, rv64, non-arm64 arch. `--use_xnnpack`/`--use_qnn`/`--build_apple_framework` correctly NOT rejected (they don't change target away from macOS arm64).
- Message assertions verified non-vacuous: each checks exit code 2 + distinctive substring in stderr. No shared-prefix false matches.
- Catalyst decision honest: CMake comment accurately states PLATFORM_NAME unavailable in plain cmake; CLI rejects `--macos Catalyst`; gap is documented.
- Single copy of rules confirmed: validation only in build_args.py:1103-1123; build.py:881-883 appends CMake define only.
- All invariants unchanged: default OFF, PRIVATE define inside guarded block, arm64 detection with CMAKE_OSX_ARCHITECTURES fallback, universal2 absent.
- Body matches code: parser.error() / exit status 2, no stale BuildError reference.
- One substantive non-blocker: ruff check reports 3 lint diagnostics in test_build_args.py (PLC0415, SIM105).
- **Verdict: READY to leave draft.**

**PR #31993 (NEON f16↔f32 cast kernel):**
- CI chain verified: mac.yml cpu job → machine=target=arm64 → build.py --test → onnxruntime_mlas_test runs natively.
- NaN assertions verified non-vacuous: catch NaN→Inf, NaN→finite, sign flips. Payload deliberately unasserted (hardware vs scalar divergence). Both F16→F32 and F32→F16 directions covered.
- QEMU correctly described as emulation. FEAT_FP16 correction accurate (vcvt is baseline ARMv8-A).
- Gate: macOS-arm64-only via TARGET_OS_OSX; MLAS_CAST_F16_NEON_SUPPORTED separate from MLAS_F16VEC_INTRINSICS_SUPPORTED; no universal2; no x86_64.
- Body disclosure honest: states tests run but per-test output unverifiable via gh.
- **Verdict: READY to leave draft.**

**Output:** `.squad/decisions/inbox/holden-final-two.md`

## 2026-08-14 — Zero-copy hybrid Linux default raised to 2 GiB (#936, merged)
Measured (#925, H200, driver 580.105.08, CUDA 13, native VMM) that the #864 zero-copy aperture ceiling
is **Windows/WDDM/VidMm-specific and absent on Linux** — generation byte-identical up to 6.795 GB
host-mapped weights (704 binds, n=3), ~15× the WDDM ~0.44 GB corruption onset. Made
`ZERO_COPY_SAFE_BUDGET_BYTES` platform-conditional in `crates/onnx-runtime-ep-cuda/src/weight_paging.rs`:
Windows stays 256 MiB, Linux/non-Windows raised to **2 GiB** (both consts referenced every build via
`cfg!(target_os="windows")`). Unlocks a ~8× memory-constrained decode win (67.04 vs ~8.5 tok/s over-budget,
byte-identical). 2 GiB bounded on purpose (>3× below measured-safe 6.795 GB, only Hopper tested); operator
override via `ONNX_GENAI_ZERO_COPY_HYBRID_BUDGET_BYTES`. Feature stays opt-in default-OFF; Windows unchanged
(do NOT inherit the Linux conclusion there — inverse of #783).
