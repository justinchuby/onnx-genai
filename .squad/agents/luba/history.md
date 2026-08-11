# Luba — History

## Role
ARM CPU / QNN EP Engineer — ARM64 CPU (NEON/SVE) perf + Qualcomm QNN NPU execution provider, edge/Windows-on-ARM. CPU & Edge pod. Joined 2026-07-26.

## Historical context

Joined during CUDA parity wave. Fixed PR #294 aarch64 build by cfg-gating the x86-only perf probe. Worked through the Mac CPU EP PR #227 wave (Apple-Silicon-general NEON paths). Triage: ARM/Apple CI failures for upstream PRs #31973 and #31974 were all confirmed infra flakes (CDN timeouts, job timeouts) — no code bugs. Helped fix B3 NxrtStatus inline buffer for PR #762.

Pre-2026-08-11 entries archived in `history-archive.md`.

## 2026-08-11 — Apple MLAS FP16 cast kernel audit and implementation (PR #31993)

Audited: Apple ARM64 genuinely excluded from NEON f16↔f32 cast kernel. Gap is real; "baseline instructions" claim was wrong (vcvt_f32_f16 needs `-march=armv8.2-a+fp16`). All Apple Silicon has FEAT_FP16, so this is a build-system issue, not hardware. Introduced `MLAS_CAST_F16_NEON_SUPPORTED`, gated on `__APPLE__ && MLAS_TARGET_ARM64`. Draft PR #31993 opened. No performance claims.

## 2026-08-11 — PR #31993 Holden review and Freysa revision

Holden reviewed `nxrt/mlas-apple-f16-cast` @ `df162d9`. All gating confirmed correct. Two substantive findings: S1 (vacuous dispatch test — 1.0 converts identically on scalar path), S2 (missing sNaN/denormal coverage). Freysa revised under lockout (Luba barred). Both issues fixed. Head: `54f2fc8`. PR remains draft pending Apple CI.

## Archive pointer

Older entries in `history-archive.md`.

## 2026-08-11T23:55Z — PR A: Apple Accelerate infrastructure option

- **PR:** https://github.com/microsoft/onnxruntime/pull/32001 (draft)
- **Branch:** `nxrt/mlas-apple-framework-option`
- **Option:** `onnxruntime_USE_APPLE_ACCELERATE` (default OFF, FATAL_ERROR on non-Apple)
- **Linkage:** `find_library(Accelerate)` → system framework (macOS/iOS/universal2)
- **Verified:** Default-OFF configure on Linux x86-64 is behaviour-identical to upstream/main
- **Cannot verify here:** Apple SDK resolution, actual linking on Apple targets
- **Next:** PR B (Accelerate cblas SGEMM/SDPA), separate branch, needs Apple hardware

## 2026-08-12 — PR #32001 review fixes (Isidore under lockout)

- Luv reviewed PR #32001 and found three substantive issues: S1 (FATAL_ERROR wrong idiom), S2 (no `build.py` argument), S3 (dangling compile definition).
- **Luba and Luv both locked out** of the revision. Isidore revised all three.
- S1: `message(WARNING ...) + set(onnxruntime_USE_APPLE_ACCELERATE OFF)` matching SVE/KleidiAI idiom.
- S2: `--use_apple_accelerate` plumbed through `build_args.py` + `build.py`.
- S3: `target_compile_definitions` line removed — no consumer exists yet.
- Head: `d16a108252`. PR remains draft. @justinchuby directed PR A stay separate from kernel PRs.
