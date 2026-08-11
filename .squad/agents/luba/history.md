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
