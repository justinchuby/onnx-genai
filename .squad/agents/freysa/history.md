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
