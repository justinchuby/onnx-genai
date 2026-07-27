### 2026-07-27: FP16 GEMV review follow-ups
**By:** Deckard
**What:** Documented each NEON f16 inline-asm conversion site with the stabilization condition for replacing it with f16 intrinsics, and tightened FP16 GEMV guards to `1e-4` relative / `1e-5` absolute. The model-scale guard now runs under 1, 3, 7, and 11 Rayon workers to cover Apple Silicon worker-count differences.
**Why:** Chew verified the asm is bit-exact but noted the maintainability hazard; the comments preserve that context until Rust `f16` and aarch64 f16 conversion intrinsics stabilize. Chew measured 2.38e-7 max relative f64-reference drift, 1.73e-6 FP16-vs-F32 parity, and 2.28e-7 odd-tail absolute error, so the new thresholds keep cross-chip headroom while catching real FP16 accumulate, lane, or tail regressions.
