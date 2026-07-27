# Decision: BNNS fp16→f32 Prefill via AMX

**Author:** Iran (Mac CPU Optimization Engineer)
**Date:** 2026-07-27
**PR:** #275
**Branch:** `squad/mac-prefill-bnns`
**Commit:** `a855f826`

## Context

Decode was won in PR #227 (native 1.42× vs ORT). Prefill remained 9.6–11.9× worse than ORT (TTFT 1034–1314 ms vs 103–110 ms), making end-to-end performance 0.40–0.45× of ORT. Prefill is compute-bound (arithmetic intensity ≈20 FLOP/byte) — the opposite of decode's bandwidth-bound character.

## Decision

**Three-regime MatMul dispatch**, replacing the binary `m > 1` gate:

| M | Path | Bound |
|---|---|---|
| M = 1 | NEON GEMV (unchanged) | bandwidth |
| M ≥ 2, macOS | **BNNS `BNNSFilterCreateLayerBroadcastMatMul` fp16→f32** | compute (AMX) |
| M ≥ 2, non-Mac | `half_gemm.rs` NEON | portable fallback |

## Why BNNS

- Standard BLAS has no half-precision GEMM (`sgemm` is f32, `dgemm` is f64)
- Apple's fp16 matrix path is BNNS, which reaches AMX
- Measured 2451 GFLOPS at M=128 vs 52 GFLOPS for NEON blocked GEMM (~47×)
- ORT links no Accelerate — this is a structural advantage they cannot match without taking an Accelerate dependency
- Projected TTFT ~37 ms vs ORT's 107 ms (2.9×)

## Why M=2 threshold

Sebastian measured the crossover at M=2 for both prefill and batch decode. The per-call GCD overhead (~50 µs) is absorbed by AMX throughput even at small M.

## Constraints upheld

1. **No BNNS/Accelerate from Rayon parallel regions** — calls from dispatch level only
2. **Decode unregressed** — `fp16_m1_decode_reaches_neon_gemv_not_half_gemm` guard passes
3. **Runtime feature detection** — `bnns_matmul_available()` probes at startup, caches result
4. **One implementation** — `half_gemm.rs` remains portable fallback for non-Mac
5. **Cross-compilation clean** — clippy passes on both aarch64 and x86_64 with `--all-targets`

## Key implementation detail: b_is_weights=false

Setting `b_is_weights: true` causes `BNNSFilterApplyTwoInput` to return -1. When true, BNNS expects B data baked into the filter descriptor at creation time. Since we pass both A and B at apply time, both must be `false`.

## Tests

- Dispatch reachability: fp16 M≥2 → BNNS (atomic counter guard)
- BF16 exclusion: verified via output parity with portable reference
- Numerics: f64 reference at shapes up to 128×896×4864, tolerance √K·2e-3
- Edge values: fp16 max (65504), denormals, NaN, zero
- Bitwise determinism
- Guard-break proof: broken M threshold caught by test

## Measurement status

System load was extreme (LA 82.97 on 10-core M1 Max) during implementation. TTFT measurement deferred to a quiet window. Sebastian's microbenchmark data (2451 GFLOPS) is authoritative for the BNNS path performance.
