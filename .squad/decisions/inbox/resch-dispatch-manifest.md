# Decision: Dispatch Manifest Lint (Phase 1–2)

**Author:** Resch (Intel CPU Optimization)  
**Date:** 2026-07-27  
**Status:** Proposed (PR pending review)  
**Depends on:** #317 (Conv tiers), #314 (kernel pre-binding), #295 (reachability lint)  
**Implements:** Roy's structural-fix-plan, Phases 1–2

---

## What

A declarative TOML manifest (`dispatch_manifest.toml`) listing every claimed `(op, variant, platform) → minimum tier` with the `_TEST_HITS` counter that proves it, plus a CI lint (`scripts/check_dispatch_manifest.py`) that blocks merge when a claim lacks its proving counter.

## Design Properties

1. **Zero runtime cost.** The manifest is a CI-only artifact. Counters remain `#[cfg(test)]`. No per-execution recording touches the hot path.

2. **Fails safe and loud.** A claim whose counter is missing produces a CI failure naming the op, platform, expected tier, counter name, and file — with a prescriptive fix message.

3. **Resists rot.** The manifest is *small and curated* (only ops we explicitly claim), not exhaustive. Rows are added at optimization-ship time. The lint enforces the pairing. Removing a row is a legitimate un-claim that passes CI — the ratchet only prevents *silent* regression, not conscious retreat.

4. **Builds on existing lints.** Each lint covers what the others cannot:
   - `check_platform_naming.py` → file with single-arch code but neutral name
   - `check_dispatch_reachability.py` → counter without paired test
   - `check_dispatch_manifest.py` → claimed optimization without proving counter

5. **Cross-EP ready.** The format is EP-agnostic — `file` can point to any crate. No CPU-EP-specific logic exists in the lint. When CUDA/Metal EPs adopt TEST_HITS counters, they add manifest rows and get identical protection.

## Historical Instance Coverage

| # | Instance | Would catch? | How |
|---|----------|:---:|-----|
| 1 | Accelerate placeholder never wired | ✅ | Manifest claims BNNS counter; counter missing → lint fails |
| 2 | gemm_generic no parallelism at M=1 | ✅ | Manifest claims GEMV counter for M=1; wrong path can't increment it |
| 3 | SDPA dot_f32/axpy_f32 x86-only | ✅ | Manifest claims SDPA_NEON; missing counter → lint fails |
| 5 | half_gemm.rs hijacked M=1 | ✅ | Test asserts GEMV counter for M=1 shape; half_gemm intercept prevents increment |
| 6 | BNNS unreachable for non-contiguous | ✅ | Test asserts BNNS counter with realistic shapes; stays 0 |
| 7 | Rescue block returning zeros | ✅ | Counter + test already caught; manifest forces counter to exist earlier |
| 8 | conv_ref.rs was only Conv on macOS | ✅ | **Central case.** Manifest says Conv/aarch64 → tier1 (BNNS counter). No counter exists → lint fails at PR time. |
| 9 | Non-Conv CNN ops on scalar | ❌ | No claimed optimization = no manifest row = invisible. Process gap. |
| 4 | Non-aarch64 compilation broke 4× | ❌ | Compilation error, not silent fallback. Caught by check_cross_compile.sh. |

**7 of 9 caught.** The two misses are covered by other mechanisms (cross-compile script) or are inherently human-judgment (adding rows for new optimizations).

## What It Still Cannot Catch

1. **Un-claimed optimizations.** If nobody adds a manifest row, the scalar reference runs without complaint. The manifest guards what you declare — it cannot invent claims.

2. **Compilation errors.** A `cfg` typo that prevents compilation is caught by `check_cross_compile.sh`, not this lint.

3. **Runtime dispatch bugs where the counter increments but the fast path is incorrect.** Correctness tests cover this — the manifest only covers reachability.

4. **Performance of the fast path itself.** The manifest proves "BNNS ran" but not "BNNS ran fast enough." Per-PR benchmarks (#306) cover this.

## Manifest Format

```toml
[[claim]]
op = "Conv"
variant = "standard"
platform = "aarch64-apple-darwin"
minimum_tier = "tier1"
counter = "CONV_BNNS_TEST_HITS"
file = "crates/onnx-runtime-ep-cpu/src/kernels/conv_ref.rs"
description = "Standard Conv dispatches to BNNS on Apple Silicon"

[[exclusion]]
op = "Conv"
variant = "depthwise"
platform = "aarch64-apple-darwin"
reason = "Depthwise conv deliberately on scalar reference pending profiling"
```

## Who Maintains It

The **op author** adds rows when shipping an optimization, alongside the counter and test. The lint enforces the commitment. The Scribe or reviewer verifies the row exists during PR review. Rot is impossible because removing the counter or test while the row exists fails CI.
