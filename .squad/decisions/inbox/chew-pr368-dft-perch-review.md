# PR #368 — DFT kernel with vDSP fast path (Perch bioacoustics)

**Reviewer:** Chew
**Date:** 2026-07-28T21:44:00-07:00
**Author:** Deckard
**Verdict:** 🟡 APPROVE WITH NON-BLOCKING NITS

## FFT Error Bound

Independently characterised f32 radix-2 FFT error vs f64 naive DFT reference:

| N    | max_abs_err | max_rel_err |
|------|-------------|-------------|
| 8    | 1.85e-7     | 1.44e-7     |
| 64   | 4.79e-6     | 2.27e-6     |
| 256  | 1.72e-5     | 7.83e-6     |
| 1024 | 6.69e-5     | 1.93e-5     |
| 4096 | 5.67e-4     | 9.47e-5     |

Error grows as O(√N · log₂N · ε), consistent with radix-2 butterfly accumulation.
At N=1024 (the Perch transform size), absolute error is ~7e-5 — well within
f32 inference tolerance. vDSP matches this bound (same sign convention, same
precision class).

**The f64 naive DFT reference is a fair ground-truth**: it uses double-precision
twiddle factors computed independently from both the f32 radix-2 and vDSP paths.
This is NOT two approximations compared to each other.

## Normalisation Convention

**Correct.** vDSP `vDSP_DFT_zop` complex DFT:
- Forward: no scaling (matches ONNX forward DFT)
- Inverse: no scaling (code manually applies 1/N to match ONNX inverse)

The dangerous factor-of-2 convention applies to vDSP's **real FFT** (`vDSP_fft_zrip`),
which packs real data into a split-complex format. Deckard correctly chose
`vDSP_DFT_zop` (complex DFT with explicit real/imag buffers), which has no hidden
scaling. **This is the right API call and avoids the classic trap.**

## Sign Convention

Independently verified with probe: `vDSP_DFT_FORWARD` uses the negative-exponent
convention (X[k] = Σ x[n]·exp(-2πi·k·n/N)), matching ONNX spec exactly.
Test: `exp(2πi·n/4)` input → energy at bin[1] = 4+0i, confirming negative exponent.

## `onesided` Parameter

**Correct for both even and odd N.**
- N=8 onesided → 5 bins (8/2+1) ✓
- N=7 onesided → 4 bins (7/2+1 = 3+1 via integer division) ✓
- DC bin and Nyquist bin correctly placed
- Perch uses N=1024 (even, power-of-two) — primary path exercised

## `axis`, `inverse`, Complex Input

- `axis` normalisation handles negative values correctly; excludes last dim (complex component)
- Non-final axis: strided extraction via `input_strides[axis]` is correct
- `inverse`: 1/N scaling applied after both vDSP and radix-2 paths; roundtrip test passes
- Complex input: `complex_dim == 2` reads both real and imaginary components ✓

## Fallback Dispatch

- vDSP fires for N ≥ 4 power-of-two: Perch's N=1024 hits this path (1000 dispatches confirmed)
- N < 4 power-of-two (e.g. N=2): falls to radix-2 FFT — `fft_fallback_reachability` test proves this
- Non-power-of-two: falls to naive O(N²) DFT — correct, no counter (acceptable for rare path)
- `vDSP_DFT_zop_CreateSetup` returns NULL on failure → correct `None` propagation

## Attribution / Amdahl

Deckard's `ONNX_GENAI_PROFILE_OPS=1` breakdown shows DFT = 0.80% of Perch model time.
Amdahl projection: max 1.008× from DFT alone — correctly stated as negligible.
The honest conclusion ("invest in elementwise vectorization, 66% of model time") is
exactly right. No inflated claims.

## Dispatch Discipline

- `DFT_VDSP_TEST_HITS`: static AtomicU64, `cfg`-gated, manifest claim present, test fires counter ✓
- `DFT_FFT_TEST_HITS`: static AtomicU64, unconditional, manifest claim present, test fires counter ✓
- Inverse rule: if vDSP guard intercepted the radix-2 path, `fft_fallback_reachability` (N=2) would fail ✓

## OS-Gating CI Break

The fix (`99c33539`) correctly gates `DFT_VDSP_TEST_HITS` import and assertions behind
`#[cfg(any(target_os = "macos", target_os = "ios"))]`. The test still compiles on Linux
(counter reads and assertions are simply skipped). `check_cross_compile.sh` passes for
the FFI-free subset; full verification requires CI's ubuntu-latest runner (which is
correct — the ep-cpu crate links Accelerate and needs a macOS sysroot).

## Portability

- No hardware-specific constants (no cache sizes, no thread counts derived from M1 Max)
- vDSP is available on all Apple Silicon (M1–M4, base through Ultra) and Intel Macs
- `vDSP_DFT_zop_CreateSetup` returns NULL for unsupported lengths → graceful fallback
- No fitted thresholds — the power-of-two guard is structural, not empirical

## Model Artifact

Perch model not present in tree or tracked in git. Fetch-measure-delete rule satisfied.

## Non-Blocking Nits (N1–N2)

**N1.** The `vdsp_matches_fft` test threshold of `1e-2` is 150× looser than necessary.
Measured max error at N=1024 is ~7e-5. Tighten to `5e-4` (still generous) to catch
real regressions like a missing 1/N normalization. Current threshold would pass even
with a 100× scaling error.

**N2.** No test covers `onesided=true` with odd N through the kernel interface (the naive
fallback is only unit-tested for pow-2). Add a kernel-level test with odd dft_length
to confirm the full path. Low risk (code is correct, verified by probe) but coverage gap.

## Summary

The numerics are sound. vDSP complex DFT is the right Accelerate API choice — it avoids
the real-FFT scaling trap, uses the same sign convention as ONNX, and the code correctly
applies inverse normalization. The implementation handles all parameter combinations
(axis, onesided, inverse, complex input) correctly. Dispatch discipline is satisfied.
The Amdahl analysis is honest and the claim is appropriately scoped.

The only material concern is the excessively loose test threshold (N1), which is a
testing quality issue, not a correctness bug. It should be tightened before or shortly
after merge to prevent silent regressions.
