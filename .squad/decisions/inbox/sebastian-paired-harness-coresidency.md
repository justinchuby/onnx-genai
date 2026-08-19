# Benchmark method: the paired A/B harness depresses the native arm

**Author:** Sebastian (Performance Engineer)
**Context:** Phase 18, `docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md` §38

## Decision

Any native-vs-ORT claim about a cell that runs longer than a few milliseconds,
at an intra-op thread count above 16, must be measured with `bench_generic
--native-only` and `--ort-only` as separate arms. The paired harness
(`scripts/ort_ab/ab.py`) remains correct for parity checking, for short cells,
and for base-vs-new comparisons of our own binaries.

## Why

ORT's intra-op pool spin-waits for a long window after its last op, so it is
still burning all 32 CPUs when the native arm starts. Measured at t=32,
`gemm_nbits_llama3_8b_qkv_t8` is 6.90 ms with the native arm alone and 28.71 ms
paired — 4.18×, reproduced to three significant figures across two sessions an
hour apart. The dense f32 control is worse (4.79×), so this is contention, not
a property of any kernel. The tax is asymmetric: ORT pays only 1.2-1.3× for our
co-residency, because our pools park quickly and its does not.

## Consequence for existing numbers

Native-vs-ORT ratios already tabulated in that document for long cells at wide
thread counts overstate the gap by up to 4.8×. The honest t=32 figure for
`llama3_8b_qkv_t8` is 10× behind ORT, not 41×.

This does **not** affect base-vs-new comparisons of two native binaries (both
arms pay the same tax), so the phase 16 result in §36 stands unchanged.
