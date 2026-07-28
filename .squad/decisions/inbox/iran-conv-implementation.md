# Conv Implementation Decision — iran-conv-implementation

**Date**: 2026-07-27
**Author**: Iran (Mac CPU Optimization Engineer)
**PR**: #317
**Branch**: `squad/conv-accelerate-dispatch`

## Decision

Implement three-tier Conv dispatch: BNNS Filter → im2col+GEMM → scalar reference.

## Context

The CPU EP's Conv kernel was a scalar single-threaded reference loop (`conv_ref.rs`) running unconditionally on Apple Silicon because the optimized `conv.rs` was `cfg(feature = "mlas")`-gated and `mlas-sys` targets x86-64 Linux only. This produced a 643× performance gap vs ORT on ResNet-18.

## Implementation

### Tier 1: BNNSFilterCreateLayerConvolution (macOS/iOS only)
- Reaches AMX hardware, measured 877–1458 GFLOPS at real ResNet-18 shapes
- Handles: 2D, f32, symmetric padding, no dilation, group=1
- Supports ReLU fusion when following op is Relu
- API is deprecated (macOS 15.0) but no replacement exists for per-op dispatch (BNNSGraph requires .mlmodelc). Leave on deprecated API — same reasoning as MatMul (#275).

### Tier 2: im2col + cblas_sgemm via CpuBackend
- ~300 GFLOPS on Apple Silicon
- Fallback for: dilation, asymmetric padding, groups, 1D convolutions
- Reuses existing `gemm_with_backend` — dispatches to Accelerate on macOS, appropriate backend elsewhere
- Durable path: `cblas_sgemm` is NOT deprecated

### Tier 3: Scalar reference (unchanged)
- Correctness baseline, reachable for edge cases (non-f32, exotic configs)
- Must remain exercised by tests

## Measurements (M1 Max, uptime 1d14h, corroborated 2×)

| Model | Before | After | ORT | native ÷ ORT |
|-------|--------|-------|-----|---------------|
| ResNet-18 | 8792 ms | 93 ms | 14 ms | 0.15× |
| Whisper-tiny encoder | 3808 ms | 3808 ms | 102 ms | 0.027× |
| LLM decode | ~1.5× ORT | unchanged | — | ~1.5× ✅ |

### Why Whisper is unchanged
Whisper-tiny encoder's bottleneck is MatMul/attention (2D Conv is only the initial feature extraction, 2 layers out of ~50 ops). The Conv fix helps those 2 layers but doesn't move the needle on the graph total.

### Why ResNet-18 is still 6.7× behind ORT
Conv itself now runs at competitive GFLOPS, but the *rest* of the graph — BatchNorm, Add, MaxPool, GlobalAveragePool, Flatten, Gemm — are all still on scalar reference paths. These are the next targets.

## Risk Assessment

- **BNNS deprecation**: Accepted risk. API deprecated macos(11.0, 15.0) but no per-op replacement exists. BNNSGraph requires compiled Core ML models (.mlmodelc), ruling it out for our use case. Same exposure as MatMul (#275). Tier 2 (cblas_sgemm) provides durable fallback if filter API is eventually removed.
- **Struct layout correctness**: BNNS FFI structs defined from SDK headers. Verified by tests producing correct numerics. Independent review (Chew) required before merge.
- **Cross-platform**: x86/Linux/Windows falls through to im2col+GEMM or scalar ref. No regression possible — `gemm_with_backend` dispatches platform-appropriately.

## Follow-up Work (not in this PR)

1. BatchNorm, Add, Pool kernels → same tiered pattern needed for ResNet-18 parity
2. Whisper encoder requires overall graph execution optimization (MatMul dominates)
3. Consider Winograd for 3×3 stride-1 to close the BNNS-vs-ORT gap (ORT uses this via MLAS)
4. Depthwise convolutions (groups > 1) remain on scalar ref — needed for MobileNet/EfficientNet

## Cross-Platform Verification Recipe

The previous recipe (`--target x86_64-apple-darwin`) was insufficient — it changes `target_arch` but not `target_os`, so `cfg(target_os = "macos")` guards are not exercised. Use:

```bash
# Arch gating (catches cfg(target_arch) issues)
cargo clippy --target x86_64-apple-darwin --all-targets -- -D warnings

# OS gating (catches cfg(target_os) issues)
cargo check -p onnx-runtime-ep-cpu --target x86_64-unknown-linux-gnu
# Note: ort-sys build script blocks full workspace check without Linux sysroot.
# Fallback: cfg-inversion test (negate target_os guards, compile natively).
```

This should be standard for any PR introducing `cfg(target_os)` guards.
