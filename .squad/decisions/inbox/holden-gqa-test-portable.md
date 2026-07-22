# Make CPU GQA SIMD tests portable

## Decision

Keep the long-context GQA reference test runnable through normal runtime dispatch on every architecture. It now verifies the scalar fallback whenever AVX2+FMA is unavailable. The direct dot-product and repeated weighted-AXPY SIMD regressions early-return with a clear skip message when the runtime gate is false, preserving their AVX2/FMA mutation-detection coverage on capable x86 hosts without executing unsupported instructions on older x86 or ARM.

A test-only `ONNX_RUNTIME_EP_CPU_FORCE_NO_SIMD_X86=1` override was added to `has_simd_x86()`. It does not exist in production builds and lets unit tests exercise normal GQA dispatch with the scalar fallback on an AVX2 host.

## Verification

- AVX2 host: `cargo test -p onnx-runtime-ep-cpu --features mlas group_query` passed (17 tests).
- Forced scalar fallback: `ONNX_RUNTIME_EP_CPU_FORCE_NO_SIMD_X86=1 cargo test -p onnx-runtime-ep-cpu --features mlas group_query` passed (17 tests); SIMD-only helper regressions cleanly skip while the long-context GQA and generic AXPY coverage execute the scalar dispatch path.
- `cargo clippy -p onnx-runtime-ep-cpu --features mlas --tests -- -D warnings` passed.
