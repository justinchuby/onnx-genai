# Decision: Default symmetric zero-point for CUDA GatherBlockQuantized (#702)

**Author:** Quaid (CUDA/Rust)
**Date:** 2026-08-11
**Issue:** #702 — Large mobius-converted models produce empty / non-finite output on native CUDA but correct output on CPU.

## Root cause

The CUDA `com.microsoft::GatherBlockQuantized` kernel
(`crates/onnx-runtime-ep-cuda/src/kernels/gather_block_quantized.rs`) left the
dequant `offset = 0` when the optional `zero_points` input was absent. The CPU
reference (`crates/onnx-runtime-ep-cpu/src/kernels/gather_block_quantized.rs`,
`default_zp = 1 << (bits - 1)`) and ONNX Runtime instead use the symmetric
midpoint. GGUF-style embedding tables (converted 14B/27B models) carry no
explicit zero-point, so CUDA dequantized every embedding against 0 instead of 8
(int4), yielding empty output (immediate EOS) on the 14B and non-finite logits
on the 27B — a native-vs-CPU correctness divergence.

## Fix

In the NVRTC kernel source, initialize `int offset = 1 << (bits - 1);` and only
override it from `zero_points` when that pointer is non-null. The
explicit-zero_points path is byte-for-byte unchanged. General fix — no
model-name gates.

## Tests

- Extended the existing CUDA parity test's host oracle to use the symmetric
  midpoint when `with_zp == false` (previously asserted offset 0). GPU parity now
  proves CUDA == CPU/ORT for absent zero_points across int4/int8, fp16/fp32.
- Added `gather_block_quantized_source_uses_symmetric_default_zero_point`, a
  host-only guard (no GPU) asserting the kernel defaults to `1 << (bits - 1)`.
- Both pass on RTX GPU (`cargo test -p onnx-runtime-ep-cuda gather_block_quantized`).

## Incidental

Fixed 4 pre-existing broken `gqa_decode_fp16.rs` test call sites (missing the
`kv_layout` arg added by #782) by passing `0` (legacy BNSH) — required to compile
the crate's test binary.
