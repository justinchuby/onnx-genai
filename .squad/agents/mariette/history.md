# Mariette — History

## 2026-07-12: Joined
Hired as a Metal/MPS kernel engineer for the new Apple Metal EP for ONNX Runtime (`../onnxruntime-mps`). Owns the heavy compute kernels: MatMulNBits (int4, the decode hot path — target llama.cpp Metal parity), GroupQueryAttention, softmax, RoPE, RMSNorm. Reference ExecuTorch (`backends/apple/mps`) + PyTorch (`aten/src/ATen/native/mps`). Correctness vs CPU reference first, then optimize with simdgroup matrix ops / threadgroup tiling. Tested via onnx-genai. Key prior context: onnx-genai's CPU recipe (accuracy_level=4 + quantized head) already beats LM Studio short-context; the Metal EP aims to beat everyone on Apple Silicon.

- 2026-07-14T19:05:00Z — Offline per-EP ONNX conformance harness and `docs/EP_CONFORMANCE.md` merged to origin/main in `1dfab0d`; process-bridge design recorded in decisions.

- 2026-07-15 — Vendored cpuinfo beneath its crate so cargo publish succeeds (merged `65cc851`).

## 2026-07-15T00:00:00Z — Cross-agent session update

- Applied final CUDA DLPack review fixes and documented the honest CPU-executor boundary; merged in the GPU-DLPack wave.

## 2026-07-16T15:39:27Z — Scribe session update

- Extended Mobius PR #404 with GLM-5.2 IndexShare DSA and improved-MTP export; it remains open and rebased on merged #398.

## 2026-07-16T18:11:48+0000 — Mobius full-IQ export review

- 🟢 Cleared Pris's Mobius PR #406 update: all ten native block formats match the onnx-genai `BlockQuantizedMatMul` format, dimension, and byte-preservation contract.
- PR #406 remains awaiting user action.

## 2026-07-16T19-27-57+0000 — Scribe session update

- 🟢 Cleared Pris's Mobius `797fff9` PR #406 fixes: mixed-native export uses a 4-bit/block-32 scaffold, native IQ bytes remain exact, serialized genai opset v1 is present, and pure-Q8 behavior remains unchanged (238 tests).

- 2026-07-18: Attention review cycle completed: initial rejection corrected and final revision approved.
