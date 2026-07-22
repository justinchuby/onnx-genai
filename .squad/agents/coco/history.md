# Coco — History

## 2026-07-12: Joined
Hired as a Metal/MPS kernel engineer for the new Apple Metal EP for ONNX Runtime (`../onnxruntime-mps`). Owns data/quantization/elementwise kernels: GatherBlockQuantized (int4 embedding), quantize/dequantize (int4/int8/fp8), KV ops, elementwise/activations, reshape/transpose/cast. Reference ExecuTorch + PyTorch MPS. Op set must match exactly what onnx-genai/Mobius models emit (MatMulNBits, GQA, GatherBlockQuantized, RoPE, RMSNorm). Tested via onnx-genai runtime.

- 2026-07-14T19:05:00Z — Tracer AutoDiagnosis and roofline module merged in `8607687`; Hodge review GREEN. Follow-up decision requires first-class missed-fast-path diagnosis from executor selection metadata.

- 2026-07-15 — Bundled oneDNN in Linux and macOS Python wheels (merged `ef89a95`).

### 2026-07-16T00:00:00Z — Performance-and-design wave
Applied CUDA coverage documentation correction for the merged kernel slice.

## 2026-07-16T00:00:00Z — onnx-rs upstream text-format port review
- Cleared merged commit `23e4995` 🟢: 10 added tests make 16 upstream-derived text-format cases and assert real parser/IR/codec behavior.
- Confirmed 89 passing onnx-rs tests with no ignored or vacuous cases; documented unsupported functions, non-tensor type forms, complex/2-bit dtypes, and literal tensor/sparse payload syntax.

### 2026-07-22T14:59:36+0000 — WP-B landed
WP-B landed: Coco's initial WP-B3 admission work was superseded by raw-protobuf authority fixes that completed the epic.
