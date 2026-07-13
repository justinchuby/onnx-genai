# Coco — History

## 2026-07-12: Joined
Hired as a Metal/MPS kernel engineer for the new Apple Metal EP for ONNX Runtime (`../onnxruntime-mps`). Owns data/quantization/elementwise kernels: GatherBlockQuantized (int4 embedding), quantize/dequantize (int4/int8/fp8), KV ops, elementwise/activations, reshape/transpose/cast. Reference ExecuTorch + PyTorch MPS. Op set must match exactly what onnx-genai/Mobius models emit (MatMulNBits, GQA, GatherBlockQuantized, RoPE, RMSNorm). Tested via onnx-genai runtime.
