# Mariette — History

## 2026-07-12: Joined
Hired as a Metal/MPS kernel engineer for the new Apple Metal EP for ONNX Runtime (`../onnxruntime-mps`). Owns the heavy compute kernels: MatMulNBits (int4, the decode hot path — target llama.cpp Metal parity), GroupQueryAttention, softmax, RoPE, RMSNorm. Reference ExecuTorch (`backends/apple/mps`) + PyTorch (`aten/src/ATen/native/mps`). Correctness vs CPU reference first, then optimize with simdgroup matrix ops / threadgroup tiling. Tested via onnx-genai. Key prior context: onnx-genai's CPU recipe (accuracy_level=4 + quantized head) already beats LM Studio short-context; the Metal EP aims to beat everyone on Apple Silicon.
