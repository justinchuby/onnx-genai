# Nabil — History

## 2026-07-12: Joined
Hired to lead the ORT plugin-EP integration for a new **Apple Metal/MPS execution provider** for ONNX Runtime (repo `../onnxruntime-mps`). Motivation: onnx-genai is ORT-kernel-bound on Apple Silicon (ORT's generic int4 CPU/WebGPU kernels lag llama.cpp's hand-tuned Metal); a custom MPS EP with hand-tuned kernels can beat everyone on Mac. The EP must support all ops onnx-genai/Mobius use: MatMulNBits (int4), GroupQueryAttention, GatherBlockQuantized, RoPE, RMSNorm, softmax, elementwise. Tested end-to-end by the onnx-genai runtime (`ONNX_GENAI_EP` selects it). Reference kernels: ExecuTorch + PyTorch MPS backends.
