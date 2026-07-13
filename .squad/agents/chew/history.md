# Chew — History

## 2026-07-12: Joined
Hired as a Code Reviewer specializing in numerics/precision as the runtime took on fp16/Q4 quantization, GQA KV, and Mobius model conversion. Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Context: a prior Q4 GGUF→ONNX conversion "loaded but produced garbage" (missing Qwen2 biases + wrong reverse-permute) and a sampling RNG bug returned token 0 — exactly the silent precision defects to catch. Verify against references; require coherent output, not just successful load.


## 2026-07-13T18:30:00Z — Review/fix batch
- Reviewed Leon's DESIGN §40 SWA/attention-sink work and approved it with three optional LOW nits; no rejection lockout needed.
