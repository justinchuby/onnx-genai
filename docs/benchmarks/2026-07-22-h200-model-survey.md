# H200 native decode model survey — 2026-07-23

## Method

- GPU: NVIDIA H200; HBM roofline assumed to be 3.35 TB/s.
- Binary: `profile_native`, release build with `bench-native,cuda`.
- Decode settings: CUDA, greedy, prompt `Hello`, 2 warmups, 3 measured runs, and steady-state timing with the first 8 emitted tokens excluded.
- Each throughput below is the median reported by the benchmark.
- Weight volume is the sum of `model.onnx` and its external data file. The first-order decode roofline is `3.35e12 / weight_bytes`; it excludes KV traffic and therefore is an optimistic upper bound.
- Generated token IDs from both output lengths were decoded with each model's tokenizer. All six outputs were readable, grammatical continuations rather than garbled token streams.

## Results

The efficiency column uses the 128-token measurement, where the requested laptop-baseline comparison was made.

| model | dtype | tok/s @128 | tok/s @1024 | weight GB | roofline tok/s | % of roofline @128 | coherent? |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-0.5B | INT4, including quantized `lm_head` | 312.87 | 84.77 | 0.866 | 3,869.60 | 8.09% | Yes, at both lengths |
| Llama-3.2-1B Instruct Q4KM | Q4KM body, FP16 tied `lm_head` | 450.61 | 439.00 | 1.105 | 3,031.42 | 14.86% | Yes, at both lengths |
| Llama-3.2-1B Instruct | FP16 | 44.35 | 44.23 | 2.489 | 1,346.14 | 3.29% | Yes, at both lengths |

At 1024 tokens, the corresponding roofline efficiencies are 2.19%, 14.48%, and 3.29%. The Qwen slowdown with sequence length is especially large, while Llama Q4KM remains nearly flat.

## Qwen laptop-baseline check

**Qwen2.5-0.5B did not beat 380 tok/s.** Its measured 312.87 tok/s at 128 output tokens is **67.13 tok/s below** the RTX 4060 laptop baseline, a 17.67% shortfall. The three measured runs were 312.87, 312.23, and 313.06 tok/s, so this result was stable rather than a single-run outlier.

## Fusion observation and optimization headroom

Llama Q4KM is the clear winner at 450.61 tok/s and sustains 439.00 tok/s at 1024 tokens. Its graph has a Q4 `MatMulNBits` body but an FP16 tied-embedding `lm_head` (`Transpose` + `MatMul`), matching the shape targeted by the new head fusion and consistent with the expected approximately 449 tok/s result. The fully FP16 Llama graph, however, reached only 44.35 tok/s rather than the expected approximately 449 tok/s.

The **FP16 Llama is furthest from its roofline**, reaching only 3.29%. The graph uses dense `MatMul` throughout all 16 transformer layers, whereas the fast Q4KM graph uses the optimized `MatMulNBits` path for its body. The near-identical 128- and 1024-token rates suggest a fixed per-token compute/kernel cost rather than growing KV traffic is dominant. A likely explanation is that the head fusion is either not selected for this graph form or is overwhelmed by slow/unfused dense FP16 matmuls, with additional launch overhead from the non-fused residual, activation, normalization, and attention sequence. Kernel tracing should next confirm the selected `lm_head` variant and identify whether dense matmuls are using cuBLAS/cuBLASLt or a fallback path.

No model failed to load or run.

