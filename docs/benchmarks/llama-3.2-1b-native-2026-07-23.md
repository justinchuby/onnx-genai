# Llama 3.2 1B Instruct generality benchmark — H200 — 2026-07-23

## Result

The selected model was **Llama-3.2-1B-Instruct**, exported from
`bartowski/Llama-3.2-1B-Instruct-GGUF`'s Q4_K_M artifact through Mobius's
standard `build-gguf --keep-quantized --runtime onnx-genai` path.

| Backend | Output tokens | Median prefill | Median decode | Median decode tok/s |
|---|---:|---:|---:|---:|
| Native CUDA, graph auto-enabled | 128 | 26.529 ms | 10.281 ms/token | **97.26** |
| ORT 1.27 CUDA, graph disabled | 128 | 3.350 ms | 1.696 ms/token | **589.66** |
| Native CUDA, graph auto-enabled | 1024 | 27.843 ms | 10.266 ms/token | **97.41** |
| ORT 1.27 CUDA, graph disabled | 1024 | 3.583 ms | 1.863 ms/token | **536.78** |

Native reached 16.5% and 18.1% of ORT at 128 and 1024 tokens. It did not
approach ORT throughput, but decode stayed flat with context length, proving
that the generic metadata-selected shared/device-KV path worked.

## Export and metadata

Mobius branch `squad/hassan-llama-metadata` at `bf4f2a6` was used:

```bash
PYTHONPATH=src python3 -m mobius build-gguf \
  Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --output /home/justinchu/llama-3.2-1b-instruct-q4km-onnx-genai-hassan \
  --keep-quantized --dtype f16 --ep cuda --runtime onnx-genai
```

The 1.1 GB package contains 112 block-32 `com.microsoft::MatMulNBits` nodes and
16 `com.microsoft::GroupQueryAttention` nodes. The source GGUF stores the tied
embedding/output table as Q6_K, which this standard path dequantizes to fp16;
the final output head is therefore `Transpose` + dense `MatMul`. No
`genai_config.json` is emitted for the onnx-genai runtime.

```yaml
required_capabilities:
- kv_cache
- grouped_query_attention
model:
  attention:
    type: grouped_query
    num_attention_heads: 32
    num_kv_heads: 8
    head_dim: 64
  architecture: llama
  max_sequence_length: 131072
kv_cache:
  native_dtype: fp16
```

Mobius previously omitted `kv_cache.native_dtype` unless a Python caller
manually supplied it. That failed the runtime gate in
`shared_kv_buffer_len_from_metadata`: GQA type + supported native KV dtype +
maximum sequence length. The generic fix derives KV dtype from the exported
model activation dtype; it is not model-name gated.

## Fast-path and coherence checks

With both `ONNX_GENAI_DEVICE_KV` and `ONNX_GENAI_CUDA_GRAPH` unset, a native
16-token diagnostic reported:

```text
cuda_graph: enabled=true captures=2 replays=26 fallbacks=0
cuda_graph_measured: captures=1 replays=13 fallbacks=0
device_kv_measured: h2d_calls=0 h2d_bytes=0 d2h_calls=0 d2h_bytes=0
generated_text: "Paris. The capital of Germany is Berlin. The capital of Italy is Rome."
```

This confirms metadata-driven shared-buffer selection, default CUDA device KV,
and automatic graph capture without a metadata override. Output was coherent,
though long greedy runs became repetitive.

## Benchmark method

- Runtime commit: `0c7be31` (`origin/main`)
- GPU: NVIDIA H200, GPU 0, 143,771 MiB
- Prompt: raw `The capital of France is` (five tokens)
- Greedy decode, EOS stopping disabled
- Two warmups and three measured runs
- Native and ORT processes ran sequentially
- Native timings used `profile_native --steady --decode-skip 1`
- ORT timings used callback timestamps added to `profile_decode`; CUDA graph
  was explicitly disabled with `ONNX_GENAI_CUDA_GRAPH=0`
- The current benchmark crate has `cuda-ort`, not the requested obsolete
  `bench,cuda` feature combination

The ORT provider library is under `.ort-cuda-1.27/root/lib`; the supplied
core-only directory lacks `libonnxruntime_providers_cuda.so`, so `root/lib` was
prepended to `LD_LIBRARY_PATH`.

## Roofline and native gap

Exact initializer bytes are 1,104,810,000. FP16 KV traffic per cached token is:

```text
16 layers * 2 (K,V) * 8 KV heads * 64 head_dim * 2 bytes = 32,768 bytes
```

At 3.35 TB/s the explicit weight-plus-average-KV rooflines are 3,026 tok/s
(128 output tokens) and 2,986 tok/s (1024). Native achieved 3.2%/3.3%; ORT
achieved 19.5%/18.0%.

A native trace attributes the dominant eager/seam time to the unfused
initializer `Transpose` feeding the dense tied output-head `MatMul`; the 112
MatMulNBits projections were not dominant. This is a generic graph
optimization/export opportunity, not a Llama dispatch issue.

## Model-name audit

The requested grep produced 175 textual hits, but inspection found no
architecture/model-type comparison or model-name dispatch. Hits were tests,
comments, benchmark defaults, compatibility type names, and provenance.
Additional searches for model-family strings adjacent to
`architecture`/`model_type` comparisons and string equality returned empty.
