# Native INT4 decode: Qwen2.5 0.5B (2026-07-16)

## Result

This is the first native CPU decode measurement using real
`com.microsoft::MatMulNBits` block-wise INT4 weights.

| Item | Value |
|---|---|
| Model | Qwen2.5-0.5B-Instruct, 24 layers |
| Weights | 121 `MatMulNBits` nodes, INT4, block size 32, `accuracy_level=4` |
| Backend | native `onnx-runtime` CPU EP (not ORT-backed) |
| Build | Cargo release profile |
| Decode throughput | **0.21 tok/s** |
| Decode latency | **4,870.933 ms/step** |
| fp32 Gemma 4 E2B reference | 0.03 tok/s, 30,437.728 ms/step |
| Improvement over reference | 6.25x lower step latency / 7x rounded throughput |

The model was built locally with the `onnxruntime-genai` model builder from
the available Conda environment:

```text
LD_LIBRARY_PATH=$HOME/.conda/envs/onnx/lib \
  $HOME/.conda/envs/onnx/bin/python -m onnxruntime_genai.models.builder \
  -m Qwen/Qwen2.5-0.5B-Instruct \
  -o /home/justinchu/qwen2.5-0.5b-int4-onnx \
  -p int4 -e cpu -c /home/justinchu/hfcache-int4bench
```

It produced an 865 MB external-data decoder and `tokenizer.json`. The original
CPU-targeted export uses the standard ONNX `SimplifiedLayerNormalization` and
packed-QKV `GroupQueryAttention`, neither of which the native CPU decode path
currently accepts. For this measurement only, a local copy at
`/home/justinchu/qwen2.5-0.5b-int4-onnx-native-unpacked` rewrites that
mathematically equivalent normalization to the supported Microsoft contrib op
and inserts 24 `Split` nodes to expose Q, K, and V separately. It shares the
same external weight data and is not committed.

```text
./target/release/profile_native \
  --model /home/justinchu/qwen2.5-0.5b-int4-onnx-native-unpacked \
  --tokens 4 --warmups 1 --runs 2 \
  --prompt "The capital of France is"
```

Output:

```text
profile_native: model=/home/justinchu/qwen2.5-0.5b-int4-onnx-native-unpacked/model.onnx ep=Cpu layers=24 prompt_tokens=[785, 6722, 315, 9625, 374] tokens=4 warmups=1 runs=2
throughput: 0.21 tok/s, 4870.933 ms/step (8 generated tokens in 38967.461 ms)
generated_token_ids: [12095, 13, 1084, 374]
```

The greedy IDs decode to ` Paris. It is`; they are deterministic across the
two measured runs and are not a single-token loop.

## Analysis and next steps

INT4 removes the fp32 Gemma path's approximately 10 GB per-token weight
re-densification burden, reducing the observed step latency by 84%. The result
is still only 0.21 tok/s, so native decode is not competitive with llama.cpp:
scalar/reference-style MatMulNBits work and per-node execution/allocation
overhead remain dominant rather than raw weight bandwidth.

The immediate compatibility gap is also clear: standard
`ai.onnx::SimplifiedLayerNormalization` and packed-QKV
`com.microsoft::GroupQueryAttention` should be handled natively so builder
output can run without the local graph adapter. After that, profile
MatMulNBits per node, prepack its weights, and parallelize M=1 decode across
the output dimension. Those changes are required before comparing this native
CPU path fairly with llama.cpp.

## Perf pass 2 (int4)

`MatMulNBits` now uses the session's initializer-only prepack signal to
dequantize each constant weight once into output-major `[N, K]` f32 storage.
M=1 decode uses a bounded-parallel GEMV over that cached matrix; activations are
never cached. Non-decode shapes retain the shared GEMM/oneDNN path.

The same release + oneDNN command and model produced:

| Measurement | Before | After |
|---|---:|---:|
| Decode throughput | 0.19 tok/s | **0.50 tok/s** |
| Decode latency | 5,297.249 ms/step | **1,993.045 ms/step** |
| Speedup | — | **2.66x** |

Both runs generated `[12095, 13, 1084, 374]` (` Paris. It is`), so the
optimized result remained deterministic and coherent. The cached fp32 matrix
trades memory for decode speed; a future packed-int8/SDOT path can reduce that
footprint and avoid fp32 weight bandwidth.
