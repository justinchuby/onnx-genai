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

## Perf pass 3: accuracy_level=4 int8/VNNI

The native CPU `MatMulNBits` kernel now honors `accuracy_level=4`. It quantizes
each fp32 activation row symmetrically with `scale = max_abs / 127`, caches
unpacked int8 weights and block scales, accumulates int8 products in int32, and
uses runtime-dispatched VNNI on x86-64. The model already contains
`accuracy_level=4`, so no environment override or graph rewrite was needed.

| Measurement | fp32 prepack | int8/VNNI |
|---|---:|---:|
| Decode throughput | 0.50 tok/s | **1.01 tok/s** |
| Decode latency | 1,993.045 ms/step | **992.584 ms/step** |
| Speedup | — | **2.01x** |

The release benchmark used 4 generated tokens, 1 warmup, and 2 measured runs:

```text
cargo run --release -p onnx-genai-bench --features bench-native \
  --bin profile_native -- \
  --model /home/justinchu/qwen2.5-0.5b-int4-onnx \
  --tokens 4 --warmups 1 --runs 2 \
  --prompt "The capital of France is"
```

It generated `[12095, 13, 1084, 374]` (` Paris. It is`), matching the fp32
prepack run. The benchmark host is an Intel Xeon Platinum 8480C; runtime
detection reported both AVX-VNNI and AVX512-VNNI/AVX512VL, selecting the
AVX512-VNNI path.

Correctness tests compare against an independently dequantized fp32 matmul for
block sizes 32 and 128, partial K, M=1, and batched M>1. They use
`0.05 + 5% * |reference|` tolerance: activation quantization is intentionally
lossy, and a 5% relative bound plus 0.05 absolute allowance covers values near
zero without weakening the existing exact default-path tests.
