# First real-model native decode: Gemma 4 E2B (2026-07-15)

## Result

This is the first successful native-runtime decode of a real pretrained model
through the in-tree CPU execution provider.

| Item | Value |
|---|---|
| Model | Gemma 4 E2B, fp32, 15 layers |
| Attention | GQA: 8 query heads, 1 KV head, head dimension 256, sliding window 512 |
| Backend | native `onnx-runtime` `ep-cpu` (not the ORT-backed path) |
| Host | 96 logical CPUs (`nproc`) |
| Build | Cargo release profile |
| Single-step validation | 29.5 s/step (0.03 tok/s), deterministic on rerun |
| Four-token sample | 0.03 tok/s, 30,437.728 ms/step |
| Generated token IDs | `[7001, 563, 7001, 563]` |

The coherence sample used:

```text
cargo build --release -p onnx-genai-bench --features bench-native
./target/release/profile_native \
  --model /home/justinchu/gemma4-e2b-onnx \
  --tokens 4 --warmups 0 --runs 1 \
  --prompt "The capital of France is"
```

Output:

```text
profile_native: model=/home/justinchu/gemma4-e2b-onnx/model.onnx ep=Cpu layers=15 prompt_tokens=[818, 5279, 529, 7001, 563] tokens=4 warmups=0 runs=1
throughput: 0.03 tok/s, 30437.728 ms/step (4 generated tokens in 121750.913 ms)
generated_token_ids: [7001, 563, 7001, 563]
```

This sample establishes model loading, KV-cache growth, deterministic greedy
generation, and repeated native execution. It is a correctness milestone, not a
competitive performance result.

## Correctness enablers

Each shared change generalizes an ONNX operation or runtime rule required by the
Gemma export rather than special-casing the model:

- `kernels/gelu.rs` — accept supported floating-point storage types by widening for computation and narrowing to the declared output type, which is the general floating-point GELU contract needed by exported decoder MLPs.
- `kernels/group_query_attention.rs` — support differing query/key sequence lengths, correct absolute rotary and causal positions, fp16/fp32 conversion, and typed outputs, which are required by cached GQA decode and follow the operator's general past/current sequence semantics.
- `kernels/reduce_ops.rs` — execute `ReduceSum` on `int64` without lossy float conversion, which is required by integer shape/mask subgraphs and preserves ONNX integer reduction semantics.
- `kernels/reshape.rs` — copy logical tensor bytes without assuming fp32 while enforcing equal input/output dtypes, which lets Gemma reshape non-fp32 values and correctly implements dtype-preserving Reshape.
- `kernels/rmsnorm.rs` — widen supported floating inputs for fp32 accumulation and narrow to the input/output dtype, which enables decoder normalization while retaining the numerically appropriate general RMSNormalization behavior.
- `onnx-runtime-session/src/executor.rs` — resolve dynamic `GroupQueryAttention` present-cache shapes from runtime query/key/past dimensions and attach node context to kernel errors, enabling safe KV allocation for serialized cached decoders generally.
- `onnx-runtime-shape-inference/src/handlers/norm.rs` — infer contrib `GroupQueryAttention` output and growing present-K/V shapes, preserving the query type and cache geometry defined by the operator for any compatible graph.

The native decode wrapper also treats `position_ids` as optional because fused GQA
exports can derive positions from cache length, validates finite logits, reports
the detected layer count, and prints generated token IDs for deterministic
correctness checks.

## Performance analysis / next steps

The observed approximately 29.5 seconds per decode step is explained by three
known limitations in the native fp32 CPU path:

1. GEMM parallelism partitions over M rows
   (`matmul.rs:89`, `c.par_chunks_mut(MC*n)`), so M=1 autoregressive decode is
   effectively single-threaded.
2. MatMul re-densifies the weight operand on every invocation
   (`matmul.rs:186-187`, `to_dense_f32_widen(b)`), copying roughly 10 GB per
   generated token.
3. The model is unquantized fp32 and the kernel does not use BLAS.

The optimization roadmap is:

1. Prepack and cache weights to eliminate per-token weight re-densification.
2. Add N-dimension-parallel GEMV for M=1 decode.
3. Implement the quantized `MatMulNBits` path.
4. Continue improving the built-in SIMD GEMM backend.

Comparisons with llama.cpp or vLLM are deferred until the native path is
quantized and threaded; comparing today's fp32 scalar-oriented path would not be
representative.

## Perf pass 1 (2026-07-15)

This historical pass added an initializer-only prepack seam and benchmarked the
now-removed oneDNN GEMM backend:

- The session marks each kernel input as constant only when its `ValueId` is in
  `Graph::initializers`. `MatMul`, `FusedMatMulBias`, and `FusedGemm` may cache
  dtype widening or stride compaction only for those marked inputs; activations
  are always read live.
- Contiguous fp32 tensors are borrowed as `Cow<[f32]>` instead of copied.
  Non-f32 or strided constant operands are materialized once in a per-kernel
  `OnceLock<Vec<f32>>`.
- The former native backend has since been removed because its benchmark
  results did not justify its source-build dependencies. `SimdX86` is now the
  default x86 fast path, with Generic as the portable fallback.

The requested one-token, zero-warmup command was measured on the same 96-core
host. The established earlier baseline was 29,500-30,438 ms/step; host load
during this pass made a same-session unmodified baseline slower:

| Build | ms/step | Speedup vs same-session baseline |
|---|---:|---:|
| Unmodified Generic baseline | 36,673.970 | 1.00x |
| Prepack/zero-copy, Generic | 40,134.062 | 0.91x |
| Prepack/zero-copy + historical oneDNN | 32,044.370 | 1.14x |

A two-token check (one prefill plus one decode continuation) measured
36,564.846 ms/step for Generic and 31,487.407 ms/step for oneDNN, a 1.16x
oneDNN speedup. Generated token IDs stayed deterministic (`[7001]` and
`[7001, 563]`).

The code now removes the diagnosed repeated dense-copy path safely, but this
host run did not show a standalone prepack speedup and the combined result did
not reach the expected large gain. Historical oneDNN verbose output confirmed
`gemm_api` execution for all MatMuls, including the `5x1536 @ 1536x262144`
head. Remaining work is to profile time outside primitive execution (including
native GEMM setup and other tensor movement), implement quantized
`MatMulNBits`, reduce attention cost, and investigate further threading where
the removed native backend is unavailable.
