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
4. Optionally add a BLAS/oneDNN backend.

Comparisons with llama.cpp or vLLM are deferred until the native path is
quantized and threaded; comparing today's fp32 scalar-oriented path would not be
representative.
