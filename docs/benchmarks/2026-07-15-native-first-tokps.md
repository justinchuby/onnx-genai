# First native-runtime decode throughput (2026-07-15)

## Result

The first architecture-representative native decode measurement is:

| Backend | Result | Configuration |
|---|---:|---|
| **nxrt native CPU** | **2,111.43 tok/s; 0.474 ms/step** | median of 5 release invocations; each used 64 generated tokens/run, 1 warmup + 5 measured runs |
| nxrt native CUDA | not available | `NativeDecodeSession` rejects CUDA because `onnx-runtime-session` still instantiates only the CPU EP |

Host: 2-socket Intel Xeon Platinum 8480C (96 cores visible), NVIDIA H200 present.
Command:

```text
cargo build --release -p onnx-genai-bench --features bench-native --bin profile_native
./target/release/profile_native --synthetic --tokens 64 --warmups 1 --runs 5 --ep cpu
```

The five aggregate results were 2,132.01, 2,093.21, 2,131.87, 2,111.43,
and 2,101.07 tok/s (range 2,093–2,132 tok/s). Median output:

```text
throughput: 2111.43 tok/s, 0.474 ms/step (320 generated tokens in 151.556 ms)
```

This is a **synthetic-model result**, not a quality-bearing pretrained model result.
It measures the engine's shared `run_decode_loop` through `NativeDecodeSession` and
the native CPU executor. It must not be compared as if it were the ORT-backed
approximately 492 tok/s H200/batch-128 result: that number used a different backend,
hardware execution path, model/workload, and batch size.

## Synthetic decoder

`onnx-genai-bench::synthetic_decoder` builds the model with the native IR
`Value`/`Node`/`Graph` API and constructs an encoder `Model` to write an inspectable
ONNX artifact (about 344 KiB). The timed session is built from that same IR graph
with `InferenceSession::from_graph`, then wrapped by
`NativeDecodeSession::from_session`.

The graph has 38 nodes and the standard cached-decoder contract:

- inputs: `input_ids`, `attention_mask`, `position_ids`, and two layers of
  `past_key_values.N.key/value`;
- outputs: `logits` and two layers of `present_key_values.N.key/value`;
- dimensions: hidden 64, 4 query heads, 2 KV heads (GQA), head dimension 16,
  MLP width 128, 2 blocks, vocabulary 32;
- token embedding: `Gather`;
- each block: `RMSNormalization`, Q/K/V `MatMul`, standard
  `ai.onnx::RotaryEmbedding`, standard `ai.onnx::Attention` with in-op
  past/present KV, residual projection, and a gated MLP;
- SiLU is decomposed as `Mul(x, Sigmoid(x))`;
- final `RMSNormalization` and `MatMul` LM head.

The ONNX file is emitted for inspection, but the timed path currently uses the IR
graph directly. Reloading this dynamic cached graph from the emitted ONNX file
loses the symbolic `past + current` output expression during shape inference
(`present_key_values.*` becomes unresolved). That loader/dynamic-shape issue is a
separate blocker before this exact generated fixture can use `NativeDecodeSession::load`.

## Prioritized real-decoder gaps

Registry presence was checked directly in
`onnx-runtime-ep-cpu/src/kernels/mod.rs` and
`onnx-runtime-ep-cuda/src/kernels/mod.rs`.

1. **P0 — exported GQA/MHA operator compatibility.**
   `com.microsoft::GroupQueryAttention`, `MultiHeadAttention`, and `QAttention`
   are not registered by either EP. CPU does register standard
   `ai.onnx::Attention` v23/v24, including GQA and past/present KV semantics, but
   real ORT Qwen exports commonly contain the contrib operators and therefore
   need kernels or a canonicalization rewrite. CUDA registers only
   `com.microsoft::Attention`, not standard `ai.onnx::Attention`.
2. **P0 — `com.microsoft::MatMulNBits`.**
   It is absent from both registries. This blocks the usual weight-only
   4-bit/8-bit Qwen2.5-0.5B deployment artifacts even after attention works.
3. **P0 CUDA — graph plumbing plus executor dispatch.**
   CUDA lacks registrations for `Gather`, `Reshape`, `Transpose`, `Concat`,
   `Slice`, `Expand`, `Unsqueeze`, and `Shape`; these are pervasive in exported
   decoder shape/KV subgraphs. More fundamentally, native session execution does
   not instantiate the CUDA EP yet, so no native CUDA tok/s number is possible.
4. **P1 — rotary export compatibility.**
   CPU registers standard `ai.onnx::RotaryEmbedding` v23, but not the
   `com.microsoft` rotary form used by some ORT exports. CUDA registers neither
   form. A real Qwen graph therefore needs a contrib alias/rewrite on CPU and an
   actual CUDA kernel.
5. **P1 — fused residual RMS normalization.**
   `com.microsoft::SkipSimplifiedLayerNormalization` is absent from both EPs.
   Both have `SimplifiedLayerNormalization` and `RMSNormalization`, so this is
   decomposable to `Add` plus RMS norm, but current fused exports will not place
   without a rewrite.
6. **P2 — Swish/SiLU convenience forms.**
   CPU registers standard `Swish` v24; CUDA does not. Both can express SiLU as
   `Mul` plus `Sigmoid`, as the synthetic graph does, so this is not a fundamental
   blocker if canonicalization runs.

Registration alone is not sufficient: the CPU standard Attention implementation
is currently f32-only, and the native cached-shape loader issue above must also be
fixed for typical fp16/bf16 serialized Qwen exports. The shortest path to a real
Qwen-class native CPU run is: contrib GQA/RoPE canonicalization (or aliases),
dynamic present-KV shape preservation, then fp16/bf16 coverage. Quantized models
add `MatMulNBits` immediately after those.
