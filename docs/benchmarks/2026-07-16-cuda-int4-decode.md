# Native CUDA int4 decode gap report

**Date:** 2026-07-16  
**GPU:** NVIDIA H200  
**Model:** `/home/justinchu/qwen2.5-0.5b-int4-onnx/model.onnx`

## Result

The end-to-end native CUDA benchmark did not reach model loading, so no CUDA
decode throughput can be reported.

Command:

```text
cargo run --release -p onnx-genai-bench --features bench-native \
  --bin profile_native -- \
  --model /home/justinchu/qwen2.5-0.5b-int4-onnx \
  --tokens 4 --warmups 1 --runs 2 \
  --prompt "The capital of France is" --ep cuda
```

Exact error:

```text
Error: load native decoder /home/justinchu/qwen2.5-0.5b-int4-onnx/model.onnx

Caused by:
    native CUDA decode is not available yet: onnx-runtime-session currently accepts GPU preferences but its executor still instantiates only the CPU EP
```

`profile_native` already exposes `--ep cpu|cuda`, with CPU as the default.
`bench-native` already includes `onnx-runtime-ep-cuda`; there is no separate
bench `cuda` feature. The blocking integration gap is that
`onnx-runtime-session` depends on and instantiates only
`onnx-runtime-ep-cpu`. `NativeDecodeSession::load` therefore rejects CUDA
before building a session.

## CUDA kernel gaps in this graph

The original graph has 299 nodes and 13 distinct `(domain, op_type)` pairs.
Cross-referencing every node against the 62 names in `CUDA_COVERED_OPS`
identified three op types without CUDA kernels:

| Priority | Domain / op | Nodes | Role |
|---:|---|---:|---|
| 1 | `ai.onnx::Gather` | 2 | Token embedding lookup and attention-mask shape indexing. The embedding gather is on the main decoder data path. |
| 2 | `ai.onnx::Shape` | 1 | Produces the attention-mask shape consumed by one of the missing `Gather` nodes. |
| 3 | `ai.onnx::Constant` | 2 | Produces two int64 constants used by the attention-mask reformat subgraph; these may be removable by constant folding but have no CUDA registration in the unoptimized graph. |

All remaining graph op types are named in `CUDA_COVERED_OPS`:

| Domain / op | Nodes |
|---|---:|
| `com.microsoft::MatMulNBits` | 121 |
| `ai.onnx::Mul` | 48 |
| `com.microsoft::SkipSimplifiedLayerNormalization` | 48 |
| `ai.onnx::Add` | 24 |
| `ai.onnx::Sigmoid` | 24 |
| `com.microsoft::GroupQueryAttention` | 24 |
| `ai.onnx::Cast` | 2 |
| `ai.onnx::ReduceSum` | 1 |
| `ai.onnx::SimplifiedLayerNormalization` | 1 |
| `ai.onnx::Sub` | 1 |

Thus CUDA decode has two independent blockers: session/executor CUDA EP
selection must first be wired, then this exact model still needs CUDA support
or an explicit CPU fallback/constant-folding strategy for `Gather`, `Shape`,
and `Constant`.

## CPU control

Omitting `--ep` selected CPU and decoded successfully with two tiny tokens:

```text
profile_native: ... ep=Cpu ... tokens=2 warmups=0 runs=1
throughput: 0.22 tok/s, 4628.422 ms/step
generated_token_ids: [12095, 13]
```

This confirms the default CPU behavior remains intact. The short control run
is not a replacement for the established optimized CPU reference of
approximately 0.50 tok/s.
