# Gemma 4 E2B native CUDA re-probe — 2026-07-21

## Summary

Re-probed on `main` commit `5c48ba5a669d65da90cb17eda764bf8452774e5e`,
which contains opset-24 `ConstantOfShape`/`Gelu`/`OneHot` (`ea4036d`) and
generic `ScatterElements` fp16/bf16 data plus int32/int64 indices (`5b01a01`).

| Graph | Optimized nodes | CUDA claimed | CUDA rejected | Result |
|---|---:|---:|---:|---|
| Gemma 4 E2B target decoder | 1,299 | **1,299** | **0** | Full placement; first forward fails in `GroupQueryAttention` |
| Gemma 4 E2B assistant sidecar | 133 | 132 | 1 | Placement stops at fp16 `TopK` |

The target decoder has advanced from a load-time stop with 90 rejected nodes
(20 `ConstantOfShape`, 70 `Gelu`) to complete CUDA placement: **+90 newly
claimed nodes and zero placement rejects**. It now reaches execution node 701.
The first end-to-end blocker is a generic query-only/shared-cache
`com.microsoft::GroupQueryAttention` shape that supplies zero-length current
K/V.

## Environment and method

- GPU: NVIDIA H200, 143,771 MiB; driver 580.105.08.
- Host: Linux x86-64.
- Rust/Cargo: 1.97.0.
- NVRTC: 13.3, loaded dynamically through the supplied conda NVIDIA library
  path.
- Target model SHA-256:
  `6a2ab727c2b491b737d15a1bacfc077f4afd10b8a41ba79f2f063f633b82775e`.
- Assistant model SHA-256:
  `74b49e0af30e65789206bed92baa42d0a6f181979d6a78c99e40746361e63920`.

Build:

```bash
cargo build --release -p onnx-genai-bench \
  --features bench-native,cuda --bin profile_native
```

All probes used:

```text
CUDA_VISIBLE_DEVICES=0
ONNX_GENAI_CUDA_GRAPH=1
ONNX_GENAI_DEVICE_KV=1
ONNX_GENAI_REQUIRE_CUDA=1
```

Target command:

```bash
./target/release/profile_native \
  --model /home/justinchu/gemma4-e2b-onnx \
  --tokens 1 --warmups 0 --runs 1 \
  --prompt Hello --ep cuda
```

`Hello` tokenized to `[9259]`. There was no CUDA-placement fallback warning or
strict-placement failure, proving that the CUDA EP claimed the complete
optimized graph. The raw ONNX graph has 1,194 nodes; the existing EP/session
passes produce the same 1,299 executable-node count reported by the preceding
probe.

## First target-decoder execution gap

The first failure is:

```text
executor node 701
model/layers.15/self_attn/GroupQueryAttention_node_846
com.microsoft::GroupQueryAttention
cuda_ep GroupQueryAttention: incompatible query/key/value batch, sequence,
or hidden dimensions
```

Node attributes:

```text
num_heads=8
kv_num_heads=1
scale=1.0
do_rotary=1
rotary_interleaved=0
local_window_size=512
```

Concrete inputs:

| Input | Dtype | Shape |
|---|---|---|
| query | fp16 | `[1, 1, 2048]` |
| key | fp16 | **`[1, 0, 256]`** |
| value | fp16 | **`[1, 0, 256]`** |
| past key | fp16 | `[1, 1, 4096, 256]` |
| past value | fp16 | `[1, 1, 4096, 256]` |
| `seqlens_k` | int32 | `[1]` |
| `total_sequence_length` | int32 | scalar `[]` |
| cosine cache | fp16 | `[131072, 128]` |
| sine cache | fp16 | `[131072, 128]` |

Outputs are fp16 attention `[1,1,2048]` and key/value caches
`[1,1,4096,256]`. The dimensions imply head size 256 for both Q and KV, so the
failure is specifically the current kernel's unconditional rejection of
`k_seq == 0`, followed by its `do_rotary` requirement that Q and K sequence
lengths be equal. This graph is using an existing shared KV cache and has no
current K/V token to append.

**Generic fix suggestion:** support unpacked GQA with `K.shape[1] ==
V.shape[1] == 0` when valid past K/V are present: derive head size from Q and
the cache, rotate Q only, skip K/V append, preserve/alias past to present, and
attend over the `seqlens_k`-bounded cache; add the same shape contract to the
claim gate and GPU parity tests.

Gemma therefore produced no token and has no throughput result yet.

## Assistant-sidecar placement check

The assistant graph confirms that the newly landed `ConstantOfShape`, `Gelu`,
and `ScatterElements` coverage is active. Its strict placement report has only
one rejection:

```text
graph/node#109 "masked_embedding/TopK_node_133"
ai.onnx::TopK: input 0 ('X') dtype Float16 unsupported; expected Float32
```

Concrete `TopK` signature after shape inference:

```text
X:       fp16 [batch, q_len, 2048]
K:       int64 [1]                 (K=32)
values:  fp16 [batch, q_len, 32]
indices: int64 [batch, q_len, 32]
axis=-1, largest=1, sorted=0
```

The downstream `ScatterElements` is now claimed with fp16 data/updates and
int64 indices:

```text
data:    fp16 [batch, q_len, 262144]
indices: int64 [batch, q_len, 4096]
updates: fp16 [batch, q_len, 4096]
```

Thus fp16 `TopK` remains a secondary blocker for the speculative assistant,
while zero-length-K/V GQA is the first blocker for ordinary target generation.

## Qwen2.5-0.5B H200 confirmation

Fixture:
`/home/justinchu/ana-bench/qwen-oga-cuda-graph-a4`
(`model.onnx` SHA-256
`0ba0908e0ce8e39fcb18462787f572bfa7ca840f98c206c43919b3bec4e83eea`).

```bash
./target/release/profile_native \
  --model /home/justinchu/ana-bench/qwen-oga-cuda-graph-a4 \
  --tokens 256 --warmups 2 --runs 1 \
  --prompt Hello --ep cuda
```

Result:

```text
804.73 tok/s, 1.243 ms/token
256 generated tokens in 318.120 ms
CUDA graphs: 3 captures, 762 replays, 0 fallbacks
```

The continuation begins, “I am a beginner in Python and I am trying to create
a simple program...”, matching the earlier coherent reference. This short
shared-H200 run confirms and slightly exceeds the prior 771.40 tok/s result;
the difference should be treated as run-to-run/shared-host variance.
