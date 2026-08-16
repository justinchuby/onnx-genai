### 2026-08-15: GQA decode aux overhead attribution and split-merge tiling
**By:** Deckard
**What:** Profiled the fixed GQA decode aux overhead after #999 and optimized the fp16/bf16 split-K merge epilogue by tiling each query row across four dimension blocks. The merge now launches `rows * 4` CTAs, precomputes split weights once per CTA in shared memory, and writes a disjoint head-dim slice per CTA.
**Why:** Nsight Systems showed the steady decode path already uses fused decode prep: standalone RoPE, append, prepare-metadata, and transposes are absent from the measured window. The fixed aux cost is dominated by the split-KV merge epilogue, not standalone RoPE/append/reshape. The original merge launched only one CTA per query head (32 CTAs/layer for glm), leaving H200 almost completely grid-starved. Splitting the head dimension raises the merge grid to 128 CTAs/layer while preserving the same per-output split order and fp32 softmax math.

#### Attribution before this change (glm-4-9b-int4, GPU2, graph ON, stacked on #999)

| KV length | core | split-K merge | fused prep | standalone RoPE | append | transpose/metadata |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 447.6 us | 280.8 us | 104.0 us | 0 us | 0 us | 0 us |
| 2048 | 1335.0 us | 280.9 us | 109.1 us | 0 us | 0 us | 0 us |

#### After this change

| KV length | core | split-K merge | fused prep | measured e2e |
|---:|---:|---:|---:|---:|
| 512 | 448.0 us | 223.4 us | 101.8 us | 5.088 ms/token, 196.56 tok/s |
| 2048 | 1335.1 us | 223.2 us | 107.6 us | 5.940 ms/token, 168.35 tok/s |

Aux cut: ~57 us/step from the merge epilogue at both KV512 and KV2048. Short base decode measured 207.57 tok/s with captures=3, replays=474, fallbacks=0.

#### Correctness / capture
- glm and qwen KV512/KV2048 generated token streams matched #999 exactly in measured before/after runs.
- CUDA graph capture stayed intact (`fallbacks=0`).
- `cargo test -p onnx-runtime-ep-cuda --features cuda,gpu-tests --test gqa_fp16_gpu --quiet` passed.
- `cargo test -p onnx-runtime-ep-cuda --features cuda,gpu-tests --test group_query_attention_gpu gqa_gpu_fp16_decode_split_k_long_context_is_deterministic_and_matches_baseline --quiet` passed.
- `cargo fmt --all -- --check` and `cargo clippy -p onnx-runtime-ep-cuda --features cuda --lib -- -D warnings` passed.
