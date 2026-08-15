### 2026-08-15: GQA fp16 decode core CTA warp retune
**By:** Deckard
**What:** Retuned the fp16 split-KV GQA decode core from 4 warps/CTA (128 threads) to 8 warps/CTA (256 threads), keeping `MAX_SPLITS=16` and the existing split/merge scratch layout. `ONNX_GENAI_CUDA_GQA_WARPS=4` remains an A/B rollback knob; default is 8. This preserves the launch grid and avoids the short-context inactive-CTA overhead seen when raising `MAX_SPLITS` to 24/32, while increasing intra-CTA parallelism over each split's KV slice.

**Why:** Nsight Compute at KV2048 showed the core was grid/issue starved, not DRAM-bandwidth bound: the 4-warp kernel launched only 512 CTAs/layer (0.39 waves/SM), achieved ~23% occupancy, and had low issue-slot utilization. Increasing splits to 32 improved long-context e2e but doubled the fixed grid and hurt shorter contexts. The 8-warp CTA raises active warps without changing split count or merge cost.

#### Nsight Compute core kernel, glm-4-9b-int4, KV2048, graph ON, GPU2

| kernel | block | grid | duration | achieved occ | issue slots | mem throughput |
|---|---:|---:|---:|---:|---:|---:|
| 4 warps/CTA baseline | 128 threads | 512 | 33.6-34.7 us | 22.9-23.2% | 21.9-23.2% | 61.7-63.6 GB/s |
| 8 warps/CTA | 256 threads | 512 | 20.1-20.4 us | 44.7-45.0% | 40.2-42.0% | 105-106 GB/s |

#### End-to-end steady decode, same GPU/session

| KV length | baseline | 8-warps/CTA | delta |
|---:|---:|---:|---:|
| 512 | 5.072 ms/token, 197.17 tok/s | 4.730 ms/token, 211.43 tok/s | +7.2% tok/s |
| 2048 | 5.962 ms/token, 167.73 tok/s | 5.200 ms/token, 192.30 tok/s | +14.7% tok/s |

CUDA graph capture stayed intact in both final runs (`fallbacks=0`). Split-count sweeps were measured and rejected for default-on: `MAX_SPLITS=32` reached up to ~189 tok/s at KV2048 in one sweep, but the doubled fixed grid/inactive CTA overhead made KV512 noisy/regressive; `MAX_SPLITS=24` was worse at KV2048.

#### Validation
- `cargo test -p onnx-runtime-ep-cuda --features cuda,gpu-tests --test gqa_fp16_gpu --quiet` passed (2/2).
- `cargo test -p onnx-runtime-ep-cuda --features cuda,gpu-tests --test matmul_nbits_marlin_numerics --quiet` passed (8/8, f64 oracle).
- qwen2.5-14b smoke, graph ON: KV512 147.46 tok/s, KV2048 136.12 tok/s, `fallbacks=0`.
- `cargo fmt --check` passed. Full `cargo clippy -p onnx-runtime-ep-cuda --features cuda --all-targets -- -D warnings` still fails on unrelated pre-existing warnings in `matmul_nbits.rs`, `standard_attention.rs`, `optimizer.rs`, `gqa_shared_prefix_parity_gpu.rs`, and `index_share_gpu.rs`; not touched here.
