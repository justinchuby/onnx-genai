### 2026-08-15: GQA decode flash-decoding split count retune
**By:** Deckard
**What:** Retuned the existing fp16/bf16 GQA decode split-K flash-decoding path from ~2 target waves to ~4 target waves, added `ONNX_GENAI_CUDA_GQA_SPLITS` as an A/B rollback knob, switched fp16 softmax exponentials to `exp2f(x * 1/ln2)`, and parallelized the fp16/bf16 split-merge output loop with 64 threads.
**Why:** Profiling showed attention is meaningful at glm decode lengths: ~20% of a KV512 step and ~38% of a KV2048 step. The current kernel was already split-KV flash-decoding, not serial, but the host split-fill target under-parallelized long-context decode on H200. Raising active splits cuts the fp16 attention core at KV2048 from ~2195 us/step to ~1315 us/step; the merge-parallelization keeps the larger-split merge overhead bounded.

#### Measurements (glm-4-9b-int4, GPU1, graph ON, `ONNX_GENAI_LMHEAD_CUBLASLT=1`)

| KV length | attention core before | attention core after | attention aux before | attention aux after | e2e after |
|---:|---:|---:|---:|---:|---:|
| 512 | ~663 us | ~442 us | ~396 us | ~446 us | 5.175 ms/token, 193.23 tok/s |
| 2048 | ~2195 us | ~1315 us | ~392 us | ~443 us | 6.035 ms/token, 165.70 tok/s |

Short-context base decode (`--tokens 160 --decode-skip 40 --runs 3`) measured 207.46 tok/s with captures=3, replays=474, fallbacks=0. Fresh ORT fair graph-off on the same GPU measured 197.07 tok/s; ORT graph-on still fails in this harness with `ort_value must contain a constructed tensor`, so the certified ~250 tok/s graph-on target remains the comparator.

#### Accuracy / gates
- Existing fp16 GQA oracle tests pass: `cargo test -p onnx-runtime-ep-cuda --features cuda,gpu-tests --test gqa_fp16_gpu --quiet` (2/2).
- Existing int4 f64 oracle still passes: `cargo test -p onnx-runtime-ep-cuda --features cuda,gpu-tests --test matmul_nbits_marlin_numerics --quiet` (7/7).
- glm generated streams differ from the old split count at long context; the first inspected divergence is a greedy near-tie class caused by split-K reduction order, not a large numerical blow-up. qwen KV512/KV2048 generated streams matched between old split count and the new default in the measured runs.
- CUDA graph capture remains intact in all measured native runs (`fallbacks=0`).

#### Notes
The fixed attention aux bucket is mostly the split merge plus fused decode prep. Increasing splits raises merge work, so the merge loop now distributes half2 output lanes across 64 threads. Future work should focus on reducing/combining the per-layer merge/prep launches; the core is improved but still latency/grid-limited rather than bandwidth-bound.
