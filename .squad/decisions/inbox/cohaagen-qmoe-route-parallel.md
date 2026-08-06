# Decision: parallelize QMoE decode router (`qmoe_route`) — ~1.9× Qwen3.6-35B-A3B decode

**Author:** Cohaagen (EP/runtime perf + numerics)
**Date:** 2026-08-06
**Branch:** squad/qmoe-route-parallel (PR against main)
**Artifact:** `/home/justinchu/qwen36-35b-a3b-qmoe-artifacts` (native sparse QMoE)
**Follows:** roofline finding `.squad/decisions/inbox/cohaagen-35b-qmoe-roofline.md`

## What
Rewrote the CUDA `qmoe_route` kernel (`crates/onnx-runtime-ep-cuda/src/kernels/qmoe.rs`) from a **row-parallel single-thread** selection into a **block-cooperative** one: one CUDA block routes one row (grid-strided over rows), the top-k selection is `k` rounds of a block-wide argmax tree reduction, and the row logits are staged once into shared memory. New `route_launch_config` launches one power-of-two block per row with dynamic shared memory (logits + picked-mask + reduction scratch). Removed the now-unused `route_value_is_better` device helper.

## Why
The roofline pass found `qmoe_route` was **65.3% of decode GPU time** — Grid=(1,1,1) Block=(256,1,1), and because it was *row*-parallel with decode rows=1, **only thread 0 ran**, doing ~2048 latency-bound global `logits[expert]` reads. ncu: 0.02% mem throughput, 0.04% SM throughput, 1.57% warp occupancy → the H200 was ~99.96% idle inside it. 40 layers × 382µs ≈ 15.3 ms/tok ≈ half the decode. The expert GEMVs (the actual "3B active") were only ~9%.

## Byte-exactness (mandatory, preserved)
- **Selection** produces bit-identical `selected_experts`: integer-key argmax with the same tie rule (higher `total_order_key`, ties → lower index) is order-independent, so the parallel reduction picks exactly what the serial scan did.
- **Routing weights** (fp32 softmax / normalize / separate-router-weight aggregation) are computed by a **single thread in the ORIGINAL sequential order**, now reading logits from shared memory instead of re-issuing global loads. The floating-point op sequence is unchanged → weights are bit-for-bit identical.
- Net: the kernel is byte-identical to the previous serial kernel, not merely within tolerance. Handles rows>1 (prefill) unchanged. General/DRY — no hardcoded 256/8; experts/k come from the op shape/attrs exactly as before.

## Verification
- **`cargo test -p onnx-runtime-ep-cuda --features cuda,gpu-tests --test qmoe_gpu`: 27/27 pass** on H200 (decode rows=1 AND prefill rows>1, empty/hot experts, separate router weights, 64-expert top-6, int1/2/4/8, f32/f16/bf16, capture-replay).
- **Oracle regression `qwen36_35b_a3b_qmoe_divergence`** (`--ignored`, real 35B QMoE+dense+fp32-oracle, CUDA): step 1 (autoregressive **token-119 == oracle 33803**) PASSES. Step 2 (teacher-forced argmax) reports 279 — **this is PRE-EXISTING on origin/main**: I ran the unmodified kernel and it produces the identical 279 (and identical step-1 pass). This both confirms the failure is environmental/pre-existing (not introduced here) AND independently confirms my kernel is byte-identical (same value before/after). Flagging the pre-existing teacher-forced lock as a separate issue, out of scope for this perf change.
- `cargo fmt --all --check`: clean. `cargo clippy -p onnx-runtime-ep-cuda --features cuda`: clean.

## Measured impact (H200 GPU0, `--steady --warmups 2 --runs 3 --tokens 128 --decode-skip 8`, greedy)
| | ms/tok (median) | tok/s |
|---|---|---|
| Before (origin/main) | 30.997 | 32.26 |
| After (this PR) | **16.142** | **61.95** |
| | **1.92× decode** | |
Generated text byte-identical to baseline. Runs after: 16.188 / 16.053 / 16.142 (<1% spread).

nsys (`--cuda-graph-trace=node`), `qmoe_route`:
- Before: **65.3%** of GPU time, 381,850 ns/call.
- After: **5.8%** of GPU time, **12,555 ns/call (~30× faster)**. Top kernels are now the real compute (`matmul_nbits_gemv` 21%, `qmoe_linear_f32/f16` 14.4%+11.1%, `linear_attention` 6.3%). Router is no longer the bottleneck.

## Follow-ups (next levers, per the roofline record)
1. The residual ~12.6µs/call is now the single-thread sequential softmax tail (256 expf on thread 0), kept serial for bit-exactness; a bit-exact parallel prefix could trim it but it is only ~0.5 ms/tok.
2. Bigger remaining lever: **complete/repair CUDA-graph capture** on the hybrid graph (shredded into eager seams + per-step alloc churn) — now the largest share of the post-fix host/launch gap.
