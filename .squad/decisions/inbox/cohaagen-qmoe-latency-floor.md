# Cohaagen memo: QMoE decode latency floor after #764

Date: 2026-08-11
Branch/worktree: `squad/moe-profile-nextlever` / `/home/justinchu/onnx-genai-wt-moeprof`
GPU: H200, `CUDA_VISIBLE_DEVICES=1`

## Result

No new code shipped. I tried an order-preserving ILP-2 unroll of `qmoe_linear_impl`'s K loop after #764. It passed the QMoE GPU tests, but missed the hard model-level gate and regressed `qmoe_linear_f32` in NCU, so I reverted it.

## Measurements

Post-#764 NCU explicit stall metrics:

- `qmoe_linear_f32`: ~27.6-28.0 us, SM throughput ~72%, DRAM throughput ~4%, no-eligible ~22-23%. Per-warp stalls: barrier ~37%, wait ~15%, not-selected ~14%, short-scoreboard ~8.5%, long-scoreboard ~8%.
- `qmoe_linear_f16`: ~24.6-25.3 us, SM throughput ~71-73%, DRAM throughput ~9%, no-eligible ~21%. Per-warp stalls: not-selected ~24%, long-scoreboard ~21%, wait ~18%, math-pipe ~9%, short-scoreboard ~8.5%, barrier ~4%.

ILP-2 experiment:

- Model steady decode, 5 runs, 128 tokens: median **11.546 ms/token** (`11.511, 11.546, 11.520, 11.592, 11.552`). This is inside the previously observed 11.38-11.56 ms/token host-noise band and below the required >3% wall-clock win.
- `qmoe_linear_f32` regressed from ~27.6 us to ~32.2 us; active warps dropped ~13.2 -> ~9.7, no-eligible rose ~23% -> ~31% (extra registers/ILP reduced occupancy more than it hid latency).
- `qmoe_linear_f16` was essentially unchanged (~24.6-25.0 us).

Conclusion: #764 already removed the easiest grid-stride/reuse-barrier symptom. The remaining QMoE decode path is a small-M GEMV latency/occupancy balance, not a simple ILP opportunity. ILP does not clear the noise floor.

## Is 35B QMoE decode at the int4 weight-read latency floor?

It is not near the *bandwidth* roofline. Approximate active expert bytes per token:

- Hidden=2048, expert intermediate=512, k=8, 40 MoE layers.
- Per expert weights: gate + up + down = `2048*512*2 + 512*2048 = 3.15M` int4 weights = **1.57 MB** packed.
- k=8 and 40 layers: **~503 MB/token** packed int4 weights. On H200 at ~4.8 TB/s, the pure packed-weight bandwidth floor is **~0.10 ms/token**.
- Including f32 scales and zero-points as currently loaded pushes the read stream closer to **~0.6-1.1 GB/token** depending on cache/reuse assumptions, still only **~0.13-0.24 ms/token** at H200 peak bandwidth.

Measured QMoE kernels are ~2.8 ms/token, far above bandwidth floor. The gap is latency/launch/low-reuse structure: M=1, many independent tiny expert/output reductions, per-output int4 decode and scale loads, and limited useful work per CTA. More scalar ILP in the current CTA shape did not help.

## Remaining structural levers (ranked)

1. **Fuse gate/up/down QMoE decode path** — highest ROI, high implementation/numerics risk. A fused per-route/per-expert kernel could keep gate/up intermediates in registers/shared memory, avoid the global `fc1_output`, `fc3_output`, `activated`, and `route_output` scratch round trips, and cut 3 linear launches + activation/combine launch overhead. Must preserve fp32 accumulator/order or prove the 33803 oracle margin survives.

2. **Int4 tensor-core / DP4A-style expert GEMV** — high ROI, highest numerics/engineering risk. Current path scalar-decodes int4 to fp32 multiply-add. A packed int4 dot path could raise work per memory transaction and use hardware dot throughput, but accumulation order and dequant scale semantics are hard to keep byte-exact.

3. **Persistent decode kernel / persistent QMoE worker** — medium ROI, high integration risk. Keep CTAs resident across token decode or across QMoE sub-ops to amortize launch/graph-node overhead and improve locality. Harder to integrate with the current ORT-style per-op executor and graph capture model.

4. **Scale/zero-point load reuse inside the current GEMV** — medium/low ROI, moderate risk. The int4 chunk path reloads scale per 8 weights even with block size 32. Reusing scale across four chunks may reduce global/L1 traffic, but it changes loop structure/register pressure and needs byte-exact validation.

Recommendation: stop micro-tuning `qmoe_linear_impl` for now. The next meaningful >3% model-level lever is a designed fused QMoE decode kernel, with 35B teacher-forced oracle as the primary acceptance gate.
