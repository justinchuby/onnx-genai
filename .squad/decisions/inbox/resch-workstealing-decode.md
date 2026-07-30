# Resch — Work-stealing decode pool integration

Timestamp: 2026-07-29T20:21:04-07:00
Branch: qwen3-perf-followups / PR #398

## Change

Integrated Deckard's `mlas_sys::WorkStealingThreadPool` into `decode_spmd.rs` behind `ONNX_GENAI_CPU_DECODE_SCHEDULE=steal` / `work-stealing`. In steal mode, the fixed SPMD worker set is not spawned; a single persistent work-stealing pool is created once and reused across tokens/nodes. `dispatch_output_rows` and `dispatch_output_rows_indexed` publish coarse output tiles directly through `WorkStealingThreadPool::parallel_for`, so MLAS QNBit still runs single-threaded per tile while unclaimed tiles can be stolen if a worker is delayed. Default tile count is one tile per worker: 2x/3x tiling made shards too narrow and regressed Qwen3 throughput.

## Result

Do not make work-stealing the default yet. With the current API (no pinning / no NUMA placement in the mlas-sys pool), it tightened the catastrophic slow tail compared with fixed SPMD but lowered peak and median. Fixed SPMD remains the default; work-stealing is an opt-in diagnostic path.

Benchmark: `profile_native --backend {native|ort} --ep cpu --steady --runs 15 --tokens 96`, qwen3-0.6b, Snapdragon/Oryon Windows ARM64 host under normal contention. Throughput percentiles are over the 15 steady runs.

| Config | Best tok/s | p90 tok/s | Median tok/s | Worst tok/s |
|---|---:|---:|---:|---:|
| native work-stealing (`ONNX_GENAI_CPU_DECODE_SCHEDULE=steal`) | 97.31 | 97.10 | 90.13 | 80.81 |
| native fixed SPMD (`ONNX_GENAI_CPU_DECODE_SCHEDULE=fixed`) | 105.98 | 105.15 | 99.38 | 40.59 |
| ORT | 108.91 | 107.29 | 99.78 | 89.46 |

Interpretation: work-stealing removed fixed-SPMD's worst outliers in this run (80.8 vs 40.6 tok/s), but its median was ~9% below fixed and ORT, and its best was ~8% below fixed. The likely missing piece is worker affinity/NUMA-local placement for the mlas-sys pool; fixed SPMD still benefits from the existing pinned worker layout and node-local first-touch.

## Validation

- `cargo check -p onnx-runtime-ep-cpu --features mlas`
- `cargo check -p onnx-runtime-ep-cpu --no-default-features --features ops-quantized`
- `cargo test -p onnx-runtime-ep-cpu --features mlas decode_spmd -- --nocapture --test-threads=1` (36 passed)
- `cargo test -p onnx-runtime-ep-cpu --features mlas matmulnbits -- --nocapture --test-threads=1` (43 passed, 2 ignored)

## Follow-up for Deckard

If we want another pass, `mlas_sys::WorkStealingThreadPool` likely needs optional worker pinning / affinity hooks (or a construction API accepting CPU ids) so it can preserve the same 6-8 Oryon worker placement and weight first-touch locality that fixed SPMD currently has.
