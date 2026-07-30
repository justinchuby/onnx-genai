# Resch — decode work-stealing pool prototype

**When:** 2026-07-29T17:55:00-07:00  
**Branch:** `qwen3-perf-followups` / PR #398

## What changed

Prototyped an opt-in persistent decode schedule,
`ONNX_GENAI_CPU_DECODE_SCHEDULE=steal`, inside `decode_spmd.rs`. It keeps the
resident SPMD workers and lightweight barrier, but decomposes output columns
into coarse dynamic tiles (`ONNX_GENAI_CPU_DECODE_STEAL_TILES_PER_WORKER`,
default 2) claimed by an atomic cursor instead of assigning one static shard to
each worker. MLAS SQNBit cached shards use the same dynamic tile list and remain
bit-identical to the full-width MLAS call.

The default remains the existing fixed SPMD schedule.

## Correctness

- `cargo test -p onnx-runtime-ep-cpu decode_spmd --features mlas` — 36 passed.
- `cargo test -p onnx-runtime-ep-cpu mlas_work_stealing_decode_matches_no_shard_full_width --features mlas` — passed.
- `cargo build --release -p onnx-genai-bench --features mlas,bench-ort --bin profile_native` — passed.
- Native fixed and work-stealing generated identical token IDs in the Qwen3 run.

## Benchmark

Model:
`C:\Users\justinchu\.cache\huggingface\hub\models--justinchuby--qwen3-0.6b-onnx-genai\snapshots\c2570eab4ae4ada4aa08b8a451d57d901dab8e83`

Command shape:
`profile_native --model <model> --ep cpu --steady --runs 12 --tokens 96`

| Config | Best tok/s | Median tok/s | Min tok/s | Spread (best-min) |
|---|---:|---:|---:|---:|
| Native fixed SPMD (`ONNX_GENAI_CPU_DECODE_SCHEDULE=fixed`) | 105.14 | 99.56 | 59.71 | 45.43 |
| Native work-stealing (`...=steal`, 2 tiles/worker) | 95.07 | 81.60 | 36.23 | 58.84 |
| ORT backend | 105.72 | 102.98 | 99.63 | 6.09 |

Extra probe: work-stealing with 1 tile/worker (dynamic queue but no extra
tiles) was still worse in a 6-run sample: best 100.43, median 86.34 tok/s.

## Decision

Do **not** make work-stealing the default. On this contended host the dynamic
tile path did not reduce tail latency and did not improve peak; it regressed
best-case and median relative to fixed SPMD, and it widened outlier spread.

The likely cost is per-tile MLAS SQNBit shard/call overhead plus the atomic
dynamic scheduler in the hot per-projection loop. ORT remains much more stable,
but this prototype did not reproduce Eigen-pool stability inside our current
per-op persistent SPMD design. Keep fixed SPMD for peak; the residual is likely
elsewhere (executor/GQA/core scheduling) or requires a broader whole-step task
graph rather than per-op dynamic tiling.
