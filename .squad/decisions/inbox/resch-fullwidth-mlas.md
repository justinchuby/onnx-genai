# Resch: full-width MLAS QNBit decode experiment

Date: 2026-07-29
Branch: qwen3-perf-followups

## Change

Deckard replaced MLAS's standalone parallel-for with the persistent work-stealing pool, so I tested the ORT-style MatMulNBits path: one full-width `MlasQNBitGemmBatch`/`sqnbit_gemm_with_workspace(..., multithread=true)` per node, with MLAS partitioning N internally instead of our static SPMD shard fan-out.

I wired this path through the existing `ONNX_GENAI_CPU_MM_MLAS_NO_SHARD=1` escape hatch and removed the extra local Rayon wrapper around full-width MLAS QNBit calls. This avoids nesting our decode pool around MLAS's own intra-op pool. The fixed static-SPMD shard path remains the default because the full-width path did not win.

## Correctness and validation

Passed:

- `cargo check -p onnx-runtime-ep-cpu --features mlas`
- `cargo check -p onnx-runtime-ep-cpu --no-default-features --features ops-quantized`
- `cargo test -p onnx-runtime-ep-cpu --features mlas matmulnbits_arm64_mlas_qnbit_reaches_qwen_decode_bits4_and_bits8 -- --nocapture --test-threads=1`
- `cargo test -p onnx-runtime-ep-cpu --features mlas matmulnbits -- --nocapture --test-threads=1`
- `cargo test -p mlas-sys qnbit_multithread_uses_work_stealing -- --nocapture`

## Profiling

`ONNX_GENAI_PROFILE_OPS=1`, qwen3-0.6b, native CPU:

| Path | MatMulNBits best | MatMulNBits p90 | MatMulNBits median | Notes |
| --- | ---: | ---: | ---: | --- |
| Static SPMD default | 6.912 ms | 12.951 ms | 7.451 ms | 380 decode samples; host was contended, with slow outliers |
| Full-width MLAS opt-in (`ONNX_GENAI_CPU_MM_MLAS_NO_SHARD=1`) | 7.505 ms | 8.599 ms | 8.072 ms | 287 decode samples; profile run later failed deterministic-run check under profiling |

Full-width did not move MatMulNBits toward ORT's ~6 ms bucket on this host; it was slower than the fixed static-shard path in the measured bucket.

## Throughput benchmark

`profile_native --steady --runs 15 --tokens 96`, qwen3-0.6b, contended host:

| Config | Best | p90 | Median | Notes |
| --- | ---: | ---: | ---: | --- |
| Native static SPMD default | 105.99 tok/s | 105.87 tok/s | 102.91 tok/s | 15-run benchmark completed; two contention outliers at 65.78 and 59.29 tok/s |
| ORT CPU | 108.61 tok/s | 108.60 tok/s | 106.80 tok/s | 15-run benchmark completed |
| Native full-width MLAS opt-in | 93.16 tok/s | 93.16 tok/s | 92.92 tok/s | 4-run probe; repeated 5/15-run attempts hung before producing steady runs, so I did not promote it |

## Verdict

Do **not** make full-width MLAS QNBit the default. With Deckard's MLAS work-stealing backend it avoids the old Rayon wrapper, but real decode still regressed versus our static-SPMD shard path and longer full-width benchmark attempts were not reliable enough to pass the honest gate. Keep the ORT-style full-width path as an opt-in diagnostic via `ONNX_GENAI_CPU_MM_MLAS_NO_SHARD=1`; default remains fixed static-SPMD shards.
