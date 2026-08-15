# MatMulNBits N-dimension threading (2026-07-16)

## Strategy

`MatMulNBits` gives each Rayon task a contiguous output-column range. The
int8/VNNI decode path and fp32 `[N,K]` GEMV path share a thread-count-aware
partition policy; no private pool is created.

The old fixed 1 Mi-dot gate left 48 of Qwen2.5-0.5B's 121 decode matmuls
serial and regressed the 24-worker result. The replacement policy uses both
work and the active Rayon pool size:

- Up to 48 workers, parallel work needs 8 Ki dot terms per worker. Tasks
  target at least 32 Ki dot terms and are capped at two tasks per worker.
- Above 48 workers, parallel work needs 64 Ki dot terms per worker and is
  capped at one task per worker. This avoids repeatedly sending medium
  projections across the dual-socket pool; only sufficiently large GEMVs use
  all workers.
- Tasks contain at least 16 outputs. One-worker and tiny products stay serial.

For this model, all 121 matmuls thread at 24 and 48 workers (versus 73 with the
old gate). At 96 workers only the 151,936 x 896 language-head GEMV threads;
parallelizing the 120 smaller projections across both NUMA nodes was slower.

## Qwen2.5-0.5B INT4 decode

- Host: 96 physical cores, 2 NUMA nodes
- Model: `/home/justinchu/qwen2.5-0.5b-int4-onnx`
- Baseline: `origin/main` (`3219673`)
- Command: `profile_native --tokens 4 --warmups 3 --runs 7 --ep cpu`
- Throughput: median of three interleaved baseline/branch invocations
- Op profile: median decode-step value with `ONNX_GENAI_PROFILE_OPS=1`

| Rayon workers | threaded nodes | baseline tok/s | branch tok/s | gain | baseline MatMulNBits ms (% node time) | branch MatMulNBits ms (% node time) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 / 121 | 11.37 | 11.50 | +1.1% | - | - |
| 24 | 121 / 121 | 29.59 | **39.81** | **+34.5%** | 21.330 (76.41%) | **17.863 (73.20%)** |
| 48 | 121 / 121 | 20.31 | **25.31** | **+24.6%** | 41.611 (83.37%) | **31.964 (80.47%)** |
| 96 | 1 / 121 | 11.90 | **14.99** | **+26.0%** | 71.144 (87.75%) | **59.707 (90.16%)** |

Greedy token IDs remained `[11576, 42740, 11, 358]`. The higher 96-worker
MatMulNBits percentage is expected: other node time also fell while absolute
MatMulNBits time improved by 16.1%.
