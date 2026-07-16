# MatMulNBits N-dimension threading (2026-07-16)

## Strategy

`MatMulNBits` now gives each Rayon task a contiguous range of output columns.
The int8/VNNI decode path and the fp32 `[N,K]` GEMV path use the same partition
policy. M=1 partitions N directly; larger-M int8 work partitions rows and also
partitions N when there are fewer rows than workers. The general larger-M fp32
path remains on the shared GEMM/optional oneDNN backend.

This follows MLAS `QNBitGemm`'s M/N range partitioning rather than parallelizing
the K reduction. oneDNN was not reused because this crate only exposes its
optional `dnnl_sgemm` f32 path, not an N-bit GEMM.

Tiny products remain serial below 1,048,576 dot-product terms. Parallel tasks
contain at least 16 outputs and otherwise use `ceil(N / rayon_threads)`, so the
existing Rayon pool configuration controls concurrency without a second pool.
A threshold sweep at 64 Ki, 1 Mi, and 4 Mi selected 1 Mi; 1 Mi gave the best
96-worker result while avoiding wake-up overhead on the small projections.

## Qwen2.5-0.5B INT4 decode

- Host: 96 physical cores, 2 NUMA nodes
- Model: `/home/justinchu/qwen2.5-0.5b-int4-onnx`
- Command: `profile_native --tokens 4 --warmups 2 --runs 3 --ep cpu`
- Rayon: `RAYON_NUM_THREADS=96`

| Version | tok/s | ms/step |
|---|---:|---:|
| `origin/main` (`edad526`) | 11.44 | 87.423 |
| N-chunked kernel | **12.49** | **80.078** |

The measured gain is 9.2%. Greedy token IDs remained
`[11576, 42740, 11, 358]`.

With `ONNX_GENAI_PROFILE_OPS=1`, steady-state `MatMulNBits` time decreased from
74.953 to 73.344 ms/step (2.1%). Its share was effectively unchanged
(87.85% to 88.07%) because it remains overwhelmingly dominant; profiling did
not substantiate a share reduction.

Thread-count sweeps showed this small model peaks around 8-24 workers
(approximately 30 tok/s) and loses efficiency across the second NUMA socket.
The kernel nevertheless honors all 96 configured Rayon workers for sufficiently
large N; NUMA-aware scheduling or fused projection dispatch is a follow-up.
