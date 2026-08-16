# Historical f32 GEMM backend comparison: MLAS, oneDNN, and SimdX86

Measured 2026-07-19 on an Intel Xeon Platinum 8480C (Sapphire Rapids). The host
supports AVX-512 F/DQ/BW/VL/VNNI/BF16/FP16 and AMX tile/int8/bf16. ORT MLAS and
oneDNN can therefore dispatch beyond AVX2; the built-in `SimdX86` backend is
AVX2/FMA-only.

> **Status:** The oneDNN backend discussed here was removed after this benchmark:
> it did not provide the desired multi-threaded parity and required a CMake and
> bindgen source build. `SimdX86` remains the default x86 fast path.

## Result

Times are medians in microseconds; lower is better. Ratios are backend latency
divided by MLAS latency.

| Shape (M×K×N) | Threads | MLAS µs | oneDNN µs | SimdX86 µs | oneDNN / MLAS | SimdX86 / MLAS |
|---|---:|---:|---:|---:|---:|---:|
| 1×256×256 | 1 | 7.228 | 9.453 | 26.248 | 1.31× | 3.63× |
| 32×512×512 | 1 | 124.237 | 135.355 | 267.008 | 1.09× | 2.15× |
| 32×1024×1024 | 1 | 706.882 | 611.416 | 1,184.602 | 0.86× | 1.68× |
| 1×256×256 | 8 | 8.585 | 41.785 | 17.725 | 4.87× | 2.06× |
| 32×512×512 | 8 | 28.374 | 106.599 | 130.884 | 3.76× | 4.61× |
| 32×1024×1024 | 8 | 117.096 | 213.523 | 320.174 | 1.82× | 2.73× |

At one thread, oneDNN is close to MLAS on the two projection shapes: 9% slower
at `32×512×512` and 14% faster at `32×1024×1024`. It is 31% slower on the
matrix-vector shape. At eight threads, it does **not** match MLAS: it is 1.82×
to 4.87× slower. SimdX86 remains 1.68× to 4.61× slower than MLAS, although its
lower parallel-call overhead beats oneDNN on the two smaller eight-thread
shapes.

For an "entirely match MLAS performance" requirement, oneDNN was not a parity
path on these inference-oriented shapes. It was removed; pursue the MLAS
AVX-512/AMX kernel/packing strategy (or an equivalent specialized
implementation) for matched multi-thread performance.

## Method

- Source revisions: repository commit
  `daab01a170b736a419e2b0c6c593d2aa40dd776a` on
  `bench/mlas-vs-onednn`; oneDNN v3.9.2, commit
  `fef486592e40c9e907e615e747118620b4611e04`.
- MLAS: `onnxruntime==1.27.0`, `CPUExecutionProvider`,
  `intra_op_num_threads={1,8}`, and `inter_op_num_threads=1`.
- oneDNN: `CpuBackend::auto_detect()` was verified by its feature-selection test
  to return `OneDnn`; the source build used the default OpenMP CPU runtime.
- SimdX86: default feature build; this host satisfies its AVX2/FMA runtime check.
- Affinity: one-thread runs used physical core 0; eight-thread runs used physical
  cores 0-7 on socket/NUMA node 0 (`taskset -c 0` or `taskset -c 0-7`).
- oneDNN OpenMP used `OMP_NUM_THREADS={1,8}`, `OMP_DYNAMIC=FALSE`,
  `OMP_PROC_BIND=close`, and `OMP_PLACES=cores`. SimdX86 ran in the harness's
  dedicated 1- or 8-worker Rayon pool.
- Rust Criterion runs used 2 seconds warmup, 5 seconds measurement, 50 samples,
  and the median point estimate from `estimates.json`. Inputs and outputs were
  allocated outside the timed loop.
- MLAS used 20 warmups followed by nine independently timed batches of 1,000
  `session.run` calls; the table reports the median batch per-call time. Session
  construction was excluded.

The MLAS baseline includes one-node ORT invocation overhead while the Rust
harness calls the kernel directly. This biases the comparison slightly against
MLAS, so it does not explain oneDNN's eight-thread deficit.

## Reproduction

```bash
RAYON_NUM_THREADS=1 taskset -c 0 cargo bench -p onnx-runtime-ep-cpu \
  --bench kernels -- 'matmul/.*/f32/threads=1' \
  --warm-up-time 2 --measurement-time 5 --sample-size 50
RAYON_NUM_THREADS=8 taskset -c 0-7 cargo bench -p onnx-runtime-ep-cpu \
  --bench kernels -- 'matmul/.*/f32/threads=8' \
  --warm-up-time 2 --measurement-time 5 --sample-size 50

taskset -c 0 python3 crates/onnx-runtime-ep-cpu/benches/ort_baseline.py \
  --filter matmul --threads 1 --warmup 20 --iterations 1000 --repetitions 9
taskset -c 0-7 python3 crates/onnx-runtime-ep-cpu/benches/ort_baseline.py \
  --filter matmul --threads 8 --warmup 20 --iterations 1000 --repetitions 9
```

`cargo bench` already uses Cargo's optimized bench profile; this Cargo version
rejects a redundant `--release` flag for the bench subcommand.

The removed oneDNN configuration is retained only as historical benchmark data;
it cannot be reproduced from the current source tree.

The Generic backend was not remeasured because the public harness has no
backend override and changing kernel selection was outside this measurement's
bench/docs-only scope. The previously recorded medium-shape context remains
2.801 ms (one thread) and 502 µs (eight threads).
