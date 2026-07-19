# CPU kernel microbenchmarks

This directory measures the pure-Rust CPU execution-provider kernels without
session/model-loading overhead. The standing quality bar for kernel rewrites is:

1. numerical regression tests remain green; and
2. the Rust kernel is not slower than ONNX Runtime's CPU EP at the same shape
   and dtype, while retaining the Rust EP's broader portable dtype coverage.

## Run Criterion

From the repository root:

```bash
cargo bench -p onnx-runtime-ep-cpu
```

Use a filter while iterating, for example:

```bash
cargo bench -p onnx-runtime-ep-cpu -- matmul/medium
```

Criterion reports the estimated time interval and change versus the prior local
baseline. HTML reports are written under
`target/criterion/report/index.html`. Compare the central time estimate, not a
single sample, and keep CPU governor, thread count, build flags, and machine
fixed. Benchmark IDs encode the operation, size class, dtype, and element or
matrix dimensions.

Coverage:

| Kernel | Shapes | Dtypes |
|---|---|---|
| Add (row broadcast) | `[1024]`, `[256,1024]`, `[1024,4096]` | f32, f16, bf16 |
| ReduceMean (axis 1) | `[32,128]`, `[128,512]`, `[256,1024]` | f32 |
| Gather (embedding rows) | `[4096,128]×32`, `[16384,256]×128`, `[32768,512]×256` | f32, f16, bf16 |
| MatMul | `1×256×256`, `32×512×512`, `32×1024×1024` | f32, f16, bf16 |

`ReduceMean` is f32-only because that is the current kernel contract. The other
three benchmarks document f16/bf16 support as well as f32 performance.

## ONNX Runtime baseline

The shared Python venv did not contain `onnxruntime` when this harness was
created. Keep ORT optional by installing it only in a disposable/local Python
environment:

```bash
python -m pip install numpy onnx onnxruntime
python crates/onnx-runtime-ep-cpu/benches/ort_baseline.py
```

The script builds one-op ONNX models and times the same f32 operations and
shapes after warmup, excluding session construction:

```bash
python crates/onnx-runtime-ep-cpu/benches/ort_baseline.py \
  --filter matmul/medium --warmup 20 --iterations 200
```

Run it on the same otherwise-idle machine as Criterion. Compare matching f32
rows in microseconds. ORT support and optimization behavior for f16/bf16 on CPU
varies by release, so f32 is the required common baseline; the Rust-only
f16/bf16 rows guard the broader dtype surface.

## Numeric regressions

Fixed golden vectors for every benchmarked kernel/dtype live in
`tests/kernel_numeric_regression.rs`:

```bash
cargo test -p onnx-runtime-ep-cpu --test kernel_numeric_regression
```
