# CPU kernel microbenchmarks

This directory measures the pure-Rust CPU execution-provider kernels without
session/model-loading overhead. The standing quality bar for kernel rewrites is:

1. numerical regression tests remain green; and
2. the Rust kernel is not slower than ONNX Runtime's CPU EP at the same shape
   and dtype, while retaining the Rust EP's broader portable dtype coverage.

## Run Criterion

From the repository root:

### Focused kernel target (excludes governed Einsum)

Name the bench target explicitly while iterating. A package-wide
`cargo bench -p onnx-runtime-ep-cpu` also selects the governed Einsum target,
so it is not a lightweight shortcut.

```bash
export ONNX_GENAI_CPU_DECODE_THREADS=16
export CARGO_TARGET_DIR="$PWD/target-cpu-kernel-criterion"
scripts/hostlock.sh run --owner cpu-bench \
  --reason "CPU EP focused Criterion target" --wait --gate 3 --strict-reap -- \
  cargo bench -p onnx-runtime-ep-cpu --bench kernels -- matmul/medium
```

Use a physical-core budget the host can realize as one logical CPU per core.
The lock remains mandatory for focused timing because even one multithreaded
kernel benchmark can contaminate another agent's measurement.

### Full package sweep (includes governed Einsum)

Use a fresh, absolute target directory. This one lock covers compilation,
warmup, and every benchmark target in the package:

```bash
export ONNX_GENAI_CPU_DECODE_THREADS=16
export CARGO_TARGET_DIR="$PWD/target-cpu-ep-suite-$(git rev-parse --short=12 HEAD)"
scripts/hostlock.sh run --owner cpu-bench \
  --reason "CPU EP full benchmark suite" --wait --gate 3 --strict-reap -- \
  cargo bench -p onnx-runtime-ep-cpu
```

Do not move the lock inside a per-target loop: the suite is one evidence sweep,
and releasing between targets makes its rows incomparable.

### Exact Einsum census and evidence run

Set the physical-core budget and a fresh absolute Cargo target directory from
the repository root. The census helper uses the same values and must report
12/12 before the evidence run:

```console
export ONNX_GENAI_CPU_DECODE_THREADS=16
EINSUM_TARGET="$PWD/target-einsum-evidence-$(git rev-parse --short=12 HEAD)"
export CARGO_TARGET_DIR="$EINSUM_TARGET"
crates/onnx-runtime-ep-cpu/benches/run_einsum.sh \
  census cpu-bench 16 "$EINSUM_TARGET"
```

The governed target has exactly one committed direct invocation. Keep the
outer lock around the Cargo child so compilation, warmup, all 26 selectors,
and both controls remain in one custody interval:

```bash
scripts/hostlock.sh run --owner cpu-bench \
  --reason "CPU Einsum evidence sweep (26 selectors)" \
  --wait --gate 3 --strict-reap -- \
  cargo bench -p onnx-runtime-ep-cpu --bench einsum -- --noplot
```

The census proves that the target lists exactly 12 Criterion selectors before
the timed sweep. The benchmark refuses
before measuring when the outer lock, gate, ownership, box scope, worktree
provenance, physical-core affinity, CPU model/frequency, or raw Criterion
destination cannot be verified. It also fails the run if any measurement window
has foreign/sibling contention, more than 20% median-frequency drift, or an
unstable MatMul control.

The runbook's `run` mode remains an equivalent convenience for local use, but
the direct command above is the governed invocation pinned by conformance.
Both paths hold one outer `hostlock.sh run` for the complete timed sweep,
including Cargo compilation. Reading the lock from the benchmark cannot
substitute for custody.

The target emits:

- exact commit/tree/branch, host-lock provenance, CPU model, realized
  logical-to-package/core mapping, and frequency samples;
- setup/planning time for optimized, forced GenericNative, and oracle modes;
  warmed allocation counts/bytes, reusable workspace, shared-input hashes,
  nonzero oracle counts, numeric error, and the native route that actually
  fired;
- three independent absolute repetitions per synthetic case, plus six
  deterministic ABBA/BAAB repetitions per arm for the equivalent
  `ik,kj->ij` Einsum/MatMul comparison and the MatMul A/A null. Criterion
  records optimized and forced GenericNative arms for every bilinear,
  trilinear, and N-ary case;
- per-window wall/CPU efficiency, `foreign_pct`, `sibling_peak_pct`, frequency,
  and two-ended host-lock attribution;
- Criterion's ten-sample raw JSON under
  `target-einsum-evidence/criterion/{einsum,einsum_view,einsum_control}`.

The f64 generic evaluator is validation-only and never appears as a timed arm.
Only the contiguous f32 `gemm_friendly` row is compared with MatMul, using the
same input buffers, shape, dtype, layout, output, and underlying native MatMul
kernel.

Criterion reports the estimated time interval and change versus the prior local
baseline. HTML reports are written under
`"$CARGO_TARGET_DIR/criterion/report/index.html"`. Compare the central time
estimate, not a single sample, and keep CPU governor, build flags, and machine
fixed. MatMul is run in dedicated Rayon pools pinned to 1 and 8 workers; its
benchmark IDs report `threads=1` or `threads=8`. Add, ReduceMean, and Gather do
not use Rayon internally and their IDs explicitly report
`threads=1-internal`. Benchmark IDs also encode the operation, size class,
dtype, and element or matrix dimensions.

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
scripts/hostlock.sh run --owner cpu-bench \
  --reason "ORT CPU kernel baseline" --wait --gate 3 --strict-reap -- \
  python crates/onnx-runtime-ep-cpu/benches/ort_baseline.py --threads 1 8
```

The script builds one-op ONNX models and times the same f32 operations and
shapes after warmup, excluding session construction:

```bash
scripts/hostlock.sh run --owner cpu-bench \
  --reason "ORT CPU MatMul baseline" --wait --gate 3 --strict-reap -- \
  python crates/onnx-runtime-ep-cpu/benches/ort_baseline.py \
    --filter matmul/medium --threads 1 8 --warmup 20 \
    --iterations 1000 --repetitions 9
```

Run it under the same host-lock and admission-gate conditions as Criterion.
Compare matching f32 rows and matching thread counts in microseconds. The
script pins and prints `intra_op_num_threads` for every result and fixes
`inter_op_num_threads=1` because each generated graph contains one node. ORT
support and optimization behavior for f16/bf16 on CPU varies by release, so
f32 is the required common baseline; the Rust-only f16/bf16 rows guard the
broader dtype surface. `--repetitions` reports the median of independently
timed iteration batches, which is preferable to a single elapsed-time sample
for recorded comparisons.

## Thread-matched MatMul comparison

For the medium f32 shape (`32×512×512`), Gaff's warm-cache measurements with
allocation outside the timed loop and ORT 1.27.0 were:

| Workers | Rust MatMul | ORT CPU EP | Rust / ORT |
|---:|---:|---:|---:|
| 1 | 2.801 ms | 131 µs | 21.4× |
| 8 | 502 µs | 30.6 µs | 16.4× |

These are matched comparisons, not default-pool results: Rust uses a dedicated
Rayon pool with the stated worker count, while ORT uses the same intra-op count
and one inter-op thread. The current gap is therefore approximately 16–21×,
depending on the matched thread count. The standing bar remains no slower than
ORT at matching shape, dtype, and thread count while preserving the Rust EP's
broader dtype coverage. Porting MLAS GEMM/MatMul is the recommended next step.

## Numeric regressions

Fixed golden vectors for every benchmarked kernel/dtype live in
`tests/kernel_numeric_regression.rs`:

```bash
export CARGO_TARGET_DIR="$PWD/target-cpu-kernel-tests"
cargo test -p onnx-runtime-ep-cpu --test kernel_numeric_regression
```
