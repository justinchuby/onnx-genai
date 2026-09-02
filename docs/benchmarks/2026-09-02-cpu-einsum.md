# Native CPU Einsum — execution and performance

## Conditions

- Commit under test: `4ff8ba7c5` plus this change.
- Host: Intel Xeon Platinum 8480C, 2 sockets, 48 cores/socket, AVX2/FMA,
  AVX-512, AVX-512 BF16, no SMT.
- Process: pinned to logical CPUs `80-87`;
  `ONNX_GENAI_CPU_DECODE_THREADS=8`.
- Harness: Criterion, 10 samples/arm, 1 s warmup and 2 s measurement per arm.
  Kernel construction/planning occurs outside the timed loop.
- Repetitions: three clean interleaved optimized/reference sweeps (reps 2, 3,
  and replacement 4). Rep 1 was discarded because `foreign_pct=10.5` exceeded
  the 5% gate. Clean reps reported `foreign_pct=0.0, 0.0, 2.2` and
  `sibling_peak_pct=0.0`.
- Host coordination: `scripts/hostlock.sh run --owner resch`, default
  box-scoped lock directory, held across each complete sweep.
- Oracle: the same canonical `EinsumPlan` evaluated by an independent generic
  f64 accumulator, then narrowed to the requested dtype. The real ORT plugin
  conformance test additionally assigns and executes `ik,kj->ij` on `cpu_ep`
  with CPU fallback disabled.

`reference_f64 / optimized` compares the optimized lowering with the generic
canonical evaluator. It is not a native-before comparison: native CPU did not
previously implement Einsum. ORT is a correctness and reachability control, not
a claimed performance baseline.

## Steady-state results

Times are Criterion point estimates in microseconds. Ranges are the three clean
process repetitions. Allocation counts/bytes bracket one warmed kernel
execution with a counting `System` allocator; they include allocations made by
the reused MatMul implementation. “Workspace” is reusable f32 storage retained
by the Einsum kernel, not all transient storage below it.

| class / equation | shape, dtype, layout | optimized median (range) | reference / optimized | setup median | workspace | allocations / bytes |
|---|---|---:|---:|---:|---:|---:|
| small GEMM `ik,kj->ij` | `4x16 · 16x4`, f32, contiguous | 1.356 (1.347–1.417) | 3.05x | 30.8 us | 0 | 30 / 17,928 |
| GEMM `ik,kj->ij` | `32x256 · 256x256`, f32, contiguous | 58.639 (58.568–58.803) | 365.64x | 21.6 us | 0 | 45 / 300,168 |
| large GEMM `ik,kj->ij` | `64x512 · 512x256`, f32, contiguous | 99.125 (98.710–100.060) | 897.44x | 21.1 us | 0 | 45 / 398,472 |
| transpose-required `ik,jk->ij` | `32x256 · 128x256`, f32, strided B view | 175.330 (174.170–179.350) | 64.11x | 19.1 us | 0 | 39 / 300,184 |
| broadcast BMM `...mk,...kn->...mn` | `[4,16,128] · [1,128,64]`, f32 | 192.340 (188.760–193.790) | 32.56x | 20.8 us | 0 | 62 / 300,392 |
| GEMM `ik,kj->ij` | `32x256 · 256x128`, f16 | 29.321 (28.541–29.483) | 386.60x | 19.8 us | 0 | 45 / 296,072 |
| GEMM `ik,kj->ij` | `32x256 · 256x128`, bf16 | 90.458 (89.465–91.270) | 126.02x | 20.1 us | 0 | 30 / 83,080 |
| reduction `ij->i` | `[512,512]`, f32, contiguous suffix reduction | 36.314 (35.924–36.600) | 33.14x | 16.5 us | 2 KiB | 12 / 952 |
| elementwise `ij,ij->ij` | two `[512,512]` f32 tensors | 198.740 (195.400–200.230) | 13.76x | 16.4 us | 1 MiB | 16 / 1,208 |
| flattened GEMM + output permutation `abxy,xycd->dcab` | `[4,4,8,8] · [8,8,4,4]`, f32 | 18.687 (18.620–19.921) | 16.66x | 22.9 us | 1 KiB | 41 / 27,792 |
| diagonal copy fallback `ii->i` | `[1024,1024]`, f32 | 3.565 (3.562–3.700) | 1.01x | 13.2 us | 0 | 9 / 4,440 |
| zero-copy permutation metadata `abc->bca` | `[32,64,128]`, f32 strided output view | 0.238 (0.237–0.238) | n/a | 14.9 us | 0 | 8 / 480 |

The unaffected MatMul control (`32x256 · 256x256`, f32) was 56.452 us
(55.410–57.717 us), a 4.1% full range. Aggregate clean-sweep wall time was
93.25–93.84 s; aggregate process CPU was 239.60–242.08 s
(`process_efficiency=2.56–2.58` cores). The low whole-sweep efficiency is
expected because it includes Criterion sleeps and the intentionally serial f64
reference arms; it is not a kernel utilization claim.

Setup is `EinsumFactory::create` only, measured separately with three locked
repetitions. The first table row is also the process's cold first factory call
(29.7–35.2 us); subsequent plan builds are 12.2–23.6 us.

## What the measurements selected

- Binary contractions use the existing MatMul implementation rather than a new
  GEMM. This is the dominant measured win and preserves the mature SIMD,
  half/bfloat widening, batching, and prepack paths.
- Contiguous single-input suffix reductions shard independent output rows and
  retain a deterministic serial accumulation order within each row.
- Aligned elementwise products use a dense loop, parallel only above 64 Ki
  elements. Outer products and broadcasted/general reduction mappings retain
  the canonical generic loop.
- View-only permutation and diagonal extraction use `view_outputs`; the copy
  timings above describe the plugin/direct-call fallback, while a native
  executor pays only the metadata path.
- Layout normalization remains zero-copy when each canonical GEMM axis group
  can be represented by a collapsed stride. Otherwise the kernel materializes
  bounded f32 operands/result and applies the requested output permutation.

## Coverage ceiling and remaining opportunities

The repository model census currently contains zero real Einsum nodes, so these
claims are limited to the canonical classes and synthetic shapes above. The
kernel intentionally declines N-way coupled contractions, mixed
operand-local-reduction contractions, and reduced ellipsis contractions.

The largest remaining CPU opportunity is allocation removal: direct f32 GEMM
inherits MatMul’s panel allocations, while axis/layout vectors add small
per-dispatch allocations. Precomputing shape-specialized layout descriptors and
adding reusable MatMul panel/output workspace could reduce call counts, but the
measured large GEMM rows are already dominated by the mature SIMD kernel.
Diagonal direct calls show no arithmetic optimization opportunity; native
executor view propagation is the correct path.
