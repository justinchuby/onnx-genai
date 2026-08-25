# CPU matmul performance vs ONNX Runtime

Measured `MatMul` / `Gemm` / `MatMulNBits` / `QLinearMatMul` ratios against ONNX Runtime's own CPU
kernels.

## What this file is now

It used to be the evidence behind an assignment policy: ranges this EP measured slower than ORT
were declined at `GetCapability` so ORT's CPU EP ran them instead, and every row below was asserted
by a test in `crates/onnx-runtime-ep-cpu/src/assignment_policy.rs`.

**That policy is gone.** When this EP is selected it claims every node it supports and never hands
one to ORT's CPU EP. Splitting a graph across an EP boundary costs a partition, and it forfeits the
fusion, prepack and buffer reuse that only hold inside one partition — so a range where this EP is
slower is a kernel to optimize, not a node to give away.

So the ratios below are no longer thresholds. They are a **work list**: every row under 1.00 is an
open gap, and closing it is the only way it stops being one.

## Rule

None, at assignment time. Everything supported is claimed.

There is also no longer a mechanism to express anything else: `ClaimPreference`,
`ExecutionProvider::claim_preference{,_node}` and the `host_fallback_available`
plumbing were deleted from `onnx-runtime-ep-api` / `onnx-runtime-ep-plugin`, so
a future "just this one shape" deferral would have to reintroduce the machinery
first. `supports_op` — a *capability* answer — is all that remains.

The bar for calling a gap *closed* is unchanged and still deliberately strict: a
**>= 5% repeatable win beyond noise at every measured thread count**. Anything
inside the noise band is not a win, and a win at one thread count that inverts at
another is not a win either — `Sqrt` is the standing example, at 1.9x
single-threaded and 0.30x at sixteen threads.

## Method

Session-level interleaved A/B: the same `.onnx` model run through ORT's CPU EP and through this EP
in the same process, alternating, warmups discarded, p50 and p90 of whole-`Run` latency over 5-9
reps x 9-11 runs. Output parity against ORT is asserted on every rep
(`1e-4 + 1e-3 * |y|` for f32), and separately against a **float64** oracle where precision itself is
under test.

- Host: AMD EPYC 9V74, 32 vCPU / 16 cores, **AVX2 + FMA + F16C only** — no AVX-512, no VNNI, no AMX.
- ONNX Runtime **1.27.0**, CPU EP, `intra_op_num_threads` matched to ours on both sides.
- `K = N = 3584` (Qwen3-8B hidden size), `M = 1` decode and `M = 128 / 512` prefill.
- The host is **shared and contended**, so ratios and dispersion are quoted, never absolute
  milliseconds as a headline. Rows with `spread > 1.0` are marked noisy.

**All ratios below are `ours / ORT`, so lower is better and `> 1.0` means we are slower.**

## Matrix

| op | dtype / bits | M | K, N | threads | p50 | p90 | status |
|---|---|---|---|---|---|---|---|
| `MatMulNBits` | 8-bit, block 32 | 1 | 3584 | 1 | **0.15** | 0.15 | **win** |
| `MatMulNBits` | 8-bit, block 32 | 1 | 3584 | 2 | **0.36** | 0.36 | **win** |
| `MatMulNBits` | 8-bit, block 32 | 1 | 3584 | 4 | **0.25** | 0.32 | **win** |
| `MatMulNBits` | 8-bit, block 32 | 1 | 3584 | 8 | **0.25** | 0.30 | **win** |
| `MatMulNBits` | 8-bit, block 32 | 1 | 3584 | 16 | **0.23** | 0.23 | **win** |
| `MatMulNBits` | 8-bit, block 32 | 1 | 4096 | 8 | **0.23** | 0.24 | **win** |
| `MatMulNBits` | 8-bit, block 32 | 128 | 3584 | 2 | **0.66** | 0.66 † | **win** (was 2.06) |
| `MatMulNBits` | 8-bit, block 32 | 128 | 3584 | 4 | **0.75** | 0.84 † | **win** (was 2.01) |
| `MatMulNBits` | 8-bit, block 32 | 128 | 3584 | 8 | **0.68** | 0.88 † | **win** (was 2.32) |
| `MatMulNBits` | 8-bit, block 32 | 256 | 3584 | 2 | **0.77** | 0.79 † | **win** (was 1.70) |
| `MatMulNBits` | 8-bit, block 32 | 256 | 3584 | 4 | **0.83** | 0.99 † | **win** (was 1.69) |
| `MatMulNBits` | 8-bit, block 32 | 256 | 3584 | 8 | **0.88** | 1.11 † | **win** (was 1.89) |
| `MatMulNBits` | 8-bit, block 32 | 512 | 3584 | 2 | **0.86** | 0.87 † | **win** (was 1.41) |
| `MatMulNBits` | 8-bit, block 32 | 512 | 3584 | 4 | **0.88** | 1.00 † | **win** (was 1.42) |
| `MatMulNBits` | 8-bit, block 32 | 512 | 3584 | 8 | 1.01 | 1.13 † | parity (was 1.54) |

† **not a p90.** On the nine rows above, that column is the per-trial *maximum* — the worst of 7-11
interleaved trials — because `ab.py` reports `p50 [min-max]` rather than a p90. It is a strictly more
pessimistic statistic than the p90 the rest of this table carries, and it is marked rather than
converted so no cell here is ever compared to a p90 as though the two were the same number.

### The 8-bit prefill loss was a 51 MB f32 weight rebuilt on every call

**Fixed.** On a native (non-`mlas`) build there was no borrowed route for `bits == 8, m > 1` at all:
`try_prefill_mlas_nt` declines and the kernel materialized the whole `k x n` weight as f32 in the
transposed `Kn` layout -- 51.4 MB for 3584x3584, written at stride `n`, **per call**, because
nothing caches that layout. Generalizing #1117's fused-dequant GEBP to 8 bits removed it: each
packed byte is read once per call into the L1-resident panel every row reuses. Kernel level, the
same `3584` geometry gets **17.1x** at `m = 2`, **13.9x** at `m = 8`, **5.8x** at `m = 64`, **2.6x**
at `m = 256` and **1.8x** at `m = 512`, and the two arms' outputs are bit-identical.

Full record, including the ORT-comparison method and the controls:
[`docs/benchmarks/2026-08-19-int8-prefill-gebp.md`](../benchmarks/2026-08-19-int8-prefill-gebp.md).

Four things about the table above are worth stating plainly.

* **The `p90` column on the nine re-measured rows is the per-trial *maximum*** (marked `†` above).
  Where it crosses 1.00 (`M = 256, t = 4/8` and `M = 512, t = 4/8`) that is one trial out of 7-11,
  not the median behaviour.
* **The `was` column is a paired re-measurement, not the historical number.** Both arms are one
  build of the current tree, differing only by `ONNX_GENAI_CPU_MM_INT8_GEBP`, interleaved trial by
  trial (7-11 trials x 9 runs). The `M = 512` prior rows reproduce the numbers this file has always
  carried almost exactly (1.41 -> 1.41 at 2 threads, 1.39 -> 1.42 at 4); the `M = 128` and `M = 256`
  prior rows do **not** -- this file recorded 0.90/0.87/0.94 and 1.17/1.15/0.99 where the paired
  arm now reads 2.06/2.01/2.32 and 1.70/1.69/1.89. Those older rows come from a tree and build
  configuration that can no longer be reconstructed here, so they are replaced rather than defended.
  A `--features mlas` build of the *current* tree does not reproduce them either (it reads 0.56 at
  `M = 128, t = 2`), which rules out "the old rows were an MLAS build" as the explanation.
* **`M = 512` at 8 threads reaches parity, not a win.** 1.01 `[0.886-1.131]` over 11 trials. By this
  file's own bar -- a >= 5% repeatable win beyond noise at *every* measured thread count -- the
  512-row row is not closed. It is no longer the shape that motivates optimizing 8-bit first.
* **Decode is untouched and still wins.** `M = 1` reads 0.153 / 0.181 / 0.219 at 2 / 4 / 8 threads
  with the switch on, against 0.154 / 0.207 / 0.244 with it off: overlapping ranges, as it must be,
  since `m == 1` never reaches this branch.

For reference, a `--features mlas` research build of the same tree reads 0.56 / 0.50 at
`M = 128` (2 / 8 threads) and 0.82 / 0.76 at `M = 512`. The pure-native path is now within
1.05-1.35x of it, where before it was 2.5-3.7x behind.

### Block sizes ORT cannot build at all

Worth recording because it shows a handover was never universally available even when the policy
wanted one. This EP accepts any power-of-two `block_size >= 16`; ORT's CPU `MatMulNBits`
`ORT_ENFORCE`s `block_size` in {16, 32, 64, 128, 256}, and that check throws at *kernel
construction*. Giving a 512-wide block back would turn a working session into a load failure, not a
slow one. `bits` needs no equivalent note -- both runtimes accept exactly {2, 4, 8}.

### 2-bit was never measured

Only 4-bit and 8-bit `MatMulNBits` were measured. 2-bit is a valid contrib value that shares the
dequant-then-GEMM structure and the threadpool with 4-bit, so its behaviour is extrapolated rather
than measured, and no row below should be read as covering it.

### Rows are folded, not read from one dimension

The row count of `[.., M, K]` is the product of every dimension but the last, so a statically
batched `[4, 100, 3584]` is 400 rows and lands in the wide-prefill region even though no single
dimension reaches 256. Any symbolic dimension anywhere in the batch makes the whole count unknown,
which is the decode case and stays claimed.
### Measured ranges below the 8-bit `MatMulNBits` win

Same host, same harness, same convention: **p50/p90 are ours/ORT, lower is better**, so every number
below 1.00 is a win and every number above it is a loss. Every range here is
claimed and executed by this EP; the column records whether we currently beat ORT, not
whether we hand the node over.

| op | dtype / bits | M | K, N | threads | p50 | p90 | vs ORT |
|---|---|---|---|---|---|---|---|
| `MatMulNBits` | 4-bit, acc 0 | 1 | 3584 | 1 | 1.00 | 1.03 | gap |
| `MatMulNBits` | 4-bit, acc 0 | 1 | 3584 | 2 | 1.52 | 1.56 | gap |
| `MatMulNBits` | 4-bit, acc 0 | 1 | 3584 | 4 | 1.74 | 1.82 | gap |
| `MatMulNBits` | 4-bit, acc 0 | 1 | 3584 | 8 | 2.23 | 2.31 | gap |
| `MatMulNBits` | 4-bit, acc 0 | 1 | 3584 | 16 | 2.21 | 3.28 | gap |
| `MatMulNBits` | 4-bit, acc 0 | 1 | 3584 | 32 | 4.28 | 4.71 | gap |
| `MatMulNBits` | 4-bit, acc 0 | 128 | 3584 | 1 | 0.99 | 1.07 | gap |
| `MatMulNBits` | 4-bit, acc 0 | 128 | 3584 | 4 | 2.35 | 2.63 | gap |
| `MatMulNBits` | 4-bit, acc 0 | 128 | 3584 | 8 | 2.41 | 2.63 | gap |
| `MatMulNBits` | 4-bit, acc 0 | 128 | 3584 | 32 | 3.79 | 4.36 | gap |
| `MatMulNBits` | 4-bit, acc 4 | 1 | 3584 | 8 | 1.78 | 1.97 | gap |
| `MatMulNBits` | 4-bit, acc 4 | 128 | 3584 | 8 | 2.11 | 2.37 | gap |
| `MatMul` | f32 | 1 | 3584 | 1 | 1.00 | 1.00 | gap |
| `MatMul` | f32 | 1 | 3584 | 8 | 2.52 | 4.51 | gap (noisy) |
| `MatMul` | f32 | 1 | 3584 | 32 | 0.57 | 0.71 | gap |
| `MatMul` | f32 | 128 | 3584 | 1 | 0.97 | 1.03 | gap |
| `MatMul` | f32 | 128 | 3584 | 2 | 2.11 | 2.35 | gap |
| `MatMul` | f32 | 128 | 3584 | 4 | 1.77 | 1.79 | gap |
| `MatMul` | f32 | 128 | 3584 | 8 | 2.76 | 2.85 | gap (noisy) |
| `MatMul` | f32 | 128 | 3584 | 16 | 1.65 | 1.73 | gap |
| `MatMul` | f32 | 128 | 3584 | 32 | 0.67 | 0.88 | gap |
| `MatMul` | f16 | 1 | 3584 | 1 | 0.86 | 0.96 | **win** (see note) |
| `MatMul` | f16 | 1 | 3584 | 2 | 1.47 | 1.89 | gap |
| `MatMul` | f16 | 1 | 3584 | 4 | 2.06 | 2.19 | gap |
| `MatMul` | f16 | 1 | 3584 | 8 | 1.93 | 2.20 | gap |
| `MatMul` | f16 | 1 | 3584 | 16 | 2.01 | 2.79 | gap |
| `MatMul` | f16 | 1 | 3584 | 32 | 3.83 | 5.95 | gap (noisy) |
| `MatMul` | f16 | 128 | 3584 | 1 | 1.00 | 1.01 | gap |
| `MatMul` | f16 | 128 | 3584 | 2 | 1.68 | 1.69 | gap |
| `MatMul` | f16 | 128 | 3584 | 4 | 1.76 | 1.80 | gap |
| `MatMul` | f16 | 128 | 3584 | 8 | 1.72 | 1.77 | gap |
| `MatMul` | f16 | 128 | 3584 | 16 | 1.30 | 1.62 | gap |
| `MatMul` | f16 | 128 | 3584 | 32 | 0.91 | 1.46 | gap |
| `Gemm` | f16 | 1 | 3584 | 1 | 0.86 | 0.94 | **win** (see note) |
| `Gemm` | f16 | 1 | 3584 | 2 | 1.13 | 1.25 | gap |
| `Gemm` | f16 | 1 | 3584 | 4 | 0.98 | 1.85 | gap |
| `Gemm` | f16 | 1 | 3584 | 8 | 1.83 | 2.08 | gap |
| `Gemm` | f16 | 1 | 3584 | 16 | 2.05 | 2.68 | gap |
| `Gemm` | f16 | 1 | 3584 | 32 | 4.20 | 7.72 | gap (noisy) |
| `Gemm` | f16 | 128 | 3584 | 1 | 1.03 | 1.03 | gap |
| `Gemm` | f16 | 128 | 3584 | 2 | 1.82 | 1.83 | gap |
| `Gemm` | f16 | 128 | 3584 | 4 | 1.90 | 1.91 | gap |
| `Gemm` | f16 | 128 | 3584 | 8 | 1.86 | 2.24 | gap |
| `Gemm` | f16 | 128 | 3584 | 16 | 1.44 | 1.46 | gap |
| `Gemm` | f16 | 128 | 3584 | 32 | 1.19 | 1.35 | gap |
| `Gemm` | f16, **transB** | 1 | 3584 | 1 | **0.66** | 0.67 | **win** (was 22.5) |
| `Gemm` | f16, **transB** | 1 | 3584 | 2 | **0.63** | 0.73 | **win** (was 29.7) |
| `Gemm` | f16, **transB** | 1 | 3584 | 4 | **0.84** | 1.02 | **win** (was 36.9) |
| `Gemm` | f16, **transB** | 1 | 3584 | 8 | **0.76** | 0.93 | **win** (was 49.5) |
| `Gemm` | f16, **transB** | 1 | 3584 | 16 | 1.71 | 1.22 | gap (noisy, was 65.3) |
| `Gemm` | f16, **transB** | 128 | 3584 | 1 | 4.04 | 4.11 | **gap** |
| `Gemm` | f16, **transB** | 128 | 3584 | 8 | 16.95 | 10.53 | **gap** |
| `QLinearMatMul` | u8 x u8 | 1 | 3584 | 1 | 1.13 | 1.18 | gap |
| `QLinearMatMul` | u8 x u8 | 1 | 3584 | 8 | 2.33 | 2.77 | gap |
| `QLinearMatMul` | u8 x u8 | 1 | 3584 | 16 | 2.34 | 2.58 | gap |
| `QLinearMatMul` | u8 x u8 | 128 | 3584 | 1 | 1.20 | 1.21 | gap |
| `QLinearMatMul` | u8 x u8 | 128 | 3584 | 8 | 2.43 | 2.80 | gap |
| `QLinearMatMul` | u8 x u8 | 128 | 3584 | 16 | 2.65 | 3.42 | gap |
| `QLinearMatMul` | u8 x u8 | 512 | 3584 | 1 | 1.20 | 1.21 | gap |
| `QLinearMatMul` | u8 x u8 | 512 | 3584 | 8 | 2.12 | 2.22 | gap |
| `QLinearMatMul` | u8 x u8 | 512 | 3584 | 16 | 2.08 | 2.18 | gap |
| `QLinearMatMul` | i8 x i8 | 1 | 3584 | 1 | **0.03** | 0.03 | **win** |
| `QLinearMatMul` | i8 x i8 | 1 | 3584 | 8 | **0.09** | 0.09 | **win** |
| `QLinearMatMul` | i8 x i8 | 1 | 3584 | 16 | **0.10** | 0.10 | **win** |
| `QLinearMatMul` | i8 x i8 | 128 | 3584 | 1 | **0.25** | 0.25 | **win** |
| `QLinearMatMul` | i8 x i8 | 128 | 3584 | 8 | **0.47** | 0.53 | **win** |
| `QLinearMatMul` | i8 x i8 | 128 | 3584 | 16 | **0.60** | 0.69 | **win** |
| `QLinearMatMul` | i8 x i8 | 512 | 3584 | 1 | **0.26** | 0.26 | **win** |
| `QLinearMatMul` | i8 x i8 | 512 | 3584 | 8 | **0.42** | 0.43 | **win** |
| `QLinearMatMul` | i8 x i8 | 512 | 3584 | 16 | **0.42** | 0.47 | **win** |
`transB = 1` is the layout every `nn.Linear` export produces, so these rows are
not a corner case — they are what a QKV, an output projection and an MLP gate
look like when a model is exported through `Gemm` rather than `MatMul`. They
were absent from this table entirely until 2026-08-19; the `transB = 0` `Gemm`
rows above them never covered them.

**The `M = 1` ratios above are the least reproducible numbers in this file, and
the `t = 16` row should not be trusted to two digits.** The decode cells now run
in 0.3-1.5 ms, which is short enough that the host's other tenants move the
ratio more than the kernel does. Three independent sweeps of the same five cells,
at load 6-13, 12-18 and 9-11, gave p50 ratios of

| threads | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| sweep A (7 trials) | 0.59 | 0.59 | 1.22 | 1.83 | 1.46 |
| sweep B (7 trials) | 0.88 | 1.51 | 1.23 | 1.04 | 2.68 |
| sweep C (9 trials, tabulated) | 0.66 | 0.63 | 0.84 | 0.76 | 1.71 |

so the honest reading is: **a consistent win at 1-2 threads (0.59-0.88), and
somewhere between a win and a ~2x loss at 4-16 threads, unresolvable on this
host.** What is *not* in doubt is the absolute change, which is three to four
orders of magnitude larger than the noise: 32-48 ms before, 0.3-1.5 ms after.


> **The `QLinearMatMul` rows above are `--features mlas` measurements.** They were taken on the
> research build, and nothing on this page said so. The build we actually ship had no integer GEMM
> at all until #1194, and measured **11.9x at M=1 and 11.9x at M=128** rather than the 1.13-1.20x
> the rows claim. See [section 3](#3-per-call-packing-and-a-missing-native-kernel--qlinearmatmul-fixed)
> for the corrected native numbers; the rows themselves are left as measured, because they are
> accurate for the build they describe.
>
> **The 11.9x figure is itself stale**: it predates #1194's native integer GEMM and stood only
> because there was no `QLinearMatMul` generator to re-measure with. Re-measured on the default
> build in [section 16](#16-qlinearmatmul-decode-had-never-been-measured-on-the-shipped-build-and-the-integer-gemm-branched-on-signedness-per-16-bytes),
> it is **1.13x at u8 M=128**, **0.11x at i8 M=1 (a win)**, and a loss confined to u8 M=1.

Ranges **outside** the measured region — `K * N < 2^20`, symbolic/dynamic weight shapes, and dense
dtypes other than f32/f16 — are simply unmeasured. They are claimed like everything else; the note
is only that no row above characterises them.

> **The `MatMul` f32 `M = 1` rows above predate the decode GEMV becoming the default and were taken
> with a build that had it compiled in.** Measured through an ORT session on the *shipped* default
> build, the same range was **5.63** at one thread, not 1.00, because the M=1 GEMV was behind a
> default-off env toggle — see [section 4](#4-the-decode-f32-gemv-was-written-but-never-shipped-fixed),
> which fixes it and re-measures the range end to end.
>
> The `QLinearMatMul` rows are likewise measured on a **`--features mlas` research build**. In the
> default native build the integer path was the widened-`i32` scalar loop, which measured **11.8x
> (M=1) / 12.0x (M=128)** against ORT at `1x2048x2048`, one thread. Section 3's "fixed" applied to
> the MLAS route only. #1194 then landed a native QGEMM and
> [section 16](#16-qlinearmatmul-decode-had-never-been-measured-on-the-shipped-build-and-the-integer-gemm-branched-on-signedness-per-16-bytes)
> is the first measurement of it through an ORT session: **1.13x at u8 M=128** and **0.11x at i8
> M=1**, with the remaining loss confined to **u8 M=1**.

## Root causes

### 1. Parallel efficiency, not kernel quality — f32 dense and int4 `MatMulNBits`

At **one thread this EP is at parity** (f32 1.00 / 0.97, int4 1.00 / 0.99). The gap opens as threads
are added and closes again only at 32, where ORT's own scaling saturates. We realise roughly **half**
of ORT's parallel speedup.

#1054 removed one cause — the standalone MLAS pool was clamped to `min(available, 8)` workers and
never saw the EP's requested thread count, which cost 2.16x -> 1.61x at 32 vCPU. The residual
1.4-2.4x (ours/ORT, lower is better) at 2-16 threads is still open.

**It is not the pool.** Driving an MLAS GEMM directly through this crate's work-stealing pool,
`K = N = 3584`, `M = 512`, warmed, on the same host:

| pool threads | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| MLAS GEMM only | 99.9 ms | 60.3 ms | 26.9 ms | 14.3 ms | 13.6 ms | 11.0 ms |
| speedup vs 1 thread | 1.0x | 1.7x | 3.7x | **7.0x** | **7.4x** | 9.1x |

MLAS requests at least `pool_threads` partitions per call and they are dispatched. The instrument
below prints them: on this host `sched_per_call` comes out at exactly the pool width (8 at
`--threads 8`) with `serial_fallback=0`. The *committed assertion* in `mlas-sys` is only the
inequality `scheduled_iterations >= pool_threads`, so the inequality is what is proven and the
equality is what is observed. Either way the primitive and the pool together scale about as well as
ORT does. The pool is therefore **not** where the 1.4-2.4x goes. What is left is the per-call work
*around* the primitive, which does not shrink with threads.

**Read the scope of that evidence carefully.** The numbers above are measured on the **integer
`QLinearMatMul`** path (`qgemm_i32_packed`), which is the only kernel here with a committed
phase-splitting instrument. That is a *proxy*, and it licenses exactly one transferable conclusion:
this crate's work-stealing pool can drive an MLAS GEMM to 7x on this host, so the pool is not the
ceiling for f32 or int4 either — all three go through the same pool. It does **not** license the
specific phase names or milliseconds for f32/int4. `QLinearMatMul`'s serial phases (requantize, the
`i32` accumulator staging) **do not exist** in `matmul.rs` or `matmul_nbits.rs`; those kernels have
their own, different per-call work (dequant staging, activation densification, executor tensor
handling) which has **not** been decomposed. Doing that decomposition is open work.

What the instrument actually shows, on `QLinearMatMul`, `K = N = 3584`, `M = 512`:

| phase | 1 thread | 8 threads |
|---|---:|---:|
| MLAS GEMM | ~100 ms | 14.5-15.7 ms |
| requantize + accumulator alloc | 1.7-2.5 ms | ~1.5 ms |
| unattributed (densify, executor, measurement) | 0-2 ms | 2.5-8.5 ms |
| **non-GEMM share of the call** | **~2-4%** | **~22-40%** |

Only the requantize + alloc slice is repeatable and genuinely thread-stable at ~2 ms. The
unattributed remainder is **not** constant in the thread count — it is larger at 8 threads than at
1, and it varied 2.5 / 4.3 / 8.5 ms across three consecutive runs of an identical configuration on
this shared, heavily contended host. So the honest statement is a bound: **non-GEMM work is 22-40%
of the call at 8 threads and ~2-4% at 1**, it is dominated by a term we have not attributed, and the
Amdahl direction is clear even though the decomposition is not.

`qlinear_phase_report` (`#[ignore]`d and `#[cfg(feature = "mlas")]`, in `kernels/qlinear_matmul.rs`)
reproduces the split.

The thread count is **not visible at capability time**, so there is no threshold that could have
described this even if we still wanted one: the same shape wins at 1 thread and loses at 32. The
only way out is to fix the parallel efficiency.

### 2. A kernel gap — f16 dense (fixed by #1080; f16 now behaves like section 1)

f16 used to be the only dense range that lost at **one** thread. ORT casts around MLAS's f32
kernels; this EP's f16 path did not reach the same primitive, and `Gemm` had no f16 GEMV at all.
#1080 routes constant-weight f16 `MatMul`/`Gemm` prefill through MLAS SGEMM with a once-only widened
and packed `B`, and adds the missing GEMV.

Same host, `K = N = 3584`, ours/ORT p50, lower is better:

| | before #1080 | after #1080 | source of the "before" |
|---|---:|---:|---|
| `MatMul` f16 M=128, T=1 | 2.47 | **1.00** | this matrix, previous revision |
| `MatMul` f16 M=128, T=4 | 6.57 | **1.76** | this matrix, previous revision |
| `MatMul` f16 M=128, T=16 | 7.10 | **1.30** | this matrix, previous revision |
| `Gemm` f16 M=1, T=1 | 6.57 | **0.86** | #1080's report |
| `Gemm` f16 M=1, T=8 | 46.67 | **1.83** | #1080's report |

The last two "before" figures are **not** re-measurements and could not be: `Gemm` had no f16 rows
in this matrix because it had no f16 GEMV, and the pre-#1080 kernel no longer exists to re-run. They
are quoted from #1080's report and are flagged so nobody mistakes them for a fresh measurement.

The one-thread kernel gap is closed: f16 is at parity at `M = 128` (1.00 / 1.03) and is **parity to
a modest win at `M = 1`** — 0.86 p50 for both `MatMul` and `Gemm` on the run recorded above, which
is ~1.16x faster than ORT, but an independent re-run of `Gemm` M=1 T=1 on the same host gave 0.96,
so the honest band is roughly **1.0x-1.16x**. It is no longer a loss, which is the point; the exact
size of the win is inside this host's noise.

**It is still a gap at higher thread counts.** f16 now loses at 2-16 threads exactly
the way f32 and int4 do (0.98-2.06 at `M = 1`, 1.30-1.90 at `M = 128`; the single sub-1.00 entry is
`Gemm` M=1 T=4 at 0.98, whose p90 is 1.85 and whose spread is 1.01, so it is noise and not a win). In other words f16 has
stopped being a kernel-quality story and has joined the parallel-efficiency story in section 1. When
section 1's per-call serial work is fixed, f16 should clear ORT at every thread count — the
`M = 1`, `T = 1` win says the primitive is now the right one.

The improvement stands on its own regardless: this EP runs its own f16 kernel in every mode, and
#1080 reports that kernel as 2.4x-14.3x faster than its predecessor. That range is ours-before over ours-after, from #1080's own report — it is a different
quantity from this table's ours/ORT ratios and cannot be re-derived by dividing them.

#### The default build never had #1080's route (**fixed natively**)

#1080's f16 prefill gate is `#[cfg(feature = "mlas")]`, and `mlas` is **off by default**. Every row
in the table above was measured with MLAS compiled in; the shipped default build had no such route
and stayed on the row-blocked half GEMM, which re-widens and re-packs the whole of `B` once per row
block (`m = 64` on 32 threads = 64 full passes over the weight).

The native path now has its own answer: `f16`/`bf16` prefill widens `B` **directly into** the packed
L1 panel of the existing AVX2/FMA f32 microkernel, so the weight is traversed once whatever `m` is,
with no widened `B` copy retained. Self-A/B on the same host, default features, production kernel,
p50 steady, `k = 4096, n = 11008` — **ours-before over ours-after**, a different quantity from this
table's ours/ORT ratios:

| dtype | m | before | after | gain |
|---|---|---:|---:|---:|
| f16 | 8 | 36.20 ms | 2.59 ms | **14.0x** |
| f16 | 64 | 118.33 ms | 7.50 ms | **15.8x** |
| f16 | 256 | 169.41 ms | 26.27 ms | **6.4x** |
| bf16 | 1 | 31.36 ms | 1.96 ms | **16.0x** |
| bf16 | 64 | 108.79 ms | 7.66 ms | **14.2x** |

After the change `f16` prefill is at parity with this EP's own f32 SGEMM at `m = 64` (7.50 vs 7.42
ms) and faster than it at `m = 8`. The **ours/ORT ratios above have not been re-measured** — this
host has no model corpus to run the session-level A/B against, so the table stands as it was and
this note is only about the native-vs-native change. Full record:
`docs/benchmarks/2026-08-19-half-prefill-fused-widen-pack-gebp.md`.

#### `bf16` decode had no GEMV at all, and `f16` decode kept a GEMV past its crossover (**both fixed natively**)

The same audit found the mirror-image hole at `M = 1`. `f16` decode has had a dedicated GEMV since
#1082 — at one row no packed panel is reused, so packing looked like pure overhead on a
memory-bound problem — but `bf16` had none, and a single decode token widened and packed the whole
weight to multiply it by one row.

`half_gemv` now serves both formats (`bf16` widens by an AVX2 shift, so it does not require the
`f16c` conversion unit the `f16` path needs). Measuring the GEMV against the fused prefill GEBP
then **inverted the assumption behind the work**, and inverted it further than the first attempt
recognised. A GEMV is the floor only while the weight is too small for the GEBP to accept at all.
Above `HALF_PREFILL_GEBP_MIN_WEIGHT` the packed route is ahead in both formats at every shape
measured, once the shared host's `f32` control row is divided out — at the very first weight above
the gate (1024x1024) it is already 2.0x/2.1x ahead. So decode does not get a threshold of its own:
it takes the GEMV exactly when the GEBP declines the weight.

| dtype | `K x N` | elements | before | after | gain (control-corrected) |
|---|---|---:|---:|---:|---:|
| bf16 | 1024x768 | 0.79M | blocked 0.252 ms | GEMV 0.086 ms | **3.0x** |
| bf16 | 512x512 | 0.26M | blocked 0.078 ms | GEMV 0.014 ms | **5.6x** |
| f16 | 2048x2048 | 4.19M | GEMV 0.223 ms | GEBP 0.177 ms | **1.27x** |
| f16 | 4096x11008 | 45.1M | GEMV 2.441 ms | GEBP 2.029 ms | **1.23x** |
| f16 | 896x151936 | 136M | GEMV 6.172 ms | GEBP 4.611 ms | **1.34x** |

Median of the per-run steady p50, two binaries interleaved rep by rep, 5 repetitions each, against
`main` at `d4cb7341d`; the `f32` control rows of the same runs agree between the binaries to within
3%, and every gain above is quoted after dividing that control out. The `bf16` rows at and above
the weight gate are **unchanged** — `main` already routed them through the fused GEBP (#1365) —
which is the honest scope of the `bf16` half of this work: it fixes decode *below* the gate, where
there was no vectorised route, not above it.

An earlier revision of this change split decode at 33.6M weight elements instead, on a sweep that
varied only `n` at `k = 4096`. Across both axes that threshold is wrong in one direction
everywhere it applies, worst of all for `bf16` at 2048x2048, where it moved decode off the GEBP
`main` already used and onto the GEMV — a 2.1x regression against `main`. It was retired rather
than retuned. Full record, including that retraction and the noisier earlier sweep kept as the
disclosure of this shared host's worst case:
`docs/benchmarks/2026-08-19-bf16-decode-gemv.md`.

### 3. Per-call packing and a missing native kernel — `QLinearMatMul` (**fixed**)

#### 3a. The research build

Everything in this subsection is behind `--features mlas` and describes the **reference** build, not
the one we ship. It is kept because it is the baseline the native kernel in 3b was measured against.

#1058 bound MLAS's integer QGEMM and took u8 x u8 from **27-119x** down to 3-4x. What remained was
structural: **ORT pre-packed the constant B once at session init; this kernel packed inside every
call**, and additionally copied the whole `K * N` weight into a dense buffer on every call. At M=1 a
12.8 MB pack and copy dominated a 1.7 ms call, which was the whole of the residual 22x.

That is now fixed. The kernel takes `set_constant_inputs`, keys a session-lifetime pack on the
weight's full identity (address, `K`, `N`, both signedness flags, and whether the bytes were sign
translated), and skips the dense copy entirely once the pack exists. Cold cost is repaid on the
**first** call:

| K = N = 3584, M = 1 | time |
|---|---|
| first call, including dense copy + MLAS pack | 6.35 ms |
| every later call | 0.108 ms |
| never-packed path (what every call used to cost) | 5.90 ms |

Signedness is no longer a cap either. MLAS documents `AIsSigned` as unsupported off ARM
(`mlas.h:610-611`), and on AVX2 without VNNI u8 x i8 is **not bit-exact** — `vpmaddubsw` sums
adjacent products into a saturating i16, and `255*(-128) + 255*(-128)` clamps. Rather than decline
those combinations to a scalar loop, the offending operand is now *translated* into the unsigned
domain: `XOR 0x80` on its bytes and `+128` on its zero point. Since the kernel computes
`sum_k (a_k - za)(b_k - zb)`, shifting an operand and its zero point by the same constant leaves
every `i32` accumulator bit-identical, and the call lands on the `u8 x u8` kernel. i8 activations
went from 5.5x slower than ORT to **1.7x-33x faster**.

What is left for u8 x u8 is thread scaling, not packing: 1.13-1.20x at one thread, widening to
2.08-2.65x at sixteen — the same root cause as [parallel efficiency](#1-parallel-efficiency-not-kernel-quality--f32-dense-and-int4-matmulnbits).

Both `QLinearMatMul` rules were measured on **x86-64 AVX2 only** and are applied on every
architecture, which is the same convention the rest of this table uses. aarch64 has native `i8 x i8`
kernels (SDOT/SMMLA) that need no translation at all, so it is if anything
better placed there — but its *speed* is unmeasured here and is not claimed to be measured. Correctness on
that lane is covered unconditionally by `qgemm_i32_matches_the_integer_oracle_for_every_signedness`.

Mixed signedness (`u8 x i8`, `i8 x u8`) was not measured either way, and should be assumed to track
the u8 gap rather than the signed win until someone measures it.

The `i8 x i8` "before" ratios above (5.20/4.97/5.22 at 8 threads) were re-measured against ONNX
Runtime 1.27.0 for this round on the same host. An earlier round recorded 2.23-3.07 for the same
scalar path; the kernel side did not change between the two, so the difference is the
baseline and the harness, not a regression.

#### 3b. The build we ship had no integer GEMM at all (**fixed by #1194**)

Section 3a was written as if it described the product. It did not. Every one of those wins is
compiled out unless `--features mlas` is set, and the default build fell through to a path that
had never been optimised at all:

- `read_quantized` widened **both** operands into `Vec<i32>` on every call — 16 MiB for a
  2048x2048 weight, allocated, filled and dropped per call;
- the accumulation was a scalar rank-1 update, so each row of `A` re-streamed the entire widened
  `B`. At M=128 that is 512 MiB of traffic for a 4 MiB weight.

Measured through an ORT session, that was **11.9x ORT at M=1 and 11.9x at M=128** — not the
1.13-1.20x the matrix above claims, and the largest single loss anywhere in the matmul family.

`kernels::qgemm_native` replaces it with a real integer GEMM on the operand *bytes*. The
instruction that fits is `vpmaddwd`: eight `i16` pairs multiplied and pairwise-summed into `i32`,
sixteen multiply-accumulates each. It is exact here, not approximate — a centred `a` is in
`[-255, 255]` and a raw `b` in `[-128, 255]`, so a pair sum cannot exceed 130050 — and unlike the
`vpmaddubsw` that section 3a has to work around, it does not saturate, so **no operand has to be
translated into another sign domain**.

Two kernels, chosen by `M`:

| `M` | strategy | why |
|---|---|---|
| `> 4` | `B` packed into 16-column, k-pair-interleaved tiles in a 256 KiB L2-resident panel | the panel is re-read once per row block, so the SIMD pack costs about 1% of the GEMM it feeds |
| `<= 4` | no pack; the interleave happens in registers, accumulators stay in registers across a `k` block | at one row block a panel would be written once and read once, and `2 * k * n` bytes of stores to serve a GEMV that reads `k * n` **is** the call |

Session A/B, `K = N = 2048`, u8 x u8, same host and harness on both sides, 61 iterations
(`ratio = ours / ORT`, lower is better):

| M | threads | before | after | our time before | our time after |
|---|---|---|---|---|---|
| 1 | 1 | 12.20 | **2.17** | 1.402 ms | 0.226 ms |
| 128 | 1 | 11.90 | **1.20** | 99.84 ms | 9.99 ms |
| 1 | 4 | 37.11 | **4.03** | 1.379 ms | 0.170 ms |
| 128 | 4 | 14.44 | **1.47** | 31.03 ms | 3.14 ms |
| 1 | 16 | 83.12 | **35.63** | 2.366 ms | 1.336 ms |
| 128 | 16 | 42.28 | **15.05** | 51.05 ms | 16.55 ms |

ORT's own side moved by under 1.5% between the two arms at one and four threads, which is what
makes the comparison a comparison. The i8 x i8 win widened too: 0.206 to **0.049** at M=1, one
thread.

The sixteen-thread rows are directional only. This host's session A/B is not usable at that width —
ORT's own p50/p90 spread there is 4.8x — and the kernel-level harness
(`bench_qgemm_ab`, `#[ignore]`) is the reliable instrument for scaling:

| shape | 1 thread | 2 | 4 | 8 | 16 | portable control, 1 thread |
|---|---|---|---|---|---|---|
| 1x2048x2048 | 0.229 ms | 0.136 | 0.090 | 0.098 | 0.166 | 4.92 ms (**21x**) |
| 4x2048x2048 | 0.565 ms | 0.311 | 0.199 | 0.237 | 0.345 | 4.58 ms (**8.1x**) |
| 128x2048x2048 | 8.911 ms | 4.773 | 2.755 | 2.780 | 1.991 | — |
| 128x5120x5120 | 53.56 ms | 27.06 | 14.35 | 8.98 | 11.51 | — |

The kernel scales to four threads and then flattens: this host has sixteen physical cores but the
sweep is pinned to eight of them, so eight and sixteen are SMT siblings. What the table also shows
is that the **session** does not reach even the four-thread kernel number (0.170 ms against 0.090
ms), which is EP-level thread plumbing rather than this kernel, and is tracked with the rest of the
oversubscription work.

Two things are deliberately left open:

1. **The constant-B pack is not cached.** `B` is a graph initializer and its packed panel could be
   built once per session, which would remove the pack from prefill entirely. That cache is keyed on
   a weight, so it has to go through `kernels::governed_weight_cache` and a budget; it is a separate
   change with its own CI gate, not a rider on this one.
2. **ORT is still ahead at M=1** (2.17x). The gap is now structural rather than sloppy: MLAS reaches
   32 multiply-accumulates per pair of instructions with `vpmaddubsw`, which *saturates*, and this
   kernel reaches 32 per four with `vpmaddwd`, which does not. Closing it means giving up exact
   integer arithmetic, and this EP does not trade determinism for throughput.

### 4. The decode f32 GEMV was written but never shipped (**fixed**)

#1091 added a native `M = 1` SGEMV to the built-in `SimdX86` backend — the same mechanism MLAS uses
(`SgemmKernelM1Avx`): stream B in place, no packed panel, because at one output row every byte of B
is read exactly once and a packed copy is reused zero times. It landed behind
`ONNX_GENAI_CPU_MM_SIMD_M1_GEMV`, **default off**, "until the win is measured". Nothing measured it,
so every shipped decode `MatMul` kept paying the packed GEBP path: a full `K * N` read-and-write copy
of B for no reuse, driven by a `6x16` microkernel with five of its six rows idle.

It is measured now, and the route is the default. There is no env probe on the dispatch any more.

**Kernel level**, `bench_f32_gemm_ab`, one rayon thread, `taskset -c 8-15`, min-of-5, Qwen2.5-14B f32
shapes. The `M = 128` rows are the harness's built-in control: the `M = 1` route cannot reach them,
so they measure this run's noise and must stay at ~1.00.

| shape (M x K x N) | packed (shipped) | GEMV (new default) | packed / GEMV |
|---|---:|---:|---:|
| 1 x 5120 x 7168 (QKV) | 31.889 ms | 5.051 ms | **6.31x** |
| 1 x 5120 x 5120 (o_proj) | 20.517 ms | 3.328 ms | **6.17x** |
| 1 x 5120 x 13824 (gate/up) | 68.102 ms | 8.911 ms | **7.64x** |
| 1 x 13824 x 5120 (down) | 65.828 ms | 10.169 ms | **6.47x** |
| 1 x 5120 x 152064 (lm_head) | 734.203 ms | 95.901 ms | **7.66x** |
| CTL 128 x 5120 x 5120 | 108.535 ms | 111.192 ms | 0.98x |
| CTL 128 x 5120 x 13824 | 292.547 ms | 291.881 ms | 1.00x |
| CTL 128 x 13824 x 5120 | 257.195 ms | 261.052 ms | 0.99x |

The packed arm runs at 2.1-2.6 GF/s against the GEMV's 13.9-16.2 GF/s, which is the `6x16`
microkernel doing one row of useful work in six plus the pack traffic.

**Session level**, through an ORT session on both sides (`plugin_path_ab_vs_plain_ort`,
`bench_matmul_f32_*`, `1x2048x2048`, 31 interleaved iterations, both pools pinned to the same
count). Convention as everywhere else here: **ours/ORT, lower is better**. `M = 128` is again the
control.

| threads | M | p50 before | p50 after | p90 before | p90 after |
|---|---|---:|---:|---:|---:|
| 1 | 1 | 5.63 | **1.11** | 5.71 | **1.12** |
| 1 | 128 (CTL) | 1.05 | 1.02 | 1.05 | 1.02 |
| 4 | 1 | 5.33 | **2.25** | 4.90 | **2.35** |
| 4 | 128 (CTL) | 1.58 | 1.55 | 1.57 | 1.58 |

At 8 and 16 threads the control moved by 8.4x and 1.4x between the two arms on this shared host, so
**no conclusion is drawn from those runs** — that is the control doing its job, not a result. What
is left after this fix is the thread-scaling gap of
[section 1](#1-parallel-efficiency-not-kernel-quality--f32-dense-and-int4-matmulnbits): one thread is
now 1.11, four threads is 2.25.

The two routes differ only in summation reassociation — the same products, summed in a different
order — so this changes f32 results at `M = 1` within the tolerance
`m1_route_matches_packed_within_tolerance` pins. Each route is itself deterministic: the GEMV's
column strips are disjoint and its K-unroll order is fixed, so the same input gives the same bits on
every run and at every thread count.

### 5. The int4 prefill re-decoded the weight once per row, and was switched off (**fixed**)

The scheduler isolated `gemm_nbits_llama3_8b_qkv_t8` at ~10x behind ORT measured native-alone, flat
across 8/16/32 threads. Reproduced on `f8f3878ba` at 6.837 ms against the note's 6.90 ms — three
significant figures apart — so the cell is stable.

Two defects, and the second is the expensive one.

**It was switched off.** `borrowed_int4_prefill_block_enabled()` defaulted to `false`, so every build
anyone ran took the row-serial `borrowed_affine_int4_matmul`. This is
[section 4](#4-the-decode-f32-gemv-was-written-but-never-shipped-fixed) again, and #1080's f16 fix
behind a `mlas` feature gate a third time: a toggle added "until the win is measured", and then
nothing measured it. §36.2 of the phase-18 benchmark document had already written down "it is dead in
the default build"; it had not been acted on.

**It re-decoded the weight for every activation row.** Per 32-lane chunk of one column the int4 inner
loop spends ~18 instructions unpacking nibbles (load, two masks, a shift, two `unpack`s, eight
widen/converts) to feed **four** FMAs. Better than 80% of the instruction stream is decode, and the
row loop sat outside all of it.

Two independent measurements say so. Time was **exactly linear in `m`** — 0.890 / 0.859 / 0.851 ms
*per row* at `m` = 1 / 8 / 128, which is what zero row reuse looks like. And the roofline: 402 MFLOP
in 6.87 ms is 58.6 GFLOP/s, **0.61 FLOP/cycle/thread** against ORT's 6.07. A kernel 10x off compute
peak whose entire 15.7 MiB working set fits in one 32 MiB L3 slice is not waiting for memory.

Worth recording what it was *not*, since two plausible causes were built and rejected:

* **Not bandwidth.** 100.7 MiB in 6.87 ms is 14.6 GB/s, an order of magnitude under this part. A
  64-column tile with rows inner — so a tile's weight and activations both sit in L2 — moved nothing
  at 16 or 32 threads. Removed.
* **Not fork/join.** The path whose whole purpose is to replace `m` fork-joins with one reads
  6.874 ms against the default's 6.897 ms.

The fix is `borrowed_int4_rowblock_avx2`: decode each K block's nibbles **once**, then drive
`PREFILL_ROW_BLOCK = 4` rows through the decoded vectors, taking the instruction budget per 32 MACs
from ~22 to ~8.5. Within a row it is instruction-for-instruction the single-column case of the
existing `NCols4` kernel. The default is flipped to on; the variable stays as a kill switch.

All cells, 32 threads, **native-alone** (`--native-only` / `--ort-only` as separate arms, per
`sebastian-paired-harness-coresidency` — paired runs depress the native arm by up to 4.8x here).
"before" is `ONNX_GENAI_CPU_MM_INT4_PREFILL=0`, exactly the path that shipped.

| cell | before | after | speedup | ours/ORT was | now |
|---|---:|---:|---:|---:|---:|
| `llama3_8b_qkv_t8` | 7.087 ms | **2.339 ms** | 3.03x | 9.35 | **3.09** |
| `llama3_8b_qkv_t128` | 103.250 ms | 28.862 ms | 3.58x | 12.49 | 3.49 |
| `llama3_8b_qkv_t512` | 509.199 ms | 97.591 ms | **5.22x** | 23.97 | 4.59 |
| `llama3_8b_mlp_t8` | 15.350 ms | 5.327 ms | 2.88x | 10.47 | 3.63 |
| `llama3_8b_mlp_t128` | 195.611 ms | 56.196 ms | 3.48x | 13.10 | 3.76 |
| `llama3_8b_mlp_t512` | 810.630 ms | 227.322 ms | 3.57x | 15.49 | 4.34 |
| `qwen3_0p6b_qkv_t8` | 0.895 ms | 0.261 ms | 3.43x | 9.32 | 2.72 |
| `qwen3_0p6b_qkv_t128` | 13.645 ms | 4.981 ms | 2.74x | 19.98 | 7.29 |
| `qwen3_0p6b_qkv_t512` | 51.084 ms | 15.337 ms | 3.33x | 8.50 | 2.55 |
| `qwen3_0p6b_mlp_t8` | 1.718 ms | 0.678 ms | 2.53x | 10.10 | 3.99 |
| `qwen3_0p6b_mlp_t128` | 26.891 ms | 6.981 ms | 3.85x | 18.83 | 4.89 |
| `qwen3_0p6b_mlp_t512` | 104.909 ms | 30.250 ms | 3.47x | 17.16 | 4.95 |

Twelve cells, twelve wins, 2.53x-5.22x, no regressions. Row reuse now shows up where its absence
used to: per-row cost falls with `m` (0.292 / 0.226 / 0.191 ms at `m` = 8 / 128 / 512) instead of
sitting flat at ~0.9.

Like section 4, the two routes differ only in summation reassociation (~2 ULP): the per-element path
reduces every 32-lane block to a scalar, the blocked kernel keeps a lanewise accumulator and reduces
once. Nibble decode is untouched. The test that asserted byte-identity is rewritten to check both
paths against an **f64 oracle** and require the blocked one to be no worse, and byte-identity is
still asserted across fan-out partitions, across row-tile boundaries, and against `NCols4` for a
single row.

What is left is the decode itself: MLAS SQNBit's CompInt8 path quantizes activations to int8 and uses
integer dot products, avoiding f32 dequantization entirely, where this kernel still widens every
nibble to f32. That is a structural step, not a tuning one, and VNNI — which this host lacks — is
what would make it pay.

Full record: [`docs/benchmarks/2026-08-19-int4-prefill-row-blocking.md`](../benchmarks/2026-08-19-int4-prefill-row-blocking.md).

Sequel in section 9: the fused pack that made this route viable was itself a scalar loop, and
vectorizing it moved both of this section's row thresholds.

### 6. `Gemm` declined the f16 fast path whenever B was transposed (**fixed**)

`GemmKernel::execute` disqualified both f16 fast paths if either transpose flag
was set, on the reasoning — written in a comment directly above the check — that
both "read B in its stored `[K, N]` order, and materialising a transpose first
would give back what they save". The premise is correct; the conclusion is not.
A transpose is only needed if you insist on reusing the `[K, N]` kernel. A
`[N, K]` weight is in fact the *better* GEMV layout: every output element is one
**contiguous** `k`-run rather than a strided gather, so the weight streams front
to back and the output partitions to any granularity, down to a single row.

So `transB = 1` at `M = 1` fell into the portable blocked half GEMM — the path
the dispatch comment in `gemm.rs` already calls "the worst dense region measured
anywhere in this EP". It measured **32-48 ms against ORT's 0.16-1.5 ms at `K = N = 3584`: 22x to 65x
slower**, and it did not improve with thread count at all (36.6 ms at 1 thread,
36.2 at 8, 48.1 at 16). This is not a corner case — `transB = 1` is what every
`nn.Linear` export produces.

The fix is a second kernel rather than a transpose: `half_gemv::gemv_f16_nk`,
four independent 8-lane FMA chains along `k`, 8 output rows per task. `M = 1`
`transB` now reads **0.63-0.84 — a win — at 1 through 8 threads**, and at 16 a
ratio too noisy on this host to quote; in absolute terms 38x to 90x faster, up
to 148x on per-run minima.

What this did **not** fix, and what was measured and rejected:

* **`transB` prefill is still broken** and got no better: `M = 128` measures
  **4.0x at 1 thread and 17.0x at 8** (156 ms vs 39 ms). Unlike the decode
  cells these are long enough to be reproducible. The GEMV
  correctly declines `M > 1`, so prefill still takes the blocked path. Closing
  it needs a packed **NT** half GEMM — the f16 analogue of #1176's transposed-B
  SGEMM. **This has since been closed — see [section
  8](#8-transposed-b-transb--1-was-packed-by-a-scalar-strided-gather-fixed) —
  though not by the NT micro-kernel predicted here: the arithmetic was never
  the problem, the *packer* was.**
* **The residual loss at high thread counts is the section 1 ceiling**, not
  anything specific to this kernel. Every one of our `M = 1` GEMVs flattens at
  ~0.7-1.1 ms past 4-8 threads while ORT keeps scaling. Measured on the same
  sweep: 4-bit goes 3.60 -> 0.79 ms across 1..16 threads while ORT goes
  1.59 -> 0.13; f32 goes 1.73 -> 0.94 while ORT goes 1.76 -> 0.17. Note what this
  rules out: we do **not** get slower as threads are added, so it is not
  fork/join overhead — we stop getting *faster*.
* **Task granularity is *not* the cause — measured and rejected.** The obvious
  suspect was `gemv_half_kn`'s fixed `STRIPE = 512`: at `n = 3584` it yields 7
  tasks however many workers are offered. Making the width adaptive
  (`n / (2 * threads)`, rounded up to a multiple of 32) raises that to 8/16/28
  tasks at 4/8/16 threads — and moves nothing. Two interleaved A/B runs of the
  same binary pair, each with a null control, disagreed in sign:

  | run | host load | t=4 | t=8 | t=16 |
  |---|---|---|---|---|
  | 1 (5 trials) | 5.0 | — | +49.1% *(noise 51.9%)* | **+41.0%** *(noise 0.5%)* |
  | 2 (9 trials) | 9.9 | -15.1% *(noise 8.9%)* | +1.1% *(noise -1.3%)* | +9.2% *(noise 33.5%)* |

  The effect is smaller than this host's between-run variability, so the change
  was **not shipped**. The absolute numbers are what actually settle it: native
  `p50` stayed at 1.15-1.64 ms in *both* arms at *every* thread count. 4x the
  tasks, same time.
* **The mechanism the rejection points at.** `gemv_half_kn` holds a `w`-wide
  `f32` accumulator *in memory* and loops `p` outermost, so every FMA is a
  load-modify-**store** against L1 — `w` is far too large for the 16 available
  `ymm` registers, at 512 lanes or at 128. Narrowing the stripe keeps the
  accumulator in L1 either way, which is precisely why it changed nothing.
  Fixing this means inverting the loop nest to hold a register-resident
  accumulator tile across the whole `k` contraction — what `gemv_f16_nk`
  already does for `[N, K]` — not tuning a constant. That is a kernel rewrite
  and is the next thing to try, but it is unproven and is claimed as nothing more.
  **It has since been done and measured: see [section
  7](#7-the-f16-decode-gemv-accumulated-through-memory-fixed).** The mechanism
  named here was the right one.

Full record: [`docs/benchmarks/2026-08-19-f16-gemm-transb-decode.md`](../benchmarks/2026-08-19-f16-gemm-transb-decode.md).

### 7. The f16 decode GEMV accumulated through memory (**fixed**)

`kernels::half_gemv::gemv_half_kn` is the x86 `M = 1` half GEMV. Its own module documentation states
the premise — at `M = 1` "each weight element is touched exactly once, so the kernel is purely
memory-bound" — and it was not true. Measured against this host's sustained read bandwidth
(`roofline_bandwidth`: **75.8 GB/s**, saturating by 4-8 threads), the kernel ran at **12-47 GB/s**,
under half the machine on every large cell. It also *anti-scaled*: `l3_3584` went 0.568 ms at t=4 to
0.830 ms at t=32.

The tell was that a 0.5 MB working set that never leaves L2 moved the same GB/s as a 134 MB one that
cannot fit anywhere. A memory-bound kernel would be far faster per byte on the small cell. The limit
was per-core and the extra threads were only contending for it.

The `p` (contraction) loop was outermost, so each output's accumulator was live across the whole
contraction but lived in `acc`. That is **three memory operations per 8-lane FMA** — load the weight,
load the accumulator, store it back — plus a store-to-load forwarding round trip from the previous
`p`. `STRIPE = 512` keeps the accumulators in L1, which the old comment offered as reassurance; L1 is
only cheap next to L3, not next to a register, and 16 `ymm` were idle.

Tiling the output into `TILE = 64` columns and hoisting `p` inside leaves eight accumulators in
registers for the entire contraction, stored once — one memory operation per FMA, which is the
minimum the problem admits. Tiling adds no traffic: the same `k * STRIPE` elements, the same stride,
and `TILE` is exactly two cache lines with `STRIPE` a whole number of tiles.

This change lives in the `stripe_simd_fn!` macro, so it is instantiated for the bf16 kernel as well
as the f16 one.

**Which routes reach this kernel matters, and changed underneath the work.** [Section
5](#5-the-int4-prefill-re-decoded-the-weight-once-per-row-and-was-switched-off-fixed)'s sibling
change, the decode handover added in #1381, sends an `M = 1` half `MatMul` of **1,048,576 elements or
more** to the fused widen-pack GEBP instead. So the `MatMul` route reaches this kernel only below
that weight; `Gemm` with `transB = 0` reaches it at **any** weight, having no gate at all. Both are
measured, the first with the GEBP switched off to isolate the kernel, the second with **no
environment set**, which is what a default build runs.

Native-alone (`ab.py --native-only --null-control`, 7 x 30, medians, latest `main`):

| route | cell | weight | t | before | after | speedup |
|---|---|---:|---:|---:|---:|---:|
| `MatMul`, GEBP off | `l2_1024` | 2.1 MB | 4 | 0.090 ms | 0.053 ms | 1.70x |
| `MatMul`, GEBP off | `l3_2048` | 8.4 MB | 4 | 0.213 ms | 0.095 ms | **2.24x** |
| `MatMul`, GEBP off | `l3_3584` | 25.7 MB | 16 | 0.813 ms | 0.312 ms | **2.61x** |
| `MatMul`, GEBP off | `dram_8192` | 134.2 MB | 16 | 4.067 ms | 2.126 ms | 1.91x |
| `Gemm`, **default** | `l2_512` | 0.5 MB | 16 | 0.026 ms | 0.014 ms | 1.86x |
| `Gemm`, **default** | `l3_2048` | 8.4 MB | 16 | 0.442 ms | 0.200 ms | **2.21x** |
| `Gemm`, **default** | `l3_3584` | 25.7 MB | 16 | 0.561 ms | 0.324 ms | 1.73x |
| `Gemm`, **default** | `dram_8192` | 134.2 MB | 16 | 3.278 ms | 1.913 ms | 1.71x |

**15 of 15** `MatMul` cells win by 1.13x-2.61x above their null control; **13 of 15** `Gemm` cells by
1.25x-2.21x. The two unclaimed cells are both t=32, with nulls of 22.7% and 48.7% — unusable controls
rather than small effects.

`dram_8192` is the only cell whose weight really comes from DRAM, so it is the only one the DRAM
roofline applies to: **33.0 to 63.1 GB/s, 44% to 83% of 75.8**. The L3-resident cells now read *above*
75.8 GB/s (up to 88.6), which is not a violation — their weight never reaches DRAM — but it does mean
the "% of roofline" figures previously quoted for them were comparing against the wrong ceiling.

Full matrix, the `TILE` sweep, and the two rejected explanations are in
[`2026-08-19-f16-decode-gemv-register-tile.md`](../benchmarks/2026-08-19-f16-decode-gemv-register-tile.md).

Unlike the int4 row-blocking change, this is **bit-identical**: tiling changes which register holds a
partial sum, never the order it is built in, so every pre-existing oracle test passes unmodified.

`TILE = 64` was chosen by sweeping 32/64/96/128 across all 15 cells. 128 needs 18 of 16 architectural
`ymm` and spills — on the smallest cell it is slower than making no change at all.

**A consequence this change does not act on.** #1381 placed the decode handover using a sweep of the
*untiled* GEMV. Re-running that same harness against the tiled kernel inverts it at the two largest
shapes — `4096x11008` goes 1.18 to 0.82 and `896x151936` 1.26 to 0.86, both now favouring the GEMV —
while `2048x2048` narrows from 1.78 to 1.26 without inverting, and the `ab.py` cells disagree with the
bench harness at that weight depending on thread count. The threshold is therefore wrong at both ends
and right in the middle, and a single weight cutoff cannot express a thread-dependent crossover.
Retuning it needs its own `k x n x` thread sweep; it is filed as a follow-up rather than folded in, and
**this change alters no routing** — every shape takes the route it took before, only faster.

What is left is the same fan-out problem as [section 1](#1-parallel-efficiency-not-kernel-quality--f32-dense-and-int4-matmulnbits):
`STRIPE = 512` gives `n = 3584` only 7 stripes, so past 7 threads the workers contend. It is now the
dominant remaining loss on these cells — every unclaimed cell above is t=32 — and an adaptive stripe
width was already measured and refuted.

### 8. Transposed `B` (`transB = 1`) was packed by a scalar strided gather (**fixed**)

`Gemm` f16 with `transB = 1` is the shape a fused QKV projection takes when the weight is stored
output-major. At `M = 128`, `K = N = 3584` it measured **4.48x** against ORT at one thread and
**15.99x** at eight — one of the few cells that gets dramatically *worse* as threads are added.

The cause was in `half_gemm::pack_b`, not in any kernel. `MatrixLayout::transposed(k)` has
`column_stride = k`, and the packer's inner loop walked `column`, so it read `B` with stride `k`,
one scalar `to_f32` per element, never reaching F16C. Transposed `B` stores each logical column
*contiguously*; the packer was walking the one axis that strides.

This was localised by emitting each shape twice — `transB = 1`, and `transB = 0` over the same
array physically pre-transposed. Identical product, identical numbers, same micro-kernel; the only
difference is the layout handed to `pack_b`. The transposed spelling cost a flat **~62 ms more at
both `t = 1` and `t = 8`**, and a constant that does not shrink under 8x the cores is not
arithmetic. It is `pack_b` being called once per row-block while `gemm_impl` scales the row-block
*count* with the thread count.

`pack_b_transposed` reads along the stored direction instead, so each column is a contiguous run
through the same `T::pack_contiguous` the row-major path uses (F16C on x86, FP16/bf16 on NEON — no
new intrinsics, no new architecture gates). Columns are handled `TRANSPOSED_PACK_GROUP = 4` at a
time so the transposing stores are contiguous too; the group width is swept in the constant's docs.

**27 cells across three geometries, 27 wins, 1.29x-2.66x** when opened; re-measured on the merged
head immediately before landing, **26 of 27 prefill cells improve beyond their own null, -24.2% to
-65.0% (1.32x-2.86x)**, with the 27th (`qwen3_0p6b_m32 t=8`) not claimed because its null moved
46.3%. The headline cell, `M = 128 / t = 8`,
goes **14.0x -> 7.0x** against ORT within one invocation (the 15.99x above is the same cell from a
separate reproduction run; before/after must come from one run). `M = 1` is unaffected and
unclaimed — it takes `gemv_f16_nk` and never reaches `pack_b`; three `M = 1` cells that read as
regressions at 5 trials were all within noise at 11 trials x 40 runs. Results are **bit-identical**
for `f16` — the F16C and scalar converters agree on all 65536 patterns — and differ for `bf16` only
on the 126 signalling NaNs, which the scalar converter quiets and the `<< 16` widening preserves;
both are NaN, and the change makes transposed `B` agree with row-major `B`. Fill order cannot
change the packed panel, and
`transposed_b_is_bit_identical_to_pre_transposed_row_major` asserts `to_bits()` equality on every
execution path rather than a tolerance.

Bounded honestly: this removes **two-thirds to three-quarters** of the layout penalty and nothing
else. The pre-transposed control is *itself* still 2.6-3.3x behind ORT, so the half GEMM remains as
far behind as section 1 describes for f32; and `B` is still re-packed per row-block, which is most
of the 9-20 ms residual. Full record, including the negative control showing row-major `B` does not
move: [`docs/benchmarks/2026-08-19-f16-nt-gemm-packing.md`](../benchmarks/2026-08-19-f16-nt-gemm-packing.md).

> The `Gemm` f16 `M = 128` rows in the matrix above (1.03-2.24) **do not describe the shipped
> default build**: on it, `transB = 1` measures 4.24x and even the row-major control 2.88x at
> `M = 128 / t = 1`. They belong to the same research-build category already flagged for `MatMul`
> f32 `M = 1` and `QLinearMatMul`. They have not been re-measured row by row, so they are annotated
> rather than rewritten.


## Precision

`MatMulNBits` bits=8 M=1 was the only range this EP won outright, and part of that margin was bought
with accuracy: the decode GEMV quantized activations to int16 even at `accuracy_level = 0`, which
ONNX defines as fp32 compute. Measured max absolute deviation from a float64 oracle:

| K = N | int16 activations | fp32 activations | ORT 1.27 |
|---|---|---|---|
| 1024 | 1.9e-3 | 2.3e-5 | — |
| 3584 | 6.0e-3 | 1.1e-4 | **1.2e-4** |
| 4096 | 6.4e-3 | 9.2e-5 | — |

The fp32 path tracks ORT's own error; the int16 path was ~55x worse for 6-18%. It is now gated on
`accuracy_level >= 2`. **The win survives the fix** — 0.23-0.25 ours/ORT at parity=PASS — which is
why bits=8 is the one range still claimed.

## Hardware gaps

Not measurable on this host, and therefore **not claimed anywhere**:

- **AVX-512 / AVX-VNNI / AMX.** The VNNI dot kernels exist and are now bit-exactness-tested by a
  forced-dispatch unit test that runs on every lane, and unrunnable requests are clamped rather than
  faulting (this was a reproducible `SIGILL`). Their *speed* is unverified.
- **Real ARM64 silicon.** QEMU gives correctness only, never performance. The aarch64 policy
  branches are exercised on the CI ARM64 lanes for correctness and by
  `ONNX_GENAI_CPU_ARM64_INT4_DIRECT=0`, which reproduces the macOS/iOS dispatch policy on Linux
  aarch64.
- **Cold / first-call cost.** Every ratio above is steady state with weights prepacked. Packing is
  measured separately and is never folded into a steady-state number.

## Reproducing

> **The recipe that stood here did not work.** It built a `bench_prec` binary
> with `--native-threads` / `--ort-intra-threads` flags. No such binary exists,
> and `git log` finds no commit that ever added or removed one, so the numbers
> above cannot be reproduced as written. The closest surviving native-vs-ORT
> harness is `onnx-genai-bench --bin compare`; see
> [`crates/onnx-genai-bench/README.md`](../../crates/onnx-genai-bench/README.md).

`native/ort` in the result line is the ratio quoted here; `parity` must read `PASS`.

The `QLinearMatMul` numbers in [3b](#3b-the-build-we-ship-had-no-integer-gemm-at-all-fixed-by-1194)
come from two harnesses instead. The session A/B — the one that counts — is

```sh
export NXRT_ORT_LIB_DIR=<ort-prebuilt>/lib
NXRT_REQUIRE_ORT_TESTS=1 NXRT_MM_BENCH=1 NXRT_MM_BENCH_CASE=qlinear \
  NXRT_MM_BENCH_ITERS=61 NXRT_MM_BENCH_THREADS=1 \
  ONNX_GENAI_MLAS_THREADPOOL_THREADS=1 RAYON_NUM_THREADS=1 \
  taskset -c 8-15 cargo test --release -p onnx-runtime-ep-cpu-plugin \
  --test plugin_ort_e2e plugin_path_ab_vs_plain_ort -- --ignored --nocapture
```

All three thread knobs must agree or the harness refuses to run: a pinned A/B needs both pools
pinned. Use `-c 8-15` at one thread and `-c 0-15` above it. The kernel-level scaling table comes
from

```sh
QGEMM_AB_ITERS=21 taskset -c 0-15 cargo test --release -p onnx-runtime-ep-cpu --lib \
  bench_qgemm_ab -- --ignored --nocapture
```

whose `portable` arm is the control: it is the same arithmetic with none of the blocking, so it must
not move when a SIMD constant is retuned. If it does, the host was busy and the run says nothing.

For the `M = 1` f16 GEMV of [section 7](#7-the-f16-decode-gemv-accumulated-through-memory-fixed),
which is a native-vs-native comparison rather than a ratio against ORT:

```sh
python3 scripts/ort_ab/gen_f16_gemv.py --out <models>
cargo build --release -p onnx-genai-bench --features bench-native \
  --bin bench_generic --bin roofline_bandwidth
./target/release/roofline_bandwidth --threads 1,2,4,8,16,32 --mib 1024 --seconds 3
python3 scripts/ort_ab/ab.py --native-only --null-control \
  --arms base=<before> tile=<after> --models <models>/*.onnx \
  --threads 4 16 32 --trials 7 --runs 30 --warmups 10 --csv <out>.csv
```

`--native-only` is not optional for a native-vs-native A/B: ORT's intra-op pool spin-waits, and on
these cells a paired run depressed the native median by up to 6x and pushed the null control past the
effect being measured.

### 9. The int4 prefill's fused dequant pack was scalar, and its row gate was measured against it (**fixed**)

Section 5 fused the dequantization into the GEBP pack, so the f32 weight is never materialized: at
`4096x11008` the route reads 22.5 MB of packed int4 and never writes a 180 MB f32 panel. Issue
#1471 proposed fusing the dequant into an L2 tile to remove that panel — but that had already
landed in #1356, and the panel it describes does not exist on `main`. The fixed cost it measured is
real; the cause is not the one it names.

Fitting `t = fixed + marginal * m` over the GEBP arm at `4096x11008` gives **fixed = 4.80 ms**.
Against 22.5 MB of packed weight that is **4.7 GB/s** — 6% of this host's 75.8 GB/s. Nothing about
it is bandwidth. It is the pack loop: `Int4Weight::dequant_column` walks one column at a time, so
per element it does a shift, a mask, a scalar widen, a subtract, a multiply and a store to
`dst[p * NR + slot]`.

The fix dequantizes eight columns at once and transposes them in registers. The arithmetic is
unchanged — the same widen, subtract and multiply, kept as separate `_mm256_sub_ps` and
`_mm256_mul_ps` so it is never contracted into an FMA — so the packed panel is **bit-identical**,
asserted directly against the scalar path rather than through the GEMM.

**Correction (measured after #1556 landed).** #1556's write-up attributed the win to the strided
store. That attribution is wrong, and the correct one is now measured. The 8-bit weight uses the
same panel, the same store pattern and the same microkernel, and differs only in having no nibble
unpack — so vectorizing *its* pack isolates the store term exactly. Over 5 interleaved reps at
`4096x11008`:

| pack | fixed cost | saved |
|---|---:|---:|
| int4 scalar | 4.92 ms | — |
| int4 SIMD (#1556) | 2.09 ms | **2.83 ms** |
| int8 scalar | 2.58 ms | — |
| int8 SIMD | 2.52 ms | **0.07 ms — inside the noise** |

The store is worth ~0.07 ms; the scalar nibble unpack was worth the other ~2.76 ms, **97.6% of the
win**. The panel is only `KC * NR * 4` = 16 KB and stays in L1, so a strided store into it is
cheap — the "separate cache line per element" reasoning was wrong about a buffer this small. The
8-bit vectorization was implemented, proved bit-identical and mutation-tested to confirm the vector
path was live, measured at 1.002x/1.010x/1.010x/0.992x/0.997x, and **discarded**: ~90 lines of
unsafe SIMD for no measured gain. Reproduce with `PROBE_BITS=8` on `benches/int4_prefill_route_ab`.

The rule this leaves behind: **vectorize a packer when its per-element arithmetic vectorizes, not
because its stores are strided.** An L1-resident panel absorbs the strided store.

`fixed` falls **4.80 ms -> 2.24 ms**, and with it both of section 5's row thresholds, which were
measured against the scalar pack: the large-weight crossover moves **12 -> 5** and the L2-resident
one **24 -> 12**. Half the win in production comes from the pack and half from rows the gate now
lets through.

Production A/B on real `MatMulNBits` models, native-alone, median of five interleaved reps,
`null` control per cell: **14 cells improve 1.02x-1.31x, 9 within noise, no regression survives
re-measurement.**

| cell | threads | base ms | new ms | speedup |
|---|---:|---:|---:|---:|
| `llama3_8b_qkv_t8` | 8 | 3.808 | 2.897 | **1.31x** |
| `llama3_8b_mlp_t8` | 8 | 8.811 | 6.793 | **1.30x** |
| `llama3_8b_mlp_t128` | 8 | 35.901 | 27.579 | **1.30x** |
| `llama3_8b_qkv_t128` | 8 | 15.092 | 12.171 | **1.24x** |
| `llama3_8b_mlp_t8` | 16 | 4.465 | 3.670 | **1.22x** |
| `llama3_8b_qkv_t512` | 8 | 45.496 | 41.752 | 1.09x |

`llama3_8b_qkv_t8` is the cell the scheduler evidence flagged; it is `m = 8`, which the old gate of
12 sent to the column-blocked kernel and the new gate of 5 sends to the GEBP.

Against ORT in the same paired invocation (paired mode depresses the native arm, so read the
before/after pair, not the absolute): `llama3_8b_qkv_t8` **4.61x -> 3.56x**, `llama3_8b_mlp_t8`
3.42x -> 2.73x, `llama3_8b_qkv_t128` 1.98x -> 1.57x, `llama3_8b_mlp_t128` 1.84x -> 1.44x,
`t512` 1.25x -> 1.17x and 1.20x -> 1.11x.

Bounded honestly: 8-bit is untouched — `Int8Weight` keeps the per-column default, so its behaviour
is byte-for-byte what it was. `m = 1` cannot reach any of this -- no int4 prefill route is gated
below `m = 2`, a floor section 11 pins with a `const` assert -- and is unchanged. (This sentence
originally read "the lowest gate of any route is 4", which was true when it was written and stale
within two PRs; the conclusion survives because it rests on the floor, not on the value.)

And the route is still behind ORT at small `m`, for the reason section 5 already gives:
MLAS SQNBit's CompInt8 path uses integer dot products where this one widens every nibble to f32,
and VNNI is what would make that pay.

Full record: [`docs/benchmarks/2026-08-19-int4-prefill-dequant-pack-simd.md`](../benchmarks/2026-08-19-int4-prefill-dequant-pack-simd.md).

_Sequel: the pack was still the dominant term at small `m` after this. See section 10._

### 10. The int4 pack's remaining fixed cost was index arithmetic, not unpacking (**fixed**)

Section 9 halved the int4 prefill pack. It was still the dominant term at small `m`: at
`4096x11008` the fitted fixed cost was 2.16 ms against 22.5 MB of packed weight — **10.8 GB/s, 14%
of this host's 75.8 GB/s roofline**, and the pack is already parallel over column strips. Not
bandwidth. The microkernel next to it runs at 1161 GFLOPS, ~95% of this host's AVX2 FMA peak, so
the pack was the whole remaining opportunity.

Two costs, both bookkeeping rather than arithmetic:

1. **The four packed bytes were four bounds-checked indexes.** `self.packed[byte_at]` ..
   `[byte_at + 3]` assembled into a `u32` compiles to four loads, four bounds checks and the shifts
   to reassemble. Taking `self.packed[byte_at..byte_at + 4].try_into()` compiles to one unaligned
   32-bit load with one bounds check. **fixed 2.155 -> 1.775 ms.**
2. **Scale and zero point were re-derived every eight depths.** They are constant across a whole
   block, but the group is eight depths, so at `block_size = 32` each was recomputed four times per
   block — and for int4 the zero point lookup is itself a nibble extract. Hoisting both to block
   scope: **fixed 1.881 -> 1.407 ms.**

Together **2.155 -> 1.407 ms, 1.53x**, on top of section 9's 2.2x. Neither touches the arithmetic;
the panel stays bit-identical, still asserted directly against the per-column path.

The block-scoped hoist introduces an invariant worth naming: the vector loop now walks whole groups
*inside one block*, so `block_size - pc % block_size` must stay a multiple of the group. It does,
because `pc` is a multiple of `KC` and both `KC` and `block_size` are multiples of 8 — the test now
covers `block_size` 24 and 40, which are multiples of the group that do **not** divide `KC`, so the
invariant is pinned rather than assumed.

**Both row gates moved again**, for the third time, exactly as section 9 predicted they would
whenever the pack's cost changes:

| regime | scalar pack | after §9 | after §10 |
|---|---:|---:|---:|
| non-resident (`INT4_PREFILL_GEBP_MIN_ROWS`) | 12 | 5 | **3** |
| L2-resident (`..._L2_RESIDENT`) | 24 | 12 | **6** |

On the non-resident shape the GEBP is now ahead at *every* `m >= 1`, so that crossover has fallen
off the bottom of the sweep. The constant is set to `3` rather than `1` deliberately: `m = 2` is a
1.7% difference, inside the noise, and `m = 1` is decode, which has its own route and should not be
re-pointed on the strength of a prefill bench.

Production A/B on real `MatMulNBits` models, 25 shapes x 3 thread counts, `--native-only` with a
null control: **60 of 75 cells improved, 15 within noise, 0 surviving regressions**, parity `PASS`
on every row.

| cell | threads | base ms | new ms | speedup |
|---|---:|---:|---:|---:|
| `qwen3_0p6b_mlp_t8` | 32 | 1.585 | 0.556 | **2.85x** |
| `qwen3_0p6b_qkv_t8` | 32 | 1.086 | 0.422 | **2.57x** |
| `qwen3_0p6b_qkv_t8` | 8 | 0.945 | 0.389 | **2.43x** |
| `qwen3_0p6b_mlp_t8` | 8 | 1.444 | 0.733 | **1.97x** |
| `llama3_8b_qkv_t8` | 8 | 3.125 | 2.193 | **1.43x** |
| `llama3_8b_mlp_t8` | 8 | 7.629 | 5.373 | **1.42x** |
| `qwen3_8b_square_t8` | 8 | 2.100 | 1.477 | **1.42x** |

The `qwen3_0p6b` `t8` cells are the largest because they are the L2-resident gate moving 12 -> 6:
at `m = 8` they were taking the column-blocked route and now take the GEBP.

Three cells first read as regressions and none survived re-measurement at 11 trials x 40 runs:
`llama3_8b_qkv_t512` t=16 +1.53% -> **-1.73%**, `qwen3_8b_square_t8` t=16 +5.33% -> **-21.21%**
(agreeing with its own t=8 and t=32 siblings, which had improved 1.42x and 1.22x — opposite signs
on the same shape at different thread counts is the noise signature). The third,
`qwen3_0p6b_qkv_t1` t=16, is `m = 1` on a 1.57 MB weight, whose gate is 6 both before and after, so
it **provably cannot reach any changed code**; at 15 trials x 60 runs it reads -3.16% against a
-7.22% null.

Versus ORT, paired before/after from the same invocation at 8 threads:

| cell | before | after |
|---|---:|---:|
| `qwen3_0p6b_qkv_t8` | 4.69x | **2.59x** |
| `llama3_8b_qkv_t8` | 2.93x | **2.05x** |
| `llama3_8b_mlp_t8` | 2.57x | **2.03x** |
| `llama3_8b_qkv_t128` | 1.53x | 1.53x |

`t128` is unchanged, which is the control: there the pack is a small fraction of the call and the
microkernel already runs near peak.

**What this does not fix.** `m = 1` is untouched. 8-bit prefill keeps the per-column scalar pack —
and per the section 9 correction, vectorizing it is measured to be worth nothing.
`INT4_PREFILL_GEBP_MIN_ROWS_UNBLOCKED = 4` is untouched; it gates a different competitor and was not
re-measured — section 11 does that, and finds it could not have been re-measured here because the
bench had no way to express its block size. The residual gap to ORT at small `m` remains
structural: MLAS `SQNBitGemm` CompInt8 wants VNNI, which this host does not have.

### 11. The gate for block sizes the row kernels reject was never measured at all (**fixed**)

Sections 9 and 10 each re-derived `INT4_PREFILL_GEBP_MIN_ROWS` and its L2-resident twin. Section 10
closed by naming `INT4_PREFILL_GEBP_MIN_ROWS_UNBLOCKED = 4` as "untouched"; section 9 did not name
it at all. It was untouched because it was **unmeasurable**: `int4_prefill_route_ab.rs` pinned
`block_size` to 32, so no run of that bench could reach the branch. This is the third instance of
one defect —
`PROBE_BITS` (#1558) existed for the same reason, and `scripts/ort_ab/gen_gemm.py` pinned
`BLOCK_SIZE = 32` *and* stepped tokens `1 -> 8`, straddling every row gate the last three PRs moved.
A threshold that no harness can express is not conservative, it is unowned.

`PROBE_BLOCK` and `gen_gemm.py --block-size/--tokens` close that. The measurement then reads very
differently from the 32-element case, because **the competitor is different**. For a block size the
column-blocked kernels reject, the route below the threshold is not a rival vectorized kernel — both
`borrowed_affine_int4_matmul_prefill` and `..._nblock` require 32-element blocks — it is
`borrowed_affine_int4_matmul`, a per-block scalar dot. GEBP's speedup over it, native-alone, steady,
median of five interleaved reps:

| m | 2048x2048 | 4096x11008 |
|---|---|---|
| 1 | 2.51x | 4.62x |
| 2 | **4.06x** | **8.96x** |
| 3 | 6.81x | 14.6x |
| 4 | 10.4x | 18.1x |
| 6 | 13.2x | 30.1x |
| 8 | 13.8x | 30.6x |

GEBP wins at every `m` on both shapes, and the margin grows linearly because the dot arm's cost is
linear in `m` (9.4 -> 76.9 ms on the large shape) while GEBP's is flat (~2 ms, the pack). Unlike the
32-element gates this one needs no residency split: both shapes agree. Gate **4 -> 2**.

The measured floor is 2048x2048 (2.1 MB). A weight far below that -- a few KB -- is unmeasured, and
there GEBP forks the global pool where the scalar fallback stays on the narrow decode pool, so the
fork/join could in principle dominate trivial compute. It is a correctness non-issue (GEBP
zero-pads partial panels) and no real projection at `block_size = 16` is that small, but the claim
above is bounded by the shapes measured.

Production A/B on 16-element-block models through the real dispatch path, `t = 8` and `t = 16`,
confirms it, and the ORT ratio on the rows that moved:

| cell | before | after |
|---|---|---|
| `llama3_8b_qkv_t3` | 20.30x | **2.60x** |
| `llama3_8b_mlp_t3` | 17.98x | **2.48x** |
| `qwen3_0p6b_qkv_t3` | 16.48x | **2.99x** |
| `llama3_8b_qkv_t2` | 13.63x | **2.53x** |
| `qwen3_8b_square_t2` | 13.20x | **2.34x** |

Parity PASS on every cell. That band — 2.3x-3.0x — is the same one the 32-element weights reach
after section 10, so this closes an outlier class rather than opening a new front: 16-element blocks
were paying 13-20x, not the 2-2.6x everything else pays.

`t1`, `t4` and `t8` are the controls and are structurally identical between the arms (gates 4 and 2
both exclude `m = 1`, and both admit `m >= 4`). Every apparent delta on them failed to survive
re-measurement — `llama3_8b_qkv_t1 t=16` read +0.95%, +5.49% and -11.80% across three runs against a
0.06% null, which is the signature of host noise, not a result.

**Set to 2, not 1, deliberately.** `m = 1` is decode, and its 2.51x/4.62x here is measured by a
single-op prefill bench that cannot see the per-token pool contention the narrow decode pool exists
to avoid: GEBP returns before `with_decode_pool` and drives the global pool. That is the same reason
`borrowed_affine_int4_matmul_prefill` is itself gated at `m >= 2`. Moving decode wants a decode-loop
measurement, and is left open.

> **Superseded by section 12.** The decode-loop measurement was run and the objection above did not
> survive it: at block 16 the contention argument points the other way, and the gate is now 1. The
> reasoning is preserved because it was right to *withhold* the change until decode was measured —
> what it got wrong was the sign, not the standard of evidence.

**What this does not fix.** The residual ~2.3x-3.0x to ORT is the same structural CompInt8/VNNI gap
as section 10. 8-bit weights at 16-element blocks were not separately retuned —
`INT8_PREFILL_GEBP_MIN_ROWS` is 2 already, so there is no equivalent gap to close.


### 12. The decode gate was held up by an objection that measured backwards (**fixed**)

Section 11 set `INT4_PREFILL_GEBP_MIN_ROWS_UNBLOCKED = 2` rather than 1, refusing a measured
2.51x/4.62x at `m = 1` on the grounds that the number came from a single-op prefill bench: GEBP
returns before `with_decode_pool`, so under a real decode loop it would fork the machine per op per
token and lose whatever it had won. That was the right thing to withhold on. It was also,
measured, backwards.

**The harness the objection required.** `int4_decode_loop_ab.rs` runs a llama3-8B projection chain
(qkv 4096x6144, o 4096x4096, gate/up 4096x14336, down 14336x4096) at `m = 1`, one token at a time,
across `PROBE_SESSIONS` concurrent threads sharing one weight set. Both arms come from **one
binary** with the gates forced to 1, selected by `ONNX_GENAI_CPU_MM_INT4_GEBP`, and a third
unmodified build is run interleaved as a control for the `current` arm. It agreed to within 1% in
every cell.

The detail that decided the result is that each token runs inside **one**
`with_decode_pool_scope`, which is exactly how `native_decode/cpu.rs` drives a single-token forward.
A first version of this bench called the kernels directly. It reported GEBP 1.28x/1.53x/1.64x ahead
at 1/2/4 sessions and the margin *growing* with concurrency — a clean, plausible, entirely
artefactual result. Installing the scope the way production installs it moved the `current` arm
alone by 1.44x at block 32 and inverted the finding. **Measuring the challenger faithfully is not
enough; the incumbent has to be measured in the pool it actually runs in.**

**Aggregate tokens/s, steady, mean of three interleaved reps.** The per-session median is
unusable here — under contention the sessions serialise on the shared pool, so the median reports
whichever token owned it (7.1 ms) while p90 reports the ones that waited (56.3 ms). Throughput and
p90 are the honest pair:

| block | competitor at `m = 1` | GEBP | current | verdict |
|---|---|---|---|---|
| **16** | generic **scalar** per-block dot | **97.3** | 30.6 | GEBP **3.2x** |
| 32 | `borrowed_affine_int4_matmul_nblock` | 108.7 | 118.2 | GEBP 0.92x |
| 64 | `borrowed_affine_int4_matmul_nblock` | 111.8 | 202.5 | GEBP 0.55x |
| 128 | `borrowed_affine_int4_matmul_nblock` | 119.3 | 256.1 | GEBP 0.47x |

So the answer is not "GEBP is good at decode" or "GEBP is bad at decode". It splits exactly along
**which kernel is on the other side**, which is the fourth time this file has had to record that a
threshold is a property of both kernels either side of it. Against a vectorized column-blocked
kernel GEBP loses at `m = 1`, and loses harder as the block grows. Against the scalar fallback it
wins, and the contention objection inverts:

| sessions | GEBP | scalar dot | speedup | GEBP p90 | dot p90 |
|---|---|---|---|---|---|
| 1 | 97.3 | 30.6 | 3.2x | 10.3 ms | 29.7 ms |
| 2 | 111.5 | 32.3 | 3.5x | 18.4 ms | 30.9 ms |
| 4 | 122.8 | 29.7 | **4.1x** | **35.7 ms** | 233.3 ms |

Tail latency improves 6.5x at 4 sessions. The global-pool fork that was supposed to sink this is
real, and it is still cheaper than leaving the work on a scalar dot inside the narrow pool.

**The gate names exactly one block size.** `MatMulNBits` rejects any block size that is not a power
of two `>= 16`, and 16 is the only such value that is not a multiple of 32 — so "unblocked" is the
block-16 gate and nothing else. That is now asserted rather than assumed: the crossover test sweeps
`[16, 32, 64, 128, 256, 512]` and requires the unblocked branch to claim 16 and only 16. Blocks 24
and 48 were tried against the bench and rejected by the kernel's own validation, which is how the
constraint was found.

`INT4_PREFILL_GEBP_MIN_ROWS` (3) and its L2-resident twin (6) are **unchanged and re-confirmed** by
the same run: the 32/64/128 rows above are the direct evidence that lowering them to 1 would be a
regression of 1.08x-2.1x.

**Confirmed end to end.** The paired-ORT production A/B, run at `--block-size 16 --tokens 1` — the
first time that harness has contained a row any int4 row gate controls — improves **all 20 cells**,
1.86x-3.43x, parity PASS on every one. Against ORT the shape goes from 6.66x-13.44x to
**3.44x-5.30x**.

**What this does not fix.** Block-16 decode is now 97-123 tokens/s where block 128 reaches 256, so
this closes a 3-8x outlier and leaves the ordinary case alone. The residual 3.4x-5.3x is still worse
than the 2.3x-3.0x band the 32-element weights sit in, worst at `t = 8`, where GEBP fans out on the
global pool while the persistent-SPMD workers spin. `m = 1` at blocks 32/64/128 keeps
today's route. The residual 2.3x-3.0x to ORT is untouched and remains the CompInt8/VNNI gap of
sections 10 and 11. Full record:
[`docs/benchmarks/2026-08-20-int4-decode-loop.md`](../benchmarks/2026-08-20-int4-decode-loop.md).

### 13. Halving the weight bytes at `accuracy_level = 4` made it slower, not faster (**rejected**)

The premise recorded in
[`docs/benchmarks/2026-08-20-int4-acc4-int8-repack.md`](../benchmarks/2026-08-20-int4-acc4-int8-repack.md)
(landed with #1590) was that the int4 `accuracy_level = 4` route wastes its advantage by
repacking 4-bit weights up to int8 before the dot: `prepack_int8_weight` doubles the bytes the
kernel streams, so a kernel reading the ONNX nibbles directly should stream half as much and win.
That premise was tested by building the kernel it implies, and it is **false on AVX2**. (Note this
is a different claim from section 10 above, which concerns the `accuracy_level = 0` prefill pack
and concluded that pack was *not* bandwidth-bound either.)

**What was built.** A complete packed-nibble int4 x int16 kernel: consumes the 0.5 B/weight ONNX
blob unchanged (no value repack — the layout is already `[n][k_blocks][blob_size]` and
block-padded), unsigned nibbles with per-block zero points folded out of the integer dot via
`sum((q - zp) * a) = sum(q * a) - zp * sum(a)` so absent zero points are bit-identical to an
explicit midpoint, per-group f32 scales, block sizes 16/32/64/128 with exact scalar tails, and
`_mm256_madd_epi16` accumulation with four orders of magnitude of i32 overflow headroom. It was
bit-exact against a scalar reference over every nibble value in every lane position, exhaustive
over all 256 byte values x 32 element positions, and matched an independent f64 dequant contract.
Three mutation guards were verified to fail loud.

**The measurement.** It lost in **every cell**, 1.5x-2.2x, against the int8 route it was meant to
replace. Three independent falsifications of the byte thesis, not one noisy result:

- The `nibble/int8` ratio is **flat** (~1.6 at block 32, ~2.1 at block 128) across an 18x weight
  footprint range, 1.6 MB to 29.4 MB. A byte-bound kernel's ratio would improve with size.
- In the one cell where the int8 arm spills the 64 MiB L3 (73.4 MB) and the nibble arm does not
  (44.1 MB) — the best case the thesis can construct — the nibble kernel is still **1.63x slower**.
- The incumbent already runs at **74-77 GB/s, 98-102% of this host's 75.8 GB/s DRAM ceiling**, at
  block 128; the nibble kernel reaches 16% of it. Fewer bytes cannot help a kernel that cannot
  saturate the bus it already owns.

**The mechanism.** `_mm256_madd_epi16` retires 16 products per instruction where
`_mm256_maddubs_epi16` retires 32, so an int16-activation kernel needs twice the multiply
instructions at the same width; and int16 activations are 2 B, so a fixed 32-weight step reads twice
the activation bytes and needs two loads where int8 needs one. Per 32 weights: int8 ~5 vector ops,
nibble ~14. The nibble unpack itself (~4 ops) is the *smaller* half of the loss. Folding each
group's i32 partial into one `f32x8` block accumulator — the 8-bit kernel's structure — recovered
only 1-3 pp and cost bit-exactness, so it was reverted too.

**What is kept.** The kernel is deleted rather than parked behind a default-off flag. The nearest
precedent — the aarch64 int8-activation diversion removed at `borrowed_affine_int4_matmul`'s
precision contract — is only a partial analogy, and worth stating precisely: that one was removed
because it was *semantically* wrong for its only caller, whereas this kernel is correct and in fact
more accurate, and is being dropped purely on speed. What carries the decision here is that the
loss is flat across every footprint tested, so no host or model size argues for keeping it, and a
flag is not free: it is a second prepack cache and a live dispatch arm to keep correct forever. The
experiment survives as this branch's first commit if anyone wants it back on AVX-512. What survives
in tree is the finding the experiment turned up by accident and
`accuracy4_int4_decode_error_envelope_is_pinned_against_f64` now pins: int4 `accuracy_level = 4`
quantizes activations to **int8** (4.75e-3 relative error) while 8-bit `accuracy_level = 4`
quantizes to **int16** — the same attribute selecting routes **9,703x** apart in accuracy. Because
`accuracy_level = 4` is a contract to be *less* accurate, no output-comparison test can fail when
that error drifts, so it was untracked; the pin's lower bound is the important half, failing if the
asymmetry ever closes so the record gets revisited instead of rotting.

**Bearing on the remaining gap.** ORT does `llama3_8b_qkv` at `accuracy_level = 4`, `m = 1` in
0.18 ms against our 0.361 ms, and our int8 arm is already at 98-102% of DRAM — so ORT is not
streaming faster. It must touch fewer bytes per output row or reuse more from cache. Weight byte
*width* is now excluded as the lever, which is what this experiment bought. Full record:
[`docs/benchmarks/2026-08-20-int4-nibble-i16-negative.md`](../benchmarks/2026-08-20-int4-nibble-i16-negative.md).

### 14. The half decode's handover to the fused GEBP was measured on one thread count and one `k` (**retired**)

`MatMul` sent every `m == 1` `f16`/`bf16` decode with `k * n >= HALF_PREFILL_GEBP_MIN_WEIGHT`
(1 048 576) to the fused widen-pack GEBP. `Gemm` did not — its `m == 1` `f16` path stayed on the
decode GEMV at every weight. That is the divergence #1381 recorded, and the obvious reading was
that `Gemm` was missing a gate. Re-measured, it was `MatMul` that was carrying a bad one.

**What the original evidence missed.** Its sweep was 32 threads and `k <= 2048`. Run through the
same production harness (`benches/half_decode_gemv_ab.rs`, arms selected by
`ONNX_GENAI_CPU_MM_HALF_GEMV/GEBP`, `f32` control divided out) at **8 threads**, the handover is a
loss at every `full`-set shape at or above the gate (`/ctl` 0.26–0.76) and in 20 of the 22 `f16`
cells of the `band` set over two independent repetitions (`/ctl` 0.20–1.14) — up to **5.0x**
slower — with the GEBP's weight bandwidth pinned at 20–34 GB/s independent of shape while the GEMV
reaches 24–155 GB/s. The two `band` cells that do exceed 1.00 sit *exactly* at the retired
threshold at `k = 1024` and do not hold across repetitions (1.14 then 0.89). At 32
threads it is still a loss at every shape a 7B-class decode issues: `k = 4096` qkv/mlp 0.51–0.89,
a 136M `lm_head` 0.86. The corner it does win — `k = 1024`, 1.05M–4.2M, 32 threads, `/ctl`
0.95–1.49 — is real but is not a shape any such model emits, and its `k = 2048` rows are not
reproducible run to run (6.3M measured 1.31 then 0.70 from the same binary).

**Why there was nothing to retune.** The GEBP earns its packing by reusing a `KC x NR` panel of
`B` across the rows of `A`. At `m == 1`, `m_panels` is 1 — every panel is consumed by exactly one
microkernel pass, so none of the packing is repaid. Stated where the cost actually lands: the
widen-pack writes `k*n` `f32` and reads it back, but per strip that is a `bpack` of at most
`KC * 16 * NR * 4 B` = 256 KiB, which is **L2-resident on this host** and so is not a DRAM story;
what does reach DRAM is line amplification, because `pack_b_half` walks `B` column-strip-major and
at the usual one panel per strip touches `NR * 2` = 32 bytes of each 64-byte line, roughly **2x**
the GEMV's read traffic. The rest is L2 bandwidth, the widening work itself, and a fork/join over
strips that a single row of `A` cannot amortise. The two axes follow: the pack work per unit of
reuse rises with `k`, and it can only be overlapped if there are workers to overlap it with, which
is why the arm collapses at 8 threads and merely loses at 32. A decode-specialised GEBP that skips
packing `B` **is** the GEMV, so the finding is structural rather than a threshold.

**Disposition.** `half_decode_prefers_gebp`/`_when` deleted, the `!half_decode_prefers_gebp(..)`
term removed from `MatMul`'s decode arm, and #1381 closed with `MatMul` adopting `Gemm`'s route
rather than the reverse — the third time this file records an in-tree gate whose evidence was
narrower than the region it governed (cf. §11, §12). `HALF_PREFILL_GEBP_MIN_WEIGHT` and
`half_prefill_gebp_selected` are untouched: they are the **prefill** gate, and they still serve a
*batched* `m == 1` half MatMul, which the non-batched GEMV declines and whose only alternative is
the row-blocked half GEMM at 16x–21x. Coverage was repaired in the same change:
`PROBE_SHAPE=band` puts 11 rows immediately below, at and above the retired threshold at three
different `k`, so a `k`-dependent effect cannot hide inside a `k * n` gate again, and the pins are
by execution (`no_half_decode_is_diverted_off_the_gemv_on_weight`,
`gemm_and_matmul_take_the_same_decode_route`) rather than by asking a predicate — the predicate is
what was deleted. Full record:
[`docs/benchmarks/2026-08-21-half-decode-gebp-retired.md`](../benchmarks/2026-08-21-half-decode-gebp-retired.md).

**Still open here.** `Gemm`'s half fast path remains `f16`-only: `bf16` operands return `None` at
the dtype check and fall into the portable blocked half GEMM, where `MatMul` serves them from the
same GEMV. That is a separate, separately mergeable gap — **closed in §15.**

### 15. `Gemm` excluded `bf16` decode by dtype, and it cost 3.0x–24x

The gap §14 left open. `GemmKernel::try_half_fast_path` opened with
`a.dtype != Float16 || b.dtype != Float16 -> None`, so a `bf16` `Gemm` never reached the decode
GEMV at any shape, while `MatMul` has served `bf16` decode from that same GEMV since the
2026-08-19 record. The identical decode got a different kernel depending on which op the exporter
emitted, and `Gemm`'s was the portable blocked half GEMM.

**The measurement.** `bench_gemm_half_decode_route` with `PROBE_DTYPE=bf16`, both arms out of one
build (`ONNX_GENAI_CPU_MM_HALF_GEMV=0` reproduces the pre-change route), `f32` control divided out.
It wins at **every shape at both thread counts**: 4.4x–24.1x at 8 threads and 3.0x–22.3x at 32,
with the model shapes at the top of the range (`llama_mlp` 36.07 → 1.80 ms, `llama_qkv`
9.85 → 0.47 ms, `qwen_qkv` 9.97 → 0.52 ms at 8 threads). The blocked GEMM holds 2.5–6.0 GB/s of
weight bandwidth regardless of shape; the GEMV reaches 26–76 GB/s against a 75.8 GB/s ceiling.

**Disposition.** The dtype gate now calls the same `half_storage_format` helper `MatMul` uses, and
the format is threaded into `simd_available`/`gemv_half_kn` rather than hard-coded. Two defects fell
out of the same term: `Gemm` also did not honour `ONNX_GENAI_CPU_MM_HALF_GEMV`, so the documented
field kill-switch for this route silently covered only `MatMul` and the `Gemm` side of the shipped
binary could not be A/B'd at all. It does now. **Still asymmetric:** a *transposed* `bf16` decode
declines, because `gemv_f16_nk` reads `f16` bit patterns and has no `bf16` twin — kernel coverage,
not policy, pinned by a test that checks both the route and the numerics. Mutation-verified in both
directions. Full record:
[`docs/benchmarks/2026-08-21-gemm-bf16-decode-gemv.md`](../benchmarks/2026-08-21-gemm-bf16-decode-gemv.md).

### 16. `QLinearMatMul` decode had never been measured on the shipped build, and the integer GEMM branched on signedness per 16 bytes

The `QLinearMatMul` rows in the matrix are `--features mlas` numbers, and the caveat under them said
the default build was 11.8x–12.0x behind ORT. That caveat predates #1194's native integer GEMM and
**nothing re-measured it**, because `scripts/ort_ab/` had no `QLinearMatMul` generator at all — the
op could not be driven through the standard harness. `gen_qlinear.py` (new) fixes the coverage hole,
with rows straddling both of the kernel's own gates (`PARALLEL_MIN_WORK`, and the `m <= MR`
fused/packed split).

**The corrected picture on the default build**, one thread, arms interleaved, parity `PASS`
everywhere: **1.13x at u8 M=128**, **0.11x at i8 M=1 (a 9.4x win**, because ORT's own i8 x i8 path is
35x slower than its u8 x u8 path at the same shape**)**, and a real loss of **2.39x–3.21x confined to
u8 x u8 at M=1** — decode. The 11.8x caveat is stale in both directions.

**Localising it.** Timed alone the kernel is ~88% of the call, so the wrapper is not the story
(the 44% an interleaved run implies is the paired-arm L3 eviction both arms pay; the ratio survives,
the absolute does not). Sweeping aspect ratio at a fixed 12.85 MB footprint gives 17–20 GB/s
everywhere, and a **1 MB weight that fits L2 runs at the same 19.5 GB/s** — so at m=1 the kernel is
**instruction-bound, not memory-bound**, up to L3.

**The defect.** `widen16(signed: bool, ..)` — the widen every byte of `B` passes through — took
signedness as a runtime argument and branched on it in the innermost loop, once per 16 bytes, even
though `Operand::signed` is fixed for the whole call and the `#[target_feature]` boundary stops the
compiler hoisting it.

**Disposition.** Signedness is now a `const SIGNED: bool` on `widen16`/`fused_strip`/
`accumulate_fused`/`pack_panel`, with the single runtime `match` moved out to the dispatcher that
already matched on `m`. No arithmetic changed — accumulation is still wrapping `i32` and the output
is bit-identical, asserted by the existing both-sign-domain oracle. Kernel A/B, two prebuilt test
binaries alternated over three repetitions with the harness's `portable` drift control steady to
1.6%: **1.13x–1.14x at m=1** and **1.03x** on the packed `m > MR` path, with non-overlapping ranges
between the arms at m=1.

**A retraction rides with it.** A first end-to-end A/B across two separately built `bench_generic`
binaries read 1.41x–1.46x at M=1 and "2.97x -> 1.61x against ORT". Both are **withdrawn**: a 1.13x
kernel cannot yield a 1.43x call, since `1 / (f/R_k + 1 - f) <= R_k` for any kernel fraction `f`.
A null control (the same binary against itself) puts this host's paired end-to-end floor at ±10%,
and re-running each binary against its own ORT reference back to back gives medians of 2.43x (base)
and 2.58x (new) — **no measurable end-to-end effect**. The kernel win is real and controlled; its
effect on the whole call is below what this box can resolve, and the u8 M=1 gap against ORT stands
at ~2.4x.

**Still open, measured but not merged:** (a) the u8 M=1 gap itself, ~2.4x and unmoved; (b) *parallel scaling at m=1* — 8
threads buy 1.3x where ORT gets 5.8x, because the fused path splits columns and hands every worker a
page-crossing strided walk of `B`. This section originally proposed a `k` split with private
accumulators as "the shape that streams"; **that was built, measured and did not pay for itself** —
see [section 17](#17-splitting-k-in-the-fused-decode-gemm-does-not-pay-for-itself-negative-result).
The loss (b) describes is still open. (c) The single-thread residual is an
instruction budget: exact full-range 8-bit needs `vpmaddwd` at ~0.25 vector uops/byte, and
`vpmaddubsw` — which would halve it — saturates unless one operand stays within +/-64
(`255 * 64 * 2 = 32640` fits `i16`, `255 * 65 * 2` does not), which is precisely why ORT's
quantizer ships `reduce_range` for non-VNNI AVX2. Full record:
[`docs/benchmarks/2026-08-21-qlinearmatmul-m1-signedness.md`](../benchmarks/2026-08-21-qlinearmatmul-m1-signedness.md).

### 17. Splitting `k` in the fused decode GEMM does not pay for itself (**negative result**)

[Section 16](#16-qlinearmatmul-decode-had-never-been-measured-on-the-shipped-build-and-the-integer-gemm-branched-on-signedness-per-16-bytes)
left the m=1 parallel-scaling loss with a named fix: the fused path splits *columns*, so at
`n = 3584` and eight workers each worker walks 448 contiguous bytes out of every 3584-byte row — a
stride past a 4 KiB page, which is where the stride prefetcher stops following, with no `A` reuse to
hide the latency. Splitting **`k`** instead hands each worker whole rows to stream, paid for with one
private `m * n` `i32` accumulator per band and a reduction.

It was built (twice), it is **correct**, and it does not pay for itself. It is not merged.

The correctness half is worth keeping on the record: accumulation is wrapping `i32`, exactly mod
2^32, so summing bands in band order is **bit-identical** to summing `k` in one pass — a `k` split
does not owe a tolerance, it owes an equality, and
`the_k_split_is_bit_identical_to_the_column_split` asserted it at every thread count.

The performance half: 36 cells (six llama/qwen decode and short-prefill shapes x `t = 1,2,4,8,16,32`),
three repetitions, two prebuilt binaries alternated, `portable` drift control at 0.999, and the
`t = 1` rows as a null control because at one thread both binaries execute identical code. That
control holds to +/-4% on five shapes and **fails on `1x1024x3072` (-29% in one matrix, +14% in the
other)**, so all five of that shape's cells are dropped — its wins *and* its losses — leaving 25
counted cells.

* **Hybrid band x column plan, serial reduction: 8 wins, 5 losses, 12 neutral, geomean 0.997.** A
  wash. And it *inverts its own prediction*: the gain should grow with thread count as the stripe
  narrows, but `t = 4` is the best column and `t = 32` the worst.
* **Pool-aligned bands plus a parallel reduction** — the fix implied by the Amdahl term
  `bands * threads / k`, ~6% at 8 bands and 32 workers — is **worse: 3 wins, 13 losses, 9 neutral,
  geomean 0.846** (3/12/9 and 0.904 with one unexplained 0.17x cell also excluded).

A second parallel plan, a scratch allocation and a reduction have to earn their place; a geomean of
0.997 does not buy them.

**Scope, stated honestly.** The experiment moved two things at once — the access pattern *and* the
addition of scratch — and the best explanation of the v2 result is that the **scratch** is what hurt,
contending for the same L3 the weights stream through. So the streaming layout itself was never
measured in isolation, and this does **not** rule out prepacked/reordered `B` (the same idea with the
scratch cost paid once, outside the call), huge pages, software prefetch, NUMA first-touch, or gating
the split to only the sub-page-stripe case. What rules those in or out is still owed.

The claim that the fused decode kernel is **instruction-bound inside L3** rests on the aspect-ratio
sweep in section 16 (12.85 MB at 17-20 GB/s at *every* aspect ratio; a 1 MB L2-resident weight at the
same 19.5 GB/s), not on this; a confounded A/B corroborates, it does not independently confirm.

The planning consequence: the instruction budget is the term that is measured — `vpmaddwd` at ~0.31
total uops per byte of `B` (~0.25 counting vector uops only), with `vpmaddubsw` unable to replace it
at full range — so paths that cut **bytes and uops per weight together**, such as the packed-nibble
`int4 x int16` kernel consuming 0.5 B/weight in place, are the better next spend. Full record, both
complete matrices, the controls and the anomaly:
[`docs/benchmarks/2026-08-21-qgemm-k-split-negative.md`](../benchmarks/2026-08-21-qgemm-k-split-negative.md).

### 18. The packed-nibble `int4 x int16` AVX2 decode kernel pays — but not for the reason it was proposed

[Section 17](#17-splitting-k-in-the-fused-decode-gemm-does-not-pay-for-itself-negative-result) closed
by naming the next spend: paths that cut **bytes and uops per weight together**, such as a kernel
consuming the ONNX int4 weight at 0.5 B/weight in place instead of expanding every nibble to a whole
`i8`. That kernel is built, measured and **merged**. It is **1.2x to 2.4x** faster than the
expanded-`i8` route it replaces across four block sizes, five llama/qwen geometries and `m = 1..64`,
with **no losing cell**. Against ORT the same cells go from ~5.8-8.1x behind to ~3.3-5.6x behind: the
gap is roughly halved, not closed.

The prediction was right about *what* to build and wrong about *why it would win*. Halving the weight
stream, on its own, bought **1-5%** — nothing. `ONNX_GENAI_PROFILE_OPS=1` put `MatMulNBits` at 99.95%
of the run, so there was no Amdahl dilution to hide behind; the kernel had genuinely not moved.

A block-size sweep at `t = 1` on a fixed geometry located it. The weight bytes are identical at every
block size and only the block count changes, yet time tracked `1/block_size` almost exactly. Fitting
`time = blocks*A + weights*B` gave `A = 17.1 cycles/block` against an inner loop already running at
**11.1 weights/cycle** — against a uop budget of 12.8. The multiply-accumulate was never the problem.
At `block_size = 32`, the common case, a block is 32 weights, so ~17 cycles of per-block tail sat on
top of ~2.5 cycles of arithmetic.

Two changes took `A` to **8.5 cycles/block** and produced the entire result:

1. `is_x86_feature_detected!` was evaluated **per block**. It caches in an atomic and is cheap in
   isolation, but it sits on a non-`target_feature` call boundary that prevents the AVX2 body from
   inlining, forcing the accumulator through memory every 32 weights. Hoisting the probe to the row
   driver and marking that driver `#[target_feature(enable = "avx2")]` fixed both.
2. Four consecutive `k` blocks now share **one** reduction tree and **one** vector tail. They are
   contiguous in every array the tail reads — weight scales, activation scales, block sums, and the
   zero-point nibbles — so the zero-point correction, the `i32`->`f32` convert, the two multiplies and
   the add all run once on four lanes instead of four times on scalars.

**The transferable finding: at decode block sizes the per-block tail, not the multiply-accumulate, is
the kernel.** Section 17 measured `vpmaddwd` at ~0.31 uops per byte of `B` and treated that as the
budget to beat; that number is real but it describes only the part of the loop that was never the
bottleneck. Any future int4/int8 decode kernel should be costed as `blocks*A + weights*B` with `A`
measured, and a byte-density change should not be expected to show at all until `A` is down.

`INT4_NIBBLE_MAX_ROWS` was **swept, not asserted**: every `m` from 1 to 64 wins ~2x, and the gate
ships at 64 because that is the largest `m` with a measurement behind it, not a modelled crossover.
The route is guarded off VNNI hosts on purpose — there `vpdpbusd` gives the expanded-`i8` route a
1-uop-per-32-MAC dot and the byte/arithmetic trade may invert, which this host cannot measure.

Three null controls are disclosed, including the one that matters: the `accuracy_level = 0` cells are
code-identical in both arms and read **+/-5%**, which is the real per-cell floor on this host — worse
than the +/-0.1% the duplicated-binary null arm reports on its lucky cells. Two cells that first
contradicted their own shape were re-measured rather than dropped, with the contended reps shown.

Still open and explicitly **not** claimed: ~3.3-5.6x behind ORT remains; `A = 8.5 cycles/block` is
still 3.4x the inner loop's own cost, and the untried lever is an `N` tile so one activation block
serves several outputs (which needs a block-major prepack of the scale and zero-point arrays, both
`N`-major today); VNNI and aarch64 hosts are gated out and untested; Miri cannot execute x86 SIMD, so
it is **no coverage** for this module rather than a pass. Full record, all matrices, the controls, the
mutation battery and the harness bug that first reported it green:
[`docs/benchmarks/2026-08-21-int4-packed-nibble-avx2.md`](../benchmarks/2026-08-21-int4-packed-nibble-avx2.md).

**Reconciliation with the 2026-08-20 rejection.** The same mechanism was built
and rejected the day before at 1.5x-2.2x *slower*
(`docs/benchmarks/2026-08-20-int4-nibble-i16-negative.md`, now marked
superseded). Two kernels, two honest measurements. The differences: that one
spent ~4 uops per 32 weights restoring `k` order in the **weights**, where this
one deinterleaves the **activation** once per row (14 uops -> 10); and neither
kernel's inner loop was the binding cost, so its "instruction-bound" diagnosis
pointed at the wrong term. Its "the incumbent is already at the DRAM roofline"
claim does not survive at all -- the 58.7 MB expanded weight is L3-resident on a
64 MiB L3, and this kernel is 2.37x faster than the arm called saturated.

**The method lesson worth keeping.** Fit `time = blocks*A + weights*B` by sweeping
block size at fixed weight bytes before concluding a kernel is arithmetic-bound.
`A` was 17.1 cycles/block against ~2.5 cycles of arithmetic at `block_size = 32`;
until `A` came down, halving bytes/weight was invisible (v1 measured 1-5%). Two
changes moved it: hoisting a per-block `is_x86_feature_detected!` off a
non-`target_feature` call boundary (17.1 -> 15.0) and tiling four `k` blocks
through one reduction tree and one vector tail (15.0 -> **8.5**).

### 19. The `[N, K]` decode GEMV existed only in an `f16` spelling, and a transposed `bf16` decode paid 21x-101x for it

The asymmetry §15 recorded as still open, closed. `Gemm` with `transB = 1` on a
`bf16` weight declined the decode GEMV and fell to the portable path: **0.038 ms
-> 0.810 ms** at `k=1024, n=768`, **3.42 ms -> 201.9 ms** at a 896x151936
`lm_head`, 21x-101x across the `full` shape set at 8 threads.

**It was never a policy question.** §15 called it "kernel coverage, not policy"
and that was exactly right. The `[K, N]` stripe kernel had already been made
per-format by a macro — `#[target_feature]` is an attribute on a concrete
function, and `bf16` must not be made to ask for the `f16c` unit it widens
without — but the `[N, K]` row kernel never got the same treatment. So `trans_b`
was selecting on **dtype** where it should have been selecting on **layout**.
The `[N, K]` row dot is now instantiated per format from one macro
(`dot_row_simd_f16` / `dot_row_simd_bf16`), `HalfFormat` is threaded through
`gemv_f16_nk` -> `gemv_half_nk` and `dot_row_scalar`, and the
`(!trans_b || format == HalfFormat::F16)` term is deleted. Both operators, both
stored orders and both 16-bit formats reach one backend.

**Why nothing caught it.** The transposed route had **no benchmark row at all** —
`half_decode_gemv_ab` built a `MatMul`, which cannot express `transB`. That is
the third time this file records an unmeasured region behind a gate (cf. §11,
§12, §14), and the repair is the same shape: `PROBE_OP=gemm_transb` now builds a
`Gemm` node with the weight transposed into `[N, K]`, so the route has a row.

**The pins are numeric, not just route counters**, because reading a `bf16`
weight through the `f16` kernel does not fault — it silently reinterprets every
bit pattern. `a_transposed_bf16_decode_takes_the_same_gemv_as_f16` checks the
route *and* an `f64`-referenced bound, `kn_and_nk_agree_on_the_same_bf16_weight`
checks the two stored orders against each other, and
`nk_simd_and_scalar_rows_agree_bit_for_bit` now runs both formats. Forcing the
`f16` kernel fails the first two and nothing else, which is the mutation that
matters.

**Disclosed:** the `f32` null control on this shared host spans 0.795-1.264
(+-26%). Several GB/s figures read above the 75.8 GB/s DRAM number, and the
reason is **not** uniform: the mid-size rows are L3-resident (a 4096x4096 `bf16`
weight is 33.6 MB against a 64 MiB L3), but the 896x151936 `lm_head` is 272 MB
and cannot be — its 79.5 GB/s is a ~5% overshoot of a quoted DRAM figure that is
itself approximate. Neither reading is a roofline violation and neither is
evidence of one; a bandwidth percentage means nothing until the denominator is
shown to bind, which is the error §18 had to correct in the other direction.
Full record:
[`docs/benchmarks/2026-08-21-gemm-transposed-bf16-decode-gemv.md`](../benchmarks/2026-08-21-gemm-transposed-bf16-decode-gemv.md).

### 20. Int4 acc4 N-tile — designed, bounded, not built; two hypotheses closed

Investigated the block-major scale/zero-point N-tile over #1619's packed-nibble
kernel. **Not built**, and the reason is recorded rather than hidden: the
analytic case is ~15% (a ~62→~53 uop reduction per 4 (block, column) pairs,
against §18's finding that the per-block term is 78% of runtime at block 32),
but §18's uop model **mispredicts measured per-block cost by 2.8x**, so it
cannot adjudicate a 15% delta. The tile also removes instructions, not the
139.7 MB/token that must cross the memory system, and the measurement that
would have bounded that failed on a contended host (block 128 at 32 threads
spread 1.135–9.035 ms, 8x). MLAS uses `NCols = 4`, so this is a statement about
available evidence, not about the design.

Two cheaper hypotheses were tested and **both are closed negative**:

- *"The kernel does not thread-scale."* An artifact: `RAYON_NUM_THREADS` does
  not size the decode pool (`configured_decode_threads` reads
  `available_parallelism` and `ONNX_GENAI_CPU_DECODE_THREADS`). That much
  stands. **The width curve originally recorded here does not, and has been
  re-measured (2026-08-23);** the numbers below replace
  `22.949/22.893/11.584/5.866/3.302 ms/token at 1/2/4/8/16`.
  - **`t=1` ≡ `t=2` was wrong, not unexplained.** It was recorded here as "an
    open question, not an explanation"; the honest disposition is that it is
    **false**. On a re-measurement with one process per cell, `t=2` is
    **1.96x** faster than `t=1` (7.278 vs 14.300 ms/token) against a **0.00%
    A/A null** over five reps per arm. The 1.96x is not itself a surprise —
    the budget confines the process to `w` CPUs, so `t=2` has twice the
    hardware of `t=1` and ~2x is the *expected* result. That is the point: the
    recorded curve reported no speedup where the ordinary one was, and the
    cause is harness-side and mechanical (#1771), not a property of the pool.
  - **`t=1` does not build a pool at all, so ratios against it are "vs
    serial".** At width 1 the decode budget confines the process to a single
    CPU, and `build_from_env` then *declines* to construct the SPMD pool —
    with one CPU there is no core to run the inline dispatcher alongside a
    spinning worker, so it would starve itself (`decode_spmd.rs`, the
    `allowed.len() == 1` fallback; the crate's own test notes "the smallest
    budget that builds a pool is 2"). Decode runs on the flat path and the
    bench reports `path=flat`. Note this is a *different* mechanism from the
    `total_workers <= 1` serial short-circuit in `dispatch_output_rows`, which
    is not what fires here: at width 1 there is no pool and no worker to
    short-circuit. Either way `t=1` is a different code path from every other
    column, so "1.96x over `t=1`" means "over the serial flat path".
  - **The `8→16 = 1.77x ±0.7%` error bar is withdrawn as unverifiable.** Over
    six independent launches `w=16` spans **1.476–9.064 ms/token (514%)**
    while `w=8` spans 3.195–3.509 (9.8%). At `w=16` the run holds all 16
    physical cores and has no headroom, so any co-tenant takes throughput
    straight out of the measurement; at `w=8` there is slack. Note the
    absolute level also moved between builds — the 2026-08-21 `t=8` of 5.87
    against today's 3.31 — so this is **not** a claim that those three
    repetitions were secretly bimodal; it is that a ±0.7% interval on `w=16`
    is not something this host can support, and the original cannot now be
    re-checked because the build has moved. **`8→16` is left unquoted** rather
    than given an interval the measurement cannot carry.
  - Trustworthy cells, five reps each, `realized=`/`as_requested` verified per
    cell: **14.300 / 7.278 / 3.784 ms/token at t=1/2/4** — 1.00x / 1.96x /
    3.78x, spreads 2.25% / 1.73% / 2.60%. **`w=8` is the widest measurable
    cell on a shared host** (9.8% over six launches); `w=16` is not. Full
    record:
    [`docs/benchmarks/2026-08-23-acc4-decode-width-remeasurement.md`](../benchmarks/2026-08-23-acc4-decode-width-remeasurement.md).
- *"Production is stuck on the narrow flat pool, so 1.77x is free."* False.
  `default_persistent_threads(32)` is 16 and `PERSISTENT_POOL_DEFAULT` is
  `true`, so `is_forced()` holds; every pool configuration reachable from this
  harness measures 3.23–3.42 ms/token = t=16, never t=8 (5.87–5.91) nor the
  6-wide flat default. **No pool-default change is proposed.** Note this is
  *not* a persistent-vs-flat A/B: `PROBE_SPMD` does not move the pool under the
  default policy, an earlier draft wrongly labelled an arm "flat", and
  adversarial review caught it — a 6-wide pool cannot produce a 16-wide time.
  (The `1.77x` in the hypothesis is the figure withdrawn above; the refutation
  does not depend on it, since it rests on which pool width is reachable at
  all, not on the ratio between two widths.)

**Shipped:** `PROBE_ACCURACY` in `int4_decode_loop_ab`. The bench hard-coded
`accuracy_level = 0`, and 4 is the only value reaching the packed-nibble route,
so #1619 had **no decode-loop row at all** — only single-op benches, which is
the wrong shape for a decode gate. The bench header now also documents the
`RAYON_NUM_THREADS` trap, because a wrong "does not scale" verdict is one
env-var name away.

**Next lever, unmeasured and deliberately unranked:** at block 32 the f32
scales are 27.26 MB against 109.05 MB of packed weights — with zero points,
**22% of all bytes moved**. Storing them as bf16 in a block-major prepack
removes ~10% of traffic, converting ~1:1 into time *wherever the kernel is
bandwidth-bound*, without touching the integer dot's exactness. It is **not**
claimed to be worth more than the N-tile: the two bet on opposite unmeasured
regimes (the byte lever pays only if bandwidth-bound; the N-tile is discounted
only if it is). Establish the regime on a quiet host first. Full record:
[`docs/benchmarks/2026-08-21-int4-acc4-ntile-design.md`](../benchmarks/2026-08-21-int4-acc4-ntile-design.md).

### 21. QLinearMatMul u8 — instruction mix, the `m = 5` cliff, and `vpmaddubsw`

**Instruction decomposition.** The pack-free kernel costs `6 + 4R` uops per `R`
rows of 32 multiply-accumulates: two load+`cvtepu8_epi16` pairs, two `unpack`
to build `k` pairs, then `madd`+`add` per row. At `m = 1` that is **0.3125
uops/MAC of which only 40% is arithmetic** — the six-uop preamble is fixed per
32 weights and has nobody to amortise against. The packed kernel is `2 + 4R`
(0.1875 at `R = 1`, **1.67x denser**), which is what the panel buys.

**`m = 1` is issue-bound, not bandwidth-bound**, for `k = n <= 2048`:
`1x2048x2048` and `4x2048x2048` read the *identical* 4.19 MB of `B` yet differ
**2.86x**, so time tracks arithmetic, not bytes. Memory only begins to bind at
`1x4096x4096` (16.8 MB of `B`), where throughput falls 21.08 → 14.72 GMAC/s.
Static model predicts 12.8 MAC/cycle against a measured 21 GMAC/s; the ~⅓
shortfall is latency and **is not attributed** — `perf` is unavailable here.

**Shipped:** the pack-free kernel had no row blocking, so `fused = m <= MR` sent
`m = 5` to the packed path with two row blocks and nothing to amortise a
`2*k*n` panel write against — a **2.0x cliff** off `m = 4` (0.543 ms ->
1.090 ms) for 25% more arithmetic. Added row blocking
and moved the boundary to `2 * MR`. Serial gains **1.46-1.48x / 1.37-1.42x /
1.16-1.19x** at `m = 5/6/8` across three independent runs, the last of them on
the merged latest-main head (both reported, not the
more flattering one); `m = 1` and `m >= 12` unchanged by construction and
measured unchanged at 0.998x/1.013x. One-thread null controls repeat to
**1.5%**; the 8-thread arms are the same code and spread **10%**, so the two
floors must not be interchanged.

**First reversal, and it inverted the rule.** The boundary drafted at 16 on
one-thread evidence **loses 0.87x/0.79x at `m = 5/8` on eight threads**.
Serially the pack is latency on the critical path; across the pool every worker
has its own issue width but one shared memory system, so the pack-free path's
extra sweep of `B` per row block binds and the L2-resident panel wins. The gate
is therefore `m <= MR || (!parallel && m <= 2 * MR)` — above `MR` the right
answer is a function of thread count, not of `m`. The 8-thread arms are the
same code, which calibrates this harness's noise floor at **±10%**.

**`vpmaddubsw` — legality derived, benefit quantified, not built.** It cannot
consume today's operands: `A` is centred into `[-255, 255]` and so cannot be the
unsigned operand. Legal form is raw `u8` `A` against a prepacked centred-`i8`
`B' = b - zb`, with `sum_k a·B' − za·sum_k B'` (exact; the column sum is a
prepack constant). Saturation is safe iff `|B'_k| + |B'_{k+1}| <= 128.5`, i.e.
**guaranteed when every `|B'| <= 64`** — exactly reduce-range 7-bit weights. At
full range it is *false*, not tight (`255*127*2 = 64770`). **The contract should
be proven by scanning the constant weight at prepack, not trusted from
metadata**; on failure the kernel keeps exact `vpmaddwd`. Payoff: the `k`-pair
interleave is unavoidable either way, but doing it in the *byte* domain halves
issue to **0.156 uops/MAC** — the largest identified `m = 1` lever. Blocked on a
constant-`B` pack cache, which the native path lacks entirely
(`pack_lookup`/`pack_build` are `#[cfg(feature = "mlas")]`). Full record:
[`docs/benchmarks/2026-08-21-qlinear-u8-m1-instruction-mix.md`](../benchmarks/2026-08-21-qlinear-u8-m1-instruction-mix.md).

### 22. Per-block bookkeeping was 1.68x of the int4 acc4 decode kernel — at block 32, at low thread counts, and nowhere else

`nibble_outputs_avx2` built **two bounds-checked slices per block** and made the
callee recover its group count from `packed.len()` on every call. Removing that —
raw pointers hoisted above the `wide` branch, group count hoisted out of the
block loop — is the whole result. It is bit-identical output.

**The headline does not generalise, and the first draft of it did not
reproduce.** Re-measured on latest main against the merged baseline, min of 6+
interleaved repetitions with a per-row A/A control:

| cell | speedup | | cell | speedup |
|---|---|---|---|---|
| t=1, block 32 | **1.686x** | | t=1, block 16 | 1.028x |
| t=4, block 32 | **1.640x** | | t=1, block 64 | **1.380x** |
| t=8, block 32 | 1.016x | | t=1, block 128 | 1.091x |
| t=16, block 32 | **1.242x** | | 2 sessions x t=4 | 1.419x |

Four of the nine rows in the originating draft did not reproduce — three
optimistic, one (t=16, claimed 1.065x) **pessimistic**. The mechanism is
consistent with the shape: `wide` requires `group >= WIDE_GROUP` (32) and
`wide_groups = blob/16`, so the removed fixed per-block cost is amortized over
1, 2 and 4 groups at block 32/64/128 and the path is not taken at all at block
16. A one-parameter fit from the block-32 row predicts block 64 within 3% and
block 128 within 8%, which is corroboration from two points rather than proof;
the block-16 null is the stronger evidence, since that path provably cannot be
entered. `accuracy_level = 0` is a 1.000x null control at t=1/4/8. **Quote the
cell, not a multiplier** (§18 is the same lesson from the other side).

**Two of my own corrections did not survive re-measurement against current
main, and both failures were mine, not the draft's.** (i) **t=8 reversed to a
wash.** It read 1.141x against `f8eb8a3e2`; against `2f94cba4d` it is 1.016x
(two windows, A/A 0.52% and 0.03%). The *baseline* arm got 7.8% faster across
that rebase window while the patched arm did not move, so the cell closed on its
own. The window contains the native-MTP series, but those are CUDA-graph and
speculative-decode commits with no stated path into a CPU int4 decode kernel;
**the cause is not established** and is recorded as correlated-with-window, not
explained. (ii) **"Specific to block_size 32" was wrong**, and wrong the same
way the three defects below are wrong: block 64 and 128 had only ever been
measured **at t=8**, the one thread count where this change buys nothing,
confounding two variables in one cell. At t=1 block 64 is **1.380x**. The rule
this yields is that a negative result measured at a single point on another axis
is not a negative result.

**A fourth defect was found in review of this very section**, and it is the
same family: the ORT gap table above originally carried the **old-base** native
arms next to the new-base A/B table, undisclosed. At t=8 that manufactured a
2.84x -> 2.49x "improvement" out of a cell this section retracts as a wash. Base
labels now travel with the numbers. **Two tables in one document may not sit on
different baselines without saying so.**

**Three measurement defects were found in the originating evidence, and all
three are protocol, not arithmetic.** (i) The t=8 baseline was its *slow* mode —
that arm is bimodal at 3.54/5.2 ms while the patched arm is stable, so pairing
against 5.880 ms produced 1.609x where the reproducible figure on that base was
1.141x — and 1.016x on current main.
(ii) The multi-session rows used the harness's **pooled median per-token
latency**, which mixes contended and uncontended tokens as sessions
desynchronise — one baseline repetition read 9.348 ms against a 15.9 ms
population and the A/A hit **52.7%**. Aggregate throughput is the only valid
multi-session statistic here, and it moves those rows by ~0.17x. (iii) A cell
whose min and median disagree in *direction* (t=16: 1.234x vs 0.730x) has not
been measured; the fix is a longer window (240 tokens -> 1.245x/1.412x), not a
choice of statistic. Also: `ONNX_GENAI_CPU_DECODE_THREADS=2` read **identical to
`=1`** on this harness. That identity was real, but its cause was not the pool —
see the correction in §27; the knob delivers 1.96x under control.

**Two corrections to this file's host model**, both of which change how earlier
roofline arguments read: **L3 is 32 MiB per CCX, not 64 MiB shared**, and
**75.8 GB/s is not achievable** (measured 31-36 GB/s within a CCX, ~56.6 GB/s
across both). §18's refutation of "the incumbent is at the roofline" stands, but
its stated reason — that a 58.7 MB working set was L3-resident — is void; the
supported reason is that the denominator was ~2x the real ceiling. **SMT
siblings are adjacent pairs**, so physical cores are the even CPUs.

**Unsafe was kept only because safe was measured and is slower.** Three safe
formulations were built: bounds-checked index sub-slicing **1.340x**, nested
`chunks_exact` **1.542x**, and the same with a hoisted group count **1.542x** —
identical to three decimals, which rules out the inner loop shape and localises
the entire cost to caller-side slice construction. The best safe form runs
**8.8% slower** than the shipped one and gives up **13% of the win** (worst:
25.3% slower, 37% of the win), so codegen is *not* equivalent. Shipped form
confines the unsafe
to one `#[inline]` function with a stated loop invariant and adds a
`validate_nibble_outputs` prevalidation pass on the safe entry, so the pointer
derivations rest on checked arithmetic (`checked_mul` throughout) rather than on
caller discipline. No integer-pointer round trips.

**The tiled path had never been executed by a test in its own module.**
`the_kernel_tracks_the_float64_contract` drives `k_blocks <= 3` against
`BLOCK_TILE = 4`, so the tile loop's trip count was always zero; coverage came
only from four `matmul_nbits` integration tests one module away. "1607 tests
pass" was true and vacuous. Six mutations now all die. Two of them only died
after the *tests* were fixed: `tiles - 1` is an **equivalent mutant** (work
migrates to the tail loop) and was replaced with `tiled_blocks + 1`, and the
fail-before-unsafe test had to assert the **panic message** rather than
`is_err()`, because malformed inputs also trip an incidental bounds check
*after* the pointers are formed.

**Miri covers this code only if you make it.** `is_x86_feature_detected!("avx2")`
is **false** under Miri, so a default run takes the generic path and the vector
tests early-return — vacuous. Under
`RUSTFLAGS=-C target-feature=+avx2` with `-Zmiri-strict-provenance` Miri
interprets the intrinsics; that it genuinely covers both raw-pointer derivations
was proved by injecting an off-by-one group read and a `+8` element offset and
confirming Miri reports UB for each. Clean as shipped.

**Retired, do not build:** the int4 acc4 **N-tile** (0.94x DRAM-resident, 0.76x
at full width; its positive number is an L3-residency artifact of undersized
benchmark shapes) and **bf16 scales / block-major prepack** for throughput
(0.96x at tile 1; the traffic arithmetic is right and the kernel is not
traffic-limited where it would help). Both remain defensible for *footprint*;
neither may be sold as speed.

**The gap against ORT, which is the number that actually matters.** There was no
int4 ORT baseline in the tree — `benches/ort_baseline.py` is f32-only — so one
was added as `benches/ort_matmulnbits_baseline.py`: the same five projections in
one graph, matched `block_size`/`accuracy_level`/thread count, one `Run` per
token. ORT's `accuracy_level` is honoured rather than assumed (30.632 ms at
acc0 vs 7.822 ms at acc4, 3.9x apart).

| threads | ORT acc4 | native before | native after | before | **after** |
|---|---|---|---|---|---|
| 1 | 7.822 ms/tok | 23.535 | 13.962 | 3.01x | **1.78x** |
| 4 | 2.154 | 6.100 | 3.720 | 2.83x | **1.73x** |
| 8 | 1.249 | 3.264 | 3.214 | 2.61x | 2.57x |
| 16 | 1.227 | 1.804 | 1.452 | 1.47x | **1.18x** |

Both native columns are the current-main (`2f94cba4d`) arms. Quoting the older
`f8eb8a3e2` arms here would credit the change with the t=8 win that the
re-measurement retracts, and an earlier revision of this section did exactly
that — see the review note below.

Three qualifications travel with that table. **ORT saturates by t=8** (1.249 ->
1.227 to t=16), so the t=16 row flatters us — parity there is worth much less
than the same ratio at t=1, and per-`Run` framing overhead is not negligible at
~1.2 ms/token, so that row carries the most uncertainty. **The worst remaining
row is t=8 at 2.57x**, and it barely moves (2.61x -> 2.57x) because the change
is a wash at t=8; that cell is now the top of the list and nothing here
addresses it. And every row is measured **without** zero-points on both sides,
which is ORT's fastest configuration (they cost ORT ~26%: 9.885 vs 7.816 ms,
min over three windows each)
and therefore the harder comparison.

A fourth qualification, added 2026-08-23: **the t=16 row should not be relied
on at all on a shared host.** Six independent launches at that width span
1.476–9.064 ms/token (514%) against 3.195–3.509 (9.8%) at t=8 — see §20. The
native before/after cells there are 1.804 and 1.452, a 1.24x improvement, and
both sit inside the launch-to-launch spread of a *single* arm at that width;
the 1.18x gap figure inherits the same problem, since it is 1.452 divided by
an ORT number. Neither is resolvable by the measurement that produced it. The
t=1 and t=4 rows have headroom and are unaffected; **t=8 is the widest row
worth arguing from.**

**The default path was the bigger target when this was written, and is not any
more.** This kernel is gated to `accuracy_level = 4`; production default is 0,
where native measured 56.307 ms/token against ORT's 30.632 — **1.84x** at the
time. That figure is now stale; the re-measurement replaces it.

> **Re-measured 2026-08-23: the acc0 gap is ~1.12x, and 1.84x is stale.** On
> `e189244ba`, llama / block 32 / one session / matched core budget, arms
> interleaved and the gap medianed over paired cells: **t=1 1.120x** (native
> 27.9 tok/s, ORT 31.2; range 1.112–1.128 over 2 retained cells of 3),
> **t=8 1.120x** (range 1.089–1.145, 3 cells), **t=4 ~1.15x but unresolved** —
> its A/A null spans 0.868–1.150, so the gap sits inside its own noise floor at
> that width. Flat across the measurable range, so acc0 is neither a scaling
> problem nor the top target any more — **conditional on t=16**, which did not
> resolve here and pointed at ~1.6x.
>
> **The condition failed. acc0 is back at the top.** A dedicated study the same
> day (`0f84888b8`, 30 launches, acceptance rule pre-registered) measured
> **~1.78x at t=16** — paired medians 1.782 and 1.773 across two runs with
> different token budgets, 1.770x best-launch-vs-best-launch, and 1.650x even in
> the half of cells where native ran fastest. **acc0 is ~1.12x at t=1 and t=8
> and ~1.78x at the width closest to an unconfined production process.** The
> scaling wall was then measured directly, both widths inside the same launch
> with the width order rotated: **native converts the t=8 -> t=16 doubling into
> 1.319x where ORT gets 1.762x, in 10 of 10 paired launches** (pre-registered
> sign test, threshold 80%). So this is not a new kernel deficiency at that
> width — it is the t=8 gap plus a scaling failure that is ours alone. It is
> **not** the host bandwidth knee (`2026-08-22-decode-width-scaling.md`): a DRAM
> ceiling is a property of the host and would have flattened both arms, and ORT
> scales 1.76x across the same doubling on the same host in the same launch.
> **What `t=8` and `t=16` physically are has since been read out of `/proc`**
> (`benches/decode_placement_census.sh`, a `Cpus_allowed_list` census on
> `0a668d54b` — categorical, not a timing). Every configuration places **one
> worker per physical core** with the reserved dispatcher CPU left clear: the
> default unpinned launch and `THREADS=16` both put 15 workers on `0,2,…,28`
> across both L3 instances, and `THREADS=8` confines the process to
> `[0,2,4,6,8,10,12,14]` and runs 7 workers on `0,2,…,12` — **entirely inside
> one 32 MiB L3**. So the `t=8 -> t=16` doubling on this host doubles cache and
> memory-controller reach as well as cores. It is **not** a confound: `ort()`
> defaults its pin to `native_pin(threads)`, so both arms get the same CPUs at
> each width, and both figures post-date that fix (`4b4dacc7e`). It does make
> the 2.0x ideal a conservative reference for both arms, so the finding is if
> anything understated. The same census refutes a cross-agent report of two
> workers per core on a single L3: that is what a **pre-#1729** build does, and
> #1729 (`6e8c31ebd`) and the related width-halving fix #1794 (`0652fdd2e`) are
> both ancestors of main.
> **The wall has since been split.** CPU-seconds attribution on both arms
> (`acc0_w8_w16_cpu_split.py`, identity-checked per cell) finds it is **two
> causes, not one**: at t=16 native burns **~30% more CPU per token** than at
> t=8 *and* leaves **~40% of the sixteen cores idle** (`busy` 0.938 -> 0.595),
> while ORT holds `busy = 0.999` and pays 11%. A first pass scored this
> BURN-DOMINATED with `R_busy = 1.057` — "the workers are not idle" — and that
> reading was an **instrument artifact**: `decode_spmd`'s wait spins and
> `sched_yield`s for up to 500 us before parking, and a yielding thread accrues
> CPU time exactly like a working one, so `busy` at t=16 reads 0.966 at the
> shipped default and 0.595 with the ramp off, at statistically identical
> throughput. **Any `busy`/occupancy reading of the decode pool taken at the
> default blocktime over-reads by tens of points.** The ramp itself costs ~20%
> of process CPU at t=16 on top of an unchanged `user_s`, and removing it is
> throughput-neutral here (ratio 0.9960 against a 5.24% A/A null) -- but this is
> a **zero-gap** decode loop, exactly where parking looks falsely free, so
> nothing here licenses changing the shipped default; that needs #1395's
> gap-aware harness. Next step is unchanged in target and sharper in aim:
> localise the *idle* half with #1859's per-worker straggler attribution.
> **That is now done** (`acc0_w16_worker_split.py`, 10/10 trusted): the idle is
> **not** a wake problem -- `wake_frac` is only 0.051 at t=16 and the
> pre-registered WAKE-BOUND condition did not fire -- it is a **barrier wait**.
> The mean worker spends **22.2%** of the window waiting for a straggler doing
> ~45% more work than the mean and holding **72%** of last-arrivals against a
> 6.7% chance share, while an Amdahl calibration shows 20.4 of the remaining
> points are constant-serial scaling and **not** a defect. So the recoverable
> figure at t=16 is ~25 points, not the 46% residual. The straggler's identity
> *moves between launches*, which excludes a static mis-partition, and
> dispatcher/worker CPU collision was tested and excluded (one partial match in
> four launches). Nothing can help it today: `DEFAULT_STEAL_TILES_PER_WORKER
> = 1` makes `target == total_workers`, so `work_stealing_segments_aligned`
> always falls back to static equal segments. Setting it to 2 measures
> **+23% at t=16** with the predicted mechanism (`sys_frac` 0.280 -> 0.192) and
> **no t=8 regression** -- but it is **REJECTED** by the pre-registered rule
> and not proposed, because the t=16 A/A null in the same run is **+-21.5%**.
> **The binding constraint at t=16 is now the measurement, not the kernel:**
> until the A/A instability is understood no improvement of realistic size can
> clear a pre-registered bar there.
>
> **The dispatcher/worker collision line above is withdrawn as unevidenced.**
> It rested on a probe that sampled `/proc/<pid>/stat` -- the process **main
> thread**, which is not the dispatcher. The dispatcher is a transient thread
> that is usually gone before a bench can report, so its placement can only be
> read from inside the dispatch path; a first attempt from the *reporting*
> thread returned the exactly **inverted** answer, because the reporter is idle
> and the scheduler parks it on the very core the reserve freed. Collision is
> now neither asserted nor excluded.
>
> **What is established is structural: the pool reserves a CPU for the
> dispatcher and binds nothing to it.** `DISPATCHER_RESERVED_CPUS = 1` and
> `reserve_single_group_headroom` keep one allowed CPU clear of workers -- CPU
> 30 at t=16 on this host, justified in-tree by a measured 1.57x -- and the
> dispatcher is then left to the scheduler. Direct measurement confirms the gap
> is real: unpinned, the dispatcher was last seen on a **worker's** core in one
> launch of four. `ONNX_GENAI_CPU_DECODE_DISPATCHER_PIN=1` closes it and
> **fails its own bar**: 15 of 15 trusted launches faster, median **1.0953**
> against a pre-registered 1.10, with no t=8 regression; the companion
> dispersion rule **failed its self-test** and certified nothing. An earlier
> 6-launch run scored 1.1910/ACCEPT and **did not replicate** -- partly small-n,
> partly because #1868's spin-deadline fix already took control `sys_frac` at
> t=16 from 0.257 to 0.198. The mechanism is **unproven**: the unpinned
> dispatcher migrates at most once per launch, so migration is not it. The knob
> ships **off**, and would need prefill in the matrix before it could ship on --
> the dispatcher is the session thread and keeps its affinity after decode ends.
> **The A/A null therefore remains open, and the +23% steal-tiles candidate
> remains blocked behind it.** **A third mechanism has now been tested and
> rejected: transparent-hugepage backing of the weight arena.** THP is
> `[always]` with `defrag=madvise` on this host, so a 2 MB fault that cannot be
> served immediately falls back to 4 KB silently and permanently — a per-process
> lottery that fits the null's shape exactly. It is not what is happening:
> across 12 quiet-host launches `thp_frac` is **0.823–0.928** (range 0.104
> against a pre-registered 0.20) with Spearman rho **-0.19** against a required
> -0.70, while `ms_token` spans 1.725x over the same launches. **83–93% of
> anonymous memory is already hugepage-backed in every launch, including every
> slow one.** A four-launch reconnaissance of the same rule returned a *perfect*
> rho of -1.0000 over a 0.023 range and was refused by the pre-registered range
> guard — the same small-n manufacture that produced the 1.1910 dispatcher-pin
> ACCEPT. **What the run did establish is sharper than the rejection: on a quiet
> host the null is not a spread, it is two modes.** The slow cluster is five
> launches agreeing to **1.05%**, the inter-cluster gap is **11x** the largest
> within-cluster gap, launch order does not predict membership, and the mode
> ratio is **1.687x** — against a 1.23x candidate it is blocking. So the target
> is now a *discrete per-launch configuration*, not a variance to average down,
> and three mechanisms are excluded (dispatcher placement, worker placement,
> page backing). **A second quiet-host run reproduces both modes to within 1%**
> (3.48-3.81 and 5.91-6.03), which makes them a property of the system rather
> than of a run, and it **separates them**: `park_frac`, `sys_frac` and
> `cpu_s_per_token` all overlap across the modes, but *effective lanes* —
> `cpu_s_per_token / ms_token` — does not. **Mode A runs on 15.30-16.10 of the
> sixteen lanes; mode B on 9.76-12.16, with no overlap and a 3.1-lane gap.** The
> slow mode is not burning more CPU; it uses *less* per token and takes 1.55x
> the wall time (**this sentence is wrong and is corrected below** — it
> generalises from one B launch undercutting one A launch; mode B's *median*
> CPU per token is above mode A's). So the question is now "why does a launch
> that builds 15
> workers on 15 verified distinct physical cores run at two-thirds of its width
> for its entire life". **The foreign-load hypothesis has now been tested and
> is REJECTED.** A per-launch `/proc/stat` read of the sixteen pinned CPUs minus
> our own `getrusage` child CPU bounds non-ours time at **0.59 CPU-seconds
> against ~34, a 1.7% ceiling in every launch**, where costing 3.7 of 16 lanes
> needs 23%; Spearman rho is **+0.0210** and the *fastest* launch carries 1.6x
> more foreign time than the slow one. That magnitude bound does not depend on
> sampling the slow mode, which matters because that run drew only 1 slow launch
> in 12 (incidence across three clean runs: 5/12, 6/14, 1/12 — **not a stable
> rate, do not quote it**). **The same run relocates the target:** splitting CPU
> per token into user and system, **user CPU per token spans 11% across both
> modes while wall spans 1.69x**, and the slow launch is **+4.6% user, +170%
> sys**. The work is identical; the lanes are lost to **waiting in
> `worker_wait`'s yield loop** — a persistent straggler *inside* the process,
> with placement, page backing and foreign load all now excluded. Remaining
> candidates: weight-arena memory placement across the two L3/CCX domains, and
> per-launch clock/boost state. Two unblocks this licenses for A/B work, neither
> needing a pool fix: **stratify on an in-launch statistic that cannot see the
> arm** (effective lanes, backed by three runs; `sys_frac` separates 0.315 vs
> 0.140-0.200 but on one slow sample, so treat as hypothesis), and **score
> work-reducing candidates on user CPU per token**, where the null is 15x
> smaller — null-immune but narrow, since a pure load-balance change like the
> +23% steal-tiles candidate moves wall and `sys` while leaving user CPU flat,
> making it that candidate's *control* rather than its score. **The +23%
> candidate has now been re-tested and is CLOSED.** 24 launches with the
> unmodified rule: **ratio 0.9883, sign 38%**; stratified to the fast mode the
> A/A half-width collapses **0.1478 → 0.0323**, an instrument 4.6x sharper that
> would resolve anything above +9.7%, and the effect is **−0.0111**. The
> original +23% is accounted for: its control arm drew the slow mode in 4 of 8
> launches against the test arm's 1 of 8, and at the 1.687x mode ratio that
> imbalance alone manufactures up to **+0.2576** — more than the +0.2327
> observed. Its `sys_frac` "mechanism confirmation" inverts at n=24 (+0.0111,
> 46% sign), because the slow mode is the high-`sys` mode and the mechanism was
> downstream of the same nuisance variable. **A directionally-correct mechanism
> reading corroborates nothing if it is not independent of the confound.**
> Stratification is nonetheless validated as a method here (4.6x sharper null,
> paired-imbalance gate passing at 8.3%) and costs ~3x the launches, since each
> launch runs three independent width-16 processes that each draw the mode. The
> 22.2-point straggler wait remains the open target; what is closed is the claim
> that spare tiles collect it.
>
> **Worker placement is also excluded (2026-08-24).** A probe capturing the
> pinned-CPU set and the mode from the *same* process found both modes on a
> byte-identical 15-worker / 15-physical-core / 8-7-L3 set across 16 trusted
> launches (21 over three runs). A 4.0 ms launch and a 6.0 ms launch are
> placed identically. The probe's first version carried a **selection bias
> that admitted only slow launches** — it sampled `/proc` at a fixed 4.0 s,
> later than a fast launch's entire 1.13 s lifetime, and reported the
> resulting misses as workload failures. Fixed by polling from t=0; the
> defect is written up because the failure mode (an instrument that
> structurally cannot see one arm, while its discards look like noise) is the
> same class as the harness defects already in this ledger. Remaining live
> candidates for the 22.2-point straggler wait: **weight-arena placement
> across the two L3/CCX domains**, and **per-launch clock/boost state**. Since
> a slow worker is slow for reasons of its own rather than holding more work,
> a static redistribution cannot collect it — the evidence points at a
> dynamic, measurement-driven steal. Full records:

> [`docs/benchmarks/2026-08-23-acc0-gap-at-width-16.md`](../benchmarks/2026-08-23-acc0-gap-at-width-16.md),
> [`docs/benchmarks/2026-08-23-acc0-width-16-cpu-attribution.md`](../benchmarks/2026-08-23-acc0-width-16-cpu-attribution.md),
> [`docs/benchmarks/2026-08-23-acc0-width-16-worker-attribution.md`](../benchmarks/2026-08-23-acc0-width-16-worker-attribution.md),
> [`docs/benchmarks/2026-08-24-acc0-dispatcher-placement.md`](../benchmarks/2026-08-24-acc0-dispatcher-placement.md),
> [`docs/benchmarks/2026-08-24-acc0-w16-null-page-backing.md`](../benchmarks/2026-08-24-acc0-w16-null-page-backing.md),
> [`docs/benchmarks/2026-08-24-acc0-steal-tiles-retest.md`](../benchmarks/2026-08-24-acc0-steal-tiles-retest.md),
> [`docs/benchmarks/2026-08-24-acc0-w16-mode-placement.md`](../benchmarks/2026-08-24-acc0-w16-mode-placement.md),
> [`docs/benchmarks/2026-08-24-acc0-lowwidth-smt.md`](../benchmarks/2026-08-24-acc0-lowwidth-smt.md),
> [`docs/benchmarks/2026-08-24-acc0-w16-clock-state.md`](../benchmarks/2026-08-24-acc0-w16-clock-state.md).
>
> **The t=2 `Percent of CPU` anomaly is retired (2026-08-24).** The old
> 98 / 71 / 186 table at widths 1/2/4 had a t=2 cell below one core. Forcing
> two workers onto one physical core was tested directly and costs 1.86x
> throughput while leaving CPU-time at **200.0%** (vs 196.5% on two cores), so
> SMT co-location cannot produce a sub-one-core reading -- it steals throughput
> without stealing CPU-time, and the work-completed ratio 0.550 replicates an
> independent scalar probe's 0.5505. Re-pointing the *retired instrument
> itself* at current main reads **99 / 196 / 372**, with w=1 reproducing
> exactly, so the anomaly was a property of an older tree, not of the
> instrument. Neither the original wake-latency attribution nor the SMT
> hypothesis survives; there is no anomaly left on this tree.
>
> Also recorded there: **decode width `w` builds `w - 1` `onnx-genai-spmd`
> threads and runs the w-th lane on the dispatcher thread**, so any instrument
> counting named worker threads is off by one.
>
> **Clock/boost state is excluded (2026-08-24).** This host has no direct clock
> instrument -- no `cpufreq` sysfs, no vPMU (`perf stat -e cycles` reports
> `<not supported>`), no readable MSR, and `/proc/cpuinfo` `cpu MHz` is a
> constant 2870.7 in all 18 launches of both modes, i.e. nominal. None is
> needed: a clock drop raises CPU-time per token by the *same* factor as wall
> time, unlike SMT contention (wall up, CPU-time unchanged) or parking (wall
> up, CPU-time down). Measured, wall/token ratio is **1.5225** while user
> CPU/token ratio is **1.0250** -- 4.8% of the required inflation. REJECT.
>
> **Correction to the null record's sys reading.** That record reported the
> slow mode as `+4.6% user, +170% sys` and I generalised the mode difference to
> yield-loop time. In this run sys/token is flat to **2.4%** while wall is
> +52%. Both runs are real; different slow launches were drawn. The statement
> that holds in both is: **user CPU per token is flat between modes** (+2.5%
> and +4.6%) while wall moves 1.5-1.7x, and in this run both user and sys are
> flat, so the ~3.3 missing lanes are consuming no CPU at all rather than
> spending longer in the kernel. The sys behaviour is not stable across slow
> launches and is not carried as a mechanism until measured over several.
>
> **The candidate list is empty (2026-08-24).** "Weight-arena placement across
> the two L3/CCX domains" was carried as the last live candidate for two
> records and is **malformed on this host**: `numactl --hardware` reports one
> NUMA node over all 32 CPUs at a single distance, so there is no second memory
> domain to place an arena in, and an L3 is a cache rather than an allocation
> target. I listed it twice without checking the host could express it.
>
> **Mode-stratifying the worker-split instrument reports nothing, and says
> why**: under `ONNX_GENAI_CPU_DECODE_WORKER_PROFILE` the width-16 `wall_s`
> values are one broad distribution (between-mode gap 0.106 s against a
> within-mode spread of 0.292 s), i.e. two clock reads per worker per op
> dissolve the bimodality. The counters that would explain the modes perturb
> them away.
>
> **New lead, not the bimodality:** the aggregate width-16 window spends
> **0.313** in straggler wait, with one worker holding **0.565** of
> `last_arrivals` against a chance share of 0.067 (`straggler_excess` 8.47) and
> `work_skew` **0.562** -- one worker doing ~56% more than the mean, with every
> other worker waiting on it. A direct per-CPU scalar census shows **cpu 0 is
> not degraded** (rel 0.965 inside a 0.965-1.032 band), contradicting an
> external report of a permanent competitor there, so the imbalance is ours.
> Deliberately *not* diagnosed from source: `output_chunk_len_for` returns
> `n.div_ceil(tasks)` and every llama width divides evenly by 16, so static
> reading predicts no skew while measurement says 0.562. That contradiction is
> recorded rather than resolved by argument. Full record:
> [`docs/benchmarks/2026-08-24-acc0-straggler-lead.md`](../benchmarks/2026-08-24-acc0-straggler-lead.md).
>
> **The contradiction is now resolved by measurement, and the source reading
> won (2026-08-24).** The worker instrument already reported `timed_ops` per
> lane and `derive()` was discarding it, taking `workers[0]` as *the* op count
> -- silently assuming the thing in question. Reading all fifteen over 24
> trusted launches gives `ops_spread` = **0.0000 in every launch**: the split
> is exactly even, `output_chunk_len_for` is exonerated, and **the excess is
> execution time on equal work rather than unequal assignment**.
>
> Two further mechanisms are excluded. **Placement**: one lane->cpu map across
> all 24 launches (lane *i* on cpu *2i*, one per physical core -- #1729 working
> as specified), yet the victim moves, with top lane and top cpu concentration
> both 0.208 against a 0.5 bar. **Address layout**: with `setarch -R` holding
> the layout byte-identical the concentration is **0.267, the same number as
> under ASLR**, so layout does not select the victim either. The `setarch -R`
> knob was verified to work before measuring, because #1792 on this project is
> a user-facing placement control that is entirely inert and an inert knob here
> would have manufactured exactly the observed REJECT.
>
> **The straggler survived its own null test.** `work_skew` is a maximum over
> fifteen lanes and so cannot return zero; `straggler_share` has no null model
> either, and both had been read as an imbalance across three records. Scaling
> the window 4x separates a slow lane from max-of-noise: chance predicts the
> excess over 1/15 decaying by 1/sqrt(4) = 0.50, and it instead **rose**,
> R = **1.690**, with one lane last on a median **72% of 3840 ops**. The
> straggler is real, persistent within a process, and worth ~0.31 of the
> width-16 window. What picks the victim at startup is not placement, not
> assignment and not address layout, and is deliberately left unnamed until
> measured. Full record:
> [`docs/benchmarks/2026-08-24-acc0-straggler-identity.md`](../benchmarks/2026-08-24-acc0-straggler-identity.md).
>
> **Post-hoc lead from the same data, recorded as a lead and not a verdict:**
> across 73 launches in three independent experiments the last-arriving lane is
> also the highest-`work_ns` lane in **0.667 / 0.667 / 0.684** of launches
> against a chance share of 0.067 -- so the victim usually computes longer
> rather than merely starting late. That constrains the selector to something
> that makes identical work, on a fixed core, at fixed virtual addresses, take
> longer. **Physical page assignment** is the family that fits and is the next
> hypothesis to test: `setarch -R` fixes virtual addresses while the kernel
> still hands out different physical frames each exec, and the large caches
> here are physically indexed. Not yet tested, and not to be cited as a cause.
>
> **Tested the same session, and rejected.** `prctl(PR_SET_THP_DISABLE)` is the
> only unprivileged lever on this host (pagemap PFNs are masked, the sysfs THP
> control is root-only), and it was verified to work before measuring
> (`AnonHugePages` 262144 kB default, 0 kB under the wrapper). 14 launches per
> arm, interleaved: `work_skew` **0.5329** with 2 MiB backing against **0.5454**
> with 4 KiB backing, ratio **1.023** against a required 0.60. Both arms held
> `ops_spread` = 0.0000 and one lane->cpu map. `work_skew` is scale-invariant,
> so the 1.027x wall cost of disabling THP cannot have manufactured this.
> **Physical page backing is not the selector**, and no candidate replaces it:
> the list is empty. The straggler is real, costs ~0.31 of the width-16 window,
> and is unexplained. What the next probe must satisfy: fixed for a process,
> different between processes, and none of lane index, CPU, virtual layout or
> page size.
>
> **The dynamic decode claim was unreachable in production (2026-08-24).**
> `ONNX_GENAI_CPU_DECODE_SCHEDULE=steal` is documented as selecting a dynamic
> tile claim and was **inert in every default build**: the parse arms
> recognising it were `#[cfg(feature = "mlas")]`, while the path they select --
> an `AtomicUsize` cursor over the tile table on the ordinary native SPMD pool
> -- has no MLAS dependency at all. Only the *optional executor* needs `mlas`,
> and `mlas` is off by default by policy. It survived because the existing
> `work_stealing_*` tests construct `DecodeSchedule::Steal` directly, bypassing
> the parser: the implementation was covered, its reachability was not. Third
> member of the family after #1792 and the latched-`OnceLock` A/B. Un-gated, with
> a test that starts from the env string and asserts parse -> build -> dispatch
> with every output claimed exactly once; the control (gate restored) fails it
> with `left: Fixed`.
>
> **What it is worth, and why the default did not change.** Four interleaved
> arms with an A/A null, 60 launches each, width 16. Every arm is bimodal with
> modes ~2.25x apart, so a pooled median estimates the *mode fraction* rather
> than the effect and the report stratifies per arm. **Inside the slow mode the
> dynamic claim removes ~20% of decode latency** (steal1 +19.44%, steal4
> +20.41%, A/A null **0.0043**), replicated at n=24 (+19.72 / +21.46, A/A
> 0.0223). The fast mode is unresolvable (A/A 0.0327) and the **A/A null on the
> mode-weighted expectation is 0.0655**, so *no end-to-end claim is licensed and
> `decode_schedule_from_raw` still falls through to `Fixed`*. The fix ships on
> contract grounds -- a documented switch that silently does nothing is wrong
> independently of whether turning it on is faster.
>
> Two things this changes for the straggler. The slow mode's excess over the
> fast mode is ~2.01 ms and the claim recovers ~0.74 of it, so **~37% of the
> width-16 slow mode is work-distribution cost that scheduling can recover** --
> the first bite taken out of the straggler after five rejected hypotheses, and
> it lands without identifying the selector. But **both granularities win
> equally**, which undercuts the discrimination the probe was built for: a
> `steal1` win (tiles == workers) is what absorbing a *late-waking* lane looks
> like, and argues against the slow-executor reading that
> `straggler_idx == slowest_idx` (0.667/0.667/0.684) supports. Recorded as a
> tension, not resolved. Full record:
> [`docs/benchmarks/2026-08-24-acc0-steal-selector.md`](../benchmarks/2026-08-24-acc0-steal-selector.md).
>
> **The straggler's selector is unknown after six rejected hypotheses, and the
> seventh cannot be tested with the EP as it stands (#2017).** Lane *i* always
> runs on cpu *2i* and always computes chunk *i*, so "slow lane" (a
> thread/core/hardware property) and "slow chunk" (a data property) predict
> *identical* observations in every dataset collected to date. Both maps are
> static for the life of a process; this is a structural ambiguity, not a
> sample-size problem, and more repetitions cannot resolve it. Both readings
> currently score ~0.208 against a 0.5 bar **only because they are the same
> number**. `ONNX_GENAI_CPU_DECODE_CHUNK_PERMUTATION` (`rotate:<k>` / `seed:<n>`)
> breaks the tie by permuting lane->chunk while holding lane->cpu fixed. Off by
> default, and its default is the *identity function*, so the shipped path is
> byte-for-byte unchanged: the permutation reorders an already-computed segment
> table (alignment is applied to the canonical order and permuted afterwards), it
> stays inside a NUMA node group, and `place_rows` and dispatch read the same
> permuted table so a lane touches and computes the same rows. **No timing claim
> is made and the experiment has not been run** -- the instrument is published
> before its verdict on purpose. Reachability is asserted from the documented env
> string in a child process, with the anti-vacuity assertion that the map must
> *change*, because this is the third knob in a family (#1792 inert affinity, the
> latched-`OnceLock` A/B, and #2014's MLAS-gated `steal`) where the
> implementation was covered and its reachability was not: **a knob is not
> verified until an observable changes when you turn it.** Both negative controls
> fired -- a dead env read, and a corrupted permutation caught by the dispatch's
> own per-row coverage counter. Full record:
> [`docs/benchmarks/2026-08-24-acc0-chunk-permutation-instrument.md`](../benchmarks/2026-08-24-acc0-chunk-permutation-instrument.md).
>
> The old figure was **not mislabelled — it was a correct measurement of a tree
> that no longer exists.** An earlier draft argued this from the ORT arm alone
> (it reproduces to +4.4%, 30.632 → 31.99 ms). **That control is not
> sufficient**: it shows the *ORT* ruler did not move and says nothing about the
> native ruler, which changed repeatedly over the same window. So the inference
> was replaced with a direct A/B — `e9754e7ef`'s bench rebuilt and run beside
> current main's, same host, same environment, `PROBE_REPS=1` on both, arms
> interleaved:
>
> | width | kernel-only, measured | published pair implies | verdict |
> |---:|---:|---:|---|
> | 1 | **1.64x** [1.61–1.88], 12 paired cells | 1.59x | apparent movement **is** kernel |
> | 8 | **1.82x** [1.78–1.89], 6 paired cells | 3.08x | **3.08x retracted** |
>
> **The `t=8` overclaim is worth knowing about.** Both old figures reproduce
> today to within 0.4% — but only **unpinned**. The old bench never called
> `EpFactory::initialize`, so it never ran `bound_process_to_decode_budget()`
> and its process was never confined; that function, physical-core selection
> included, **already existed at `e9754e7ef`**, and only the bench was missing
> the call (added by #1766 `11cb8e5f3`). So the old `t=8` row measured eight
> decode workers scattered over 32 logical CPUs onto SMT siblings — a topology
> **no served session ever ran in**. Pinning the same old binary to eight
> physical cores gives 8.430 ms against its unpinned 14.115, and forcing it onto
> `0-7` gives 16.121: 1.67x of the claimed 3.08x was that, not kernel work.
> Today's binary is pin-insensitive (4.619 unpinned vs 4.664 pinned) because it
> confines itself.
>
> Two corrections that look right and are not, recorded so they are not
> re-applied here: the ~11% warmup/spawn handicap of §27 is in
> **`tokens_s_total`**, and both published figures are **`ms_token`** (the old
> ORT harness's docstring names "the native harness's `steady` column-2 median"
> as its comparand, and the reproductions land on it to 0.4%), so deducting 11%
> yields a number neither tree produces. The asymmetry that *is* real —
> old ORT took `min` over reps, old native was single-shot — biases in **ORT's**
> favour, making the old gap look worse rather than better.
>
> Eight merges landed after `e9754e7ef` published the number: three direct acc0
> kernel work (#1667 broke the serial f32 reduction chain, 5.75x t=1; #1679
> enabled the register-blocked kernel *at acc0*; #1783 folded the zero-point
> unpack), three changing what a width means (#1728, #1794, #1746), and two
> changing the ruler (#1722 made the two arms measure one quantity; #1766 put
> the benches in the production decode topology).
>
> The 2026-08-23 scope note this replaces said the figure was `t=1`-only and
> that the production-width gap was unmeasured. Both were true; both missed
> that the `t=1` number was itself three kernel merges out of date. **A number
> that was right when taken is more durable than one that was wrong** — its
> provenance looks impeccable, so it is never challenged, and it keeps a closed
> problem at the top of the list while the real top item goes unexamined. The
> lesson is not "label the width", it is **re-measure before ranking work off a
> figure you did not take today**.
>
> Still true and unchanged: at t=1 the native side runs `path=flat`, confined by
> the decode budget to a single CPU with no pool built at all (§20), so that row
> compares native-serial against ORT-single-thread and every scaling figure
> quoted against it is "vs serial", not "vs a one-worker pool". **t=16 does not
> resolve, and it is the row that matters** — two of its three cells passed the
> load guard cleanly and read 1.831x and 1.456x (median **1.643x**), but the
> width's A/A null spans 0.969–1.295 (±30%, against 3.6% at t=1 and 2.8% at t=8)
> and native sits in different modes of its 514% launch bimodality across the two
> cells. So it is ~1.6x on an instrument too loose to call it, not "no data" —
> an earlier draft wrote the width off as wholly contaminated, which was wrong in
> the direction that flattered us. **The re-ranking below is conditional on
> that**: a confirmed 1.64x at t=16, the width closest to an unconfined
> production process, would put acc0 back at the top. A dedicated quiet-host
> study of t=16 with launch distributions and a pre-registered A/A threshold is
> the next action. Full record:
> [`docs/benchmarks/2026-08-23-acc0-gap-vs-ort-by-width.md`](../benchmarks/2026-08-23-acc0-gap-vs-ort-by-width.md).
>
> **2026-08-24 — the width-16 straggler is a lane property, not a chunk
> property.** Using the chunk-permutation knob (#2030) and the arm observable
> (#2041), 832 trusted samples across three datasets on a quiet host. The
> pre-registered primary rule returns NEITHER in all three: no single index
> dominates in either frame. But the lane frame is significantly non-uniform in
> every dataset (chi-square p<0.0001, three replications) while the chunk frame
> never is (p=0.78/0.87/0.57) — pooling in the lane frame preserves the
> structure, pooling in the chunk frame destroys it. An arm-set defect found
> mid-analysis (all-even rotations make odd-lane and odd-chunk the same claim)
> was fixed with odd-`k` arms, which settle it: odd lane p=0.00018, odd chunk
> p=0.99989. The victim is core-anchored, drawn from a biased distribution over
> lanes (concentrated in the first 32 MiB L3 and on cpus = 2 mod 4), not fixed
> to one. Within a process it holds a median 74.8% of last-arrivals, 11.2x
> chance. This retires the whole data-property class (cache colouring, page
> interleave, NUMA placement of the weight range) as the selector — the 8th
> rejection and the first to close a class rather than a candidate. Remaining
> limit: placement is one map, so lane and cpu are still the same claim; the
> analogous lane->cpu permutation is the next instrument. Full record:
> [`docs/benchmarks/2026-08-24-acc0-chunk-permutation-instrument.md`](../benchmarks/2026-08-24-acc0-chunk-permutation-instrument.md).

Full record for the acc4 table above:
[`docs/benchmarks/2026-08-21-int4-acc4-execution-regime.md`](../benchmarks/2026-08-21-int4-acc4-execution-regime.md).

### 23. The register-blocked int4 decode kernel shipped default-off and stayed dormant for its whole life (**fixed**)

`accuracy_level = 0` is the production default, and route counters instrumented
from operator entry through the kernel said it never reached the N-blocked
kernel: `entry_bits4=95 percolumn=95 nblock=0 block_simd=129,499,136`.
#1104 built that kernel, measured 1.46x on a 14B model, proved it
byte-identical, and then shipped it behind a **default-off** env toggle "until
the win is measured, exactly like the toggles that preceded it". The
measurement never happened. Nothing asserted which route production took, so a
finished, proven kernel sat unreachable in the tree while the default path took
the per-column loop.

The repair is one line of default (`unwrap_or(false)` -> `unwrap_or(true)`) plus
`acc0_decode_reaches_the_nblocked_kernel_by_default`, which makes the default
route a **checked property** rather than a comment. Merged as `99f105d52`.

**The first attribution I produced was wrong, and the instrument is what was
wrong.** The route probe's `fetch_add` fired 129 million times in the per-column
arm and zero in the N-blocked arm, so the counter inflated the baseline it was
supposed to measure by ~3x and produced a self-consistent 3.14x/4.80x. Rebuilt
without the probe in the timed path:

| route | t=1 | t=4 | vs per-column |
|---|---|---|---|
| per-column (previous default) | 56.528 | 28.518 | 1.00x |
| N-blocked, group of 1 | 55.156 | 27.861 | 1.02x |
| N-blocked, group of 2 | 47.679 | 23.885 | 1.19x |
| N-blocked, group of 4 | 38.114 | 19.238 | **1.48x** |

**The group-of-1 row rules out the explanation the structure suggests.**
Restructuring the reduction — keeping the scale in a vector accumulator and
reducing once per column instead of once per 32-weight block — is worth
**1.02x, i.e. nothing measurable**, even though the per-column path's hreduce
(`extractf128`/`movehl`/`shuffle`, each dependent on the last) sits on the
critical path every four FMAs. The block loop has enough independent work for
the out-of-order engine to hide it. The entire win is the **four-column
activation reuse**: 1.45x from group 1 to group 4, which also matches #1104's
independently measured 1.46x.

**The numerics move the wrong way and are shipped anyway, disclosed.** The
N-blocked kernel's separated correction is **3.59x worse** relatively against
f64 on the worst cell measured (2.422e-5 vs 6.739e-6). It stays within the
pinned envelope, the envelope guard was sized to the *measurement* (8x) rather
than to hope, and the tradeoff is stated rather than buried — a 1.48x decode win
for a 3.59x relative error increase inside an envelope is a defensible trade
only if both numbers are on the table.

Two lessons. An instrument in the timed path is a **measurement error, not
overhead** — distinct from §18's probe, which was real kernel overhead that
broke inlining and whose removal made the shipping kernel genuinely faster; this
one changed nothing in production and only corrupted its own baseline. And
"default off until measured" is a decision that **expires silently** unless a
test asserts the default.

Full record:
[`docs/benchmarks/2026-08-21-int4-acc0-dormant-nblock.md`](../benchmarks/2026-08-21-int4-acc0-dormant-nblock.md).

### 24. The t=8 "wash" is worker-to-CPU placement, not the kernel (**negative result; runtime-owned**)

The premise handed to me was that #1628's int4 acc4 win vanishes at t=8 while
holding at t=1/4/16, and that the kernel should be looked at. **It does not
reproduce.** Measured against explicit pool widths rather than a thread-count
label, the win is flat through width 8 and collapses *after* it:

| pool width | 1 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| speedup | 1.66x | 1.65x | 1.66x | 1.238x | 0.993x |

So there is no t=8 anomaly to tune for. There is a **width-12-and-above**
collapse, and it is not the kernel.

Two hypotheses were tried and discarded before the right one. Memory bandwidth:
a STREAM-style all-thread sweep read 83 GB/s against a 41 GB/s decode draw —
but that figure **does not reconcile with §22**, which measured this host at
31-36 GB/s within a CCX and ~56.6 GB/s across both, and §22's numbers are the
ones this file stands behind. Against those, 41 GB/s is 72% of the across-CCX
ceiling and *above* the within-CCX one, so bandwidth cannot be dismissed by my
83 GB/s sweep and is not dismissed here on that basis. It is ruled out by the
placement A/B below instead, which holds shapes, bytes, thread count and binary
constant and moves **only** which CPUs the workers sit on — a bandwidth ceiling
does not care about that, and the result does. Task grain: the shard count and
barrier time move as expected.

**Root cause: `decode_spmd.rs::node_shards` pins worker *i* to
`allowed_cpus()[i]` in logical order.** On this host SMT siblings are adjacent
(CPUs 0 and 1 are the two threads of core 0), so 16 workers land on CPUs 0-15,
which is **8 physical cores**, and half of them contend for a sibling's
execution units. Verified by reading `/proc/<pid>/task/*/stat`, and confirmed
decisively by comparison: default placement 0.982x versus one-worker-per-
physical-core 1.225x on the same binary and the same shapes.

**No kernel change was made**, which is the point. Filed as **#1680** with the
measurement and the placement evidence, for the runtime owner. Every timing in
§23 and §25 is `taskset`-pinned to even CPUs as a consequence — an unpinned
multi-thread number on this host is measuring the scheduler, not the kernel.

Full record:
[`docs/benchmarks/2026-08-21-decode-worker-cpu-placement.md`](../benchmarks/2026-08-21-decode-worker-cpu-placement.md).

### 25. `f16`/`bf16` decode diverged by *layout*, and the slow side was also the less accurate one (**fixed**)

#1381's dispatch comment claimed the divergence was closed: "both operators,
both stored orders and both 16-bit formats reach the same GEMV backend". Same
**backend**, not same **kernel**. Enumerated with route counters:

| operator | stored order | kernel taken |
|---|---|---|
| `Gemm` transB=1 | `[N,K]` | `gemv_half_nk` |
| `Gemm` transB=0 | `[K,N]` | `gemv_half_kn` |
| `MatMul` | `[K,N]` | `gemv_half_kn` |
| `FusedMatMulBias` | `[K,N]` | **no 16-bit GEMV at all** |

The fourth row was found empirically, not inferred — `count_half_decode_gemv()`
lives in `MatMulKernel::execute_with_backend` and `fused_matmul_bias.rs` calls
the free `matmul_dense_prepacked_into`, so a probe read `matmul_gemv=1
fused_gemv=0`. It matters because the optimizer fuses `MatMul + Add(bias)`. The
MatMul-side transposed row had **no test**: `decode_matmul` only ever built B as
`[k,n]` — the fifth unmeasured-region-behind-a-gate this file records (cf. §11,
§12, §14, §19).

**`[K,N]` crosses a page every `p`.** The stride between consecutive `p` is
`n*2` bytes — 12 KB at n=6144 — and the L2 prefetcher does not cross page
boundaries. Direct kernel A/B on identical bytes, same call, only the stored
order differing:

| shape | `gemv_half_kn` us | `gemv_half_nk` us | penalty |
|---|---|---|---|
| qkv 4096x6144 | 5022 | 1688 | **2.98x** |
| down 14336x4096 | 11015 | 7068 | **1.56x** |

**The zero-memory alternative was tried first and is a negative result.**
`_mm_prefetch` at distance 12 into the strided inner loop made `kn` *worse*
(5580 vs 5022 us on qkv). The stride penalty is not prefetch-recoverable, which
is what justifies paying `2*K*N` bytes for a transpose.

**Accuracy moves the same way, so there is no trade to weigh.** `kn` carries one
serial accumulator per column across the whole contraction; `nk` carries four
combined pairwise. Against an f64 oracle, `nk` is **2.7-9.3x** more accurate.
The first version of that test used `*0.125` operands — exactly representable,
every partial sum exact — and reported zero error and 100% bit-identity. It
could not detect the effect it existed to measure. Hostile data shows ~3%
bit-identity. That is the same failure mode as the instrument in §23: **check
that the measurement can see the effect at all.**

Production-path A/B, two builds, `taskset`-pinned, `steady_ms`:

| dtype | shape | before | after | speedup |
|---|---|---|---|---|
| **f32 (null control)** | attn_out 1024x768 | 0.068 | 0.067 | 1.01x |
| **f32 (null control)** | mlp 4096x11008 | 6.605 | 6.628 | 1.00x |
| f16 | attn_out 1024x768 | 0.063 | 0.028 | **2.25x** |
| f16 | square 2048x2048 | 0.314 | 0.058 | **5.41x** |
| f16 | mlp 4096x11008 | 2.867 | 1.791 | **1.60x** |
| f16 | lm_head 896x151936 | 8.553 | 7.089 | **1.21x** |
| bf16 | attn_out 1024x768 | 0.077 | 0.026 | **2.96x** |
| bf16 | mlp 4096x11008 | 2.782 | 1.719 | **1.62x** |
| bf16 | lm_head 896x151936 | 8.635 | 7.064 | **1.22x** |

**The best row is not the interesting one.** `square` at 5.4x is a shape nobody
runs; the large model-shaped rows (`mlp`, `lm_head`) are 1.2-1.6x, and
`lm_head` — which gains least at 1.21x — is also the row that pays most,
**272 MB** resident for one weight. The small `attn_out` projection sits between
them at 2.25x/2.96x.

**The memory-plan coupling is the part that could have gone badly.**
`node_weight_transpose_cache_bytes` is what `engine/load.rs` budgets against
under #1056, and it was `cfg(macos/ios)` for `MatMul`. On x86 this transpose
would have been **completely invisible to the plan**, which would have
under-budgeted by gigabytes. Any change that makes a kernel retain a
weight-scaled buffer must update that predictor. `FusedMatMulBias` is
deliberately excluded — it has no x86 16-bit GEMV, so budgeting it would
over-reserve every fused projection.

**Two guard tests encoded the opposite decision** ("a transposed variant would
cost a permanent 2*K*N bytes"). They were not deleted: they now assert the
stronger invariant they were reaching for — no **unbudgeted** copy, and never an
f32 widening, checked against the predictor itself.

**Admission is no longer numerically neutral on this path.** Declining the cache
changes *which kernel* runs and therefore output bits. Three contract comments
still claimed neutrality and were corrected; adversarial review caught that the
commit message disclosed it but the comments a maintainer actually reads did
not.

**A real bug, introduced and fixed here.** The f16 transpose cache stores raw
`u16` keyed `(addr, k, n)` with **no dtype** — safe only while one dtype used
it. Routing bf16 through it let a bf16 weight hit an f16 entry left at a
recycled address: `-0.000021640852 != -0.8984375`, reproducible **only in
company**. Guarding the view dtype is not enough; the *key* needs the
discriminator.

**Still divergent, deliberately:** `FusedMatMulBias` takes no 16-bit GEMV on
x86 — 2845 us on qkv against MatMul's 1830 us after this change. It is a
separate mechanism with its own memory-plan consequence and is not folded in
here. Merged as `2e1cfb67c`.

Full record:
[`docs/benchmarks/2026-08-21-half-decode-layout-divergence.md`](../benchmarks/2026-08-21-half-decode-layout-divergence.md).

## 26. `FusedMatMulBias` had no 16-bit decode GEMV — the fused form of a projection was the slow one (#1702)

The optimizer fuses `MatMul + Add(bias)` into `FusedMatMulBias`. On x86 that
fusion was a **pessimisation for every 16-bit weight**: `MatMul` had a 16-bit
decode GEMV and `FusedMatMulBias` did not, so the fused node widened the whole
constant weight to a resident `4 * K * N` f32 copy and ran an f32 GEMV over it.
Applying the optimizer made the model slower, which is the exact inversion its
cost model is supposed to prevent.

Measured on the quiesced host (16 physical cores, `steady_ms`, three reps,
interleaved before/after/`MatMul` in one binary):

| shape | dtype | FMB before | FMB after | gain | `MatMul` | after/`MatMul` |
|---|---|---|---|---|---|---|
| mlp 4096x14336 | f16 | 2.687 | 1.730 | **1.55x** | 1.759 | 0.98x |
| mlp 4096x14336 | bf16 | 2.845 | 1.723 | **1.65x** | 1.709 | 1.01x |
| lm_head 4096x128256 | f16 | 8.595 | 6.862 | **1.25x** | 6.789 | 1.01x |
| lm_head 4096x128256 | bf16 | 8.698 | 6.885 | **1.26x** | 6.784 | 1.01x |
| qkv 4096x4096 | f16 | 1.407 | 0.175 | **8.04x** | 0.171 | 1.02x |
| qkv 4096x4096 | bf16 | 1.427 | 0.152 | **9.39x** | 0.171 | 0.89x |
| attn_out 4096x4096 | f16 | 0.289 | 0.026 | **11.12x** | 0.025 | 1.04x |
| **f32 mlp (null control)** | f32 | 6.551 | 6.538 | **1.00x** | 6.656 | 0.98x |
| **f32 lm_head (null control)** | f32 | 14.896 | 14.840 | **1.00x** | 14.795 | 1.00x |

The `after/MatMul` column is the point: **0.98x–1.04x on every model-shaped
row**. The fix is not a new kernel, it is deleting a divergence — `MatMul`'s
decode arm was extracted into `try_half_decode_gemv` and both operators now
call it, so the fused form is exactly as fast as the unfused one and the
bias is a post-reduction epilogue that never touches summation order.

The f32 rows are a **null control that was built into the workload rather than
added afterwards**: `FusedMatMulBias`'s f32 path was already *faster* than
`MatMul`'s, and it is unchanged at 1.00x. That rules out "the fused operator's
plumbing is slow" as an explanation and localises the entire penalty to the
missing 16-bit route.

### The headline in #1702 was understated, and the reason is worth recording

#1702 reported 1.55x. That is the mlp row and it is correct, but the range is
1.25x–11.12x and the *first* measurement of the small shapes said 1.17x. The
difference was **host load**: the first pass ran while the acc0 matrix was
still executing at load average 20.94. Every number in it was void. This is the
third time in this assignment that a contended host has produced a plausible,
publishable, wrong number, so the rule is now explicit — `uptime` before every
benchmark, and never run two of them at once. The cheap diagnostic that
survives contention is the **cold/steady shape**, not the steady number:
`MatMul` cold 4.673 / steady 0.602 (it builds a transpose) versus pre-fix FMB
cold 1.832 / steady 1.691 (it never does), which identified the missing route
correctly even while the timings were garbage.

### A second defect, found by the guard rather than by review

Routing `FusedMatMulBias` through the shared helper makes it retain the same
`2 * K * N` transpose `MatMul` retains, so `node_weight_transpose_cache_bytes`
owed it a prediction. It had an explicit arm saying the opposite —
"`FusedMatMulBias` has no 16-bit GEMV on this target and retains nothing, so
counting it would over-budget every fused projection" — a comment that was true
when written and false the moment the operators were unified.

Fixing the op name was **not sufficient, and believing it was would have shipped
a dead predictor.** `node_weight_transpose_cache_bytes` opens with a blanket
`if !node.is_default_domain() { return 0 }`, and the optimizer emits
`FusedMatMulBias` into `com.microsoft`. The one node the predictor most needed
to see was the one node it structurally could not. The unit test that caught
this passed when written with `domain = ""` and failed the instant it used the
domain the operator actually ships in; tests for domain-scoped code must use the
shipping domain or they validate a code path that never executes.

### The guard: classification derived from the registry, not from memory

Both defects are the same shape — *an existing, governed cache gains a new
route, and the accounting is not updated*. `weight-cache-guard.yml` cannot see
it: that guard greps for new `OnceLock<..Cache..>` declarations, and no new
cache was declared. A hand-written per-route reconciliation test cannot see it
either, because the missing predictor arm and the missing test are the same
omission — whoever forgets one forgets the other.

The guard that does work derives its obligation from the **real registry**:
`every_registered_cpu_op_is_classified_for_weight_cache_accounting` walks
`OpRegistry::keys()` and fails until every registered op has been classified
`CachesTransposedWeight` or `NoWeightCache(reason)`. Registering a kernel
without deciding what it retains is now a test failure naming the op and the
function to edit. It found 27 unclassified ops on first run.

Three mutations confirm it is not decorative: excluding `FusedMatMulBias` from
the shared arm, restoring the pre-#1702 op-name allowlist, and restoring the
blanket non-default-domain bail each fail at least two of the four tests.
Live reconciliation (`retained_transpose_bytes_are_bytes_the_plan_predicted`)
closes the loop from the executing kernel's side, so the predictor and the
kernel cannot agree on a number neither produces.

### Honest limits (updated after Opus review)

* `node_matmul_dense_cache_bytes` had a second, independent instance of the
  same domain-gate defect #1702 fixed in `node_weight_transpose_cache_bytes`:
  its blanket `!node.is_default_domain()` bail silently zeroed every
  `FusedMatMulBias` node (which always ships in `com.microsoft`), so it was an
  **under**-prediction, not the over-prediction this section originally
  claimed. Fixed by gating each op to the domain it actually ships in
  (`node.op_type == "MatMul" && node.is_default_domain()` /
  `node.op_type == "FusedMatMulBias" && node.domain == "com.microsoft"`), with
  `fused_matmul_bias_dense_cache_is_budgeted_in_its_shipping_domain` guarding
  the regression. With that fixed, the predictor now counts `FusedMatMulBias`
  at `4 * numel * MATMUL_DENSE_DECODE_INSTANTIATIONS` as documented; since the
  decode path no longer fills that cache (it takes `try_half_decode_gemv`
  instead), this is now correctly an **over**-prediction on decode-only
  workloads — the #1056-mandated safe direction — and prefill still fills it,
  so it is left alone deliberately rather than tightened on a guess.
* **The same defect existed twice, which changes what the guard has to be.**
  The transpose-predictor fix and the dense-predictor fix are the identical
  mistake — gate on `is_default_domain()`, ship the op in `com.microsoft` —
  written independently in two functions. The registry-derived guard as first
  built only walked the *transpose* predictor and would not have found the
  second one; it was found by adversarial review, which is not a control that
  scales. `no_predictor_zeroes_a_caching_op_because_of_its_shipping_domain`
  now takes the domain from `OpRegistry::keys()` and asserts **both**
  predictors are non-zero for every caching op in the domain the registry says
  it ships in. A test that hardcodes `domain = ""` passes against a predictor
  that is dead in production; a test that reads the domain from the registry
  cannot. Both mutations (restoring either blanket bail) now fail two tests
  each.
* The `after/MatMul` ratios include two cells at 0.89x and 1.04x-1.09x on the
  smallest shapes, where absolute times are 20-30 microseconds and run-to-run
  noise exceeds the difference. Parity is claimed on the model-shaped rows
  (0.98x-1.01x), not on those.

Full record:
[`docs/benchmarks/2026-08-22-fused-matmul-bias-half-gemv.md`](../benchmarks/2026-08-22-fused-matmul-bias-half-gemv.md).

## 27. The acc0 single-session "gap" was measured with two different rulers (**retraction**, #1712)

The 24-cell acc0 matrix posted to #1679/#1676 divided a native number by an ORT
number that **was not the same statistic**, and the disagreement between the two
definitions was concentrated at exactly `sessions = 1` — the configuration the
whole "single-session gap" conclusion rests on.

### The four biases

| | native (before) | ORT (before) |
|---|---|---|
| denominator | wall included thread spawn + 3 warmup steps | warmup ran before `t0` |
| session start | no barrier; staggered spawn absorbed into `wall` | `threading.Barrier` |
| over repetitions | single shot | `min` (s=1) / `max` (s≥2) — the luckiest run |
| **statistic** | wall-clock aggregate at every `s` | **`1000/median_ms` at s=1, wall-clock aggregate at s≥2** |

The last row is the one that manufactures a result. The baseline switched from a
best-case statistic (a median per-token time, which excludes every straggler) to
a realistic one (wall-clock aggregate, which includes them) **at `sessions = 2`**.
A baseline that does that is guaranteed to look strongest at `sessions = 1`, and
"strongest at `sessions = 1`" is precisely the shape that was reported as *"the
gap is concurrency-dependent; we lose at one session and win at two and four."*

That reading is withdrawn. It is an artifact of the instrument.

The native side was independently penalised: at `tokens = 24`, charging three
warmup steps to the clock means 27 steps of work counted against 24 tokens, a
flat ~11% handicap the ORT arm never paid at any session count.

### What the headline actually was

`qwen t=16 s=1 acc=0` was published as **0.436x**. Under a single definition the
same cell reads **0.70x**, and six independent runs per arm show the ratio cannot
honestly be quoted more precisely than a **range of 0.48x–0.86x**, because the
ORT arm lands in two clusters 1.79x apart:

```
native  190.9 195.6 197.8 200.3 201.5 220.6   unimodal, ±8%
ORT     218.3 229.8 246.2 396.9 414.9 427.7   two clusters, 1.79x apart
```

**The 0.436x figure was never a measurable quantity.** It is `max`-over-reps of
ORT's fast cluster divided by a single-shot native run carrying an 11% warmup
handicap.

### Why the second conclusion is *also* being withheld

The obvious next claim — "ORT is bimodal here, so the anomaly is in the baseline"
— is not supported either, and it is worth recording why, because it was nearly
published. The three slow ORT runs reported intra-run spreads of 67.9%, 4.7% and
12.0%; the three fast ones reported 1.4%, 1.1% and 0.7%. **Elevated intra-run
spread confined to the slow cluster is the signature of external contention, not
of an internal bimodality** — a genuinely bimodal implementation would be stable
in both of its modes. The parsimonious explanation is that the slow cluster is
other load on the host, which was later confirmed to be present.

So this cell has **no published ratio** until both arms are re-run interleaved on
a quiet host.

### The measurement environment was the dominant error term

Mid-investigation the host was found to be running, concurrently: another agent's
full `cargo test`/`llvm-cov` on this same crate (~2470% CPU), **another agent's
run of this same `int4_decode_loop_ab` benchmark** (~1275% CPU), and a stray
`while :; do :; done` spinner. Peak load average 31.25 on a 16-physical-core box.

The same cell — qwen, block 32, acc 0, s=1, 16 decode threads, pinned to even
CPUs — measured **197.2 tok/s** in one window and **22.8 tok/s** in another. An
8.6x swing, from the environment alone.

The trap worth naming: **several of those contaminated runs reported an intra-run
`spread_%` under 6%.** A tight spread means the contention was *steady* during
the run, not that the host was idle. Intra-run spread is not a contention
detector, and treating it as one is how a 8.6x environmental artifact acquires a
credible-looking error bar. `acc0_gap_matrix.py` therefore refuses to start a
cell while any other process is above 150% CPU, and marks the cell `UNTRUSTED`
rather than silently proceeding if it cannot wait one out.

### What survives contention, and a claim withdrawn on review

Absolute throughput on this host is not stable, so the instinct was to fall back
on a **ratio between the two arms measured back-to-back in the same window** on
the grounds that both arms eat the same contention. Interleaved native / ORT /
native, native A/A partner at **1.018**:

| arm | tok/s | achieved GB/s on the same 145.7 MB/token footprint |
|---|---|---|
| native acc0 | 196.0 | 28.5 |
| ORT | 279.5 | 40.7 |

That was originally written up as a "decisive observation" that ORT reaches a
bandwidth we do not, and that this direction was stable even if the multiplier
was not. **Opus review rejected that, correctly, and it is withdrawn.** The
refutation is in this document's own bistability data below:

* Pair 8 has native at **264.9** against ORT at **255.0** — native out-bandwidths
  ORT in the same shared window. So the direction is not universal.
* Native's own fast placement, **335.6 tok/s = 48.9 GB/s**, is *higher* than the
  40.7 GB/s ORT figure quoted above as decisive.

The A/A of 1.018 does not rescue it. That partner controls native-vs-native
placement stability across two native launches; it says nothing about which L3
placement the separately-launched **ORT** process drew. Quoting native at 28.5
GB/s against ORT at 40.7 GB/s compares native's slow placement with ORT's fast
one, and then calls the difference stable. That is precisely the objection used
two subsections above to withhold the "ORT is intrinsically bimodal" and "~14% of
FMA peak" claims — a placement-depressed number is a **lower bound**, and I
applied that standard to two claims and then failed to apply it to a third.

**What the evidence actually supports**, stated at the strength it can carry:

* In some shared windows ORT out-bandwidths native; in at least one, native
  out-bandwidths ORT. Both arms are placement-bistable per process launch.
* **We are not memory-bound, and this conclusion is now *stronger*, not weaker.**
  Native's own fast mode demonstrates the memory system supplies at least 48.9
  GB/s to *our* kernel. So our common-case 28.5 GB/s is not a hardware ceiling —
  it is something we are leaving on the table. That is a better-founded statement
  than the retracted one, and it does not depend on the ORT arm at all.

### The MLP-starvation hypothesis is provisionally falsified

The leading explanation for a single-session deficit was memory-level-parallelism
starvation: one session cannot keep enough cache misses outstanding, while two or
four supply independent streams that do. That predicts aggregate bandwidth rising
with session count.

Measured across `s = 1, 2, 4, 8` it does not rise — it stays flat at ~22–28 GB/s
aggregate. Marked **provisional** because the sweep was taken in a contended
window, but the shape is a within-sweep comparison and the prediction is
directional, so a real MLP effect should still have been visible.

If it holds, it has a second consequence worth stating plainly: **the `s=2`/`s=4`
"wins" were never us scaling.** Our absolute throughput is flat in session count,
so those cells were the baseline degrading, not the kernel improving — and no
kernel change should be justified by them.

### The mechanism: the acc0 kernel is per-*block* bound, not per-*nibble* bound

This is the one result of this segment that is both quantitative and robust. It
is a **within-sweep comparison of a single arm**, all four points taken
back-to-back in one quiet window, so it does not depend on the ORT baseline or
on absolute throughput being stable.

Sweeping `block_size` varies how often the per-block epilogue runs while leaving
the number of nibbles to unpack essentially unchanged. qwen, `t=16`, `s=1`,
`accuracy_level = 0`:

| block | tok/s | spread | bytes/token | achieved GB/s |
|---|---|---|---|---|
| 16 | 93.3 | 8.7% | 174.8 MB | 16.3 |
| 32 | 195.2 | 7.0% | 145.7 MB | **28.4** |
| 64 | 428.9 | 16.3% | 131.1 MB | **56.2** |
| 128 | 352.8 | 57.6% | 123.8 MB | (spread too high to quote) |

Single launch per row, so the *magnitudes* below are superseded by the paired
measurement that follows; the table is kept because the monotone direction and
the block-16 scalar cliff are both real.

That single-shot sweep suggested ~2x per doubling. **It overstated the effect**,
because block 64 happened to draw a fast placement (see the bistability section);
this is precisely the trap this section is otherwise about. Re-measured properly
as six interleaved, independently launched pairs:

| pair | block 32 | block 64 | ratio |
|---|---|---|---|
| 1 | 190.7 | 307.0 | 1.61x |
| 2 | 195.5 | 283.3 | 1.45x |
| 3 | 204.0 | 281.9 | 1.38x |
| 4 | 203.9 | 298.0 | 1.46x |
| 5 | 192.5 | 278.0 | 1.44x |
| 6 | 199.2 | 303.4 | 1.52x |
| **median** | **197.4** | **290.7** | **1.47x** |

**1.47x, six pairs out of six, with no overlap between the two distributions.**
That is the load-bearing number; the 2.20x from the single sweep is withdrawn.

Doubling the block size moves 1.11x less traffic and does 1.47x more work, and
the nibble count is identical in both rows. A per-nibble cost cannot produce
that. **The dominant cost is per-block work.**

Worth noting the model this *fails* to match, since it bounds how well the
mechanism is understood: counting instructions in the loop below predicts only
~1.14x for 32→64 (block 64 amortises one epilogue over two chunks instead of
one). The measured 1.47x is larger, so per-block cost is not just the visible
instruction count — the branchy `BorrowedScales::get` discriminant test, the
bounds-checked indexing, and the dependency chain through `blk[c]` into `acc[c]`
are all per-block too. The *direction* is established; the full decomposition is
not, and should not be claimed until a prototype confirms it.

Block 16 is a separate effect and is not evidence for anything here: the
dispatch gate at `matmul_nbits.rs:1568` requires `block_size.is_multiple_of(32)`,
so block 16 **never enters `borrowed_int4_nblock4_avx2` at all** and is routed to
`borrowed_affine_int4_matmul` instead. It is slow because it is not taking this
kernel, not because of anything inside it. (An earlier draft of this section
blamed `chunks = block_size / 32` evaluating to zero; that line is unreachable
for block 16 and the attribution was wrong, though the exclusion is right.)

The code says the same thing. In `borrowed_int4_nblock4_avx2` the inner chunk
loop runs `block_size / 32` times, so **at block 32 it runs exactly once**, and
each block then pays a full epilogue:

```rust
for c in 0..group {
    let scale = scales.get(scale_bases[c] + block);
    let zero_point = layout.zero_point(zp_rows[c], block) as f32;
    acc[c] = _mm256_fmadd_ps(blk[c], _mm256_set1_ps(scale), acc[c]);
    correction[c] += scale * zero_point * activation_sums[block];
}
```

At block 32 that epilogue — a bounds-checked scale fetch, a branchy
`zero_point` lookup, a broadcast, an FMA, and three scalar float ops — runs
**once per four FMAs of useful work**, per column. At block 64 it amortises over
eight, at block 128 over sixteen. That is exactly the observed curve.

Two further notes:

* The benchmark passes **no zero-point tensor**, so `zp_rows[c]` is `None` and
  `layout.zero_point` returns a constant. It is still being called per block per
  column, inside the hot loop. That is pure overhead on the default production
  path, with no numerical consequence to hoisting it.
* `correction[c] += scale * zero_point * activation_sums[block]` is a scalar
  chain across blocks. With `zp` constant it is `zp * dot(scales, activation_sums)`
  — a reduction that could be computed 8-wide in a separate pass. That one is
  **not** numerically free: f32 addition is not associative, so it must be
  weighed against the acc0 exactness contract rather than assumed.

This is §22's finding ("per-block bookkeeping was 1.68x of the int4 acc4 decode
kernel — at block 32, at low thread counts") reappearing in the **acc0** kernel,
where it was never fixed. It supersedes the unpack/convert hypothesis stated
earlier in this section, which this sweep **falsifies**: if the nibble unpack
dominated, block size would barely move the number.

### The host is bistable, and a cpuset mask is not enough

Eight interleaved ORT/native pairs on a quiet host:

```
ORT     338.6  421.8  442.0  428.6  440.1  223.0  293.5  255.0
native  198.3  196.0  240.9  335.6  200.3  197.8  197.6  264.9
```

Both arms are bistable, and the tell is that the runs reporting *low* intra-run
spread cluster at each arm's fast mode (ORT 442.0/428.6/440.1 at 0.4%/3.5%/3.0%;
native 198.3/200.3/197.8/197.6 at 16%/0.2%/2.4%/4.0%). A run is either placed
well and then internally stable, or placed badly and then also noisy.

This is thread-to-L3-domain placement at process start. `taskset -c 0,2,...,30`
constrains the pool to 16 distinct physical cores spanning **both** 32 MiB L3
instances, but it does not determine which thread lands in which domain, and the
weight set (145.7 MB) is far larger than either. So the earlier attribution of
the slow cluster to "external contention" was itself incomplete: contention was
real and was the dominant error term, but a second bistability survives on an
idle host.

**Consequence for all future numbers in this document: a cpuset mask is not
sufficient pinning.** Per-thread placement has to be fixed, or every cell has to
be reported as a distribution over independent process launches. Single-run
ratios on this host are not reproducible, whichever arm they favour.

### Disposition

* Harness unified and the definition written into the module docs as a table, so
  the two arms cannot silently drift apart again.
* Both arms print `spread_%`; the driver blocks on a quiet host and labels
  untrusted cells.
* The 24-cell matrix, the 0.436x headline, and the "concurrency-dependent gap"
  reading are **withdrawn**. A quiet-host re-measurement under the unified
  definition puts `qwen t=16 s=1` at **0.625x** (A/A 1.042), but see the
  bistability section: even that is one draw from a distribution, and the
  matching `llama t=16 s=1` cell is **rejected by its own A/A of 0.796**, whose
  20% self-noise is comparable to the 24% effect it was supposed to support.
* We win `s=2`/`s=4` decisively (qwen 1.86x/2.18x, llama 2.27x/2.40x, effects far
  larger than their A/A noise) — but our *absolute* throughput falls with session
  count (qwen 272.7 → 192.2 → 152.9). Those cells are the baseline collapsing
  (436.2 → 103.1 → 70.0), not us scaling, and must not be cited as kernel wins.
* **Next mechanism, now identified rather than guessed: amortise the per-block
  epilogue in `borrowed_int4_nblock4_avx2`.** The block sweep shows the kernel
  reaches the machine's bandwidth ceiling at block 64 and half of it at the
  production block 32, and the code shows why — at block 32 the chunk loop runs
  once, so every four FMAs pay a scale fetch, a branchy `zero_point` lookup, a
  broadcast, an FMA and three scalar ops. Hoisting the `zp_rows[c] == None` case
  out of the block loop is numerically free and is the first thing to do;
  vectorising the `correction` reduction is not free and must be weighed against
  the acc0 exactness contract.
* The earlier unpack/convert hypothesis in this section is **falsified** by that
  same sweep and should not be picked up by the next reader.
* The dispatch-width question was handed to the runtime owner with CPU-time
  evidence (total CPU-seconds flat across widths, `sys` time rising ~20x from
  `t<=2` to `t>=4`) rather than tuned around in the kernel. The `sys` rise is
  real and is the `worker_wait` yield ramp (a `sched_yield` syscall per
  iteration for the remainder of the blocktime window); it remains open with the
  runtime owner.
  **Superseded in part (2026-08-22, #1771): the mechanism stands, the width and
  the magnitude do not.** That `sys` column was taken pre-#1766 on an unbounded
  process, and the shape it actually had was the *opposite* of what this bullet
  records: `sys` was **highest at `t=2`** (2.19s) and **decreased** with width.
  On post-#1766 main the onset is at **`t=16`**, not `t>=4`, and it is an order
  of magnitude smaller (1.14s). The `yield_now`-per-iteration ramp is still
  there in the source, and the runtime owner's blocktime `500us -> 0` A/B
  (`sys` 28.1s -> 5.1s, latency-neutral) is a genuine within-configuration
  result — but it was measured with the pool oversubscribed against a
  full-width Rayon pool, which inflates barrier time on its own. Treat the cost
  as unquantified until the blocktime A/B is re-run on bounded topology. The
  "~20x jump" figure is mine and should not be re-quoted.
* The accompanying claim that `ONNX_GENAI_CPU_DECODE_THREADS=2` is a knob that
  "silently does nothing" is **withdrawn (2026-08-22)**. The acc4 regime document
  recorded `=2` as identical to `=1` (23.529 vs 23.527 ms/token) and that
  reproduced here (23.6 vs 23.6 tok/s), but a controlled re-measurement on a
  quiet host — one process per launch, interleaved arms, per-rep load guard,
  over-guard cells discarded — gives **20.447 ms/token at `=2` against 40.039 at
  `=1`, a 1.96x speedup, with a 0.6% A/A null**. Per-thread attribution shows
  both `=2` workers **99% busy**, not parked. The "71% of one core" figure came
  from `/usr/bin/time`'s `Percent of CPU`, which is `(user+sys)/wall` and so is
  wall-derived: under contention it degrades exactly like the wall time it was
  being used to corroborate, rather than independently confirming it. **There is
  a second, independent reason not to read it as utilisation on an SMT host**,
  found 2026-08-24 while checking a cross-agent report of a permanent competitor
  on cpu 0: a logical CPU whose sibling is busy is granted a full 100% *share*
  while delivering roughly half the *work*, and no CPU-time instrument can see
  that — the scheduler really is handing over the CPU; the contention is in
  hardware, below its view. Only a work-completed probe distinguishes them
  (`benches/cpu_work_probe.py`: iterations against `CLOCK_THREAD_CPUTIME_ID`). The
  competitor itself did **not** reproduce on a quiet host — cpu 0 reads
  `cpu_share` 0.999–1.000 at 9429/9482/9489 iterations, inside the 8744–9499
  band spanned by ten other CPUs, with one transient outlier that two re-probes
  cleared — so on a host shared by several agents, "permanent" was load rather
  than topology. The instrument point is the durable part. The
  strongest clue that this was harness-side was arithmetic — 23.529 vs 23.527
  agree to four significant figures, and contention is random. Full curve,
  method and the categorical non-vacuity check:
  [2026-08-22-decode-width-scaling.md](../benchmarks/2026-08-22-decode-width-scaling.md).
  **Root cause identified (2026-08-22, #1771), and it is mechanical rather than
  statistical.** Those rows predate **#1766**: the benchmark process never
  called `CpuExecutionProvider::initialize()`, so the decode budget sized the
  SPMD pool without *bounding the process*, and the global Rayon pool ran full
  width at every budget. The `t=1` arm was therefore never 1-wide, which is why
  the bottom of the sweep read flat. An independent falsifier against
  `9747b4971` (main minus #1766) reproduces the correction here from the other
  direction: **1.98x at `t=2` and 187% CPU**, two lanes genuinely computing.
  This supersedes "contention" as the explanation — contention was a real
  confound in that window, but it is not what produced the flat line.
* The general point survives intact and is worth keeping: a knob that silently
  does nothing is worse than a slow one, because every sweep through it prints a
  flat line that reads as "this kernel does not scale". The fix is to verify a
  knob **structurally** (`w` in must give `w` SPMD threads out — categorical, so
  valid even on a loaded host) rather than by comparing timings.
  **Now enforced (#1747, `b9d9c48`).**
  `every_benchmarked_decode_width_realizes_the_worker_count_it_requests` sweeps
  every published width in its own child process and asserts the lanes the pool
  *built* equal the width requested, so the label can no longer drift from the
  run. It is mutation-proved against production: reverting #1748's dispatcher
  lane fails it with `requested width 4 realized 3 compute lanes`. Rows also
  now print `requested=/realized=/path=` inline (#1764, #1770), so a reader can
  check a width without re-deriving it.
* **Benchmarking on this host now requires coordination.** Three agents share
  16 physical cores; two were running heavy jobs on this crate during this
  investigation, one of them the very same benchmark binary. Cross-agent notice
  before a long run is not politeness, it is a correctness requirement for
  anyone publishing a ratio.
