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
> default native build the integer path is the widened-`i32` scalar loop, which measures **11.8x
> (M=1) / 12.0x (M=128)** against ORT at `1x2048x2048`, one thread. Section 3's "fixed" applies to
> the MLAS route only; a native QGEMM is open work.

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

```sh
export ORT_ROOT=<ort-prebuilt>            # the ort-sys download under target/
cargo build --release -p onnx-genai-bench --bin bench_prec --features mlas
LD_LIBRARY_PATH=$ORT_ROOT/lib ./target/release/bench_prec \
  --model <model>.onnx --runs 11 --warmups 4 \
  --native-threads 8 --ort-intra-threads 8
```

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
is byte-for-byte what it was. `m = 1` cannot reach any of this (the lowest gate of any route is 4) and is
unchanged. And the route is still behind ORT at small `m`, for the reason section 5 already gives:
MLAS SQNBit's CompInt8 path uses integer dot products where this one widens every nibble to f32,
and VNNI is what would make that pay.

Full record: [`docs/benchmarks/2026-08-19-int4-prefill-dequant-pack-simd.md`](../benchmarks/2026-08-19-int4-prefill-dequant-pack-simd.md).
