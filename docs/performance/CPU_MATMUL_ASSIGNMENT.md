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
| `MatMulNBits` | 8-bit, block 32 | 128 | 3584 | 2 | **0.90** | 0.91 | **win** |
| `MatMulNBits` | 8-bit, block 32 | 128 | 3584 | 4 | **0.87** | 0.87 | **win** |
| `MatMulNBits` | 8-bit, block 32 | 128 | 3584 | 8 | **0.94** | 1.06 | **win** |
| `MatMulNBits` | 8-bit, block 32 | 256 | 3584 | 2 | 1.17 | 1.18 | gap (static M) |
| `MatMulNBits` | 8-bit, block 32 | 256 | 3584 | 4 | 1.15 | 1.16 | gap (static M) |
| `MatMulNBits` | 8-bit, block 32 | 256 | 3584 | 8 | 0.99 | 1.05 | gap (static M) |
| `MatMulNBits` | 8-bit, block 32 | 512 | 3584 | 2 | 1.41 | 1.42 | gap (static M) |
| `MatMulNBits` | 8-bit, block 32 | 512 | 3584 | 4 | 1.39 | 1.39 | gap (static M) |
| `MatMulNBits` | 8-bit, block 32 | 512 | 3584 | 8 | 1.25 | 1.32 | gap (static M) |

### The 8-bit win is bounded by row count

The 8-bit keep is **not** unconditional in `M`. The win erodes as rows are added and crosses over
between 128 and 256 -- and the crossover rows are the *low*-dispersion measurements in this whole
document (spread 0.01-0.04), so it is not a contended-host artefact.

A node whose row count is **statically** >= 256 is pure prefill: there is no decode traffic on it to
amortise the loss against, so this is the shape where the gap is felt undiluted, and the one to
optimize first.

A **dynamic** row count is the LLM case, where a single node serves both phases. There the loss is
already repaid in practice: at 8 threads decode saves **6.06 ms per token** (1.78 ms ours vs 7.84 ms
ORT) while a 512-row prefill costs **7.69 ms once** (38.58 vs 30.89). The *second* generated token
has repaid the entire prefill loss.

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

### 3. Per-call packing — `QLinearMatMul` (**fixed**)

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
