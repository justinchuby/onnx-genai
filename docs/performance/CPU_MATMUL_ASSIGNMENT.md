# CPU matmul assignment matrix vs ONNX Runtime

Which of `MatMul` / `Gemm` / `MatMulNBits` / `QLinearMatMul` this EP should take from a plugin host,
and which it should leave to ONNX Runtime's own CPU kernels.

The policy is **encoded**, not just documented: `crates/onnx-runtime-ep-cpu/src/assignment_policy.rs`
is the source of truth and this file is its evidence. Every row below is asserted by a test in that
module.

## Rule

A range is claimed only if it has a **>= 5% repeatable win beyond noise at every measured thread
count**, or a correctness reason (the host has no kernel at all). Losing ranges, and ranges we have
not measured but can size, defer to the host. Unmeasured-and-unsizable ranges keep the historical
claim, because deferring on no evidence is a guess in the other direction.

Deferral is only safe when the host actually has a kernel. Under
`session.disable_cpu_ep_fallback=1` it does not, and the plugin switches this gate off entirely
(`onnx-runtime-ep-plugin`'s `ExportedEp::host_fallback_available`). **The native-only runtime is
unaffected**: `supports_op` still answers "yes" for everything implemented, so declining to
*advertise* a kernel never makes it unreachable when this crate is the whole runtime. A deferred
range is therefore never advertised as faster, but it is still there when there is nothing else.

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

| op | dtype / bits | M | K, N | threads | p50 | p90 | assignment |
|---|---|---|---|---|---|---|---|
| `MatMulNBits` | 8-bit, block 32 | 1 | 3584 | 1 | **0.15** | 0.15 | **claim** |
| `MatMulNBits` | 8-bit, block 32 | 1 | 3584 | 2 | **0.36** | 0.36 | **claim** |
| `MatMulNBits` | 8-bit, block 32 | 1 | 3584 | 4 | **0.25** | 0.32 | **claim** |
| `MatMulNBits` | 8-bit, block 32 | 1 | 3584 | 8 | **0.25** | 0.30 | **claim** |
| `MatMulNBits` | 8-bit, block 32 | 1 | 3584 | 16 | **0.23** | 0.23 | **claim** |
| `MatMulNBits` | 8-bit, block 32 | 1 | 4096 | 8 | **0.23** | 0.24 | **claim** |
| `MatMulNBits` | 8-bit, block 32 | 128 | 3584 | 2 | **0.90** | 0.91 | **claim** |
| `MatMulNBits` | 8-bit, block 32 | 128 | 3584 | 4 | **0.87** | 0.87 | **claim** |
| `MatMulNBits` | 8-bit, block 32 | 128 | 3584 | 8 | **0.94** | 1.06 | **claim** |
| `MatMulNBits` | 8-bit, block 32 | 256 | 3584 | 2 | 1.17 | 1.18 | defer (static M) |
| `MatMulNBits` | 8-bit, block 32 | 256 | 3584 | 4 | 1.15 | 1.16 | defer (static M) |
| `MatMulNBits` | 8-bit, block 32 | 256 | 3584 | 8 | 0.99 | 1.05 | defer (static M) |
| `MatMulNBits` | 8-bit, block 32 | 512 | 3584 | 2 | 1.41 | 1.42 | defer (static M) |
| `MatMulNBits` | 8-bit, block 32 | 512 | 3584 | 4 | 1.39 | 1.39 | defer (static M) |
| `MatMulNBits` | 8-bit, block 32 | 512 | 3584 | 8 | 1.25 | 1.32 | defer (static M) |

### The 8-bit win is bounded by row count

The 8-bit keep is **not** unconditional in `M`. The win erodes as rows are added and crosses over
between 128 and 256 -- and the crossover rows are the *low*-dispersion measurements in this whole
document (spread 0.01-0.04), so it is not a contended-host artefact.

A node whose row count is **statically** >= 256 is pure prefill: there is no decode traffic on it to
amortise the loss against, so it defers to the host.

A **dynamic** row count is the LLM case, where a single node serves both phases and the choice
cannot be split. Claiming is correct there by a wide margin: at 8 threads decode saves **6.06 ms per
token** (1.78 ms ours vs 7.84 ms ORT) while a 512-row prefill costs **7.69 ms once** (38.58 vs
30.89). The *second* generated token has already repaid the entire prefill loss.

### Block sizes ORT cannot build stay claimed

This EP accepts any power-of-two `block_size >= 16`; ORT's CPU `MatMulNBits` `ORT_ENFORCE`s
`block_size` in {16, 32, 64, 128, 256}, and that check throws at *kernel construction*. Deferring a
512-wide block would therefore turn a working session into a load failure, so those keep the claim
regardless of the measured ratio -- the same safety rule the dense arm applies to dtypes ORT has no
kernel for.

This gate is evaluated **before every** `MatMulNBits` deferral, not only the 4-bit one. An 8-bit
node that is statically wide *and* carries a block size ORT cannot build would otherwise escape
through the row gate above and fail `ORT_ENFORCE` on the host;
`wide_eight_bit_with_a_block_size_ort_rejects_stays_claimed` pins that ordering. `bits` needs no
equivalent guard -- both runtimes accept exactly {2, 4, 8}.

### 2-bit is deferred by extrapolation, and says so

Only 4-bit and 8-bit `MatMulNBits` were measured. 2-bit is a valid contrib value that shares the
dequant-then-GEMM structure and the threadpool with 4-bit, so it defers -- but its deferral reason
states that it is extrapolated rather than borrowing 4-bit's numbers.

### Rows are folded, not read from one dimension

The row count of `[.., M, K]` is the product of every dimension but the last, so a statically
batched `[4, 100, 3584]` is 400 rows and lands in the wide-prefill region even though no single
dimension reaches 256. Any symbolic dimension anywhere in the batch makes the whole count unknown,
which is the decode case and stays claimed.
| `MatMulNBits` | 4-bit, acc 0 | 1 | 3584 | 1 | 1.00 | 1.03 | defer |
| `MatMulNBits` | 4-bit, acc 0 | 1 | 3584 | 2 | 1.52 | 1.56 | defer |
| `MatMulNBits` | 4-bit, acc 0 | 1 | 3584 | 4 | 1.74 | 1.82 | defer |
| `MatMulNBits` | 4-bit, acc 0 | 1 | 3584 | 8 | 2.23 | 2.31 | defer |
| `MatMulNBits` | 4-bit, acc 0 | 1 | 3584 | 16 | 2.21 | 3.28 | defer |
| `MatMulNBits` | 4-bit, acc 0 | 1 | 3584 | 32 | 4.28 | 4.71 | defer |
| `MatMulNBits` | 4-bit, acc 0 | 128 | 3584 | 1 | 0.99 | 1.07 | defer |
| `MatMulNBits` | 4-bit, acc 0 | 128 | 3584 | 4 | 2.35 | 2.63 | defer |
| `MatMulNBits` | 4-bit, acc 0 | 128 | 3584 | 8 | 2.41 | 2.63 | defer |
| `MatMulNBits` | 4-bit, acc 0 | 128 | 3584 | 32 | 3.79 | 4.36 | defer |
| `MatMulNBits` | 4-bit, acc 4 | 1 | 3584 | 8 | 1.78 | 1.97 | defer |
| `MatMulNBits` | 4-bit, acc 4 | 128 | 3584 | 8 | 2.11 | 2.37 | defer |
| `MatMul` | f32 | 1 | 3584 | 1 | 1.00 | 1.00 | defer |
| `MatMul` | f32 | 1 | 3584 | 8 | 2.52 | 4.51 | defer (noisy) |
| `MatMul` | f32 | 1 | 3584 | 32 | 0.57 | 0.71 | defer |
| `MatMul` | f32 | 128 | 3584 | 1 | 0.97 | 1.03 | defer |
| `MatMul` | f32 | 128 | 3584 | 2 | 2.11 | 2.35 | defer |
| `MatMul` | f32 | 128 | 3584 | 4 | 1.77 | 1.79 | defer |
| `MatMul` | f32 | 128 | 3584 | 8 | 1.38 | 2.49 | defer |
| `MatMul` | f32 | 128 | 3584 | 16 | 1.65 | 1.73 | defer |
| `MatMul` | f32 | 128 | 3584 | 32 | 0.67 | 0.88 | defer |
| `MatMul` | f16 | 1 | 3584 | 8 | 2.04 | 2.16 | defer |
| `MatMul` | f16 | 128 | 3584 | 1 | 2.47 | 2.48 | defer |
| `MatMul` | f16 | 128 | 3584 | 2 | 5.38 | 5.43 | defer |
| `MatMul` | f16 | 128 | 3584 | 4 | 6.57 | 6.78 | defer |
| `MatMul` | f16 | 128 | 3584 | 8 | 7.77 | 8.77 | defer (noisy) |
| `MatMul` | f16 | 128 | 3584 | 16 | 7.10 | 7.46 | defer |
| `MatMul` | f16 | 128 | 3584 | 32 | 5.34 | 6.24 | defer |
| `QLinearMatMul` | u8 x u8 | 1 | 3584 | 1 | 1.13 | 1.18 | defer |
| `QLinearMatMul` | u8 x u8 | 1 | 3584 | 8 | 2.33 | 2.77 | defer |
| `QLinearMatMul` | u8 x u8 | 1 | 3584 | 16 | 2.34 | 2.58 | defer |
| `QLinearMatMul` | u8 x u8 | 128 | 3584 | 1 | 1.20 | 1.21 | defer |
| `QLinearMatMul` | u8 x u8 | 128 | 3584 | 8 | 2.43 | 2.80 | defer |
| `QLinearMatMul` | u8 x u8 | 128 | 3584 | 16 | 2.65 | 3.42 | defer |
| `QLinearMatMul` | u8 x u8 | 512 | 3584 | 1 | 1.20 | 1.21 | defer |
| `QLinearMatMul` | u8 x u8 | 512 | 3584 | 8 | 2.12 | 2.22 | defer |
| `QLinearMatMul` | u8 x u8 | 512 | 3584 | 16 | 2.08 | 2.18 | defer |
| `QLinearMatMul` | i8 x i8 | 1 | 3584 | 1 | **0.03** | 0.03 | **claim** |
| `QLinearMatMul` | i8 x i8 | 1 | 3584 | 8 | **0.09** | 0.09 | **claim** |
| `QLinearMatMul` | i8 x i8 | 1 | 3584 | 16 | **0.10** | 0.10 | **claim** |
| `QLinearMatMul` | i8 x i8 | 128 | 3584 | 1 | **0.25** | 0.25 | **claim** |
| `QLinearMatMul` | i8 x i8 | 128 | 3584 | 8 | **0.47** | 0.53 | **claim** |
| `QLinearMatMul` | i8 x i8 | 128 | 3584 | 16 | **0.60** | 0.69 | **claim** |
| `QLinearMatMul` | i8 x i8 | 512 | 3584 | 1 | **0.26** | 0.26 | **claim** |
| `QLinearMatMul` | i8 x i8 | 512 | 3584 | 8 | **0.42** | 0.43 | **claim** |
| `QLinearMatMul` | i8 x i8 | 512 | 3584 | 16 | **0.42** | 0.47 | **claim** |

Ranges **outside** the measured region keep the historical claim, and the code says so explicitly:
`K * N < 2^20`, symbolic/dynamic weight shapes, and dense dtypes other than f32/f16 (where a
deferral could be a load failure rather than a slow session, because ORT's CPU EP may have no kernel
at all).

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

Because the thread count is **not visible at capability time**, a claim would have to hold at every
count, and none of these do.

### 2. A kernel gap — f16 dense

f16 is the only dense range that loses at **one** thread (2.47x), so it is not the scaling story. ORT
casts around MLAS's f32 kernels; this EP's f16 path does not reach the same primitive. This is the
largest single dense loss in the matrix and the most tractable: the fix is to route f16 `MatMul`
through the same MLAS `sgemm` the f32 path already uses, with a cast, rather than a bespoke kernel.

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
went from 5.5x slower than ORT to **1.7x-33x faster**, and are now claimed.

What is left for u8 x u8 is thread scaling, not packing: 1.13-1.20x at one thread, widening to
2.08-2.65x at sixteen — the same root cause as [parallel efficiency](#1-parallel-efficiency-not-kernel-quality--f32-dense-and-int4-matmulnbits).

Both `QLinearMatMul` rules were measured on **x86-64 AVX2 only** and are applied on every
architecture, which is the same convention the rest of this table uses. aarch64 has native `i8 x i8`
kernels (SDOT/SMMLA) that need no translation at all, so the claim there is if anything
conservative — but its *speed* is unmeasured here and is not claimed to be measured. Correctness on
that lane is covered unconditionally by `qgemm_i32_matches_the_integer_oracle_for_every_signedness`.

Mixed signedness (`u8 x i8`, `i8 x u8`) was not measured either way. It follows the u8 rule rather
than borrowing the signed win: deferred in the measured region, claimed below it like every other
unmeasured shape here.

The `i8 x i8` "before" ratios above (5.20/4.97/5.22 at 8 threads) were re-measured against ONNX
Runtime 1.27.0 for this round on the same host. An earlier round recorded 2.23-3.07 for the same
declined scalar path; the kernel side did not change between the two, so the difference is the
baseline and the harness, not a regression.

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
