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

### Deferred ranges

Same host, same harness, same convention: **p50/p90 are ours/ORT, lower is better**, so every number
below 1.00 is a win and every number above it is a loss.

| op | dtype / bits | M | K, N | threads | p50 | p90 | assignment |
|---|---|---|---|---|---|---|---|
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
| `MatMul` | f32 | 128 | 3584 | 8 | 2.76 | 2.85 | defer |
| `MatMul` | f32 | 128 | 3584 | 16 | 1.65 | 1.73 | defer |
| `MatMul` | f32 | 128 | 3584 | 32 | 0.67 | 0.88 | defer |
| `MatMul` | f16 | 1 | 3584 | 1 | 0.86 | 0.96 | defer (see note) |
| `MatMul` | f16 | 1 | 3584 | 2 | 1.47 | 1.89 | defer |
| `MatMul` | f16 | 1 | 3584 | 4 | 2.06 | 2.19 | defer |
| `MatMul` | f16 | 1 | 3584 | 8 | 1.93 | 2.20 | defer |
| `MatMul` | f16 | 1 | 3584 | 16 | 2.01 | 2.79 | defer |
| `MatMul` | f16 | 1 | 3584 | 32 | 3.83 | 5.95 | defer (noisy) |
| `MatMul` | f16 | 128 | 3584 | 1 | 1.00 | 1.01 | defer |
| `MatMul` | f16 | 128 | 3584 | 2 | 1.68 | 1.69 | defer |
| `MatMul` | f16 | 128 | 3584 | 4 | 1.76 | 1.80 | defer |
| `MatMul` | f16 | 128 | 3584 | 8 | 1.72 | 1.77 | defer |
| `MatMul` | f16 | 128 | 3584 | 16 | 1.30 | 1.62 | defer |
| `MatMul` | f16 | 128 | 3584 | 32 | 0.91 | 1.46 | defer |
| `Gemm` | f16 | 1 | 3584 | 1 | 0.86 | 0.94 | defer (see note) |
| `Gemm` | f16 | 1 | 3584 | 2 | 1.13 | 1.25 | defer |
| `Gemm` | f16 | 1 | 3584 | 4 | 0.98 | 1.85 | defer |
| `Gemm` | f16 | 1 | 3584 | 8 | 1.83 | 2.08 | defer |
| `Gemm` | f16 | 1 | 3584 | 16 | 2.05 | 2.68 | defer |
| `Gemm` | f16 | 1 | 3584 | 32 | 4.20 | 7.72 | defer (noisy) |
| `Gemm` | f16 | 128 | 3584 | 1 | 1.03 | 1.03 | defer |
| `Gemm` | f16 | 128 | 3584 | 2 | 1.82 | 1.83 | defer |
| `Gemm` | f16 | 128 | 3584 | 4 | 1.90 | 1.91 | defer |
| `Gemm` | f16 | 128 | 3584 | 8 | 1.86 | 2.24 | defer |
| `Gemm` | f16 | 128 | 3584 | 16 | 1.44 | 1.46 | defer |
| `Gemm` | f16 | 128 | 3584 | 32 | 1.19 | 1.35 | defer |
| `QLinearMatMul` | u8 x u8 | 1 | 3584 | 8 | 22.04 | 27.30 | defer |
| `QLinearMatMul` | u8 x u8 | 128 | 3584 | 8 | 4.19 | 4.23 | defer |
| `QLinearMatMul` | u8 x u8 | 512 | 3584 | 8 | 4.05 | 4.16 | defer |
| `QLinearMatMul` | i8 x i8 | 1 | 3584 | 8 | 2.23 | 2.37 | defer |
| `QLinearMatMul` | i8 x i8 | 128 | 3584 | 8 | 3.02 | 3.05 | defer |
| `QLinearMatMul` | i8 x i8 | 512 | 3584 | 8 | 3.07 | 3.21 | defer |

Ranges **outside** the measured region keep the historical claim, and the code says so explicitly:
`K * N < 2^20`, symbolic/dynamic weight shapes, and dense dtypes other than f32/f16 (where a
deferral could be a load failure rather than a slow session, because ORT's CPU EP may have no kernel
at all).

## Root causes

### 1. Parallel efficiency, not kernel quality — f32 dense and int4 `MatMulNBits`

At **one thread this EP is at parity** (f32 1.00 / 0.97, int4 1.00 / 0.99). The gap opens as threads
are added and closes again only at 32, where ORT's own scaling saturates. We realise roughly **half**
of ORT's parallel speedup.

That is a threadpool/partitioning problem, not a kernel problem. #1054 removed one cause — the
standalone MLAS pool was clamped to `min(available, 8)` workers and never saw the EP's requested
thread count, which cost 2.16x -> 1.61x at 32 vCPU — but the residual 1.4-2.4x at 2-16 threads is
still open. Because the thread count is **not visible at capability time**, a claim would have to
hold at every count, and none of these do.

### 2. A kernel gap — f16 dense (fixed by #1080; f16 now behaves like section 1)

f16 used to be the only dense range that lost at **one** thread. ORT casts around MLAS's f32
kernels; this EP's f16 path did not reach the same primitive, and `Gemm` had no f16 GEMV at all.
#1080 routes constant-weight f16 `MatMul`/`Gemm` prefill through MLAS SGEMM with a once-only widened
and packed `B`, and adds the missing GEMV.

Measured on the same host, `K = N = 3584`, ours/ORT p50, lower is better:

| | before #1080 | after #1080 |
|---|---:|---:|
| `MatMul` f16 M=128, T=1 | 2.47 | **1.00** |
| `MatMul` f16 M=128, T=4 | 6.57 | **1.76** |
| `MatMul` f16 M=128, T=16 | 7.10 | **1.30** |
| `Gemm` f16 M=1, T=1 | 6.57 | **0.86** |
| `Gemm` f16 M=1, T=8 | 46.67 | **1.83** |

The one-thread kernel gap is closed: f16 is now at parity at `M = 128` (1.00 / 1.03) and is a
**genuine win at `M = 1`** (0.86 for both `MatMul` and `Gemm`, i.e. we are ~1.16x faster than ORT).

**It is still deferred, and that is not a contradiction.** The thread count is not visible at
capability time, so a claim has to hold at *every* count, and f16 now loses at 2-16 threads exactly
the way f32 and int4 do (1.13-2.06 at `M = 1`, 1.30-1.90 at `M = 128`). In other words f16 has
stopped being a kernel-quality story and has joined the parallel-efficiency story in section 1. When
section 1's per-call serial work is fixed, f16 should be re-evaluated for a claim at the same time as
f32 and int4 — the `M = 1`, `T = 1` win says the primitive is now the right one.

The improvement is not wasted in the meantime: deferral only applies where a host fallback exists
(`ep.rs` gates `claim_preference_node` on `host_fallback_available`). In native-only mode this EP
still runs its own f16 kernel, and that kernel is now 2.4x-14.3x faster than it was.

### 3. Per-call packing — `QLinearMatMul`

#1058 bound MLAS's integer QGEMM and took u8 x u8 from **27-119x** down to 3-4x. What remains is
structural: **ORT pre-packs the constant B once at session init; this kernel packs inside every
call.** At M=1 a 12.8 MB pack dominates a 1.7 ms call, which is the whole of the residual 22x.
`QLinearMatMulFactory::create` ignores its `Node`, so the kernel is stateless and prepacking would
need a constant-initializer hook with its own correctness surface (weight identity, **address
reuse**, dynamic weights).

Signed x signed is separately capped: MLAS documents `AIsSigned` as unsupported off ARM
(`mlas.h:610-611`), so on x86 only u8 x u8 reaches the fast path at all. On AVX2 without VNNI, u8 x
i8 is additionally **not bit-exact** — `vpmaddubsw` sums adjacent products into a saturating i16, and
`255*(-128) + 255*(-128)` clamps — so it is excluded by a runtime exactness probe rather than an ISA
table, which lets VNNI/AMX hosts pick the fast path up automatically.

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
