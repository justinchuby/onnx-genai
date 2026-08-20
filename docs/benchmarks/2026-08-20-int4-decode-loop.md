# int4 decode (`m = 1`) through the prefill GEBP — decode-loop A/B

Closes [#1565](https://github.com/justinchuby/onnx-genai/issues/1565).

## The question

[#1563](https://github.com/justinchuby/onnx-genai/pull/1563) measured GEBP 2.51x/4.62x faster than
the incumbent at `m = 1` for 16-element blocks, and then deliberately set
`INT4_PREFILL_GEBP_MIN_ROWS_UNBLOCKED = 2` anyway — refusing its own measurement. The reason: the
number came from `int4_prefill_route_ab.rs`, a **single-op prefill** bench. GEBP returns before
`with_decode_pool` and dispatches on the global pool, so in a real decode loop it would fork the
whole machine once per projection per token. A single-op bench cannot see that cost.

#1565 asked for the measurement that could settle it.

## Harness

`crates/onnx-runtime-ep-cpu/benches/int4_decode_loop_ab.rs`. A llama3-8B projection chain executed
per token at `m = 1`:

| projection | N x K |
|---|---|
| qkv | 4096 x 6144 |
| o | 4096 x 4096 |
| gate / up | 4096 x 14336 |
| down | 14336 x 4096 |

~109 MB of int4 weight read per token. Knobs: `PROBE_BLOCK`, `PROBE_SESSIONS`, `PROBE_TOKENS`,
`PROBE_LAYERS`, `PROBE_SPMD`. Sessions are `std::thread::scope` threads sharing one weight set,
which is the contention the objection was about. Reports cold/steady ms-per-token, p90, and
aggregate tokens/s.

### Two things that had to be right

**One binary, two arms.** Both arms are the same build with all three gates forced to 1;
`ONNX_GENAI_CPU_MM_INT4_GEBP=0` selects today's routes. A third *unmodified* build runs interleaved
as a control for the `current` arm — it agreed within 1% in every cell (block 16: 30.7 vs 30.6;
block 64: 199.7 vs 202.5; block 128: 255.3 vs 256.1), which is what makes `current` trustworthy as
"today's behaviour".

**The scope.** Each token runs inside one `with_decode_pool_scope(spmd, ..)`, exactly as
`native_decode/cpu.rs` wraps a single-token `session.run`. This is not a detail:

| bench shape | GEBP vs current, block 32 |
|---|---|
| kernels called directly (**wrong**) | 1.28x / 1.53x / 1.64x at 1/2/4 sessions, margin growing |
| inside `with_decode_pool_scope` (**production**) | 0.92x / 0.99x / 0.96x |

Installing the scope moved the **`current` arm alone** by 1.44x at 1 session and inverted the
conclusion. The challenger was measured faithfully in both; the incumbent was not. A decode kernel
measured outside the decode pool is measuring a configuration that never ships.

## Result

Aggregate tokens/s, steady phase, mean of three interleaved reps, 1 session:

| block | competitor at `m = 1` | GEBP | current | clean (control) | verdict |
|---|---|---|---|---|---|
| **16** | generic **scalar** per-block dot | **97.3** | 30.6 | 30.7 | GEBP **3.2x** |
| 32 | `borrowed_affine_int4_matmul_nblock` | 108.7 | 118.2 | — | GEBP 0.92x |
| 64 | `borrowed_affine_int4_matmul_nblock` | 111.8 | 202.5 | 199.7 | GEBP 0.55x |
| 128 | `borrowed_affine_int4_matmul_nblock` | 119.3 | 256.1 | 255.3 | GEBP 0.47x |

The result splits on **which kernel is on the other side of the gate**. GEBP's own throughput is
nearly flat across block sizes (97-119 tok/s) — it dequantizes into a packed panel, so the block
size barely matters to it. The incumbent's varies 8.4x (30.6 -> 256.1), because at block 16 it is
not a vectorized kernel at all: both `borrowed_affine_int4_matmul_prefill` and
`borrowed_affine_int4_matmul_nblock` require `block_size % 32 == 0`, so block 16 falls through to
`borrowed_affine_int4_matmul`, a per-block scalar dot.

### Block 16 under contention — the objection inverts

| sessions | GEBP | scalar dot | speedup | GEBP p90 | dot p90 |
|---|---|---|---|---|---|
| 1 | 97.3 | 30.6 | 3.2x | 10.3 ms | 29.7 ms |
| 2 | 111.5 | 32.3 | 3.5x | 18.4 ms | 30.9 ms |
| 4 | 122.8 | 29.7 | **4.1x** | **35.7 ms** | 233.3 ms |

The advantage *grows* with session count and tail latency improves **6.5x** at 4 sessions. The
global-pool fork is real; it is simply cheaper than leaving the work on a scalar dot.

### Read the throughput, not the median

Under contention the per-session median becomes meaningless. Block 32, 4 sessions:

| arm | median ms/token | p90 ms/token | aggregate tok/s |
|---|---|---|---|
| gebp | 25.97 | 30.57 | 142.1 |
| current | **7.12** | **56.31** | 150.1 |

The `current` distribution is bimodal — sessions serialise on the shared pool, so the median reports
whichever token owned the pool and p90 reports the ones that queued. Its median is flat at ~7.1 ms
from 1 to 4 sessions, which cannot be true of a scaling system: 4 sessions at 7.1 ms/token would be
563 tok/s, and the measured aggregate is 150. Throughput and p90 together are the honest pair.

## Disposition

`INT4_PREFILL_GEBP_MIN_ROWS_UNBLOCKED`: **2 -> 1**.

`INT4_PREFILL_GEBP_MIN_ROWS` (3) and `INT4_PREFILL_GEBP_MIN_ROWS_L2_RESIDENT` (6) are **unchanged**,
and the 32/64/128 rows above are positive evidence for that rather than an absence of evidence:
lowering them to 1 would cost 1.08x-2.1x.

### The gate names exactly one block size

`MatMulNBits` rejects block sizes that are not a power of two `>= 16` (blocks 24 and 48 were tried
against this bench and rejected by that validation, which is how the constraint surfaced). The only
power of two `>= 16` that is not a multiple of 32 is **16**. So this gate is the block-16 gate and
nothing else, and the crossover test now sweeps `[16, 32, 64, 128, 256, 512]` and asserts the
unblocked branch claims 16 and only 16.

## Coverage

`matmulnbits_int4_prefill_gebp_covers_the_unblocked_block_size` already ran `m = 1` and asserted
`took_gebp == (m >= INT4_PREFILL_GEBP_MIN_ROWS_UNBLOCKED)` against the
`INT4_PREFILL_GEBP_TEST_CALLS` counter, so lowering the gate turns it into positive proof that
decode now takes the packed microkernel, with the same tolerance check on the numerics. Verified
non-vacuous by mutation: with `ONNX_GENAI_CPU_MM_INT4_GEBP=0` it fails at exactly `m = 1`.

## What this does not fix

Block-16 decode reaches 97-123 tok/s where block 128 reaches 256, so this closes a 3-8x outlier
class and leaves the ordinary case alone. `m = 1` at blocks 32/64/128 keeps today's route. The
residual 2.3x-3.0x to ORT is untouched.

## Reproduce

```bash
# gates forced to 1 -> roy_d_forced; unmodified -> roy_d_clean
cargo build --release -p onnx-runtime-ep-cpu --bench int4_decode_loop_ab

PROBE_BLOCK=16 PROBE_SESSIONS=4 PROBE_TOKENS=48 ./roy_d_forced --bench          # GEBP
ONNX_GENAI_CPU_MM_INT4_GEBP=0 PROBE_BLOCK=16 PROBE_SESSIONS=4 ./roy_d_forced --bench  # current
PROBE_BLOCK=16 PROBE_SESSIONS=4 ./roy_d_clean --bench                            # control
```

Host: AMD EPYC 9V74, 32 vCPU (16 cores x 2 SMT), AVX2 + FMA + F16C, no AVX-512/VNNI, 75.8 GB/s DRAM,
shared. At 4 sessions block 16 GEBP moves 122.8 tok/s x 109 MB = 13.4 GB/s, ~18% of roofline — this
is not a bandwidth result, and neither arm is close to the memory ceiling.
