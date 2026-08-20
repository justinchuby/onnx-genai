# The half decode GEMV accumulated through memory, not registers

2026-08-19. AMD EPYC 9V74, 32 vCPU (16 cores x 2 SMT), AVX2 + FMA + F16C, no
AVX-512, no VNNI. L1d 32 KiB/core, L2 1 MiB/core, L3 2 x 32 MiB. Shared host;
load average is disclosed per run.

## Summary

`kernels::half_gemv::gemv_half_kn` is the x86 `M == 1` GEMV for an f16 or bf16
weight in its stored `[K, N]` order. Its module documentation states the design
premise plainly: at `M == 1` "each weight element is touched exactly once, so
the kernel is purely memory-bound". It was not. It ran at **12-47 GB/s against a
measured 75.8 GB/s ceiling**, because the inner loop kept its accumulators in
`acc` and therefore paid three memory operations per 8-lane FMA instead of one.

Holding the accumulators in `ymm` registers across the contraction is worth
**1.13x-2.61x** on 15 of 15 cells with the kernel isolated, and **1.25x-2.21x**
on 13 of 15 cells of a **default** build with no environment set. The one cell
whose weight genuinely comes from DRAM goes from 44% to **83%** of the measured
memory roofline.

The tiling lives in the `stripe_simd_fn!` macro, so it is instantiated for the
bf16 kernel as well as the f16 one — the original revision of this work predated
#1381 and was f16-only.

Numerics are unchanged: every output element still accumulates over `p` in
strictly increasing order, so the result is bit-identical, and the pre-existing
bit-identity tests pass untouched.

## The measurement that started it

There was no f16 GEMV benchmark cell. `gen_gemm.py` covers block-quantised and
f32 dense GEMM, so the one kernel whose entire premise is a bandwidth claim had
nothing checking it. `scripts/ort_ab/gen_f16_gemv.py` adds five, sweeping the
*weight working set* — the only variable that matters for a kernel that reads
each weight once:

| cell | k | n | weight | resident in |
|---|---:|---:|---:|---|
| `l2_512` | 512 | 512 | 0.5 MB | one core's L2 |
| `l2_1024` | 1024 | 1024 | 2.1 MB | L2/L3 |
| `l3_2048` | 2048 | 2048 | 8.4 MB | L3 |
| `l3_3584` | 3584 | 3584 | 25.7 MB | L3 (Qwen3-8B hidden) |
| `dram_8192` | 8192 | 8192 | 134.2 MB | past any LLC here |

Native-alone, `--native-only`, before the change:

| cell | t=4 | t=16 | t=32 | GB/s at t=32 |
|---|---:|---:|---:|---:|
| `l2_512` | 0.028 ms | 0.017 ms | 0.017 ms | 19.4 |
| `l2_1024` | 0.099 ms | 0.111 ms | 0.116 ms | 12.3 |
| `l3_2048` | 0.201 ms | 0.356 ms | 0.366 ms | 28.1 |
| `l3_3584` | 0.568 ms | 0.672 ms | 0.830 ms | 23.9 |
| `dram_8192` | 7.877 ms | 4.105 ms | 3.996 ms | 34.9 |

Two things are wrong here and they point the same way.

**The L2-resident cell runs at the same GB/s as the DRAM-resident one.** A
0.5 MB working set that never leaves L2 has no business moving 19 GB/s. If the
kernel were memory-bound, the small cells would be far faster per byte than the
large ones. They are not, so the limit is not memory.

**More threads make it slower.** `l2_1024`, `l3_2048` and `l3_3584` all
*regress* from t=4 to t=32. A kernel that is not bandwidth-bound and does not
scale is being limited by something per-core that extra cores only contend for.

## The roofline

`roofline_bandwidth --threads 1,2,4,8,16,32 --mib 1024 --seconds 3`:

| threads | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| GB/s | 33.7 | 42.3 | 71.4 | 74.7 | 75.6 | 75.8 |

The host sustains **75.8 GB/s** and saturates by 4-8 threads. The kernel was
delivering 12-47 GB/s — between 16% and 62% of what the machine will give, and
under half of it on every large cell.

## Root cause

The inner loop was:

```rust
for p in 0..k {
    let av = _mm256_set1_ps(*ap.add(p));
    let row = bp.add(p * n + j0);
    while j + 8 <= w {
        let h = _mm_loadu_si128(row.add(j) as *const __m128i);
        let bw = _mm256_cvtph_ps(h);
        let cur = _mm256_loadu_ps(cp.add(j));                       // load acc
        _mm256_storeu_ps(cp.add(j), _mm256_fmadd_ps(bw, av, cur));  // store acc
        j += 8;
    }
}
```

`p` is the outer loop, so the accumulator for a given `j` is live across the
whole contraction but lives in `acc`. Per 8-lane FMA that is **three memory
operations** — load the weight, load the accumulator, store it back — against
one arithmetic operation. The load/store ports, not the FMA units, set the
rate, and each FMA additionally waits on store-to-load forwarding from the same
address one `p` earlier.

`STRIPE = 512` means 2 KiB of accumulators, which the existing comment
correctly notes is "comfortably inside L1". That was the problem, not the
reassurance it reads as: being in L1 is only cheap relative to L3. It is not
cheap relative to a register, and 16 registers were sitting unused.

This is the same defect I had just fixed in the int4 prefill and described
there as re-doing work per row. It is worth naming the general shape, because
that is now twice: **a value that is live across a loop belongs in a register;
if the loop nest puts it in memory, the loop nest is wrong.** In the int4 case
the repeated work was nibble decoding; here it is the accumulator round trip.

## The fix

Tile the output columns and hoist the `p` loop inside the tile, so `TILE / 8`
accumulators stay in `ymm` for the entire contraction and reach memory exactly
once:

```rust
while j + TILE <= w {
    let mut sums = [_mm256_setzero_ps(); TILE / 8];
    let base = bp.add(j0 + j);
    for p in 0..k {
        let av = _mm256_set1_ps(*ap.add(p));
        let row = base.add(p * n);
        for (lane, sum) in sums.iter_mut().enumerate() {
            let h = _mm_loadu_si128(row.add(lane * 8) as *const __m128i);
            *sum = _mm256_fmadd_ps(_mm256_cvtph_ps(h), av, *sum);
        }
    }
    for (lane, sum) in sums.iter().enumerate() {
        _mm256_storeu_ps(cp.add(j + lane * 8), *sum);
    }
    j += TILE;
}
```

One memory operation per FMA — the weight — which is the minimum the problem
admits, since the weight is read once and nothing else is touched.

**Tiling costs no extra traffic.** A stripe's weight is swept once per tile over
`STRIPE / TILE` disjoint column ranges: the same `k * STRIPE` elements in total,
the same `n`-element stride between consecutive `p`, and — because `TILE = 64`
f16 is exactly two 64-byte lines and `STRIPE` is a whole number of tiles — no
fetched line is ever partially consumed. `a_tile_never_straddles_a_stripe` pins
all three of those divisibility facts.

## Why 64 columns

Measured, not asserted — 15 cells, native-alone, interleaved, medians of 7
trials, percentages relative to the unmodified build:

| cell | t | `TILE=32` | `TILE=64` | `TILE=96` | `TILE=128` |
|---|---:|---:|---:|---:|---:|
| `l2_512` | 4 | -11.1% | **-22.2%** | -16.7% | +16.7% |
| `l2_512` | 16 | -27.3% | **-36.4%** | -31.8% | +4.5% |
| `l2_512` | 32 | -11.1% | **-22.2%** | -11.1% | +16.7% |
| `l2_1024` | 4 | -34.0% | **-44.7%** | -29.8% | -36.2% |
| `l2_1024` | 16 | -31.5% | **-37.7%** | -28.5% | -33.8% |
| `l2_1024` | 32 | -30.3% | **-36.4%** | -30.3% | -34.1% |
| `l3_2048` | 4 | -7.2% | **-48.3%** | -21.1% | **-48.3%** |
| `l3_2048` | 16 | -22.1% | -45.3% | -28.7% | **-45.6%** |
| `l3_2048` | 32 | -23.8% | **-45.1%** | -29.3% | -44.4% |
| `l3_3584` | 4 | -26.3% | **-36.5%** | -28.4% | -33.2% |
| `l3_3584` | 16 | -50.0% | -50.3% | -53.5% | **-56.6%** |
| `l3_3584` | 32 | -45.5% | -47.0% | **-49.0%** | -48.1% |
| `dram_8192` | 4 | -38.7% | **-46.4%** | -33.3% | -46.0% |
| `dram_8192` | 16 | -32.8% | **-50.8%** | -39.0% | -48.3% |
| `dram_8192` | 32 | -34.0% | **-42.0%** | -32.0% | -40.2% |

64 wins 11 cells outright and ties a twelfth; the three it loses, it loses by
0.3, 6.3 and 2.0 points. 128 is the instructive one: it needs 16 accumulators
plus the broadcast and the widened weight, which is 18 of 16 architectural
`ymm`, so it spills — and on the smallest cell it is *slower than doing
nothing*. 64 leaves 10 registers in use and never regresses.

## Which paths actually reach this kernel

`gemv_half_kn` is not the only decode route any more, and the surface changed
underneath this work while it was open. #1381 landed a decode handover: an
`M == 1` `MatMul` on a half weight of **1,048,576 elements or more** now goes to
the fused widen-pack GEBP instead, and only smaller weights keep the GEMV. So
four of the five `MatMul` cells below no longer reach the kernel in a default
build, and measuring them as if they did would be measuring nothing.

What still reaches it:

| route | weight range |
|---|---|
| `MatMul` f16/bf16 decode | below 1,048,576 elements |
| `Gemm` f16 decode, `transB=0` | **any** — this path has no weight gate |
| any decode under `ONNX_GENAI_CPU_MM_HALF_GEBP=0` | any |
| any decode on 32-bit `x86` | any — there is no GEBP to hand off to |
| bf16 decode on an AVX-512 BF16 host | any — the handover declines there |

Both are therefore measured below, and separately: the `MatMul` cells with the
GEBP switched off, which isolates the kernel over the whole weight range, and
the `Gemm` cells with **no environment set at all**, which is what a default
build runs. `gen_f16_gemv.py --op gemm` emits the second set.

## Result

`scripts/ort_ab/ab.py --native-only --null-control`, 7 trials x 30 runs,
medians, re-measured on latest `main` after the merge. `null` is the baseline
binary under a second name; its delta is the host's noise floor for that cell,
and a speedup no larger than it is not claimed.

### `MatMul` cells, GEBP switched off — the kernel over its whole range

| cell | t | before | after | speedup | null | before GB/s | after GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `l2_512` | 4 | 0.021 ms | 0.014 ms | 1.50x | 14.3% | 25.0 | 37.4 |
| `l2_512` | 16 | 0.027 ms | 0.019 ms | 1.42x | 0.0% | 19.4 | 27.6 |
| `l2_512` | 32 | 0.021 ms | 0.014 ms | 1.50x | 4.8% | 25.0 | 37.4 |
| `l2_1024` | 4 | 0.090 ms | 0.053 ms | 1.70x | 2.2% | 23.3 | 39.6 |
| `l2_1024` | 16 | 0.163 ms | 0.098 ms | 1.66x | 12.3% | 12.9 | 21.4 |
| `l2_1024` | 32 | 0.144 ms | 0.112 ms | 1.29x | 13.2% | 14.6 | 18.7 |
| `l3_2048` | 4 | 0.213 ms | 0.095 ms | **2.24x** | 9.4% | 39.4 | 88.3 |
| `l3_2048` | 16 | 0.406 ms | 0.187 ms | **2.17x** | 4.4% | 20.7 | 44.9 |
| `l3_2048` | 32 | 0.503 ms | 0.358 ms | 1.41x | 1.6% | 16.7 | 23.4 |
| `l3_3584` | 4 | 0.592 ms | 0.290 ms | **2.04x** | 9.3% | 43.4 | 88.6 |
| `l3_3584` | 16 | 0.813 ms | 0.312 ms | **2.61x** | 0.6% | 31.6 | 82.3 |
| `l3_3584` | 32 | 0.790 ms | 0.701 ms | 1.13x | 1.8% | 32.5 | 36.6 |
| `dram_8192` | 4 | 7.413 ms | 4.221 ms | 1.76x | 1.8% | 18.1 | 31.8 |
| `dram_8192` | 16 | 4.067 ms | 2.126 ms | 1.91x | 9.7% | 33.0 | 63.1 |
| `dram_8192` | 32 | 3.702 ms | 2.222 ms | 1.67x | 3.2% | 36.3 | 60.4 |

**15 of 15 above their null control, 1.13x-2.61x.**

### `Gemm` cells, shipped default — no environment set

| cell | t | before | after | speedup | null | before GB/s | after GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `l2_512` | 4 | 0.021 ms | 0.014 ms | 1.50x | 4.8% | 25.0 | 37.4 |
| `l2_512` | 16 | 0.026 ms | 0.014 ms | 1.86x | 3.8% | 20.2 | 37.4 |
| `l2_512` | 32 | 0.022 ms | 0.017 ms | *not claimed* | 22.7% | 23.8 | 30.8 |
| `l2_1024` | 4 | 0.087 ms | 0.055 ms | 1.58x | 2.3% | 24.1 | 38.1 |
| `l2_1024` | 16 | 0.110 ms | 0.083 ms | 1.33x | 1.8% | 19.1 | 25.3 |
| `l2_1024` | 32 | 0.154 ms | 0.112 ms | 1.38x | 4.5% | 13.6 | 18.7 |
| `l3_2048` | 4 | 0.222 ms | 0.178 ms | 1.25x | 1.8% | 37.8 | 47.1 |
| `l3_2048` | 16 | 0.442 ms | 0.200 ms | **2.21x** | 16.5% | 19.0 | 41.9 |
| `l3_2048` | 32 | 0.343 ms | 0.352 ms | *not claimed* | 48.7% | 24.5 | 23.8 |
| `l3_3584` | 4 | 0.566 ms | 0.409 ms | 1.38x | 12.0% | 45.4 | 62.8 |
| `l3_3584` | 16 | 0.561 ms | 0.324 ms | 1.73x | 1.1% | 45.8 | 79.3 |
| `l3_3584` | 32 | 0.823 ms | 0.594 ms | 1.39x | 16.6% | 31.2 | 43.2 |
| `dram_8192` | 4 | 7.702 ms | 4.171 ms | 1.85x | 0.4% | 17.4 | 32.2 |
| `dram_8192` | 16 | 3.278 ms | 1.913 ms | 1.71x | 2.8% | 40.9 | 70.2 |
| `dram_8192` | 32 | 3.764 ms | 2.140 ms | 1.76x | 1.1% | 35.7 | 62.7 |

**13 of 15 above their null control, 1.25x-2.21x.** The two unclaimed cells both
had unusable controls (nulls of 22.7% and 48.7%) rather than small effects; at
t=32 these are ~0.02 ms and ~0.35 ms of work split across 32 threads, short
enough that scheduler jitter dominates. Every other cell settled.

### On the bandwidth figures

The GB/s column counts the `2 * k * n` weight, which is all this kernel reads.
Three cells now exceed the host's **75.8 GB/s DRAM** ceiling — `l3_2048` at 88.3
and `l3_3584` at 88.6 and 82.3. That is not a measurement error and not a
roofline violation: at 8.4 MB and 25.7 MB those weights are L3-resident, so they
never go to DRAM at all, and the DRAM roofline is simply the wrong ceiling for
them. Before this change they ran at 20-43 GB/s, far enough below the DRAM
figure that the distinction never showed up. It does now, which is itself the
result: the kernel stopped being the limit, so the cells separated by where
their weight actually lives.

`dram_8192` is the only cell whose weight genuinely comes from DRAM, and it is
the one to read against the roofline: **63.1 GB/s of 75.8, or 83%**, up from
33.0 GB/s (44%).

The small `l2_*` cells read low GB/s in *both* arms because at 0.014-0.02 ms the
per-run fixed cost is a large fraction of the measurement. Their speedups are
real and above noise; their absolute bandwidth figures are not meaningful.

## Numerics

Bit-identical, and this is a stronger claim than the int4 row-blocking change
could make. Tiling changes only *which register* holds a partial sum, never the
order it is built in: within any one output element the contraction still runs
`p = 0, 1, ... k-1` with the same FMA at each step. So the existing oracle tests
— `gemv_is_bit_identical_to_a_naive_reference`,
`simd_and_scalar_stripes_agree_bit_for_bit`,
`widths_below_across_and_beyond_the_stripe_are_exact`,
`results_do_not_depend_on_the_thread_count` — pass **unmodified**. Nothing was
weakened to a tolerance.

Two tests were added for the new structure:

- `stripe_widths_around_the_tile_boundary_are_exact` sweeps every width from 1
  to `2 * TILE + 9` at four stripe offsets and requires the SIMD stripe to match
  the scalar one bit for bit. A contiguous sweep rather than picked widths is
  what makes an off-by-one in the tile loop's `j + TILE <= w` bound impossible
  to miss. The top width is two whole tiles plus one 8-lane block plus one
  scalar, so the sweep ends having exercised all three in a single call, and it
  asserts its own combination count so a future `n` cannot silently shrink it.
- `a_tile_never_straddles_a_stripe` pins `STRIPE % TILE == 0`, `TILE % 8 == 0`
  and `TILE * 2 % 64 == 0`, the three divisibility facts the no-extra-traffic
  argument rests on.

## Tooling

`ab.py` grew `--native-only`. Per `sebastian-paired-harness-coresidency`, ORT's
intra-op pool spin-waits, so a paired run steals cores from the native arm; on
these cells it depressed the native median by up to **6x** (`l3_3584` t=32 reads
4.44 ms paired against 0.71 ms alone) and drove the null control to 27%, which
is larger than most of the effects here. A native-vs-native A/B has no reason to
pay that, and the first paired attempt at this measurement produced a table with
three sign errors in it. The flag lives in the shared driver rather than a
private script so the next person does not repeat the mistake.

## The decode handover is now measured on a stale premise

#1381 set the `MatMul` decode handover at `HALF_PREFILL_GEBP_MIN_WEIGHT` on a
13-shape sweep showing the GEBP faster at every weight at or above 1.05M — by up
to 2.27x. That sweep measured the **untiled** GEMV. Since this change is worth
1.13x-2.61x to the GEMV, the comparison has to be re-run, and it partly inverts.

Re-running #1381's own harness (`bench half_decode_gemv_ab`, 5 interleaved
repetitions, median of the per-run steady p50, `ratio = GEMV / GEBP` so below
1.00 favours the GEMV) against the tiled kernel:

| `K x N` | elements | f16 ratio | f16 /ctl | bf16 ratio | bf16 /ctl | #1381 f16 /ctl |
|---|---:|---:|---:|---:|---:|---:|
| 1024x768 | 0.79M | 0.25 | 0.25 | 0.32 | 0.33 | 0.40 |
| 2048x2048 | 4.19M | 1.12 | 1.26 | 1.16 | 1.31 | 1.78 |
| 4096x11008 | 45.1M | 0.88 | 0.82 | 0.88 | 0.82 | 1.18 |
| 896x151936 | 136M | 0.86 | 0.86 | 0.88 | 0.88 | 1.26 |

**The two largest shapes invert** — `mlp` and `lm_head`, the shapes that
actually dominate decode — and 2048x2048 narrows from 1.78 to 1.26 without
inverting. The `ab.py` cells disagree with the bench harness at 4.19M, and the
disagreement is thread count: `cargo bench` runs at the rayon default of 32,
where `l3_2048` reads -2.13% *within* noise, while at t=4 and t=16 the same cell
reads -83.5% and -61.8% in the GEMV's favour.

So the honest statement is that the threshold is **wrong at both ends and right
in the middle**, and that its correct value is thread-count dependent, which a
single weight cutoff cannot express. Retuning it needs its own k x n x thread
sweep and its own control, and stacking a routing change on evidence that
conflicts between two harnesses is exactly the kind of unproven change this
ledger exists to refuse. **Filed as a follow-up rather than folded in here.**
This PR changes no routing at all: every shape takes the same route it took
before, only faster.

## What this does not fix

- **The parallel decomposition still anti-scales**, just from a higher floor:
  `l3_3584` is 0.290 ms at t=4 and 0.701 ms at t=32. With `STRIPE = 512` and
  `n = 3584` there are 7 stripes, so beyond 7 threads the extra workers only
  contend. That mattered when the kernel was at 24 GB/s; against a 75.8 GB/s
  ceiling there is less left to win, and an earlier attempt to make the stripe
  width adaptive was refuted outright (see the adaptive-stripe rejection in
  `2026-08-19-f16-gemm-transb-decode.md`). Worth revisiting only after something
  else moves the ceiling — but note it is now the *dominant* remaining loss on
  the L3 cells, since t=32 is where every unclaimed cell sits.
- **`l2_1024` at t=32 is still only 18.7 GB/s.** Two stripes, so two working
  threads and a fork/join for 0.1 ms of work.
- **The f16 *prefill* path is untouched.** This kernel is `M == 1` only.
- Nothing here applies to aarch64: the module is `#[cfg(target_arch = "x86")]` /
  `x86_64` and NEON hosts take a different route entirely.
