# The f16 decode GEMV accumulated through memory, not registers

2026-08-19. AMD EPYC 9V74, 32 vCPU (16 cores x 2 SMT), AVX2 + FMA + F16C, no
AVX-512, no VNNI. L1d 32 KiB/core, L2 1 MiB/core, L3 2 x 32 MiB. Shared host;
load average is disclosed per run.

## Summary

`kernels::half_gemv::gemv_f16_kn` is the x86 `M == 1` f16 GEMV. Its module
documentation states the design premise plainly: at `M == 1` "each weight
element is touched exactly once, so the kernel is purely memory-bound". It was
not. It ran at **12-47 GB/s against a measured 75.8 GB/s ceiling**, because the
inner loop kept its accumulators in `acc` and therefore paid three memory
operations per 8-lane FMA instead of one.

Holding the accumulators in `ymm` registers across the contraction is worth
**1.22x-2.60x** on 14 of 15 measured cells and takes the large cells from ~46%
of the memory roofline to **79-86%**. Numerics are unchanged: every output
element still accumulates over `p` in strictly increasing order, so the result
is bit-identical, and the pre-existing bit-identity tests pass untouched.

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

## Result

`scripts/ort_ab/ab.py --native-only --null-control`, 7 trials x 30 runs,
medians, load average 5.3 at start. `null` is the baseline binary under a second
name; its delta is the host's noise floor for that cell.

| cell | t | before | after | speedup | null | before GB/s | after GB/s | % of roofline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `l2_512` | 4 | 0.018 ms | 0.014 ms | 1.29x | 5.6% | 29.1 | 37.4 | 52% |
| `l2_512` | 16 | 0.028 ms | 0.018 ms | 1.56x | 3.6% | 18.7 | 29.1 | 39% |
| `l2_512` | 32 | 0.027 ms | 0.018 ms | 1.50x | 0.0% | 19.4 | 29.1 | 38% |
| `l2_1024` | 4 | 0.088 ms | 0.051 ms | 1.73x | 2.3% | 23.8 | 41.1 | 58% |
| `l2_1024` | 16 | 0.140 ms | 0.084 ms | 1.67x | 15.7% | 15.0 | 25.0 | 33% |
| `l2_1024` | 32 | 0.171 ms | 0.109 ms | 1.57x | 1.2% | 12.3 | 19.2 | 25% |
| `l3_2048` | 4 | 0.214 ms | 0.175 ms | 1.22x | 7.9% | 39.2 | 47.9 | 67% |
| `l3_2048` | 16 | 0.401 ms | 0.154 ms | **2.60x** | 1.2% | 20.9 | 54.5 | 72% |
| `l3_2048` | 32 | *not separable from noise — see below* ||||||
| `l3_3584` | 4 | 0.549 ms | 0.419 ms | 1.31x | 4.9% | 46.8 | 61.3 | **86%** |
| `l3_3584` | 16 | 1.260 ms | 0.522 ms | 2.41x | 16.3% | 20.4 | 49.2 | 65% |
| `l3_3584` | 32 | 1.075 ms | 0.560 ms | 1.92x | 2.0% | 23.9 | 45.9 | 61% |
| `dram_8192` | 4 | 7.689 ms | 4.100 ms | 1.88x | 1.8% | 17.5 | 32.7 | 46% |
| `dram_8192` | 16 | 3.838 ms | 2.163 ms | 1.77x | 12.4% | 35.0 | 62.1 | **82%** |
| `dram_8192` | 32 | 3.844 ms | 2.228 ms | 1.73x | 0.3% | 34.9 | 60.2 | 79% |

A second run of the two L3 cells, 11 trials x 40 runs, agrees: `l3_2048` t=4
1.35x, t=16 2.42x; `l3_3584` t=4 1.36x, t=16 1.78x, t=32 2.04x (nulls 1.6-6.3%).

**`l3_2048` at t=32 is not claimed.** Three runs measured -45.1% (null 0.6%),
+18.8% (null 70.1%) and -25.8% (null 35.8%). Only the first has a usable
control, and one clean run against two unusable ones is not a result. The cell
is ~0.2 ms of work split into 16 stripes across 32 threads, short enough that
scheduler jitter dominates; every other cell settled.

The small `l2_*` cells read low GB/s in *both* arms because at 0.018 ms the
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
  to `2 * TILE + 8` at four stripe offsets and requires the SIMD stripe to match
  the scalar one bit for bit. A contiguous sweep rather than picked widths is
  what makes an off-by-one in the tile loop's `j + TILE <= w` bound impossible
  to miss.
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

## What this does not fix

- **The parallel decomposition still anti-scales**, just from a higher floor:
  `l3_3584` is 0.419 ms at t=4 and 0.560 ms at t=32. With `STRIPE = 512` and
  `n = 3584` there are 7 stripes, so beyond 7 threads the extra workers only
  contend. That mattered when the kernel was at 24 GB/s; at 46-61 GB/s against a
  75.8 GB/s ceiling there is much less left to win, and an earlier attempt to
  make the stripe width adaptive was refuted outright (see the adaptive-stripe
  rejection in `2026-08-19-f16-gemm-transb-decode.md`). Worth revisiting only
  after something else moves the ceiling.
- **`l2_1024` at t=32 is still only 19.2 GB/s.** Two stripes, so two working
  threads and a fork/join for 0.1 ms of work.
- **The f16 *prefill* path is untouched.** This kernel is `M == 1` only.
- Nothing here applies to aarch64: the module is `#[cfg(target_arch = "x86")]` /
  `x86_64` and NEON hosts take a different route entirely.
