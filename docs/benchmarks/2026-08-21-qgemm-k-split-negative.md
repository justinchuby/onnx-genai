# Splitting `k` in the fused decode GEMM: a negative result

Date: 2026-08-21. Host: AMD EPYC 9V74, 32 vCPU (16 cores x 2 SMT), AVX2 + FMA +
F16C, **no AVX-512 / VNNI / AMX**. L1d 32 KiB/core, L2 1 MiB/core, L3 64 MiB,
75.8 GB/s DRAM, shared with other tenants.

**Verdict: not merged.** Stated precisely, because the two matrices do not say
the same thing. Over the 25 cells whose null control holds (see
[Method](#method)): the first variant is a **wash** -- 8 wins, 5 losses, 12
neutral, median ratio 1.029, **geomean 0.997** -- which loses hardest exactly
where the hypothesis predicted it would win most; the second, which removes the
first one's identified overhead, is a **net loss** -- 3 wins, 13 losses, 9
neutral, median 0.918, geomean 0.846 (3/12/9, median 0.947, geomean 0.904 with
the one unexplained cell also removed). A mechanism that costs a scratch allocation, a
reduction and a second parallel plan has to earn its place, and neither variant
does. Closed in favour of `qgemm` keeping the column split.

The experiment's scope is narrow and is stated as such in
[what this does and does not rule out](#what-this-does-and-does-not-rule-out).

## The hypothesis

`qgemm`'s fused path (`m <= MR`, i.e. decode) reads every byte of `B` exactly
once. The existing parallel plan splits **columns**, so with `n = 3584` and
eight workers each worker walks a *narrow vertical stripe*: 448 contiguous bytes
out of every 3584-byte row. The stride between two useful lines is larger than a
4 KiB page, which is where the hardware stride prefetcher stops following, and
at `m = 1` there is no reuse of `A` to hide the miss latency behind.

The proposed fix: split **`k`** instead. Each worker gets a contiguous
*horizontal band* of `B` -- whole rows, streamed -- paid for with one private
`m * n` `i32` accumulator per band plus a final reduction. Because accumulation
is wrapping `i32` (exactly mod 2^32), summing bands in **band order** is
bit-identical to one pass, so the module's "deterministic at every thread count"
property survives.

If the hypothesis were right, the fused path should get materially faster at the
thread counts where the column stripe is narrowest.

## What was built

Two variants, both correct (a bit-identity test against the column split passes
at every thread count, and both directions of the plan gate are mutation-checked).

* **v1** -- hybrid band x column plan. Bands take what `k` affords
  (`K_SPLIT_MIN_BAND_ROWS = 512`), column blocks fill whatever workers the bands
  left idle, never cut below 512 contiguous bytes. Serial reduction.
* **v2** -- v1 plus (a) finer bands (`K_SPLIT_MIN_BAND_ROWS = 256`) with the band
  count chosen as the largest divisor of the pool size, so `bands *
  column_blocks` is a whole number of waves instead of a half-empty final one,
  and (b) a **parallel** reduction, chunked by column at 4096 `i32`.

v2 was motivated by an Amdahl estimate: the serial reduction is `bands * m * n`
adds against `m * n * k / threads` parallel multiply-accumulates, i.e. a serial
fraction of `bands * threads / k` -- 6% at eight bands, 32 workers, `k = 4096`.

## Method

Two prebuilt test binaries (`cargo test --release --no-run`, copied aside) run
alternately, so no rebuild happens between samples. `bench_qgemm_ab`, 21
iterations, p50, **3 reps**, arms interleaved at rep granularity. Each run also
reports the `portable` scalar arm as a **drift control**: it is unchanged code,
so its ratio between the two binaries must be 1.

The `t = 1` rows are a second, stronger control: at one thread `KSplit::plan`
returns `None`, so **both binaries execute identical code**. Any deviation there
is the noise floor, not the mechanism.

```
QGEMM_AB_SHAPES=1x3584x3584,1x4096x14336,1x1024x3072,1x512x4096,4x3584x3584,2x4096x4096
QGEMM_AB_THREADS=1,2,4,8,16,32
QGEMM_AB_ITERS=21
```

Controls measured: `portable` arm ratio **0.999** (v1 matrix). The `t = 1`
identical-code rows, which must all read 1.00:

| shape | v1 | v2 |
|---|---|---|
| 1x3584x3584 | 0.99 | 0.98 |
| 1x4096x14336 | 1.04 | 1.00 |
| **1x1024x3072** | **0.71** | **1.14** |
| 1x512x4096 | 1.04 | 0.96 |
| 4x3584x3584 | 1.00 | 1.02 |
| 2x4096x4096 | 1.01 | 0.99 |

Five of six shapes hold to +/-4%. **`1x1024x3072` does not**: its 3.1M MACs are
short enough that binary code layout alone moves it **-29% in one matrix and
+14% in the other**. Every `1x1024x3072` row below -- wins *and* losses -- is
therefore **excluded from the tallies and from the argument**, rather than
excluded only where it is inconvenient. It is left in the tables, labelled, so
the exclusion is visible instead of silent.

Win/loss labels use +/-5% on the median of three reps, applied uniformly by
`.roy_sum.py`. Tallies cover the **25** cells that are neither controls nor on
the excluded shape. For completeness, over all 30 non-control cells the totals
are 10/6/14 (v1) and 3/15/12 (v2) -- the exclusion moves v1's geomean from 0.975
to 0.997 and v2's from 0.862 to 0.846, i.e. it works *against* the verdict on v1
and barely moves v2.

## v1: hybrid plan, serial reduction

GMACS, three reps each, ratio of medians (k-split / column split).

| shape | t | column split | k split | ratio | |
|---|---|---|---|---|---|
| 1x3584x3584 | 1 | 23.6, 17.1, 22.5 | 23.6, 22.4, 21.8 | 0.99 | control |
| 1x3584x3584 | 2 | 38.3, 39.3, 38.5 | 38.4, 41.8, 39.9 | 1.04 | |
| 1x3584x3584 | 4 | 62.4, 63.0, 61.6 | 73.3, 61.2, 66.1 | 1.06 | win |
| 1x3584x3584 | 8 | 89.5, 86.5, 94.2 | 75.8, 68.2, 83.3 | 0.85 | **loss** |
| 1x3584x3584 | 16 | 49.7, 64.8, 63.8 | 62.7, 65.5, 65.8 | 1.03 | |
| 1x3584x3584 | 32 | 30.9, 43.4, 35.8 | 29.8, 34.3, 40.9 | 0.96 | |
| 1x4096x14336 | 1 | 7.0, 7.5, 7.4 | 7.7, 7.6, 7.5 | 1.04 | control |
| 1x4096x14336 | 2 | 13.7, 17.7, 13.9 | 14.0, 14.0, 15.7 | 1.01 | |
| 1x4096x14336 | 4 | 32.4, 27.7, 25.9 | 29.6, 34.9, 28.3 | 1.07 | win |
| 1x4096x14336 | 8 | 48.6, 45.9, 49.2 | 47.2, 48.1, 46.5 | 0.97 | |
| 1x4096x14336 | 16 | 64.3, 65.8, 64.7 | 61.9, 61.5, 58.3 | 0.95 | |
| 1x4096x14336 | 32 | 67.1, 75.9, 79.2 | 67.6, 61.3, 65.1 | 0.86 | **loss** |
| 1x1024x3072 | 1 | 25.3, 24.3, 15.5 | 17.3, 15.7, 17.3 | 0.71 | control (noise) |
| 1x1024x3072 | 2 | 31.5, 34.2, 31.0 | 30.9, 32.2, 31.9 | 1.01 | |
| 1x1024x3072 | 4 | 44.5, 44.5, 46.1 | 48.8, 48.1, 50.3 | 1.10 | win |
| 1x1024x3072 | 8 | 56.8, 56.9, 58.1 | 64.1, 64.3, 62.5 | 1.13 | win |
| 1x1024x3072 | 16 | 57.8, 58.4, 53.8 | 23.8, 23.1, 22.8 | 0.40 | **loss** |
| 1x1024x3072 | 32 | 10.9, 12.1, 8.7 | 10.7, 11.1, 11.2 | 1.03 | |
| 1x512x4096 | 1 | 13.3, 15.4, 13.5 | 14.0, 13.9, 14.5 | 1.04 | control |
| 1x512x4096 | 2 | 21.7, 24.7, 21.9 | 22.9, 22.9, 22.2 | 1.04 | |
| 1x512x4096 | 4 | 32.5, 35.5, 32.5 | 33.8, 35.3, 33.6 | 1.04 | |
| 1x512x4096 | 8 | 43.1, 44.8, 43.7 | 44.2, 45.1, 43.0 | 1.01 | |
| 1x512x4096 | 16 | 45.5, 41.4, 41.4 | 46.0, 48.8, 51.9 | 1.18 | win |
| 1x512x4096 | 32 | 28.1, 39.4, 36.7 | 41.3, 38.7, 36.7 | 1.05 | win |
| 4x3584x3584 | 1 | 43.8, 42.7, 42.9 | 43.6, 42.7, 43.0 | 1.00 | control |
| 4x3584x3584 | 2 | 72.4, 73.9, 75.4 | 79.5, 77.1, 73.6 | 1.04 | |
| 4x3584x3584 | 4 | 117.2, 112.2, 115.0 | 133.7, 117.5, 125.3 | 1.09 | win |
| 4x3584x3584 | 8 | 157.3, 148.2, 156.6 | 159.6, 153.0, 151.8 | 0.98 | |
| 4x3584x3584 | 16 | 109.7, 107.5, 110.9 | 133.5, 133.7, 129.6 | 1.22 | win |
| 4x3584x3584 | 32 | 96.2, 88.0, 96.5 | 74.0, 77.9, 78.7 | 0.81 | **loss** |
| 2x4096x4096 | 1 | 15.3, 15.0, 15.1 | 15.3, 15.3, 15.0 | 1.01 | control |
| 2x4096x4096 | 2 | 28.7, 29.4, 29.0 | 29.8, 30.1, 29.7 | 1.03 | |
| 2x4096x4096 | 4 | 53.5, 53.3, 56.4 | 56.3, 57.9, 55.7 | 1.05 | win |
| 2x4096x4096 | 8 | 87.1, 84.7, 86.8 | 92.0, 91.4, 89.4 | 1.05 | win |
| 2x4096x4096 | 16 | 91.4, 104.5, 100.6 | 111.1, 76.6, 85.5 | 0.85 | **loss** |
| 2x4096x4096 | 32 | 80.4, 79.3, 87.1 | 65.2, 78.1, 64.0 | 0.81 | **loss** |

**8 wins, 5 losses, 12 neutral** over the 25 counted cells; **geomean 0.997**.
That is a wash, not a win: no shape wins at every thread count, no thread count
wins at every shape, and the only consistent column is `t = 4` (1.04-1.10 on all
six shapes) -- 5-10% against a +/-4% control floor.

The shape of the disagreement matters more than the tally. The hypothesis
predicts the gain grows as the column stripe narrows, i.e. **with thread count**.
The measurement does the opposite: `t = 4` is the best column and `t = 32` the
worst (three of the five counted losses, and the only two counted losses at
`t = 8`/`t = 16` are on the widest shapes). A mechanism that inverts its own
predicted trend has not been confirmed by a positive geomean.

## v2: pool-aligned bands, parallel reduction

The Amdahl fix made it **worse**, decisively. Full matrix:

| shape | t | column split | k split | ratio | |
|---|---|---|---|---|---|
| 1x3584x3584 | 1 | 23.4, 23.0, 23.4 | 22.6, 22.9, 23.8 | 0.98 | control |
| 1x3584x3584 | 2 | 44.0, 42.6, 39.0 | 43.6, 47.1, 42.5 | 1.02 |  |
| 1x3584x3584 | 4 | 59.3, 63.1, 62.9 | 69.5, 66.6, 69.1 | 1.10 | win |
| 1x3584x3584 | 8 | 88.5, 96.4, 97.3 | 75.4, 79.3, 95.6 | 0.82 | **loss** |
| 1x3584x3584 | 16 | 55.7, 66.9, 59.5 | 58.0, 56.8, 67.7 | 0.98 |  |
| 1x3584x3584 | 32 | 32.8, 36.2, 37.2 | 29.8, 33.8, 33.0 | 0.91 | **loss** |
| 1x4096x14336 | 1 | 7.4, 7.6, 7.5 | 7.5, 7.4, 7.4 | 1.00 | control |
| 1x4096x14336 | 2 | 13.1, 15.8, 13.7 | 14.8, 14.1, 14.8 | 1.08 | win |
| 1x4096x14336 | 4 | 25.5, 26.4, 26.4 | 28.2, 26.1, 26.5 | 1.00 |  |
| 1x4096x14336 | 8 | 48.2, 48.7, 49.2 | 42.6, 43.0, 44.2 | 0.88 | **loss** |
| 1x4096x14336 | 16 | 63.0, 79.8, 64.0 | 44.0, 46.3, 42.0 | 0.69 | **loss** |
| 1x4096x14336 | 32 | 64.8, 75.0, 75.6 | 59.7, 61.1, 50.3 | 0.80 | **loss** |
| 1x1024x3072 | 1 | 19.0, 15.0, 15.8 | 15.8, 19.6, 18.1 | 1.14 | control |
| 1x1024x3072 | 2 | 30.9, 31.1, 31.4 | 31.3, 31.1, 34.5 | 1.01 |  |
| 1x1024x3072 | 4 | 45.8, 45.0, 47.5 | 45.2, 47.1, 47.4 | 1.03 |  |
| 1x1024x3072 | 8 | 58.0, 60.0, 57.1 | 59.2, 62.1, 60.1 | 1.04 |  |
| 1x1024x3072 | 16 | 57.7, 61.7, 57.0 | 47.8, 25.8, 60.4 | 0.83 | **loss** |
| 1x1024x3072 | 32 | 10.8, 8.7, 11.1 | 9.4, 9.5, 10.6 | 0.88 | **loss** |
| 1x512x4096 | 1 | 13.8, 13.3, 14.5 | 13.0, 13.5, 13.3 | 0.96 | control |
| 1x512x4096 | 2 | 21.9, 21.9, 22.6 | 23.6, 21.6, 22.1 | 1.01 |  |
| 1x512x4096 | 4 | 33.2, 32.1, 33.3 | 32.8, 33.0, 32.4 | 0.99 |  |
| 1x512x4096 | 8 | 41.2, 41.6, 44.5 | 41.6, 41.8, 41.5 | 1.00 |  |
| 1x512x4096 | 16 | 48.3, 43.6, 47.5 | 42.8, 43.6, 48.0 | 0.92 | **loss** |
| 1x512x4096 | 32 | 38.1, 36.4, 38.6 | 6.5, 6.0, 7.0 | 0.17 | **loss** |
| 4x3584x3584 | 1 | 43.2, 43.3, 42.7 | 44.0, 43.9, 43.3 | 1.02 | control |
| 4x3584x3584 | 2 | 72.7, 72.6, 71.2 | 76.0, 81.3, 76.9 | 1.06 | win |
| 4x3584x3584 | 4 | 114.7, 101.2, 113.3 | 123.8, 115.2, 118.7 | 1.05 |  |
| 4x3584x3584 | 8 | 153.7, 151.5, 153.5 | 133.6, 84.0, 124.3 | 0.81 | **loss** |
| 4x3584x3584 | 16 | 107.7, 119.3, 107.1 | 91.8, 95.5, 98.9 | 0.89 | **loss** |
| 4x3584x3584 | 32 | 91.9, 100.2, 96.5 | 73.3, 72.6, 44.9 | 0.75 | **loss** |
| 2x4096x4096 | 1 | 15.2, 15.1, 15.3 | 15.3, 14.9, 15.0 | 0.99 | control |
| 2x4096x4096 | 2 | 30.2, 29.1, 30.2 | 31.2, 30.2, 29.9 | 1.00 |  |
| 2x4096x4096 | 4 | 53.3, 52.0, 53.6 | 55.0, 54.2, 53.4 | 1.02 |  |
| 2x4096x4096 | 8 | 85.2, 83.2, 84.9 | 87.7, 77.9, 77.4 | 0.92 | **loss** |
| 2x4096x4096 | 16 | 95.1, 100.4, 103.2 | 62.7, 63.4, 74.4 | 0.63 | **loss** |
| 2x4096x4096 | 32 | 81.8, 103.5, 87.6 | 56.4, 57.1, 55.5 | 0.64 | **loss** |

**3 wins, 13 losses, 9 neutral** over the same 25 cells (median 0.918, geomean
0.846); **3 / 12 / 9**, median 0.947, geomean 0.904, once the anomaly below is
also removed. Either way v2 is worse than v1, which is the comparison this
variant existed to make. The `1x512x4096` `t = 32` cell (0.17x) is consistently reproduced across all
three reps and is **not explained**: 32 KiB of scratch over 2 bands x 8 column
blocks is neither a scratch-traffic nor a wave-imbalance story, and the same
cell was a 1.05 *win* under v1. Because it is unexplained it is **excluded from
the tally above** rather than merely annotated -- it contributes ~6 points of
geomean, and leaving it in while calling it "not evidence" would be having it
both ways. The comparative claim (v2 worse than v1) survives its removal, which
is why the anomaly does not need to be resolved before closing this out.

More bands and a parallel reduction should have helped under the Amdahl model
and did the opposite. The most probable reading is that the scratch is not free
bandwidth -- it **contends for the same L3 the weights are streaming through**,
and the fused kernel at `m <= MR` has no reuse to absorb the loss -- so more
bands buys smoother streaming with the very resource the streaming needs. That
is a hypothesis consistent with the data, not a measurement; nothing here
isolates the scratch's cache footprint from the band count.

## What this does and does not rule out

**What it rules out, narrowly:** *this* mechanism -- a `k` split with private
`m * n` accumulators and a reduction -- does not beat the column split on this
host, at any band granularity tried, at any thread count. `qgemm` keeps the
column split.

**What it does not rule out.** The experiment changed **two** things at once:
(a) the access pattern, to streaming whole rows, and (b) the addition of private
scratch plus a reduction. The best available explanation of the v2 result is
that **(b)** is what hurt. So (a) -- the streaming layout itself, which is the
actual hypothesis -- was never measured in isolation, and this matrix cannot
convict it. Untouched levers that are still memory-layout work:

* **Prepacked / reordered `B`.** Gives the streaming layout with *no* scratch
  penalty at all, by paying the reordering once outside the call. This is the
  obvious next form of the same idea and it is untested here.
* **Huge pages.** The hypothesis is literally "a stride past a 4 KiB page is
  where the prefetcher stops". 2 MiB pages test exactly that and were not tried.
* **Software prefetch** into the column stripe; **NUMA first-touch and
  pinning** (this is a two-socket-like EPYC part and `B`'s placement is
  unaddressed); a **finer band-granularity sweep** than the two points taken;
  and gating the split to **only** the case the hypothesis names, a column
  stripe under one page.

**What it is consistent with.** Earlier this cycle an aspect-ratio sweep at
fixed footprint found 12.85 MB of `B` runs at 17-20 GB/s at *every* aspect
ratio, and a **1 MB L2-resident** weight at the same 19.5 GB/s; only past L3
(58.7 MB) does it fall to 7.2 GB/s. That sweep is the evidence that the fused
decode kernel is **instruction-bound inside L3**, and it stands on its own. This
experiment is *consistent with* it and would have been awkward for it had the
`k` split won -- but a confounded A/B is corroboration, not a second independent
measurement, and it is not cited as one.

**The planning consequence, stated at the strength the evidence supports.** The
instruction budget is the known-real term: exact full-range `u8 x i8` on AVX2
without VNNI must use `vpmaddwd`, because `vpmaddubsw` computes
`sat_i16(a0*b0 + a1*b1)` and is exact only when one operand stays within +/-64
(`255 * 64 * 2 = 32640` fits `i16`; `255 * 65 * 2 = 33150` does not) -- which is
why ORT's quantizer ships `reduce_range` for non-VNNI AVX2. That costs 2 loads +
2 `vpmovzxbw` + 2 unpacks + 2 `vpmaddwd` + 2 `vpaddd` per 32 bytes of `B`:
**~0.31 total uops/byte, ~0.25 counting vector uops only** (the figure section 16
quotes). Paths that cut **bytes and uops per weight together** -- the
packed-nibble `int4 x int16` kernel consumes 0.5 B/weight in place -- attack a
term that is measured, so they are the better next spend. That is a priority
call, not a proof that layout work is dead.

## Disposition

* Code: **#1617**, opened for reproducibility and **closed unmerged**. Branch
  `squad/roy-qgemm-ksplit`, commit `f21524d08`.
* No production behaviour changed. `qgemm` keeps the column split.
* Retained for reuse if a future prepacked layout changes the arithmetic:
  the wrapping-`i32` band-order argument (a `k` split is *bit-identical*, not
  merely close) is a real and reusable result.

## Open items

* The `1x512x4096` `t = 32` 0.17x anomaly in v2 is unexplained, and the same
  cell was a 1.05 win in v1.
* `1x1024x3072` has a **-29% / +14%** binary-layout swing at `t = 1` across the
  two matrices, on code that is identical at one thread. Any future claim on
  that shape needs a null control, not just reps -- and any harness change that
  narrows that floor would make this shape usable again.
* The streaming layout has not been tested without the scratch penalty. A
  prepacked `B` is the way to do that, and is the open question this experiment
  leaves behind rather than answers.
* Nothing here measures a pool shared with other sessions; all numbers are a
  dedicated pool on a shared host.
