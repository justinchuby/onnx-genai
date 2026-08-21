# Int4 acc4 N-tile: design, analytic bound, and two closed hypotheses

**Date:** 2026-08-21 · **Owner:** Roy (CPU MatMul) · **Host:** AMD EPYC 9V74,
32 vCPU (16c x 2 SMT), AVX2/FMA/F16C, **no AVX-512/VNNI**, 64 MiB L3.

**Verdict: the N-tile is designed and analytically bounded, but not built.**
Two cheaper hypotheses raised along the way were tested and **both are closed
negative**. The one shippable artifact here is benchmark-coverage repair: the
accuracy-4 packed-nibble route (#1619) had **no decode-loop row at all**.

---

## 1. Coverage repair (the shippable part)

`int4_decode_loop_ab` hard-coded `accuracy_level = 0`. Accuracy 4 is the *only*
value that reaches the packed-nibble kernel, so the route merged in #1619 was
measurable only through single-op benches -- never through a steady-state
decode loop, which is the shape of measurement its own gate (`m <= 64`,
decode-dominated) is designed for. `PROBE_ACCURACY` adds that axis.

This is a precondition for deciding the N-tile at all: there was previously no
harness in which an N-tile could be A/B'd against #1619 in decode.

## 2. Closed hypothesis A -- "the kernel does not thread-scale"

An early sweep of `RAYON_NUM_THREADS` (1/2/4/8/16) returned 22.9 / 3.5 / 3.8 /
3.4 / 3.3 ms/token -- flat past 2 threads -- and an accuracy-0 control that was
flat across the board. Read naively this says the kernel hits a shared wall and
that instruction-level work is pointless.

**It was an instrument error.** `configured_decode_threads` reads
`available_parallelism()` and `ONNX_GENAI_CPU_DECODE_THREADS`. It never reads
`RAYON_NUM_THREADS`. The sweep held the pool width fixed while appearing to
vary it; the "flat line" was one configuration measured five times.

Sweeping the real knob, block 32 / accuracy 4:

| `ONNX_GENAI_CPU_DECODE_THREADS` | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| ms/token | 22.949 | 22.893 | 11.584 | 5.866 | 3.302 | see §4* |

The kernel scales. Three interleaved repetitions confirm t=8 -> 5.872/5.910/
5.906 and t=16 -> 3.344/3.328/3.297: **1.77x** by median (5.906/3.328) and by
mean (5.896/3.323), **+-0.7%** (t=16, the worse arm: half-range 0.0235 on a
mean of 3.323). The single-shot table above gives 5.866/3.302 = 1.78x; take
1.77x as the measured figure, since it rests on the repeated arms.

*The t=32 column is deliberately not given a headline value: its samples are
1.801 / 3.178 / 3.170, with no stable central value. See §4.

**Two things this table does not explain.** First, **t=1 and t=2 are
identical** (22.949 vs 22.893, 1.002x) while 2->4 and 4->8 are clean 1.98x
doublings. That shape is what a granularity floor looks like -- the pool
declining to split at width 2 (`output_chunk_len` returns `n` when
`tasks < 2`, and `MIN_OUTPUTS_PER_TASK` / `MIN_PARALLEL_DOT_PRODUCTS_PER_THREAD`
impose minimum task sizes) -- but I did not confirm it, and it is recorded here
as an open question rather than an explanation. No conclusion in this document
depends on it. Second, scaling decays past 16 (§4 could not measure t=32
reliably).

**Lesson, now written into the bench header: verify that the knob you are
sweeping moves the thing you think it moves.** A wrong "does not scale" verdict
is one env-var name away.

## 3. Closed hypothesis B — "production is stuck on the narrow flat pool"

`configured_decode_threads` sizes the flat pool from `default_decode_threads`,
which on a 32-vCPU host resolves to `min(ceil_log2(32) + 1, 8, 32)` = **6**,
while `configured_persistent_decode_threads` uses `default_persistent_threads`
= `available / 2` = **16**. If decode landed on the 6-wide flat pool, §2's
numbers would mean production leaves well over 1.77x on the table — a far
larger lever than any N-tile, and nearly free.

Measured at the **default** width, no `ONNX_GENAI_CPU_DECODE_THREADS` override:

| `PROBE_SPMD` | `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL` | rep 1 | rep 2 |
|---|---|---|---|
| 1 | unset (default) | 3.298 | 3.313 |
| 0 | unset (default) | 3.316 | 3.300 |
| 0 | `0` | 3.293 | 3.226 |
| 0 | `1` | 3.417 | 3.333 |

Every reachable configuration lands on 3.23–3.42 ms/token, which is t=16
(3.297–3.344) and **nowhere near** t=8 (5.87–5.91), let alone a 6-wide flat
pool. **The hypothesis is false and the lever does not exist.** No change to
the pool defaults is proposed.

**What this table does *not* show, stated plainly.** It is *not* a
persistent-vs-flat comparison, and an earlier draft of this document wrongly
labelled the `PROBE_SPMD=0` rows "flat". `PERSISTENT_POOL_DEFAULT` is
`AtomicBool::new(true)`, so with the env unset `persistence_mode()` is `On` and
`is_forced()` returns true; `spmd_pool_eligible` is then true regardless of
`PROBE_SPMD`, and both of the first two rows run the **same** persistent pool.
Setting the env to `0` does not rescue the comparison either — the resulting
rows are still ~3.3 ms, so that path is not the 6-wide flat pool either.
**I could not construct a genuine flat-pool arm from this bench, so no arm here
is labelled flat.**

The surviving claim is the weaker but sufficient one: *every decode pool
configuration reachable from this harness runs 16-wide-equivalent*, which is
what kills the hypothesis. The mechanism — `default_persistent_threads(32) = 16`
and `is_forced()` — is read from the source, not inferred from the timings.

This correction is the same class of error §2 catches, committed one section
later: **`PROBE_SPMD` did not move the pool.** It was found by adversarial
review, and the tell was arithmetic, not code — a 6-wide pool cannot produce a
16-wide time, so two arms reading 3.3 ms could not be two different pools.

## 4. What the contended host refused to tell us

Later runs degraded badly: block 128 at t=32 returned 9.035, 2.355, 2.389,
1.135, 2.364 ms/token across five interleaved repetitions -- an **8x** spread.
Block 32 at t=32 gave 1.801 then 3.178/3.170. t=8 and t=16 stayed within
+-0.7% throughout; only the all-vCPU configuration is unstable, which is what
sharing a 32-vCPU host with other jobs looks like.

Consequently **two questions are left open rather than answered wrongly**:

- **Is the kernel near the DRAM roofline?** The projection set
  (qkv 4096x6144, o 4096x4096, gate/up 4096x14336, down 14336x4096 = 218.1 M
  weights) is 109.05 MB packed + 27.26 MB f32 scales = **136.3 MB/token** as
  *this bench actually runs it*, comfortably past the 64 MiB L3. Note the
  harness passes three inputs (activation, B, scales) and no zero-point tensor,
  so the measured configuration is **symmetric**; an asymmetric weight would
  add 3.41 MB of zero points for 139.7 MB, and that larger figure is the right
  one for the §5 traffic argument but the wrong one here. Dividing the measured
  136.3 MB by the quiet t=32 sample gives 75.7 GB/s; by the contended ones,
  ~43 GB/s. The first is ~100% of this host's oft-quoted 75.8 GB/s figure, and
  block 128's quiet sample computes to ~136% of it. A ratio at or above 100%
  means the *denominator* is wrong, not that the kernel is superhuman.
- **Does time track bytes across block size?** Medians say 1.35x for
  32-vs-128 against a 1.20x byte ratio, but under an 8x spread that is not a
  measurement.

**Standing rule, applied to myself here: a bandwidth percentage means nothing
until the denominator is shown to bind.** This is exactly the error that let
the 2026-08-20 "dead on AVX2" verdict stand (an L3-resident working set
measured against a DRAM ceiling), and it was corrected once already in ledger
§19. Publishing "76.6% of roofline" off a borrowed 75.8 GB/s constant would
have been the same mistake a third time.

## 5. The N-tile design, and why it was not built on this evidence

**Design.** Keep the weights N-major -- 4 strided but sequentially walked
streams, which the prefetcher handles -- so the 29 MB weight is *never*
repacked. Add only two block-major side arrays indexed `[block][column]`:

- **scales**, `N * k_blocks * f32` (already exactly that size today, merely
  transposed), giving one contiguous 4-column load per block;
- **zero points, pre-expanded to one byte per (n, block)**, `N * k_blocks`
  bytes (~1.8 MB for the llama MLP shapes).

The zero-point expansion is worth calling out independently of the tile:
today's per-tile assembly is
`_mm_setr_epi32(low & 0xf, low >> 4, high & 0xf, high >> 4)`, roughly **8
scalar uops**, which a pre-expanded array turns into one load plus
`_mm_cvtepu8_epi32` = **2 uops**. The tail also uses `_mm_mullo_epi32`, 2 uops
and 10-cycle latency.

Per (block, 4 columns) the tile then shares 2 activation loads, folds one
`hadd` tree across the 4 columns, and turns the activation scale and block sum
into broadcasts.

**Registers/L1/L2.** 4 columns x one `__m256i` accumulator + 2 activation
registers + 2 weight registers fits the 16 YMM registers with room for the
`hadd` tree; 8 columns does not, without spilling the accumulators. The
block-major scale/zp stripes for a 4-column tile are 16 B + 4 B per block, so a
k=4096/bs=32 column group touches 128 blocks x 20 B = 2.5 KB of side data
against 32 KB L1 -- not a constraint. Tails: `N % 4` columns fall to the
existing single-column driver; `k_blocks % 4` already falls to the scalar
remainder loop, unchanged.

**Analytic bound.** From ledger §18's fit `time = blocks*A + weights*B` with
`A = 8.5` cycles per (block, column) and `B = 13.4` weights/cycle, the
per-block term is **78% of runtime at block 32**. The current 4-block k-tile
costs ~62 uops per 4 (block, column) pairs = 15.5 each; the 4-column N-tile
projects ~53 uops = 13.25 each, a **~15%** reduction (~27% under a more
generous accounting).

**Why that is not sufficient to build on.** Three reasons, in order of weight:

1. **The uop model is known to be wrong by 2.8x.** It predicts ~3.9 cycles per
   block where measurement says 10.9. A model that mispredicts absolute cost by
   that much cannot adjudicate a projected 15% delta.
2. **The projected win is close to the noise floor.** The published
   single-cell floor for this harness is +-5.5% on a quiet host; today's host
   is at +-0.7% for t<=16 but 8x at t=32. A 15% effect is measurable at t=16
   and invisible at t=32.
3. **It reduces instructions, not bytes.** The N-tile removes activation
   *re-reads* (L1 traffic) and shared arithmetic; it does not shrink the
   139.7 MB/token that must cross the memory system. If the kernel is anywhere
   near bandwidth-bound at production width, the tile's ceiling is small -- and
   §4 is precisely the measurement that would have settled this, and it failed.

**A byte-level lever is the natural alternative, and it is not the N-tile.** At
block 32 the f32 scales are **27.26 MB against 109.05 MB of packed weights**;
with an asymmetric weight, scales plus zero points are **22% of all bytes
moved** (30.67 of 139.72 MB). Storing scales as bf16/f16 in the same
block-major prepack removes ~10% of total traffic, which converts roughly 1:1
into time *wherever the kernel is bandwidth-bound*, and does not touch the
integer dot's exactness (only the per-block f32 multiply, whose tolerance the
acc4 envelope already governs).

**But it cannot be ranked above the N-tile on current evidence, and saying
otherwise would be the same overreach this document is trying to avoid.** The
two levers bet on *opposite* unmeasured regimes: the byte lever pays off only
if the kernel is bandwidth-bound, while reason 3 above discounts the N-tile
only if it is. §4 is exactly the measurement that would decide which regime
holds, and it failed. So the honest ordering is: **establish the regime first,
on a quiet host, then pick the lever.** The byte lever is cheaper to build and
settles its own question faster, which is an argument for trying it first — not
an argument that it is worth more.

**MLAS's `SQ4BitGemmM1Kernel_CompInt8_avx2` uses `NCols = 4`**, so the N-tile
is not a bad idea and this is not a negative result about the design. It is a
statement that on this host, today, the evidence needed to justify ~500 lines
of unsafe intrinsics does not exist, and manufacturing it from a 2.8x-wrong
uop model would be dishonest.

## 6. Reproduce

```bash
cargo build --release -p onnx-runtime-ep-cpu --bench int4_decode_loop_ab
BIN=target/release/deps/int4_decode_loop_ab-*

# thread width (the real knob; RAYON_NUM_THREADS does nothing here)
for t in 1 2 4 8 16 32; do
  ONNX_GENAI_CPU_DECODE_THREADS=$t PROBE_ACCURACY=4 PROBE_BLOCK=32 PROBE_TOKENS=48 $BIN
done

# default width, across every reachable pool configuration
for s in 1 0; do PROBE_SPMD=$s PROBE_ACCURACY=4 PROBE_BLOCK=32 PROBE_TOKENS=48 $BIN; done
for e in 0 1; do
  ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=$e PROBE_SPMD=0 \
    PROBE_ACCURACY=4 PROBE_BLOCK=32 PROBE_TOKENS=48 $BIN
done
```

`PROBE_SPMD` alone does **not** toggle the pool: `PERSISTENT_POOL_DEFAULT` is
`true`, so `is_forced()` holds and the persistent pool wins regardless. That is
why §3 declines to label any arm "flat".

Interleave repetitions rather than running one arm to completion; this host's
all-vCPU configuration drifts by up to 8x between neighbouring runs.
