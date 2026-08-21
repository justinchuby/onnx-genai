# QLinearMatMul u8: instruction decomposition, the `m = 5` cliff, and the
# `vpmaddubsw` legality question

**Date:** 2026-08-21 · **Owner:** Roy (CPU MatMul) · **Host:** AMD EPYC 9V74,
32 vCPU (16c x 2 SMT), AVX2/FMA/F16C, **no AVX-512/VNNI**, 64 MiB L3.
`perf` is unavailable on this host, so every uop count below is a static read of
the emitted intrinsics, not a hardware counter. Where a static count and a
measurement disagree, the measurement wins and the disagreement is stated.

**Shipped: one mechanism, and it is narrow.** Serial `QLinearMatMul` at
`MR < m <= 2 * MR` gets **1.16-1.48x** (`m = 8` to `m = 5`), across two
independent runs. **`m = 1` is not improved by this
change at all** -- see §4 for what it would take, and why that is a separate,
unbuilt piece of work.

---

## 1. Instruction decomposition of the pack-free (`m <= MR`) kernel

`fused_strip::<R, T, SIGNED>` per (`k` pair, 16-column tile):

| step | intrinsics | uops |
|---|---|---|
| load + widen `b[k][c..c+16]` | `_mm_loadu_si128` + `_mm256_cvtepu8_epi16` | 2 |
| load + widen `b[k+1][c..c+16]` | same | 2 |
| interleave to `k` pairs | `_mm256_unpacklo_epi16`, `_mm256_unpackhi_epi16` | 2 |
| arithmetic, per row | `_mm256_madd_epi16` x2, `_mm256_add_epi32` x2 | 4R |

So **`6 + 4R` uops per `R` rows of 32 multiply-accumulates**:

| | `R = 1` (decode) | `R = 4` |
|---|---|---|
| uops / MAC | 0.3125 | 0.172 |
| share of uops that is arithmetic | **40%** | 73% |

**At `m = 1`, 60% of issued uops are the load/widen/interleave tax and only 40%
do arithmetic.** That is the structural statement of the `m = 1` problem: the
six-uop preamble is fixed per 32 weights and has nobody to amortise against.
The packed kernel's inner loop (`tile::<R>`) is `2 + 4R` uops for the same work
because its panel is already widened and interleaved -- 0.1875 uops/MAC at
`R = 1`, **1.67x denser** -- which is exactly what it buys with the pack.

## 2. Is `m = 1` bandwidth-bound or issue-bound?

Bandwidth-bound would mean instruction work is pointless, so this is settled
before anything else. Two measurements, one thread:

- `1x2048x2048` and `4x2048x2048` **read the identical 4.19 MB of `B`**. If `B`
  traffic bound the kernel their times would match. They are 0.199 ms and
  0.550 ms -- **2.86x apart**, tracking arithmetic, not bytes. Not
  bandwidth-bound.
- Throughput across `B` footprint at `m = 1`: 4.54 / 11.68 / 17.99 / 21.08 /
  **14.72** GMAC/s at `n = k =` 256 / 512 / 1024 / 2048 / 4096. It climbs while
  fixed per-call cost amortises, then **falls at 4096**, where `B` is 16.8 MB.
  So memory does begin to bind, but above the shapes in question, not at them.

**Verdict: issue-bound at `m = 1` for `k = n <= 2048`.** The static model
predicts 12.8 MACs/cycle at 4 uops/cycle against a measured 21 GMAC/s, which
would imply a ~1.6 GHz clock -- below this part's base. So the kernel is
running at roughly two-thirds of its issue ceiling and the remainder is
latency, not width. **I could not attribute that remainder without `perf`, and
I am not going to guess at it.**

## 3. The `m = 5` cliff (the shipped fix)

`fused = m <= MR` sent `m = 5` to the packed path, which then had two row
blocks and nothing to amortise a `2 * k * n`-byte panel write against. The
packed path costs 0.543 ms at `m = 4` and 1.090 ms at `m = 5` -- a **2.0x
cliff for 25% more arithmetic**. Measured
at `k = n = 2048`, one thread, packed vs pack-free, two repetitions:

| `m` | 4 | 5 | 6 | 8 | 9 | 10 | 11 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| pack-free speedup | 0.99x | **1.48x** | **1.42x** | **1.16x** | 1.06x | 1.02x | 0.90x | 0.94x | 0.84x |

Every cell is the mean of two repetitions divided by the mean of two. `m = 4`
is a **null control**: both arms run the identical path there, so its 0.99x is
this harness's one-thread noise floor -- **about 1.5%**, and 0.2% on the `m = 1`
control in the confirmation run below. That floor is what the `m = 8` claim has
to clear, and 16% clears it by an order of magnitude.

The pack does repay itself -- at about ten rows, not five. Fix: give the
pack-free kernel a row-block loop (it previously dispatched on exact `m` and
physically could not take a fifth row) and move the boundary to `2 * MR`.

**First reversal, and it inverted the answer.** My first draft set the boundary
at 16 on the strength of the table above. Re-measured across the pool it
**loses**:

| 8 threads, `m` | 5 | 8 |
|---|---|---|
| pack-free speedup | **0.87x** | **0.79x** |

Serially the pack is latency on the critical path, so skipping it wins. Across
the pool each worker has its own issue width but they share one memory system,
so what binds is the pack-free path's extra sweep of `B` per row block, and the
packed panel -- written once, re-read from L2 -- wins from `MR + 1` up. The
same shape wants **opposite answers at one thread and at eight**.

So the gate is `m <= MR || (!parallel && m <= 2 * MR)`. Final state -- a
**separate confirmation run** from the boundary sweep above, so its ratios
differ slightly from that table's (`m = 5/6/8` land at 1.46x/1.37x/1.16x here
against 1.48x/1.42x/1.16x there). Both runs are reported rather than the more
flattering one; the honest headline is the range across them, and for `m = 6`
that is **1.37-1.42x**. Two repetitions:

| shape | 1 thread before | 1 thread after | 8 threads before | 8 threads after |
|---|---|---|---|---|
| `1x2048x2048` | 0.205 / 0.205 | 0.206 / 0.205 | 0.071 / 0.093 | 0.093 / 0.083 |
| `5x2048x2048` | 1.072 / 1.091 | **0.737 / 0.744** | 0.214 / 0.227 | 0.220 / 0.221 |
| `6x2048x2048` | 1.139 / 1.157 | **0.837 / 0.834** | 0.249 / 0.245 | 0.242 / 0.232 |
| `8x2048x2048` | 1.257 / 1.256 | **1.084 / 1.076** | 0.274 / 0.274 | 0.264 / 0.270 |
| `12x2048x2048` | 1.516 / 1.551 | 1.522 / 1.508 | 0.337 / 0.343 | 0.347 / 0.321 |

`m = 1` and `m = 12` are unchanged by construction (both stay on the path they
were already taking) and measure unchanged -- 0.998x and 1.013x, two more null
controls.

**Two different noise floors, and they must not be confused.** At one thread
the same-path controls (`m = 1`, `m = 4`, `m = 12`) repeat to within **1.5%**.
At 8 threads the two arms are also the same code, and there they spread by
**about 10%** -- so the 8-thread columns above say nothing at all, and are
printed only to show the parallel regression is gone. Applying the 8-thread
figure to the one-thread cells would understate them by nearly an order of
magnitude; applying the one-thread figure to the 8-thread cells would
manufacture findings out of noise.

**Scope, stated honestly.** `m * n * k` for these shapes is far above
`PARALLEL_MIN_WORK`, so `!parallel` here means a genuinely single-worker pool.
This is a real configuration and a real win, but it is **not** the decode shape
and **not** the multi-threaded one.

### Confirmation on latest main

Re-measured after merging `origin/main` (`9cd2a7d13`) into the branch, since a
stale measurement is not evidence about the head that ships. Third independent
run, same protocol, two repetitions, one thread, arms in separate processes
(`fused_max_rows` is a `OnceLock`, so the two arms cannot be interleaved inside
one process):

| `m` | 1 | 4 | 5 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| off (ms) | 0.2053 | 0.5471 | 1.0898 | 1.1612 | 1.2805 | 1.5364 |
| on (ms) | 0.2024 | 0.5493 | 0.7456 | 0.8422 | 1.0766 | 1.5396 |
| speedup | 1.01x | 1.00x | **1.46x** | **1.38x** | **1.19x** | 1.00x |

Three null controls (`m = 1`, `4`, `12`) all land inside the 1.5% floor. Across
all three runs the win is **1.46-1.48x / 1.37-1.42x / 1.16-1.19x** at
`m = 5/6/8`.

## 4. `vpmaddubsw`: when it is legal, what it would buy, why it is not here

`vpmaddubsw` computes `sat_i16(u8 * i8 + u8 * i8)`. The saturation is the whole
question.

**It cannot consume this kernel's operands as they stand.** `A` is *centred*
(`a - za`) into `[-255, 255]`, which is not `u8`, so it cannot be the unsigned
operand. Making it legal means restructuring, not just swapping an intrinsic:
keep `A` **raw** `u8` and centre `B` instead, into `i8`, at prepack time:

```text
sum_k (a - za)(b - zb)  ==  sum_k a * B'  -  za * sum_k B'      where B' = b - zb
```

Both terms are exact integers and `sum_k B'` is a per-column constant computed
once at prepack, so nothing here weakens `u8` semantics.

**The legality bound.** With `a in [0, 255]`, a pair accumulates to at most
`255 * (|B'_k| + |B'_{k+1}|)`. Staying inside `i16` needs that `<= 32767`, i.e.
`|B'_k| + |B'_{k+1}| <= 128.5`. **Guaranteed when every `|B'| <= 64`** -- which
is precisely what reduce-range (7-bit) weight quantization produces, `[-64, 63]`.
At full range it is not merely tight, it is false: `255 * 127 * 2 = 64770`
saturates, and a saturating result is a wrong result, not an approximate one.

**Do not take this from metadata -- take it from the bytes.** `B` is constant on
the decode path, so the honest test is to *scan the actual weight once at
prepack* and check `max |b - zb| <= 64`. That is a proof about the tensor in
hand rather than trust in a producer's `reduce_range` flag, it costs one pass
over data the prepack already walks, and when it fails the kernel simply keeps
exact `vpmaddwd`. `reduce_range` stays explicit and no path is weakened.

**What it would buy.** The `k`-pair interleave is *not* avoided --
`B` is `[k][n]` row-major, so pairing `b[k][c]` with `b[k+1][c]` is required
whichever instruction consumes them. The win is doing that interleave in the
**byte** domain, where a 256-bit register holds 32 elements instead of 16:

| | current (`vpmaddwd`) | proposed (`vpmaddubsw`) |
|---|---|---|
| per 64 bytes of `B` | 20 uops | 2 load + 2 `unpack_epi8` + 2 `maddubs` + 2 `madd(ones)` + 2 add = 10 |
| uops / MAC at `R = 1` | 0.3125 | **0.156** |

A 2x issue-side reduction on a kernel §2 places at roughly two-thirds of its
issue ceiling. That is the single largest identified lever for `m = 1`.

**Not implemented, and not claimed.** It needs the centred-`i8` prepack, a
cached pack keyed on weight identity (the native path has **no** constant-`B`
pack cache today -- `pack_lookup`/`pack_build` are `#[cfg(feature = "mlas")]`),
the scan-and-verify gate, the `a_signed` case (raw `i8` cannot be the unsigned
operand either; it needs a `+128` offset and a second correction term), and its
own numerics and mutation coverage. Shipping the boundary fix from §3 does not
depend on any of it, so they are separate.

## 5. Reproduce

```bash
cargo test -p onnx-runtime-ep-cpu --lib --release bench_qgemm_ab -- --ignored --nocapture
```

Knobs: `QGEMM_AB_SHAPES` (`mxkxn`, comma-separated), `QGEMM_AB_THREADS`,
`QGEMM_AB_ITERS`. Arms: `ONNX_GENAI_CPU_QGEMM_FUSED_ROWS=0` restores the
historical `m <= MR`; unset is the shipped `2 * MR`. The harness prints a
`portable` null control per shape -- it is the same arithmetic with none of the
blocking, so if it moves between arms the box was busy and the run says nothing.

Alternate arms rather than running one to completion; this host is shared.
