# The int4 pack's eliminated modulo: the matrix #1809 could not finish

**Date:** 2026-08-25 · **Refs:** #1809 (merged `71bbec062`), #1676
**Reproduce:** `crates/onnx-runtime-ep-cpu/benches/int4_modulo_arms.sh`, then
`int4_modulo_matrix.py`

## What this corrects

#1809 removed the per-group integer division from
`Int4Weight::dequant_panel_avx2`, the innermost loop of the int4 GEBP B-panel
pack, and reported it as **1.015x on block-16 decode and a null on prefill**.

Its prefill rows were `m = 64/256/512`. The two smallest, `m = 1` and `m = 8`,
**failed their A/A null at 5.31% and 4.62% and were withheld**. The pack is
amortized over `m` rows, so the mechanism puts the effect at *small* `m`: the
sweep had a hole exactly where the answer was.

Three corrections come out of filling it in.

1. **Prefill is not a null.** At block 16 the elimination is worth
   1.0067/1.0060/1.0046 at `m = 1/8/16`, decaying to an exact null by `m = 64`.
   Block 32 agrees. Every A/A interval in both sweeps brackets 1.000.
2. **Decode is ~1.010x, not 1.015x**, and the instrument is worse than either
   number's interval suggests: the decode-loop A/A null **excludes 1.000 in
   both sweeps taken here, with opposite signs** (+0.63% at 21 launches, −0.28%
   at 41). Its real floor is about ±0.6%.
3. **A source-level A/B on this codebase carries a per-build code-layout
   component that reaches ~2%, and no same-binary A/A can see it.** Found by a
   control that was supposed to be boring. Detailed below, because it bears on
   every A/B in this directory and not just this one.

## Matrix

`2048x2048`, `bits=4`, `accuracy_level=0`, pinned to one physical core (cpu 4),
**61 independent launches per arm**, arms interleaved and rotated per round,
percentile bootstrap over launches (20 000 resamples, seed 20260825). 0
launches discarded by the CPU-efficiency gate. Ratio is `before / after`, so
above 1.000 means the elimination is faster.

### Block 16 — GEBP at every row, so the whole sweep is one kernel

| m | before ms | after ms | speedup | 95% CI | verdict | A/A | A/A 95% CI |
|---:|---:|---:|---:|---|---|---:|---|
| 1 | 2.110 | 2.096 | **1.0067** | [1.0038, 1.0096] | **gain** | 1.0014 | [0.9990, 1.0081] |
| 8 | 2.685 | 2.669 | **1.0060** | [1.0037, 1.0094] | **gain** | 1.0019 | [0.9989, 1.0060] |
| 16 | 3.273 | 3.258 | **1.0046** | [1.0025, 1.0077] | **gain** | 1.0006 | [0.9982, 1.0037] |
| 32 | 5.002 | 4.992 | 1.0020 | [0.9996, 1.0040] | null | 1.0012 | [0.9982, 1.0034] |
| 64 | 7.922 | 7.922 | 1.0000 | [0.9974, 1.0049] | null | 0.9999 | [0.9946, 1.0027] |
| 128 | 14.381 | 14.376 | 1.0003 | [0.9985, 1.0029] | null | 1.0012 | [0.9985, 1.0034] |
| 256 | 26.662 | 26.658 | 1.0002 | [0.9984, 1.0018] | null | 1.0004 | [0.9955, 1.0021] |
| 512 | 51.830 | 51.854 | 0.9995 | [0.9978, 1.0013] | null | 0.9958 | [0.9935, 1.0008] |

### Block 32

| m | before ms | after ms | speedup | 95% CI | verdict | A/A | A/A 95% CI |
|---:|---:|---:|---:|---|---|---:|---|
| 1 | 0.713 | 0.727 | **0.9807** | [0.9794, 0.9835] | **loss — see below** | 1.0000 | [0.9973, 1.0014] |
| 8 | 2.424 | 2.401 | **1.0096** | [1.0067, 1.0113] | **gain** | 1.0012 | [0.9979, 1.0037] |
| 16 | 3.006 | 2.988 | **1.0060** | [1.0037, 1.0087] | **gain** | 0.9983 | [0.9967, 1.0010] |
| 32 | 4.748 | 4.727 | **1.0044** | [1.0015, 1.0076] | **gain** | 0.9998 | [0.9979, 1.0023] |
| 64 | 7.660 | 7.659 | 1.0001 | [0.9983, 1.0039] | null | 0.9980 | [0.9957, 1.0012] |
| 128 | 14.110 | 14.106 | 1.0003 | [0.9979, 1.0025] | null | 0.9998 | [0.9976, 1.0012] |
| 256 | 26.399 | 26.386 | 1.0005 | [0.9981, 1.0024] | null | 0.9992 | [0.9975, 1.0015] |
| 512 | 51.568 | 51.556 | 1.0002 | [0.9949, 1.0030] | null | 0.9990 | [0.9972, 1.0022] |

**Every A/A interval brackets 1.000 at every row of both sweeps.** The harness
checks this itself and warns if it ever stops being true, because a row whose
A/A excludes unity has an instrument bias and its verdict is not readable.

### Decode loop, block 16

Driven through the decode-loop harness rather than the single-op one, 32 tokens
per launch:

| launches per arm | speedup | 95% CI | A/A | A/A 95% CI |
|---:|---:|---|---:|---|
| 21 | 1.0116 | [1.0094, 1.0123] | 1.0063 | [1.0045, 1.0081] |
| 41 | 1.0095 | [1.0073, 1.0109] | 0.9972 | [0.9951, 0.9994] |

**Both A/A intervals exclude 1.000, and they do so in opposite directions.**
That is not a bias to divide out; it says the estimator is unstable at this
sample size under the decode loop's per-launch bimodality, and that the true
floor is around ±0.6% rather than the ±0.2% either interval advertises.

The defensible decode statement is therefore **≈1.01x**, not #1809's 1.015x —
and it is corroborated by an instrument that *does* pass its own null: block-16
`m = 1` in the single-op harness above is the same route, and reads
1.0067 [1.0038, 1.0096] with a clean A/A.

## Why the shape of the curve is the result

Both block sizes give a monotone decay in `1/m`, reaching an exact null and
staying there for four consecutive rows:

```
block 16   1.0067  1.0060  1.0046  1.0020  1.0000  1.0003  1.0002  0.9995
block 32      —    1.0096  1.0060  1.0044  1.0001  1.0003  1.0005  1.0002
m              1       8      16      32      64     128     256     512
```

That is not sixteen independent measurements that happened to include six
positive ones. It is the amortization curve the mechanism predicts, sampled at
eight points, in two block sizes, from the same pair of binaries.

It is also the argument that rules out code layout, which is the one competing
explanation a rebuild-between-arms A/B cannot dismiss by construction. **The
same `quant_prefill_gebp` runs at every row of the block-16 sweep.** A layout
difference between the two binaries is a fixed property of that code; it cannot
be +0.6% at `m = 1` and exactly 0.0% at `m = 64` through `m = 512`. What does
scale that way is the pack, which runs once per panel however many rows are
multiplied against it.

## The control that was supposed to be boring

Block 32 at `m = 1` takes `borrowed_affine_int4_matmul_nblock` and never calls
the pack at all. The poisoned build confirms this rather than asserting it: at
that row it is **bit-identical** to `after` under a build that is deliberately
wrong. The two binaries therefore differ, on that row, only in code that does
not execute.

It reads **0.9807, CI [0.9794, 0.9835]** — a 1.9% *loss*, reproduced at
0.9821 [0.9807, 0.9862] under `ONNX_GENAI_CPU_MM_INT4_GEBP=0`, with its own
A/A sitting at 1.0000 [0.9973, 1.0014].

**An earlier pair of binaries, built from the same source change against a
main from three commits earlier, read 1.0000 and 1.0028 [0.9986, 1.0042] on
that same row.** Same change, same row, same route-not-taken: +0.3% one build,
−1.9% the next.

So this is code layout, not a property of the change — and that is the finding,
because:

* **It is invisible to an A/A.** The A/A arm is the *same file*, so it measures
  everything except the thing that differs between builds. Every A/A in this
  document brackets 1.000 while a 1.9% artifact sits in the same table.
* **It is larger than most kernel results this repository ships.** Any
  source-level A/B here whose claim is under ~2%, on a route with no
  route-not-taken control, cannot distinguish its result from this.
* **It is not stable across rebuilds**, so it cannot be calibrated out once and
  reused. It has to be re-measured per build, which is what a route-not-taken
  row does for free when one exists.

The practical rule: **include a row the change provably cannot reach, prove it
with a poisoned build, and read it as the experiment's real floor.** Where no
such row exists — which is the usual case — a sub-2% single-block-size result
should be treated as unconfirmed until it reproduces with a different
amortization slope, as this one does across two block sizes.

## Route proof, per row

Timing cannot tell "the change ran and cost nothing" from "the change never
ran", and a source-level A/B rebuilds between arms, so *which arm ran* is an
assumption unless something makes it an observation.

`int4_prefill_route_ab` prints an FNV-1a fold over the raw output bytes per row,
and `int4_modulo_matrix.py --route-proof` builds a deliberately poisoned third
arm that drops the `+ q` term:

```
 block     m  before==after  poison moves
    16   1..512      True          True
    32     1         True         False   <-- control: pack not on this route
    32   8..512      True          True
route proof: PASS
```

Exit 0. Two things fall out:

1. **`before == after` on all 16 rows, both block sizes.** The elimination is
   exact, as the algebra says, so the speedup is numerically free rather than
   bought.
2. **The poison moves exactly where the pack is on the route and nowhere
   else.** The one row that stays bit-identical under a deliberately wrong
   build is what proves the poison is not simply perturbing the whole binary.

Decode checksums are `844.536810` in all three arms — the constant #1809
recorded, unchanged by #1729 and #1794.

## Mechanism

The identity needs no assumption about `block_size`, in particular not that it
is a power of two, which this kernel's contract does not guarantee and its own
tests falsify at `block_size` 24 and 40:

* `run` is capped at `block_size - offset_base`, so `offset_base + q < block_size`
  for every `q < run`;
* `depth - offset_base` is a multiple of `block_size`.

Hence `(depth + q) % block_size == offset_base + q` exactly.

`div`/`idiv` in `dequant_panel_avx2`, disassembled from the executables that
produced the numbers above rather than from a separately emitted `.s`:

| arm | divisions |
|---|---:|
| before | 6 |
| after | **4** |
| poison | 4 |

LLVM cannot do this itself: `block_size` is a runtime field, and a
`#[target_feature]` function is never inlined into a caller that might have
narrowed it, so constant propagation never reaches the body. Different
mechanism from #1783, where the value was a literal at every call site and only
the inlining barrier hid it.

## Method

**61 launches, not one careful pairing.** The per-launch spread on this host is
enormous and the median is not — at block 32 `m = 1` the spread between
launches of the *same* binary reaches 102% while 61 of them agree on a median
to 0.14%. A single paired A/B, however tight its intra-run spread looks, can be
dominated by which mode each side's launch landed in.

**The host gate is the CPU efficiency of the run itself** — `os.wait4` rusage
`(utime + stime) / wall`, launches below 0.95 discarded — not an instantaneous
runnable count sampled at run boundaries. A short run has room for a burst that
starts after the opening sample and ends before the closing one; that is how a
52% A/A null passed a "host clean" check during #1809. Every launch in this
document passed the gate.

**The null arm is a separate file**, not a second run of the same path, so it
is a genuinely independent launch that pays every per-launch cost the real arms
pay. And `int4_modulo_arms.sh` fails hard if any two arms come out
byte-identical: two identical arms still produce a full table of numbers, and
every one of them is a null between a binary and itself.

The whole sweep runs under `scripts/hostlock.sh`, held for its duration by the
measuring process. The pin is cpu 4 — not cpus 0–1, where cpu 0 has a permanent
external competitor on this host and cpu 1 is its SMT sibling.

## Disposition

**No kernel change.** #1809's code is correct and already on main; what was
incomplete was its account of where the change pays. Corrected scope:

* **prefill `m = 1/8/16`: 1.005x–1.010x**, both block sizes, A/A clean
* **prefill `m >= 64`: null**, now bounded rather than asserted
* **block-16 decode: ≈1.010x**, against an instrument floor of ~0.6% — real,
  but smaller than the 1.015x #1809 reported
* **an incidental ~1.9% layout loss on the block-32 `m = 1` decode route in
  this build**, which is not attributable to the change and will not survive
  the next unrelated commit, but which bounds what any sub-2% A/B in this
  directory can claim

Shipped here: the per-row `fnv` route fingerprint in `int4_prefill_route_ab`,
the bootstrap and A/A self-check in the harness, and the two scripts that make
all of it reproducible.
