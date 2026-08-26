# The int4 pack's eliminated modulo: the matrix #1809 could not finish

**Date:** 2026-08-25 · **Refs:** #1809 (merged `71bbec062`), #1676
**Reproduce:** `crates/onnx-runtime-ep-cpu/benches/int4_modulo_arms.sh`, then
`int4_modulo_matrix.py`. For an independent layout, rebuild the arms with
`RUSTFLAGS=-Cllvm-args=-align-all-functions=5 MOD_ARMS_OUT=...`

## What this corrects

#1809 removed the per-group integer division from
`Int4Weight::dequant_panel_avx2`, the innermost loop of the int4 GEBP B-panel
pack, and reported it as **1.015x on block-16 decode and a null on prefill**.

Its prefill rows were `m = 64/256/512`. The two smallest, `m = 1` and `m = 8`,
**failed their A/A null at 5.31% and 4.62% and were withheld**. The pack is
amortized over `m` rows, so the mechanism puts the effect at *small* `m`: the
sweep had a hole exactly where the answer was.

Three corrections come out of filling it in.

1. **Prefill is not a null.** The elimination saves a roughly *fixed*
   **0.017–0.033 ms per packed panel**, independent of `m`, which is 0.5%–1.4%
   at `m = 1/8/16` and disappears into the noise floor by `m ≈ 64`. It
   reproduces in **three independently built pairs of binaries**, one of them
   deliberately perturbed in layout only. Every A/A interval in every sweep
   brackets 1.000.
2. **Decode is ~1.010x, not 1.015x**, and the instrument is worse than either
   number's interval suggests: the decode-loop A/A null **excludes 1.000 in
   both sweeps taken here, with opposite signs** (+0.63% at 21 launches, −0.28%
   at 41). Its real floor is about ±0.6%.
3. **A source-level A/B on this codebase carries a per-build code-layout
   component that reaches ~2%, and no same-binary A/A can see it.** Found by a
   control that was supposed to be boring. Detailed below, because it bears on
   every A/B in this directory and not just this one — and because it is the
   reason a single build pair is not enough to establish a sub-2% result.

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
1.0067 [1.0038, 1.0096] here and 1.0115 [1.0076, 1.0134] on the
layout-perturbed build pair, both with a clean A/A.

## What the curve does and does not prove

Both block sizes give a monotone decay in `1/m`, reaching a null and staying
there:

```
block 16   1.0067  1.0060  1.0046  1.0020  1.0000  1.0003  1.0002  0.9995
block 32      —    1.0096  1.0060  1.0044  1.0001  1.0003  1.0005  1.0002
m              1       8      16      32      64     128     256     512
```

An earlier draft of this document claimed that shape **rules out** code layout,
on the grounds that the same `quant_prefill_gebp` runs at every row of the
block-16 sweep, so a layout difference in it could not be +0.6% at `m = 1` and
0.0% at `m = 512`. **That argument is wrong and is withdrawn.** A layout
difference in any *fixed-cost* region — setup, allocation, the pack itself —
costs a constant number of milliseconds regardless of `m`, and a constant
absolute cost divided by a total that grows with `m` produces exactly this
1/m ratio decay. The curve is consistent with the mechanism; it does not
discriminate between the mechanism and layout, because both predict it.

Worse, the two are the same *size* here. Read the table in milliseconds
instead of ratios:

| block | m = 1 | m = 8 | m = 16 | m = 32 | m = 64 | m = 512 |
|---|---:|---:|---:|---:|---:|---:|
| 16 | +0.014 | +0.016 | +0.015 | +0.010 | 0.000 | −0.024 |
| 32 | **−0.014** | +0.023 | +0.018 | +0.021 | +0.001 | +0.012 |

The claimed saving is ~0.015 ms. The layout artifact on the route-not-taken row
is −0.014 ms. One build pair cannot separate them.

## Three build pairs, one perturbed in layout only

What separates them is that layout does not survive a rebuild and the saving
does. Three pairs of arms, built from the same one-line change:

* **A** — against a `main` three commits older, and a bench source without the
  checksum assertions. 61 launches/arm.
* **B** — against current `main`. 61 launches/arm. The pair the tables above
  are taken from.
* **C** — current `main`, built with
  `RUSTFLAGS=-Cllvm-args=-align-all-functions=5` applied identically to every
  arm. 41 launches/arm. This changes nothing an instruction executes; it moves
  where functions start. 50.6% → 90.8% of `FUNC` symbols land on a 32-byte
  boundary, and the executable grows 33 KB of padding. **A designed layout
  perturbation**, rather than an accidental one.

Block 32, `speedup` and the same figure in absolute ms:

| m | A | B | C | A ms | B ms | C ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 *(route-null control)* | 1.0028 | **0.9807** | 1.0028 | +0.002 | **−0.014** | +0.002 |
| 8 | 1.0071 | 1.0096 | **1.0137** | +0.017 | +0.023 | +0.033 |
| 16 | 1.0064 | 1.0060 | 1.0090 | +0.019 | +0.018 | +0.027 |
| 32 | 1.0034 | 1.0044 | 1.0068 | +0.016 | +0.021 | +0.032 |
| 64 | 1.0000 | 1.0001 | 1.0048 | 0.000 | +0.001 | +0.037 |

Block 16, pair B against pair C:

| m | B | C | B ms | C ms |
|---:|---:|---:|---:|---:|
| 1 | 1.0067 | 1.0115 | +0.014 | +0.024 |
| 8 | 1.0060 | 1.0045 | +0.016 | +0.012 |
| 16 | 1.0046 | 1.0077 | +0.015 | +0.025 |
| 32 | 1.0020 | 1.0042 | +0.010 | +0.021 |
| 64 | 1.0000 | 1.0013 | 0.000 | +0.010 |

Route proof passes identically on pair C, `m = 1` through 512, both block
sizes, including the `poison == after` control at block 32 `m = 1`. Every A/A
interval in both pair-C sweeps brackets 1.000; 0 launches discarded.

Three readings of the same experiment, and they separate cleanly:

* **The row the change provably never executes swings.** +0.28%, −1.93%,
  +0.28%. A 2.2-point range on a row where the two binaries differ only in code
  that does not run.
* **Every row the change does execute keeps its sign in all three pairs**, at
  the same absolute magnitude to within a factor of two: `m = 8` is
  +0.017/+0.023/+0.033 ms, `m = 16` is +0.019/+0.018/+0.027 ms. Deliberately
  moving every function in the binary did not remove it, flip it, or leave it
  unchanged — it made it slightly larger, which is what a per-call saving does
  when the loop around it is realigned.
* **The saving is flat in `m` where it is resolvable.** Pair C reads
  +0.033/+0.027/+0.032/+0.037 ms at `m = 8/16/32/64` — a constant, which is
  what a once-per-panel cost looks like. Pairs A and B lose it at `m = 64`
  because 0.02 ms against 7.6 ms is 0.26% and those rows' intervals are ±0.3%
  wide: at the resolution limit, not absent.

That is the argument. Not the shape of the curve — the fact that the
route-not-taken row is the only one that moves when the layout does.

## The control that was supposed to be boring

Block 32 at `m = 1` takes `borrowed_affine_int4_matmul_nblock` and never calls
the pack at all. The poisoned build confirms this rather than asserting it: at
that row it is **bit-identical** to `after` under a build that is deliberately
wrong. The two binaries therefore differ, on that row, only in code that does
not execute.

It reads **0.9807, CI [0.9794, 0.9835]** — a 1.9% *loss*, reproduced at
0.9821 [0.9807, 0.9862] under `ONNX_GENAI_CPU_MM_INT4_GEBP=0`, with its own
A/A sitting at 1.0000 [0.9973, 1.0014].

**Two other pairs of binaries, built from the same source change — one against
a `main` three commits earlier, one against current `main` with every function
32-byte aligned — read 1.0000 / 1.0028 [0.9986, 1.0042] and 1.0028
[0.9986, 1.0070] on that same row.** Same change, same row, same
route-not-taken: +0.3%, −1.9%, +0.3%.

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
such row exists — which is the usual case — a sub-2% result is unconfirmed
until it reproduces across **independently built pairs of binaries**, and the
cheapest way to get an independent pair on demand is to rebuild every arm under
`-Cllvm-args=-align-all-functions=5`, which perturbs layout and nothing else.
Reproducing across two block sizes, or two amortization slopes, from a *single*
pair does not substitute for it: one pair of binaries has one layout.

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
document passed the gate: the discard count is **0** for the 61-launch sweep
and 0 again for both pair-C sweeps, so the kept set is the attempted set and
no launch here was selected by the gate at all.

That is a property of this run, not of the instrument, and the instrument
could not have told the difference. Until now the harness kept a single
aggregate discard counter, which answers "how many were thrown away" but not
"were they thrown away evenly" -- and only the second closes the question,
because the arms have genuinely different runtimes and a fixed efficiency
floor can therefore admit them at different rates. At a zero discard rate the
two are the same number; at any non-zero rate they are not. `admission` in the
result JSON now carries `by_arm`, `attempts_by_arm`, `rate_by_arm` and
`rate_spread`, so a future sweep states its own answer rather than leaving a
reader to assume this one's. A gate that discards every launch of an arm now
says so by name instead of failing inside the ratio with `no median for empty
data`.

**That gate was blind to SMT contention, and these numbers were taken under
it.** `(utime + stime) / wall` measures time spent *on a logical cpu*. A
competitor on the pinned cpu's hyperthread sibling shares the physical core's
execution units, so it takes throughput without taking time: the run keeps its
timeslice, scores a perfect 1.000, and is admitted. Measured directly on this
host, `PIN=4` with a spinner on its sibling cpu5:

| cell | throughput | `eff` | gate verdict |
|---|---|---|---|
| quiet | 1.000x | 1.000 | KEPT |
| competitor on cpu5 (SMT sibling) | **0.536x** | **1.000** | **KEPT** |
| competitor on cpu8 (other physical core) | 0.976x | 1.000 | KEPT |

The third row is the control: the same load on a different physical core costs
2.4%, so the 46% is SMT specifically and not load in general. A rep delivering
half its work was indistinguishable from a clean one, and the reps worth
discarding were exactly the ones the gate kept. A tight A/A null does not rule
this out — a *persistent* competitor produces a consistently wrong number, and
consistency is what a null measures.

Two things bound the damage to what is published here. Arms are **rotated
within each round**, so a persistent sibling competitor lands on all three arms
about equally and largely cancels in a *ratio*, which is what the verdict here
rests on. That protects the ratio, not every column: the **absolute `before ms`
and `after ms` medians printed above are not immune**, and a persistent sibling
competitor would inflate both. Read them as scale, not as this host's achievable
throughput. And the
mechanism argued above predicts the null at m≥64 independently of timing. What
rotation does *not* cancel is a competitor correlated with the arms — one whose
duty cycle happens to beat against a particular arm's launch length. **The
unresolved m=1 `0.981x` is a signature consistent with exactly that**, as well
as with the code-layout reading given there, and the two have not been
separated. The discriminator is cheap and is not yet run: swap which core each
arm is pinned to and see whether the 1.9% follows the arm or the core. Until
then m=1 stays open, and it is now open for two reasons rather than one.

The harness has since grown the second gate this needs: it samples the pinned
cpu's SMT siblings from `/proc/stat` across each launch and discards a rep
whose sibling was busy, recording those discards separately from efficiency
discards because they are different contention modes. Where no sibling exists —
no SMT, or a pin already covering both — it reports the gate `INACTIVE` rather
than reporting zero discards, because a gate that cannot fire reads exactly
like a gate that passed. The driver also parks itself off the measured core and
its sibling: unpinned it was measured contending with its own benchmark, at
0.040 sibling-busy versus 0.007 parked.

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

* **prefill: a fixed ~0.017–0.033 ms per packed panel**, independent of `m`,
  reproduced in three independently built pairs of binaries including one
  perturbed in layout only. As a ratio that is **1.004x–1.014x at
  `m = 1/8/16`** (block 16 at `m = 1`; block 32's `m = 1` is off-route), and it
  is the *absolute* figure that is the result — the ratio is just that constant
  over a total that grows with `m`
* **prefill `m >= 64`: at or below the instrument's resolution**, ~0.3% at
  those rows, so bounded rather than shown to be zero
* **block-16 decode: ≈1.010x**, against an instrument floor of ~0.6% — real,
  but smaller than the 1.015x #1809 reported
* **an incidental ~1.9% layout loss on the block-32 `m = 1` decode route in one
  of the three builds**, not attributable to the change, absent from the other
  two, and the reason the claim above rests on three build pairs rather than
  one

Shipped here: the per-row `fnv` route fingerprint in `int4_prefill_route_ab`,
the bootstrap and A/A self-check in the harness, and the two scripts that make
all of it reproducible.
