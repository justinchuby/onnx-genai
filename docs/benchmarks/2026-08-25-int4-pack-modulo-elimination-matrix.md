# The int4 pack's eliminated modulo: the matrix #1809 could not finish

**Date:** 2026-08-25 (busy-host arm added 2026-08-27) · **Refs:** #1809 (merged
`71bbec062`), #1676, #1802
**Reproduce:** `crates/onnx-runtime-ep-cpu/benches/int4_modulo_arms.sh`, then
`int4_modulo_matrix.py` (add `--co-tenant smt|dram` for the busy-host arms).
For an independent layout, rebuild the arms with
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
3. **The per-build code-layout component this document reported at ~2% does not
   survive being measured directly.** Found by a control that was supposed to be
   boring, and detailed below because it bore on every A/B in this directory.
   **Measured head-on, 2026-08-26:** build the *same* source at two independent
   layouts — default, and `-Cllvm-args=-align-all-functions=5`, which moves 50%
   → 91% of `FUNC` symbols onto a 32-byte boundary — and run one against the
   other as an **A/B′ null**. Through the SMT-sibling gate (#2216) that null is
   **1.0014 [1.0000, 1.0042]** at prefill block 32 `m = 1` and **0.9993 [0.9971,
   1.0005]** on the block-16 decode. Both null; all arms bit-identical, which is
   what proves only the layout moved. Layout sensitivity at these cells is under
   half a percent, not two. The original ~2% was the spread across three build
   pairs taken *before* that gate existed, and the one cell since re-taken
   shrank 1.93% → 0.28%; the honest reading is that the spread was SMT
   contention wearing layout's clothes. **This retracts the credibility bar**
   earlier versions of this bullet exported: do not require a result here to
   clear 2% before believing it. What stands unchanged is the narrower claim
   that a *same-binary* A/A cannot see between-binary layout at all — so the
   control to run is the A/B′ null above, not a longer A/A.

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
| 1 | 0.713 | 0.727 | **0.9807** | [0.9794, 0.9835] | **loss — superseded, see below** | 1.0000 | [0.9973, 1.0014] |
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
  +0.28% — and −0.28% in a fourth build re-taken later through the SMT-aware
  gate (see the caveat section). A 2.2-point range on a row where the two
  binaries differ only in code that does not run, and where the widest reading
  is the one taken through the weakest gate.
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

The boolean table above is not the whole control, because on its own "this row
did not move" is consistent with "the poison is inert at `m = 1`" or "the poison
is inert at block 32". Both are excluded by bracketing the cell on *both* axes.
Raw FNV folds, re-taken on `9b577b2c5` from arms rebuilt for this purpose:

| block | m | GEBP | `after` | `poison` | moves? |
|---:|---:|---:|---|---|---|
| 16 | 1 | on | `01c5b99695edbba2` | `7439f2cba5901add` | **yes** |
| 32 | 8 | on | `a504a827d5336008` | `8bc857141884caa2` | **yes** |
| 32 | 1 | on | `4cb1dcffe7454cff` | `4cb1dcffe7454cff` | no |
| 32 | 1 | off | `4cb1dcffe7454cff` | `4cb1dcffe7454cff` | no |
| 32 | 8 | off | `06effc7d87d80700` | `06effc7d87d80700` | no |

Hold `m = 1` and change the block: it moves. Hold block 32 and change `m`: it
moves. The poison is live at `m = 1` and live at block 32; the single cell where
it goes quiet is exactly the one `int4_prefill_gebp_min_rows` predicts.

The last two rows close the remaining gap, and they are the reason the
env-var control is not the argument here. `ONNX_GENAI_CPU_MM_INT4_GEBP=0`
demonstrably takes the pack off the route — at `m = 8` it collapses `poison` onto
`after` (`06effc7d87d80700` both), which is the same signature the control cell
shows. But at block 32 `m = 1` the output is `4cb1dcffe7454cff` **with the flag
on and with it off**: toggling the flag does not change what that row computes,
so it does not change which kernel that row runs. Any timing movement observed
at that cell under the flag is therefore movement on a bit-identical
computation — layout or noise, not a route change, and not a residual pack path.
That matters because the flag swaps a whole algorithm and drags cache behaviour
and layout along with it, so its *timing* delta at any row is unattributable by
construction; only the checksum makes it an observation. A route claim rests on
the poison, never on the env var.

It reads **0.9807, CI [0.9794, 0.9835]** — a 1.9% *loss*, reproduced at
0.9821 [0.9807, 0.9862] under `ONNX_GENAI_CPU_MM_INT4_GEBP=0`, with its own
A/A sitting at 1.0000 [0.9973, 1.0014]. Both readings are of a row whose bytes
are provably identical in both arms, so neither is a cost of the change.

**Most of that 1.9% was the SMT hole, and re-measuring through it shrinks the
row by 7x.** The Method section below discloses that every number in the tables
above was taken under a gate blind to hyperthread-sibling contention. #2216
closed that hole after this document was written, so the row has now been
re-taken through a gate that can see it — arms rebuilt from scratch on
`533546095`, 61 launches per arm, same cell, same pin:

| taking | gate | m = 1, block 32 | A/A | discards |
|---|---|---|---|---|
| original (above) | rusage only, SMT-blind | 0.9807 [0.9794, 0.9835] | 1.0000 [0.9973, 1.0014] | 0 / 183 |
| re-take, `533546095` | rusage **+ SMT sibling** | **0.9972 [0.9930, 0.9993]** | 1.0000 [0.9958, 1.0021] | 4 / 183 |

The intervals do not overlap. A 1.9% loss became a 0.28% one against an A/A that
brackets unity in both takings, which is what a contended-sibling artifact looks
like when the contention finally becomes visible to the gate: the A/A cannot
reveal it, because both A/A arms inherit the same sibling. Four launches were
discarded here against zero originally — not because this host was busier, but
because the instrument could finally see the thing it was discarding for. The
artifact records which gate fired: **`smt_total` 4, CPU-efficiency discards 0.**
Every one of the 183 launches passed the rusage floor, exactly as all 183 did in
the original taking, and the four that were thrown out were caught only by the
sibling-jiffy gate. Per-arm admission spread 0.016 (before 1, after 2, aa 1), so
the gate did not select one arm over another.

This does not rescue the row as a result, and it is not meant to. 0.9972's
interval still excludes 1.000, and the route proof was re-run on these same
rebuilt arms and still reports `before == after == poison == 4cb1dcffe7454cff`
at that cell — so whatever remains is still measured on a row where the two
binaries are bit-identical. The correction is to the *magnitude of the artifact*,
not to its attribution, which the poison settled independently of any timing.

**Two other pairs of binaries, built from the same source change — one against
a `main` three commits earlier, one against current `main` with every function
32-byte aligned — read 1.0000 / 1.0028 [0.9986, 1.0042] and 1.0028
[0.9986, 1.0070] on that same row.** Same change, same row, same
route-not-taken: with the SMT-gated re-take above as a fourth independent
build, the four readings are +0.3%, −1.9%, +0.3%, −0.3%.

So this is code layout, not a property of the change — and that is the finding,
because:

* **It is invisible to an A/A.** The A/A arm is the *same file*, so it measures
  everything except the thing that differs between builds. Every A/A in this
  document brackets 1.000 while a 1.9% artifact sits in the same table.
* **It is mostly not layout.** ~~It is larger than most kernel results this
  repository ships.~~ Retracted 2026-08-26 by direct measurement: an A/B′ null
  holding the source constant and moving only the layout is 1.0014 [1.0000,
  1.0042] here and 0.9993 [0.9971, 1.0005] on the decode row — under half a
  percent, against the 1.9% this bullet was written to explain. Two independent
  builds of this tree disagreeing by ~2% on a bit-identical row is a real
  observation, but the cause was SMT-sibling contention, not code placement.
  A/B′ nulls are cheap; run one before attributing a small delta to layout.
* **It is not stable across rebuilds**, so it cannot be calibrated out once and
  reused. It has to be re-measured per build, which is what a route-not-taken
  row does for free when one exists.

The practical rule: **include a row the change provably cannot reach, prove it
with a poisoned build, and read it as the experiment's real floor.** Where no
such row exists — which is the usual case — the control is an **A/B′ null**:
rebuild every arm under `-Cllvm-args=-align-all-functions=5`, then point the
matrix at a directory whose `before` is the default-layout `after` and whose
`after` is the aligned one. The headline ratio is then a pure layout null on
the exact cell being claimed, and the harness's cross-arm bit-identity check
proves the semantics were held constant. That is strictly better than the older
advice to distrust anything under 2%: it measures the floor for your cell
instead of importing someone else's. Reproducing across two block sizes, or two
amortization slopes, from a *single* pair still does not substitute for it: one
pair of binaries has one layout.

### The A/B′ null, measured (2026-08-26)

Everything above about layout was inferred from *disagreement between pairs*:
three build pairs read +0.3%, −1.9%, +0.3% on a row proven bit-identical, and
the spread was attributed to code placement. That is an indirect argument, and
it was reached with an instrument later found blind to SMT-sibling contention.
The direct experiment holds the source constant and moves only the layout:

| arm | binary | layout | semantics |
| --- | --- | --- | --- |
| `before` | default build of `after` | 2546/5136 `FUNC` on 32B = 50% | `offset_base + q` |
| `after` | `-Cllvm-args=-align-all-functions=5` | 4690/5136 `FUNC` on 32B = 91% | `offset_base + q` |
| `aa` | copy of the aligned build | same as `after` | `offset_base + q` |

Both binaries compile the same line, so the harness's cross-arm bit-identity
check is what certifies the manipulation: if the arms had differed in anything
that reaches the output, the checksums would have parted. They did not.

```
prefill block 32, m = 1, 41 launches/arm   0.712 -> 0.711  1.0014 [1.0000, 1.0042]  null  bit-id True
decode  block 16,      15 launches/arm   126.070 -> 126.163  0.9993 [0.9971, 1.0005]  null  bit-id True
```

**Layout sensitivity at these cells is under half a percent**, against the ~2%
this document previously exported as a credibility bar. Two consequences, and
they point in opposite directions, so both are stated:

* The **decode result stands and is strengthened**. ~1.9–2.3% over a layout
  floor of ≤0.4% is a real effect, not a build artifact — the objection that it
  is the same size as the layout noise was correct to raise and does not hold.
* The **`m = 1` −0.28% is inside the floor** and should not be read as a
  regression. It already had a route proof saying the changed line does not
  execute there; it now also has a floor wide enough to contain it.

Caveats, since this is one perturbation and not a survey of layout space:
forced 32-byte alignment is a large, systematic, whole-binary change, so it
bounds this tree's sensitivity at these two cells rather than sampling ordinary
build-to-build drift. The decode row is 15 launches with an uneven admission
rate (`before` 14/15, `after` 15/15, `aa` 13/15, spread 0.133) and is the
weaker of the two; prefill is 41/41/40 with spread 0.024. Three of the four
discards were the SMT gate, one the CPU-efficiency floor.

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

## Does it survive a busy host? (2026-08-27)

Everything above this line was measured on a quiet host: pinned to an idle
core, under two gates whose entire job is to discard any launch that was not
alone on it. The recorded user correction on #1729 says that is not sufficient
evidence for something that ships as a default:

> CPU scheduling and performance policy **must not assume exclusive access to
> the machine.** Edge deployments commonly share CPUs with other programs and
> may be busy. [...] **A policy that wins only under exclusive quiet-host
> conditions is not a valid default.**

`dequant_panel_avx2`'s eliminated modulo is on by default with no opt-out, so
it owes that evidence, and until now it did not have it — every number in this
document was a quiet-host number, including the ones used to justify keeping
the change. `int4_modulo_matrix.py --co-tenant` supplies the missing arm by
**injecting** load rather than gating it out, in the two contention modes this
host actually has:

* **`smt`** — one pinned scalar-throughput spinner on the measured core's SMT
  sibling (cpu 5). Takes execution-unit throughput without taking timeslices;
  this is the mode `(utime+stime)/wall` is structurally blind to, and it is the
  one that bears directly on an ALU-work claim.
* **`dram`** — eight pinned streaming-memcpy hogs on other physical cores
  (2, 6, 8, …, 18), buffers far larger than either 32 MiB L3. Takes memory
  bandwidth and shared cache, and does not touch the measured core at all.

Arms stay interleaved and rotated inside each regime, so the co-tenant is
common-mode and the ratio remains a fair A/B. What changes is the regime the
kernel runs in, and the whole question is whether the ranking survives it.

### Pre-registered, before the first contended launch

```
PASS   no row regresses -- every cell is a gain or a null under injected load.
FAIL   any cell's 95% interval lies entirely below 1.000. The change is then a
       quiet-host-only win and is not valid as an unconditional default.
VOID   any row's A/A interval excludes 1.000 -- re-take the matrix rather than
       read around it. VOID outranks FAIL and PASS, so a loss can never be
       reached through an instrument that was already biased.
```

The rule is deliberately **not** "the win must be the same size". Magnitude is
reported whichever way the rule goes; what the correction forbids is a default
that is *worse* on a shared box, and that is what this tests.

### Block 16 prefill — 61 launches per arm, in each of three regimes

Ratio is `before / after`, so above 1.000 means the elimination is faster.

| m | quiet | 95% CI | `smt` co-tenant | 95% CI | `dram` co-tenant | 95% CI |
|---:|---:|---|---:|---|---:|---|
| 1 | **1.0041** | [1.0019, 1.0063] | **1.0089** | [1.0077, 1.0101] | 0.9975 | [0.9882, 1.0134] |
| 8 | **1.0034** | [1.0008, 1.0053] | **1.0072** | [1.0050, 1.0082] | 0.9991 | [0.9913, 1.0108] |
| 16 | **1.0042** | [1.0015, 1.0059] | **1.0057** | [1.0040, 1.0065] | 1.0066 | [1.0016, 1.0145] |

Per-regime A/A null, and what the co-tenant cost in absolute terms:

| regime | A/A m=1 / m=8 / m=16 | `before ms` m=1 | vs quiet | admitted | co-tenant busy (median / min) |
|---|---|---:|---:|---|---|
| quiet | 0.9998 / 1.0008 / 1.0005 | 2.085 | — | 176/183 | n/a |
| `smt` | 1.0015 / 1.0014 / 1.0006 | 3.413 | **1.64x slower** | 183/183 | 1.000 / 1.000 |
| `dram` | 0.9991 / 1.0031 / 1.0054 | 3.731 | **1.79x slower** | 177/183 | 1.000 / 0.995 |

Both co-tenants are real: they cost the *unmodified* kernel 64% and 79% of its
runtime at prefill `m = 1`, and 54% and 64% on the decode loop. The `smt`
regime discarded nothing in either workload, and the busy fraction its floor
measured never dropped below 1.000 on any admitted launch, so no cell in this
section is a quiet-host cell wearing a busy-host label.

**Verdict: PASS in both contended regimes.** No interval anywhere lies below
1.000, every A/A brackets unity, and every arm is bit-identical.

### Block 16 decode — 15 launches per arm, 32 tokens, same three regimes

The decode loop is where this change is worth the most, because every token is
a fresh `m = 1` pack with nothing to amortize it over.

| regime | before ms/tok | after ms/tok | speedup | 95% CI | A/A null | admitted | co-tenant busy (median / min) |
|---|---:|---:|---:|---|---|---|---|
| quiet | 123.064 | 121.905 | **1.0095** | [1.0077, 1.0135] | 0.01% [0.9980, 1.0021] | 45/45 | n/a |
| `smt` | 189.591 | 185.441 | **1.0224** | [1.0213, 1.0237] | 0.07% [0.9994, 1.0024] | 45/45 | 1.000 / 1.000 |
| `dram` | 201.735 | 200.970 | 1.0038 | [0.9956, 1.0155] | 0.09% [0.9929, 1.0095] | 38/45 | 1.000 / 1.000 |

Decode reproduces the prefill pattern and amplifies it: under SMT contention
the win goes from 0.95% to **2.24%**, intervals nowhere near overlapping, on an
A/A of 0.07%. Under DRAM contention it is a null. Bit-identical in all three.
The `dram` regime lost 7 of 45 launches to the efficiency floor and its
discard-rate spread is 0.133, the least even admission anywhere in this study —
another reason that column is read as "no resolvable effect" and not mined for
a number.

### The two regimes move in opposite directions, and the mechanism predicts both

* **Under SMT contention the win roughly doubles** — 1.0041 → 1.0089 at
  `m = 1` prefill, and 1.0095 → 1.0224 on the decode loop, with
  non-overlapping intervals in both. The eliminated work is integer division:
  pure ALU issue. A hyperthread sibling competes for exactly those issue slots,
  so slots the kernel no longer needs are worth more when they are scarce than
  when they are free. This is the regime an ALU-saving change should win
  hardest in, and it does, on both workloads.
* **Under bandwidth contention the win fades to a null** — 0.9975 and 0.9991 at
  prefill `m = 1` and `m = 8`, 1.0038 [0.9956, 1.0155] on decode. Same
  mechanism from the other side: with eight hogs on DRAM the pack is waiting on
  memory, and a fixed ALU saving disappears into the stall it no longer sits on
  the critical path of.

So the honest summary is not "the win holds at the same size" — it does not.
It is that **the regime where the change is worth less is the regime where it
is worth nothing, not the regime where it is worth negative**, which is exactly
the distinction the correction cares about for a default.

**One cell I decline to claim**, despite the rule mechanically reading it as a
gain: `dram` `m = 16` at 1.0066 [1.0016, 1.0145]. That regime's own A/A at the
same row is **1.0054** — the instrument's floor there is roughly 4x the quiet
host's, and it is the same size as the effect. The rule uses A/A only to decide
whether a row is *readable* for a regression, which it is; that is not a
licence to report a 0.66% gain measured through a ±1.4% null. Read the whole
`dram` column as "no resolvable effect, and no regression".

### What this does not establish

The co-tenants are synthetic: a spinner and a memcpy loop, not another
inference process. A real neighbour would contend for L2, the TLB and the
memory controller in proportions neither of these reproduces, and would itself
be bursty rather than constant. What the arm rules out is the specific failure
the correction names — a result that exists only because the box was empty —
and it does that for the two contention modes this host can produce. It does
not measure cgroup or cpuset co-tenancy, where the scheduler, not the hardware,
is what is shared; the efficiency floor would reject those launches rather than
measure them, and covering that regime needs a different instrument.

### Reproducing

```bash
crates/onnx-runtime-ep-cpu/benches/int4_modulo_arms.sh
for mode in none smt dram; do
  crates/onnx-runtime-ep-cpu/benches/int4_modulo_matrix.py \
      --rounds 61 --block 16 --m-list 1,8,16 --skip-decode \
      --co-tenant "$mode" --out target/int4-modulo-arms/cot_prefill_$mode.json
  crates/onnx-runtime-ep-cpu/benches/int4_modulo_matrix.py \
      --skip-prefill --block 16 --decode-rounds 15 --tokens 32 \
      --co-tenant "$mode" --out target/int4-modulo-arms/cot_decode_$mode.json
done
```

Deviation from the matrix above: the decode arm is 15 launches at 32 tokens,
not 41 at 64, because a contended decode launch costs ~30 s and the three
regimes are 135 launches. The A/B ratio is internally consistent within each
regime — all three use the same rounds and token count — and the quiet regime
reproduces the published 1.010x, which is the check that the shortened form
did not move the answer.

The co-tenant is spawned by the harness, pinned by `taskset`, torn down when
the run ends, and exits by itself if it is ever orphaned — a load generator
that outlives its harness on a box with eight agents on it is a worse problem
than the one it was measuring. The harness still holds `scripts/hostlock.sh`
for the whole run: the lock is what keeps *other* agents' load off the box, and
that is what makes the injected load a controlled variable rather than the
uncontrolled one every other gate in this file exists to reject.

The gate is inverted rather than removed, which is the part worth copying. An
arm whose injected load silently failed to start is a quiet-host arm with a
busy-host heading: it runs clean, passes, and is indistinguishable in the
artifact from a real result — while being precisely the number the arm exists
to replace. So each contended launch must *demonstrate* its contention, per
launch, against a floor, and the gates that are still meaningful in that mode
keep firing (a stray competitor on the sibling still invalidates a `dram`
launch; descheduling still invalidates any launch). `--self-test` asserts all
of that on synthetic records, and is mutation-tested against the five ways an
implementation of this arm plausibly goes wrong.

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
* **an incidental layout loss on the block-32 `m = 1` row** (which takes the
  decode route, and is measured in the prefill sweep), **read at ~1.9% in one
  of the three original builds and at 0.28% when that cell was re-taken through
  the SMT-aware gate** — not attributable to the change in either case, since
  the arms are bit-identical there; near-absent from the other two builds, and
  the reason the claim above rests on multiple build pairs rather than one
* **valid as a default on a busy host, not only a quiet one (2026-08-27)** —
  under injected SMT contention the win roughly doubles on both workloads
  (prefill `m = 1` 1.0089, decode 1.0224); under injected DRAM contention it
  fades to a null with no regression anywhere. Pre-registered rule, PASS in
  both regimes. This is the evidence #1802 item 4 asks for, and until it
  existed every number in this document was a quiet-host number

Shipped here: the per-row `fnv` route fingerprint in `int4_prefill_route_ab`,
the bootstrap and A/A self-check in the harness, the `--co-tenant` arm and its
mutation-tested `--self-test`, and the two scripts that make all of it
reproducible.
