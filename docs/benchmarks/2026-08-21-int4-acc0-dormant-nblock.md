# int4 `accuracy_level = 0` decode: the route the production default actually
# takes, and the correction to its attribution

**Date:** 2026-08-21 · **Owner:** Roy (CPU MatMul) · **Host:** AMD EPYC 9V74,
32 vCPU (16c x 2 SMT), AVX2/FMA/F16C, **no AVX-512/VNNI**.
Ledger entry: §23 of [`CPU_MATMUL_ASSIGNMENT.md`](../performance/CPU_MATMUL_ASSIGNMENT.md).
Merged as `99f105d52` (#1679), for #1676.

Every timing here is `taskset -c 0,2,4,...`-pinned to distinct physical cores.
Unpinned multi-thread numbers on this host measure worker placement, not the
kernel — see the [placement record](2026-08-21-decode-worker-cpu-placement.md).

---

## 1. The route, by counter

The question was where the 1.84x acc0 gap sits. The first thing to establish is
which kernel the production default even runs. Route counters instrumented from
operator entry through to the innermost arm, over a real decode step at
`accuracy_level = 0`:

> **Note (2026-08-23, revised):** the `1.84x` inherited here **no longer
> holds** — re-measured on `e189244ba` the acc0 gap is **1.12x at t=1, 1.15x at
> t=4, 1.12x at t=8**. Part of that closure is the work this very document
> motivated: enabling the register-blocked kernel at `accuracy_level = 0`
> (#1679) was one of three acc0 merges that landed after the 1.84x was taken.
> The route findings below are categorical (which kernel runs) and stand
> unchanged; only the *sizing* of the gap moved, and it moved because the
> dormant kernel was woken up. See
> [2026-08-23-acc0-gap-vs-ort-by-width.md](2026-08-23-acc0-gap-vs-ort-by-width.md).

| counter | count |
|---|---|
| `entry_bits4` | 95 |
| `percolumn` | 95 |
| `nblock` | **0** |
| `block_simd` | 129,499,136 |

`nblock = 0`. The register-blocked N-blocked kernel added by #1104 — measured
there at 1.46x on a 14B model and proven byte-identical — **was never reached at
the production default**, from the day it merged.

The cause is one line. #1104 shipped it behind
`ONNX_GENAI_CPU_MM_INT4_NBLK`, defaulted `false`, with the comment that this was
"until the win is measured, exactly like the toggles that preceded it". The
measurement did not happen. Nothing in the tree asserted which route production
took, so the toggle's expiry condition was invisible.

**Fix:** default `true`, plus
`acc0_decode_reaches_the_nblocked_kernel_by_default`, which reads the counters
and fails if the default route changes. The default is now a checked property.

## 2. The first attribution was wrong, and the instrument is why

The initial A/B reported 3.14x (t=1) and 4.80x (t=4) for the N-blocked route.
Those numbers are **void**.

The route probe is an `AtomicU64::fetch_add`. In the per-column arm it sits
inside the block loop and fired **129,499,136 times** per measurement window; in
the N-blocked arm it fired 95 times. The probe inflated the *baseline it was
measuring against* by roughly 3x, and — because it did so consistently across
repetitions — produced a stable, self-consistent, wrong answer.

This is distinct from §18's probe, which was genuine kernel overhead: a
per-block `is_x86_feature_detected!` that broke inlining and forced the
accumulator through memory, so removing it made the *shipping* kernel faster.
This one never affected production at all. It only corrupted its own baseline.

Rebuilt with the probe out of the timed path, counters read once at the end:

| route | t=1 (ms) | t=4 (ms) | vs per-column |
|---|---|---|---|
| per-column (previous default) | 56.528 | 28.518 | 1.00x |
| N-blocked, group of 1 | 55.156 | 27.861 | 1.02x |
| N-blocked, group of 2 | 47.679 | 23.885 | 1.19x |
| N-blocked, group of 4 | **38.114** | **19.238** | **1.48x** |

Ratios are from the t=1 column; t=4 agrees to within 0.01x on every row, which
is the check that this is a per-core kernel effect and not a scheduling one.

## 3. What the win is, and what it is not

The group-of-1 row is the control that matters, and it is the reason the obvious
explanation is wrong.

The per-column path carries a horizontal reduction on the critical path:
`extractf128` -> `movehl` -> `shuffle`, each dependent on the last, once per
32-weight block, every four FMAs. The N-blocked kernel removes it — it keeps the
scale in a vector accumulator and reduces once per column instead. That is a
real latency chain, it is textbook, and it is the thing you would name if asked
to predict the win.

**It is worth 1.02x. Nothing measurable.** Group-of-1 is the N-blocked kernel
with the reduction restructured and *no* column grouping, and it lands on top of
the baseline. The block loop carries enough independent work for the
out-of-order engine to hide the chain entirely.

The entire win is **four-column activation reuse**: 1.45x from group 1 to group
4 (55.156 -> 38.114). Each activation block is loaded once and used against four
weight columns.

Two independent confirmations that this is the right decomposition: the
group-2 row sits where a load-amortisation model predicts (1.19x, roughly the
square-root-ish midpoint rather than half the gain), and the group-4 total,
1.48x, matches #1104's independently measured 1.46x on entirely different
hardware and a different model.

## 4. Numerics: the trade, stated

The N-blocked kernel applies the block scale as a separated correction rather
than folding it per block. Against an f64 oracle, worst cell measured:

| route | worst relative error vs f64 |
|---|---|
| per-column | 6.739e-6 |
| N-blocked, group of 4 | **2.422e-5** |

**3.59x worse.** It is shipped anyway, and disclosed rather than buried, on
three grounds: it remains inside the pinned accuracy envelope; the envelope
guard was sized to this *measurement* (8x headroom) rather than to hope, so a
future regression trips it; and a 1.48x decode win for a 3.59x relative error
increase inside an envelope is a defensible trade **only if both numbers are
visible to whoever inherits it**.

`accuracy_level = 0`'s exact contract is unaffected — the contract is the
envelope, and this stays inside it. No reduced-precision diversion is involved;
this is f32 accumulation throughout.

## 5. What this does not close

The acc0 gap against ORT is not closed by this. The dormant kernel was a free
1.48x that had been sitting in the tree unclaimed, so it is the correct first
move, but the remaining gap after it is a separate mechanism and is not
attributed here.

**Negative-result note for the next person:** do not spend time on the
horizontal reduction in the acc0 int4 path. It is measured at 1.02x. Column
grouping is where the arithmetic intensity is.
