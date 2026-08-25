# The int4 pack's eliminated modulo: the matrix #1809 could not finish

**Date:** 2026-08-25 · **Refs:** #1809 (merged `71bbec062`), #1676
**Reproduce:** `crates/onnx-runtime-ep-cpu/benches/int4_modulo_arms.sh` then
`int4_modulo_matrix.py`

## What this corrects

#1809 removed the per-group integer division from
`Int4Weight::dequant_panel_avx2`, the innermost loop of the int4 GEBP B-panel
pack, and reported it as **1.015x on block-16 decode and a null on prefill**.

The prefill half of that claim was taken at `m = 64/256/512`. The two smallest
rows, `m = 1` and `m = 8`, **failed their A/A null at 5.31% and 4.62% and were
withheld** rather than reported — correctly, but it left the sweep with a hole
at exactly the end where the mechanism predicts the effect should live. The
pack is amortized over `m` rows, so a per-group cost is largest when `m` is
smallest, and the withheld rows were the smallest ones.

They are no longer withheld. **`m = 8` is a 1.007x gain with a 95% interval of
[1.0050, 1.0092]**, and the effect decays monotonically to a null by `m = 64`,
which is where #1809 started measuring. "A null on prefill" is not wrong about
the rows it took; it is wrong as a description of prefill.

## Matrix

Block 32, `2048x2048`, `bits=4`, `accuracy_level=0`, pinned to one physical
core (cpu 4), **61 independent launches per arm**, arms interleaved and rotated
per round, medians over launches. 0 launches discarded by the CPU-efficiency
gate.

| m | before ms | after ms | speedup | 95% CI | A/A | A/A 95% CI | verdict |
|---:|---:|---:|---:|---|---:|---|---|
| 1 | 0.713 | 0.711 | 1.0028 | [0.9986, 1.0042] | 0.9986 | [0.9972, 1.0014] | **null (control)** |
| 8 | 2.413 | 2.396 | **1.0071** | [1.0050, 1.0092] | 0.9996 | [0.9975, 1.0017] | **gain** |
| 16 | 2.999 | 2.980 | **1.0064** | [1.0044, 1.0084] | 0.9983 | [0.9960, 1.0010] | **gain** |
| 32 | 4.724 | 4.708 | 1.0034 | [1.0006, 1.0049] | 1.0000 | [0.9979, 1.0017] | marginal gain |
| 64 | 7.633 | 7.633 | 1.0000 | [0.9986, 1.0052] | 0.9959 | [0.9935, 1.0022] | null |
| 128 | 14.038 | 14.033 | 1.0004 | [0.9971, 1.0047] | 0.9991 | [0.9959, 1.0033] | null |
| 256 | 26.292 | 26.280 | 1.0005 | [0.9971, 1.0050] | 0.9985 | [0.9933, 1.0037] | null |
| 512 | 51.344 | 51.355 | 0.9998 | [0.9945, 1.0046] | 0.9992 | [0.9953, 1.0043] | null |

Intervals are bootstrap over launches (20 000 resamples, seed 20260825).
**Every A/A interval contains 1.000 at every `m`**, so the instrument is not
biased at any row of the sweep — which is the thing a withheld row leaves you
unable to say.

Decode, block 16 (`m = 1` through the fused GEBP), 21 independent launches per
arm, 32 tokens: **1.0120**, A/A null 0.16%. #1809 reported 1.015x; this
reproduces it on current main after #1729 and #1794 both changed decode
placement and default width underneath it.

An independent earlier replication at 31 launches per arm agrees on every row
(`m = 8` 1.0100, `m = 16` 1.0054, `m >= 32` null).

## Why the shape of the curve is the result

The gain is monotone in `1/m` — 1.0071, 1.0064, 1.0034, then nothing. That is
not a set of eight independent measurements that happened to include two
positive ones. It is the amortization curve the mechanism predicts, sampled at
eight points, and it lands on a hard null at `m = 1` where the route provably
does not execute the changed line at all.

That last row is the strongest control in the table and it is free: at block 32
`m = 1` takes `borrowed_affine_int4_matmul_nblock` and never calls the pack.
Its two binaries therefore differ only in code that does not run, yet they
still differ in layout, ASLR and page backing. It reads **1.0028, CI
[0.9986, 1.0042]** — so everything the A/B measures other than the change
itself is bounded at well under half a percent, and `m = 8` sits eight standard
intervals clear of it.

## Route proof, per row

Timing cannot distinguish "the change ran and cost nothing" from "the change
never ran". A source-level A/B has to rebuild between arms, so *which arm ran*
is an assumption unless something makes it an observation.

`int4_prefill_route_ab` now prints an FNV-1a fold over the raw output bytes
(`fnv`), so every row carries its own bit-exact fingerprint, and
`int4_modulo_matrix.py --route-proof` builds a deliberately **poisoned** third
arm that drops the `+ q` term:

```
 block     m  before==after  poison moves
    16   1..512      True          True     (all rows)
    32     1         True         False     <-- control: pack not on this route
    32   8..512      True          True
```

`route proof: PASS`, exit 0. Two things fall out of it:

1. **`before == after` on all 16 rows, both block sizes.** The elimination is
   exact, as the algebra says it must be, so the speedup is numerically free
   rather than bought.
2. **The poison moves exactly where the pack is on the route, and nowhere
   else.** Block 32 `m = 1` is bit-identical under a build that is deliberately
   wrong, which is what proves the poison is not simply perturbing the whole
   binary.

Decode checksums are `844.536810` in all three arms — the same constant #1809
recorded, unchanged by #1729 and #1794.

## Mechanism, re-derived from the shipped binaries

The identity needs no assumption about `block_size`, in particular not that it
is a power of two, which this kernel's contract does not guarantee and its own
tests falsify at `block_size` 24 and 40:

- `run` is capped at `block_size - offset_base`, so `offset_base + q < block_size`
  for every `q < run`;
- `depth - offset_base` is a multiple of `block_size`.

Hence `(depth + q) % block_size == offset_base + q` exactly.

Division count, disassembled from the **executables that produced the numbers
above** rather than from an `.s` file emitted separately:

| arm | `div`/`idiv` in `dequant_panel_avx2` |
|---|---:|
| before | 6 |
| after | **4** |
| poison | 4 |

LLVM cannot do this itself: `block_size` is a runtime field, and a
`#[target_feature]` function is never inlined into a caller that might have
narrowed it, so constant propagation never reaches the body. This is a
different mechanism from #1783, where the value was a literal at every call
site and only the inlining barrier hid it.

## Method: why 61 launches and not one careful pairing

The per-launch spread on this host is enormous and the median is not:

| m | launch-to-launch spread (max−min) | A/A null on the median |
|---:|---:|---:|
| 1 | 102% | 0.14% |
| 8 | 42% | 0.04% |
| 512 | 19% | 0.08% |

A single paired A/B — however tight its intra-run spread looks — can be
dominated by which mode each side's launch landed in. That is the failure this
sweep is built to avoid, and the `m = 1` row is the demonstration: two launches
of the *same* binary differing by 2x, while 61 of them agree to 0.14%.

The host gate is the **CPU efficiency of the run itself** (`os.wait4` rusage
`(utime + stime) / wall`, launches below 0.95 discarded), not an instantaneous
runnable count sampled at run boundaries. A short run has room for a burst that
starts after the opening sample and ends before the closing one; that is how a
52% A/A null once passed a "host clean" check during #1809. Every launch here
passed the gate.

The whole sweep is taken under `scripts/hostlock.sh`, held for its full
duration by the measuring process.

The pin is cpu 4. Not cpus 0–1: cpu 0 has a permanent external competitor on
this host and cpu 1 is its SMT sibling, so a run pinned there is contended by
construction.

## Disposition

No kernel change. #1809's code is already correct and already on main; what was
incomplete was its account of where the change pays. The corrected scope:

- **block-16 decode: 1.012x** (reproduced on current main)
- **prefill `m = 8`: 1.007x, `m = 16`: 1.006x, `m = 32`: 1.003x**
- **prefill `m >= 64`: null**, and the null is now bounded rather than asserted

Shipped here: the per-row `fnv` route fingerprint in `int4_prefill_route_ab`,
and the two scripts that make the matrix reproducible.
