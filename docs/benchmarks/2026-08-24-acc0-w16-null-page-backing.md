# The width-16 A/A null is bimodal, and it is not page backing

**Date:** 2026-08-24
**Tree:** `2c3968afb` (main)
**Host:** AMD EPYC 9V74, 16 physical / 32 logical, SMT siblings adjacent, L3 in
two 32 MiB instances (`0-15`, `16-31`). AVX2 + FMA + F16C; **no AVX-512, no
VNNI** — the VM does not expose Zen 4's AVX-512.

## Verdict first

**REJECT: transparent-hugepage backing of the weight arena does not explain the
width-16 A/A null.** Pre-registered rule, 12 trusted launches: `thp_frac` range
**0.104** against a required 0.20, Spearman rho **-0.1888** against a required
-0.70. Both conditions fail.

**Established alongside it, and more useful than the rejection:** on a quiet
host the null is not a spread, it is **two modes**. Twelve identical launches
split into a slow cluster of five agreeing to **1.05%** and a fast cluster of
seven, with a mode ratio of **1.687x**. A slow arm whose members agree to one
percent is a discrete alternative configuration selected per process launch, not
noise.

## Why hugepages were the hypothesis

The null's every measured property said *per-process startup state*: position
independent, internally consistent to 0.5% while running slow, decided before
the first token. Two candidate mechanisms had already been tested and rejected —
dispatcher CPU placement (1.0953 against a 1.10 bar,
[`2026-08-24-acc0-dispatcher-placement.md`](2026-08-24-acc0-dispatcher-placement.md))
and worker placement (one worker per physical core in every configuration,
`benches/decode_placement_census.sh`).

Page backing fits that shape better than either:

```
$ cat /sys/kernel/mm/transparent_hugepage/enabled
[always] madvise never
$ cat /sys/kernel/mm/transparent_hugepage/defrag
always defer defer+madvise [madvise] never
```

`always` + `defrag=madvise` means anonymous memory gets 2 MB pages
*opportunistically at fault time*, and when the buddy allocator cannot hand over
a free 2 MB block immediately it silently falls back to 4 KB — no error, no log,
and no second chance, because the weights are faulted once at load and never
again. Whether a launch wins that lottery is decided by host fragmentation at
the instant it starts, is fixed for the life of the process, and is invisible to
every timing instrument. It also predicts the width dependence: at 4 KB a 22 MB
projection chain needs ~5600 PTEs against ~11 at 2 MB, and sixteen workers
share the page-walk caches.

## The measurement

`benches/acc0_w16_page_backing.py`. Each launch is the width-16 acc0 decode arm; while it runs,
`AnonHugePages` and `Anonymous` are read from `/proc/<pid>/smaps_rollup` and the
maximum-residency sample is kept. Those are categorical kernel accounting, not
timings, so the page reading is not corrupted by a busy host — but the
`ms_token` it is correlated against is, so the run took the hostlock and the
gate reported `runnable=3`.

| launch | ms/token | spread | AnonHugePages | Anonymous | `thp_frac` |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.965 | 4.9% | 142.0 MB | 172.5 MB | 0.823 |
| 2 | 5.903 | 17.5% | 146.0 MB | 172.5 MB | 0.846 |
| 3 | 3.690 | 20.3% | 142.0 MB | 172.5 MB | 0.823 |
| 4 | 5.953 | 36.8% | 144.0 MB | 172.5 MB | 0.835 |
| 5 | 3.558 | 5.6% | 144.0 MB | 172.5 MB | 0.835 |
| 6 | 5.926 | 41.7% | 160.0 MB | 172.5 MB | 0.928 |
| 7 | 5.944 | 20.8% | 144.0 MB | 172.5 MB | 0.835 |
| 8 | 3.868 | 7.5% | 144.0 MB | 172.5 MB | 0.835 |
| 9 | 3.508 | 5.2% | 142.0 MB | 172.5 MB | 0.823 |
| 10 | 3.458 | 1.5% | 152.0 MB | 172.5 MB | 0.881 |
| 11 | 3.519 | 7.5% | 146.0 MB | 172.5 MB | 0.846 |
| 12 | 3.497 | 0.4% | 144.0 MB | 172.5 MB | 0.835 |

`thp_frac` is 0.823–0.928 — **83% to 93% of anonymous memory is already
hugepage-backed in every launch, including every slow one**, while `ms_token`
spans 1.725x over the same twelve. There is no lottery to lose. The hypothesis
is dead on the range condition before the correlation is even consulted.

## The pre-registered range guard is the reason this is a rejection

A reconnaissance pass of **four** launches returned **rho = -1.0000**, a perfect
rank correlation, over a `thp_frac` range of 0.023. Condition (3) — *the range
must span at least 0.20, so the correlation is measured over a real spread
rather than fitted to noise* — is what refused it. Extending to twelve launches
collapsed rho to **-0.1888**, confirming the perfect correlation was four points
happening to sort.

That is worth stating plainly because it is the failure mode this whole line of
work keeps producing: **a small-n run of a heavy-tailed quantity manufactures a
clean result.** It produced the 1.1910 dispatcher-pin ACCEPT that did not
replicate, it produced a 2.24x "regression" elsewhere that a fourth launch
destroyed, and it produced a perfect rho here. The guard has to be written down
*before* the run, because after the run it is indistinguishable from moving the
bar.

## What the same run established

Sorted, the twelve `ms_token` values are:

```
fast   3.458  3.497  3.508  3.519  3.558  3.690  3.868
slow                            5.903  5.926  5.944  5.953  5.965
```

The slow five span **1.05%**. The gap between the clusters is 2.035 ms; the
largest gap *within* either cluster is 0.178 ms — **11x smaller**. Launch order
does not predict membership (slow launches were 1, 2, 4, 6, 7), so it is not a
warm-up transient; a four-launch reconnaissance that happened to descend
monotonically had suggested exactly that, and twelve launches refuted it.

**Mode ratio 5.926 / 3.512 = 1.687x.** For scale, the +23% steal-tiles candidate
that this null is blocking is a 1.23x effect.

## Bounding what this does and does not license

- It does **not** identify the mechanism. Three are now excluded — dispatcher
  placement, worker placement, page backing — and the field is not exhausted.
- It does **not** license a harness change on its own. Knowing the null is
  bimodal rather than continuous suggests stratifying launches by mode before
  comparing arms, but a stratifier needs a *pre-launch or non-timing* classifier;
  splitting on the timing you are trying to measure is circular.

## The modes replicate, and what separates them is effective width

A second quiet-host run (`benches/acc0_w16_mode_split.py`, 14 launches, taken
~40 minutes later at `runnable=5`) reproduces **the same two levels**:

```
mode A   3.482  3.738  3.760  3.808                          (~3.7 ms)
mode B   5.911  5.917  5.920  5.925  6.024  6.029            (~5.95 ms, 2.0% apart)
tail     7.853  13.011  13.353  14.958                       (see below)
```

Mode B's six members span 2.0%, and both levels land within 1% of the first
run's 3.512 / 5.926. **A bimodal null whose two modes reproduce to within one
percent across independent runs is a property of the system, not of a run.**

**The `park_frac` lead is rejected.** `parks / (parks + spin_hits)` does not
separate the modes: launch 4 sits in mode B at 5.911 ms with 1579 parks, *fewer*
than launch 8 which sits in mode A at 3.808 ms with 2904. Mode A spans
296–2904 parks and mode B spans 1579–10341 — heavily overlapping. The apparent
monotonicity in the contaminated run was the external load, which raises parking
and wall time together. `sys_frac` and `cpu_s_per_token` overlap across the
modes too.

**One derived quantity separates them with no overlap at all** — effective
lanes, `cpu_s_per_token / (ms_token / 1000)`, i.e. how many of the sixteen
lanes were actually busy:

| launch | ms/token | cpu·s/token | **effective lanes** | mode |
|---:|---:|---:|---:|:--|
| 10 | 3.482 | 0.05542 | **15.91** | A |
| 7 | 3.738 | 0.05724 | **15.30** | A |
| 1 | 3.760 | 0.05932 | **15.77** | A |
| 8 | 3.808 | 0.06130 | **16.10** | A |
| 4 | 5.911 | 0.05771 | **9.76** | B |
| 6 | 5.917 | 0.05995 | **10.12** | B |
| 2 | 5.920 | 0.06318 | **10.68** | B |
| 3 | 5.925 | 0.06807 | **11.49** | B |
| 9 | 6.024 | 0.07297 | **12.12** | B |
| 5 | 6.029 | 0.07328 | **12.16** | B |
| 11 | 7.853 | 0.06766 | 8.62 | tail |
| 12 | 13.011 | 0.07385 | 5.68 | tail |
| 13 | 13.353 | 0.07401 | 5.54 | tail |
| 14 | 14.958 | 0.07438 | 4.97 | tail |

Mode A: 15.30–16.10. Mode B: 9.76–12.16. **No overlap, and the gap is 3.1
lanes.** The slow mode is not burning more CPU — launch 4 uses *less* CPU per
token than launch 8 and takes 1.55x the wall time. **It is running on about
ten of the sixteen lanes.**

That reframes the target. The question is no longer "why is this launch slow",
it is **"why does a launch that builds 15 workers on 15 verified distinct
physical cores run at two-thirds of its width, for its entire life, decided
before the first token".**

**The leading hypothesis is foreign load on a subset of the pinned CPUs, and it
is unproven.** It fits every property — a competitor stable over a ~4 s launch
makes those workers permanent stragglers, the barrier waits on them, and
effective width drops in discrete steps with the number of contended cores; it
is decided before the run and constant within it; and it is a property of a
shared host, which is why the same levels recur. It also predicts the tail:
launches 12–14 are consecutive and sit at 5 lanes, which is what more foreign
load looks like, and `sys_frac` there is 0.40–0.45 against 0.15–0.32 elsewhere.
The **falsifier is direct**: sample per-CPU busy time from `/proc/stat` for the
sixteen pinned CPUs across a launch and subtract our own workers' CPU time; if
mode B launches show foreign time on ~5 pinned cores and mode A shows none,
the null is contention the hostlock gate does not catch, and the acc0 width-16
numbers need to be taken with that measured per launch rather than gated once
at acquire. **That has not been run and nothing here asserts it.**

Note what this does *not* rescue: if the mechanism is external, the +23%
steal-tiles candidate is still blocked, but the block becomes a solvable
measurement problem (reject or stratify launches by measured foreign time)
rather than an open question about the pool.

## Reproducing

```bash
scripts/hostlock.sh run --gate 6 --reason "thp a/a" -- \
  python3 crates/onnx-runtime-ep-cpu/benches/acc0_w16_page_backing.py \
    --binary target/release/deps/int4_decode_loop_ab-<hash> \
    --launches 12 --tokens 192 --reps 2
```

The probe refuses a launch that does not report `decode_width ... as_requested`
and prints the child's output on any discard, because the first version of it
silently discarded all three launches of its first run — it was resolving the
model fixtures relative to the wrong working directory, and a harness that
reports a workload failure and a parse failure identically cannot tell you
which one it hit.
