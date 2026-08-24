# The width-16 A/A null is bimodal; it is not page backing and it is not foreign load

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

**REJECT: foreign load on the pinned CPUs does not explain it either.** A
direct `/proc/stat` measurement of non-ours CPU time on exactly the sixteen
pinned CPUs bounds it at **0.59 CPU-seconds against ~34**, a ceiling of 1.7% in
every launch. Costing 3.7 of 16 lanes needs 23%. Spearman rho **+0.0210**, and
the *fastest* launch in the run carries 1.6x more foreign time than the slow
one.

**The finding that relocates the target:** the slow mode does the **same work**.
User CPU per token spans **11% across both modes** while wall time spans
**1.69x**; the slow launch's `sys` per token is **+170%** and its user per token
is **+4.6%**. The lanes are lost to **waiting in the yield loop**, not to work
going missing — the signature of a persistent straggler inside the process, with
placement, page backing and foreign load all now excluded.

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
lanes.** **It is running on about ten of the sixteen lanes.**

An earlier draft of this paragraph read "the slow mode is not burning more CPU —
launch 4 uses *less* CPU per token than launch 8". That is true of those two
launches and false of the modes: mode B's median CPU per token (0.0631) is
above mode A's (0.0592), and the later user/sys split below shows the excess is
**entirely `sys`** while user CPU per token is nearly mode-invariant. Picking
the one B launch that undercuts the one A launch is the small-n manufacture
trap in miniature, inside a document about that trap. Corrected rather than
deleted, because the corrected version is the interesting one.

That reframes the target. The question is no longer "why is this launch slow",
it is **"why does a launch that builds 15 workers on 15 verified distinct
physical cores run at two-thirds of its width, for its entire life, decided
before the first token".**

**The leading hypothesis was foreign load on a subset of the pinned CPUs. It
has now been tested directly and is REJECTED.** See the next section.

## Foreign load on the pinned CPUs: REJECTED

`benches/acc0_w16_foreign_load.py`, 12 trusted launches on a gated host. For
each launch, per-CPU busy jiffies are read from `/proc/stat` for exactly the
pinned set before and after the child, and the child's own total CPU time
(`getrusage(RUSAGE_CHILDREN)` deltas, whole process, not just the steady phase)
is subtracted. The child is pinned to the same sixteen CPUs, so all of its CPU
time lands inside the sum, and the remainder is CPU time on our cores that is
not ours.

| # | ms/tok | lanes | pinned busy | child cpu | **foreign** | sys frac |
|---|---|---|---|---|---|---|
| 1 | 3.534 | 15.93 | 33.80 s | 33.21 s | **0.59 s** | 0.147 |
| **2** | **5.910** | **12.43** | 40.36 s | 40.00 s | **0.36 s** | **0.315** |
| 3 | 3.473 | 16.05 | 34.52 s | 34.09 s | 0.43 s | 0.154 |
| 4 | 3.406 | 16.19 | 33.38 s | 33.41 s | 0.00 s | 0.162 |
| 5 | 3.749 | 15.06 | 34.09 s | 34.09 s | 0.00 s | 0.200 |
| 6 | 3.646 | 15.73 | 35.89 s | 35.87 s | 0.02 s | 0.195 |
| 7 | 3.556 | 15.48 | 34.40 s | 34.23 s | 0.17 s | 0.154 |
| 8 | 3.496 | 15.39 | 33.44 s | 33.26 s | 0.18 s | 0.150 |
| 9 | 3.796 | 14.57 | 34.91 s | 34.86 s | 0.05 s | 0.155 |
| 10 | 3.580 | 15.81 | 33.08 s | 32.98 s | 0.10 s | 0.167 |
| 11 | 3.560 | 15.71 | 34.23 s | 34.10 s | 0.13 s | 0.140 |
| 12 | 3.664 | 15.76 | 33.85 s | 33.79 s | 0.06 s | 0.161 |

Spearman rho between foreign time and effective lanes is **+0.0210** against a
required -0.70 — no relationship, and the sign is wrong. The slow launch is not
the one with the most foreign time; launch 1, at 15.93 lanes, has **1.6x more
foreign time than the slow launch does**.

**The magnitude argument is stronger than the correlation and does not depend
on n at all.** Foreign time never exceeds **0.59 CPU-seconds against ~34
CPU-seconds** of pinned busy time — a ceiling of **1.7%** in every launch,
including the slow one. Losing 3.7 lanes out of 16 requires **23%** of the
pinned CPU time to go to someone else. The measured ceiling is **thirteen times
too small**. Even a run that had sampled only fast launches would license this
rejection, because it bounds the cause rather than correlating it with the
effect.

One condition did pass in isolation and is worth naming as a trap: the slow
launch's foreign time is 3.43x the fast-mode median, clearing the pre-registered
2x bar. That "3.4x" is 0.36 s versus 0.10 s — a ratio of two quantities that are
both noise against a 34-second denominator, from a single slow sample. A rule
that fired on the multiple alone would have manufactured an ACCEPT here. It
survived only because the rule required the correlation *and* checked the
absolute spread first.

### The rule had an ordering defect, fixed before the first launch

As first written, the range guard was tested before the both-modes-present
check, which would have printed `REJECT` on a run that sampled only one mode.
A narrow range in the *cause* is only evidence against a hypothesis once the
*effect* has been observed to vary; on a single-mode run it means nothing.
Note this is the opposite reading of a narrow range from the page-backing
probe above, where a narrow range invalidated an ACCEPT. A narrow range can
never support an ACCEPT, and can only support a REJECT after the effect is
known to have moved. The order is now (1) n, (2) both modes, (3) range,
(4) correlation, and the docstring says why.

## What the slow mode actually spends, and why that relocates the target

The falsifier's `sys_frac` column carries the sharper finding. Splitting each
launch's CPU per token into user and system:

| | fast mode (11 launches) | slow launch |
|---|---|---|
| wall ms/token | 3.406 – 3.796 | **5.910** (1.69x) |
| cpu s/token | 0.0538 – 0.0578 | 0.0735 (+31%) |
| **user s/token** | **0.0452 – 0.0485** | **0.0503** (+4.6% on median) |
| **sys s/token** | **0.0081 – 0.0113** | **0.0232** (+170%) |
| sys frac | 0.140 – 0.200 | 0.315 |

**The useful work is the same. Across both modes, user CPU per token spans 11%
end to end while wall time spans 1.69x** — the metric is roughly fifteen times
tighter than the one the null is measured on. Essentially all of the slow
mode's extra CPU is `sys`.

That is the signature of **stragglers, not of less parallelism per se**: the
same user work is performed, one or a few workers arrive late, and the rest
spend the difference in `worker_wait`'s yield loop, which is exactly the `sys`
consumer Sebastian isolated with his blocktime sweep (sys rises 12x with the
window while parks move the *opposite* way, so it is the yield remainder and
not futex traffic). The earlier framing in this document — "it is running on
about ten of the sixteen lanes" — is right arithmetically but reads as though
work went missing. It did not. The lanes are lost to **waiting**, and the
question is now:

> what makes one worker in a 15-worker pool persistently late, for the entire
> life of a process, when placement is verified one-per-physical-core and there
> is no foreign load on its CPU?

That is an internal question. Foreign load is out; so is page backing; so is
placement (categorical census, `2c3968afb`). The remaining candidates are the
memory placement of the weight arena across the two L3/CCX domains, and
per-launch clock or boost state — both of which are per-process, decided at
startup, and constant for the life of the process, which the mechanism must be.

### The unblock this licenses, stated precisely

Two usable consequences for A/B work at width 16, both of which need no fix to
the pool:

1. **Stratify on an in-launch statistic that cannot see the arm.** Effective
   lanes and `sys_frac` each separate the modes with no overlap on a clean run
   (lanes 12.43 vs 14.57–16.19; `sys_frac` 0.315 vs 0.140–0.200), and both are
   computed from the launch's own counters without reference to which arm it
   is. Rejecting slow-mode launches by a pre-registered threshold is therefore
   not arm-selective. **Caveat: one slow launch.** The lane classifier is
   backed by three runs; the `sys_frac` separation is backed by this one and
   should be treated as a hypothesis until a run samples several slow launches.
2. **Score work-reducing candidates on user CPU per token, where the null is
   15x smaller.** This is null-immune but *narrow*: it measures work, not
   speed. A candidate whose whole mechanism is better load balance — which
   is what the +23% steal-tiles candidate is — moves wall time and `sys` while
   leaving user CPU per token flat, so this metric is the wrong score for it
   and the right *control* for it. It is how you check that a load-balance
   change did not quietly add work.

**Neither of these is a claim that the +23% candidate is real.** It remains
blocked on a mode-stratified re-test against its own unmodified rule.

Note what this does *not* rescue: the mechanism being internal means the block
on the steal-tiles candidate is a measurement problem to be solved by
stratification, not a pool defect to be fixed.

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

The foreign-load falsifier:

```bash
HOSTLOCK_OWNER=roy scripts/hostlock.sh run --gate 4 --wait \
  --reason "acc0 w16 null: foreign-load falsifier" -- \
  python3 crates/onnx-runtime-ep-cpu/benches/acc0_w16_foreign_load.py \
    --binary target/release/deps/int4_decode_loop_ab-<hash> \
    --launches 12 --json bb/w16_foreign.json
```

It reads `/proc/stat` per-CPU rows for exactly `acc0_gap_matrix.PIN` rather than
the aggregate `cpu` row, so an idle-elsewhere host cannot mask a competitor
sitting on one of our sixteen; and it subtracts `getrusage(RUSAGE_CHILDREN)`
deltas rather than the bench's own steady-phase counters, so startup and
teardown CPU are attributed to us and not to a phantom competitor.

**Mode incidence is not stable across runs and should not be quoted as a rate.**
The three clean runs to date produced 5/12, 6/14 and 1/12 slow launches. Any
protocol that depends on sampling the slow mode must check that it did, rather
than assume a hit rate.
