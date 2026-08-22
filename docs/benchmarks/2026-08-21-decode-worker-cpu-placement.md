# The "t=8 wash" is worker-to-CPU placement, not the kernel

**Date:** 2026-08-21 · **Owner:** Roy (CPU MatMul) · **Host:** AMD EPYC 9V74,
32 vCPU (16c x 2 SMT, **siblings adjacent**), AVX2/FMA/F16C, no AVX-512/VNNI.
Ledger entry: §24 of [`CPU_MATMUL_ASSIGNMENT.md`](../performance/CPU_MATMUL_ASSIGNMENT.md).
**Negative result. No kernel change.** Handed to the runtime owner as **#1680**.

---

## 1. The premise, and why it does not reproduce

The reported observation was that #1628's packed-nibble int4 acc4 win holds at
t=1, t=4 and t=16 but **washes out at t=8**, with the implied action being to
look at the kernel.

A thread-count *label* is not a pool width. Re-measured against explicitly set
pool widths on the same binary and shapes:

| pool width | 1 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| acc4 speedup vs acc0 | 1.66x | 1.65x | **1.66x** | 1.238x | 0.993x |

**There is no t=8 anomaly.** The win is flat through width 8 and collapses
*after* it. The shape of the real effect — flat, then a cliff between 8 and 12,
reaching parity at 16 — is not the shape that was reported, and tuning a kernel
against the reported shape would have been tuning against noise.

Width 8 is exactly the physical core count of one half of this machine, which is
the tell.

## 2. Hypotheses tried and discarded

**Memory bandwidth.** A STREAM-style all-thread sweep read 83 GB/s against a
41 GB/s decode draw, which would make the loop comfortably compute-bound. That
figure **does not reconcile with §22 of the ledger**, which measured this host
at 31-36 GB/s within a CCX and ~56.6 GB/s across both — and §22's numbers are
the ones the ledger stands behind, because they were taken with the access
pattern the decode loop actually uses. Against §22, a 41 GB/s draw is 72% of the
across-CCX ceiling and *above* the within-CCX one.

So bandwidth is **not** dismissed by the 83 GB/s sweep, and is not dismissed on
that basis. It is ruled out by §3's placement A/B, which holds shapes, bytes,
thread count and binary constant and changes **only which CPUs the workers sit
on**. A bandwidth ceiling is indifferent to that. The result is not.

**Task grain.** Shard count and per-shard barrier time were read directly and
move as expected with width — no straggler, no grain cliff at 8 or 12. Discarded.

## 3. Root cause: logical-order pinning

`decode_spmd.rs::node_shards` pins worker *i* to `allowed_cpus()[i]` — **logical
order, no topology awareness**.

On this host SMT siblings are adjacent: CPUs 0 and 1 are the two hardware
threads of physical core 0. So a 16-worker pool lands on CPUs 0-15, which is
**8 physical cores**, and every worker contends with a sibling for the same
execution units. A width-12 pool puts 4 of its 12 workers on shared cores, which
is exactly where the table above starts to bend.

Verified two ways.

**Observationally**, by reading `/proc/<pid>/task/*/stat` field 39 (`processor`)
for every worker thread during a run: the 16 workers report CPUs 0 through 15.

**Causally**, by changing nothing but placement on the same binary and shapes:

| placement, 16 workers | speedup |
|---|---|
| default (`allowed_cpus()[i]`, CPUs 0-15 = 8 physical cores) | 0.982x |
| one worker per physical core (`taskset -c 0,2,4,...`) | **1.225x** |

Same kernel, same data, same worker count. The only variable is which CPUs they
sit on, and it moves the result by 1.25x.

## 4. Disposition

**No kernel change was made.** The kernel is not implicated by any measurement
here, and distorting it to compensate for scheduler placement would bake a
host-topology artefact into shipped code — the specific failure mode the
directive named. Filed as **#1680** with the pool-width sweep, the `/proc`
evidence and the placement A/B, for whoever owns `decode_spmd.rs`.

## 5. The part that affects everyone else's measurements

**Every unpinned multi-thread number taken on this host above pool width 8 is
contaminated**, and the contamination is silent — it looks like a kernel that
stops scaling rather than like a placement bug.

Two workable rules, either sufficient:

- pin explicitly with `taskset -c 0,2,4,6,...` (even CPUs are distinct physical
  cores on this machine), or
- keep pool width at or below 8, where the default placement happens to be
  correct by accident.

Both §23 and §25's records were taken under the first rule. Any older
multi-thread figure in this directory taken above width 8 without pinning should
be treated as unreconstructed.
