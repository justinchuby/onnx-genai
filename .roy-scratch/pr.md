## Why

A cross-agent note asked for **every `t>=8` row published before #1729 (`6e8c31ebd`) to be re-taken**, on the grounds that the default decode pool pinned 16 workers to cpus 0–15 — 8 physical cores with both SMT siblings loaded.

The premise is correct, and it is the same defect this repo found independently and filed as **#1680** (§24 of `CPU_MATMUL_ASSIGNMENT.md`). The blanket conclusion is too broad for the rows in that file, and — the point of this PR — **which rows survive is decidable from the source, without spending host time re-measuring.**

## The argument

Every multi-thread timing in §23/§25 is `taskset`-pinned to the even CPUs (§24's closing note). On this host SMT siblings are adjacent pairs, so that mask is 16 CPUs that are *already* 16 distinct physical cores.

**The spread half of #1729 is the identity on such a mask.** Pre-#1729 the SPMD shard builder used `allowed_cpus()` in raw ascending order (confirmed at `6e8c31ebd^`); post-#1729 the same list goes through `order_pin_targets`. Its `Spread` arm is `leaders_within(cpus)` plus the non-leader remainder — and on an all-even mask each core group contributes its single allowed member, the remainder is empty, and `leaders_within` ends `sort_unstable(); dedup()`. Same ascending list. `build_decode_pool` pins worker `i` to `cpus[i % len]` either way, so **#1729 cannot move a number taken under that pin.**

**The reserve half is not the identity, and it bites at exactly one width.** With `allowed = cores = 16`, `reserve_single_group_headroom(total, 16, 16)` returns `total.min(15)`:

| explicit width | pre-#1729 | post-#1729 |
|---|---|---|
| 1, 2, 4, 8 | 1, 2, 4, 8 | unchanged |
| 16 | 16 | **15** + a free core for the inline dispatcher |

**#1794 does not apply.** It fixed `default_persistent_threads` returning `available / 2`. These sweeps set `ONNX_GENAI_CPU_DECODE_THREADS` explicitly, and an explicit count bypasses the default entirely. The rows that defect corrupted are pinned runs that left the width to the default — of which this file has none, because the pin and the explicit width were adopted together.

**Disposition: `t<=8` survives unchanged; `t=16` does not** — and `t=16` was already withheld for an unrelated reason (its A/A null spans 0.969–1.295, ±30%, against 3.6% at `t=1` and 2.8% at `t=8`). Second time this week a figure has been retired by two independent arguments at once.

## The test, and why it is the real content

That argument was **load-bearing and untested**. `cargo test order_pin_targets` matched **zero** tests, and none of the existing placement tests covered the fixed point.

So it is asserted rather than argued: **a cpuset that already holds one CPU per physical core is unchanged by either placement policy.**

This is not a niche property. It is the guarantee *every* pinned benchmark in this repository rests on — the house rule for a clean multi-thread number is `taskset` to one CPU per core, and that rule is worth nothing unless the pool then pins workers to the CPUs that were reserved, **in the order they were reserved**. It is also exactly what separates "#1729 invalidates this measurement" from "#1729 leaves it alone", which is a question several of us are now asking about our own records.

The test also asserts the two policies **still disagree on a full mask**, so it is a property of the *mask*, not a policy that never reorders anything.

**Live, not vacuous:** reversing the leader order inside the `Spread` arm fails it with

```
`spread` reordered a mask that was already one CPU per core, so a pinned
benchmark's workers do not land on the CPUs it reserved
```

## Also recorded

`t=2` is **closed**, by two methods sharing no apparatus: **1.96x** measured here on a quiet host (20.447 vs 40.039 ms/token, 0.6% A/A null, both workers 99% busy by per-thread attribution) and **1.94x / 97% efficiency** reported independently by the runtime owner on a post-#1729 baseline. ~1% apart. The withdrawn "71% of one core" is now over-determined.

One standing caveat is reinforced rather than revised: `t=1` runs `path=flat` and `t>=2` runs `path=spmd-pool`, so a `t=1` vs `t=2` comparison crosses routes as well as widths. §20 already reads that row as "vs serial" rather than "vs a one-worker pool".

## Validation

- `cargo test -p onnx-runtime-ep-cpu --lib` — **1828 passed, 0 failed** (one new test), under `scripts/hostlock.sh`
- `cargo clippy -p onnx-runtime-ep-cpu --all-targets --all-features -- -D warnings` — clean
- `cargo fmt --all --check` — clean

No production code changed: one test plus a documentation section.
