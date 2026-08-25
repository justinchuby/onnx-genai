---
name: "measurement-discipline"
description: "Make a performance or memory number trustworthy before you report it — or believe it"
domain: "quality"
confidence: "high"
source: "earned (#834 knob artifact, #851 self-contended retraction, #853 2x weight bytes, #877→#880 proxy correction, #886 silent corruption, #1619/#1982 selector vacuity)"
---

# Measurement discipline

## Context

This project has retracted several confident performance claims, including two by
the coordinator on the same day. Every one failed the same way: **a number was
interpreted before the conditions that produced it were established.** The
corrections were each worth more than the original claims, but they cost real
time and they are avoidable. Below is the checklist, with the incident behind
each item, because the abstract rule is forgettable and the incident is not.

> **On the CUDA backend, pair this with
> [`cuda-perf-measurement`](../cuda-perf-measurement/SKILL.md).** This skill tells
> you whether a number means what you think; that one tells you which instrument
> to use and how each of them lies on this hardware.

## The failure modes, and what catches them

### 1. A knob you set produced the number you then interpreted

The most expensive one. #834 concluded "the engine commits the full KV bucket
eagerly" from `committed_len == capacity` — but the harness had set
`ONNX_GENAI_KV_MIN_BUCKET = capacity`, and `kv_capacity_bucket` is
`len.next_power_of_two().max(min_bucket).min(hard_max)`. The knob **forced** the
observation. Re-running at the default bucket gave `committed_len = 256` with
`max_len = 8192` — the engine commits on demand, exactly the opposite conclusion.

**Check:** for every knob you set, ask what the metric would read if the system
did nothing at all. If that equals what you measured, you measured the knob.

**The knob is often not a knob.** It can be which constructor, entry point, or
default you happened to call. #930 concluded "capture and weight-streaming are
mutually exclusive on this build" from a genuine decline message —
`weight_offload_enabled && !weight_offload_stable_va`. But the sweep harness
called `load_with_resolved_io`, which passes `cuda_offload_policy: None`, and
`stable_va` then falls to its deliberately conservative `unwrap_or(false)`. The
harness asserted pointer-instability; the runtime was never asked. A run on the
normal engine path, same box and model, had `captures=2 fallbacks=0` alongside
`htod_bytes_per_token = 1,714,132,992` — they coexist, which is what #796 built
and #836 verified at scale. Had it merged, a false negative would have closed
the most valuable question on the #750 line.

**Check:** when a subsystem declines, print the inputs to the predicate that
declined it, not just the predicate's name. `!stable_va` and "this hardware
cannot do it" look identical in a log and mean opposite things.

**Corollary — build a control that could falsify you.** #891 measured `1/N`
amortization *and* showed it identical at `past_len` 0, 512 and 2048, ruling out
a KV-pressure artifact. A result that survives a control it could have failed is
worth far more than one that merely looks good.

### 2. The instrument measured a real thing that was not the thing named

`h2d_enqueue_copy_ms` reported host-to-device transfer at **1.7%** of step time
and retired an entire line of work. It bracketed an *asynchronous enqueue
returning*, not a transfer completing. Measured properly: **18.8%**.

**Check:** divide the bytes by the time and ask whether the answer is a real
bandwidth. A number near link speed, near memcpy speed, or ten times either tells
you what the counter actually bracketed. This takes seconds.

**Check:** state in a comment on every timing counter exactly what lies between
start and stop, and whether anything there blocks the host.

### 3. Arithmetic that cannot close

`total_weight_bytes` was **2.00× too large**: it measured the external-data *file
size*, and that file was 50% orphaned prefix from a re-export (#853). What
exposed it was that measured traffic sat **below** its own theoretical floor,
which is impossible.

**Check:** compute the bound your number must obey. When it is violated, suspect
the instrument before the system.

### 4. A rate over a population with wildly different member costs

Raising the weight budget moved `hit_rate` **57.09% → 81.31%** while the gap to
the streaming floor **widened** 1.78× → 2.30×. Hits skew to ~10 KiB norms, misses
to ~11.9 MB projections. A count-based rate is not a cost metric.

**Check:** weight the rate by what you actually pay for — here `byte_hit_rate`
(#869), not `hit_rate`.

### 5. Contention you did not know was there

The coordinator filed an issue claiming a mandatory gate failed ~1 in 5 runs
solo, then retracted it: two loops it had started itself were contending. A
strict re-run with `nvidia-smi` verified empty **before every individual run**
was 8/8, and wall-clock tightened from a 24–223 s spread to 74–141 s. The spread
itself had been the signal.

**Check:** verify the device is idle **before every individual run**, not once
per loop. A window that is clear when you begin is not evidence it stayed clear.

**Check:** a RED under contention is invalid evidence — re-run solo before
calling it a regression. A GREEN under contention is weak evidence too.

### 6. Wall-clock on a box that pages your own memory

Identical configurations here have ranged **3.9–28 tok/s**, and #863 showed the
OS pages out our own VMM granules under system-wide pressure. Wall-clock is not
evidence on this machine.

**Check:** lead with deterministic counters. #884 proved the byte and page
counters are contention-*invariant* process-local accounting; wall-clock-derived
ones are not. If you must report throughput, give medians with n ≥ 3 and the full
range, with the counters beside it.

### 7. A proxy standing in for the real access pattern

A sequential `cuMemcpyDtoD` from host-mapped memory measured **11.41 GB/s**; the
**real strided int4 GEMV** measured **~5.6 GB/s** (#880). The proxy was
optimistic by ~2×, and it had been published as an upper bound — a loose one.

**Check:** name the difference between your proxy and the real workload, and
treat the gap as unknown until measured. Sweep the size if caching could differ:
the realistic ~12 MiB per-tensor read fits this GPU's L2, so a single point there
compares a cached read against a PCIe read.

### 8. A precondition observed rather than established

A regression test relied on the allocator handing back the same address. glibc
does; the Windows heap does not. It was green on two platforms and vacuous on the
third (#906). The test was right about the invariant and wrong about how to reach
it.

**Check:** if your test needs a condition, **construct** it, and assert
non-vacuously that it held.

### 9. A selector that selected nothing, reporting as success

`cargo test --lib <name> -- --exact` with a **bare** test name matches nothing
when the test lives in a module. It prints `running 0 tests`, `test result: ok`,
and **exits 0**. To a script reading the exit code that is indistinguishable from
the test passing — and inside a mutation battery it is indistinguishable from
*the mutant surviving*, which is the dangerous direction: every arm reports "your
test does not cover this", so you go and write coverage you already have, or
conclude a guard is unreachable and delete it.

```
cargo test -q --lib a_continuing_turn_is_admitted -- --exact
  running 0 tests
  test result: ok. 0 passed; 0 failed; 0 ignored; 2 filtered out      exit 0

cargo test -q --lib tests::a_continuing_turn_is_admitted -- --exact
  running 1 test
  test result: ok. 1 passed; 0 failed; 0 ignored; 1 filtered out      exit 0
```

Two properties make it silent rather than noisy. **A renamed test and a test that
never existed produce byte-identical output**, so the battery repeats its verdict
forever after a rename. And **whether a bare name matches depends only on module
nesting** — a `#[test]` at crate root *is* its own full path and matches; the same
name inside `mod tests` does not — so one battery can hold working arms and
vacuous arms at once, which reads as partial coverage rather than a broken
instrument.

Found independently at least three times here: guarded in code with a comment
saying why (`agrees_with_hostlock_sh.rs`, #1950), recorded as a harness bug after
it reported seven of seven mutations as undetected (#1619), and paid for in full
again in #1982.

**Check:** prove the selector resolved, and prefer a count to a string. `--list`
enumerates matches without running them, so the precheck is separable from the
result:

```sh
n=$(cargo test -q --lib "$FILTER" -- --exact --list 2>/dev/null | grep -c ': test$')
[ "$n" -eq 1 ] || { echo "FILTER-DRIFT: '$FILTER' selected $n, expected 1"; exit 2; }
```

Asserting `1 passed` in the run output is the same idea and is what
`agrees_with_hostlock_sh.rs` does, but on its own it is a substring match on a
*result*, and it fails in the direction this whole section is about. Measured:

```
filter selects 11 passing tests -> "11 passed; 0 failed"   contains "1 passed" -> accepted
filter selects 2, one failing   -> "1 passed; 1 failed"    contains "1 passed" -> accepted
```

Both are false greens: the first accepts a filter that resolved to eleven tests
instead of one, the second accepts a run containing a genuine failure. It is
sound in `agrees_with_hostlock_sh.rs` only because that child selects exactly one
test by construction — `window_probe_child` is a `#[test]` at the integration
crate's root.

So the two checks **compose**, and neither is sufficient alone: the listing pins
the selection to exactly one, which is what makes the substring reading of the
run output trustworthy afterwards. Pin the count first, then read the result.

One residual gap in the count, also measured: `--list` prints an `#[ignore]`d
test with the same `: test` suffix, so a test that gets ignored still resolves to
`n = 1` while the run executes nothing (`0 passed; 0 failed; 1 ignored`). The
listing proves the *name* resolves, never that the arm *ran* — which is why the
run-output half is not optional.

**Check:** every battery carries a **vacuity arm** — a mutation so destructive
that survival is impossible (refuse every request; empty the function body). Its
expected result is the one you know independently of the code under test, so it
is the only arm that can report that the apparatus itself is lying.

**Check:** cite the commit that *introduced* a claim, not the last one to touch
the file. Review of this section caught two wrong PR numbers in it, both produced
by `git log -1 -- <file>`, which resolves to the file's most recent commit and
answers a question nobody asked. `git log -S '<the exact line>' -- <file>` finds
the change that made the claim. Same failure as the rest of §9, one level up: the
command returned a real, correct, confidently-formatted answer to the wrong
query.

## Reporting rules

- **State the conditions with the number** — platform, model, budget, solo or
  not. A figure without them is not reusable.
- **Report a ceiling, not just a gain.** #891 gave `N_max ≈ 19 @ 2048 ctx` and
  said the win saturates; #901 bounded eviction-order tuning at ~10% of the
  recoverable gap. Both stopped someone spending a week for a tenth of the prize.
- **A truthful negative is a first-class deliverable.** Several of the most
  valuable results here were "this does not work, and here is the mechanism."
- **If your result is favourable and rests on one measurement, say so.** #901
  measured MRU as better and still recommended keeping LRU, because the magnitude
  was budget-specific. That is the harder and more useful call.
- **Correct the record where the claim lives.** A superseded number left in a
  design doc will be quoted back as fact.

## Correctness constraints that outrank performance

A performance change on this codebase must also hold:

- **Token IDs byte-identical** for the same prompt. This caught a residency
  policy that silently collapsed generation 16 → 3 tokens (#886) — no error, just
  wrong output.
- `captures > 0`, `fallbacks == 0` (#796): a speedup that silently disables graph
  capture is a regression.
- `peak_committed_physical_bytes < managed_limit_bytes`, `oversubscribed_bytes == 0`
  (#798), and `ref_underflows` / `byte_underflows` / `unaccounted_committed_bytes`
  all 0.
- Never assert inside `Drop` (causes `STATUS_STACK_BUFFER_OVERRUN` here).

## Anti-patterns

- Reporting a speedup without its ceiling or its conditions.
- Re-running a measurement until it comes out favourable.
- Treating a design document as evidence — where a doc and a measurement
  disagree the measurement wins, **including** the "single source", which has
  been wrong at least three times.
- Skipping a mandatory gate silently. Say you skipped it and why: the gate that
  let a real regression through was one that got skipped (#810/#814).
