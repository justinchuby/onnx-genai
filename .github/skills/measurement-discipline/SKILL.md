---
name: "measurement-discipline"
description: "Make a performance or memory number trustworthy before you report it — or believe it"
domain: "quality"
confidence: "high"
source: "earned (#834 knob artifact, #851 self-contended retraction, #853 2x weight bytes, #877→#880 proxy correction, #886 silent corruption, #1619/#1982 selector vacuity, #1995/#2000 selector identity)"
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

**On a shared host, verifying is not enough — you must also claim.** Checking
that the box is idle is a *pre*-check, and a pre-check cannot see a competitor
that starts after you do. Several agents share this machine and used to
coordinate by announcing runs to each other. That protocol has a delivery step,
and the delivery step failed: messages arrived four times over via three
different agents, replies landed in uninvolved sessions, and both parties ended
up idling on each other while behaving correctly. A release that is never
received is indistinguishable from one that was never sent.

**Rule: any saturating benchmark or full EP test matrix must hold the host lock
for its whole duration.** "Saturating" means it takes cores or a GPU that
another agent's measurement would notice — every `benches/acc0_*.py` sweep, any
`--steady` decode loop, and the full EP conformance matrix all qualify.

```sh
scripts/hostlock.sh run --owner <you> --reason "what you are running" -- <cmd>
```

`run` is the form to prefer: it anchors liveness to its own pid, releases on
every exit path including signals, and needs no expiry. If you must use
`acquire`/`release` directly, pass `--ttl 0` and `--pid $$`, because the default
anchor is the invoking shell and the default TTL is one hour — a TTL means
"release this on the clock whether or not I am still running", so on a
multi-hour sweep the default hands the host to a second measurer mid-run and
contaminates *both* sets of numbers while every log line still reads as held.

Take it for the **whole sweep**, not per cell: a per-cell acquire hands the box
back between cells and lets a competitor land inside a matrix whose cells are
only comparable to each other if they all saw the same machine.

Hold it **and** keep the after-the-fact check. They answer different questions —
the lock stops a competitor landing, the load watch notices one that landed
anyway (a co-tenant outside the protocol, or a build nobody announced). Record
`scripts/hostlock.sh provenance` *with* the numbers rather than asserting a quiet
host beside them, so a row taken on a shared box stays identifiable after the
scrollback is gone. The row names the lock directory too, because a private lock
(`HOSTLOCK_DIR` set) coordinates with nobody while producing rows otherwise
byte-identical to real ones.

This is not theoretical: the first two sweeps run after the harnesses started
taking the lock were both refused, against two different agents, at moments when
the host had been announced as free.

#### Why a lock, and not an announce protocol plus a good guard

This gets re-litigated roughly once a week, usually by someone who has just had
a good day with announce-before/announce-after. Three arguments, each from a
measured failure rather than a preference, so the next round is short.

**1. Announcement is pairwise; the host is not.** "I'm taking the box" / "host
free" is a two-party protocol, and this machine is shared by a roster of
eighteen agents with eight or so active on any given day. On 2026-08-25 one
agent correctly yielded the host to a second, and a third then negotiated with
the first for a box that had already been given away. Nobody defected,
everybody was polite, and the protocol still could not represent the state,
because pairwise etiquette has nowhere to *put* a third party. A lock has
exactly one holder and every non-holder reads the same answer.

**2. An announcement describes an edge; a lock covers the interval.** Both
false host-state claims recorded on 2026-08-25 were assertions that outlived
their measurement: one agent read the host, and sent "host free, I checked
independently" from a reading that was by then **74 minutes stale** — a hung
process had started 14 minutes before the message went out. Three test
processes ran for 75, 61 and 51 minutes against a ~7-minute baseline and were
found only because somebody went looking. Note whose claims those were: one
came from the agent who proposed the announce discipline, one from the agent
policing it. A protocol that its own author and its own enforcer each break
within an hour is not being defeated by carelessness.

This is the same defect as `ps`-based liveness, one layer up, and it is why the
outer harness holds the lock rather than each benchmark child: an arm that has
exited because the harness advanced to the next arm looks exactly like a clear
host to anyone sampling processes.

**3. A per-run efficiency guard is self-protective, not preventive.** Per-run
rusage `(utime+stime)/wall` is a genuinely excellent instrument and it belongs
in every harness — it took an A/A null from 52% to 0.04–0.56% on this box. But
it tells you when somebody contaminated **your** run. It does nothing about
**you** contaminating **theirs**, so it does not compose across agents: if
everyone adopts it and nobody locks, every run is correctly labelled and half
of them are discarded. It converts a correctness problem into a throughput
one, which is the right trade on a quiet box and the wrong one here, where
processes hang for an hour unnoticed — you can discard 100% of your reps and
never learn why. It is also blind to SMT-sibling contention and to steady
external load, both of which hold efficiency near 1.0 while moving the number.

So: hold the lock, **and** keep the efficiency guard and the A/A null. The
lock decides whether you may **start**; the guard decides whether to
**believe** the reps you got. Neither substitutes for the other, and the
guards are supplementary — a run with a clean efficiency trace and no lock is
not a defensible measurement.

#### There is a third axis, and it is the one that answers the SMT blind spot

The paragraph above says the efficiency guard is blind to SMT-sibling
contention. That was written as a caveat, and it stayed a caveat for a while
because nothing measured the thing it named. Something does now, so the
guidance is no longer two axes and a known hole.

Ask three questions, in this order, because each is blind to what the next
one sees:

| axis | question | instrument | blind to |
|---|---|---|---|
| **lock** | does anybody *claim* this box? | `scripts/hostlock.sh` | anyone outside the protocol |
| **gate** | is anything *runnable* right now? | `hostlock.sh --gate N` | anything that starts after you do |
| **foreign CPU** | is anyone on **your cores specifically**? | `onnx-runtime-hostmon` | nothing on this list, which is why it is last |

The third is the one that survives a bounded, well-behaved co-tenant — the
case that defeats runnable-count-as-admission-test outright. A deliberately
bounded 4-of-32-CPU protocol is a good citizen and still trips a `-le 3` gate;
it is also invisible to the lock if it never took one. What matters to your
number is not whether the host is busy but whether the busy part overlaps the
cores you were confined to, and that is a different question from both of the
others.

`onnx-runtime-hostmon` (`crates/onnx-runtime-hostmon`) answers it by reading
`/proc/stat` for the CPUs in the process's own `Cpus_allowed_list` and
subtracting the process's own time. Two things about it are worth knowing
before you cite a column it produced:

* **`foreign_pct` cannot see an SMT sibling, by construction.** A decode
  budget of `N` confines the process to `N` *physical* cores, one logical CPU
  per core — so the partner logical CPU of every core you run on is **outside
  your mask**, is never counted, and shares the core's execution units with
  your worker anyway. Measured on a 16-core/32-thread host at budget 12, a
  verified 100%-busy spinner pinned to a sibling slowed the predicted worker
  in five of six arms (p ≈ 2e-5 against a uniform choice among 12) with
  in-shard time up ~1.7x for an exactly equal row segment — while
  `foreign_%` read as low as **0.0**. Use `sibling_peak_pct`, which needs no
  own-time subtraction (you cannot run there, so every busy jiffy is foreign)
  and takes a peak rather than a sum, because under a barrier one saturated
  sibling gates the whole dispatch.
* **It reads the lock at both ends of the window, not once.** A single
  reading at the end reports a plausible holder for a window that changed
  hands halfway through — the stale-snapshot error moved out of `ps` and into
  the row, where it is harder to spot. `hostlock::field` reports `Changed`
  when the two readings disagree, and `Unverified` (rendered
  `unverified:<owner>`; `ab.py` spells the same fact `unverified-end` in its
  CSV) when the second read fails — because an unreadable lock is evidence
  neither for a handoff nor against one.

The library only **reads**. It does not acquire or enforce, and it must not:
taking a lock is a decision the harness makes, and a library that took one as
a side effect of formatting a field would be worse than no lock at all. The
outer harness still holds the lock across every A/B/null arm.

Note the failure mode this axis was built out of, because it is the general
one: `scripts/hostlock.sh` sat on `main` for some time while
`grep -r hostlock crates/` returned **nothing**. No benchmark, no harness and
no result row consumed it. A capability that exists, is `pub`, and has no
caller is indistinguishable in the output from one that was never built — and
the absence reads as success. Shipping the lock was not the same as measuring
under it.

Do not infer HOST FREE from "nothing of mine is running", from a point-in-time
`ps`, or from `/proc/loadavg`. A deliberately-bounded 4-of-32-CPU protocol
shows runnable ≈4–5 and trips a `-le 3` gate while being a good citizen; a
single-threaded 100% CPU hog shows ≈1 and passes. Load average measures the
wrong thing for admission control. Ask the lock.

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
# -p scopes the count to one test binary: without it a workspace-root run
# enumerates every member's lib target and a correct filter reports n = 2.
list=$(cargo test -q -p "$PKG" --lib "$FILTER" -- --exact --list 2>&1) || {
  # The header is echoed separately: piping it through the same `tail` that
  # truncates the compiler output silently ate it whenever the error ran long.
  echo "BUILD-FAILED: could not enumerate tests for '$FILTER'"
  printf '%s\n' "$list" | tail -5
  exit 3
}
n=$(printf '%s\n' "$list" | grep -c ': test$')
[ "$n" -eq 1 ] || { echo "FILTER-DRIFT: '$FILTER' selected $n, expected 1"; exit 2; }
```

Both guards in that snippet were added after measurement. An earlier form sent
stderr to `/dev/null` and read only the count, which gave the **right verdict for
the wrong reason** when the crate did not build: `cargo … --list` exits 101,
prints nothing on stdout, `grep -c` returns 0, and the guard reported
`FILTER-DRIFT: 'root_level_probe' selected 0, expected 1` — naming the filter as
the cause of a syntax error four files away. Checking cargo's status first
separates *the tree does not build* from *the name does not resolve*; they are
different bugs with different fixes, and only one of them is about your filter.

That `||` reads the *assignment's* status, which is the command substitution's
only while the assignment is bare. Wrapping the recipe in a helper — the obvious
way to reuse it — puts a builtin in front, and the builtin's status is its own:

```sh
cmd() { return 101; }                      # a crate that does not build
f() { list=$(cmd);       echo "$?"; }; f   # 101  guard fires
g() { local list=$(cmd); echo "$?"; }; g   # 0    guard silent
h() { local list; list=$(cmd); echo "$?"; }; h   # 101  guard fires again
```

`export`, `readonly`, `declare` and `typeset` mask it the same way, in `dash` as
well as `bash`. The function wrapper above is not decoration: `local` outside a
function is itself an error, so the masking is only reachable in the context that
makes it likely.

Measured through the whole recipe: with `local`, a crate that fails to build
reports `FILTER-DRIFT: '$FILTER' selected 0, expected 1` and exits 2 — the
identical wrong verdict the `||` was added to eliminate, because `grep -c` still
counts the empty string. **`set -e` does not rescue it.** There is no failed
command for `-e` to trip on; `local` succeeded. That is worth knowing here
because bare `run:` steps are `bash -e`, so the shell option people assume is
catching this is not.

Assign bare, or declare separately (`local list; list=$(cmd) || …`). This is the
pipeline rule one step over: **a status is only yours if nothing ran after the
thing you meant to measure** — and `local` runs after it.

Asserting `1 passed` in the run output is the same idea and is what
`agrees_with_hostlock_sh.rs` does, but on its own it is a substring match on a
*result*. Measured — note both arms use **substring** filters, with no `--exact`:

```
filter selects 11 passing tests -> "11 passed; 0 failed"   contains "1 passed" -> accepted
filter selects 2, one failing   -> "1 passed; 1 failed"    contains "1 passed" -> accepted
```

Both are false greens against a substring filter. Against `--exact` they are
**unreachable**, and that is worth stating precisely, because it inverts which
half of the composed check is load-bearing. A test binary's full test paths are
unique and `--exact` demands equality, so **one `--exact` filter selects at most
one test per binary** — `11 passed` cannot occur. `agrees_with_hostlock_sh.rs` is
therefore sound structurally, not merely by construction of `window_probe_child`:
no rename or re-nesting of that test can make its check over-count.

What survives is narrower, and it belongs to the count rather than the string.
Measured per binary under `--exact`:

| situation | `--list` count | `1 passed` / `1 failed` in run output |
| --- | --- | --- |
| name resolves and the test runs | `n = 1`, accepts | accepts |
| bare-name drift, nothing selected | `n = 0`, refuses | refuses |
| the test gained `#[ignore]` | **`n = 1`, accepts** | refuses |

The last row is the one to keep in mind: `--list` prints an `#[ignore]`d test
with the same `: test` suffix, so the count accepts a run that executed nothing
(`0 passed; 0 failed; 1 ignored`) and the result string refuses it. **The listing
proves the name resolves, never that the arm ran.** So compose the two for their
diagnostics — the count distinguishes *the name is gone* from *the test failed*,
which one exit code cannot — but do not describe the count as the half that makes
the string trustworthy. Under `--exact` it is the half with the false green.

**All of the above is about cardinality, and cardinality is the weaker half.** A
count answers *how many tests ran*; it never answers *whether they were the ones
that cover the mutated code*. A filter can resolve to exactly one test, satisfy
every check in this section, and still be pointed at the wrong test — and then a
surviving mutant reads as a clean `PASS`. Measured on a two-test crate, filter
`--exact` on a test that does not touch the mutated function:

```
precheck  n = 1                                    (guard satisfied)
mutate    covered(a) -> a + 1  becomes  a + 99
arm       expects FAIL, gets   1 passed; 0 failed  -> "survived"
control   unfiltered            1 passed; 1 failed -> the mutant IS caught
```

A non-zero count is worse than a zero one here, because it *suppresses* the
suspicion an empty result would have raised. #2000 (closing #1995) hit this: a
substring filter on the word the arm was named after selected a double-digit
number of tests — the battery's own output read `18 passed, 0 failed` — and not
one of them covered the arm under test. The tests that did cover it are the
streaming ones, and none of their names contains that word, which is checkable
in the tree without rerunning anything. A `selected >= 1` guard passes that. So
does `selected == 1`, when the one is wrong.

**Check:** name the test, don't just count it. An arm that expects FAIL must
state *which* test it expects to fail and assert that test appears in the
failures; an arm that expects PASS needs it more, because it has no failure
output to inspect and nothing else distinguishes "the guard held" from "nothing
relevant ran". Cardinality proves the selector resolved; only identity proves it
resolved to the subject.

```sh
: "${FILTER:?}" "${EXPECTED_TEST:?}"   # an empty filter runs the whole suite

# expect-FAIL arm: the named test must be among the failures, not just some test
cargo test -q --lib "$FILTER" > arm.out 2>&1 || true
grep -qE '^test result: FAILED' arm.out \
  && grep -qE "^ +$EXPECTED_TEST\$" arm.out \
  || { echo "ARM-DRIFT: '$EXPECTED_TEST' did not fail under the mutation"; exit 2; }
```

Redirect rather than pipe, and swallow the status with `|| true`, for a reason
that is this section one level up. The obvious form pipes `cargo` into `tee` into
`grep` and reads the pipeline's status — which is `grep`'s status only while
`pipefail` is off. Under `set -euo pipefail`, the idiom in every hardened
battery, the pipeline reports **cargo's** 101 instead, so the `&&` short-circuits
and *every correct expect-FAIL arm reports ARM-DRIFT*. Measured on the same
crate, right filter, mutation live:

```
pipefail off   arm OK      (exit 0)      <- what the author sees
pipefail on    ARM-DRIFT   (exit 2)      <- what the copier sees
```

Review caught that in this very snippet, after I had tested it three ways —
because I ran it in my shell, which had no `pipefail`, and the copier's shell
does. The failure direction is safe (loud, never a false green), which is also
why it could have survived a long time in someone's script as a check that
always fires. The redirect form is verified at 6/6 across both settings.

The expect-PASS arm needs the same naming and cannot use `-q`: quiet mode prints
passing tests as dots and emits no per-test line, so there is nothing to match.
Drop it and assert the name resolved *and* passed.

```sh
cargo test --lib "$FILTER" -- --exact > arm.out 2>&1 || true
grep -qE "^test $EXPECTED_TEST \.\.\. ok\$" arm.out \
  || { echo "ARM-DRIFT: '$EXPECTED_TEST' did not run and pass"; exit 2; }
```

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
