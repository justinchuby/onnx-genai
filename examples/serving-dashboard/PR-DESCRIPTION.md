# Serving dashboard demo: an honest view of a live inference server

Adds `examples/serving-dashboard/` — a browser dashboard that shows what an
`onnx-genai` server is actually doing while it serves — plus the engine and
server work needed to make those numbers real rather than plausible.

The bulk of the diff is the demo directory and its tests; the crate changes are
comparatively small and are listed under *Dependency and contract changes*
below. (Commit and line counts are deliberately not restated here — GitHub
renders them live above, and a hand-copied total is stale the moment the branch
moves. It moved three times while this description was being written.)

---

## The rule this demo is built on

**A dashboard that shows a number it did not measure is worse than one that
shows nothing.** A flat zero and a real zero look identical, and the visitor
cannot tell them apart — so every field here must be able to say *where it came
from*, or say that it is unavailable and why.

That rule generated most of the engineering below. It is also the reason this
PR is long: several fields turned out to be unmeasurable, and saying so
correctly took more work than printing a plausible value would have.

---

## Headline technical result: paged KV and continuous batching are mutually exclusive

This is the most important thing we learned, and it was **not** the thing we set
out to build.

The continuous-batching path and the paged-KV cache **cannot both be active on
the same engine.** `ContinuousBatchManager` never touches `engine.kv_cache`; the
driver takes one path or the other (`driver.rs`, `continuous_batch_manager`).
So a single server demonstrates *either* continuous batching *or* paged KV with
prefix caching — never both at once.

**That is why the demo runs two servers rather than one**, and why the
dashboard's scenario tabs navigate between origins instead of switching panels
in place. The two-server layout is not a presentation choice; it is the shape of
the runtime. `ARCHITECTURE.md` §5.6 has the full statement.

The visible consequence is that several KV fields are **structurally
unavailable** on the batching server — not missing, not zero, not "still
loading". The dashboard says so, names the endpoint the value would have come
from, and explains why it cannot arrive. Details in the README's field-state
section.

---

## Performance: what we are claiming, and what we withdrew

**We are not shipping a throughput ratio.** An earlier draft of this PR led with
one. It has been deleted rather than hedged, and the reason is not the usual
"results vary by hardware."

> **The model the numbers were measured against cannot be rebuilt.** It was
> assembled by accident from **two builds seventeen days apart**, and its
> inference metadata file was edited **fifty-four minutes after the model was
> built, inside the measurement window**, while every sibling file in that
> directory is timestamped within a minute of the build. **Nobody has established
> that measurements taken either side of that edit are comparable.**
>
> That makes this worse than unreproducible. **We cannot show the figure is
> internally consistent with itself.** A performance claim whose artifact nobody
> can rebuild is not a measurement — it is a rumour with a decimal point, and the
> decimal point is doing the persuading. Two decimal places asserted a resolution
> this setup cannot support, which is the same defect as drawing a 4-slot batch
> as `75 %`: **a format is a claim about how finely a quantity can be known.**

**Why deleted rather than footnoted.** A hedged number is still quoted; the hedge
is dropped in the retelling and the digits survive. And the failure mode is
asymmetric — if a reader tries to reproduce a figure and cannot, **every other
claim in this PR becomes suspect, including the ones that are true and were
expensive to get right.** The mutual-exclusivity finding above is the result
worth having, and it is a *mechanism* claim: it is checked by reading
`driver.rs`, not by trusting our machine.

**What we claim instead, all of it checkable by reading code:**

- **Continuous batching edits a running batch between steps rather than between
  batches** (`crates/onnx-genai-engine/src/batched.rs`). `submit` queues a
  request; each `step` calls `admit_available_rows`, which retires finished rows
  and admits queued ones in the same pass. The static contrast is
  `generate_batched_static`. **The effect is on waiting, not on speed:** under
  static batching an arrival waits on the longest member of the batch in flight.
- **Paged attention draws the KV cache from a shared pool of fixed-size pages**
  (`crates/onnx-genai-kv/src/page_table.rs`, `allocate` / `free`) instead of one
  contiguous per-sequence buffer, so capacity is taken as tokens are generated
  rather than reserved for the longest generation a request might produce — and
  identical pages can be shared instead of copied.
- **The tradeoff, which needs no number:** batching raises *aggregate*
  throughput and makes no single request faster. Per-stream throughput falls as
  total throughput rises. **A tradeoff presented as a pure win is a lie told with
  true numbers**, and the half an engineer actually needs is the second one.

**The raw capture stays in**
[`examples/serving-dashboard/perf-baseline.md`](examples/serving-dashboard/perf-baseline.md)
**as a lab notebook** — commands, hardware, load average, ordering, and the
binary and model SHA-256s. **Read it as a record of what was run, not as a
result.** Deleting it would suppress the evidence that the claim was unsafe.

`check-perf-claims.test.js` now enforces the withdrawal instead of the figure: it
fails if any shipping document reintroduces the ratio.

---

## Results that were inconclusive, stated as inconclusive

**Prefix caching produced no measurable improvement in the scenario we built for
it, and we are not claiming one.**

Reading the code explains why, and the explanation is more useful than the
measurement would have been. There are two prefix branches, and **only one of
them restores anything**: the token-prefix branch scans cached token sequences
and keeps the longest overlap, but **never touches the page table and
materialises no KV** — no prefill is skipped. The other branch calls
`lookup_shared`, matches pages, materialises them, and genuinely shrinks
prefill. Our scenario exercised the first.

A null result tells you nothing about *why*. The mechanism does, and it predicts
every number we recorded. Full analysis in the README's Scenario B section.

**We did not promote a faster model build.** A rebuilt variant verified fine,
but the withdrawn baseline was measured against the original artifact — swapping it would
have invalidated the comparison. Verified does not mean it should be shipped.

---

## Dependency and contract changes

**One new direct dependency:** `tower-http = { version = "0.6",
default-features = false, features = ["fs"] }`, for serving the dashboard's
static files at `GET /demo`. Only the `fs` feature is enabled — a hand-rolled
directory server is a well-known source of path-traversal bugs and is not worth
writing for a handful of static files.

**`Cargo.lock`: +45 lines, four transitive packages** — `futures-sink`,
`http-range-header`, `mime_guess`, `tokio-util`. No existing package changed
version.

### ⚠️ Contract change for the cluster router: `kv_usage` may now be absent

`NodeStatus` previously published a hardcoded `0.0` for `kv_usage`. It now
**omits** the field when the value is unknown.

**Omission specifically, rather than `null`:** the router's deserialization
mirror fills missing keys via `#[serde(default)]` but **rejects an explicit
`null` into a bare `f32`**, which would fail the parse and mark the node
unhealthy.

**This is not cost-free, and reviewers of the router should know.** `kv_usage`
feeds load balancing, and a missing value defaults to `0.0` — "this node's KV is
empty" — which biases traffic *toward* a node that simply cannot report. **That
was equally true of the hardcoded `0.0` it replaces, so it is not a
regression**, but the router still needs to distinguish *unknown* from *empty*
before it can route on this field honestly. Same applies to `kv_pages_used`,
`kv_pages_total`, and `model_id`.

---

## Tests

The JavaScript suite — documentation-drift checks plus the dashboard's own
tests — has exactly one way to run it:

```bash
./examples/serving-dashboard/run-tests.sh
```

> **This section no longer states the command, only cites it.** It previously
> gave `node --test check-*.test.js` — a *single glob*, which does not recurse
> and silently omitted every suite under `dashboard/` and `ui/` while exiting 0.
> The runner discovers files instead of listing them, reconciles the discovered
> count against the suites Node actually executed, and refuses to report a total
> that includes untracked files. `CONTRACT.md` §7 records the five incompatible
> forms this replaced and why each one was green and wrong.
>
> **Counts below are anchored, because a test count is a claim with a shelf
> life.** Re-run rather than believe them.
>
> - **632 tests across 95 suites, 0 failures** — measured at `02b54684` on a
>   clean detached worktree (`porcelain 0`), Node v25.6.1, 47 discovered files,
>   0 untracked, 0 tracked-but-missing.
> - **136 Rust tests** in the server crate — carried from `e6dd848e` and **not
>   re-measured for this revision**; treat it as the older claim it is.

The drift checks exist because **documentation rots faster than code and nothing
tells you.** They bind the README to the repository: every cited file and line
must exist, **every cited line must still sit beside the symbol the prose names**,
every documented CLI flag must be a real flag, every field state's glyph must
match what the renderer prints, and every performance figure must be recomputable
from the raw samples.

**Every check in this suite has been watched fail.** A test that has never failed
is a hypothesis, not a check — several of these were green on first write and
turned out to be asserting nothing.

`repair-citations.mjs` recomputes stale line numbers from the symbols the prose
names, because a line number in documentation is a hand-maintained copy of a
fact that lives in a source file, and copies rot. It is deliberately not wired
into the suite: a check that repairs its own subject can never fail.

---

## Known gaps

- **No browser pass has been recorded against the final build.** Code review is
  not QA, and we are not presenting it as such. The last unobserved surface is
  the rendered page in grayscale and colourblind simulation — if
  `not-applicable` and `unavailable` render identically, the five-state model
  silently becomes four on screen. Reading CSS tells you what you specified;
  only looking tells you what a visitor gets.
- **`ARCHITECTURE.md` §8.9 previously said `max_batch` was not configurable.**
  Corrected in this PR: `--max-batch` exists (`ONNX_GENAI_MAX_BATCH`, default
  4). The same two sentences also cited `DEFAULT_MAX_BATCH` as
  `state.rs::DEFAULT_MAX_OUTPUT_TOKENS` — a different constant with a different
  value — and asserted that batch utilization was computed against `max_batch`.
  It is not; it divides by `effective_batch_capacity()`.
- **Batch occupancy mixes scopes.** The numerator is a process-global counter;
  the denominator is one configuration's ceiling, so the numerator can
  legitimately exceed it. The value is clamped, which hides a scope mismatch
  rather than a bug. Documented in the README rather than silently corrected.
- **Most fields on the page are `—`, and the two headline panels are the
  emptiest.** Measured two independent ways that agree: a live `data-state`
  census of the rendered page (**40 of 51** field states `unavailable`), and a
  static count of keys `dashboard/kv-memory.js` passes to `field()` that have no
  catalogue entry (**10 of 13, 77 %**). One counts pixels, one counts bindings.
  Every dash is a metric no endpoint publishes, and the page saying so is the
  point — **but a dashboard that degrades honestly is still a dashboard not
  showing you much yet.** The degradation machinery is complete and tested; the
  telemetry coverage behind it is early. Those are two different maturity
  levels and the page's calm presentation can blur them. Quantified in the
  README under *How much of the page is actually populated*, and gated there so
  it cannot drift — **including upward**, since a figure that improves silently
  is one nobody reports.
- **`dashboard/field-keys.test.js` verifies only field keys written as string
  literals.** Its extractor requires a quoted argument, so the 15 keys
  `dashboard/throughput.js:274` builds by template interpolation
  (`` `${definition.prefix}_${percentile}` ``) are invisible to it; none of the
  15 has a producer anywhere in the tree. **The subtle part is the allowlist:
  `NOT_YET_PUBLISHED` contains exactly two of those 15 — precisely the two that
  also appear as literals — each with a written rationale that applies equally
  to the other 13.** It reads as a surveyed exemption and is actually the
  extractor's visible subset. The guard is sound and narrower than it looks;
  treat it as covering the literal half of the field-key class.
- **A green gate is not a satisfied reviewer.** The mechanical gate reached 6/6,
  and the code reviewer's standing verdict is still REQUEST CHANGES on findings
  the gate does not encode. These are different instruments measuring different
  things, and reporting the gate alone would be the *all-clear-terminates-inquiry*
  failure this PR spends its length arguing against.
- **`demo-spec.md` is committed as a snapshot.** Do not quote a total from
  anywhere else, **including from this sentence** — which said `159` until
  `05:38` and was wrong by fifty-one, having been true when written. Count it
  yourself: `grep -cE '^- \[[ x]\] \*\*AC' demo-spec.md`. **A number in prose is
  a copy of a fact, and copies rot; this bullet rotted inside the section that
  warns that documentation rots.**
- **Prefix caching is demonstrated, not proven beneficial.** See above.

---

## What we do not claim

**This section is not a disclaimer. It is the product.**

This is a dashboard about honest measurement, built by a crew that spent a
session discovering its own instruments failed in exactly the way the product's
worst bug did. That rhyme is the story, so it belongs here rather than in a
postmortem nobody reads.

**The one diagnosis behind almost every defect we found: we could not tell *no
data* from *data*.** An em-dash drawn over live observations. A `git archive`
extract that disarmed ten guards and dropped sixty-six tests **while the suite
count stayed identical**. A citation harness that printed *"OK — every anchored
citation resolves"* at exit 0 when the sources were missing, **its confidence
scaling with how much of the checkout was absent**, because a universal claim
over an empty set is vacuously true and reads as thorough. A guard *refusing to
run* rendering identically to a guard *finding a defect*. A launcher testing
that a binary **exists** to answer a question about which **source tree** built
it.

> **Every instrument we built checks that what is present is true. Not one
> checked that what is true is present.**

The product now has a third state, and so do the tools: `0 = clean`,
`1 = a defect was found`, `2 = cannot run`. **Two-state thinking was the root
defect of this work — in the tools, in the reviews, and in the page.**

**And the law we would most like carried elsewhere, because no guard we own can
enforce it:**

> ### A layout is a claim.
> Putting two series on one locked axis asserts *these are commensurable* as
> loudly as a sentence would — and every checker we have reads strings. **A
> confound expressed in geometry is invisible to all of them.** It was caught
> with a stopwatch and four arms, not a grep.

### The specific things this PR does not assert

- **The throughput ratio is withdrawn, not softened.** It is absent from this
  document as a live figure and must stay absent. The model it was measured
  against cannot be rebuilt, so **the claim is immune to re-running** — which is
  worse than unreproducible. Full reasoning in *Performance* above.
- **The two panes are NOT comparable, and this is by construction.** Same
  request stream, two cache strategies, **two different capabilities** — never
  *"A vs B"*. Static/scatter cache lets rows share one buffer so continuous
  batching runs; a dynamic cache grows per sequence, so paged KV and batching
  cannot coexist. **Four rows on one side and one row on the other is the
  architecture, not an artefact.** No caption, legend, header or footer may
  present a cross-pane delta: **any number derived from both panes is a defect
  by construction.**
- **Some fields are `MISATTRIBUTED`, and the word is new because we lacked it.**
  Our vocabulary had `DOCUMENTED_ZERO` (a constant), `NOT_PLUMBED` (absent) and
  `STRUCTURALLY_BYPASSED` (never asked) — and **no word at all for *asked,
  answered, answering something else.*** `MEASURED` was the only remaining
  option, so three fields were classified `MEASURED` **by the shape of the enum
  rather than by anyone's judgement.** `prefix_cache_hits` is the case: it
  increments off a shared chat preamble, so it **reads the same with and without
  the reuse it purports to measure.** *When you cannot find the right label,
  that is a finding — not a prompt to pick the nearest one.*
- **Two designed behaviours ship SPECIFIED-NOT-BUILT** — the provenance-unknown
  badge (D298) and panel-header attribution (D300). They are disclosed in-file
  where a reader meets them, rather than implied by a design document that says
  they exist.
- **L5, L8 and L9 ship unmeasured.** In those words. No predicate arrived for
  them, and *"shipped unmeasured"* is the accurate phrase; anything softer would
  be the euphemism this document exists to refuse.
- **We cannot fully say which code was running.** The demo servers observed
  during this work **can name no commit**, so a fixed source tree and a leaking
  process coexisted for hours. **The code is fixed; a running process is not the
  code, and that distinction is a restart, not a commit.**
- **And the reason we were slow to see that is worth more than the incident.**
  For hours this was described — in this document's own earlier draft — as
  processes *"started from a sibling checkout."* **There is no sibling
  checkout.** There are **nine working trees of one repository**, sharing a
  single object store: `git rev-parse --show-toplevel` returns a **different**
  path in each, while `--git-common-dir` returns the **same** one in all of
  them. **Everyone ran the first command, correctly, and got the correct
  answer; nobody ran the second.** *An instrument that answers "which directory
  am I standing in" was read as answering "which repository is this," and it
  was right every time it was asked.* **The consequence is the hazard: because
  the object store is shared, every commit from every other tree resolves here
  — so a read against the wrong tree returns a confident, well-formed, wrong
  answer, and no error is ever printed.** ***The two most dangerous words in a
  measurement are the ones nobody thought were a measurement.***

### Verification, at one revision, with denominators

**Every number here carries its revision and its denominator, or it is not
here.** Re-run rather than believe.

- **JavaScript: 744 tests / 114 suites / 743 pass / 1 fail — raw unpiped exit
  `1`** at `8a309ce0`. **The tree is red at that revision and we are not
  rounding that off.**
- **That single failure is a guard working correctly**, and it is this PR's
  thesis in miniature: `check-perf-claims.test.js` reports that the refusing
  match arm is no longer `PastPresent { .. } | Legacy => bail!` while the README
  still tells the reader the unresolved question is *exactly two ways wide.* Its
  own message names the direction of the error — **the stated width would be
  wrong toward claiming more certainty than we have.** The guard is defending a
  *width*, not a value.
- **⚠️ Read those two bullets as a snapshot, and here is the proof they must
  be.** Four minutes earlier, the same suite at `090e68ea` reported **740 / 113 /
  1 fail** — and **the failing test was a different one**
  (`check-review-freshness.test.js`, whose stale `KNOWN_ABSTAINERS` entry was
  retired in between). **The count moved, the denominator moved, and the
  identity of the red moved, inside four minutes, with nobody doing anything
  wrong.** ***A test result is a measurement of a revision, not a property of a
  project*** — which is why every figure here carries a SHA, and why *"the suite
  is green"* is not a sentence anyone should write without one.
- **Rust: `cargo test -p onnx-genai-server` — 264 pass / 0 fail / 4 ignored,
  raw unpiped exit `0`, at `964cad4a`**, published with its per-binary split
  (`lib` 211·3 · `demo_dashboard` 15 · `http` 28·1 · `vlm_image_bundle` 10 ·
  `main` 0). **Scope is one package, not the workspace** — other crates are
  unmeasured here.
- **The 4 ignored are named in *Tests* above rather than netted off**, because
  **a skip is invisible to a pass rate**: `264/264` and `264/268` are the same
  screen. **Three of the four are the multimodal pipelines** — the least
  exercised surface on the branch. Not blocking; stated *beside* the 264, not
  after it.
- **There is deliberately NO combined total.** The JavaScript suite reaches no
  Rust code, so one figure would imply a coverage that does not exist. **Two
  suites, two denominators, both quoted.**
- **⚠️ And one discrepancy we are publishing precisely because we could not
  close it.** A static census of `#[test]`/`#[tokio::test]` attributes in the
  crate source, at the same revision, counts **216** in `src/` where cargo ran
  **214**. **The instrument is not simply wrong: on the three integration
  binaries it agrees with cargo exactly, 54 and 54.** It disagrees only on the
  lib target, and only by two. Ruled out: branch drift (the census is identical
  at a revision 111 commits later), a `.rs` file declared by no `mod` (none),
  feature gating (no test-bearing file carries one, and `default = ["metrics"]`
  while `metrics.rs` has no tests), duplicate function names (none), and
  attributes sitting inside raw-string fixtures. ***Two test functions appear to
  exist in the source and not in the run, and we do not know why.*** It is two
  tests out of 270 and it blocks nothing — **but the honest form of a number
  you cannot reconcile is to print it with its denominator and say so, not to
  round it to the figure you prefer.**
- **A green gate is not a satisfied reviewer**, and a red one is not an
  unsatisfied product. Different instruments measure different things.
