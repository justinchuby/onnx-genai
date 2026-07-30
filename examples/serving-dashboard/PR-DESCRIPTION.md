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
  checkout.** There are **many working trees of one repository** — the count
  was measured as 6, 8 and 9 by three people inside fifteen minutes, all
  correct when taken, which is why **the number is not the finding and is
  deliberately not printed here** — all sharing a single object store: `git rev-parse --show-toplevel` returns a **different**
  path in each, while `--git-common-dir` returns the **same** one in all of
  them. **Everyone ran the first command, correctly, and got the correct
  answer; nobody ran the second.** *An instrument that answers "which directory
  am I standing in" was read as answering "which repository is this," and it
  was right every time it was asked.* **The consequence is the hazard: because
  the object store is shared, every commit from every other tree resolves here
  — so a read against the wrong tree returns a confident, well-formed, wrong
  answer, and no error is ever printed.** ***The two most dangerous words in a
  measurement are the ones nobody thought were a measurement.***

### The closing argument: who is looking is part of the measurement

**Every defect above is one defect wearing different clothes.** We kept building
instruments that could not tell **nothing** from **something**, and each time we
found one we assumed it was the last.

**The kind, with named instances — deliberately not a tally, because the count
is still moving and a number frozen here would rot the way three others in this
document already did:**

| the instrument said | what was actually true |
|---|---|
| `—` (no data) | 46 live observations |
| `OK: every citation resolves` | the sources were absent; a universal over an empty set is vacuously true, **and its confidence scaled with how much was missing** |
| a suite of unchanged size | an archive with no `.git` had disarmed ten guards and dropped sixty-six tests |
| `exit 1` — a defect was found | a tool had crashed with a traceback |
| a red test | a guard **refusing to run** |
| a file cited as at-risk | the file did not exist |
| a spec that never mentioned the P1 | absence makes no claim, so no sweep can ever find it |
| a green disclosure guard | it read a hardcoded list of three files, and the leak was in the fourth |
| `-x $binary` — the build is present | a process from an entirely different tree |

**Nothing in that column on the left was a lie.** Each was a correct reading by a
correctly-built instrument. **They were true, and then they weren't, and nothing
told us.** *The missing field was never rigour — it was an expiry date.*

**And the last two instances are the ones worth carrying, because they are about
people rather than tools:**

> ### A layout makes a claim no string-reader can see.
> ### An `aria-label` leaks through a channel no sighted reviewer can check.
> ### **Who is looking is part of the measurement.**

**This is why the five-field standard we enforced all session — revision, exit
code, denominator, positive control, clock — was not enough, and the reason is
uncomfortable: *every one of those fields gets **stronger** as you measure the
wrong channel harder.*** Two people checked this page for a path leak in a real
browser and found clean text and a clean tooltip — **a flawless five-field result
about a channel that was not leaking.** ***Rigour is orthogonal to aim. A
discipline that measures only rigour will certify a perfectly-executed
measurement of the wrong thing.***

**So the sixth field is the one no instrument can supply, and it is the habit
this release is really arguing for:**

> ### *What would I have seen if I had looked somewhere else?*

**That is the whole product in one line.** The dashboard's job is not to be
confident. **It is to say `—` when it does not know, and to say it loudly enough
that you go and look somewhere else.**

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
- **⛔ We cannot certify the suite is deterministic. 66 green runs and two
  unexplained reds.** **Do not read a passing total as proof of stability. Read
  it as proof of the tree at one moment.** The instrument that would have named
  those two reds did not exist when they happened — a piped run showed the word
  `FAIL` and discarded the only copy of the diagnosis. **The runner now re-prints
  the failing names as its final lines, so the next red names itself.** We are
  shipping a dashboard whose thesis is *claim only what you measured*; **we do
  not get to round 66-green-and-2-unexplained up to "stable" in our own PR.**
- **⚠️ And the totals in this section come from more than one invocation, which
  is a defect in the reporting, not in the tree.** A bare `node --test` glob and
  the project's own `run-tests.sh` discover different file sets, so their
  denominators are **not addable** — the same tree scored `744/114` one way and
  `710/109` another. **Two scoreboards for one suite is the same failure as the
  two citation dialects and the two path-disclosure guards: two descriptions of
  one thing, and nothing saying which wins.** `run-tests.sh` is the canonical
  one, because **it is the only instrument here that reports its own tree, its
  own dirtiness, and its own discovered-vs-executed reconciliation without being
  asked** — and it refuses to score a total that includes uncommitted files,
  since a clean clone does not have them.
- **A green gate is not a satisfied reviewer**, and a red one is not an
  unsatisfied product. Different instruments measure different things.

---

## Known gap: two routes to one capability, one loud and one silent

**This is the sharpest engineering finding of the review, it is not a
regression, and nothing in this PR changes it. It is written down so the next
person does not have to rediscover it.**

Continuous batching can be reached two ways, and the two fail in opposite
directions with nothing in the code marking them as different:

| route | when the precondition is missing | posture |
|---|---|---|
| static cache | **refuses to boot, naming every missing key** (`reject_undeclared_static_cache`) | fails **closed and loud** ✅ |
| shared buffer | **silently degrades to per-request** | fails **open and quiet** ⚠️ |

**The loud one is genuinely excellent and should be the model.** A stale config
cannot silently switch off the headline capability on that axis: the server
declines to start and tells you exactly what it wanted. **That is the same
three-state discipline this dashboard argues for, already implemented in the
engine by someone who got there first.**

**The quiet one carries a second, subtler defect: the reason it reports is
mis-signed at birth.** Three independent preconditions are collapsed into a
single boolean, and the boolean keeps no record of which one failed:

```
crates/onnx-genai-engine/src/decode/metadata.rs::shared_buffer
    supports_present_binding
 && session.past_present_share_buffer_supported()   <- a fact about the HOST
 && metadata_max_context.is_some()                  <- a fact about the MODEL
```

Downstream, `crates/onnx-genai-engine/src/batched.rs` renders one sentence for
all three: *"continuous batching requires a STATIC-CACHE or shared-buffer
past/present model."* **That sentence blames the model file, and it is printed
even when the model was fine and the execution provider declined.** No test
pins the attribution — `past_present_share_buffer_supported` appears in **zero**
test files.

**So an operator whose execution provider declined is told to fix their model,
at warn level, on the wire, and on this page.** ***This is the `MISATTRIBUTED`
classification happening one floor below the layer that has a word for it — and
the honesty layer cannot detect it, because it can only be as honest as the
reason it is handed.*** **Every improvement we make to the disclosure surface
makes the wrong attribution more visible and more authoritative.**

**And the reason this one cannot be fixed by anything this project has learned
tonight:** the capability is negotiated with the execution provider at load
time, so **the same model bytes batch on one machine and refuse on another, and
the refusal names the bytes.** The one artefact the operator can inspect, diff
and re-download is the one the message accuses — **and it is identical on both
machines.** ***Unfalsifiable from the operator's side.***

> **We pinned SHAs, detached worktrees, hashed response bodies and stamped build
> IDs. Here is a capability that identical bytes do not determine. You cannot
> pin your way out of this one — it has to be read off the running process.**

**The fix is small and keeps the existing plumbing, which is good:** return a
reason rather than a boolean — `EpDeclined`, `SessionDeclined`, `NoMaxContext` —
and let the renderer show the one it actually got. **The decision is correct;
only its explanation is lossy.** ***A capability that turns on the host should be
reported as a fact about the host.***

---

## Known gap: a documented YAML config surface with zero production callers

**Asked by @732c7548 as a one-line question about one flag. Measured at
`78f335d7`, it is bigger than the flag.**

```
crates/onnx-genai-engine/src/config.rs

:612  /// Decode the `serving.memory.limits` YAML surface documented in §26.11.4.
:615  pub fn from_yaml(yaml: &str) -> Result<Self, EngineConfigError>
:628      config.allow_runtime_override = yaml_limits.allow_runtime_override;

  CALL SITES OF from_yaml:      729 · 749 · 759 · 767
  #[cfg(test)] BEGINS AT:       654          <- ALL FOUR ARE BELOW IT
  OUTSIDE config.rs, workspace-wide:  **0**

[POSITIVE CONTROL] EngineConfig::default has callers in five other crates
                   (bench/model.rs, compare.rs, multiturn.rs, profile_*.rs)
                   -> the instrument reaches the workspace; the zero is real.
[NEG CONTROL] from_zqx_9901 -> 0
```

**So `serving.memory.limits` is parsed, validated, tested and unreachable.**
No shipped binary can construct an `EngineConfig` from YAML — the server's only
config flag is `--models-config`, which builds a *different* type
(`ModelsConfig::from_file`).

**The consequence for the flag that was actually asked about:**
`allow_runtime_override` is not a hardcode, and it is not "assigned from the
YAML". **It is a `Default` impl value that no shipping code path can override,
because the sole mechanism that could override it has no production caller.**
The runtime-override refusal at `governor.rs:168`
(`if !self.allow_runtime_override`) is therefore **unconditional in practice**.

**Its error message offers two remedies and we quoted only the first when we
first wrote this section. The full text at `governor.rs:191-192` reads:**

> *"set `serving.memory.limits.allow_runtime_override: true` **or construct
> `EngineConfig` with `allow_runtime_override = true` before calling
> `set_vram_limit`**"*

**Stating it in full makes the gap sharper, not softer.** The first remedy is a
YAML key that no shipping binary reads. **The second is real — and it is a
*recompile*.** So an operator who hits this error at runtime is offered one
instruction that does nothing and one that requires them to be a developer with
a build environment. ***Neither is an operator remedy, and the message is
addressed to an operator.***

**We record the truncation deliberately:** quoting the first clause made the
message look simply wrong, when what it actually is, is *correct and addressed
to the wrong audience* — a worse defect and a harder one to notice.

> **An error message that names a remedy the product cannot perform is worse
> than a bare refusal. The bare refusal sends you to the maintainers; the
> helpful one sends you to a config file that will never be read.**

**This is the third confirmed member of the SPECIFIED-NOT-BUILT class in this
release** (with D298 and D300), and it is the first one where **the gap is
invisible from every angle a reviewer normally looks**: the parser exists, the
struct exists, the tests pass, the documentation cross-references a numbered
spec section, and the doc comment is accurate about what the function *does*.
***Everything is true except that nobody calls it.***

**Not a regression. Nothing in this PR changes it. Named so the next person
does not spend an afternoon writing a YAML file that has no reader.**

---

## Four additions closing this document out

**Ordered onto the gate by @c7a654ed and @12e42da8. Seven of their eight
required disclosures were already here; these are the four that were not.**

### 1. The test fixtures default to a retired state — measured, not relayed

```
dashboard/testing/fake-store.js:65   state: options.state ?? 'ok'
field-state.js                       OK: 'measured'
   ⬅ 'ok' IS THE RETIRED SPELLING. THE LIVE ENUM IS 'measured'.
test files importing fake-store:  **8**
   accessibility · model-path-disclosure · panel-bypass · panel-kit ·
   panels · scheduling · stylesheet · fake-store-contract
[POSITIVE CONTROL] test files importing panel-kit: 6   [NEG] fake-zqq-store: 0
```

**The `~112 fixture fields` figure reached me relayed, inside a checklist, and
I could not establish who took it — so under AC218 this document names no one
for it.** *(It also cited `fake-store.js:26`; the default I measured is at
`:65`.)* **I have not reproduced the count and I am not publishing it as mine — what I measured is the default, the retirement, and the
blast radius of eight files.** *The direction is not in doubt; the magnitude is
somebody else's number and it is labelled as such.*

**This is the highest-leverage instance of the fallback bias in the whole
release, because it is upstream of the tests rather than in the product:
*a fixture that defaults to a claim teaches every test written against it to
expect a claim.*** The bias reproduces itself in the suite faster than in the
code, and each new test makes it harder to remove.

### 2. Length is a claim we never guarded

At 320px the leaked model path rendered **four wrapped lines / 81px**; the model
name above it rendered **one line / 20px**. **Nobody chose to emphasise the
presenter's home directory — string length chose it.**

**And the CSS is correct.** `overflow-wrap: anywhere` refuses to hide data, which
is exactly right for honest values. **A rule adopted to protect honesty became a
megaphone the moment the content was something we should never have shown.**

> **Every guard in this repository asks whether a value is TRUE. Not one asks how
> much SPACE it takes.**

**It generalises past the row that was deleted: every unbounded-length field
inherits it — an error `reason`, a provenance warning, a stale-field
explanation.** ***So the most emphatic element on most of our panels is the
honesty layer apologising for not having a number.*** **This is *a layout is a
claim* arriving from the opposite direction, and it is visible only in a
browser** — which is why thirteen instruments missed it.

### 3. Two sentences about how this work actually got done

> ***The cheapest fix on the board was the only one with no owner — and that is
> not a coincidence. It is cheap because it is a deletion, and our tracking
> apparatus is built to observe additions.***

> ***A search for the defect never shows you its neighbour — and the neighbour
> was the fix.***

**The second fired four times, most tellingly on a ruling rather than on a
grep.** **We were rescued by individuals, not by the system, and that stands as a
finding even though every defect got fixed.**

### 4. What this PR is pinned to, and what it is not

**This document quotes every number with the revision it was taken at, and it
quotes more than one, because more than one is true.** `744/114` and `710/109`
are **two quantities, not two readings** — different runners over different
file sets. **Neither is wrong and they must not be added, averaged or
reconciled.**

**The pin itself is not mine to set** and was still moving when this was written.
**Whatever revision is finally announced, the numbers in §2 must be re-measured
there with `./run-tests.sh` and no hand-written glob** — the canonical runner is
the only one that reports its own tree, its own dirtiness, and refuses to score
a total containing uncommitted files.

> **A number in this document without a revision beside it is a defect in this
> document. If you find one, it is mine.**

---

## Known gap: this repository cannot tell you who did any of this

**Measured, not recalled. Over the last 200 commits:**

```
distinct commit AUTHORS      **1**       distinct COMMITTERS  **1**
commit messages naming a contributor id   65 / 200
```

Fourteen contributors produced this branch. **The permanent record shows one
person.** Author-based attribution is not degraded here — it is *absent*, and
every credit and blame assigned during the work was settled from conversation
and memory rather than from the tree.

**The obvious workaround is worse than nothing, and we measured that too.**
Searching commit *messages* for a contributor's id looks like an authorship
query and is a **citation** query:

```
--grep <my own id>          -> 19 commits.  Touching my files: **1**
                               The other 18 are other people's files.
--grep <a reviewer's id>    -> **34 commits**
                               THAT REVIEWER WROTE **ZERO** COMMITS ALL SESSION.
[NEG CONTROL] a nonsense token -> 0
```

> **The contributor who wrote nothing scores highest, because everyone thanked
> them. Citation and authorship here are not merely different sets — they are
> close to inverted, and the inversion rewards exactly the reviewing work that
> leaves no commits behind.**

**So the only recoverable attribution in this repository is file ownership**:
resolve a SHA, list its files, ask whose file that is. **That makes one-writer-
per-path a correctness property rather than a tidiness preference** — it is the
sole mechanism by which any claim about who did what can be checked at all.

**Why this belongs in a PR description and not in a retro:** three separate
contributors were credited in writing tonight with work they did not do, and
each correction required someone to run `git show --name-only` against their own
reputation. **Two of those were caught only because the person being praised
declined the praise.** *A misattribution that hands you credit produces no
complaint from the receiver and no alarm from the giver — every incentive in the
loop lets it stand,* which is why it is the one defect class here that survived
the entire session while ten others were closed.

**The remedy we can offer the next reader is small and mechanical:** an
attribution is a claim, so it carries the SHA and pathspec that establish it, or
it names no one. **This document follows that rule — including in one place
where following it meant deleting a name we could not verify.**

---

## Known gap: the specification says where assets come from and never what must not leave

**Found by a reviewer auditing our specification rather than our code, and
confirmed here with both controls firing.** In `demo-spec.md`, at HEAD:

```
dotfile              0        SERVABLE_EXTENSIONS   0
percent-encoding     0        .env                  0
allowlist            0        "must not be served"  0
the three static-asset security findings, by id   **0 · 0 · 0**
[POSITIVE CONTROL] "REQUIREMENT" 7 · "refuse" 17 · a known id resolves 1
[NEGATIVE CONTROL] a nonsense id 0
```

**The one asset requirement we did write is about freshness:**

> *"ASSETS ARE READ FROM DISK AT REQUEST TIME AND NEVER BAKED INTO THE BINARY."*

**That is a requirement about where bytes come from. There is no requirement
anywhere in the specification about which bytes must never leave.** Every
static-asset boundary this project has — the extension allowlist, the dotfile
rule, the traversal defence — **was invented during implementation and review,
and none of it was ever asked for.**

> **The acceptance criteria specify the happy path of asset serving in detail
> and its boundary not at all. Everything that protects the boundary today
> exists because an engineer thought of it, not because anyone required it.**

**This is the most consequential thing on this list, and it is a product defect
rather than an engineering one — it is mine.** The consequences were not
hypothetical: at one point the server returned the contents of a dotfile with an
allowlisted extension, and the three refusals that made it *look* safe were
refusals by coincidence — the file names simply happened to end in extensions
nobody had allowlisted. **A specification that never states a boundary cannot
have that boundary reviewed, tested, or regression-guarded, and the absence
looks exactly like agreement.**

**What we would want from the next revision, stated as the requirement we should
have written:** *the demo asset server serves an enumerated set and refuses
everything else, the refusal is by rule rather than by coincidence, and the rule
is stated in terms a reviewer can check without reading the implementation.*

**One correction to the report that found this, offered because it is the same
class as the finding:** the reviewer also measured 20 occurrences of the
model-path symbol in the specification and read it as the specification carrying
the defect. **Only 4 of those 20 are code; the rest are the record of the fix —
closure notices, controls, and an explicit instruction not to reinstate the
field.** *A document that records a defect's death contains that defect's name,
and a token census cannot tell an epitaph from a body.* The reviewer's aim was
right and their instance was not — which is the cheapest kind of error to fix
and, on this branch, the most common.
