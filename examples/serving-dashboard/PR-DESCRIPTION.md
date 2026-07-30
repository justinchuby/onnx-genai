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
- **`demo-spec.md` is committed as a snapshot** at 159 acceptance criteria. Its
  authoritative count lives in the file's own header and is machine-generated;
  do not quote a total from anywhere else.
- **Prefix caching is demonstrated, not proven beneficial.** See above.
