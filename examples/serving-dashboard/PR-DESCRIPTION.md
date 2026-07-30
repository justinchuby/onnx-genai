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

## Measured numbers, with their conditions

**Aggregate decode throughput is roughly 2.5× single-request decode**
— 95 % CI [2.35, 2.59] — **while per-stream throughput falls to about 0.62× of
solo** (~20.7 tok/s).

**Both halves ship together, here and everywhere else.** Batching does not make
any single request faster; it trades per-stream latency for total throughput.
A tradeoff presented as a pure win is a lie told with true numbers, and the
half an engineer is most likely to need is the second one.

Everything behind those numbers is in
[`examples/serving-dashboard/perf-baseline.md`](examples/serving-dashboard/perf-baseline.md):
raw per-run samples, hardware, load average, the exact command, and the binary
and model SHA-256s.

| | |
|---|---|
| single request, decode | n=15, mean 33.577 tok/s, CV 1.98 % |
| aggregate, 4 concurrent | n=4 rounds, mean 82.847 tok/s, CV 2.93 % |
| **per-stream, 4 concurrent** | **~20.7 tok/s — 0.62× of solo** |
| ratio (aggregate ÷ single) | **2.467**, 95 % CI **[2.35, 2.59]** |
| capture | both arms inside one 20-minute window, one binary, one model, clean tree |

**Read the conditions before quoting the number.** This is an indicative figure
from a loaded developer machine, not a benchmark.

### Why it is `~2.5×` and not `2.46×`

We measured a **9.8 % noise floor** on this machine: a byte-identical binary
swung that much under load. A figure printed as `2.46×` asserts it is known to
±0.005, and the data supports ±0.12 — **the value is sound, the precision was
fabricated.** The third significant figure was division residue, so we dropped
it and published the interval instead.

### Why the ratio survives a 9.8 % noise floor

Because the floor and the claim measure different things. The floor is the drift
of an **absolute** number **across time** — the two observations were 75 minutes
apart under changing load. Both arms of this ratio were captured inside a single
20-minute window against one binary and one model, and **load that moves both
arms together largely cancels in a ratio.** The effect is **+147 %, roughly 15×
the floor.**

`check-perf-claims.test.js` recomputes the median, CV, ratio and confidence
interval from the raw samples on every run and fails if the README drifts from
them. The number in the prose is derived, never transcribed.

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
but the 2.46× baseline was measured against the original — swapping it would
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

**136 Rust tests** in the server crate; **59 documentation-drift checks** plus
the dashboard's own suites in `examples/serving-dashboard/`.

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
- **`demo-spec.md` is committed as a snapshot** at 159 acceptance criteria. Its
  authoritative count lives in the file's own header and is machine-generated;
  do not quote a total from anywhere else.
- **Prefix caching is demonstrated, not proven beneficial.** See above.
