# Architecture & Security Review — `feat/genai-demo-dashboard`

MEASURED-AT: 1f8d3f94

**What that header does and does not promise.** It is the SHA of my most recent
measurement, so the freshness guard can prove this document describes a tree this
branch actually passed through. It is **not** a claim that every section was
measured there. This document spans six hours; §7 is hours older than §22. A
single header on a long document is a **lower bound on staleness**, and the
oldest section is the real exposure. Each section names its own SHA — those are
the load-bearing dates. I adopt the header because being the holdout costs the
crew more than the imprecision costs me, and I state the imprecision because a
document-wide assertion that outruns its evidence is the defect this branch
exists to catch.

**Reviewer:** Critical Reviewer (agent `f6527cc9`) — architecture, security, structural
design, performance, failure modes.
**Not in scope for this document:** implementation correctness (Code Reviewer) and
naming/organisation/documentation (Readability Reviewer). Where those reviewers own a
finding, this document says so and does not restate it.

> **Which tree this document is about.** Every `crates/…` and `examples/…` reference here
> means the **`onnx-genai-demo`** checkout on **`feat/genai-demo-dashboard`**. A sibling
> checkout at `onnx-genai` contains files at *identical paths* — `routes/admin.rs`,
> `driver.rs`, `cli.rs` all resolve in both — and it disagrees with this one. Per
> @086345a5's measurement, a fully-qualified path does **not** disambiguate here, so
> citations below are **symbol-anchored and tree-scoped by this paragraph**, which is the
> only form that cannot rot: a line number rots when code moves, and a per-citation
> repository prefix rots the same way.

**Verdict — and it differs by tree, so both are stated rather than one being implied:**

- **At `review-0` (`6ecd9183`), the pinned review artifact: APPROVE WITH ONE BLOCKER — C5.**
  `may_disclose_model_paths()` is present there and keys on the server's bind address, and
  `run-demo.sh` binds loopback by default, so the disclosure branch is the one every demo
  run takes. See §7.7.
- **At `HEAD`: APPROVE WITH COMMENTS. Blocking set zero.** C5 was closed by `2da3e851`,
  which is **not** an ancestor of the tag.

**I reported C5 struck against the tag and it is live there — a false green, corrected in
§7.7.** My recommendation to the Lead is to re-cut the tag above `2da3e851`: the fix is
real and better than anything three reviewers proposed, and a tag that excludes it makes
this review describe a tree nobody will run.

This document opened at
REQUEST CHANGES with three blocking findings; two were fixed while it sat unedited, and
the third was downgraded on evidence. **The stale verdict was live in this file for
roughly ninety minutes, and a stale red is not a safe error** — it either holds a ship
that is ready or burns a reviewer's hour chasing a corpse. Status, re-derived at the
SHA in each row rather than recalled:

| # | Was | Now | Closed by | How I know |
|---|---|---|---|---|
| **C2** stalled-server hang | BLOCKING | ✅ **CLOSED** | `6ecd9183` | ancestor-of-HEAD; re-executed at `c1323e7f`: blackhole socket → typed `RequestTimeoutError` @ 2004 ms, normal-server control → HTTP 200 @ 14 ms — see §1 |
| **C1** `parseOrigin` host validation | BLOCKING | ✅ **CLOSED** | `023db167` + `be3ab37c` | ancestors-of-HEAD; predicate re-run — see §2 |
| **C11** router re-fabricates zeros | BLOCKING | 🟡 **real, not blocking** | — | shipping, but zero router processes ran tonight — see §3 |
| **P1** model-path disclosure | *(not filed)* | 🔴 **LIVE at `review-0`** · 🟡 caption-only at HEAD | `2da3e851` *(post-tag)* | at the tag the Rust still discloses on a loopback bind — see §7.7. The deletion is real but is **not** in the artifact — see §5 |

**Blocking set at `review-0` is one: C5.** At HEAD it is zero. The strongest single change
on this branch landed while this document sat stale: the server's path-disclosure
*conditional* was not fixed, it was **deleted**, and a source-level test now forbids its
return. That is the design principle this whole review argued for — **make the wrong
state unconstructible rather than guard it** — executed more thoroughly than I proposed.
§7.2 below retracts C5 as a blocker and notes a residual (a proxy or port-forward makes
the bind address loopback while the peer is remote). **The author did not argue about how
likely that residual was; they removed the axis it lived on.** My retraction and their
deletion are both right, and that is the honest reading — not "I was correct all along."

> **How to read this document.** It is a *consolidated current state*, not a log. My
> working notes were append-only and superseded their own severities several times; a
> reviewer reading them raw would encounter a retracted finding written in the present
> tense and act on it. Retractions below are marked as retractions and kept, because a
> withdrawn finding that silently disappears is indistinguishable from one that was
> never made.

---

## 0. What I actually did, so you can discount it appropriately

- Diff scope: merge-base `f55e459b`, 119 files, +35,892 / −143.
- Prioritised by trust boundary: Rust server → shell scripts → dashboard JS.
- `cargo check`: clean.
- XSS-sink audit across all 24 non-test dashboard JS modules: **zero sinks**
  (`innerHTML`, `outerHTML`, `insertAdjacentHTML`, `document.write`, `eval`,
  `Function`, `srcdoc`, `javascript:`). The dashboard builds DOM through
  `textContent` and `createElement` throughout. This is a genuinely good result and
  it is not an accident of scale — it holds across every module.
- Findings C1, C2, C10, C11 were **executed against the real modules**, not inferred
  from reading. C3, C4 are read-only and are labelled as such.

**Instrument caveat, stated because it invalidated several of my own earlier claims:**
there are five checkouts of this repository on this machine and the demo exists in
exactly one. A path-relative search in any other returns a clean, reproducible,
**wrong** negative. Every observation in this document was taken in worktree
`onnx-genai-demo` on branch `feat/genai-demo-dashboard`.

---

## 1. ✅ CLOSED (`6ecd9183`) — C2: a server that stalls *after* connecting freezes the dashboard permanently, and the dashboard keeps reporting `connected`

**Severity: blocking. Highest-priority finding in this review.** Measured, with
controls, and with the fix validated.

**Mechanism.** `runPollCycle`'s `finally` re-arms the timer correctly — this is *not* a
lost-timer bug, and the polling loop is well built. `fetchAllSources` awaits a
`Promise.all` over `fetchSource`, and `fetchSource` awaits `fetchImpl` **with no
`signal`**. A never-settling promise means `finally` never runs, `pollInFlight` stays
`true`, and the timer is never re-armed.

The store's failure model — `transportError`, the unreachable state, reconnect
backoff, endpoint suppression — is **well designed and unreachable by construction**
for this case, because every one of those paths is entered from a *settled* rejection.
**The failure model assumes every fetch settles. Nothing enforces that assumption.**

**Measured, by store publishes over ~14 s (fetch counts are a proxy and misled me
badly — see §7.1):**

| injected condition | publishes | connection state | verdict |
|---|---|---|---|
| healthy 200 | 56 | `connected` | alive |
| HTTP 500 | 57 | `connected` | **alive — an earlier alarm of mine, withdrawn** |
| accepts, never replies | **1** | `connecting` | **dead, permanently** |

**The case that matters — same server, healthy first, then stalling:**

```
healthy phase : publishes=17  conn=connected
after stall   : publishes=17   (+0 over 15 s)   conn=connected
```

**The dashboard does not merely stop updating. It continues to display the last values
it received, under a `connected` indicator, indefinitely.** For a project whose thesis
is "never display a number you cannot stand behind," this is the most consequential
failure mode in the tree: every number on screen is stale, and the one widget whose job
is to tell you that says everything is fine.

**Fix, validated by simulation — an `AbortSignal.timeout` deadline in `fetchSource`,
nothing else changed:**

```
without deadline: +0 publishes,  conn stays "connected"
with    deadline: +4 publishes,  conn -> "unreachable"
```

**The fix adds no new failure path** — an abort *is* a transport rejection, which the
store already handles well. It belongs in `fetchSource`, the single chokepoint every
endpoint flows through, **not at the call sites**: a deadline that each caller must
remember to pass is one careless caller away from reopening exactly this hole.

---

## 2. ✅ CLOSED (`023db167`, `be3ab37c`) — C1: `parseOrigin` validates the scheme but not the host

**Severity: blocking.** Executed.

`parseOrigin` in `scenario-origins.js` checks `url.protocol` and nothing else. A query
parameter therefore redirects the dashboard's polling at an arbitrary host:

```
resolveOrigins({href: '...?dynamic-origin=http://127.0.0.1:64017', selfClasses: ['scatter']})
  -> {"scatter": "http://127.0.0.1:8123", "dynamic": "http://127.0.0.1:64017"}
```

That origin becomes `baseUrl` and is interpolated into the un-deadlined `fetchImpl`.
**A link is the entire attack** — no server compromise required. The dashboard then
renders attacker-supplied telemetry with full "measured by the server" framing.

**Fix:** compare `url.hostname` against the page's own hostname. **Hostname, not
origin** — the demo is legitimately one host on two ports, so an origin comparison
would break the scenario switch, which is the feature this parameter exists to serve.
`resolveOrigins` already receives the page `href`, so the input needed for the check is
in hand at the call site; no plumbing is required.

**Note on ranking:** I carried C1 as my most severe finding for most of the review
*because it has an attacker in it*. That was a bias worth naming. **C2 outranks it:**
C2 requires no adversary and fires during an ordinary production incident with our own
server.

---

## 3. 🟡 REAL, NOT BLOCKING — C11: the router re-fabricates the zeros the server deliberately omits, and routes live traffic on them

**Severity: blocking, if the router is in this PR's scope** — the demo itself runs no
router, so scope is the Lead's call. The *contract defect* is real either way.

This branch taught the server to be honest per-field: `/v1/status` **omits** what it
cannot measure via `skip_serializing_if`, and carries an `unavailable` map giving a code
and a reason for each omission. That is excellent design.

**The consumer never learned.** The router's `NodeStatus` wire mirror carries **14
`#[serde(default)]` fields**, byte-identical to its state at merge-base — the branch
changed the producer and never the consumer. A search for `unavailable` across the
**entire router crate returns zero** (positive control: `kv_usage` appears across five
files, so the search reaches this class of symbol).

**The fabricated zero is load-bearing.** It flows into `accepts_affinity`,
`RoutingPolicy::LeastKvUsage` — which selects the *lowest* `kv_usage` — the weighted
policy, and two Prometheus gauges.

> **A node that loses telemetry is scored as a node with an empty KV cache, and
> therefore becomes the most attractive routing target on the fleet. Failure produces
> attraction rather than avoidance.**

**Two green tests certify the opposite halves of the same contradiction:** the server's
`status_omits_kv_fields_rather_than_reporting_an_empty_pool`, and the router's
`deserializes_minimal_status_with_defaults`, which asserts `kv_usage == 0.0`. Each is
correct about its own side. Nothing tests the seam.

**Recommended fix — narrow, in `apply_status` (`crates/onnx-genai-router/src/node.rs`), not in the type.** The wide
fix (making 14 fields `Option`) touches ~8 scoring sites, 2 gauges, ~15 test
constructors and the `api.rs` re-exports. Instead: **`apply_status` should treat an
omitted load-bearing field as a missed poll**, feeding the existing
`consecutive_misses` machinery rather than writing `0.0` into `NodeState`.

**The sharpest framing, and the reason this is structural rather than a bug:**
`last_poll: Option<Instant>` already answers *"did we hear from this node?"* — that is
**poll granularity**. What is now needed is *"did it make a claim about **this
field**?"* — **field granularity**. **The producer moved to per-field honesty; the
consumer's only vocabulary for absence is per-node.**

---

## 4. The structural cause, which is one cause and not three

The same defect appears at all three consumers of the server's output:

| the server does the honest thing | consumer | outcome |
|---|---|---|
| **omits** what it cannot measure | **router crate** | `serde(default)` → `0.0`; `unavailable` never read; a quiet node attracts *more* traffic |
| publishes `batch_capacity`, and **warns in writing** against `max_batch` | **JS dashboard** | binds `scheduler.max_batch`; a *correct* fallback in `renderOccupancy` hides the miss |
| — | **test fixtures** | supply what the wire never sends, certifying both consumers as working |

The `batch_capacity` doc comment in the server's `routes/mod.rs` predicted the dashboard
defect **in advance and in writing**:

> *"Deliberately NOT named `max_batch`. It is `min(max_batch, max_queue_depth)` —
> because admission is often the tighter constraint, and `max_batch` alone overstates
> the ceiling. **A client that received this as `max_batch` would re-derive the wrong
> quantity from an honest one.**"*

The client binds `scheduler.max_batch` anyway, four fixtures supply the value the wire
never sends, and the suite is green.

> **One cause: the server's output contract is not an artefact anywhere. Every consumer
> hand-maintains a private model of it, and every test validates against the model
> rather than the wire.**

Two consequences worth carrying past this PR:

1. **A recorded capture is a fixture too.** It has better provenance the moment it is
   taken and rots identically afterwards. The useful distinction is not
   recording-versus-fixture — it is **whether anything notices when it expires**.
2. **A unit suite structurally cannot validate a cross-process contract.** It lives
   entirely on one side of the boundary, so every instrument it holds is a *model* of
   the other side. This is why nine JS suites import the field-state vocabulary, all
   pass, and none can see `routes/mod.rs`. **The end-to-end gate item is not the last
   checkbox on the list; it is the only instrument in the project that can see this
   defect class at all.**

---

## 5. Should-fix and lower

| # | Finding | Severity | Evidence |
|---|---|---|---|
| **C3** | KV block mirror caps at 4096 while true `hot_capacity` is ~14,612 — the visualisation silently under-represents the pool | should-fix | read |
| **C10** | `--demo-assets-dir` validates *existence*, not *identity* — a wrong-but-real directory is accepted silently, yielding a fully working API and a 404 on the demo path with nothing in the log naming a directory | should-fix | executed |
| **C9** | Poll fan-out isolates failure but not latency: `Promise.all` means the slowest endpoint gates the cycle | **downgraded** — root cause fixed by `bd2197a4`; `/metrics` worst case now 71.3 ms. **The server fix removes today's instance; C2 removes the class** | measured |
| **C4** | `batch_in_flight` ÷ `batch_capacity` scope mismatch | low, latent | read |
| **C7/C8** | `ServeDir` publishes the whole assets directory; no CSP header | minor | read |
| **C5** | `may_disclose_model_paths()` keys on **bind address rather than peer** — and the demo binds `127.0.0.1` by default, so the disclosure branch is the **default path for every demo run**, not an edge case | 🔴 **LIVE AT `review-0`** · closed at HEAD by `2da3e851` (after the tag) — see §7.2 and §7.7 | executed |
| **C6** | `0.0`-on-zero-capacity | **RETRACTED — false positive** | executed |
| **P1** | **Model-path disclosure — server half CLOSED by deletion, client half is now a caption defect, not a leak.** The server no longer has a disclosure switch at all: `model_path_for_display()` in `routes/admin.rs` takes one argument and returns `file_name()` unconditionally, and `crates/onnx-genai-server/src/tests.rs` `no_configuration_can_re_enable_full_path_disclosure` asserts at *source* level that neither `may_disclose_model_paths` nor `bind_addr` reappears in `crates/onnx-genai-server/src/state.rs`, `crates/onnx-genai-server/src/routes/admin.rs` or `crates/onnx-genai-server/src/cli.rs`. **No absolute path reaches the wire in any configuration.** What survives is that `ui/model-card.js` still labels the value `Directory` and `dashboard/system.js` labels it `model directory`, while the value is now a *basename* — @376a0297 predicted this exact caption defect before it landed | 🟡 **caption, not disclosure** — severity collapsed by the server fix | executed |
| **C12** | ~~`fetchWithDeadline` is the only network path by discipline, not by construction; nothing asserts it~~ | **RETRACTED — false when filed. See §7.6** | executed |
| **C15** | `fetchWithDeadline` **silently discards a caller-supplied `signal`** — `{ ...init, signal: controller.signal }` spreads the caller's key and then overwrites it. The docstring promises *"everything else is passed through to the underlying fetch untouched"*; that promise is false for the one key that controls cancellation. Executed at `review-0`: caller's `abort()` leaves the request **PENDING**. 🟡 latent — zero shipped callers pass a signal today | 🟡 NEW, latent, structural — see §8.1 | executed |

**On C10 and five checkouts:** this deserves more weight than its severity suggests.
With five checkouts of this repository on one machine, launching from the wrong one
produces a working API and a broken demo path, and **no log line ever says the word
"directory."** That is the most likely way this demo fails in front of an audience.

---

## 6. What I checked and found healthy

Recorded deliberately, because this crew has repeatedly issued edits against correct
code, and "I looked and it was fine" is information a reviewer needs.

- **`/v1/resources` emits three explicit `null`s.** Not defects. The router does not
  fetch this endpoint and the JS handles `null` correctly. The "omission, not null"
  rule was broadcast without its scope and does not reach here.
- **`status_never_emits_null_for_an_unmeasurable_field`** is correctly scoped.
- **Server↔client state vocabulary.** The server emits `not-applicable` and
  `unavailable`; both are members of the client's field-state set. **The contract
  holds.**
- **`healthy: bool` defaulting to `true`** is the same shape as C11, but
  `consecutive_misses` already covers it.
- **No XSS sinks in any of 24 non-test modules.**

---

## 7. My own errors, kept in the record

A review is a measurement, and an unaudited measurement is exactly what this branch is
about.

**7.1 — I measured a proxy and nearly retracted a true finding.** My standing C2
evidence was a *count of `fetchImpl` calls*. A control I had not run showed a healthy
404-returning server produces an identical signature. Widening the probe appeared to
reveal an inverted recovery taxonomy — 404/500/stall frozen, refused/malformed
recovering. **Three of those five readings were wrong:** endpoint suppression
deliberately replays a known HTTP failure for 10 s and my probe window was 6 s. C2 was
re-proven on **publishes**, the observable that actually corresponds to the user-visible
claim, and the 404/500 alarm was withdrawn.

**7.2 — C5 retracted; the code was better than my finding.**
`may_disclose_model_paths()` returns `bind_addr.is_loopback()`. **A loopback-bound
socket cannot be reached from another host, so the bind address *entails* peer locality
— it is not a proxy that can fail in the dangerous direction.** Bound to `0.0.0.0` it
discloses nothing. Its doc comment already named the exact leak, the missing auth, and
the reasoning. Residual: a port-forward makes the bind address loopback while the peer
is remote — narrow, and out of scope for a local demo.

**What survives from C5 is a different channel, and it is LOW.** The model path renders
in visible page text by design on loopback. But panel screenshots go into the README and
the PR, and a screenshot crops the address bar. **The guard reasons correctly about who
can reach the socket and structurally cannot reason about where the rendered output
travels afterwards.** Since the leaked username is already the public account name, the
marginal disclosure is a directory layout: **low, not a blocker.** Cheap construction-side
fix if anyone is in the file: render the model **basename** with the full path as a
`title` attribute.

**7.2a — UPDATE, and it corrects a prediction I made loudly.** The gate is gone. I
warned that deleting the Rust conditional before the client render would "delete a
working control," and I was wrong about the outcome while right about the class. The
author did not delete the *control*, they deleted the *capability*: `model_path_for_display()`
lost its boolean parameter and returns `file_name()` unconditionally. **A conditional
that can only ever be wrong in the disclosing direction was replaced by no conditional
at all.** My proposed `title`-attribute fix above is now actively bad — it would
reintroduce the full path into the DOM that the Rust deletion just removed from the
wire. **Struck.** This is the second time tonight a fix of mine was beaten by a deletion,
and both times the deletion won for the same reason: it removes the axis instead of
choosing a value on it.

**7.3 — My named recurring error: I verify a *definition* and infer its *use*.** Three
instances — reading `field-state.js` and never opening `format.js`; reading `NodeStatus`
and never opening `router.rs`; filing C5 without reading the comment directly above the
function. **One search for consumers before any prescription costs far less than the
retraction.**

**7.4 — A false count caught before broadcast.** I nearly reported the router's
`serde(default)` fields as growing "10 → 14." The 14 are byte-identical at merge-base;
my "10" counted numerics only. **Two different questions wearing one number.**

## 9. TRIPLE REVIEW — PASS 1, CRITICAL-REVIEWER ARM

Measured in a clean detached worktree at **`review-1` = `fca13038`**, resolved repo root
`/private/tmp/rv1-crit`, porcelain 0, `run-tests.sh` **raw unpiped exit 0 · 627 tests ·
94 suites · 0 fail · 47 files discovered · provenance 0 untracked / 0 missing**.
(The dispatch quoted 599/91 at this same immutable SHA; my count comes from the canonical
runner in a clean tree. The difference is the instrument, not the artifact.)

**P1-A — The provenance schema demands evidence to say "I don't know" and demands nothing
to say "this is real."** Executed against the module itself, all 10 `byOrigin` overrides:

```
OVERRIDES TO A SUPPRESSED STATE   5 of 5 carry a documented reason   ✅
OVERRIDES TO 'MEASURED'           5 of 5 carry ONLY the word itself  ⛔
  prefix_cache.hits · prefix_cache.lookups · prefix_cache.hit_rate ·
  metrics.prefix_cache_hits · metrics.prefix_cache_lookups   keys={classification}
```

A perfect 5/5 split is a mechanism, not an oversight. **This is the branch's own stated
failure shape — a default that degrades toward confidence — sitting in the data model of
the honesty layer itself**, not in the CSS where it was first found.

`resolveForOrigin()` makes it worse: `{ ...entry, ...override }` means an upgraded
override **inherits the parent's evidence string**, which was written about the
un-upgraded claim. Executed:

```
prefix_cache.hit_rate resolved for 'dynamic'
  classification: MEASURED
  reason:         (none)
  evidence:       "...emits a literal 0.0 when lookups == 0, so an undefined rate and
                   a genuine 0% are the same bytes."
```

**The entry carries its own refutation and nothing reconciles the two.** Fixing the one
entry now in flight leaves the mechanism, and the next override will do this again.
`scripts/check_provenance.py` — 310 lines, the main guard — contains **zero** occurrences
of `byOrigin`. It cannot see the mechanism at all.

**Fix: require `reason` on any override that raises confidence, and forbid an override
from inheriting `evidence` it does not restate.** Cheapest form: make the guard assert
`reason` present on every `byOrigin` entry whose classification is `MEASURED`, with a
non-zero floor on the override count so it cannot pass vacuously (denominator today: 10).

**P1-B — Cross-package contract drift: three fields claim MEASURED and their producer was
renamed out from under them.** `/v1/debug/kv` on the dynamic arm returns HTTP 200 and does
not contain any of the three keys the catalogue reads:

```
CATALOGUE READS (path:)        WIRE ACTUALLY SERVES
  prefix_cache_hits              generations_with_prefix_reuse
  prefix_cache_lookups           generations_completed
  prefix_cache_hit_rate          prefix_tokens_reused
telemetry-store.js:1228  readPath(body, entry.path)   -- no alias map anywhere in JS
```

`generations_with_prefix_reuse` appears in `crates/onnx-genai-server/src/routes/admin.rs`, `crates/onnx-genai-server/src/routes/mod.rs`, `crates/onnx-genai-server/src/tests.rs`,
`docs/ARCHITECTURE.md`, `README.md` and `demo-spec.md` — and in **zero JavaScript files**.
**Every artifact in the repository tracked the rename except the one whose entire job is
to state what is real.** The store degrades honestly (`"responded, but carried no value"`),
so the visible harm is limited — but the footer's "what's real, what's not" table is
built from `allFieldKeys()` and will list all three as MEASURED.

**Disclosed instrument failure, mine:** my first pass also flagged `throughput.observed` as
MEASURED-without-producer. **False positive — it is `derived: true` by design and carries
no `metric` key.** My predicate was wrong for derived entries. 1 false positive in 5; I
caught it by opening the hit instead of banking it.

**8.5 — C11 elevated: the router's fabricated zero is not a display defect, it is a
traffic-routing decision, and it fails toward *send everything here*.**

@73e77d95 identified their four failing dashboard tests as one invariant — *measured zero
vs no data vs not applicable* — and named it as the same defect I filed against the
router's `serde(default)`. They are right, and the router is the dangerous instance.
Measured at HEAD, unchanged:

```
node.rs:150-172   #[serde(default)]      kv_usage, queue_depth, active_sessions,
                                         kv_pages_*, tokens_per_second, batch_utilization
                  #[serde(default = "default_true")]   healthy

WHERE THOSE VALUES GO:
  node.rs:136   self.healthy && self.kv_usage < overload_threshold   -> ACCEPTING
  config.rs:110 "least_kv_usage"  -> picks the LOWEST kv_usage
  config.rs:122 weighted: affinity x0.5 + kv_usage x0.3 + queue_depth x0.2
```

**A node that omits these fields deserializes as: healthy, empty cache, no queue, no
sessions — the most attractive node in the fleet under all three routing policies
simultaneously.** It does not merely appear idle; it wins. A node that has silently
stopped reporting, or a version-drifted node that renamed a field, becomes the preferred
target and absorbs traffic *because* it stopped answering.

**This is the same defect as the dashboard's and the opposite in consequence.** A
fabricated zero on a panel misinforms a human who can be sceptical of it. A fabricated
zero here misdirects traffic automatically, with no human in the loop and no symptom
until the node is saturated.

**And the docstring names the intent, as `state.rs:87` did:** *"most numeric fields
default so the router degrades gracefully across versions."* Graceful degradation was
considered; the direction of the default was not. **This is the third place on this branch
where a safe fallback was available and the convenient value was chosen** — C5 (bind
rather than peer), `default_node_id()` (hostname rather than random), and here.

**Fix: `Option<T>`, and rank `None` last rather than first.** A node that cannot state its
load is not a node with no load. The comment already anticipates the shared
`onnx-genai-node-contract` crate; the type is the place to fix this, not the call sites.
**Not tonight — no router process ran this session, so this is latent in deployment and
live in the code.**

**8.4 — `default_node_id()` is hostname-first, and I had the consequence backwards.**
I warned that any restart would publish the operator's machine name, and @c0de4c2e has
folded that warning into the P0 restart remedy. **Measured, it does not happen here, and
the restart must not be delayed for it.**

```
node_id, four binary generations spanning 01:41 -> 04:11:
  :9123 node-d7c121cc605cce2c   :9451 node-198c22552fc05de0
  :8123 node-32e3e9904095ca1b   :8133 node-8a3670f4899ca2e7
hostname: JustindeMacBook-Pro   -- appears in none of them
```

**My source reading was right; my conclusion was wrong.** `state.rs:89` does prefer
`HOSTNAME`/`COMPUTERNAME`. But bash sets `HOSTNAME` as a *shell* variable and does not
export it, and macOS sets no `COMPUTERNAME`, so `var_os` returns `None` and the CSPRNG
branch is the only branch this machine takes. Verified from a child process: both
unexported.

**The risk is real, and it is inverted from where I put it.** This function exists for the
§34 *cluster* router — Linux and containers, where `HOSTNAME` *is* routinely exported
(Docker sets it to the container id). **So the code is safe exactly where it does not
matter, a demo laptop, and leaks by default exactly where it is designed to run.** That is
why the macOS reading is not reassurance: this is a latent default, not an absent one.

**Structurally it is the same defect as C5, one crate over: the safe value is the
fallback and the identifying value is preferred.** The comment at `:87` says *"Never
derived from a model"* — one disclosure vector was considered and the other was not.
The explicit path already exists (`ONNX_GENAI_NODE_ID`, `cli.rs:54`), so the hostname
tier is a convenience tier between explicit config and a safe random id, and **deleting
it costs nothing**: anyone wanting a stable id sets the variable. Deny by default.

**Not blocking, not tonight.** `env -u HOSTNAME -u COMPUTERNAME` is harmless but
unnecessary on this host; it should not gate the restart.

**8.3 — The fix is proven on the wire, and the discriminator is build time, not repository.**
Three running servers, differing in exactly one respect — when they were built, straddling
`2da3e851` (03:50:55):

```
:9123  demo repo, built 04:11:06  POST-fix   path='qwen2.5-0.5b-scatter-v2'      ✅ basename
:9451  demo repo, built 03:31:42  PRE-fix    path='/Users/<user>/…/qwen2.5-0.5b' ⛔ leaks
:8123  sibling repo, 01:41:44     control    path='/Users/<user>/…/scatter-v2'   ⛔ leaks
```

This is the first evidence tonight that the path fix works **as executed** rather than as
read. It also corrects the prescription now on the board: `:9451` **is** a demo-repo
binary and it leaks. *Rebuild from the demo repo* is not the remedy — **built after
`2da3e851`** is. Repository was a confounder that happened to correlate.

**A correct pair is already running: `:9123` and `:9124`, both basenames, both arms.** The
runtime item can be closed by pointing the demo at them; no new build is required.

**This also answers "these processes cannot name any commit."** They cannot, but for this
defect they do not need to — the `path` field is itself a behavioural fingerprint that
dates the binary against `2da3e851`. One request, no `/v1/version`, no `vergen`:

```
R=$(curl -s --max-time 3 :PORT/v1/models)
leak=$(printf %s "$R" | grep -c '/Users/')   # subject   MUST be 0
floor=$(printf %s "$R" | grep -c qwen)       # control   MUST be >0 — a dead port
                                             # also returns leak=0 and would "pass"
```

Demonstrated against four origins: `:9123`/`:9124` PASS, `:8133`/`:8134` FAIL, and the
floor separates both from a dead port. **Standing limit: this probe dates a binary against
exactly one commit. It is not a general version check, and it must not be cited as one.**

**8.2 — L10 is closed in code and open on the wire, and those need different actions.**
Measured against the live servers at 04:13, `/v1/models` (metadata only — no generation
counters moved):

```
:8123  path='/Users/<user>/…/onnx-genai/models/qwen2.5-0.5b-scatter-v2'   ABSOLUTE
:8124  path='/Users/<user>/…/onnx-genai/models/qwen2.5-0.5b'              ABSOLUTE
HEAD   model_path_for_display(path) -> file_name() unconditionally         BASENAME
```

Both processes started 01:41:44; `2da3e851` landed 03:50:55. **They are pre-fix binaries.**
So @bb2ee824's browser sighting of a home directory was real and remains real *against
these processes*, and it is **not** reproducible from HEAD's source.

**The consequence for the gate: the two-line client deletion does not close the live
leak.** It removes the render, which is correct on design grounds (the row the card's own
IA docstring does not account for) and correct as defence in depth — but the absolute path
is on the wire at `/v1/models` regardless of what the dashboard renders, reachable by
anyone who can reach the port. What closes the live leak is **restarting the arms onto a
binary built after `2da3e851`**.

**Stated limit:** a restart only helps if it *rebuilds*. If `run-demo.sh` re-execs a
cached binary the leak persists with no visible difference. The post-restart check is one
line and must assert on content, not on the restart succeeding:

```
curl -s :8123/v1/models | grep -c '/Users/'   -> MUST BE 0
curl -s :8123/v1/models | grep -c 'qwen'      -> MUST BE >0   (anti-vacuity floor)
```

**8.1 — C15: the deadline seam silently discards a caller's `signal`, and its own
guard cannot see that it does.** Filed at @c0de4c2e's request to read C2 for
*correctness* rather than *landedness*.

The `finally`/`clearTimeout` pairing is **correct on every path**, and its comment is the
most honest paragraph in the dashboard: it states that a leaked timer would be inert
because each call owns its controller, that what the line prevents is timer accumulation,
and that *no behavioural test can distinguish its presence from its absence, so the limit
is written here rather than asserted somewhere it would look stronger than it is.* That is
the standard, and it needs no change.

The defect is one line up. `fetchWithDeadline` destructures `fetchImpl` and `timeoutMs`
out of `options`, then calls `fetchImpl(input, { ...init, signal: controller.signal })`.
A caller-supplied `signal` lands in `init` and is **overwritten by position**. Executed
against `review-0` bytes:

```
caller signal === signal fetch received : false
caller signal was REPLACED             : true
after caller abort()                   : PENDING     <- cancellation silently dead
```

**Reachability, stated before severity: zero shipped callers pass a `signal` at
`review-0`.** The refactor removed `telemetry-store.js`'s own `AbortController` rather
than leaving it stranded, so this is **latent, not live** — no regression shipped. The
`@param` typedef lists `fetchImpl`, `timeoutMs`, `headers`, `cache` and does not invite
`signal`, which mitigates it further.

**What makes it worth filing anyway is that the prose and the guard both assert the
opposite.** The docstring says *"everything else is passed through to the underlying fetch
untouched."* That is a universal claim, and it is false for exactly one key — the key a
future caller reaches for when they need to cancel on unmount, navigation, or a scenario
switch. This is the module every network call must now go through, so the first caller who
wants cancellation gets silence rather than an error.

**And the test that should catch it carries a caption broader than its value.**
`request-deadline.test.js` — *the caller options reach the underlying fetch untouched* —
passes only `headers` and `cache`, and asserts `assert.ok(seen.signal, 'no signal was
attached')`. That assertion is **true whether the signal is the caller's or the module's**,
so it is structurally incapable of distinguishing the two states that matter. It is the
same shape @73e77d95 found in F2's guard: a check written to be robust to a value is blind
to that value being wrong. A reader auditing "does this module forward options correctly?"
finds a green test whose name answers yes and whose body asked two thirds of the question.

**Fix, in preference order.** (1) Compose: `signal: init.signal ?
AbortSignal.any([init.signal, controller.signal]) : controller.signal` — one line, and it
makes the caller's intent survive by construction. (2) Refuse: throw on a caller-supplied
`signal`. Either converts a silent wrong answer into a loud one. What should **not**
happen is a documentation fix — the interface currently works only because no caller
exercises the promise it makes, which is an unwritten rule enforced by nothing.

**7.7 — I reported C5 STRUCK against `review-0` and it is LIVE there. That is a false
green in my own lane, and the mechanism is new tonight.**

`may_disclose_model_paths()` exists at `review-0:crates/onnx-genai-server/src/state.rs`
and reads `self.bind_addr.is_some_and(|addr| addr.ip().is_loopback())`. It was deleted at
`2da3e851` — *"delete the path-disclosure conditional, because one branch was already
proven sufficient"* — which is **not an ancestor of the tag**. I measured at `HEAD`, saw
the deletion, and reported the finding struck **against an artifact that predates the
fix.**

**The mechanism is created by the pin itself and did not exist before it.** While the
review target was "HEAD, whatever it is now," measuring at HEAD was merely racy. Once the
target is pinned to `review-0` and the tree keeps moving, **measuring at HEAD is
systematically optimistic**: every defect fixed after the tag reads as struck. The tag was
introduced to stop drift, and it does — but only for readers who measure *at the tag*.
Habit measures at HEAD.

**This is the inverse of the error the rest of the crew hit.** Eleven stale-order reports
tonight came from measuring at a SHA *older* than the fix and reporting a defect still
live — a **stale red**, which costs rework. Mine came from measuring at a SHA *newer* than
the artifact and reporting a live defect closed — a **false green**, which costs the
finding. A stale red gets argued down by the next reader; a false green stops the next
reader looking. Same root cause, opposite sign, and the false green is the expensive one.

**The substance is also worse than I originally filed it, and my retraction was wrong on
its own terms.** I withdrew C5 because the residual needed a proxy or port-forward, which
I judged unlikely. But `run-demo.sh:29` sets `BIND_HOST="${BIND_HOST:-127.0.0.1}"`, so
`is_loopback()` is **true on every default demo run** and the disclosure branch is the
one that always executes. This is **allow-by-default**: the gate's safe branch is the one
the demo never takes. @bb2ee824's browser observation of a full home directory is not a
separate finding — **it is this defect's observable effect**, and the two were tracked
independently all night without being connected.

**7.6 — C12 was false when I filed it, and my exclusion pattern is why.** I claimed
nothing asserts that `fetchWithDeadline` is the only network path. `request-deadline.test.js`
asserts exactly that, in a test named *every fetch in shipped dashboard code carries a
deadline*, with a `fetchSites > 0` floor whose comment states my own rule back to me:
*a scan that matches no files and a tree with no defects are byte-identical from here.*
It excludes the helper by negative lookbehind rather than a name blocklist "that would
rot." **It is a better guard than the one I proposed as missing.**

**The mechanism generalises and is the most useful thing in this section.** Every census
I ran tonight carried `':!*test*'`. That is *correct* for the question "what do we ship" —
and I then used the same corpus to answer "what is *enforced*." **Enforcement lives
exactly in the files that exclusion removes.** The corpus that answers "what ships" is
the complement of the corpus that answers "what is guaranteed," so an exclusion that is
right for one question is maximally wrong for the other. This is §7.3's error — verify a
definition, infer its use — with the inference running the other way: I verified an
absence in a corpus chosen to exclude the thing whose absence I was claiming.

**7.5 — This document was itself the defect it describes.** For most of this review my
deliverable lived in an agent artifact directory that is **not inside any git
repository**. `git status` in the repo was clean throughout — byte-identical to the
output of work that committed successfully. **A failed commit leaves a dirty file you
can see. A file written outside the repository leaves nothing to notice.** Everything
above existed only in chat until this commit.

---

## 8. Credit where it is due

- **The absence vocabulary.** Five field states with distinct visual treatments, and a
  server that omits rather than zeroes. Most dashboards fabricate a `0` here. This one
  refuses to, at real cost in complexity, and the cost is worth paying.
- **`skip_serializing_if` plus the `unavailable` map carrying a code *and* a reason.**
  Machine-readable and human-readable in one structure, and the reason is what makes the
  omission actionable instead of merely honest.
- **The `batch_capacity` doc comment**, which reasoned its way to a defect prediction
  that later came true in another language. That comment is the best single artefact in
  the diff.
- **Zero XSS sinks across 24 modules**, achieved by construction rather than by review.
- **`runPollCycle`'s `finally`, the reconnect backoff, and endpoint suppression** are
  correctly built. C2 is not a criticism of them — it is that they all sit downstream of
  an assumption nothing enforces.
- **The panel registry's pinned count**, which forces a change in what the demo claims
  to be a deliberate act rather than a merge artefact.

## 10. Pass 1, re-measured after `MISATTRIBUTED` landed: the fix moved three fields and left the mechanism exactly where it was

Measured by **executing** the catalogue at HEAD `c97305d8`, walking every `byOrigin`
override rather than grepping for them. Denominator published; a zero here is a broken
instrument, not a clean tree.

```
total byOrigin overrides ......... 10   (RED at 0)
downgrades ........................ 8   with their own `reason`: 8
upgrades to MEASURED .............. 2   with their own `reason`: 0
```

Before @bb2ee824's fix the split was 5/5 with reasons and 5/5 without. It is now 8/8 and
0/2. **Three fields moved from the evidence-free column into the evidence-carrying one and
the split stayed perfect.** That is the strongest confirmation the asymmetry is structural
rather than clerical: a real fix, made by a careful author who was not thinking about my
finding, landed on exactly the side the schema makes expensive and did not disturb the
side it makes free. `scripts/check_provenance.py` still contains `byOrigin` zero times, so
nothing in the guard would have noticed either way.

### 10.1 The two survivors are not the same defect, and one of them vindicates the author who declined it

@bb2ee824 deliberately left the `lookups` pair alone, reasoning that it *honestly counts
completed generations and is already labelled that — a true number with a fixable name is a
relabel, not a reclassification.* Measured against four live servers spanning both
repositories and both build generations, that reasoning is **correct for one entry and
false for the other**, because the two read different endpoints:

```
metrics.prefix_cache_lookups  -> /metrics   onnx_genai_prefix_cache_lookups_total
    SERVED. 3 occurrences. control: 83 onnx_genai_ metrics present.   RESOLVES ✅

prefix_cache.lookups          -> /v1/debug/kv   path 'prefix_cache_lookups'
    NOT SERVED by any of :9123 :8123 :8124 :8133 :8134 — all four agree.
    the key is now `generations_completed`.                          DOES NOT RESOLVE ⛔
```

So the rename landed on the JSON debug endpoint and **not** on the Prometheus endpoint.
The same logical quantity is published by the same process under the old name in one
surface and a new name in the other. This is the part worth carrying past the demo: we have
been discussing this as *the dashboard is stale relative to the server*, and it is not.
**The server disagrees with itself, and the dashboard is faithfully following one half of
it.** A consumer cannot be "updated for the rename" because there is no single rename to
be updated for.

The floor mattered here. My first read of `:9124` returned zero bytes and zero occurrences
of the key — which is byte-identical to the finding I was looking for. The port was dead.
An absent key and an absent server produce the same number, and only the length check told
them apart.

## 11. An audit of my own citations, prompted by the Lead's phantom-`dashboard/` finding

I extracted every path token from this document and reconciled it against `git ls-files`.
**19 of 22 were bare basenames.** Most resolve uniquely, but three do not, and the
ambiguity is not theoretical:

```
tests.rs   -> 10 tracked candidates
state.rs   ->  4   (engine/decode, router, server, runtime-session/executor)
node.rs    ->  2   (onnx-genai-router, onnx-runtime-ir)
CONTROL: driver.rs, admin.rs, cli.rs, api.rs, router.rs -> exactly 1 each
```

Every ambiguous citation in this file is now fully qualified, each resolved **by symbol**
and each with a control that correctly returned 0 in the crate I did not choose —
`apply_status` is in the router's `node.rs` and not in `onnx-runtime-ir`'s;
`no_configuration_can_re_enable_full_path_disclosure` is in the server's `tests.rs` and
not the engine's.

The reason this is worse than untidiness is the Lead's: these basenames exist at
**identical paths in the sibling repository**, so a bare `state.rs` does not fail to
resolve for a reader — it resolves, in a tree where we do not leak. An unqualified
citation is not a citation with a missing prefix; it is a citation that will confidently
answer a question about the wrong artifact. My deliverable was 19/22 exposed to that and
I did not notice until someone else found it in their own output.

I repaired these by locating each symbol first and asserting the expected occurrence count
before substituting, because the mechanical version of this exact repair put *"X hardcodes
X"* into the README an hour ago. The substitution is the easy half; knowing which of ten
`tests.rs` you meant is the whole job.

### 11.1 The repair caught me one step short, twice, and both are worth recording

**First:** after qualifying three citations I re-counted the bare tokens expecting zero and
got *more than I started with*. §11 itself discusses `tests.rs` and `state.rs` by name, so
the section warning about bare basenames manufactures bare basenames. That is @c8d9a40e's
obituary pattern arriving inside the repair for it, and it means the ratchet I would have
been tempted to write — *count of bare basenames must not increase* — would go red on the
document that fixes the problem. **Separating the two required a line-number split against
the section boundary, i.e. exactly the coordinate-based reasoning we spent the night
proving unreliable.** The honest form of that guard scores citations, not occurrences, and
we do not currently have one.

**Second, and worse:** I resolved the last ambiguous `state.rs` by searching for
`may_disclose_model_paths` and `bind_addr` — and got **0 in the server's `state.rs` and 0
in the router's**. The symbol had been deleted by the C5 fix. A zero in both arms is not a
resolution; it is an instrument that cannot see. I had already written the qualified path
by then, and it happened to be right. I only learned it was right afterwards, from
`effective_batch_capacity` and `default_node_id` (2 and 2 in the server, 0 in both others).
**Note that `AppState` would not have discriminated either — 3 in the server, 5 in the
router.** The lesson is not *check your work*; it is that **the symbol most natural to
search for is the one the fix removed**, so resolving a citation by the defect it discusses
is guaranteed to fail exactly when the defect has been repaired.

## 12. C16 (NEW, MAJOR, non-blocking) — two modules own the same vocabulary and answer opposite things about the same input

@c8d9a40e found that `dashboard/testing/fake-store.js` mints `state: 'ok'` and that the
product refuses to render it. I verified it independently and **the cause is not the one
that reading the fixture suggests.** Executed at HEAD `e9bf0285`:

```
telemetry-field.js  FIELD_STATES values -> ["measured","pending","stale",
                                            "unavailable","not-applicable"]
     'ok' among them ................... false

dashboard/field-state.js normalise('ok') -> "measured"     ⬅ ACCEPTS IT
format.js       formatField({state:'ok'}) -> "—" + console.error  ⬅ REFUSES IT
```

`format.js:55` builds its gate as `KNOWN_STATES = new Set(Object.values(FIELD_STATES))` —
strictly the canonical values. `field-state.js:101-102` maintains a **separate** alias map
whose comment reads *"BOTH SPELLINGS OF THE MEASURED STATE ARE ACCEPTED, DELIBERATELY."*
Neither table derives from the other. **The same string is legal in one module and a
reportable defect in the other, and both behaviours are deliberate and commented.**

So the fixture author was not careless. They wrote `'ok'` against a module that documents
`'ok'` as supported, and it *is* supported there. The defect is that support is not a
property of the product — it is a property of which module you reach first. **A tolerance
that lives in one consumer and not in its sibling is worse than either strictness or
tolerance chosen consistently, because it makes the correct-looking fixture wrong only on
the path nobody asserts against.**

### 12.1 Blast radius, counted rather than inherited

```
stylesheet.test.js 51 · panels.test.js 36 · scheduling.test.js 16 · field-state.test.js 6
panel-kit.test.js 2 · accessibility.test.js 1 · model-path-disclosure.test.js 1
TOTAL 113 fixture fields   (RED at 0)   CONTROL: 8 files import fake-store
```

@c8d9a40e reported ~112 across 6 files; I get 113 across 7, the extra being the
`model-path-disclosure.test.js` they had just landed. Their bound was right.

**Every one of those 113 fields exercises the unknown-state branch, not the measured
branch.** The suite is green because panel tests overwhelmingly assert structure and class
names rather than rendered values. This is the largest vacuity on the branch: our most
common render path — a healthy number — is substantially less tested than the count
suggests. I state that as a bound, not a measurement; establishing the true figure means
flipping the fixture and reading what breaks, which is exactly what must not be done at
04:30 by someone who owns none of those six files.

### 12.2 The ruling, because the obvious fix is the wrong one

There are two ways to reconcile the vocabularies and **only one is correct**:

- ⛔ **Do not widen `format.js`.** Its strictness is the thesis of this entire branch — a
  terminal branch that refuses to render an unrecognised state, with a comment saying *"a
  default branch that renders as fine is how AC6 dies quietly."* Adding `'ok'` to
  `KNOWN_STATES` would resolve the disagreement by deleting the only component that
  noticed it.
- ✅ **Delete the alias in `field-state.js`, then fix the fixture.** An alias map that
  accepts a retired spelling is what keeps the retired spelling alive; it is the mechanism
  by which a nine-times-retracted enum value is still executing. The fixture should not
  mint a raw string literal for an enum-valued field at all — it should import
  `FIELD_STATES.MEASURED`, at which point the two tables cannot disagree because there is
  only one.

**The general form, which is the third time this exact shape has appeared in my review:** a
value's meaning is decided by a permissive layer the author read and a strict layer they
did not. `panel-kit.js`'s `options.label ?? field?.label`, `resolveForOrigin`'s
`{...entry, ...override}`, and now `normalise('ok')` versus `KNOWN_STATES` are one defect
wearing three costumes — **two places that both get to decide, and no statement anywhere
about which one wins.**

I am filing this, not fixing it. It spans six test files I do not own, in a freeze, and
@c8d9a40e's reason for declining it is the correct one: some of those tests may have been
written honestly against the em-dash behaviour they were accidentally getting, and nobody
can know which without running it.

## 13. The `ONNX_GENAI_SHARED_KV_PRESENT_BINDING` escalation is not a demo blocker, and the real defect is a stale safety rationale

@c0de4c2e escalated an env var to P0, reasoning that on Apple Silicon the capability
allowlist denies fixed-capacity present binding, so continuous batching reduces to one
string in the operator's shell. I measured the chain in the sibling repo at
`f55e459b` (toplevel asserted, sibling deliberately) and **two of the load-bearing
premises are false at HEAD.**

**① The demo does not run Metal.** `run-demo.sh:234` and `:243` both launch with
`ONNX_GENAI_EP="${ONNX_GENAI_EP:-cpu}"`. CPU is on the allowlist —
`ep_compat.rs:98`, `EpCapabilities::host()` carries `FIXED_CAPACITY_PRESENT_BINDING`
outright.

**② Metal is on the allowlist too.** `ep_compat.rs:315` grants Metal
`FIXED_CAPACITY_PRESENT_BINDING`, with a comment saying the MLX plugin now implements the
fixed-capacity in-place-write GQA contract. So the env var changes nothing for either
provider the demo could plausibly use, and no runbook needs to pin it.

**This matters because the proposed remedy was to pin the variable in the launch command.**
Pinning it to `1` would not have restored a capability — it would have switched off the
allowlist for every provider in the session, which is the one thing the flag actually does.
A remedy derived from a false premise, aimed at production launch parameters, an hour
before a demo.

### 13.1 The genuine finding underneath it: two comments in one crate describe opposite states

```
mod.rs (the WHY block)   "The external Metal plugin's growing-shape GQA kernel …
                          crashed Metal E2E … Metal therefore declares NO
                          fixed-capacity present support by default, preserving
                          today's ZeroCopyRebind behavior."

ep_compat.rs:306-316     "The MLX plugin implements the fixed-capacity
                          in-place-write GQA contract, so Metal carries
                          FIXED_CAPACITY_PRESENT_BINDING."
```

Both are careful, both explain themselves, and **they state opposite capability facts.**
The code follows `ep_compat.rs`; `mod.rs` preserves the safety rationale from before the
plugin was fixed. Anyone reasoning about the crash class from `mod.rs` — which is where the
reasoning lives, and therefore where a reviewer goes — concludes Metal is protected by
default. It is not, deliberately.

This is the obituary pattern inverted. The usual case is a comment describing a defect that
is fixed, which makes a clean tree look broken. Here a comment describes a **protection**
that has been deliberately lifted, which makes a permissive configuration look guarded.
**The first failure wastes a reviewer's time; the second spends their trust.** Fixing it is
a comment edit in a crate I do not own and is not a demo-day action.

### 13.2 What I got right and wrong about my own claim here

I have said twice tonight that nobody has run `cargo test`, and @c7a654ed has said it
three times. That remains true and it is why every statement in this section is a read of
committed bytes and a read of the launcher, not an execution. **I cannot prove the
resolved capability at runtime; I can only prove what the allowlist declares and what the
launcher requests.** The honest scope is: the premise that Metal is denied is false in the
source, and the premise that the demo requests Metal is false in the launcher. Whether the
plugin actually honours the contract on this machine is unmeasured, by me and by everyone.

## 14. Auditing my own retractions, because @086345a5 proved a withdrawal can be the error

@086345a5 discovered that their R12 retraction was wrong and the original finding was
right: they re-measured after the fix landed, found the defect gone, and concluded *I was
mistaken* rather than *it was repaired*. The two are indistinguishable from the result
alone. Their conclusion is the sharpest methodological result of the session and it
indicts me directly: **I have filed four retractions tonight and ran a control on none of
them.** We instrumented findings thirteen ways and withdrawals zero.

I owed C12 an audit, so here it is. C12 claimed nothing asserts that `fetchWithDeadline`
is the only network path; I withdrew it as false-when-filed.

```
refuting test 'every fetch in shipped dashboard code carries a deadline'
    first appears ......................... 6ecd9183   03:41:04
my C12 filing ............................. b8d643ed   03:59:01

BY ANCESTRY, NOT TIMESTAMP (clocks can lie; the graph cannot):
  6ecd9183 is an ancestor of b8d643ed ........... YES
  CONTROL, must be false: reverse direction ..... no  ✅
  test string present in the file at b8d643ed ... 1
  CONTROL, absent string at same SHA ............ 0  ✅
```

**The retraction holds.** The guard existed eighteen minutes and one ancestor before I
claimed it was missing, so C12 was genuinely false when filed and withdrawing it was
correct.

The reason this is worth committing rather than just knowing is that it makes the audit
mean something. **An audit that always reverses the withdrawal is as useless as no audit —
it just relocates the bias.** @086345a5's R12 failed this check and mine passed it, which
is what a working instrument looks like: it can return either answer. My remaining three
withdrawals are unaudited and I am recording that as an open debt rather than implying the
one I checked was representative.

I also want to name the asymmetry @086345a5 found, because it is a claim about incentives
and not about git: this crew built large social reward for retracting your own findings,
correctly, and **a retraction is the one kind of claim that reward makes cheaper to get
wrong.** Calling a fixed defect *my mistake* deletes the evidence the fix was ever needed
and collects credit for candour in the same motion.


## 15. Pass 2 at `review-2` = `0bc86726` — path disclosure across all three layers

Measured in a clean detached worktree, porcelain 0, raw unpiped exit 0,
646 tests / 98 suites / 0 failures. Exact agreement with the Lead's number.

### 15.1 All three code layers are clean, and the fix is proven on the wire

| Layer | Predicate | Result |
|---|---|---|
| 1 server (Rust) | `path:` emission in `admin.rs` | `:63` -> `model_path_for_display()` -> `file_name()`, unconditional |
| 1 control | any other raw path in `routes/*.rs` | the only `to_string_lossy` in all routes is inside that fn |
| 2 catalogue | `server.model_path` in `NEVER_BIND` | 1 (banned); `projectServedModel()` defined -> 0 (deleted) |
| 3 render | non-test, non-comment bindings | 0 |
| 3 control | `server.model_id` bindings | 5 files (instrument reaches the tree) |

Proven by execution, not by reading. Two binaries, same model, same field,
same instant:

    demo worktree binary (built from the fixed tree)
        path = 'qwen2.5-0.5b-scatter-v2'                    floor 175, zero /Users/
    sibling binary, running on :8123
        path = '/Users/justinc/.../models/qwen2.5-0.5b-scatter-v2'

The predicate discriminates in both directions. A fix that only made the
leak stop appearing would be indistinguishable from an instrument I had broken.

### 15.2 CORRECTION to @bb2ee824's scope note

They wrote: "the wire field is unchanged, /v1/models still returns the
absolute path on loopback." That is TRUE OF THE RUNNING PROCESS and FALSE
OF THE SOURCE AT `review-2`. They measured the wire, which was correct, and
inferred the source, which was not. The Rust fix `e556b7f4` landed at 00:18:30.

### 15.3 THE FINDING — `run-demo.sh:207` is an existence check standing in for a freshness check

    if [[ ! -x "${SERVER_BIN}" ]]; then ... cargo build --release ... fi

`-x` tests that a file is executable. It cannot test WHICH TREE IT WAS BUILT
FROM. A binary compiled from a checkout that lacks the path fix satisfies this
condition exactly as well as a correct one, and the launcher then runs it and
prints nothing.

This is not hypothetical. Measured on this machine right now:

    sibling repo /Users/justinc/Documents/GitHub/onnx-genai
      branch justinchu/demo, HEAD f55e459b
      e556b7f4 is an ancestor of HEAD          -> NO
      fn model_path_for_display in that tree   -> 0
      CONTROL pub(crate) async fn models       -> 1   (file exists, grep resolves)

So the absence is real, not a bad pathspec. All four servers currently
listening were started 01:41 and 02:07 from that tree's binary and all four
leak the operator's absolute home directory.

Today the launcher happens to be safe: `REPO_ROOT` resolves to the demo
worktree, `CARGO_TARGET_DIR` is unset, and someone rebuilt that tree at
04:38. **That is luck, not a control.** The two binaries are different
inodes 480 bytes apart. Nothing in the launcher can tell them apart.

FIX, structural, and it is small: after resolving `SERVER_BIN`, do not branch
on existence. Either always `cargo build --release` (it is a no-op when fresh,
which is the point -- cargo already knows what the launcher is guessing at), or
have the server print its build SHA on `/v1/debug/*` and have `run-demo.sh`
refuse to launch when it does not match `git rev-parse HEAD`. The first is one
deleted `if`. **Let the tool that tracks source freshness answer the question
about source freshness, instead of re-deriving it from a filesystem mode bit.**

### 15.4 Severity, scoped honestly

The disclosure is LOOPBACK-ONLY and I verified it rather than assuming:
`127.0.0.1:8123 (LISTEN)`, and the host LAN address 192.168.1.195:8123
returns floor 0. It is not remotely reachable. The exposure is the projector
and anyone with local access -- which is exactly the demo threat model, so it
still matters, but it is not a network vulnerability and should not be
described as one.

### 15.5 Where the layering genuinely earns its keep

The client still fetches `/v1/models`, so the response body reaches the
browser on every poll. `NEVER_BIND` makes the field UNBINDABLE; it does not
make it UNFETCHED. Against a leaking server the path would still sit in the
network tab. That is the correct division -- a client cannot fix a server
disclosure -- and the three layers together mean **no single layer failing
puts the path on the projector.** Deleting `projectServedModel()` rather than
the row is what makes layer 2 hold: a ban stops a field being bound, removing
the projection stops it being addressable.

## 16. The prescribed pre-flight for the last P0 is vacuous — and its positive control passes

@c0de4c2e promoted the runtime P0 correctly and prescribed a three-part
restart predicate. Step 2 does not work, and I have both binaries on disk
to prove it:

    strings <binary> | grep -c may_disclose_model_paths     "MUST BE 0"

      FIXED   demo worktree binary (basenames on the wire)  -> 0
      LEAKING sibling binary (running on :8123 now)         -> 0

      POSITIVE CONTROL, strings did read both:
        demo    18455 strings      sibling 18412 strings

The check passes identically on the binary that leaks and the binary that
does not. `model_path_for_display` scores 0 in both as well: **private Rust
function names do not survive into a release binary's string table**, so
every symbol probe of this shape returns 0 regardless of the fix. I searched
for any discriminating token and found none; the set difference is Rust
string-table blob noise, not source symbols.

**THIS IS THE IMPORTANT PART, AND IT IS A NEW CLASS.** The prescription
already carried the positive control the Lead's four-part standard demands,
and the control PASSES, and the check is still worthless. A positive control
proves the instrument READ THE CORPUS. It does not prove the instrument CAN
SEE THE PROPERTY. Those are different questions and we have been treating
them as one. This is the Lead's doctrine correction 3 -- *a control must vary
the instrument, not just the subject* -- landing on the remedy for the only
remaining red.

### 16.1 The replacement, which I ran rather than proposed

Behaviour is the only discriminator, and it costs one curl:

    launch the binary, then
      curl -s 127.0.0.1:PORT/v1/models | grep -c '"path":"/'    MUST BE 0
      FLOOR FIRST: the response must be non-empty, or a dead port
      scores 0 and reads as a pass

Measured, both arms, same model, same instant:

      demo worktree binary   path = 'qwen2.5-0.5b-scatter-v2'   floor 175
      sibling binary :8123   path = '/Users/justinc/.../qwen2.5-0.5b-scatter-v2'

`strings` cannot tell these apart. The wire tells them apart in one field.
**Test the property you care about, not a symbol you hope implies it.**

### 16.2 Correction to @bb2ee824

They wrote that my "~1-line admin.rs fix has NOT landed". It landed at
`e556b7f4`, 00:18:30, and I executed a binary built from that tree and read
the basename off the wire. Their 4-of-4 leak measurement is correct and is a
measurement of processes started 01:41 and 02:07 from the OTHER checkout.
The fix is real, shipped, and not running. That distinction is the whole P0.

## 17. The Rust suite, executed — my longest-standing open debt, closed

I have said "cargo test has not run tonight" in every message since 02:00 and
carried it as an open limit. It has now run, by me.

    cd /Users/justinc/Documents/GitHub/onnx-genai-demo   HEAD 964cad4a
    porcelain WHOLE TREE 1  ·  porcelain crates/ 0
      ^ the corpus I executed is committed bytes even though the desk is dirty
    cargo test -p onnx-genai-server      RAW UNPIPED EXIT 0

    lib.rs                211 passed  0 failed  3 ignored
    main.rs                 0          0         0
    tests/demo_dashboard.rs 15          0         0
    tests/http.rs           28          0         1
    tests/vlm_image_bundle.rs 10        0         0
    TOTAL                 264 passed  0 failed  4 ignored

**THE LEAD'S RED IS GONE.** They reported 185 pass / 1 FAIL / 2 ignored and
said they were converting the unsatisfiable `vlm-executable` fixture to a
visible `#[ignore]` naming its gitignored prerequisite. That landed:
`tests.rs:1109` now reads *"requires a real vision encoder ... which
.gitignore (*.onnx) excludes from every clone."*

### 17.1 A denominator disagreement I am reporting rather than resolving

The Lead measured 188 total. I measure 268 (264 + 4). I am not claiming
their number was wrong and I am not picking mine because it is mine — that
is the exact move @e00032a4 refused when their instrument said 15 and the
Lead said 546. My total is auditable: six test binaries, each printed
above, summing to 264. **Whoever re-runs this should publish the per-binary
split, because a single total cannot show which binary went missing.**

### 17.2 Layer 1 is now proven BY EXECUTION, not by reading

In §15 I verified the server-side path fix by reading committed bytes and
probing the wire. The branch's own Rust test asserts it, and it ran:

    test tests::no_configuration_can_re_enable_full_path_disclosure ... ok

`tests.rs:4252`. That test forbids the *token* rather than sampling the
configuration space, which is the strongest form available: a sampled
config space only proves the settings you remembered to list.

### 17.3 The 4 ignored, by name, because a skip is invisible to a pass rate

    audio_endpoints_route_through_tiny_whisper_pipeline
      "synthetic Whisper-contract smoke test; run explicitly"
    sidecar_free_compatibility_package_builds_server_pipeline...
      "requires a real vision encoder ... .gitignore (*.onnx) excludes it"
    vision_request_routes_through_tiny_vlm_pipeline
      "requires gitignored models/tiny-vlm; run scripts/build_tiny_vlm.py first"
    qwen_real_model_tool_use_chain_end_to_end
      "requires gitignored models/qwen2.5-0.5b real model fixture"

All four name their missing prerequisite. That is the right convention and
it is followed consistently. **But four tests that never run are four claims
nobody checks, and three of them are the multimodal pipelines** -- the least
exercised surface on the branch. Not blocking; worth stating beside the
264 rather than after it.

### 17.4 Scope

`-p onnx-genai-server` only. I did NOT run the workspace: item 1 on the gate
is a known `cargo check` exit 101 from vendored x86 AVX2 on arm64 in other
crates, and I did not re-measure it. **264/0 is a claim about one package.**

## 18. C17 — a faithful pipe carrying a misattributed message

Filed at @d7cf9b84's prompt. They observed that the shared-buffer route
fails silently where the static-cache route fails loudly. They are right,
and the cause is one line further up than the silence.

### 18.1 The plumbing is excellent. I checked it first, expecting to find the defect there.

`batch_driver.rs:69 explain()` propagates the engine's own words verbatim:
*"continuous batching is INACTIVE ... The engine refused it: {reason}"*, and
its doc comment is the best sentence in the subsystem -- **"A reason shown
only on failure teaches a reader that silence means success, and silence is
also what a field that was never wired up looks like."** The enabled case
explains itself too. `driver.rs:694` logs it at `warn`, and it reaches
`/v1/status` as `batch_driver_detail`. **Nothing is lost in transit.**

### 18.2 The reason is wrong where it is BORN, not where it is carried

`decode/metadata.rs:160`:

    let shared_buffer = supports_present_binding
        && session.past_present_share_buffer_supported()
        && metadata_max_context.is_some();

**THREE INDEPENDENT CONDITIONS COLLAPSE INTO ONE BOOLEAN, AND THE BOOLEAN
KEEPS NO RECORD OF WHICH ONE FAILED.** Two are facts about the environment
(the EP declined; the session declined). Only the third is a fact about the
model.

Downstream, `batched.rs:653` composes the refusal:

    "continuous batching requires a STATIC-CACHE or shared-buffer
     past/present model"

That sentence blames **the model**. It is reached whenever ANY of the three
failed. So an operator whose *execution provider* declined is told to fix
their *model file* -- and the message is now warn-level, on the wire, and on
the page. **Every improvement to the honesty layer makes the wrong
attribution more visible and more authoritative.**

This is the branch's own `MISATTRIBUTED` classification occurring one floor
BELOW the layer that has a word for it. The honesty layer can only be as
honest as the reason handed to it, and it has no way to know this one is
mis-signed.

### 18.3 The asymmetry, which is the structural half

    static-cache route   reject_undeclared_static_cache (ort/decode/io.rs:194)
                         REFUSES TO BOOT, naming every missing key.
                         @d7cf9b84 proved this by execution: delete the
                         metadata and the server will not start.
    shared-buffer route  degrades to `false` and blames the model.

**TWO ROUTES TO ONE CAPABILITY, OPPOSITE FAILURE POSTURES, AND NOTHING IN
THE TYPE SYSTEM OR THE PROSE MARKS THEM AS DIFFERENT.** A reader who learns
the static-cache route's behaviour will generalise it, and be wrong.

### 18.4 Why this is worse than an ordinary bad error string

`supports_fixed_capacity_present_binding()` is negotiated with the EP at load
time. **The same model bytes batch on one machine and refuse on another, and
the refusal names the bytes.** The one artefact the operator can inspect,
diff and re-download is the one the message accuses, and it is identical on
both machines. That is unfalsifiable from the operator's side.

No test pins the attribution: zero test references to either capability
predicate. Control: `reject_undeclared_static_cache` is real, present code.

### 18.5 Fix, and it is small

Return a reason from the decision, not a boolean -- `Result<(), NoSharedBuffer>`
with three variants (`EpDeclined`, `SessionDeclined`, `NoMaxContext`), and let
`batched.rs` render the one it got. The plumbing to carry it **already exists
and is good**; it is being handed a string that averaged three causes into one.
`explain()` would then say which subsystem refused, which is the only thing
the reader actually needs.

**I am not asking for the boolean to become a config knob.** The decision is
correct; only its *explanation* is lossy. A capability that turns on the host
should be reported as a fact about the host.

---

## §19 — C18: a security comment that described a branch deleted six hours earlier

Measured at `4e93b97b`, fixed at `fa1fd425`. Raised by @c7a654ed, who twice
published it as open and unowned. It is in my lane and I took it.

### The defect

`routes/mod.rs`, on the `path` field of the models response:

> Configured directory. **Absolute on loopback; the basename otherwise**, so a
> non-loopback deployment does not leak the operator's username and filesystem
> layout on an endpoint with no authentication in front of it.

There is no such conditional. It was deleted at `2da3e851` (03:50:55).

    loopback mentions in routes/            4   ALL FOUR ARE PROSE
    branches inside model_path_for_display  0   unconditional basename
    CONTROL, the field is still declared    1   the grep resolves

The doc did not merely go stale. It documented a **disclosure policy** and a
**trust boundary** — "safe because of where we are bound" — on a field that is
now unconditionally redacted.

### Why the harmless direction was not harmless

The code was SAFER than its documentation. That is supposed to be the direction
that costs nothing. It cost most of the night.

@bb2ee824 reported "`/v1/models` STILL RETURNS THE ABSOLUTE PATH **ON LOOPBACK**".
That is this comment's wording, not a property of the code. Four wire
measurements were read through a model this sentence supplied: reviewers saw a
real absolute path on a real loopback port and had a sentence in the source
telling them it was *by design and conditional on the bind address*. The true
cause — a binary built from a sibling checkout where the fix is not an ancestor —
requires noticing that the running process is not the reviewed tree, and the doc
offered a cheaper explanation that fit every observation.

**A stale comment is worst when it is plausible, not when it is wrong.** This one
predicted the exact symptom under observation.

### The matched pair, and the ruling

This is the mirror of §13 (the Metal safety rationale describing a protection
that had been lifted), and the pair is the actual finding:

| | describes | fails toward |
|---|---|---|
| §13 Metal | a protection that no longer runs | **reassurance** |
| §19 loopback | a disclosure that no longer happens | **alarm** |

Both are one commit's prose left behind by one commit's code. The direction the
error points is **uncorrelated with its danger**, so "we would notice the
dangerous ones" is false — you notice neither, and here the *safe* one burned
more hours than the unsafe one has.

**The remedy was already in the tree, one file away.** `admin.rs` documents this
same decision correctly, and the difference is structural, not editorial: the
epitaph sits **on the function that enforces the property**, which is the field's
only writer. `mod.rs` restated the policy at a distance from its enforcement, and
a restatement has no reason to be updated when the enforcement changes — nothing
links them, no test reads either, and the compiler is indifferent to both.

> **A security property is documented at its enforcement point. Every other site
> points at that one and restates nothing.** A second description of one decision
> is not redundancy — it is an unowned copy that will diverge, and it diverges
> silently in whichever direction the next commit happens to take.

Fourth sighting this session of *two places describe one thing and nothing says
which wins* (C16's two vocabularies; the two `run-demo.sh` binaries; the two
`scenario-switcher.test.js`; this). It is the branch's dominant structural
failure and it has never once announced itself.

### Disposition

Fixed at `fa1fd425`. Doc-only: 8 insertions, 3 deletions, **zero non-comment
lines** in the diff, asserted mechanically. `cargo test -p onnx-genai-server`
raw unpiped exit 0, **264 passed / 0 failed / 4 ignored** — byte-identical to the
denominator recorded at §17, so the change is inert to behaviour. The new prose
names the deleting SHA and points at `admin::model_path_for_display` rather than
restating what it enforces.

I did not add a note above the stale line. Per @c7a654ed's rule — a retraction
that lives anywhere except beside the retracted string has been filed, not
applied — the false sentence is **gone**, not annotated.

---

## §20 — C19: the dotfile rule is bypassed by percent-encoding, PROVEN ON THE WIRE

Measured at `fa1fd425`, against a binary built from the demo worktree at 04:38,
newer than the dotfile fix `1384f7aa` (04:28:56). Probe server on `127.0.0.1:8791`,
started and reaped by me; the four demo origins untouched.

**@c0de4c2e measured `demo_assets.rs` growing 122 -> 566 lines and declined to
file a finding. I took it, because 78% of the code deciding which files a browser
can pull off this disk had never been read by a reviewer.**

### The result

    POSITIVE CONTROL  /demo/app.js                                  200
    NEGATIVE CONTROL  /demo/zzz-no-such-file.js                     404
    dot segment, decoded    /demo/node_modules/.vite/…/results.json 404   rule works
    dot segment, ENCODED    /demo/node_modules/%2Evite/…/results.json 200 ** BYPASS **
    lower-case %2e                                                  200   ** also **

    bytes on the wire 88  ==  bytes on disk 88   byte-identical, real content
    {"version":"4.1.10","results":[[":state-channel.test.js",…]]}

### Why

`restrict_demo_assets` authorises on `request.uri().path()` — the **raw**,
still-percent-encoded string. `ServeDir` then **decodes** it before opening a
file. Two parsers, one string, and nothing requires them to agree. The
middleware sees a segment `%2Evite` that does not begin with `.`; the file system
sees `.vite`.

This is the fourth-and-a-half sighting of the branch's dominant structural
failure — *two places describe one thing and nothing says which wins* — in its
sharpest form yet: **the two places are two PARSERS OF THE SAME BYTES.** A
divergence here is not a stale comment. It is an authorisation decision made
about a string that is not the string that gets used.

### The part that makes this worth a finding rather than a nit

The author **already identified this exact danger and wrote it down**:

> `.env`, `.npmrc` and `.git/config` were refused only incidentally, because
> their extensions are not on the list. **That is a refusal by coincidence**, and
> it inverts the moment someone adds `json` to a dotted config directory.

They were right, they built the segment rule to end the coincidence — and the
segment rule does not reach the encoded form. **So the tree is back in exactly
the state its own author declared unacceptable, and the only thing still standing
between a visitor and `.vscode/settings.json` is the extension allowlist that the
comment says must not be relied on.** I confirmed the allowlist does still hold:
`%2Egit/config` and `%2Egit/HEAD` are 404, and `.md` is 404 — so my own review
document is no longer served, which is a real repair. The blast radius is
bounded to *files inside dot-directories carrying an allowed extension*, and
`.vscode/settings.json` is squarely inside it.

### Why fifteen passing tests could not see it

Every one of the fifteen tests calls `demo_path_is_servable("…")` with an
**already-decoded literal**. The premise that the predicate's input equals
`ServeDir`'s input is shared by all fifteen and asserted by none — the Lead's
first-frame blindness, in Rust, on the security boundary. **A guard tested only
through its own front door never meets the decoder that sits in front of it.**

### The fix, and why the obvious one is the wrong one

The obvious fix is to percent-decode inside `demo_path_is_servable`. **Do not.**
That creates two decoders that must stay byte-compatible forever, which is the
same defect with an extra moving part.

**Refuse any `/demo/` path containing `%` at all.** Measured: **91** assets with
a servable extension outside `node_modules`, of which **0** require
percent-encoding in their names (control: zero filenames containing a space).
The ban costs nothing real, needs no decoder, and **cannot drift**, because it
removes the differential rather than trying to keep two parsers in agreement.

> **Do not authorise on a string that something downstream will transform.**
> Either authorise on the post-transform value, or forbid the transform. Keeping
> two parsers in sync is not a fix, it is a maintenance obligation nobody signed.

Severity: MAJOR, non-blocking for the demo — loopback-only, bounded by the
extension allowlist, and no secret in this tree currently sits in a dot-directory
under an allowed extension. It blocks nothing tonight. It should not ship.

---

## §21 — gate item 1 adjudicated: not this branch's defect, and structurally so

Measured at `cb03105f`. This is the last 🟡 on the board and it has been carried
by everyone, re-measured by nobody, all night. I carried it too. It is now
settled by execution.

    cargo check --workspace --all-targets      RAW UNPIPED EXIT 101
    error in cc-rs: c++ --target=arm64-apple-macosx … qgemm_kernel_avx2.cpp

### Three questions, three measurements

**Is it ours?** Files this branch changed under `crates/mlas-sys`: **0**.
Control, files changed under `crates/onnx-genai-server`: **17** — so the diff
instrument reaches the tree and the zero is real. Pre-existing, merge-base
`f55e459b`.

**Does it reach the demo?** `mlas-sys` is depended on by exactly one crate,
`onnx-runtime-ep-cpu`, and is not in `onnx-genai-server`'s graph at all —
`cargo tree -p onnx-genai-server -i mlas-sys` cannot even resolve the spec.
And the decisive positive: `cargo check -p onnx-genai-server --all-targets`
**raw exit 0**.

**Why does it fail?** `crates/mlas-sys/build.rs` has **no concept of the target
architecture**:

    target_arch 0 · CARGO_CFG_TARGET_ARCH 0 · aarch64 0 · arm64 0 · TARGET 0
    CONTROL avx2 -> 14   ⬅ the grep reaches this file; the zeros are real

`:36` unconditionally does `lib.join("x86_64")` and `:112` passes
`-mavx2 -mfma -mf16c -mavxvnni`. cc-rs correctly supplies
`--target=arm64-apple-macosx`; the build script never asks. **It is not a broken
build, it is a build script that was written when there was one architecture and
has never been told otherwise.** A vendored kernel directory named `x86_64`
joined without a conditional is a hardcoded assumption wearing a path separator.

### Disposition

**Gate item 1 is NOT a defect of this branch and cannot affect the demo.** It
should move from 🟡 *carried, not re-measured* to **🟡 adjudicated: pre-existing,
out of scope, unreachable from the served artefact** — a different status, and
the difference is that nobody needs to look at it again tonight.

### A correction to the record, offered as a denominator and not a dispute

@c0de4c2e generously closed what they understood to be my standing limit by
running `cargo test -p onnx-genai-server --no-fail-fast` → 264/0/4. **That is the
same single package I ran, and it closes a different limit than the one I
carried.** The limit I published was *workspace-wide cargo has never been run by
anyone*. It had not been. It has now, by me, and the answer is exit 101.

This is tonight's dominant class one more time — **a correct measurement of an
adjacent subject** — and it is worth naming precisely because @c0de4c2e's
measurement was rigorous, their intent was to relieve me of a debt, and the
result was that a real open item was very nearly retired by agreement. Two
parties agreeing that a limit is paid is not the same as the limit being paid.
The Lead's rule applies verbatim: **name the path, never say "the suite".**

---

## §22 — @1cb42f0e's duplicate-key exposure: 2 of 4, and the other 2 are immune BY DESIGN

Measured at `1f8d3f94`. @1cb42f0e shipped `dashboard/testing/object-keys.js` and
named four unguarded literals — `NEVER_BIND`, `CARD_FIELDS`, `STATE_ALIASES`,
`DECLARED_ROUTING_VIOLATIONS` — addressing me directly as the measurable answer
to my *a shared helper is only as good as its adoption*. I took it because
`NEVER_BIND` is the list holding P1 closed, and a silently dropped ban there is a
disclosure.

### The result, and it is better news than the ask

    NEVER_BIND      Object.freeze([ …   ARRAY   -> IMMUNE, cannot silently overwrite
    CARD_FIELDS     Object.freeze([ …   ARRAY   -> IMMUNE
    PROVENANCE      Object.freeze({ …   OBJECT  -> exposed · 35 keys · duplicates 0
    STATE_ALIASES   Object.freeze({ …   OBJECT  -> exposed ·  6 keys · duplicates 0

    POSITIVE CONTROL on STATE_ALIASES: inject a duplicate of an existing key
      clean    keys=6  duplicates 0
      mutated  keys=7  duplicates REPORTED   ⬅ the helper fires

**The two security-critical lists are structurally immune.** `NEVER_BIND` and
`CARD_FIELDS` are arrays of frozen records, and an array has no key to overwrite.
That is not luck — an array-of-records was the right shape for an ordered ban
list with a `why` per row, and choosing it **removed the entire silent-overwrite
class instead of guarding against it.** This is the single best structural
decision I have reviewed on this branch: the defect cannot be reintroduced by a
careless edit, because the container does not have the failure mode. Praise where
it is due, and it belongs to whoever wrote `NEVER_BIND`, not to a guard.

### The defect I found instead, and it is in the instrument

`findLiteralOpener(src, 'NEVER_BIND')` **does not refuse.** It returns an opener,
`declaredKeys` returns **7**, and `duplicatesAmong` returns **0** — which reads
exactly like a clean audit of a seven-entry ban list. It is the property names of
the *first record inside the array*: `endpoint`, `field`, `why`, …

I caught it only because my positive control failed. I injected a duplicate
`'server.model_path': true` into a copy of `NEVER_BIND` and the count stayed
**7 → 7**. Under @732c7548's rule — *a mutation is evidence only if you proved it
landed on the subject* — that mutation was void, so the zero was void, and **I
was one broadcast away from publishing "the ban list holding P1 closed is
duplicate-free" on evidence that had never touched the ban list.**

The helper's own header says it throws rather than returning a clean answer when
it loses sync. **It should throw on an array too.** An object-key auditor handed
an array has not lost sync — it has been asked a question its subject cannot be
asked, and answering is worse than failing, because the answer is a plausible
zero. @1cb42f0e's own law, applied one level up:

> A control is only as wide as the instrument it controls — it proves the
> instrument is looking, never that it is looking everywhere.

**And a control is only as valid as the KIND of subject it was taken on.** My
`STATE_ALIASES` control transfers to `PROVENANCE` because both are
`Object.freeze({`. It transfers to `NEVER_BIND` not at all. Stated so that my
`PROVENANCE` zero is read with the right warrant.

### Disposition

Not a defect in the dashboard. Two of the four named call sites need no guard and
should be recorded as immune rather than as unaudited — **an unguarded structure
and an unguardable-because-immune structure render identically on a coverage
list, and only one of them is a gap.** The remaining work is one `throw` in the
helper.
