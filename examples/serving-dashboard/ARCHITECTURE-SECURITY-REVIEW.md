# Architecture & Security Review — `feat/genai-demo-dashboard`

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

**Recommended fix — narrow, in `apply_status` (`node.rs`), not in the type.** The wide
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
| **P1** | **Model-path disclosure — server half CLOSED by deletion, client half is now a caption defect, not a leak.** The server no longer has a disclosure switch at all: `model_path_for_display()` in `routes/admin.rs` takes one argument and returns `file_name()` unconditionally, and `tests.rs` `no_configuration_can_re_enable_full_path_disclosure` asserts at *source* level that neither `may_disclose_model_paths` nor `bind_addr` reappears in `state.rs`, `routes/admin.rs` or `cli.rs`. **No absolute path reaches the wire in any configuration.** What survives is that `ui/model-card.js` still labels the value `Directory` and `dashboard/system.js` labels it `model directory`, while the value is now a *basename* — @376a0297 predicted this exact caption defect before it landed | 🟡 **caption, not disclosure** — severity collapsed by the server fix | executed |
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
