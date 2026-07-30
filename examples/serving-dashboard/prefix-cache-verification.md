# Prefix-Cache Verification — DYNAMIC model (`models/qwen2.5-0.5b`)

**Author:** QA Tester (@fc8b5d97) · **Date:** 2026-07-29 23:30–23:36 PDT
**Question:** Does prefix caching actually fire on the dynamic model, and does it produce
the TTFT collapse Scenario B is built around?

## VERDICT: 🔴 RED — proven absent, not merely unobserved

| claim under test | result |
|---|---|
| `prefix_cache_hits` rises above 0 on dynamic | ✅ yes — 19 hits / 20 lookups (95 %) |
| …but does it indicate genuine prefix **reuse**? | ❌ **no — it increments for every request, including controls that share nothing** |
| Second identical request shows materially lower TTFT | ❌ **no — +7.0 %, i.e. no benefit at all** |
| Scenario B's "TTFT collapse" payoff | ❌ **does not exist** |
| Scenario B's "hit rate climbing off zero" payoff | ❌ **does not exist** — pins ~95–100 % from the first request and never moves |

**The inference the crew was working from — "prefix caching lives in the paged KV manager →
paged KV is active on dynamic → therefore prefix caching works on dynamic" — is false.**
Page *allocation* is alive (@e00032a4 verified that correctly). Page *reuse* is not.

---

## 1. Setup

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai-demo
ONNX_GENAI_EP=cpu \
  <artifact-dir>/clean-binary/onnx-genai-server-clean-d49d3c8 \
  --model ../onnx-genai/models/qwen2.5-0.5b --model-id qwen-dynamic \
  --addr 127.0.0.1:8126 --enable-debug-endpoints
```

No build performed — used the preserved clean binary (`d49d3c8f…`). Server log confirmed:
```
INFO onnx_genai_server::driver: continuous batch driver disabled; using per-request engine path
```
This independently confirms the batching/paged-KV mutual exclusivity that justifies the
two-server demo. Counters started at `prefix_cache_hits: 0, prefix_cache_lookups: 0`.

Measurement: streaming SSE, timestamp of first content token = TTFT. `temperature=0`,
`max_tokens=8` (TTFT-dominated by design — decode is irrelevant to this question).

---

## 2. Experiment 1 — naive cold/warm pairs (the requested procedure)

Four distinct prefixes; each fired twice, second only after the first completed.

| pair | TTFT cold | TTFT warm | delta | `prefix_cache_hits` before→mid→after |
|---|---|---|---|---|
| 0 | 1694.8 ms | 1331.1 ms | −21.5 % | 0 → 0 → 1 |
| 1 | 1313.8 ms | 1336.5 ms | **+1.7 %** | 1 → **2** → 3 |
| 2 | 1572.5 ms | 1550.6 ms | **−1.4 %** | 3 → **4** → 5 |
| 3 | 1498.2 ms | 1311.5 ms | −12.5 % | 5 → **6** → 7 |

Median cold 1535.3 ms → warm 1333.8 ms (**−13.1 %**). Final: 7 hits / 8 lookups (87.5 %).

**Taken at face value this looks like a PASS.** It is not — two red flags:
1. The per-pair deltas are wildly inconsistent (−21.5 %, +1.7 %, −1.4 %, −12.5 %). A real
   cache does not help sometimes.
2. **From pair 1 onward the COLD request also scored a hit** (bold above). A brand-new
   prefix cannot legitimately hit a cache. The counter is measuring something other than
   reuse.

This is why the naive procedure alone is insufficient, and why I ran a control.

---

## 3. Experiment 2 — controlled A/B (the decisive test)

- **ARM A (shared):** one identical ~900-token prefix, fired 6×. Request 0 = cold, 1–5 = warm.
- **ARM B (control):** six prefixes that **differ from token 0** (`Bravo…`, `Charlie…`,
  `Delta…`, `Echo…`, `Foxtrot…`, `Golf…`), so there is no meaningful shared prefix.
  A prefix cache matches from the start of the sequence, so these must not benefit.

| ARM A (identical prefix ×6) | TTFT | | ARM B (all different) | TTFT |
|---|---|---|---|---|
| req 0 (cold) | 1383.2 ms | | req 0 | 1246.7 ms |
| req 1 (warm) | 1424.5 ms | | req 1 | 1290.0 ms |
| req 2 (warm) | 1060.3 ms | | req 2 | 1261.1 ms |
| req 3 (warm) | 1090.9 ms | | req 3 | 1066.0 ms |
| req 4 (warm) | 1393.2 ms | | req 4 | 1085.2 ms |
| req 5 (warm) | 1341.2 ms | | req 5 | 1446.4 ms |
| **warm median** | **1341.2 ms** | | **median** | **1253.9 ms** |

> **ARM A warm is +7.0 % SLOWER than the no-sharing control.**
> Reusing a 900-token prefix confers **zero** measurable benefit.

**Every one of the 12 requests incremented `prefix_cache_hits`** — including all six ARM B
controls that share nothing. Final counter: **19 hits / 20 lookups = 95 %**.

---

## 4. Sensitivity control — proving the test *could* detect the effect

Without this, "no effect observed" could just mean an insensitive test.

| prompt | TTFT median |
|---|---|
| short (~10 tokens) | **139.6 ms** (raw 109, 140, 140) |
| long (~900 tokens) | **1380.1 ms** (raw 1380, 1298, 1501) |

**Prefill of the ~900-token body costs 1241 ms — 90 % of long-prompt TTFT.**

A working prefix cache reusing that body would collapse TTFT from ~1380 ms toward ~140 ms,
a **~90 % drop**. That is enormous and unmissable against the ~±10 % run-to-run noise.

**Observed: +7.0 %.** The effect is therefore **proven absent**, not merely unobserved.

---

## 5. Why the hit counter reads 95 % while nothing is reused

`prefix_cache_hits` appears to increment on any non-zero prefix match. Every request shares
the chat-template preamble (a few tokens), so effectively **every** request scores a "hit"
regardless of whether meaningful reuse occurred. The counter measures *"did any token
match"*, not *"was useful work skipped"*.

**Consequence for the dashboard:** a hit-rate panel would display ~95–100 % from the very
first request, look excellent, be precisely computed, and be **completely meaningless**.
That is the exact failure mode QA-PLAN.md exists to catch — a correctly-computed number
under a name that implies something untrue.

**The two counters disagree in opposite directions:**

| server | reading | what it actually is |
|---|---|---|
| scatter (`qwen2.5-0.5b-scatter-v2`) | 0 hits / 135 lookups | **neither number is measured.** Numerator is a hardcoded literal `0` (`batched.rs:262`/`:486`); denominator counts *completed generations* (`metrics.rs:130-132`) and would read 135 with the cache deleted. Cache genuinely is bypassed here — but that is proven from source (`batched.rs:101-110`), **not by this ratio** |
| dynamic (`qwen2.5-0.5b`) | 19 hits / 20 lookups (95 %) | records a hit for **every completed generation**, including prompts sharing nothing (measured +12/12, 6 unique) |

Neither is trustworthy. **Do not ship a prefix-cache hit-rate panel in any form until the
metric is fixed** — both available numbers would mislead a viewer.

**A meaningful metric would need `prefix_cache_hit_len`** (tokens actually reused) or
reused-tokens ÷ prompt-tokens — not a boolean hit count. Recon flagged
`GenerateResult::prefix_cache_hit_len` as existing internally; it is not exposed via any API.

---

## 6. Impact and recommendation

**Scenario B as designed cannot work.** Both halves of its payoff fail:
- ❌ TTFT collapse — does not happen (+7.0 % vs control)
- ❌ Hit rate climbing off zero — pins near 100 % immediately and never moves, meaninglessly

**Recommendation:** re-scope or cut Scenario B now, while the panel is still cheap. Options:
1. **Cut it.** Cleanest. Two scenarios that are true beat three where one is fiction.
2. **Re-scope to paged-KV page allocation**, which @e00032a4 *did* verify is genuinely live
   (allocated 3, freed 3, 14 612 pages). That is a real, defensible paged-KV story — it is
   just not a *prefix-reuse* story.
3. **Fix prefix caching first**, then demo it. Correct but almost certainly out of scope.

**Do not** proceed on the assumption that this is a metrics-plumbing gap that telemetry work
will incidentally fix. The TTFT evidence shows the underlying *reuse* is absent — this is a
functional gap in the engine, not a reporting gap.

## 7. Raw data

| file | contents |
|---|---|
| `raw/qa-prefix-dynamic.json` | experiment 1 — four cold/warm pairs |
| `raw/qa-prefix-controlled.json` | experiment 2 — ARM A vs ARM B control |
| `raw/qa-prefix-sensitivity.json` | sensitivity control (short vs long TTFT) |
| `raw/qa-dynamic.log` | dynamic server log incl. `continuous batch driver disabled` |

Server stopped. No repository files modified.

---

## §7. ADDENDUM — ROOT CAUSE FOUND IN SOURCE (post-verdict)

My verdict in §1-§6 was empirical: "no TTFT benefit, effect proven absent."
I then read the source to stress-test my own RED, because the repo ships
`crates/onnx-genai-engine/tests/prefix_speedup.rs`, which *asserts* prefix reuse
works. A passing test that contradicts a measurement means one of them is
mismeasuring. It was the test's scope, not my measurement.

### 7.1 The gate

`crates/onnx-genai-engine/src/decode/state.rs:206`

```rust
pub(crate) fn uses_token_prefix_cache(&self) -> bool {
    self.has_runner() || self.is_windowed()
}
```

`prepare_session_prefix` (`engine/runtime.rs:997`) forks on this into two
*completely different* prefix mechanisms:

| Branch | Condition | What it does |
|---|---|---|
| **A — token prefix cache** | `uses_token_prefix_cache()` | Computes a `common_prefix_len` over `self.token_prefix_cache` and **returns it as a number**. |
| **B — paged prefix cache** | `else` + `kv_model.is_some()` + `page_table.tensor_config.is_some()` | `lookup_shared` → `retain` pages → `attach_pages_to_sequence` → `materialize_sequence` → `load_materialized_past` → sets `loaded_prompt_prefix`. |

### 7.2 THE DEFECT — Branch A reports a hit it never serves (P1)

Branch A (`runtime.rs:1017-1024`) is **reporting-only**. It loads no KV,
materializes no pages, and — critically — never sets `loaded_prompt_prefix`.
Twenty lines later:

```rust
if started_empty {
    state.tokens.extend_from_slice(&prompt_tokens[loaded_prompt_prefix..]);
```

`loaded_prompt_prefix` is still `0`, so **the full prompt is queued and prefill
recomputes every token**. The returned `prefix_cache_hit_len` is a number with
no compute saving behind it.

Worse, the value is:

```rust
.map(|cached| common_prefix_len(cached, prompt_tokens).min(cached.len()))
.filter(|&len| len > 0)
.max()
```

Any single shared leading token scores a hit. Every `/v1/chat/completions`
request shares the chat-template preamble, so **every request reports a hit
forever**. That is precisely what I measured: 19 hits / 20 lookups (95%),
including all six controls whose prefixes differ from token 0, with a
**+7.0% slower** shared-prefix arm.

Note the code itself draws exactly this distinction for the *connector* path
30 lines below — "If injection is not possible we fall back to the
reporting-only `lookup_extension`, **never claiming a hit we can't serve**"
(`runtime.rs:1097-1099`). Branch A violates the rule its own neighbour states.

### 7.3 Which branch did my dynamic server take?

Honest answer: **I did not isolate it, and the product conclusion does not
depend on it.** Either
(a) Branch A was taken (`is_windowed`) → hit is reporting-only by construction; or
(b) Branch B was taken but gated off by `tensor_config` / matched too few tokens
to materialize a page.

Both are consistent with every number I recorded, and **both mean no reuse**.
The distinguishing experiment, if anyone wants it, is one `tracing::debug!` of
`loaded_prompt_prefix` in `prepare_session_prefix` — I did not add it because I
modify no Rust. I flag this as an open sub-question rather than assert a
mechanism I did not observe.

### 7.4 Why the shipped test passes anyway

`prefix_speedup.rs` asserts `warm.prefix_cache_hit_len > 0` and, in
`greedy_output_matches_with_and_without_prefix_reuse`, asserts output equality.
**Neither asserts that prefill got shorter.** Under Branch A both pass while
zero work is saved — a hit counter that is always nonzero satisfies `> 0`, and
recomputing the prefix trivially produces identical greedy output. The test
fixture is `tiny-llm-scatter` with 4-token prompts, where a timing assertion
would be meaningless anyway.

**This is the bug class the test was written to catch, and it does not catch it.**

### 7.5 Consequences for the demo

1. **`prefix_cache_hits` MUST NOT be bound to any panel.** On the dynamic path
   it reads ~95% while delivering nothing. A panel showing "95% cache hit rate"
   next to a flat TTFT is a fabricated measurement — the exact AC6/AC8/AC11
   failure the honesty bar exists to prevent. It is more dangerous than the
   hardcoded `0.0` stubs, because a zero looks broken and invites scrutiny while
   95% looks like success.
2. **AC7 provenance audit:** classify `prefix_cache_hits` as
   **MISLEADING — reports unserved hits**, a third category beyond
   GENUINELY-MEASURED and DOCUMENTED-ZERO. It is not a zero to be fixed; it is a
   nonzero to be distrusted.
3. **Scenario B stays CUT or re-scoped**, per §6. Nothing in this addendum
   rescues it; the addendum only explains *why* it cannot be rescued by wiring a
   counter. This is a functional gap in prefix reuse on the served paths, not a
   reporting gap, so no telemetry PR will incidentally fix it.
4. **A real hit metric must be `loaded_prompt_prefix`** (tokens whose prefill was
   actually skipped), not `prefix_cache_hit_len` (tokens that happened to match).

### 7.6 The configuration that would genuinely work — and why we cannot reach it

Branch B is the only branch that truly skips prefill. Reaching it needs a model
on the paged-KV path **and** the per-request driver path. `max_batch == 0` makes
`continuous_batch_manager` bail (`batched.rs:603`), which flips
`continuous_batch_supported` false (`driver.rs:415`) and falls back to the
per-request path — so a `--max-batch 0` escape hatch would let a scatter model
reach it.

**But `DEFAULT_MAX_BATCH` is a compile-time `const` (`state.rs:25`) with no env
override**, so this is untestable without a rebuild, and I modify no Rust.

Flagging for whoever implements the approved `--max-batch` flag: **accept 0 and
document it as "disable continuous batching."** It costs nothing, it is the
only lever that can put a real model on the prefix-reuse path, and without it
that path is unreachable from the server at any setting. I am **not** claiming
this would make Scenario B work — that requires the measurement above, which
requires the flag first.

---

## §8. REPLICATION OF THE PM'S GREEN RESULT — AND A CORRECTION TO MY OWN VERDICT

@376a0297 measured `hits 0→1`, `e2e 1.53s → 1.22s (~20% faster)` and called
Scenario B real. That contradicted §1-§7, so I replicated their protocol and
added the control arm it was missing. The outcome corrects **both** of us.

### 8.1 PROVEN, and immune to CPU noise: the counter is not evidence

Two prompts sharing **nothing** with anything previously sent
("Zebra quantum ledger 9917 vortex marmalade telemetry drift." and
"Kaleidoscope 4471 tungsten pelican archipelago sonata."), fired on a warm server:

```
before      : 15 / 16
after uniq1 : 16 / 17
after uniq2 : 17 / 18
hits gained by two prompts that share no prefix: 2
```

Earlier in the same run, 12 requests (6 repeated + 6 deliberately unique)
produced **+12 hits, hit_rate 0.9375**. **Every completed generation scores a
hit, including prompts that differ from token 0.**

⇒ `prefix_cache_hits` going `0 → 1` is **exactly what an unrelated prompt would
also produce.** It cannot distinguish reuse from no reuse, so it cannot support
"the cache genuinely hit." This is a counting fact — it does not depend on
timing, load, or sample size. It confirms §7.2 from the black box.

### 8.2 The control arm the original test lacked

Fresh server per run, PM's prompt shape (~30x repeated sentence), request 1 cold,
request 2 either identical (`repeat`) or a **different** prefix (`unique`):

| Replicate | `repeat` Δ(req2 vs req1) | `unique` Δ(req2 vs req1) |
|---|---|---|
| 1 | **−5.89%** | **−9.45%** |
| 2 | **−39.15%** | **−16.22%** |

**In replicate 1 the unique prefix sped up MORE than the repeated one**, while
doing slightly more work (611 vs 580 prompt tokens). The second request is
faster **either way** — that is first-request warmup (ORT arena growth, lazy
init), and the PM's design places the entire measured effect on request #1.
Replicate 2 reverses the ordering. Spread across replicates is ~33 points.

By the PM's own AC33 rule — *"if spread > effect size, the result is
INCONCLUSIVE, not a pass"* — a single n=1 pair cannot carry this claim.

### 8.3 I AM DOWNGRADING MY OWN VERDICT: RED → INCONCLUSIVE (timing only)

A warm, strictly interleaved A/B (n=6/arm, PM prompt shape) gave
**repeated prefix 16.98% FASTER** (medians 2.351s vs 2.832s) — the **opposite**
of my §5 result (+7.0% slower). Raw values ranged 2.05s-5.34s.

I will not sign off a GREEN on that, and I no longer defend my RED as
"proven absent" either. **Two of my own runs disagree, so my instrument is not
resolving the effect right now.** Applying to myself the standard I applied to
the baseline: *ask whether the instrument can resolve the effect before
reporting a number.*

**Cause: the machine is saturated.** `load average: 22.56 28.63 28.47` on a
10-core box — 2-3x oversubscribed by concurrent crew work. My AC33 work already
established that a **byte-identical binary** swung **−9.8%** under load changes
alone. At load 22+, a 20% e2e delta is well inside noise.

### 8.4 Honest status and what would actually settle it

- **PROVEN:** `prefix_cache_hits` / `hit_rate` are unusable as evidence of reuse
  (§7.2 in source, §8.1 empirically). **Do not bind them to a panel, and do not
  cite them as proof the scenario works.**
- **UNPROVEN, EITHER WAY:** whether prefix reuse delivers a real speedup on the
  dynamic profile. Not GREEN, not RED — **INCONCLUSIVE pending a quiet window.**
- **The decisive measurement is TTFT, not e2e.** Prefix reuse can only shorten
  **prefill**. e2e buries it under decode, which is the majority of the request
  and is unaffected. My §5 sensitivity control put prefill at ~90% of TTFT, so a
  genuine full-prefix hit should collapse TTFT by roughly that much — an effect
  far too large to be lost in noise **if measured on a quiet machine**.
- **Protocol to settle it (~10 min, no build):** quiet window, one server, warm
  it, then interleaved TTFT (streaming, first-chunk timestamp) repeated-vs-unique,
  n>=15/arm, report medians + CV + 95% CI. Overlapping CIs ⇒ cut Scenario B.

Until that runs, Scenario B's payoff is **unverified**. Building the panel is
fine; **claiming the number is not.**

---

## §9. 🔴 THE MEASUREMENT NOBODY HAD TAKEN — DYNAMIC MODEL, TTFT. VERDICT: **RED**

Ordered by @12e42da8: *"NOBODY HAS EVER MEASURED PREFIX CACHING ON THE DYNAMIC MODEL."* Correct —
§8 measured the *counter* on dynamic but not the *payoff*. This section takes the payoff
measurement. **Scenario B's headline does not survive it.**

Server: `models/qwen2.5-0.5b` (dynamic), preserved clean binary, `--enable-debug-endpoints`,
port 8129. Startup log confirms the right path is under test:
`continuous batch driver disabled; using per-request engine path` — i.e. **not** the batching path
where the counter is a hardcoded literal. This is the configuration most favourable to prefix reuse.
Raw: `raw/qa-prefix-dynamic-ttft.json`, harness `harness/qa_prefix_dynamic_ttft.py`.

### §9.1 Q1 — do hits move off zero on dynamic? **YES — and that is exactly why the counter is worthless**

`prefix_cache_hits_total` reached **31 with 32 lookups (hit rate 0.96875)**. It moves. But 15 of
those 32 generations were **deliberately share-nothing prompts** (random preamble, differing from
the first token). Then a clean per-request probe with three wholly-unrelated prompts:

| event | hits | lookups |
|---|---|---|
| start | 31 | 32 |
| after unrelated prompt #1 | **32** | 33 |
| after unrelated prompt #2 | **33** | 34 |
| after unrelated prompt #3 | **34** | 35 |

**+1 hit per generation, every generation, for prompts sharing nothing.** The counter counts
completed generations. It is **not evidence of reuse on the dynamic profile any more than on
scatter** — on scatter it is stuck at a literal `0`, here it is pinned at ~100%. **Two different
mechanisms, both uninformative.** This confirms §8 on the correctly-named metric
(`*_total` suffix — see §9.4).

### §9.2 Q2 — does a repeated prefix actually reduce TTFT?

Interleaved REPEAT/UNIQUE, n=15/arm, `max_tokens=4` so TTFT dominates, ~1500-token shared body.
The UNIQUE arm differs at the **first token** and is **matched in length** (same 120-word random
preamble construction from the same vocabulary), so both arms do the same amount of prefill work.

| analysis | repeat | unique | delta | p |
|---|---|---|---|---|
| all data (n=15) | 4.283 s | 5.081 s | **−15.71%** | 0.052 |
| warm, first 2 pairs dropped (n=13) | 4.167 s | 5.031 s | **−17.16%** | 0.034 |
| **paired**, warm (n=13) | — | — | **−4.90% median** | sign test 10/13, p≈0.09 |

*(Dropping the first two pairs is declared post-hoc: they were 15.9 s / 15.5 s / 14.0 s / 6.9 s,
plainly cold-start, and CV fell 63%→24% on their removal. Both analyses are reported.)*

So there **is** a small repeat advantage, somewhere between 5% and 17% depending on the estimator,
of marginal significance. **The question is not whether it is non-zero. It is whether it is the
effect Scenario B promises.**

### §9.3 THE CONTROL THAT DECIDES IT — how big *should* the effect be?

If a ~1500-token prefix were genuinely reused, prefill would be **skipped**, and TTFT would fall to
roughly the cost of a single forward pass. That is directly measurable — TTFT vs prompt length on
this exact server:

| prompt | TTFT (median of 3) |
|---|---|
| ~15 tokens | **0.222 s** |
| ~380 tokens | 0.736 s |
| ~1500 tokens | **3.913 s** |

**Prefill dominates TTFT at these lengths.** Therefore genuine full-prefix reuse predicts
**3.913 s → ~0.222 s, a −94% collapse.**

| | value |
|---|---|
| predicted if prefix reuse works | **−94%** |
| measured (paired median) | **−4.9%** |
| measured (unpaired medians, warm) | −17.2% |
| **observed / predicted** | **≈ 1/19** |

**The measured effect is an order of magnitude too small to be prefill elision.** It is the size
one expects from incidental warmth — allocator reuse, ORT arena state, page residency — not from
skipping 1500 tokens of attention. And this argument is **load-independent**: a −94% effect cannot
hide under a 25% CV. The noise floor is irrelevant at that effect size, which is precisely why the
prediction was worth computing.

**This matches the source root cause in §7 exactly:** `runtime.rs:1017-1024` computes a hit length
and returns it **without setting `loaded_prompt_prefix`**, so the full prompt is queued and prefill
recomputes every token. The counter reports a hit; the work is done anyway. §9 is the empirical
confirmation of §7, taken on the profile where the cache was supposed to be real.

### §9.4 Harness defect I caught in my own run — reported because it nearly became a finding

The first pass printed `counters before: {}` and `hits gained: +0`. **That "+0" was a parser bug,
not a measurement** — I matched `onnx_genai_prefix_cache_hits`, but the Prometheus names carry a
`_total` suffix. Had I not checked an empty dict against a live `/metrics`, I would have reported
"hits never move on dynamic" — **a dramatic, wrong, and very quotable result**, and the exact
inverse of the truth. Filed as a standing rule for this project's harnesses:
**a counter that reads zero must be distinguished from a counter that was never found. Assert the
metric exists before reporting its value.** Same failure class as the field-name defects we have
been auditing all day, arriving through the test harness instead of the server.

### §9.5 VERDICT — RED

- **`prefix_cache_hits` / `hit_rate` must not be displayed as measurements on the dynamic profile.**
  Confirms and closes the escalation I raised three times: the ratified `unavailable when
  lookups == 0` rule does not fire here (lookups is never 0) and would have shipped **96.9%** as a
  genuine reading.
- **Scenario B's promised payoff — a TTFT collapse and a hit rate climbing off zero — is not
  present.** The hit rate does not climb off zero, it is *pinned near 1.0 regardless of input;* and
  the TTFT effect is ~1/19th of what real reuse would produce.
- Per the 🔒 ruling the **panel still ships**, rendering `not-applicable` on batching and
  `unavailable` on dynamic. **The SCENARIO is the cuttable part, and this is the evidence for
  cutting it** — obtained while the panel is still cheap to change.
- Recommended replacement if a third scenario is wanted: **none needed** — Scenario A (batching)
  and Scenario C are both backed by real numbers. Better two honest scenarios than three with one
  fabricated.

---

## 10. THE MECHANISM, NAILED AT HEAD — AND A CORRECTION TO MY OWN §9

In §9 I reported that on the dynamic server `prefix_cache_hits_total` *"increments +1 for every
generation, with wholly unrelated prompts."* **That is true only on the CHAT endpoint, and I did not
say so.** Re-measured at HEAD on a freshly booted dynamic server (`:8124`, `run-demo.sh` pair):

### 10.1 Raw completions — the counter is HONEST

`POST /v1/completions`, five wholly unrelated prompts:

| prompt | hits | lookups | hit_tokens |
|---|---|---|---|
| (before) | 0 | 0 | 0 |
| `The capital of France is` | **0** | 1 | 0 |
| `def quicksort(arr):` | **0** | 2 | 0 |
| `Photosynthesis converts` | **0** | 3 | 0 |
| `XYZZY plugh frotz` | **0** | 4 | 0 |
| `1729 is famous because` | **0** | 5 | 0 |

**Zero hits across five unrelated prompts. On this path the counter does exactly what its name says.**

### 10.2 Chat completions — a constant 24-token phantom hit on every request

`POST /v1/chat/completions`, same class of wholly unrelated prompts:

| prompt | hits | lookups | hit_tokens | delta |
|---|---|---|---|---|
| (before) | 0 | 5 | 0 | |
| `The capital of France is` | 0 | 6 | 0 | +0 |
| `def quicksort(arr):` | **1** | 7 | **24** | **+1 hit, +24 tok** |
| `XYZZY plugh frotz` | **2** | 8 | **48** | **+1 hit, +24 tok** |
| `1729 is famous because` | **3** | 9 | **72** | **+1 hit, +24 tok** |

**+24 hit_tokens exactly, every request, for prompts sharing no content whatsoever.** 24 tokens is the
Qwen chat-template preamble (`<|im_start|>system … <|im_end|><|im_start|>user`). Every chat request
carries it, so `common_prefix_len` finds it every time.

**The counter is not lying — it is answering a question nobody asked.** There genuinely IS a 24-token
common prefix. What there is not is any *work avoided*: per §7, the hit path at `runtime.rs:1017-1024`
returns a hit length **without setting `loaded_prompt_prefix`**, so prefill recomputes the whole
prompt anyway — which is why §9 measured TTFT at **−4.9%** against a **−94%** prediction.

### 10.3 Why this is the demo's most dangerous field

Drive the dashboard with chat completions — **which is what a chat-shaped demo does** — and the hit
rate converges to **~100%** with `hit_tokens` climbing a tidy 24 per request. A visitor reads
*"prefix cache: 100% hit rate, 24 tokens reused"* and concludes compute was saved. **It was not, and
we measured that it was not.** The number is arithmetically defensible and materially false.

**⚠️ `telemetry-provenance.js:527` classifies `metrics.prefix_cache_hits` as
`dynamic: { classification: 'MEASURED' }`.** The `scatter` branch is excellent — `STRUCTURALLY_BYPASSED`
with engine tests cited as evidence. **The `dynamic` branch is the one that ships the falsehood**, and
it does so through a field whose provenance badge asserts it is trustworthy. Recommendation: on
dynamic this must be qualified — the honest label is *"tokens matched, reuse not realised"* — or the
field must be withheld until `loaded_prompt_prefix` is actually set.

**And note the shape: a render guard keyed on `lookups == 0` protects nothing here.** Lookups is 9.
The scatter profile is protected because someone hard-classified it, not because a zero was detected.
