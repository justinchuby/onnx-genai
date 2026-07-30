# QA Test Plan — Serving Dashboard Demo

**Owner:** QA (@fc8b5d97) · **Drafted by:** Secretary (@c0de4c2e) at the Lead's request
**Structure:** Option D — Scenario A on the scatter origin, Scenarios B and C on the dynamic origin.

> **The governing rule for this entire plan.** Every other test suite on this project verifies
> that a thing *works*. This one verifies that a thing *is true*. A panel that renders a
> correctly-computed number under a name that means something else passes every unit test in
> the repo and still ships a lie. **For every displayed field, the tester must open the code
> that increments it and confirm the NAME matches the QUANTITY.** "It's a real number" is not
> a pass.

> **THE EXECUTION STANDARD (binding): "I verified it in a real browser" beats "I read it and it
> looks right."** Four bugs — including a first-paint TDZ crash that meant the page did not render
> at all — were found by driving the real page in a real browser, and were invisible to reading.
> **Nothing in this plan may be signed off from source inspection alone.**

> **AND POINT EVERY SAFEGUARD AT ITSELF.** Four times this session the *safeguard* was where the
> bug lived: the canonical measured-zero example was itself fabricated; the "match the conditions"
> methodology rule could not survive its own noise floor; provenance keyed on field name would
> have called a hardcoded zero a measurement; and a reason string — the thing that *explains*
> honesty to the visitor — was itself dishonest. **Nobody audits the audit.** If a mechanism
> exists to guarantee correctness, assume it is broken until you have watched it run.

---

## 0.1 🔴 CITATION HYGIENE — TWO CHECKOUTS, TWO BRANCHES, DIVERGED

Every `file:line` citation in our docs is ambiguous unless it names the checkout. Measured:

| File | Differing lines between checkouts |
|---|---|
| `crates/onnx-genai-server/src/routes/admin.rs` | **37** |
| `crates/onnx-genai-server/src/metrics.rs` | **33** |
| `crates/onnx-genai-server/src/lib.rs` | **29** |
| `crates/onnx-genai-engine/src/batched.rs` | 0 (identical — engine citations are safe) |

```
/Users/justinc/Documents/GitHub/onnx-genai        branch justinchu/demo
/Users/justinc/Documents/GitHub/onnx-genai-demo   branch feat/genai-demo-dashboard   ← WE SHIP THIS
```

- [ ] **0.1a Re-resolve every server-side line citation against `feat/genai-demo-dashboard`
      before review.** Worked example: the `/metrics` stall is `admin.rs:391-399` in the main
      checkout and **`admin.rs:414-416`** on the shipping branch. **The mechanism is identical;
      only the line numbers move.** A reviewer checking a main-tree citation against the branch
      finds unrelated code and reasonably concludes the claim is wrong.
- [ ] **0.1b This applies to shipped documents, not just notes.** `README.md` and
      `ARCHITECTURE.md` both cite `file:line` **as evidence for honesty claims**. A citation
      that resolves to the wrong line is a broken proof, and it fails in the most damaging
      place — the argument that we verified rather than assumed.
- [ ] **0.1c Cite by symbol where possible** (`GenerationMetrics::start`,
      `prometheus_metrics`) rather than by line. Symbols survive both divergence and drift.

## 0.2 🔴 CAPABILITY GATING IS MODEL-ID INFERENCE, AND IT FAILS OPEN

The ruling preserves "`meta.requires` gating, re-evaluated on origin switch." **There is no
`meta.requires` on disk** — no panel `meta` declares it (the six metas carry `id, title, group,
span, cadence, staleCeilingMs, defaultOpen, acronyms`). The real mechanism is
`panel.modes` in `dashboard/index.js` (`PANELS`), filtered against a `mode` derived from
`SERVER_MODE_BY_CLASS[serverClass]`, where `serverClass` comes from
`selfClassesFromModelId(modelId)` in `scenario-origins.js`.

- [ ] **0.2a The gate keys on WHICH MODEL, not WHAT THE SERVER SUPPORTS.** These are different
      questions and they come apart in one verifiable case: **`--enable-debug-endpoints`.**
      Two servers running the *identical* model differ in real capability depending on that
      flag — with it off, `/v1/debug/kv` and `/v1/debug/config` are gated and the KV,
      prefix-cache and context-length fields cannot be fed. **A model ID cannot observe a
      command-line flag.**
- [ ] **0.2b The unknown-server fallback mounts EVERYTHING.** `dashboard/index.js` returns the
      full `PANELS` list when `mode` is falsy, and `mode` is `undefined` whenever
      `selfClassesFromModelId` doesn't recognise the model ID. So a visitor pointing the page
      at their own server — the exact case the ruling names — **gets every panel, including
      ones their build cannot feed.** `dashboard/prefix-cache.js` has the same default:
      `telemetryStore.capability?.('prefix-cache') ?? { available: true }`.
      **Both fail open, and failing open is indistinguishable from working.**
- [ ] **0.2c Fail CLOSED: absence of a positive capability signal is `unavailable`, not
      assumed-supported.** Detect from what the server actually answers (probe the endpoints /
      read the config response), not from what its model is called.
- [ ] **0.2d The lifecycle half of the ruling is already satisfied for free.** Because the
      switch is a **navigation**, a full page load re-runs detection by construction — there is
      no "re-evaluate on origin switch" code to write, and no stale-capability path to defend.

**Already built and correct — do not re-request:** `scenario-origins.js` implements the whole
navigation switcher (`resolveOrigins`, `planScenario` → `requiresNavigation`, `scenarioHref`,
`currentScenarioId`), consumed by `app.js`. Per-panel `staleCeilingMs` (AC45(c)) ships in every
panel meta.

## 0. Blocking unknowns — resolve before executing §3 onward

These are open as of drafting. Each one changes the expected result of a test, so a tester who
guesses will record a false pass.

- [x] **B1 — Does the page poll a second origin?** ✅ **RESOLVED — NAVIGATION.** Switching
      scenario navigates the browser to the other server's own `/demo`, carrying the scenario in
      the URL (`http://127.0.0.1:8124/demo?scenario=paged-kv`). Both servers serve the same
      static demo via ServeDir, so **every page is always same-origin with the server it talks
      to**. Zero CORS code. Test consequences:
      - **AC42 (no cross-mode leakage) is now true BY CONSTRUCTION** — a full page load cannot
        carry a stale value. Still test it, but a failure here means something is persisting
        state deliberately (storage, URL, cache).
      - **Scenario state lives in the URL query string** → the URL is shareable and bookmarkable.
        **Test deep-linking directly**: paste `?scenario=paged-kv` into a fresh tab and confirm
        it opens on the right panel against the right origin.
        ⚠️ **This step previously deep-linked the withdrawn `prefix-cache` id, and IT COULD NOT
        FAIL.** That id was cut (it is in `CUT_SCENARIOS` with no `id:`), and `currentScenarioId`
        silently falls back to the origin's own scenario rather than erroring. A tester pasting
        it would have seen a flawless, correctly-labelled page and recorded a PASS — **the step's
        success criterion was satisfied BY the defect it should have caught.** Deep-link a
        registered id here; the withdrawn one has its own step below.
      - **Deep-link the WITHDRAWN id** — set the `scenario` query parameter to `prefix-cache` by
        hand and confirm the page does NOT present itself as prefix caching. Substitution is the
        INTENDED behaviour for a link that outlived the cut (a bookmark, a chat log, the design
        doc), so the pass condition is that nothing on screen claims prefix caching — **not**
        that an error appears.
        📌 Written as prose rather than as a pasteable URL on purpose: `scenario-routes.test.js`
        scans this file for advertised routes and cannot distinguish a link we are OFFERING from
        one we are TESTING AGAINST. Spelling the literal here would redden a guard for describing
        the very thing it guards.
      - **Capability detection is RE-EVALUATED ON ORIGIN SWITCH**, not once at page load.
        `meta.requires` gating must re-run. Test switching A→B→A and confirm panels re-mount
        correctly each time, not just on first arrival.
      - All three scenario tabs are **always visible**; the switcher chooses the origin, detection
        decides what mounts against whatever answers.
      - The page must **never** assume capability from a hardcoded scenario map — point it at a
        server whose profile contradicts the tab and confirm it believes the server, not the map.
      - ✅ **VERIFIED AT HEAD BY QA (not inferred): `GET /demo` works — 8/8 `demo_dashboard`
        integration tests pass**, `/demo` → 307 → `/demo/` → 200 `text/html`, modules served as
        `text/javascript`, **path traversal refused in all three encodings** (`../`, `..%2f`,
        `%2e%2e%2f`). HEAD compiles clean with **zero dangling `cors` references**.
      - 📌 **HISTORICAL NOTE so nobody rebuilds it:** a working, hand-rolled CORS layer (212 lines,
        **no new dependency**, loopback-reflecting, preflight-aware, 6 passing tests) was committed
        at **`0ec16375`** and **removed 4 minutes later** by two commits titled `docs(...)` that do
        not mention CORS (`1cc63c08`, `54d8ba5a`). QA measured it working before removal: preflight
        **204** with correct headers, a real cross-origin **POST returning 200**, and
        `localhost.evil.example` / `example.com` correctly **refused**. Removal is *consistent with
        the navigation ruling*, so *do not restore it* — but if navigation ever fails,
        **recover with `git show 0ec16375`; do not write it a second time.**
      - 🔴 **UNVERIFIED AND LOAD-BEARING — the `no-cors` reachability probe (D76,
        `demo-ux.md:2227`).** The navigation design depends on
        `fetch(origin+'/demo', {mode:'no-cors'})` **resolving opaquely** against a live server.
        Only the *rejection* side has been verified. **This cannot be tested with curl or Node**
        (Node ignores `mode`) — it is browser-only, the same invisible-in-the-terminal class as the
        original CORS trap. **Must be asserted in the Playwright harness in BOTH directions:** live
        origin → opaque resolve, dead port → reject. If the resolve side does not hold, the ruled
        architecture has no reachability probe and AC37's actionable error state becomes wrong advice.
- [ ] **B2 — Is `--enable-admin-endpoints` still required?** The VRAM-limit knob it gated is
      dead twice over. If dropped, the launch command, `run-demo.sh`, the README, the UI error
      string and `check-launch-command.test.js` all change together (AC38 requires them
      identical, verbatim).
      **PARTIALLY RESOLVED — `--enable-debug-endpoints` IS required, on evidence.** @376a0297
      measured the prefix counters live, but they are served from `/v1/debug/kv`, which is
      debug-gated; `/v1/status` returns `prefix_hashes: []` and zeros. So **Scenario B silently
      requires the debug flag** — which breaks the property `/v1/status` was chosen to guarantee
      ("headline works on a first run with NO flags"). Either the prefix fields ride into
      `/v1/status`, or `run-demo.sh` and the AC38 command must carry the flag. *Still open:*
      `--enable-admin-endpoints` (the admin one) is a separate question and remains unanswered.
- [ ] **B3 — AC15 per-cell block ownership** needs ~10 engine-crate lines. If not ruled in,
      the KV panel cannot colour cells by owning sequence and §5.2 drops to occupancy only.

---

## 1. Pre-flight — environment and model

- [ ] **1.0 🔴 `models/` IS PER-CHECKOUT, NOT SHARED — AND THE DEMO WORKTREE'S IS EMPTY.**
      `models/` is gitignored, so each worktree has its own. Verified: the main checkout holds
      36 entries; **the demo worktree's `models/` contains only `.hf_cache` and `.scratch`
      (no models at all).** `run-demo.sh` resolves `--model` relative to the repo root it runs
      from, so **launching the demo from the worktree fails with no model, while the identical
      command succeeds from the main checkout.** Confirm which checkout QA is running in before
      diagnosing anything else — this failure looks exactly like a broken model or a broken
      script and is neither.
- [ ] **1.0a Two `models/` directories means two different meanings for `models/qwen2.5-0.5b`.**
      Any bug report, launch command, or measurement must name the **checkout**, not just the
      model. A path alone is ambiguous on this machine.

Failures here masquerade as application bugs. Check them first, every run.

- [ ] **1.1 Correct model directory.** Use `models/qwen2.5-0.5b-scatter-v2`.
      *Trap:* the committed `qwen2.5-0.5b-scatter` does **not** load — root cause is
      `scripts/build_qwen.sh:32` passing `--runtime ort-genai`, producing a model missing
      `model.io.static_*`. If you see a load failure, check the directory name before filing a bug.
- [ ] **1.2 `--model` points at a DIRECTORY, never a config file.** The server has no equivalent
      of the CLI's `resolve_model_dir` (`crates/onnx-genai-cli/src/lib.rs`, `fn resolve_model_dir`). Passing a `.json` fails in a way
      that reads like a corrupt model.
- [ ] **1.3 Wrong-model silent-degradation check.** Loading plain `qwen2.5-0.5b` or `tiny-llm`
      does **not** error — it silently falls back to the per-request path, and the batching panel
      goes flat. **A flat batching panel is more likely the wrong model than a broken panel.**
      Verify the loaded model before filing any batching bug.
- [ ] **1.4 Memory pressure.** Two resident models plus a rebuild can exhaust memory. A
      `-scatter-v2` load failure during concurrent cargo work is far more likely memory pressure
      than a broken model. Re-run in isolation before filing.
- [ ] **1.5 Ports are not hardcoded anywhere.** Grep the JS for `8123`/`8124`. Any literal port
      in client code is a defect regardless of whether the demo runs.
- [ ] **1.6 🔴 THE DEMO ASSETS RESOLVE FROM THE PROCESS CWD, NOT FROM THE BINARY.** The static
      root is a **relative** path, so the documented launch command **404s on `/demo` when run
      from a different directory** while the server otherwise looks perfectly healthy — model
      loaded, `/v1/models` fine, generation fine. Verified by QA: same binary, same flags, one
      directory apart ⇒ 200 vs 404. **Every launch instruction must state the required CWD**, and
      the harness must `cd` explicitly rather than inheriting the operator's shell. This is a
      likely first-five-minutes failure for anyone following the README verbatim.
- [ ] **1.7 Confirm you own the port before you believe any result.** `lsof -nP -iTCP:8123 -sTCP:LISTEN`.
      In a shared checkout a **stale server from an earlier run keeps the port**, the new one dies
      with `Address already in use`, and you are then testing **the old binary while reading the new
      code**. This cost QA a nearly-filed false P1. Applies double after any rebuild.

## 2. Launch — `run-demo.sh`

- [ ] **2.1** `run-demo.sh` starts **both** servers (Option D) and reports both as ready.
- [ ] **2.2** The command printed by `run-demo.sh`, the command in the README, and the command in
      the UI's error text are **byte-identical**. (AC38 — asserted mechanically by
      `check-launch-command.test.js`.)
- [ ] **2.3** Ctrl-C tears down both servers; no orphan process holds a port on re-run.
- [ ] **2.4** `GET /demo` serves the page. *Currently unverified:* `demo_assets.rs` exists but
      has no `mod demo_assets;` declaration, so it is not compiled. Confirm this is wired before
      testing anything downstream — every UI test depends on it.

## 2.5 `GET /demo` — verify on BOTH origins, and know the THIRD gate

The navigation ruling makes same-origin hosting load-bearing on *both* servers. Verified
implemented (`crates/onnx-genai-server/src/lib.rs`, `state.config.demo_assets_dir`; `run-demo.sh:159-169`); these tests keep it that way.

- [ ] **2.5a `GET /demo/` returns the page on `:8123` AND on `:8124`.** Not one. There is no
      "demo server and a spare" — two peers, each hosting the page it talks to. If only one
      serves it, the mode switch navigates into a 404 and the navigation ruling collapses.
- [ ] **2.5b Bare `/demo` redirects to `/demo/`.** `demo_assets.rs:80-81` issues a temporary
      redirect via middleware, because `nest_service("/demo", ..)` already claims the bare path.
      Test the bare form — it is what a human types.
- [ ] **2.5c 🔑 THIRD GATING MECHANISM — a CONFIG PATH, not a flag and not a feature.** Static
      serving is gated on `state.config.demo_assets_dir` (`crates/onnx-genai-server/src/lib.rs`, `state.config.demo_assets_dir`). We now have **three
      independent ways for an endpoint to be absent**, each with a different remediation:
      | Gate | Example | Fix |
      |---|---|---|
      | Runtime flag | `/v1/debug/kv` | pass `--enable-debug-endpoints` |
      | Compile-time feature | `/metrics` | rebuild with the `metrics` feature |
      | **Config path** | `/demo/` | pass `--demo-assets-dir`, or launch from the repo root |
      **"Endpoint missing" is not one error state — it is three, and the advice differs.**
      Confirm the UI never collapses them into a single message.
- [ ] **2.5d Launch from the WRONG working directory** (without `--demo-assets-dir`).
      `resolve_demo_assets_dir` falls back to `./examples/serving-dashboard` **relative to the
      server's CWD**, so this is the likeliest real-world failure. It must serve the
      `missing_assets` explainer (`crates/onnx-genai-server/src/lib.rs`, `demo_assets::missing_assets`), not a bare 404 and not a blank page.
- [ ] **2.5e Ports must stay overridable.** `run-demo.sh:23-24` uses `${SCATTER_PORT:-8123}` /
      `${DYNAMIC_PORT:-8124}`. Run the whole demo on two non-default ports and confirm nothing
      in the page hardcodes 8123/8124.

## 3. Blocking failure states — FIRST-CLASS TESTS, NOT AFTERTHOUGHTS

These are the two states a first-time visitor is most likely to hit, and the demo is judged on
how it behaves here more than on the happy path.

- [ ] **3.1 Server unreachable.** Kill the server, load the page.
      - Page still renders (does not white-screen).
      - Every field is `pending` or `unavailable` — **never `ok` with stale numbers**.
      - The error names the fix: the exact launch command from §2.2.
      - Recovery: restart the server; the page recovers **without a manual reload**.
- [ ] **3.2 Debug endpoints disabled.** Launch without `--enable-debug-endpoints`.
      - **The 404-vs-403 distinction is the whole test.** An unregistered route returns **404**
        (flag missing). A registered-but-disabled route returns **403** (the dead VRAM knob).
        These are trivially confused and produce opposite remediation advice.
      - The UI must tell the visitor *which* flag is missing, not "an error occurred."
- [ ] **3.2a THE FLAG FAILURE IS PARTIAL, NOT TOTAL — AND THAT IS WHAT MAKES IT DANGEROUS.**
      Verified against the router (`lib.rs`), which endpoints the flag actually gates:
      | Endpoint | Demo call sites | Gated by `--enable-debug-endpoints`? |
      |---|---|---|
      | `/v1/status` | 36 | **NO** — registered unconditionally (`crates/onnx-genai-server/src/lib.rs`, `routes::status`) |
      | `/v1/resources` | 7 | **NO** — registered unconditionally (`crates/onnx-genai-server/src/lib.rs`, `routes::resources`) |
      | `/v1/debug/kv` | 6 | **YES** (`crates/onnx-genai-server/src/lib.rs`, `routes::debug_kv`) |
      | `/v1/debug/config` | 3 | **YES** (`crates/onnx-genai-server/src/lib.rs`, `routes::debug_config`) |
      So omitting the flag does **not** break the page. **Roughly 43 of 52 telemetry calls keep
      working**, the dashboard looks healthy, and *only the KV paging panels go dark* — i.e.
      **Pillar 2 silently disappears while the page still looks correct.** Test that the demo
      detects this and says so; a visitor will not infer it.
      - **The specific lie to hunt:** a 404 from `/v1/debug/kv` must NOT render as `unavailable`.
        `unavailable` means *"we plan to measure this"* — a promise. The truth here is
        *"you didn't pass a flag"* — a config error the visitor can fix in five seconds.
        **Rendering a fixable misconfiguration as a roadmap promise is the same failure shape as
        the CORS-vs-server-down misdiagnosis**, and it is worse, because it tells the visitor
        the feature does not exist yet when it is running fine three feet away.
- [ ] **3.2b `--enable-debug-endpoints` is CONFIRMED load-bearing; `--enable-admin-endpoints` is
      CONFIRMED droppable.** The demo makes zero `/v1/admin/*` calls (only prose references at
      `README.md` (the *"deliberately not used"* paragraph) and `check-launch-command.test.js`, `Deliberately excluded`, both correctly *explaining the
      exclusion*). @732c7548's drop of the admin flag is verified correct at source — do not
      re-add it. Both flags were inherited together from the retired `/v1/debug/live` design;
      one survived the reversal on merit, one did not.
- [ ] **3.2c Stale endpoint in a user-visible string.** `telemetry-provenance.js:157` renders the
      reason *"planned for the `/v1/debug/live` `server` block."* **`/v1/debug/live` no longer
      exists** — it is absent from the router and was retired in favour of `/v1/status`
      (`telemetry-contract.md` updated, commit `2208cdd9`). The honesty layer is currently
      promising a visitor a specific endpoint that will 404 if they go looking. Fix the string.
- [ ] **3.3 Partial availability.** One server up, one down (Option D). Scenarios on the live
      origin must work normally; the dead origin's scenarios must degrade per-panel, not blank
      the page.
- [ ] **3.4 Slow/hanging server.** Does the UI distinguish `pending` (never arrived) from `stale`
      (arrived once, now old)? These render differently by design; confirm both are reachable.

## 4. Scenario A — continuous batching (scatter origin)

- [ ] **4.1** Batch occupancy shows a real percentage with `max_batch` as the denominator
      (requires `--max-batch` surfaced).
- [ ] **4.2** The KV panel on this profile is **"static KV decode rows"**, never "pages."
      *Rationale:* continuous batching takes KV out of the pageable pool entirely
      (`batched.rs:101-110` — `ContinuousBatchManager` has no `kv_cache`). Calling rows "pages"
      re-imports the exact confusion the design removes.
- [ ] **4.3** Prefix hit rate renders **`not-applicable`**, not `unavailable`, and not `0%`.
      *Evidence:* `DecodeLoopState::with_rng(0, …)` at `batched.rs:262` and `:486` — the first
      arg IS `prefix_cache_hit_len`, a hardcoded literal. It is a compile-time constant wearing
      the costume of a measurement.
- [ ] **4.4** Preemption counter is `not-applicable` — `crates/onnx-genai-engine/src/batched.rs`,
      `PreemptionPolicy::Disabled` is set structurally, per the comment directly above it.
- [ ] **4.5** `active_sessions` — **fire 4 concurrent requests and expect the panel to read 0.**
      It counts persistent `X-Session-Id` sessions, not concurrent requests. At the busiest moment
      of Scenario A this panel is empty and *correct*. Either the label changes or the field goes.
      **A tester who "fixes" this has introduced a bug.**

## 5. Scenario B — paged KV (dynamic origin)

- [ ] **5.1 🔴 CORRECTED BY MEASUREMENT — THE PREFIX COUNTERS ARE NOT GENUINE ON THE DYNAMIC
      PROFILE EITHER. Do not test for "a real hit rate" here; test that it is NOT displayed.**
      This item previously read "prefix hit rate is a genuine measurement here." **That is false,
      and a tester who verifies the old wording will record a false pass on a fabricated number.**
      *Measured (@fc8b5d97, live dynamic server):* two prompts sharing **nothing** with anything
      previously sent each incremented `prefix_cache_hits` by exactly 1 — `15/16 → 16/17 → 17/18`.
      Across 12 requests (6 repeated prefix + 6 deliberately unique) the counter gained **+12 hits,
      `hit_rate` 0.9375**. **Every completed generation scores a hit.**
      *Root cause in source:* `prepare_session_prefix` (`crates/onnx-genai-engine/src/engine/runtime.rs`, `fn prepare_session_prefix`) forks on
      `uses_token_prefix_cache()` (`decode/state.rs:206`). The token-prefix branch
      (`crates/onnx-genai-engine/src/engine/runtime.rs`, `loaded_prompt_prefix`) computes a hit length and **never sets `loaded_prompt_prefix`**, so
      twenty lines later the *full* prompt is queued and **prefill recomputes every token**. The
      value is `common_prefix_len(...).filter(|&len| len > 0).max()`, so **any single shared leading
      token scores a hit** — and every `/v1/chat/completions` request shares the chat-template
      preamble. The same file states the correct rule for the connector path 30 lines below:
      *"never claiming a hit we can't serve"* (`crates/onnx-genai-engine/src/engine/runtime.rs`, `loaded_prompt_prefix = materialized_len`).
      ⇒ **The counters are broken in OPPOSITE directions on the two profiles** — a hardcoded
      literal `0` on batching (§4.3), and always-hit on dynamic. **`(field, profile)` provenance is
      still the right mechanism; it just has to resolve to NOT-GENUINE on BOTH profiles.**
- [ ] **5.2** Paged block table renders real page occupancy.
- [ ] **5.3 🔴 THE RULE AS RATIFIED IS INSUFFICIENT — IT ONLY CATCHES THE STATIC PROFILE.**
      The binding rule is "`hit_rate` is `unavailable` when `lookups == 0`". On the **dynamic**
      profile `lookups` is **never** 0, so the rule **passes 0.9375 straight through to the UI as a
      genuine measurement — on the exact profile Scenario B demos.** Verify the stronger rule:
      **`prefix_cache_hits` / `hit_rate` are unavailable on BOTH profiles, unconditionally**, until
      a metric exists that counts tokens whose prefill was actually skipped (`loaded_prompt_prefix`),
      not tokens that merely matched. `admin.rs:126-130` emits a literal `0.0`, so "no data" and
      "0% hit rate" are also byte-identical on the wire.
- [ ] **5.4 `prefix_cache_hit_rate` IS ITSELF A MISNAMED FIELD — and it survives on the dynamic
      profile, where everything else is real.** Its denominator is `prefix_cache_lookups`, which
      counts **completed generations**. So the server's `hit_rate` is *hits ÷ generations*, i.e.
      **"the fraction of generations that got a prefix hit"** — a genuinely useful number, but
      **not a cache hit rate**, and it will diverge from one as soon as a generation performs more
      or fewer than one lookup. @376a0297's measured `1 hit / 2 lookups → 0.5` is correct *as a
      per-generation rate* and coincides with a hit rate only because n=2.
      **Required:** either relabel in our UI to what it measures, or don't display it.
      ⚠️ **This item used to end "bind 'is the cache working?' to `prefix_cache_hits_total`, which
      is a genuine hit counter." THAT IS NOW DISPROVEN — see §5.1.** `hits_total` has the *same*
      defect as the denominator: it increments for prompts that share no prefix at all. **Bind
      confidence to NEITHER counter.**
      *This is the standing naming rule applied to the one field where the numbers looked real —
      the hardest case to catch, because nothing looks wrong.*
- [ ] **5.5 Scenario B's headline is measured CLIENT-SIDE** — TTFT delta on a repeated prefix,
      never the server's hit-rate field. The demo drives its own load and knows exactly what it
      sent, so attribution is exact.
      🔴 **DO NOT USE "e2e 1.53s → 1.22s (~20%)" AS A REFERENCE VALUE — IT DID NOT REPLICATE.**
      That figure is n=1 with no control arm, and its "before" is the first request ever on a fresh
      server, which carries one-time warmup (ORT arena growth, lazy init). @fc8b5d97 re-ran the
      exact protocol and added the missing control (req2 with a *different* prefix):
      | Replicate | `repeat` | `unique` (control) |
      |---|---|---|
      | 1 | −5.89% | **−9.45%** |
      | 2 | −39.15% | −16.22% |
      **In replicate 1 the UNIQUE prefix sped up MORE**, while doing more work (611 vs 580 prompt
      tokens). The second request is faster **either way**; the ordering flips between replicates;
      spread ≈33 points. Per the project's own rule (*spread > effect size ⇒ INCONCLUSIVE, not a
      pass*), **Scenario B's payoff is currently UNVERIFIED — neither confirmed nor refuted.**
      **Required protocol before any speedup number is displayed or documented:** quiet machine
      (this was taken at `load average 22.56` on 10 cores, where a **null A/B on a
      byte-identical binary — true delta zero by construction — swung +52.30 % / −40.17 % across
      six pairs**, `perf-baseline.md` §8.1), warm the server first to remove cold-start, **measure TTFT and not e2e** (reuse can
      only shorten *prefill*; e2e buries it under decode), strictly interleaved, **n ≥ 15/arm**,
      report **median + n + CV + 95% CI**. **Overlapping CIs ⇒ cut Scenario B.**
      > ⚠️ **`n ≥ 15` was sized against a noise floor of −9.8 %, which `perf-baseline.md` §6f later
      > RETRACTED as evidence (that run overlapped two CPU-heavy ONNX exports, so its swing has a
      > cause and is not ambient). The clean floor is ~5× larger, so treat `n ≥ 15` as a MINIMUM
      > that has not been re-derived, never as sufficiency.** The criterion that survives the
      > correction untouched is the **CI-overlap** rule, because it is calibrated by the data it is
      > handed rather than by a number written here in advance — and §8.2 is the proof it is
      > needed: on that null run the *mean* passed a ±2 % band while the CI was 17.5× wider than
      > the band. **A tolerance test that reports only a central tendency cannot fail for the right
      > reason.** A genuine
      full-prefix hit should collapse TTFT by ~90% (prefill is ~90% of TTFT, measured), so the
      real effect is far too large to hide in noise on a quiet machine — if it is not obvious,
      it is not there.

      🔒 **THE DECISION THIS PROTOCOL WAS WRITTEN TO MAKE HAS ALREADY BEEN MADE, AND IT SHIPPED.**
      `scenario-origins.js` now carries `'prefix-cache'` in `CUT_SCENARIOS`, keyed with **no
      `id:` field so it cannot be addressed**, giving the reason *"Prefix reuse was measured and
      found absent on both execution paths, so a tab advertising it would promise a capability
      the engine does not have."* That is a **stronger and later finding than the "UNVERIFIED"
      verdict above** — "absent on both paths" is a conclusion, "neither confirmed nor refuted"
      is a pending question. **Do not run the n ≥ 15/arm protocol to decide whether the scenario
      ships. It does not ship, and no result you obtain can change that today.**
      The protocol is retained for exactly one purpose: **falsifying the cut.** If you do observe
      a clean, replicated, non-overlapping TTFT collapse, that contradicts shipping code and is a
      finding worth filing loudly — but it is a *reversal*, not a *gate*.
      ⚠️ Note the shape of this staleness, because it is the failure §11 opens by warning
      against: **this section did not become wrong, it became ANSWERED, and an answered question
      reads exactly like an open one.** Nothing here was ever false. A tester who works top-down
      would have spent an hour on thirty timed requests to re-derive a verdict already frozen in
      an `Object.freeze` — which is the precise cost §11 exists to prevent, reproduced by the
      document that contains §11.

## 6. Scenario C — dynamic origin (paged-KV pressure)

- [ ] **6.1** Executes per spec; all §7 honesty checks apply.
- [ ] **6.2** Eviction/sharing behaviour visible where the design claims it.

### 6.3 🔴 STAGE PRESSURE WITH SESSIONS AND REPEATED PREFIXES — **NOT** CONCURRENCY

Supersedes the earlier "raise concurrency and prompt length" instruction, which is **wrong on
this origin**. The dynamic server **serialises generations**
(`crates/onnx-genai-server/src/driver.rs`, `handle_driver_command` — it takes `&mut Engine` and
runs generation inline under that exclusive borrow): concurrent
requests do **not** overlap, they **queue**.

- [ ] **6.3a Drive the pool with SEQUENTIAL requests** that (i) reuse `X-Session-Id` sessions,
      (ii) share long leading prefixes so the prefix trie has something to reuse, and
      (iii) carry long prompts so blocks are actually allocated.
- [ ] **6.3b Do NOT script a concurrency ramp here, and do not ship a concurrency slider on
      this scenario.** Raising concurrency on the dynamic origin produces a **queue**, not block
      sharing — **the paged-KV panels stay flat and read as broken.** We would be handing a
      visitor a repeatable way to *disprove* a feature that works.
- [ ] **6.3c Concurrency-driven pressure demonstrates something only on the SCATTER origin**,
      which does no paged-KV work at all. **This is the batching ⊥ paged-KV exclusivity biting
      for the fourth time** — the two axes of pressure belong to opposite servers, and each is
      inert on the other.
- [ ] **6.3d Regression guard:** a flat block table under sequential repeated-prefix load is a
      real defect; a flat block table under a concurrency ramp is **the tester staging the wrong
      pressure.** Record which was used, or the result cannot be interpreted.

## 7. Honesty audit — NAME vs QUANTITY (the highest-value section)

Run this against **every** displayed field. Known traps, each of which would have shipped a
confident lie:

| Displayed as | What the code actually counts | Required action |
|---|---|---|
| `vram.used` | KV **byte-budget accounting** — not GPU memory | Must not be labelled "VRAM" |
| `host_ram.used` | **Whole machine**, every other process included | Must not be presented as our footprint |
| `active_sessions` | Persistent `X-Session-Id` sessions, **not** concurrent requests | Relabel or remove |
| `prefix_cache_lookups` | **Completed generations** — increments unconditionally, `crates/onnx-genai-server/src/metrics.rs`, `prefix_cache_lookups` | Never label "cache lookups". Would read 5 with the cache deleted |
| `prefix_cache_hits` (batching) | Hardcoded literal `0` | `not-applicable` |
| `prefix_cache_hits` (**dynamic**) | **Increments on EVERY completed generation, including prompts sharing no prefix** — measured `+12 hits / 12 requests`, 6 of them deliberately unique. `crates/onnx-genai-engine/src/engine/runtime.rs`, `loaded_prompt_prefix` returns a hit length without ever setting `loaded_prompt_prefix`, so prefill still recomputes everything | **Not a hit counter. Unavailable on this profile too** — do not bind, do not cite as proof the cache works |
| `tokens_per_second` (`/v1/status`) | Hardcoded `0.0` | `unavailable` |

- [ ] **7.1** `/v1/status` — **10 of 13 fields are hardcoded to zero** (`admin.rs:53-81`). Only
      `queue_depth`, `active_sessions`, `healthy`, `node_id` are real. **The struct definition in
      `routes/mod.rs` is NOT the handler** — reading the struct and concluding the fields are
      populated is the single easiest mistake to make on this codebase.
- [ ] **7.1a** `sessions[].id` (`admin.rs:69`) is a **fifth genuinely-populated field**, but it is
      **deliberately truncated** — full session ids are bearer credentials. It is *real and
      intentionally partial*. **Rendering it as a complete identifier is its own small lie**;
      either label it as truncated or don't show it. (Caught by @e00032a4; @d7cf9b84's audit
      counts 4 real fields, the true count is 5.)
- [ ] **7.2** `/v1/resources` is **degenerate on the scatter server**: `total_pages: 359128175`
      with 16-byte pages. Confident, precise garbage. **No Scenario A panel may bind to it.**
- [ ] **7.3** No panel renders a stark `0%` anywhere we cannot establish the subsystem was
      *actually consulted*.
- [ ] **7.4** Every field carries `{value, state, source, reason_code}` from the server —
      provenance travels with the data, no client-side stub list.

## 7.5 `/metrics` — REAL data, but two derivation traps

`/metrics` (Prometheus text) carries genuinely measured values that `/v1/status` fabricates.
It is the single biggest honesty upgrade available. But:

- [ ] **7.5.0 🔴 P0 — `/metrics` AND `/v1/resources` BLOCK FOR THE ENTIRE GENERATION. EXPECT THE
      DASHBOARD TO FREEZE EXACTLY WHEN IT MATTERS MOST.** Measured by QA: a poll issued during a
      generation returned after **14.8 s** — the full duration of the in-flight request — instead
      of ~1 ms. Root cause: the handler round-trips a **oneshot command to the decode driver**
      (`admin.rs:396`), and the driver is single-threaded in its decode loop, so it cannot answer
      until the generation completes. **This is not a client bug and no amount of frontend polish
      hides it.** Consequences for testing:
      - A 1 Hz poller will appear **hung**, then deliver a burst of stale-timestamped samples.
        **Do not file this as a frontend defect** — and do not "fix" it by raising the timeout.
      - Any live-updating panel is **dead air during every generation**, i.e. during 100% of the
        demo's interesting moments. Verify the UI degrades honestly (last-known + staleness age)
        rather than rendering stale numbers as current.
      - ⚠️ **Design review, before telemetry is built:** routing new KV/prefix telemetry through a
        `DriverCommand` (e.g. `KvSnapshot`) **inherits this stall and makes it worse**. Telemetry
        must be read from **shared atomics/snapshot state**, never by asking the decode driver.
        Ask QA to re-measure the moment the sink lands.

- [ ] **7.5a `/metrics` is a COMPILE-TIME gate, not a runtime flag.** `crates/onnx-genai-server/src/lib.rs`, `cfg(feature = "metrics")` is
      `#[cfg(feature = "metrics")]`. Default-on (`Cargo.toml: default = ["metrics"]`), so it
      normally works — but if it is ever missing, **no launch flag can fix it.** The remediation
      is *rebuild with the feature*, not *pass a flag*. The demo's error taxonomy currently only
      knows how to say "you're missing a flag." Confirm a 404 on `/metrics` is not reported as a
      flag problem. Distinct from §3.2 — same symptom, unrelated fix.
- [ ] **7.5b 🔴 TTFT MUST NOT BE DERIVED AS `sum / count`.** The histogram is **monotonic and
      never reset** (`crates/onnx-genai-server/src/metrics.rs`, `fn observe` — `count.fetch_add`, `sum_ns.fetch_add`, no decay).
      So `sum/count` is the **all-time mean since process start**, not current TTFT. Two
      consequences, both fatal to a "live" panel:
      1. **It freezes.** After 200 requests, a new slow request moves the displayed value by
         1/200. The number converges and then visibly stops responding — **a live-labelled panel
         that stops moving is worse than an em-dash**, because it reads as a stable measurement.
      2. **It is permanently polluted by the first request**, which includes model warmup. That
         cost never ages out.
      **Correct derivation: differentiate BOTH `sum` and `count` between polls** —
      `Δsum / Δcount` = mean TTFT *over the interval*. Same technique already used for
      tokens/sec; it just has to be applied to the histogram, not only to the counters.
      **Test:** run a slow request after many fast ones. The panel must move. If it barely
      budges, it is the cumulative mean.
- [ ] **7.5c Same defect applies to `onnx_genai_e2e_request_latency_seconds`** — identical
      cumulative structure, identical fix.
- [ ] **7.5d `onnx_genai_batch_size_current` is NOT the engine's batch.** `crates/onnx-genai-server/src/metrics.rs`, `onnx_genai_batch_size_current`
      `fetch_add(1)` in `start()`, decrement in `Drop` (`:145`) — it counts **HTTP generation
      requests in flight**, not `ContinuousBatchManager`'s batch. With max batch 4, firing 8
      concurrent requests makes this read **8** while the true engine batch is **4**. Must be
      labelled `batch.in_flight`; `queued = max(0, in_flight - 4)`. **The engine's true batch
      size stays unavailable — nothing exposes it.** Never render this gauge as "batch size."

## 7.6 MISLABELLED FIELDS — real numbers under false captions

The class every provenance check is blind to: the value is measured, fresh, correctly typed and
genuinely moves. Only the *name* is wrong. Nothing automated catches these.

- [ ] **7.6a 🔴 `ttft` EXCLUDES QUEUE WAIT — and it biases in our favour exactly when the demo
      makes its point.** `GenerationMetrics::start()` (`crates/onnx-genai-server/src/metrics.rs`, `pub(crate) fn start()`) sets
      `started: Instant::now()` **and decrements `REGISTRY.pending` in the same breath** — so the
      clock starts when a request is **admitted**, not when it **arrived**. On the batched path
      `start()` is called at batch admission (`crates/onnx-genai-server/src/driver.rs`,
      `GenerationMetrics::start`). **Time spent queueing is
      therefore invisible to `ttft`.** Under the 4-concurrent scenario — the one that carried
      the now-withdrawn throughput headline — requests queue, so the reported TTFT is **better than the latency a user
      actually experiences**, and the gap *widens* as concurrency rises. This sits directly
      against the ratified rule that the per-stream cost ships beside the aggregate gain.
      **Either label it `time to first token (from admission)` or add the queue wait.** Do not
      ship it as unqualified "TTFT".
      *Confidence: mechanism verified at `crates/onnx-genai-server/src/metrics.rs`,
      `impl GenerationMetrics` + `crates/onnx-genai-server/src/driver.rs`,
      `GenerationMetrics::start`. Confirm the exact
      enqueue point before quoting a magnitude.*
- [ ] **7.6b 🟡 On the FIM path, `ttft` records TOTAL GENERATION TIME, not TTFT.**
      `run_fim_generation` (`crates/onnx-genai-server/src/driver.rs`, `run_fim_generation`)
      calls `start()` then `result()` and **never
      calls `token()`**. `result()` back-fills the missing observation
      (`crates/onnx-genai-server/src/metrics.rs`, `fn result` — `if completion_tokens > 0 { self.token(); }`), so `ttft.observe()`
      fires at the **end** of generation — making TTFT equal e2e latency for those requests.
      **Scope, stated honestly: the demo never issues FIM requests, so this does NOT affect our
      numbers.** Recorded because it silently pollutes the shared histogram if anyone ever does.
- [ ] **7.6c `prefix_cache_hit_rate` is hits ÷ COMPLETED GENERATIONS, not hits ÷ lookups.**
      `crates/onnx-genai-server/src/metrics.rs`, `prefix_cache_lookups` increments the denominator unconditionally on every completed
      generation, with no predicate. Live, plausible, correctly typed, **and it moves when you
      exercise the cache** — so every check we own passes it. See §5.4.
      🔴 **AND THE NUMERATOR IS BROKEN TOO — so this field is not "a real number under a wrong
      name", it is a wrong number under a wrong name.** `prefix_cache_hits` increments on every
      completed generation as well, including prompts sharing nothing (measured: `+12 hits / 12
      requests`, 6 unique). **Both halves of the ratio are unconditional counters, so the ratio
      trends to 1.0 regardless of cache behaviour** — QA measured **0.9375**. It "moves when you
      exercise the cache" only because it moves when you do *anything*. See §5.1.
- [ ] **7.6d `batch_size_current` is HTTP requests in flight, not the engine's batch.**
      `fetch_add` in `start()` (`crates/onnx-genai-server/src/metrics.rs`, `pub(crate) fn start()`), decrement in `Drop` (`crates/onnx-genai-server/src/metrics.rs`, `impl Drop for GenerationMetrics`). See §7.5d.

## 8. Five-state enum rendering

Enum: `ok | pending | stale | unavailable | not-applicable`.

- [ ] **8.1** All five states are visually distinct: pending=blank, ok=live, stale=dimmed-live,
      unavailable=hatched, not-applicable=explained.
- [ ] **8.2 The gate the designer set for themselves:** **screenshot in grayscale.** If
      `unavailable` (hatched) and `not-applicable` (explained) are indistinguishable, the fifth
      state has failed and must be collapsed or redesigned. This is also AC25 — colour must never
      be the sole carrier; verify the colour+pattern+shape triple.
- [ ] **8.3** `unavailable` vs `not-applicable` assert **opposite** things and must never be
      swapped: `unavailable` is a **promise** (someone will do this work); `not-applicable` is an
      **architectural fact** (this path never consults that subsystem). Spot-check every
      em-dashed field for which one it should be.
- [ ] **8.4 Derivation contagion.** A derived value computed from a `not-applicable` input must
      be `not-applicable` — not `unavailable`. Confirm the precedence order is implemented and
      documented, not inferred. *This is the gap most likely to surface as a wrong badge.*
- [ ] **8.5** Wire name is `measured`/`ok` consistently between server and client — confirm one
      spelling, not two.
- [ ] **8.6 AC42 — no cross-mode leakage.** Switch scenarios and confirm no value from the
      previous profile survives the transition, even for one frame.
- [ ] **8.7 AC41 — source attribution** present on every field.

## 9. Presentation integrity

- [ ] **9.1 Every stylesheet linked from `index.html` exists, and every `.css` on disk is
      referenced exactly once.** *This is a live defect at time of drafting:* `styles/panels.css`
      (21KB) is referenced by nothing, so panels render unstyled — silently, with no 404. Both
      devs' test suites pass, because `stylesheet.test.js` loads the file **by path**. A test that
      imports an artifact directly can never observe that the page fails to wire it.
- [ ] **9.2** Exactly one `tokens.css`. No stylesheet defines a raw colour, spacing value or font
      size outside it.
- [ ] **9.3** The prefix-cache panel **ships unconditionally**, showing whatever is true —
      including 0%. Only the *scenario* is cuttable. A panel quietly dropped because its numbers
      are unflattering is the one genuinely dishonest move available on this project.
- [ ] **9.4** Profile banner names **both** halves ("batching LIVE · paged KV bypassed by design").
      Naming only the absent half reads as breakage.
- [ ] **9.5** No scenario tab renders disabled — filtered out entirely or fully working.

## 9.5 STATE TRANSITIONS — test the EDGES, not the states

**@bb2ee824 found a bug that existed ONLY in a transition:** after going no-model → unreachable,
hovering `queue.depth` reported *"the server has no model loaded"* — an unavailable field
**inheriting a reason from an unrelated earlier state**. A confident, specific, wrong explanation,
living inside the mechanism built to prevent exactly that. **No static check could see it. Every
individual state was correct.**

**Rule: every check in §3 and §8 must be run as a TRANSITION, not a snapshot.** Reach each state
*from a different prior state* and confirm nothing is inherited.

- [ ] **9.5.1** healthy → unreachable → healthy. Full auto-recovery, **no manual refresh**.
- [ ] **9.5.2** no-model → unreachable. **Check the hover reason on every field**, not just the
      one that changed. This is the exact bug above.
- [ ] **9.5.3** unreachable → no-model → healthy (three-step). Reasons re-derived each time.
- [ ] **9.5.4** measured → stale → measured. The value must visibly change appearance and change back.
- [ ] **9.5.5** Profile S → Profile D via origin navigation → back. Panels re-mount; no value,
      reason, or provenance survives the crossing (AC42).
- [ ] **9.5.6** Any state → unknown/garbage state. Must fail closed (`—` + `console.error`),
      **never fall through to rendering the value**.
- [ ] **9.5.7** Backoff behaviour: a 500/404 endpoint must **not** be re-requested at 4 Hz forever.
      Confirm it backs off and replays the cached failure. Watch the console error count over
      ~12 polls; it should be single digits, not dozens.

**Every reason string must be re-derived per evaluation, never carried forward.** A stale reason
is worse than no reason, because the visitor believes it.

---

## 10. Regression and baseline

- [ ] **10.1 🔴 AC33 PERF COMPARISON — READ `perf-baseline.md` §5, §6, AND §6c BEFORE RUNNING
      ANYTHING. THE OBVIOUS PROTOCOL GIVES A FALSE VERDICT.**
      Mandatory reading, not a pointer to the broadcast log: §5 acceptance spec, §6
      threats-to-validity, **§6c protocol defect**.
      - **The BEFORE arm is a preserved binary, not a rebuild.** `clean-binary/
        onnx-genai-server-clean-d49d3c8` (SHA-256 `d49d3c8f…`, boot-tested). No code freeze is
        needed and none should be requested. Rebuilding `f55e459b` to recreate the arm is the
        wrong move — you would be comparing a different toolchain state.
      - **512-token generations, 15 samples/arm, decode-only tok/s from SSE per-token
        timestamps.** At 128 tokens CV is 4.95%; at 512 it is 1.98%. Do not shorten.
      - 🔴 **ONE RUN PER ARM CANNOT SETTLE `<2%`, NO MATTER THE SAMPLE COUNT.** Two runs of the
        **byte-identical** binary differ by **−2.07%** at the mean — the criterion fails against
        itself. Within-run CV (1.98%) is the wrong variance component; the criterion depends on
        **dispersion of run means**, which n-per-run does not touch.
      - ✅ **Required: alternate binaries RUN BY RUN — A B A B A … ≥5 runs per arm.** Unit of
        analysis is the **run MEDIAN** (not mean — see below), compared by exact permutation on
        run labels. **VALIDATED:** a null test of the clean binary against itself (10 runs, 55 min,
        `raw/qa-runlevel-null.json`) produced a naive **+6.23%** delta from pure noise, and the
        permutation test correctly returned **p = 0.643, indistinguishable.** The decision rule
        caught what a point estimate would have shipped as a 3×-over-budget regression.
      - 🔴 **NEVER report a bare delta of run means.** One pathological run (CV 37.95%, samples
        down to 9.6 tok/s) moved the estimate from **+1.27% (medians)** to **+6.23% (means)** on
        identical data.
      - 🔴 **"CI straddles 0" IS NOT "passes `<2%`". This is the most likely way this criterion
        gets falsely certified.** `<2%` is an **equivalence** claim; a non-significant difference
        test only says *we could not detect a difference*. The null test's CI was
        **[−5.55%, +28.08%]** — a +20% regression fits inside it. **Correct rule: the ENTIRE 95%
        CI of the difference must lie inside ±2%.** Outcome is three-state —
        **REGRESSION / EQUIVALENT / UNRESOLVED** — and **`UNRESOLVED` is not a pass.**
      - ✅ **This is how the "can't certify a quiet machine" problem is solved:** don't gate on
        quiet in advance (unverifiable). **Let the CI width certify the window retrospectively** —
        contention inflates variance, inflates the CI, and forces `UNRESOLVED`. A noisy window
        can no longer produce a confident answer. Re-running on `UNRESOLVED` is legitimate;
        dropping individual runs for "looking bad" is not.
      - **Runs needed per arm ≈ (between-run CV / 1.02)²:** ~4 on a quiet machine (~25 min),
        **~20** at the 4.58% CV measured here (~7 h). Run it when the crew is idle.
      - ❌ **Do NOT gate on a "quiet machine" and do NOT interleave per-request.** Measured:
        `loadavg` has **no** correlation with our throughput (ρ = +0.079, p = 0.785), so no
        instrument certifies a window as quiet — gating on it converts an acknowledged threat
        into an invisible assumption. And per-request pairing **doubles** delta variance (paired
        stdev 7.01 pp) because per-request jitter is high-frequency. **Interleave at the
        granularity that matches the noise you are fighting, and average below it.**
      - Drift direction **flips** between runs (+2.24%, +3.07%, −5.05%), so it is autocorrelated
        wander on the ~8-minute timescale of a run — not a correctable fixed bias.
- [ ] **10.1a** Any tok/s displayed by the dashboard is **client-derived** from SSE token
      timestamps. `/v1/status.tokens_per_second` is a hardcoded `0.0` (§7.1) and must never be
      the source. Verify the number on screen moves when generation speed moves.
- [ ] **10.2** `./examples/serving-dashboard/run-tests.sh` exits 0 with zero dependencies; no
      Vite/TS/npm build step introduced. **Use the runner, not a bare `node --test`** — a bare
      invocation can pass while running a subset, and a glob form can pass while running nothing.
      The acceptance signal is the runner's `PASS:` line, which is emitted only after the
      discovered-file / executed-suite reconciliation and the untracked-file check also pass.
- [ ] **10.3** Full run from a clean clone by someone who has not read this thread — the real
      acceptance test.

---

## 11. KNOWN-ABSENT — do NOT file these as defects

The most expensive thing a tester can do on this project is spend an hour proving something is
broken that we already know is not built. Each item below is verified absent, with the reason.
**If you find one of these, the correct action is to confirm it degrades honestly — not to file it.**

- **11.1 Scenario C's block-table grid has no data source.** The server no longer publishes ANY
  page-table field: `/v1/debug/kv` carries nine keys and none of them describes page usage
  (`fixtures/captures/{scatter,dynamic}.json`, recorded from live origins). It once answered
  `engine_kv_introspection: "unavailable: ..."`, and that string is quoted in older reports;
  the server retired it at `17d1f895` when it replaced stale "not yet implemented" claims with
  structured `FieldUnavailable::pending` entries. **The absence is now total rather than
  self-declared, which makes this item MORE true, not less** -- but a tester grepping for the
  old string will find nothing and must not read that as the grid having gained a source.
  `SequenceUsage` (`crates/onnx-genai-kv/src/page_table.rs`, `SequenceUsage`) consumes
  `Vec<PageId>` into a length, and the raw map
  is behind a private `Engine.kv_cache`. **Test that the panel says so honestly; do not test the grid.**
  *(Scenario B is NOT in this category — its block table has its own source and IS in scope.
  Do not take that on anyone's word; re-derive it, because a wrong exclusion here costs a defect
  rather than a tester's time:*
  `curl -s -o /tmp/b.json -w '%{http_code}\n' http://127.0.0.1:8124/v1/debug/kv/blocks && grep -o '"applicable":[a-z]*\|"code":"[a-z-]*"' /tmp/b.json`
  *`200` + `"applicable":true` → this exclusion holds. `200` + `"applicable":false` with
  `"code":"not-applicable"` → the exclusion is stale and 11.1 governs B too. `200` +
  `"applicable":false` with `"code":"pending"` → the driver has not finished selecting a decode
  path; poll again rather than concluding either way. **Both of those last two report
  `applicable:false`, so that flag alone does not distinguish them** —   `pending` is built by mutating a `not_applicable` response
  (`crates/onnx-genai-server/src/routes/mod.rs`, `impl BlockTableResponse`). `404` → the server was launched without
  `--enable-debug-endpoints`, so the whole debug router is unregistered
  (`crates/onnx-genai-server/src/lib.rs`, `enable_debug_endpoints`); that is the **instrument**
  missing, not the data, and it answers nothing.*
  ***Use `/blocks`.*** *Plain `/v1/debug/kv` cannot answer this — it is the nine-key response
  described two lines above, carrying prefix counters and a `block_table_endpoint` pointer but no
  page usage. It returns healthy-looking JSON and would read as a confirmation.)*
  ***Use `/blocks`.*** *Plain `/v1/debug/kv` cannot answer this — it is the nine-key response
  described two lines above, carrying prefix counters and a `block_table_endpoint` pointer but no
  page usage. It returns healthy-looking JSON and would read as a confirmation.)*
- **11.2 `tokens_per_second` and `batch_utilization` on `/v1/status` are stubs** — literal `0.0`.
  The dashboard derives tok/s client-side. A zero here is a stub, not a measurement.
- **11.3 `/v1/debug/kv` returns the literal string `"unavailable"`** on the scatter profile.
- **11.4 The runtime VRAM-limit knob.** `crates/onnx-genai-engine/src/config.rs`,
  `allow_runtime_override` is `false` — but read the mechanism before trusting this exclusion:
  the `false` is the **`Default` impl value, not a hardcode**. `EngineConfig::from_yaml` assigns
  it from `serving.memory.limits.allow_runtime_override`, so YAML can turn it on.
  ⚠️ **Whether the demo's launch path ever calls `from_yaml` is UNVERIFIED, so the "it 403s" half
  of this exclusion is unconfirmed — if you get a 403, that is consistent; if you do NOT, this
  item is wrong rather than you having found a bug. Confirm before filing either way.**
  The second leg IS verified: `ByteBudget::reconfigure`
  (`crates/onnx-genai-scheduler/src/byte_budget.rs`, `reconfigure`) sets `state.limit` but never
  touches `state.used` — the repo's own
  test is named `reconfigure_lower_reports_overage_without_evicting`. **To fill the KV pool
  honestly, use sequential requests with repeated prefixes, persistent sessions and long
  prompts — see §6.3. NOT concurrency: the dynamic server serialises generations
  (`crates/onnx-genai-server/src/driver.rs`, `handle_driver_command`), so a concurrency ramp
  fills a queue, not the pool.**
- **11.5 Preemption is structurally disabled on the batching path** —
  `crates/onnx-genai-engine/src/batched.rs`, `PreemptionPolicy::Disabled` (see the comment
  directly above the assignment). The preemption counter is
  `not-applicable` on Profile S, not broken.
- **11.6 `/v1/resources` on scatter reports `total_pages: 359128175` with 16-byte pages.**
  Precise, confident garbage. No Scenario A panel may bind to it.

**Prefer `/metrics` wherever it carries the number** — it is healthy and trustworthy (TTFT
histogram and token counters all live), unlike `/v1/status`.

---

## Exit criteria

1. §3 (both blocking failure states) passes — these are judged before the happy path.
2. §7 honesty audit complete for **every** displayed field, with the incrementing code opened
   and the name confirmed against the quantity.
3. §8.2 grayscale screenshot gate passes.
4. §9.1 passes — no orphaned stylesheet.
5. Zero fields rendering a number whose name does not match what the code measures.
6. §5.1/§5.3/§5.4 settled — **neither `prefix_cache_hits` nor `prefix_cache_hit_rate` is
   displayed as a measurement on EITHER profile.** The ratified `lookups == 0` rule is not
   sufficient on its own: it does not fire on the dynamic profile, which is the one Scenario B
   demos. The panel still **ships** (§9.3) — it renders `not-applicable` on batching and
   `unavailable` on dynamic, which is the honest reading of a counter that scores a hit for
   every generation.
7. **§5.5 settled — and it settled by CUTTING.** The criterion was "a replicated TTFT effect
   under the §5.5 protocol (quiet machine, warm server, interleaved, n ≥ 15/arm, non-overlapping
   95% CIs) **or** the scenario is cut." The ~20% e2e figure did not replicate and must not ship
   as a claim, and **the cut has landed in shipping code** — `'prefix-cache'` sits in
   `CUT_SCENARIOS` (`scenario-origins.js`) with no `id:`, so it is not addressable.
   **Discharge this criterion by confirming the cut is present and the panel still degrades
   honestly — NOT by re-running the protocol.** Per the 🔒 ruling, cutting the SCENARIO is
   permitted; cutting the PANEL is not, and the panel does still ship.
8. Every item in §11 confirmed to **degrade honestly**, rather than confirmed absent.
