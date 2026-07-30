<!-- Copyright (c) Microsoft Corporation. -->

# Serving dashboard

A live, browser-based demonstration of three things the onnx-genai runtime
actually does: **continuous batching**, **paged KV block allocation**, and
**prefix caching**.

Every number on the page is measured on your machine, while you watch. Nothing
is simulated, pre-recorded, or hardcoded — including the comparison baselines.
Where the runtime cannot measure something, the page says so in that field's
place rather than showing you a zero.

**One constraint, stated here rather than discovered later:** batching and paged
KV run on **different execution paths** in this runtime today, so they cannot be
observed in the same process. The demo therefore runs **two servers** and shows
you both — including exactly why they are separate, which turns out to be the
most interesting thing on the page. A caveat you find for yourself discounts
everything you read above it, so this one goes first.

```
./examples/serving-dashboard/run-demo.sh
```

Then open **<http://127.0.0.1:8123/demo>**.

If that worked, skip to [What you are looking at](#what-you-are-looking-at). If
it did not, it is almost certainly the models — read on.

---

## Prerequisites

| You need | Notes |
|---|---|
| A Rust toolchain | `cargo build --release -p onnx-genai-server` takes ~1 minute warm, considerably longer cold. |
| `curl` | Used only to wait for the servers to become ready. |
| A browser with ES module support | Any current browser. There is nothing to install. |
| **Two model directories** | These do **not** come with the repository. See below. |

### Getting the models

`models/` is gitignored, so a fresh clone contains no models at all. The demo
needs two, and they are not interchangeable — see
[Why two servers](#why-two-servers).

| Directory | Kind | Drives |
|---|---|---|
| `models/qwen2.5-0.5b-scatter-v2` | static cache | continuous batching |
| `models/qwen2.5-0.5b` | dynamic cache | paged KV, prefix caching |

If you already have them somewhere else, point the script at that checkout
instead of copying anything:

```bash
MODELS_DIR=/path/to/onnx-genai/models ./examples/serving-dashboard/run-demo.sh
```

To build the static-cache model from scratch you need **Mobius**, and the way
you install it matters more than it should:

```bash
pip install git+https://github.com/onnxruntime/mobius.git
```

> ⚠️ **Do not `pip install mobius`.** The distribution is named **`mobius-ai`**
> (the *import* name is `mobius`), it is **not published on PyPI**, and an
> unrelated squatter package called `mobius` *is*. So the obvious command
> installs a completely different library and then fails with
> `No module named mobius.__main__` — an error that tells you nothing about the
> real cause. Install from source, from **`onnxruntime/mobius`**; other repo
> paths for it are stale and 404.

```bash
HF_HOME=models/.hf_cache TMPDIR=models/.scratch \
python -m mobius build --model Qwen/Qwen2.5-0.5B-Instruct \
  models/qwen2.5-0.5b-scatter-v2 \
  --dtype f32 --ep default --static-cache --max-seq-len 4096 \
  --runtime onnx-genai
```

> **Three known traps, each of which costs an hour if you hit it blind.**
>
> 1. `scripts/build_qwen.sh` passes `--runtime ort-genai`, which emits only
>    `genai_config.json`. The runtime needs `inference_metadata.yaml` with a
>    `model.io.static_cache` declaration, and rejects the model at load time
>    with a message about *"a TensorScatter static-cache scatter ABI but
>    declares no `model.io.static_cache`"*. Use `--runtime onnx-genai`, as
>    above. The pre-existing `models/qwen2.5-0.5b-scatter` directory has this
>    defect and does not load.
> 2. Even under `--runtime onnx-genai`, the builder omits the `io:` block for
>    static-cache builds, so it has to be appended by hand. The graph follows
>    the same naming convention as the test fixture, so
>    `tests/fixtures/tiny-llm-scatter/inference_metadata.yaml` is a working
>    24-layer template.
> 3. **On macOS, `scripts/build_qwen.sh` aborts before it does anything.** Stock
>    `/bin/bash` is 3.2.57, and under `set -u` bash 3.2 treats an empty array
>    expansion as an unbound variable. The documented no-argument invocation has
>    therefore never worked on a Mac without Homebrew bash 5 on `PATH`. If you
>    write `set -u` bash with arrays, the safe idiom is
>    `${arr[@]+"${arr[@]}"}`.

The demo does no model loading, downloading, conversion, or management of its
own. It is a pure client of a server you start yourself.

---

## Running it

`run-demo.sh` is the supported entry point. It builds the server if needed,
checks both model directories exist before starting anything, starts both
servers, waits for each to answer `/health`, and prints the URL. `Ctrl-C` stops
both.

It is also **the canonical definition of the launch command**. The same string
appears in this README, in the `file://` guard inside `index.html`, and in the
demo's two blocking failure states — and `check-launch-command.test.js` fails
the build if they ever disagree. A command a visitor is told to paste has to be
the command that actually works.

Under the hood it runs, per server:

```bash
ONNX_GENAI_EP=cpu ./target/release/onnx-genai-server \
  --model models/qwen2.5-0.5b-scatter-v2 \
  --model-id qwen-scatter \
  --addr 127.0.0.1:8123 \
  --demo-assets-dir examples/serving-dashboard \
  --enable-debug-endpoints
```

| Flag | Why |
|---|---|
| `--model` | The server takes a model **directory**, not a config file. Unlike the CLI it does not normalise the path for you, so passing `.../genai_config.json` is a real footgun. |
| `--demo-assets-dir` | Where the dashboard's static files live. Without it the server looks for `./examples/serving-dashboard` **relative to its working directory**, so a server started from anywhere else serves an explanatory error at `/demo` instead of the page. `run-demo.sh` always passes it, so it works from any directory. |
| `--enable-debug-endpoints` | The dashboard polls `/v1/debug/kv` (KV and prefix-cache fields) and `/v1/debug/config` (context length on the model card). Without it those fields correctly degrade to unavailable — nothing breaks, there is just much less to see. |
| `--addr` | Defaults to `127.0.0.1:8080`. The demo uses `:8123` and `:8124` so it does not collide with a server you already have running. |

Overridable via environment: `MODELS_DIR`, `SCATTER_MODEL`, `DYNAMIC_MODEL`,
`SCATTER_PORT`, `DYNAMIC_PORT`, `BIND_HOST`, `ONNX_GENAI_EP`,
`READY_TIMEOUT_SECONDS`.

**No CORS configuration is needed**, even though the page is served by one
server and reads telemetry from two. Loopback origins are always permitted
(`crates/onnx-genai-server/src/cors.rs`), so the two-server demo works out of
the box. `--cors-allow-origin` exists for serving the dashboard from a
non-loopback host; the demo does not use it.

`--enable-admin-endpoints` is **deliberately not used.** The demo never calls
`/v1/admin/*`, and the server ships without authentication, so enabling an
administrative surface for a read-only visualisation would widen the attack
surface for nothing.

Both servers bind loopback by default for the same reason.

---

## Why two servers

This is the most interesting thing on the page, and it is worth reading before
you conclude that half the dashboard is broken.

**Continuous batching and paged KV are mutually exclusive in this runtime
today.** `ContinuousBatchManager` holds a batched decode session, a tokenizer
and its rows — and never touches `engine.kv_cache`
(`crates/onnx-genai-engine/src/engine/batched.rs:101-110`). Static-cache models
use runtime-owned in-place KV buffers, which bypass the page table and the
prefix trie entirely.

Measured in both directions:

| Model | Continuous batching | Paged KV activity |
|---|---|---|
| `qwen2.5-0.5b` (dynamic) | driver **disabled** | pages allocated and freed, real budget |
| `qwen2.5-0.5b-scatter-v2` (static) | **enabled**, `max_batch=4` | no page activity at all |

So a single server could only ever light up half the dashboard, and the dark
half would look like a wiring bug rather than a genuine property of the runtime.
Running both models lets the demo show each capability on the path where it is
real, and say plainly why they are separate.

`/v1/status` is a **node-level** contract with no model dimension — its own
documentation says *"all values are model-agnostic; `node_id` names this node,
never a model."* Multi-model mode gives each model its own engine driver, so one
server's `/v1/status` could only report one engine's numbers or blend two,
leaving you unable to tell which model you were looking at. Two origins make the
attribution unambiguous by construction, and require no additional server code.

**This is a current property of the runtime, not a permanent law.** Nothing in
the design prevents the continuous-batch path from using the paged KV manager;
it simply does not today. Stated as a design tradeoff rather than a defect:

> Continuous batching buys throughput by taking KV out of the pageable pool and
> putting it in fixed in-place rows. Paged KV buys sharing, prefix reuse and
> eviction by keeping KV in a managed page table. This runtime implements both,
> and today they are alternatives rather than a composition.

That sentence sounds abstract until you watch it collect victims. It is one
architectural fact, and this project met it **three separate times, from three
directions**, each time expecting something different:

| What we set out to show | How the exclusivity killed it |
| --- | --- |
| Prefix caching on the batching server | The paged KV cache owns *both* the page table and the prefix trie. Bypass the cache and you bypass the trie — one cause, two symptoms. |
| A `preempted` state in the batching swimlane | `batched.rs:757` hardcodes `PreemptionPolicy::Disabled`, and `:713-717` documents it as structural: a batch owns its KV in physical rows that cannot be swapped out and resumed in place. |
| A memory-pressure knob that shrinks the KV budget | The override moves the accounting *ceiling* only; resident KV is never released, so nothing observable happens. |

Three features, three investigations, one root cause. That is not bad luck —
**it is a single architectural property expressing itself repeatedly**, and
finding it three times from three directions is far stronger evidence than
reading it once in a design document. It is also the most interesting thing this
project learned about its own runtime, which is why it is documented here rather
than quietly worked around.

### The page detects a profile — it does not offer you a mode

You do not choose which capabilities are live. **The loaded model decides**, so
the page detects a **capability profile** from whichever server it is polling
and adapts to it. There is no mode toggle, because a toggle would misrepresent
what you actually control.

| | **Profile S** — static cache (`-scatter-v2`) | **Profile D** — dynamic |
|---|---|---|
| Continuous batching | live, `max_batch=4` | per-request path |
| KV panel | decode **row** occupancy | paged **block** table |
| Prefix caching | `—` unavailable | genuine measurement |
| Scenarios present | batching | paged KV, prefix caching |

A banner names both halves in one sentence — what is live *and* what is bypassed
— because naming only the live half is what leads a visitor to conclude
something is broken. Panels declare the capability they require and are simply
not mounted when it is absent, so nothing on screen is a dead control.

### Switching scenarios navigates to the other server

Scenario tabs are always visible — all three. Selecting one points the page at
that scenario's configured origin, **as a real page navigation**, carrying the
scenario in the URL:

```
http://127.0.0.1:8124/demo/?scenario=prefix-cache
```

Both servers serve the same static demo, so each page is always **same-origin
with the server it is describing**. That is a feature rather than a workaround,
and it buys four things at once:

- Every fetch is same-origin, so a stuck visitor's browser console shows
  something comprehensible instead of an opaque cross-origin failure.
- **The URL bar becomes the label**, and it is the strongest attribution
  available to us: it is browser chrome, rendered by software this project does
  not control and cannot fake. Every other label on the page is drawn by us and
  therefore has to be *trusted*. You cannot misread which server produced a
  number when the address bar says which server you are talking to.
- The demo becomes **shareable and bookmarkable** — a URL captures the scenario.
- No value from the previous server can survive the switch, because the document
  is destroyed. Stale cross-profile numbers are not prevented by discipline;
  they are impossible.

On arrival the page re-detects the capability profile from whichever server
answered. It never assumes capability from the scenario you clicked — so
pointing this at *your own* server shows you what your server can actually do,
not what ours could.

The cost is a full page load per switch, and any client-accumulated series
(tokens/sec history, the swimlane timeline) restarts. Each scenario drives its
own load, so this is a reset rather than a loss.

---

## What you are looking at

### Scenario A — continuous batching *(static-cache server, `:8123`)*

A swimlane timeline, one lane per request: sent → first token → streaming →
complete, all client-observed. Toggle between a **naive static-batching** request
policy (fixed waves of *B*; wave *k+1* is not sent until wave *k* has fully
completed) and **continuous** (fire immediately). Head-of-line blocking shows up
as a visible staircase.

**The lane has four states here, permanently, and that is a fact rather than a
gap.** There is no `preempted` state on this path: `batched.rs:757` hardcodes
`PreemptionPolicy::Disabled`, and the comment above it explains why this is
structural rather than a default — a batch owns its KV in physical decode rows
that cannot be swapped out and resumed in place. The demo labels preemption
*not applicable here* rather than drawing a fifth lane state that never fires,
because **a state that never appears is indistinguishable from a bug.**

**Both sides of that comparison are real requests to a real server.** The
baseline is never simulated. Whatever the delta turns out to be on your machine
is what the page reports — including an unflattering one.

**What the numbers looked like on one machine.** Measured on this repository
before the dashboard existed, CPU execution provider, `qwen2.5-0.5b-scatter-v2`,
512-token generations, median of 15 iterations:

| | median | spread |
|---|---|---|
| Single request, decode | 33.415 tok/s | CV 1.98 % |
| 4 concurrent, aggregate decode | 82.130 tok/s | CV 2.93 % |
| 4 concurrent, wall-clock throughput | 52.506 tok/s | CV 1.93 % |
| Time to first token | 2141 ms | σ 137 ms |

Read that honestly: four concurrent requests produced about **2.46× the
aggregate decode throughput** of one, while **per-stream throughput fell to
about 0.62×** of solo (~20.7 tok/s). Batching does not make any single request
faster — it trades per-stream latency for total throughput. That trade *is* the
lesson of the scenario, and a demo that showed only the 2.46× would be telling
you half of it.

These are one machine's numbers on a CPU, at a generation length chosen because
shorter runs were too noisy to be conclusive (at 128 tokens the coefficient of
variation was 4.95 %, and two runs of an identical binary differed by 1.98 % —
enough to manufacture a false result). They are **not** a performance claim about
onnx-genai, and yours will differ. The page measures your machine; that is the
only number that should persuade you.

### Scenario B — prefix caching *(dynamic profile, `:8124`)*

Reuse across a shared prompt prefix. The cache is a token radix trie, so reuse
happens over the common *prefix*, not merely by appending to a previous
conversation.

This one is measured, not asserted. On a dynamic server, firing an identical
long prefix twice moved `prefix_cache_hits_total` from 0 to 1 and cut end-to-end
latency from 1.53 s to 1.22 s — roughly 20 % faster on the repeat.

The hit rate you see is **derived client-side** from the TTFT delta on a
repeated prefix, not read from the server's `prefix_cache_hit_rate` field. That
field cannot be trusted: its denominator, `prefix_cache_lookups`, increments on
every completed generation whether or not any cache was consulted, so it counts
generations rather than lookups. The demo drives its own load and knows exactly
what it sent, which makes client-side attribution exact rather than merely
adequate.

> **This scenario cannot be driven by concurrency.** The dynamic server
> serialises generations — one engine, one driver thread — so concurrent
> requests queue rather than overlap. Prefix reuse and block sharing are driven
> by *sequential* requests that repeat a prefix. Raising concurrency here shows
> you a queue, not sharing.

### Scenario C — paged KV block table *(dynamic profile, `:8124`)*

The block grid: which blocks each sequence holds, which are shared, and what
gets evicted under pressure.

> It is called the **paged KV block table**, never "paged attention". The
> allocator — allocation, sharing, tiering, materialisation — is real and is
> what you are watching. True paged-attention *kernels* are not implemented in
> this runtime, and the repository's own README says so.

Blocks render **partially filled**, because the last block of a sequence usually
is. That gap is the actual cost of paging, and hiding it would make the picture
prettier and less true.

**Pressure here is real, and there is deliberately no slider.** An earlier design
exposed a control that lowered the KV budget to force eviction. That control is
gone, because it did not work and could not have: the override moves the
accounting *ceiling* only — resident KV is never released, and the repository's
own test says so in its name
(`reconfigure_lower_reports_overage_without_evicting`). A slider that visibly
moved while nothing was evicted would have been a fabricated *interaction*,
which is the same failure as a fabricated number wearing a costume you can drag.

Instead the demo fills the pool the honest way: **more concurrent sequences and
longer prompts, until real allocation genuinely runs out of blocks.** Slower to
reach, harder to stage, and every eviction you see actually happened.

---

## Honesty, and how it is enforced

The runtime is further along than its telemetry. When this demo was written,
**nine of the thirteen top-level fields in `GET /v1/status` were literal
constants** rather than measurements — `kv_usage`, `tokens_per_second` and
`batch_utilization` each returned a hardcoded `0.0` with a `// not yet tracked`
comment beside them in the server source, and `sessions[].state` returned the
string `"unknown"`. Some of those are being plumbed through for real; the
footer table described below is generated from code, so it is always current
even when this paragraph is not.

On the wire, `"kv_pages_used": 0` and `"queue_depth": 0` are the same six
characters of JSON and mean entirely different things: one is a placeholder, the
other is a real and meaningful measurement. No client-side test can separate
them.

So the demo does not try to remember the difference — it makes the difference a
property of the data. Nothing in the page ever receives a bare number. Every
field arrives as:

```js
{ value, state, source, reason, unit, observedAtMs }
```

where `state` is `measured`, `pending`, `stale` or `unavailable`. Panels branch
on `state` before reading `value`. `telemetry-provenance.js` holds the
classification of every field and emits documented zeros as `unavailable` **even
when the response carried a parseable number**, so a panel cannot accidentally
bind to a placeholder.

### A measured zero and a fabricated zero are opposite things

This is the distinction the whole design exists to protect, and it is worth
being concrete about.

- **`prefix_cache_hits: 0` on the batching server is a real measurement.** The
  cache genuinely did not hit, because on the static-cache path there is no
  prefix trie to hit. That zero is *information*. The repository asserts it in
  both directions:
  `crates/onnx-genai-engine/tests/batched_static_decode.rs:53,88` asserts
  `prefix_cache_hit_len == 0` on the batched static path, and
  `crates/onnx-genai-engine/tests/prefix_speedup.rs:50,84` asserts
  `warm.prefix_cache_hit_len > 0` on the dynamic path. Two tests, opposite
  assertions, both passing — that is what makes this a finding rather than a bug.
- **`tokens_per_second: 0.0` is a placeholder.** Nobody measured anything. The
  server records cumulative token totals and never computes a rate, and the
  source says so in a comment beside the literal.

Both are the character `0` in an HTTP 200 response. The page renders the first
as a number at full contrast and the second as `—`, because presenting them the
same way would be the single most misleading thing this demo could do.

### What the model card can and cannot tell you

The header shows which model produced the numbers you are looking at. It is a
good, small illustration of the rule above, because most of what you would want
to know is not reachable:

| Field | Where it comes from |
|---|---|
| Model id | `/v1/models` — ungated, always available. Defaults to the model directory's basename unless `--model-id` is passed. |
| Context length | `/v1/debug/config` — needs `--enable-debug-endpoints`. |
| Pipeline flag | `/v1/debug/config` — same. |
| **Directory path** | **No endpoint exposes it.** Renders `—`. |
| **Execution provider** | **No endpoint exposes it.** Renders `—`. |
| **Quantization, decode backend** | **No endpoint exposes them.** Not shown. |

So the card will tell you the model's *name* but not, today, which execution
provider produced the throughput figure beside it. That is a real limitation and
the card says so with an em-dash rather than guessing from the `ONNX_GENAI_EP`
you happened to set — the server never confirms it acted on that variable, and
printing it back to you would be a fabricated measurement dressed as identity.

### Three kinds of empty

An empty field is not one situation, and your next action differs in each:

| You see | Meaning | What it tells you |
|---|---|---|
| **`—`** *unavailable* | We cannot measure this here. Stubbed, not plumbed, or structurally fabricated on this profile. | The runtime may well do this; the telemetry does not report it. Hover for the specific reason. |
| **greyed number + age** *stale* | We measured it, but the most recent poll did not refresh it. | The number is real but old. The age is shown so you can judge it. |
| **`···`** *pending* | Measurable; no sample has arrived yet. | Wait a moment. This one resolves on its own. |

`pending` and `unavailable` are deliberately different states: telling you to
wait for a number that is never coming would be its own small dishonesty.

**A metric that is meaningless on the running profile is not shown as a zero**,
and the reason is worth knowing, because it is the more interesting design
decision. Such a metric is either explained in place as not applicable to this
profile, or its panel is **not mounted at all** and the profile banner explains
why. A scenario that cannot run is absent rather than disabled, because an
unclickable tab is an invitation to feel excluded.

The one place this could have gone wrong is the KV panel, which has no pages to
count on the static-cache profile. Rather than em-dashing it, the panel is
*redefined* there: it shows **decode row occupancy** — active rows against
`--max-batch` — which is real, measurable, and moves under load. Same component,
different noun, no fabricated numbers, and nothing on screen that looks broken.

So provenance is keyed by **(field, profile)**, not by field name alone. The same
field legitimately has different states on the two profiles — prefix cache hit
rate is a genuine measurement on the dynamic profile and `unavailable` on the
static one, because there it is a hardcoded literal rather than a reading.

The page footer renders the full field-by-field provenance table, generated from
`telemetry-provenance.js` so it cannot drift from the code. A few of the traps
that table exists to prevent:

- **`active_sessions`** counts persistent `X-Session-Id` sessions, not in-flight
  requests. Four concurrent requests show `0` unless the client opted in.
- **`prefix_cache_lookups`** increments on every completed generation, whether or
  not a cache was consulted. It is a real counter of the wrong noun, so the demo
  does not label anything "cache lookups" with it.
- **`vram.used`** is the scheduler's KV byte accounting, not a device query. It
  is labelled "KV bytes reserved", never "GPU memory used".
- **`host_ram`** is whole-machine capacity, including every other process.
  Attributing it to onnx-genai would be a fabrication of attribution.
- **`/v1/resources` page counts are degenerate on the static-cache profile**,
  where the endpoint reports a page size of 16 bytes and `total_pages` in the
  hundreds of millions. That is not a small-but-real number, it is arithmetic
  applied to a subsystem the profile does not use — **confident, precise
  garbage**, and the most dangerous shape a wrong number can take, because
  precision reads as care. The demo does not render page counts on that profile,
  and no example output in this document is taken from it.

---

## No build step — and why that is deliberate

Vanilla ES modules, served statically by the Rust server at `GET /demo`. No
bundler, no TypeScript compilation, no `npm install`, no `node_modules`. Run the
server, open the page; that is the whole setup.

**This deliberately diverges from `examples/diffusion-demo/`, which uses Vite and
TypeScript with a dev-server proxy. That divergence is a decision, not an
oversight — please do not "fix" it by adding a bundler.** The reasoning:

- A proxy means running `npm run dev` before the flagship demo will run at all,
  and adds a stale-`dist` failure mode that presents as mysteriously wrong data.
- Serving the page from the server that produces the data keeps them in step.
  A separately-served dashboard can silently be a build behind the API it is
  describing, which is a bad property for a tool whose entire job is telling you
  the truth about that API.
- The demo's whole claim is *"run the server and look at it."* A build step
  quietly makes that claim false.

We adopt diffusion-demo's *conventions* and decline its *toolchain*: copyright
headers, a README with real measured numbers, loopback-by-default with an opt-in
override, and indexing in `examples/README.md`.

Losing `tsc` is mitigated by small single-responsibility modules, JSDoc types
where they earn their keep, and no clever code. `dashboard/package.json` exists
solely so `node --test` treats these files as the ES modules they already are —
it has no dependencies and nothing to install.

Third-party JavaScript is capped at one CDN charting library, pinned to an exact
version with an SRI hash. The block grid and the swimlane timeline are
hand-rolled canvas, because no charting library does either of them well.

---

## Layout

```
examples/serving-dashboard/
├── run-demo.sh              Canonical launcher. Starts both servers.
├── index.html               Page shell, scenario and panel mount points.
├── app.js                   Boot, connection state, failure states.
├── telemetry-store.js       The single polling loop. Panels never fetch.
├── telemetry-field.js       Field constructors and the read-state-first guards.
├── telemetry-provenance.js  Field-by-field classification. The honesty backstop.
├── CONTRACT.md              The store/panel interface. Read before writing a panel.
├── ui/                      Model card, failure states, launch command.
├── dashboard/               Telemetry panels. One mount() per file.
├── css/                     Design tokens and the page shell.
└── design/                  Design reference. Does not ship.
```

Two seams keep this navigable:

- **`CONTRACT.md`** is the interface between the page shell and the panels. Every
  file in `dashboard/` default-exports `mount(rootElement, telemetryStore)` and
  returns `{ unmount }`.
- **`telemetry-provenance.js`** is the only place that decides whether a field is
  real. When server-side plumbing lands, one classification changes there and the
  affected panels go live with no panel-side edit.

## Tests

```bash
node --test 'examples/serving-dashboard/*.test.js' 'examples/serving-dashboard/dashboard/*.test.js'
```

Node's built-in runner. No dependencies, consistent with having no build step.
The tests that matter most assert that documented zeros can never surface as
measurements, that a genuine `0` still can, and that the launch command has not
drifted between its four appearances.

## Troubleshooting

| Symptom | Cause |
|---|---|
| **The batching timeline is flat — no overlap at all.** | Wrong model. Continuous batching engages *only* on static-cache (`-scatter`) models; on any other model the server silently falls back to the per-request path. Check this before debugging anything else. |
| A model fails to load, mentioning `model.io.static_cache`. | The static-cache build trap above. `models/qwen2.5-0.5b-scatter` has this defect; use `-scatter-v2`. |
| `model directory does not exist`, but the path is obviously there. | You pointed `--model` at a **file** — usually `.../genai_config.json`. The `onnx-genai` CLI silently coerces that to its parent directory; the server does not. Pass the directory. |
| KV and prefix panels show `n/a` on the batching scenario. | Correct and expected — see [Why two servers](#why-two-servers). Those metrics are meaningless on that execution path, which is why they read `n/a` rather than `—`. |
| A panel shows `—` where you expected a number. | That field is not measurable today. Hover it: the reason is specific, and it is never "we forgot". |
| Numbers are greyed out with an age next to them. | The last poll did not land. The values are real but stale; the connection indicator in the header shows the reconnect state. |
| "Opened from disk" blocks the page. | The page was opened as a `file://` URL. It has to be served by the server; open the printed URL. |
| The script says a model directory does not exist. | Models are gitignored. Build them, or set `MODELS_DIR`. |
| Port already in use. | `SCATTER_PORT` / `DYNAMIC_PORT`, or stop the server still holding it. |
| A telemetry endpoint returns **404**. | A missing gate flag — almost always `--enable-debug-endpoints`. An **unregistered route 404s**, so this is what a forgotten flag looks like. |
| A telemetry endpoint returns **403**. | A route that *is* registered but whose feature is disabled server-side. **This is not a missing flag** and adding flags will not fix it. Checking for 403 to detect a closed gate is the common misdiagnosis — the gate you can open is the 404. |

## Accessibility

Meaning is never carried by colour alone — the swimlanes and the block grid pair
every colour with a shape, pattern or label. The palette is colourblind-safe, the
page is keyboard navigable with a sensible focus order, and unavailable fields
expose their explanation to assistive technology rather than only on hover.

## Further reading

This demo visualises the runtime; it does not re-document it. For the runtime
itself see the repository `README.md` and the architecture documentation. For the
demo's internal contracts, `CONTRACT.md` is the place to start.
