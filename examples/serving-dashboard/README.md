<!-- Copyright (c) Microsoft Corporation. -->

# Serving dashboard

A live, browser-based demonstration of two things the onnx-genai runtime
actually does — **continuous batching** and **paged KV block allocation** — and
one thing it turns out **not** to do, measured and published rather than
quietly dropped.

Every number on the page is measured on your machine, while you watch. Nothing
is simulated, pre-recorded, or hardcoded — including the comparison baselines.
Where the runtime cannot measure something, the page says so in that field's
place rather than showing you a zero.

That third result is not an apology. We expected prefix caching to be the
headline, measured it against a control arm that shared nothing from token 0,
and **could not detect any reuse**. What we can say precisely is what the
counter proves: it scores a hit for prompts that share nothing, so it cannot
tell reuse from no-reuse. What we deliberately do **not** yet say is that reuse
is absent — repeated timing runs disagree with each other by more than the
effect they were looking for, so the honest word today is **unverified**. Both
arms and the detection floor ship on screen, because a result you can check is
worth more than one you must take on faith.

**One constraint, stated here rather than discovered later:** batching and paged
KV run on **different execution paths** in this runtime today, so they cannot be
observed in the same process. The demo therefore runs **two servers** and shows
you both — including exactly why they are separate, which turns out to be the
most interesting thing on the page. A caveat you find for yourself discounts
everything you read above it, so this one goes first.

```
./examples/serving-dashboard/run-demo.sh
```

Then open **<http://127.0.0.1:8123/demo/>**.

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
| `models/qwen2.5-0.5b` | dynamic cache | paged KV block allocation |

If you already have them somewhere else, point the script at that checkout
instead of copying anything:

```bash
MODELS_DIR=/path/to/onnx-genai/models ./examples/serving-dashboard/run-demo.sh
```

To build the models from scratch you need **Mobius**, the ONNX exporter, which
lives in a separate repository:

```bash
pip install "git+https://github.com/onnxruntime/mobius"
```

> ⚠️ **Do not `pip install mobius`.** The distribution is named **`mobius-ai`**
> (the *import* name is `mobius`), it is **not published on PyPI**, and an
> unrelated squatter package called `mobius` *is*. So the obvious command does
> not fail — it **succeeds at installing a completely different library**, which
> then fails much later with `No module named mobius.__main__`, an error that
> says nothing about the real cause. Install from source, from
> **`onnxruntime/mobius`**; other repository paths for it are stale and 404.

Then build the two models. `scripts/build_qwen.sh` does the whole job — export,
static-cache metadata, and tokenizer companions:

```bash
# dynamic profile (:8124)
scripts/build_qwen.sh

# static-cache / scatter profile (:8123)
STATIC_CACHE=1 MAX_SEQ_LEN=4096 OUT_DIR=models/qwen2.5-0.5b-scatter-v2 \
  scripts/build_qwen.sh
```

`DRY_RUN=1` prints the exact commands without building, which is the fastest way
to check your environment is wired up before committing to a 2 GB export.

> **`OUT_DIR` is not optional for the scatter build.** Its default is
> `models/qwen2.5-0.5b-scatter`, and a directory of that name **already exists
> and does not load** — it was produced by an older version of this script that
> emitted no `model.io.static_cache` declaration. `run-demo.sh` deliberately
> looks for `-scatter-v2` so a stale artifact can never be mistaken for a fresh
> one.

> **`STATIC_CACHE=1` is required, not optional.** Continuous batching engages
> *only* on static-cache models, so a scatter directory built without it
> produces the demo's quietest failure: the server starts, loads it, and serves
> it, and the batching scenario correctly reports that this path never batches.
> Nothing errors and nothing is fabricated — but an honest `n/a` on Scenario A
> is indistinguishable from continuous batching not existing. `run-demo.sh` now
> reads `inference_metadata.yaml` and refuses to start unless the declaration is
> actually there, because checking the directory *name* cannot detect this: the
> wrong model in a correctly-named directory passes a name check every time.

**Three traps that used to live here are now fixed in the script**, and they are
worth a sentence because you may still meet them in older notes:

| Was | Now |
| --- | --- |
| The script passed `--runtime ort-genai`, producing a model the runtime rejects at load time. | It passes `--runtime onnx-genai` and **verifies the export afterwards**, failing loudly with the repair command rather than emitting a model that dies later. |
| The `io:` block had to be appended by hand from a test fixture. | `scripts/lib/write_static_cache_metadata.py` writes it automatically. |
| On stock macOS `/bin/bash` 3.2.57, the documented no-argument invocation **aborted at line 31** under `set -u` — so it had never worked on any Mac without Homebrew bash 5. | Runs clean on bash 3.2, and is regression-tested under both 3.2 and 5.3. Missing dependencies now produce an actionable message naming the interpreter and the install command. |

That third one is the one worth remembering. It was invisible to everyone
capable of finding it, because anyone who develops this project has bash 5 on
`PATH` — **the failure was reserved exclusively for first-time readers.**

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
# Asset paths are relative to THIS checkout, so start at its root.
cd "$(git rev-parse --show-toplevel)"

# Models are NOT. `models/` is gitignored, so it is empty in a fresh clone and
# in every git worktree -- the models usually live in one primary checkout.
# Point this at whichever directory actually holds them.
MODELS_DIR=/path/to/onnx-genai/models

ONNX_GENAI_EP=cpu ./target/release/onnx-genai-server \
  --model "${MODELS_DIR}/qwen2.5-0.5b-scatter-v2" \
  --model-id qwen-scatter \
  --addr 127.0.0.1:8123 \
  --demo-assets-dir examples/serving-dashboard \
  --enable-debug-endpoints
```

> **The two paths in that command have opposite roots, which is why the `cd`
> alone is not enough.** `--demo-assets-dir` must resolve inside *this* checkout,
> because that is where the assets are. `--model` must resolve inside whichever
> checkout holds the (gitignored, unshared) models. In a plain clone those are
> the same directory and a bare `models/…` works; in a worktree they are not,
> and a bare `models/…` fails with a missing-model error that reads like a
> corrupt download. `run-demo.sh` handles this for you: it falls back to a
> sibling `../onnx-genai/models`, and it selects a candidate by checking that it
> **contains** the model — not merely that `models/` exists, which is always true
> and would defeat the fallback.

Run it from anywhere else and you get the demo's most confusing failure mode:
**the API works perfectly and only `/demo` is broken.** `--model` and
`--demo-assets-dir` are both resolved against the working directory, and
`resolve_demo_assets_dir` (`demo_assets.rs:54-59`) treats a directory that isn't
there as *no assets configured* rather than as an error — so the server boots
happily, `/v1/status` answers, and `/demo` serves an explanatory 404. Nothing in
the log says the word "directory". `run-demo.sh` sidesteps this entirely by
passing an **absolute** path derived from its own location, which is why it
works from any directory.

This also means **`/demo` working on one server is not evidence it works on the
other.** The two are separate processes and can be started from different
working directories, so verify both:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8123/demo/
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8124/demo/
```

| Flag | Why |
|---|---|
| `--model` | The server takes a model **directory**, not a config file. Unlike the CLI it does not normalise the path for you — but as of `cli.rs:145` it now detects the mistake up front and tells you the exact directory to use instead, rather than failing deep in the loader. |
| `--demo-assets-dir` | Where the dashboard's static files live. Without it the server looks for `./examples/serving-dashboard` **relative to its working directory**, so a server started from anywhere else serves an explanatory error at `/demo` instead of the page. `run-demo.sh` always passes it, so it works from any directory. |
| `--models-dir` | **Never used, and worth knowing why.** It loads *every* valid model directory it finds, eagerly. `models/` holds twenty-odd of them, so this means many gigabytes and a very long startup before the demo can serve anything. Each server is pointed at one model with `--model`. |
| `--max-loaded-models` | **Never set.** The default is unlimited; setting it risks evicting a model mid-demo, which presents as a scenario mysteriously going dead. |
| `--enable-debug-endpoints` | The dashboard polls `/v1/debug/kv` (KV and prefix-cache fields) and `/v1/debug/config` (context length on the model card). Without it those fields correctly degrade to unavailable — nothing breaks, there is just much less to see. |

The dashboard polls **`/v1/status`, `/v1/debug/kv` and `/health`**, and
deliberately **not `/metrics` or `/v1/resources`**, even though both expose
genuinely measured data. Those two answer in under a millisecond when the server
is idle and **block for the entire duration of a generation** when it is not —
because each one round-trips a command to the driver thread (`/v1/resources`
awaits `resource_snapshot()` at
`crates/onnx-genai-server/src/routes/admin.rs:265`; `/metrics` does the same at
`:503-508`). Measured during a single 384-token generation: `/v1/status`,
`/v1/debug/kv` and `/health` completed **61 polls at a clean 4 Hz, ~1.8 ms
each**, while `/v1/resources` and `/metrics` completed **5, blocking
14,784 ms**.

**That stall is now fixed on one server and not the other, and the split is the
opposite of convenient.** `ResourceSnapshot` is answered in two different
places:

| path | handler | borrow | behaviour |
|---|---|---|---|
| batching (`:8123`) | `driver.rs:753` `handle_or_defer_during_batch` | **`&Engine`** — shared | answered **inline, during the batch loop**. Fixed. |
| dynamic (`:8124`) | `driver.rs:830` `handle_driver_command` | **`&mut Engine`** — exclusive | generation runs inline under the borrow, so the command channel is not serviced until it finishes. **Still stalls.** |

The shared-vs-exclusive borrow *is* the fix — nothing else differs.

> **The server that still stalls is the only one where the interesting
> telemetry lives.** Paged KV, block tables, page allocation — everything
> Scenarios B and C exist to show — runs on the dynamic origin. So "the hang is
> fixed" is true, and acting on it unqualified would put the demo's most
> important panels on the one path that still freezes. **A partial fix measured
> on the healthy half certifies the broken half.** If you re-measure this,
> measure it *per server*.

Polling either endpoint would freeze the dashboard during exactly the activity
it exists to show, and it would look perfect in every idle test. If you add a
panel, check what your endpoint costs *under load*; the fast path and the slow
path are indistinguishable at rest.
| `--addr` | Defaults to `127.0.0.1:8080`. The demo uses `:8123` and `:8124` so it does not collide with a server you already have running. |

Overridable via environment: `MODELS_DIR`, `SCATTER_MODEL`, `DYNAMIC_MODEL`,
`SCATTER_PORT`, `DYNAMIC_PORT`, `BIND_HOST`, `ONNX_GENAI_EP`,
`READY_TIMEOUT_SECONDS`.

**The demo needs no CORS configuration** — not a permissive default, not a flag.
Not because cross-origin requests could not be authorised, but because the demo
never makes one: each page only ever talks to the server that served it, and
switching to a scenario on the other server *navigates* rather than fetching.
There is nothing to authorise. Whether the server grows a CORS layer for other
callers is a separate question that does not change anything here.

That independence is worth having, because the alternative design fails in a way
almost nobody catches. A cross-origin `GET /v1/status` **is sent, is handled,
and logs a clean `200 OK`** — the browser then discards the response before
JavaScript can read it. A `POST /v1/completions` never leaves the browser at
all: `application/json` plus the `X-Session-Id` header trigger a preflight
`OPTIONS`, which must be answered separately. **Neither reproduces under
`curl`**, which does not implement the same-origin policy, so every endpoint
tests perfectly from a terminal while the page is dead in a browser. A demo that
depends on cross-origin fetches is one header away from that failure at all
times; one that never makes them cannot reach it.

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

**And it is not one endpoint — it is the whole metrics layer.** `Registry`
(`crates/onnx-genai-server/src/metrics.rs:74-89`) is a single `static` struct of
**thirteen** counters and histograms with **no model dimension at all**: a search
for `with_label_values`, `const_label` or any labelling construct in that file
returns **zero**. `ttft`, `e2e`, `completion_tokens`, `pending`, `rejections`,
`batch_size` — every one is process-wide.

> **Two processes mean two registries, so cross-model blending does not get
> mitigated — it ceases to exist.** Every counter acquires a model dimension for
> free, enforced by the operating system rather than by anyone remembering. That
> matters most for the fields we lean on hardest: the headline throughput
> comparison is computed from `ttft` and `completion_tokens`, and on a single
> server those histograms would be **fed by both models at once** — not a slow
> measurement but a blend of two populations with no physical referent.
>
> **The dangerous version of that bug is the one where you fix the labelling.**
> Echo the resolved model id onto each response and a panel can show a correct,
> accurate `model:` badge **beside a number that still sums both models** — the
> request really did go to that model, so a reviewer who checks the provenance
> finds it sound and stops looking. **Provenance that certifies a contaminated
> quantity is worse than no provenance**, because it spends the credibility we
> built the provenance layer to earn.

**This is a current property of the runtime, not a permanent law.** Nothing in
the design prevents the continuous-batch path from using the paged KV manager;
it simply does not today. Stated as a design tradeoff rather than a defect:

> Continuous batching buys throughput by taking KV out of the pageable pool and
> putting it in fixed in-place rows. Paged KV buys sharing and tiering by keeping
> KV in a managed page table — **and it does evict**, by two independent LRU
> mechanisms: `PrefixCache::evict_lru`
> (`crates/onnx-genai-kv/src/prefix_cache.rs:151`), called from
> `crates/onnx-genai-engine/src/pipeline/paged_decode.rs:53`, releases pages back
> to the pool; and `PageTable::evict_lru_hot`
> (`crates/onnx-genai-kv/src/page_table.rs:1254`), called at `:1078`, `:1241` and
> `crates/onnx-genai-kv/src/paged_cache.rs:497`, demotes the least-recently-used
> page from GPU to CPU. This runtime implements both paths, and today they are
> alternatives rather than a composition.

> **This paragraph previously claimed the allocator "computes an eviction order
> but nothing consumes it", and that was wrong.** The claim is true of a
> *different* subsystem — the VRAM byte-budget governor, whose `eviction_order`
> (`crates/onnx-genai-scheduler/src/governor.rs:166`) really is produced and
> then read by nothing outside its own tests. **Two subsystems, both with an
> "eviction order", and the dead one lent its deadness to the live one.**
>
> It is recorded here rather than quietly corrected because of the direction of
> the error. **Every check in this project guards against claiming a capability
> we lack; not one guards against deleting a capability we have.** Understating
> reads as scrupulous, so it attracts no challenge — which makes it the cheaper
> mistake to make and the more expensive one to find. This one entered as an
> honesty fix.

That sentence sounds abstract until you watch it collect victims. It is one
architectural fact, and this project met it **three separate times, from three
directions**, each time expecting something different:

| What we set out to show | How the exclusivity killed it |
| --- | --- |
| Prefix caching on the batching server | The paged KV cache owns *both* the page table and the prefix trie. Bypass the cache and you bypass the trie — one cause, two symptoms. |
| A `preempted` state in the swimlane | Dead on **both** profiles, for four independent reasons — any one sufficient. Batching: `batched.rs:759` hardcodes `PreemptionPolicy::Disabled` (`:713-717` calls it structural — a batch owns its KV in physical rows that cannot be swapped out and resumed in place), and more decisively, `ContinuousBatchManager` (`batched.rs:101-110`) **has no scheduler field at all**, so the component that could preempt is not present. Dynamic: the server enters via the single-request FCFS path, and its driver runs generations serially, so there is never a second sequence to preempt. |
| A memory-pressure knob that shrinks the KV budget | Not merely ineffective — **unreachable**. `EngineConfig::from_yaml` is the only code that can set a KV limit or flip `allow_runtime_override`, and it has **no callers outside its own unit tests**. The server builds its config at `cli.rs:127-133` from two fields plus `..Default::default()`, so `allow_runtime_override` is always `false` (`config.rs:602`). There is no flag, file, or env var that reaches it. |

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
| Prefix caching | `—` not applicable | `—` not applicable |
| Scenarios present | batching | paged KV allocation, block table |

A banner names both halves in one sentence — what is live *and* what is bypassed
— because naming only the live half is what leads a visitor to conclude
something is broken. Panels declare the capability they require and are simply
not mounted when it is absent, so nothing on screen is a dead control.

### Switching scenarios navigates to the other server

Scenario tabs are always visible — all three. Selecting one points the page at
that scenario's configured origin, **as a real page navigation**, carrying the
scenario in the URL:

```
http://127.0.0.1:8124/demo/?scenario=paged-kv
```

Both servers serve the same static demo, so each page is always **same-origin
with the server it is describing**. This was chosen over an in-page toggle, not
settled for, and it buys four things at once:

- Every fetch is same-origin, so a stuck visitor's browser console shows
  something comprehensible instead of an opaque cross-origin failure.
- **The URL bar becomes the label**, and it is the strongest attribution
  available to us: it is browser chrome, rendered by software this project does
  not control and cannot fake. A rendered `Mode B` badge is a *claim the page
  makes about itself*, and claims can be wrong — this demo shipped a bug of
  exactly that shape, where a field's explanation still described a server state
  the page had already left. `http://127.0.0.1:8124/demo/` is not a claim about
  which server the numbers came from; it is the origin every fetch on that page
  will actually use. It cannot drift out of sync with the page's own behaviour,
  because it *is* the page's behaviour.
- The demo becomes **shareable and bookmarkable** — a URL captures the scenario,
  so you can send a colleague the exact view rather than a list of clicks.
- No value from the previous server can survive the switch, because the document
  is destroyed. Stale cross-profile numbers are not prevented by discipline;
  they are impossible.

It is also simply **honest about the architecture**. The page genuinely *is*
talking to a different server. Navigation makes that visible, where a toggle
would imply one system with a view setting — which is the misconception the two
servers exist to correct.

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

**The lane has four states, permanently, on both profiles — and that is a fact
rather than a gap.** There is no `preempted` state anywhere in this demo. On the
batching path, `batched.rs:759` hardcodes `PreemptionPolicy::Disabled` and the
comment above it explains why that is structural rather than a default — a batch
owns its KV in physical decode rows that cannot be swapped out and resumed in
place. That alone would be enough, but the stronger fact is that
`ContinuousBatchManager` (`batched.rs:101-110`) **holds no scheduler at all**:
preemption is not disabled here so much as absent. On the dynamic path the
server enters through the single-request FCFS entry point and its driver runs
generations serially, so there is never a second sequence to preempt.

The demo therefore ships **no preemption counter**, rather than one pinned at
zero. This is the least visible form of the fabricated zero and the most
dangerous: a stub reads as missing, but `0 preemptions` reads as *a healthy
system*, and nothing on screen would distinguish "never happened" from "cannot
happen". Preemption is labelled *not applicable here* instead, because **a state
that never appears is indistinguishable from a bug.**

**Both sides of that comparison are real requests to a real server.** The
baseline is never simulated. Whatever the delta turns out to be on your machine
is what the page reports — including an unflattering one.

#### The demo never says "batch size", and the reason is worth reading

The most convincing wrong number available to this project is
`onnx_genai_batch_size_current`. It is `fetch_add(1)` in
`GenerationMetrics::start()` and decremented on drop (`metrics.rs:115`,
`:156`), so it counts **generations in flight across the whole process** — it
has no connection to `ContinuousBatchManager` and cannot see the engine's real
batch. With `--max-batch 4`, eight concurrent requests make it read **8** while
the engine is decoding **4**.

Every other trap in this project was a static zero, which looks suspicious. This
one is real, live, and moves beautifully under load — and it is named after the
exact thing the demo exists to prove. **A gauge reading 8 labelled *batch size*
would teach the precise opposite of the truth**, which is that eight
simultaneous requests are not eight simultaneous decodes.

So the page reports `batch.in_flight` (measured) and `batch.queued` (derived),
and the engine's true batch size as **unavailable**, because nothing exposes it.

⚠️ **Do not derive queueing from `/v1/status`'s `batch_utilization`.** It is a
genuine measurement — `current_batch_size / effective_batch_capacity()`, where
the capacity is `max_batch.min(max_queue_depth)` (`state.rs:205`), which is
correctly *tighter* than `max_batch` alone. But it is **clamped to `1.0`** by
`batch_utilization` (`admin.rs:80`). The clamp is well-reasoned for a multi-model node, where
in-flight work legitimately exceeds any single batch. It also means the value
saturates at exactly the moment queueing begins, and the overflow — the entire
queueing signal — is discarded before it reaches the wire. `max(0, in_flight -
max_batch)` cannot be recovered from it. Read `current_batch_size` directly.

🔴 **And do not render this pair as `n of m`, even though `m` is a small
integer.** The house rule is that a ratio over a small integer denominator
should show both terms — `3 of 4` rather than `75 %` — because a percentage over
five reachable values invents a resolution the quantity does not have. That rule
is right, and **this field is the exception that proves why it must be checked
against provenance rather than applied on sight**: here the two terms *do not
come from the same scope.*

| term | scope |
|---|---|
| `current_batch_size` (numerator) | the **process-global** registry (`metrics.rs:82`) — sums every generation on the node, across every loaded model and every driver |
| `effective_batch_capacity()` (denominator) | **one config's** batch ceiling (`state.rs:205`) |

**The numerator can therefore legitimately exceed the denominator**, which is
exactly what the clamp in `batch_utilization` (`admin.rs:80`) exists to absorb. Rendered as `n of m`
that produces **`7 of 4`** — visibly absurd — where the percentage form silently
shows `100 %`. **The clamp is not hiding a bug; it is hiding a scope mismatch,
and `n of m` un-hides it.** Both forms are wrong here for the same underlying
reason, and neither is fixable in the client.

> **If you are asked to emit `max_batch` on the wire so the client can render a
> denominator: emit `effective_batch_capacity()` instead, under a name that says
> what it is.** `max_batch` is *not* the number the server divided by. Shipping
> it would put a client-side `3 of 4` beside a server-side `100 %` — two honest
> fields, computed from two different denominators, disagreeing on screen with
> no way for a reader to tell which one is lying. **That is this project's whole
> failure mode: not a wrong value, but a correct value in a wrong relationship.**

On this demo the mismatch is **latent rather than active**, and only because of
the one-model-per-server split described in [Why two servers](#why-two-servers):
with a single driver per process the numerator's scope collapses onto the
denominator's. **That is the two-server ruling paying for itself in a place
nobody chose it for**, and it is worth noticing that the guarantee comes from
the process boundary rather than from anyone remembering this caveat.

**What the numbers looked like on one machine.** Full method, raw per-run
values, hardware, binary and model SHA-256s, and the git state at capture are in
[`perf-baseline.md`](perf-baseline.md) — measured on this repository before the
dashboard existed, CPU execution provider, `qwen2.5-0.5b-scatter-v2`, 512-token
generations:

| | median | spread | n |
|---|---|---|---|
| Single request, decode | 33.415 tok/s | CV 1.98 % | 15 |
| 4 concurrent, aggregate decode | 82.130 tok/s | CV 2.93 % | **4** |
| 4 concurrent, wall-clock throughput | 52.506 tok/s | CV 1.93 % | **4** |
| Time to first token | 2141 ms | σ 137 ms | 15 |

Read that honestly: four concurrent requests produced roughly **2.5×** the
aggregate decode throughput of one — **95 % CI [2.35, 2.59]** — while
**per-stream throughput fell to about 0.62×** of solo (~20.7 tok/s). Batching
does not make any single request faster; it trades per-stream latency for total
throughput. That trade *is* the lesson of the scenario, and a demo showing only
the speedup would be telling you half of it.

> **Why this says `~2.5×` and not `2.46×`, which is what the raw division
> gives.** The concurrent arm is **n = 4**, so the interval around that ratio is
> about **±0.12**. Writing `2.46×` claims the quantity is known to ±0.005 — it is
> known to ±0.12, and **the third significant figure is arithmetic from the
> division, not resolution from the data.** The number is not wrong; the
> *precision* is invented. That is the same defect as rendering a 4-slot batch as
> `75 %`: **a format is a claim about how finely a quantity can be known**, and
> here the claim was 24× finer than the measurement.

**Why the ratio survives the noise floor, when a single number would not.** The
*byte-identical* binary, re-run on the *same* machine 75 minutes later, produced
**30.151 tok/s instead of 33.415 — a 9.8 % drop caused entirely by background
load** (a backup running, an indexer at 91 % CPU). That is nearly five times the
margin anyone would treat as a meaningful regression, and it is why **no absolute
figure in that table should be treated as reproducible.**

The speedup is a different kind of claim, and it is worth being precise about
why it is sturdier rather than waving at the same caveat twice:

| | |
|---|---|
| what the 9.8 % measures | drift of an **absolute** number **across time**, under changing load |
| how the two arms here were captured | **within one 20-minute window**, one binary SHA-256, one model SHA-256, clean tree ([`perf-baseline.md`](perf-baseline.md) §4c) |
| effect size of the speedup | **+147 %** |
| **effect ÷ noise floor** | **≈ 15×** |

Load that moves both arms together largely cancels in a ratio, and an effect
fifteen times the drift is not something drift can manufacture. **So the honest
split is: the speedup is real and roughly two and a half fold; the absolute
tok/s figures are one machine's afternoon.** Applying the cross-time noise floor
to the within-window ratio would be **a guard shaped like the incident rather
than like the fault** — the same error, mirrored, as trusting a single number
because it came out of a careful-looking table.

These are still one machine's numbers on a CPU, at a generation length chosen
because shorter runs were too noisy to be conclusive (at 128 tokens the
coefficient of variation was 4.95 %, and two runs of an identical binary differed
by 1.98 % — enough to manufacture a false result). They are **not** a performance
claim about onnx-genai, and yours will differ. The page measures your machine;
that is the only number that should persuade you. It is also why the demo
re-measures rather than shipping a recorded baseline: **a stored number is a
claim about a machine that no longer exists.**

### Scenario B — paged KV page allocation *(dynamic profile, `:8124`)*

Pages being allocated, shared and freed as sequences run: the block allocator
doing its actual job. This is a **paged-KV** story, not a prefix-reuse one, and
it is fed by `/v1/debug/kv` rather than `/v1/status` — the `kv_*` fields on
`/v1/status` are placeholders (see [`QA-PLAN.md`](QA-PLAN.md) §7.1, which lists
which four of the thirteen are real).

The allocator is genuinely observable: page allocation and release were verified
directly against a running dynamic server, against a pool of ~14,600 pages. That
is the payoff this scenario shows.

> **This scenario used to be "prefix caching", and it was re-scoped after the
> feature failed to survive verification.** The material below is kept because how it
> failed is considerably more instructive than the feature would have been, and
> because a re-scope that leaves no trace is indistinguishable from a feature
> that never existed.

#### Why this is no longer a prefix-caching scenario

⚠️ **Prefix reuse is unverified — and the counter that claims it is provably
not evidence.**

Those are two separate findings with very different strengths, and collapsing
them is how this section was wrong twice.

**The counter finding is airtight, because it involves no timing at all.** On a
warm server, two prompts sharing nothing with anything sent before still scored
a hit each:

| | hits / lookups |
| --- | --- |
| before | 15 / 16 |
| after a unique prompt | 16 / 17 |
| after a second unique prompt | 17 / 18 |

Twelve requests — six repeated, six deliberately unique — produced **+12 hits,
a 0.9375 "hit rate"**. Every completed generation scores a hit, including
prompts differing from token 0. **So `hits 0 → 1` is exactly what an unrelated
prompt produces, and the counter cannot distinguish reuse from no-reuse.** That
is a counting fact: no sample size, no load, no noise.

**The timing finding is weaker than the counter finding, but it is not as weak
as "inconclusive" — and separating those two claims is the whole trick.**
An early check fired one identical long prefix twice, saw the counter move and
latency fall 1.53 s → 1.22 s, and concluded the cache worked — with no control
arm, so warm-up and a cache hit were the same observation. A later controlled
run put the shared-prefix arm **7 % slower** than a control sharing nothing. A
third run, warm and interleaved, put it **17 % faster**.

**Those last two contradict each other, so the instrument cannot resolve a
difference of that size.** The measurements were taken on a machine at load
average 22 on ten cores, where a *byte-identical* binary has been observed to
swing 9.8 %. A control arm rules out warm-up; it does not rule out noise larger
than the signal.

**But the effect we were looking for is not that size, and that is what closes
the question.** A prefill/decode split on the same long prompt put prefill at
**1241 ms of a 1380 ms TTFT — about 90 %** *(measured by QA; reported here, not
re-derived at `file:line` by the author of this document)*. A prefix cache that
actually restored KV would therefore have collapsed TTFT to roughly **140 ms**.

| | magnitude |
|---|---|
| noise floor on this machine | ~10 % |
| smallest difference we can resolve | ~10 % |
| observed shared-vs-control difference | 7 % → **below the floor, unresolvable** |
| **effect a working prefix cache must produce** | **~90 % → 9× the floor** |

> **This is a sensitivity check, and it is the difference between "we did not
> observe it" and "it is not there."** The same run that cannot tell 7 % from
> 17 % can trivially tell 1380 ms from 140 ms. **Reporting a null result without
> establishing that your instrument could have detected the alternative is not a
> measurement, it is an absence of one** — and it is the same error, in the
> mirror, as the uncontrolled n=2 that started this: both hand you the answer
> you went in with.

So the honest reading is **two findings at two strengths**: small timing
differences are unresolvable here and we make no claim about them, while the
large effect that prefix reuse is *supposed* to produce is **ruled out by a test
that demonstrably could have seen it**.

> **We were careful not to resolve this in the modest direction either.**
> Understating a capability feels like the safe way to be wrong, and it is not:
> **an honesty process that only ever ratchets toward understating is not
> calibrated, it is just a differently-biased claim.** The earlier draft of this
> section said "inconclusive" for *both* findings, which undersold the one that
> was actually solid. **The fix was not to soften or harden the section but to
> stop letting the weaker half set the confidence of the stronger half.**

**And the mechanism explains the measurement, which is why this is settled
rather than merely observed.** A null result tells you nothing about *why*.
Reading the code supplies the why, and it predicts every number above. There
are two prefix branches in `prepare_session_prefix`
(`crates/onnx-genai-engine/src/engine/runtime.rs:1020`), and **only one of them
restores anything**:

| | branch | what it does |
|---|---|---|
| **1** | `runtime.rs:1040` — taken when `uses_token_prefix_cache()` | Scans cached token sequences with `common_prefix_len` and keeps the longest overlap. **It never touches the page table and materialises no KV.** No prefill is skipped. |
| **2** | `runtime.rs:1048` — the `else if` | `prefix_cache.lookup_shared(…, &mut page_table)` — the real one. Matches pages and materialises them, so prefill genuinely shrinks. |

**Branch 1 wins first, and it wins for our models.**
`uses_token_prefix_cache()` is `has_runner() || is_windowed()`
(`crates/onnx-genai-engine/src/decode/state.rs:206-208`), so a model with a
decode runner takes branch 1 and **the `else if` is never evaluated at all**.

This predicts every number we measured, which is why it settles the question:

| Observation | Mechanism |
|---|---|
| ~95 % hit rate from the very first request | *any* nonzero overlap counts, and the chat template shares its opening tokens |
| control prompts that differ from token 0 still scored hits | they still share that template prefix |
| **TTFT unchanged (+7 %)** | **branch 1 skips no prefill — there is nothing to speed up** |
| the predicted ~90 % collapse never appeared | branch 2, the only branch that materialises pages, is unreachable |

> **This is a finding about our configuration, not a verdict on the feature.**
> Branch 2 is real, wired and covered by its own tests (`prefix_speedup.rs`). It
> is simply not reachable from either server path as we run them. So the honest
> claim is *"on this path, the code that runs computes a textual overlap and
> restores nothing"* — **not** *"prefix caching does not work"*, which is a
> product-correctness claim that a configuration finding cannot support.
> Cutting the result loses something real; leading with it overclaims.

**And both hit-rate counters are unusable, in opposite directions.** On the
scatter profile the counter records nothing — `prefix_cache_hit_len` is a
hardcoded literal `0` (`batched.rs:262`, `:486`), passed as the *first*
positional argument of a call named `with_rng` and read back 300 lines later, so
the branch that increments hits (`metrics.rs:136`) is **statically dead**. Not
"did not fire" — *cannot*. On the dynamic profile it records *everything*: it
counts any nonzero match, so the few shared tokens of the chat template make it
read ~95 % from the very first request, including for prefixes that differ from
token 0. One counter is pinned low, the other pinned high, and **neither is
measuring prefix reuse.**

The denominator is independently broken, and that is the part worth
generalising. `prefix_cache_lookups` (`crates/onnx-genai-server/src/metrics.rs:136-138`) increments
**unconditionally on every completed generation**, outside any predicate — it
counts generations, not lookups. **It would read 135 with the prefix cache
deleted from the codebase.**

> **The wire has since been made honest and the internal counters have not, so
> what you grep for depends on where you look.** `/v1/debug/kv` now returns
> `generations_completed`, `generations_with_prefix_reuse` and
> `generation_prefix_reuse_rate` — names that say what the numbers actually
> count. The registry behind them is still `prefix_cache_lookups` /
> `prefix_cache_hits` (`metrics.rs:83-86`). **That is the right order to fix it
> in** — the name a visitor can see was wrong in a way the internal one is not,
> because nobody reads an atomic and concludes anything about a cache. It does
> mean a search for "the hit rate field" finds an honest name at the boundary
> and a misleading one two files in.

> **A ratio has two halves and they usually have different provenance. Audit
> them separately.** Ours was a real count over a compile-time constant. The
> obvious guard — *"suppress the rate when `lookups == 0`"* — never fires,
> because the denominator is the half that works: 135 is a true count of true
> generations. **A safeguard derived from an incident tends to be shaped like
> the incident rather than like the fault.**

> **So the demo displays no prefix cache hit rate, in any form, on either
> profile.** A precisely-computed 95 % would have been the most convincing
> number on the page and the least true. This is the failure this project exists
> to catch, caught — and caught *before* the panel was built rather than at
> sign-off.

What remains real and reportable here is the **counter's behaviour**, which is
what the panel shows: a hit scored for prompts that share nothing. That is a
finding about the instrument, and it never depended on timing at all.

**The timing result is no longer unverified — it is explained.** The two
measurements that looked like contradictory noise (+7 % on one run, −17 % on a
warm interleaved run, on a machine at load average 22) are both consistent with
branch 1, which cannot produce a speedup in either direction. **We did not need
a quieter machine; we needed to read the branch.** The wider lesson is worth
more than the result:

> **When a measurement is inconclusive, the next move is not always more
> measurement.** We had specified an n ≥ 15 interleaved re-run on an idle
> box — real work, and it would have produced a tighter confidence interval
> around a number that was never going to move. **Reading the code that
> produces the effect settled in minutes what more samples could not have
> settled at all**, because the samples were all drawn from a path with no
> effect in it. Noise made it *look* like a statistics problem.

> **This scenario cannot be driven by concurrency.** The dynamic server
> serialises generations — one engine, one driver thread — so concurrent
> requests queue rather than overlap. Any prefix reuse would have to come from
> *sequential* requests repeating a prefix. Raising concurrency here shows you a
> queue, not sharing.

### Scenario C — paged KV block table *(dynamic profile, `:8124`)*

The block grid: which blocks each sequence holds, which are shared, and what
happens when the pool runs out.

> It is called the **paged KV block table**, never "paged attention". The
> allocator — allocation, sharing, tiering, materialisation — is real and is
> what you are watching. True paged-attention *kernels* are not implemented in
> this runtime, and the repository's own README says so.

Blocks render **partially filled**, because the last block of a sequence usually
is. That gap is the actual cost of paging, and hiding it would make the picture
prettier and less true.

**The payoff is admission backpressure, not budget reclamation.** The pool
*stops accepting*; the VRAM budget is never *reclaimed*. Under pressure
`queue.depth` climbs, `admission.slots_available` falls to zero and
`admission.rejections` increments — all server-measured, no privileged endpoint,
no engine changes. **No budget-driven eviction occurs, and the demo does not
claim otherwise** — the allocator's own LRU eviction is a separate mechanism and
is live.

That distinction is worth more than it sounds. Eviction is internal housekeeping:
a visitor has to **take our word** for what the animation means. Backpressure is
externally observable — it propagates out to admission and the visitor **feels it
in their own requests slowing down**. We trade an animation they must trust for a
consequence they can verify, which is the trade this whole demo keeps making.

**There is deliberately no KV-budget slider**, and the reason hardened while this
was being written. An earlier design exposed a control that lowered the budget to
force eviction. First it turned out not to work: lowering the limit moves the
accounting *ceiling* only, resident KV is never released, and the repository's own
test says so in its name (`reconfigure_lower_reports_overage_without_evicting`).
Then it turned out the control **cannot succeed**, which is a different and
more interesting claim. There *is* a route — `POST /v1/admin/resources/vram-limit`
(`lib.rs:124`) — and reaching for it is the natural thing to do. It fails at
three independent points, any one of which is sufficient:

1. It is **admin-gated**, and the demo deliberately ships without
   `--enable-admin-endpoints`, so it is a **404**.
2. With the flag, the governor refuses: `governor.rs:168` returns
   `RuntimeOverrideDisabled` unless `allow_runtime_override` is set — a **403**.
3. That flag is **unsettable**. `EngineConfig::from_yaml` is the only code that
   can enable it, and **it has no callers outside its own unit tests**; the
   server assembles its config from two fields plus defaults
   (`cli.rs:127-133`). No CLI flag, config file, or environment variable
   reaches it. (`--models-config` looks like it should, but carries only a list
   of models.)

And even past all three, the code says so itself: `set_vram_limit` carries a
`TODO` noting that the returned eviction order is never executed. **The
*governor* computes a plan and discards it** — its `eviction_order`
(`crates/onnx-genai-scheduler/src/governor.rs:166`) is read by nothing outside
its own tests — which is why the repository's own test is named
`reconfigure_lower_reports_overage_without_evicting`.

> **Say "governor", not "allocator", and the distinction survives being
> repeated.** The paged KV allocator *does* evict, by two live LRU mechanisms
> (see *Why two servers*). It is the VRAM byte-budget governor, a different
> subsystem, whose eviction plan goes unexecuted. **Both own something called an
> "eviction order", and this README has already once let the dead one's
> reputation attach to the live one.** Whenever two subsystems share a noun, the
> defect that follows is not a wrong fact but a true fact filed under the wrong
> owner — and it survives review, because every individual sentence checks out.

So the slider would have been a **fabricated interaction** — the same failure as
a fabricated number, wearing a costume you can drag. Instead the demo fills the
pool the honest way: **a long shared prefix, then sequential requests branching
off it, until allocation genuinely runs out of blocks.** Slower to reach, harder
to stage, and every stall you see actually happened.

> **Concurrency is not the lever here, and that is a fact about this runtime
> rather than a staging preference.** The dynamic server runs generation
> *inline* on the driver thread (`run_fallback_generation`, `driver.rs:852`), so concurrent requests
> **queue rather than overlap** — raising concurrency against the paged-KV
> server adds waiting, not pressure. Concurrency drives Scenario A, on the
> scatter server, which has no block table at all. Pressure on the pool comes
> from **prompt length and sequential branching**, both of which work with a
> single request in flight.

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
{ value, state, source, sourceClass, origin, originModelId,
  label, reason, unit, observedAtMs, derivedFrom, provenanceWarning }
```

where `state` is one of five: **measured**, `pending`, `stale`, `unavailable` or
`not-applicable`. Panels branch on `state` before reading `value`.

> ⚠️ **The measured state's wire value is the string `'measured'`.** Compare
> against the constant `FIELD_STATES.MEASURED`
> (`examples/serving-dashboard/telemetry-field.js`), never against a literal.
> **That constant spelled its value `'ok'` until very recently** — a constant
> named `MEASURED` whose value was `'ok'` — so any code or documentation you
> find comparing `field.state === 'measured'` may predate the rename and may
> have been silently false for every measured field on the page. **A constant's
> name is documentation with no test coverage**: the name was written when the
> value matched, the value changed, and the name kept vouching for it.
> `check-field-states.test.js` reads the value out of the constant and fails if
> this sentence disagrees, so the two cannot drift apart again.
`telemetry-provenance.js` holds the classification of every field and emits
documented zeros as `unavailable` **even when the response carried a parseable
number**, so a panel cannot accidentally bind to a placeholder.

### A measured zero and a fabricated zero are opposite things

This is the distinction the whole design exists to protect. **The example this
section used to give was itself wrong, and correcting it in place is more
useful than replacing it quietly.**

We originally offered `prefix_cache_hits: 0` on the batching server as the
canonical *genuine* zero — the cache truly did not hit, because that path has no
prefix trie. **That reading was wrong.** The batching path does not measure a
miss: it passes a literal `0` as the `prefix_cache_hit_len` argument of
`DecodeLoopState::with_rng` (`crates/onnx-genai-engine/src/batched.rs:262,486`;
the parameter is first in the signature at `decode_loop.rs:39-42`, and at the
call site it reads `with_rng(0, rng, …)`, which is easy to mistake for a seed).
So the zero is not the answer to a question nobody could answer differently —
**it is a constant that was never a measurement at all.** It is also not
`unavailable`, because nobody will ever instrument it: `ContinuousBatchManager`
(`batched.rs:101-110`) has no prefix-cache field at all, so a lookup there is
not merely absent but *impossible*. That is what `not-applicable` is for.

**So our flagship example of a real measurement was a fabricated number, sitting
inside the section that argues this demo does not fabricate numbers.** It is
worth stating plainly, because it is the most instructive thing in this file:
the safeguard is exactly where a defect hides best, since nobody audits the
audit.

- **A genuine measured zero:** `queue_depth: 0`. Something computed it, the
  answer is zero, and it would have been a different number under load. It
  renders at full contrast, because a real zero is *information*.
- **A placeholder:** `tokens_per_second: 0.0`. Nobody measured anything. The
  server records cumulative token totals and never computes a rate, and the
  source says so in a comment beside the literal. It renders `—`.
- **A structural non-answer:** prefix reuse on the batching path. The subsystem
  is never consulted, so there is nothing to measure and never will be. It
  renders `n/a`, which teaches rather than apologises.

All three are the character `0` in an HTTP 200 response. Rendering them the same
way would be the single most misleading thing this demo could do.

> **The naming traps behind these distinctions are maintained in
> [`QA-PLAN.md`](QA-PLAN.md) §7 (displayed name → what the code actually counts
> → required action, with `file:line`) and the known-absent register in §11.**
> Those sections are normative. This file deliberately does not restate the
> table: two copies of a trap list drift, and the stale copy is the one that
> gets believed, because it reads exactly as authoritatively as the live one.

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

### Four kinds of empty

Every number on the page carries its own provenance, in one of **five states** —
`measured`, `pending`, `stale`, `unavailable`, `not-applicable`. Only the first
is a plain number; the other four are the kinds of empty, and your next action
differs in each:

| You see | Meaning | What it tells you |
|---|---|---|
| **`—`** *unavailable* | We cannot measure this here. Stubbed, not plumbed, or structurally fabricated on this profile. | The runtime may well do this; the telemetry does not report it. Hover for the specific reason. |
| **greyed number + age** *stale* | We measured it, but the most recent poll did not refresh it. | The number is real but old. The age is shown **in words**, so you can judge it — and it stops being a number entirely once it is too old (below). |
| **`···`** *pending* | Measurable; no sample has arrived yet. | Wait a moment. This one resolves on its own. |
| **`n/a` + an on-screen caption** *not applicable* | Meaningless on this execution path — the question was never asked. | Not a gap in our work. This is the architecture, and the caption says which part. |

**`not-applicable` is the one state that is not just a different glyph — it
changes scale.** When *every* field in a panel is not-applicable, the panel does
not fill with markers: the header stays, **the body is replaced by the
explanation** (`collapseNotApplicableBody`, `dashboard/panel-kit.js:597`). The
field-level `n/a` survives only for a structurally-pinned field sitting inside
an otherwise-live panel (`panel-kit.js:597`).

That is worth a sentence of *why*, because the first design used an em-dash for
both:

- **An em-dash leads with "nothing here", and readers scan past absence
  glyphs.** A panel of them reads as breakage. Under two servers this state is
  the **normal** case rather than the exception, so a visitor's first run would
  show a dashboard half-covered in apparent failure — rendering our single most
  interesting finding as a bug.
- **It also makes `unavailable` and `not-applicable` impossible to confuse by
  construction rather than by contrast tuning.** One is a field-level marker
  inside a live panel; the other replaces a panel body with prose. **They cannot
  render identically because they do not render at the same scale** — which is a
  far stronger guarantee than picking two glyphs that look sufficiently
  different.

#### Stale values stop being numbers

A greyed number with an age beside it is honest for a few seconds and dishonest
after a few minutes, because **`20.7 tok/s · 4m old` is read by every human as
`20.7 tok/s`.** An age suffix does not stop a number being a number.

So each panel carries a **staleness ceiling** (`dashboard/field-state.js:87`,
per-panel overrides at `panel-kit.js:271`). Past it the **value is removed and
the age remains** — because *why* it disappeared is the useful half. A field
whose age cannot be determined at all counts as past the ceiling immediately: if
we cannot say how old it is, we cannot claim it is fresh.

The block grid is deliberately **exempt from a wall-clock ceiling**. It is
event-sampled on request completion rather than polled, so it legitimately does
not change during a generation, and an elapsed-time rule would mark a
**correct** panel stale. Its staleness test is *no sample since the last
completed request*. A false stale badge on the most technical panel on the page
costs more credibility than a missing one.

`pending` and `unavailable` are deliberately different states: telling you to
wait for a number that is never coming would be its own small dishonesty. And
`not-applicable` is deliberately distinct from `unavailable` — one is our gap,
the other is a fact about the runtime. It is the **only** state whose
explanation is always on screen rather than behind a hover, because a fact
nobody hovers over is a fact nobody learns, and this is the one most worth
reading.

**Staleness is tracked per server, not per panel** — which matters here because
there are two. One server can stall or die while the other stays perfectly
healthy, and the dead half would otherwise keep showing its last good frame
indefinitely. A frozen chart and a saturated one are pixel-identical, so a stall
marks **every** panel fed by that origin, and past a per-panel age ceiling a
stale field stops rendering as a number at all: an unbounded age suffix is still
a number on screen. The ceiling differs by panel because the tolerable ages do —
a 4 Hz sparkline is worthless at three seconds old, while a page-table total is
fine at thirty.

That is the honesty rule arriving through the transport layer rather than the
data: every panel can be individually truthful and the page can still lie,
because correctness was enforced per field while the failure is per connection.

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
- **An uncomputed field is `null` on the wire, never `0`.** This one is a rule
  rather than a trap, and it is the rule that makes the rest of them expire
  correctly. While a stub emits a literal `0.0`, "no data" and "measured zero"
  are *byte-identical* in the response, so every client has to carry a
  hardcoded list of fields it distrusts — a copy of server state that drifts the
  moment the server improves. The failure is guaranteed and silent: **the day a
  field becomes real, a stale distrust-list hides it.** A nullable field cannot
  be accidentally summed, averaged or plotted; a `0.0` silently can.
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
├── styles/                  `tokens.css` (design tokens), `shell.css` (page
                             shell), `panels.css` (panel treatments). One CSS
                             tree, deliberately — two invite a second
                             `tokens.css`, and a forked design system announces
                             itself with no error.
├── QA-PLAN.md               Normative for naming traps and known-absent fields.
└── design/                  Design reference. Does not ship.
                             `demo-ux.md` is normative: it governs what each
                             panel is ALLOWED to render.
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
| `--model expects a model DIRECTORY, but '...' is a file.` | You pointed `--model` at a file — usually `.../genai_config.json`. The `onnx-genai` CLI coerces that to its parent directory; the server does not. The message names the directory to use; pass that. |
| KV and prefix panels show `n/a` on the batching scenario. | Correct and expected — see [Why two servers](#why-two-servers). Those metrics are meaningless on that execution path, which is why they read `n/a` rather than `—`. |
| A panel shows `—` where you expected a number. | That field is not measurable today. Hover it: the reason is specific, and it is never "we forgot". |
| Numbers are greyed out with an age next to them. | The last poll did not land. The values are real but stale; the connection indicator in the header shows the reconnect state. |
| **Every panel on one server freezes while the other keeps updating.** | That server stalled or died. Correct behaviour, not a rendering bug — staleness is marked per origin precisely so half a demo cannot keep showing a confident last frame. |
| "Opened from disk" blocks the page. | The page was opened as a `file://` URL. It has to be served by the server; open the printed URL. |
| The script says a model directory does not exist. | Models are gitignored. Build them, or set `MODELS_DIR`. |
| Port already in use. | `SCATTER_PORT` / `DYNAMIC_PORT`, or stop the server still holding it. |
| A telemetry endpoint returns **404**. | **Three different causes with three different fixes — check which gate before reaching for a flag.** See the table below. |
| A telemetry endpoint returns **403**. | A route that *is* registered but whose feature is disabled server-side. **This is not a missing flag** and adding flags will not fix it. Checking for 403 to detect a closed gate is the common misdiagnosis — the gate you can open is the 404. |

### A 404 is three different bugs

An unregistered route 404s, and **every one of these gates produces the same
404**. Reaching for a flag is right one time in three; the other two send you to
fix something that is not broken.

| Gate | Example | What actually fixes it |
|---|---|---|
| **Runtime flag** | `/v1/debug/kv` | Pass `--enable-debug-endpoints`. |
| **Compile-time feature** | `/metrics` — `#[cfg(feature = "metrics")]` at `crates/onnx-genai-server/src/lib.rs:128` | **Rebuild with the feature on. No flag can help**, and the server will not tell you the difference. (It is on by default, so this bites people who trimmed features.) |
| **Config path** | `/demo/` — gated on `state.config.demo_assets_dir` at `crates/onnx-genai-server/src/lib.rs:92` | Pass `--demo-assets-dir`, or launch from the repo root. The path is resolved **relative to the current working directory**, and a missing directory is treated as *"no assets configured"* rather than as an error — so the server boots happily and only `/demo` is missing. |

The last row is the nastiest, because it is the only one where the server had
the information to warn you and chose not to.

## Accessibility

Meaning is never carried by colour alone — the swimlanes and the block grid pair
every colour with a shape, pattern or label. The palette is colourblind-safe, the
page is keyboard navigable with a sensible focus order, and unavailable fields
expose their explanation to assistive technology rather than only on hover.

## Further reading

This demo visualises the runtime; it does not re-document it. For the runtime
itself see the repository `README.md` and the architecture documentation. For the
demo's internal contracts, `CONTRACT.md` is the place to start.
