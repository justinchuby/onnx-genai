# Telemetry store + panel contract

**Owner:** demo developer (@bb2ee824) · **Consumers:** every panel in `dashboard/`, every scenario module.
**Status:** ratified. Change it by talking to the owner first — two developers build against it in parallel.

This is the seam between the page shell and the panels. If you are writing a
`dashboard/*.js` panel, everything you need is here.

---

## 1. The one rule

> **A panel never receives a number. It receives a `TelemetryField`, and it must
> branch on `field.state` before rendering anything.**

Not a style preference. `GET /v1/status` returns hardcoded `0.0` for `kv_usage`,
`tokens_per_second` and `batch_utilization` — each with a `// not yet tracked`
comment beside it in
`crates/onnx-genai-server/src/routes/admin.rs:54-64`. Those are real numbers in a
real HTTP 200 response that nobody measured.

If the store handed out bare numbers, `0.0` from `tokens_per_second` and `0.0`
from an idle server would be the same value, and "never render a fabricated
number" would be an instruction nobody could actually follow. The envelope makes
the difference a property of the data, so the rule can be obeyed mechanically
instead of remembered.

`telemetry-provenance.js` enforces it: fields classified `DOCUMENTED_ZERO`,
`NOT_PLUMBED` or `MISATTRIBUTED` are emitted as `unavailable` **even when the
response carried a parseable number for them.** You cannot accidentally bind a
panel to a lie.

---

## 2. `TelemetryField`

```js
{
  value:        3,              // null unless state is 'measured' or 'stale'
  state:        'measured',     // see the five states below
  source:       '/v1/status',   // the ENDPOINT -- precise enough to curl
  sourceClass:  'server',       // the CLASS -- 'server'|'client'|'derived'|'estimated'
  origin:       'scatter',      // which server we asked: 'scatter' | 'dynamic'
  originModelId:'qwen2.5-0.5b-scatter-v2',  // what that server CALLED ITSELF
  label:        'Queue depth',
  reason:       null,           // required sentence whenever state !== 'measured'
  unit:         'requests',     // or null
  observedAtMs: 1785390093123,  // null when absent; ORIGINAL time when stale
  derivedFrom:  null,           // field keys, when sourceClass === 'derived'
  provenanceWarning: null       // set when the wire contradicted the audit
}
```

> ⚠️ **`state` for a good reading is the string `'measured'`, and the constant
> is `FIELD_STATES.MEASURED`.** The constant was once spelled `MEASURED` while
> its value stayed `'ok'`, which made `field.state === 'measured'` silently
> false for every measured field on the page. Name and value now agree, and the
> transitional alias was deleted rather than deprecated — an alias is a fork
> with a deprecation notice.
>
> The enum value and the `[data-state='measured']` selector in
> `styles/shell.css` are an **atomic pair**: changing either alone reintroduces
> the bug in one direction or the other, and neither half fails loudly. A
> mismatch renders every genuine measurement at *muted* contrast wherever a
> panel sets a colour, so real numbers look de-emphasised — the exact honesty
> inversion this dashboard exists to prevent. `state-treatments.test.js`
> compares the two on every run.
>
> Never global-replace the bare string `'ok'` to make this change: `status:
> 'ok'` is the HTTP health payload, and renaming that fakes an unreachable
> server. Three unrelated vocabularies share that token.
> **Always compare against `FIELD_STATES.*`, never a string literal.**

| `state` | Meaning | Render as |
|---|---|---|
| `measured` | The server computed this, just now. **Includes a genuine zero.** | The number, full contrast, no apology. |
| `pending` | Measurable, but no sample has arrived yet. **Resolves on its own.** | `···` |
| `stale` | Was measured; the latest poll did not refresh it. `value` is the last good reading. | The number, de-emphasised, **with its age in words** (`41 · 12s old`). |
| `unavailable` | The server hardcodes a placeholder and never computes it. **Plumbing would fix it.** | **Em-dash `—`** + `reason` on hover. Never `0`, never blank. |
| `not-applicable` | The subsystem exists but **this code path never consults it**, so the question is never asked. Plumbing would *not* fix it. | **`n/a`** + `reason` explaining *why*. Deliberately distinct from `unavailable`, see the note below. |

> ⚠️ **`not-applicable` renders `n/a`, NOT an em-dash.** This table said em-dash
> for both until it was caught; the code (`format.js`, `NOT_APPLICABLE_TEXT`)
> has always rendered `n/a`, and `state-treatments.test.js` asserts the two are
> distinguishable **on the surface, not only in the hover**.
> The reason is demo-spec.md:714 — *structurally not applicable must not look
> like broken*. Under the two-server design, structurally-empty panels are the
> NORMAL case, so if they render identically to a stubbed field, a visitor reads
> half a correctly-working dashboard as defective. A hover cannot fix that: it
> is not reachable by touch or keyboard scanning, and nobody hovers a field they
> have already concluded is broken.

### The three kinds of zero (demo-spec.md §3, binding)

A `0` on the wire is byte-identical in all three cases, so neither the wire nor
a panel can tell them apart. Only the provenance table can:

1. **`measured` with value `0`** — the question was asked, the answer really is zero.
   This is DATA. Hiding it is as dishonest as fabricating it.
2. **`unavailable`** — a hardcoded stub (`tokens_per_second: 0.0`,
   `admin.rs:63`). A gap in the server.
3. **`not-applicable`** — e.g. the prefix cache on the scatter server, whose
   batching path bypasses the trie entirely. A property of the architecture,
   and often the most interesting thing on the page.

Collapsing 2 and 3 turns our single most interesting finding into an apologetic
"not implemented yet". **The same wire value lands in different buckets on
different servers**, so classification depends on `origin`, not on the number.

`pending` and `unavailable` are also deliberately separate: pending resolves by
itself, unavailable never will, and telling a visitor to wait for a number that
is never coming is its own small dishonesty.

`observedAtMs` on a stale field is the **original** observation time, so age
keeps growing across repeated failed polls instead of resetting.

`provenanceWarning` is set when a field the audit classified as a placeholder
arrived carrying something else — meaning the server plumbing landed and
`telemetry-provenance.js` is out of date. The value is **shown** (hiding a real
measurement is the mirrored fabrication), with the contradiction attached.
`store.provenanceWarnings()` lists them for the AC10 footer.

Build fields with the helpers in `telemetry-field.js` — `measuredField`,
`pendingField`, `unavailableField`, `staleField`, `derivedField`. Never
construct the object literal by hand; the helpers reject a `measuredField(null)`
and an `unavailableField()` with no reason, which is exactly where this would rot.

**Read state before value, always.** `hasValue(field)` and
`numericValueOf(field)` are the guards — calling either *is* reading the state,
so a panel that reaches a value through them is correct by construction, and a
reviewer can grep for a `.value` access not preceded by one. `numericValueOf`
returns `null` rather than a number, because a bare `null` value coerces to `0`
in `+` and `<`, which is precisely how a fabricated zero reaches the screen.

**Derivation is contagious.** `derivedField` returns `unavailable` if any input
is unavailable, and `pending` if any input is pending. A ratio computed from a
documented zero is still a fabricated number.

**And it has a PRECEDENCE, not just propagation.** When inputs disagree:

```
not-applicable  >  unavailable  >  pending  >  stale  >  measured
```

`not-applicable` dominates **everything**, and the asymmetry is the point.
`unavailable`, `pending` and `stale` are all statements about our MEASUREMENT
PIPELINE — "not plumbed", "not polled yet", "poll failed" — and every one of
them implies the number could still arrive, so a derivation over them may yet
succeed. `not-applicable` is a statement about the EXECUTION PATH: the question
is not being asked at all, so the derivation can never succeed either.

Reporting `pending` for a quantity that can never exist is a small lie with a
spinner on it — the same argument that won `pending` its place against
`unavailable`.

`stale` is the one that binds LAST rather than first: it is applied to a result
that computed successfully, so it never suppresses a value, it only ages it.

---

## 3. The store

```js
import { createTelemetryStore } from '../telemetry-store.js';

const telemetryStore = createTelemetryStore();  // same-origin, 250 ms
telemetryStore.start();
```

One polling loop for the whole page, regardless of how many panels are mounted.
Panels **must not** fetch on their own: independent polling drifts, and two
panels showing different instants makes the dashboard contradict itself.

| Member | Contract |
|---|---|
| `start()` / `stop()` | Idempotent. The store is inert until `start()`. |
| `getSnapshot()` | The current frozen `TelemetrySnapshot`. Never null. |
| `field(key)` | A `TelemetryField`, **always**. An unknown key returns an explained `unavailable` field rather than `undefined`, so a typo degrades one tile instead of throwing. |
| `subscribe(fn)` | Returns an `unsubscribe` function. `fn` is called **immediately** with the current snapshot, then once per poll. |
| `pollOnce()` | One cycle, no scheduling. For tests. |
| `pollIntervalMs`, `baseUrl`, `isRunning` | Read-only. |

Guarantees you may rely on:

- **At most one poll cycle in flight.** A slow server delays the next cycle; it
  never queues cycles behind it.
- **Snapshots are frozen.** Mutating one throws in strict mode. Never mutate.
- **A subscriber that throws is caught and logged.** Your panel bug cannot take
  down another panel. Do not rely on this — it is a safety net, not a feature.
- **Backoff while unreachable.** 500 ms → 8 s, so a dead server does not produce
  a console flood or a request storm.
- **Every field key exists in every snapshot**, including the very first one
  before any poll completes. There is no "not loaded yet" hole to code around —
  measurable fields start `pending`, documented zeros start `unavailable`.

### Field keys

Keys are stable and defined in `telemetry-provenance.js` — reference keys, never
raw JSON paths. A server-side rename is then one edit in one file. Ask the owner
to add a key rather than reaching into `snapshot.raw`.

Measured today: `queue.depth`, `sessions.active`, `server.model_id`,
`server.healthy`, `server.node_id`, `server.context_length`, `server.pipeline`,
`batch.active_size`, `admission.slots_available`, `admission.rejections`,
`prefix_cache.hits`, `prefix_cache.lookups`.

From `GET /metrics` (Prometheus text, parsed by `prometheus-parse.js`):
`metrics.ttft`, `metrics.e2e_latency`, `metrics.tokens_generated_total`,
`metrics.completion_tokens_total`, `metrics.requests_waiting`,
`metrics.prefix_cache_hits`, `metrics.prefix_cache_lookups`, `batch.in_flight`.
From `GET /v1/resources`: `resources.kv_budget_bytes`,
`resources.vram_limit_bytes`.

Derived in the browser: `throughput.observed` — a real tokens/sec, obtained by
differentiating `metrics.tokens_generated_total` between polls. The server
hardcodes `tokens_per_second: 0.0` because it keeps totals but no rate; the
totals are genuine, so the rate is recoverable. It carries
`derivedFrom: ['metrics.tokens_generated_total']`, and it is `pending` on the
first poll because a rate needs two samples.

Unavailable today (documented zeros / not plumbed): `kv.usage`, `kv.pages_used`,
`kv.pages_total`, `kv.pages_shared`, `kv.introspection`, `batch.utilization`,
`throughput.tokens_per_second`, `sessions.paused`, `prefix_cache.hashes`,
`server.execution_provider`, `batch.effective_size`.

### A true value the page must never show

Every classification above answers one question: **is this value true?** Some
wire fields are perfectly true and must still never reach a visitor, so no
classification can ever describe them. Those are listed in `NEVER_BIND` in
`telemetry-provenance.js`, and the rule is a **ban, not a state**:

- `/v1/models` → `created` is `now_unix()`, recomputed per request. True, and a
  wrong fact about when the model was built.
- `/v1/models` → `path` is the configured model directory. Its own Rust doc
  comment says *"Absolute on loopback; the basename otherwise"* — and every
  origin this demo ships is loopback, so the protective branch never runs here.
  The page wants **identity**, and `server.model_id` already carries it.

Two properties make the ban enforceable rather than advisory. First, no
`PROVENANCE` row may read a banned field, so a row and a ban cannot coexist —
which is why `server.model_path` was **deleted** rather than reclassified.
Second, no shipping module may read the field name off a response body. A
banned field may declare narrow `exemptions` naming exact safe tokens (`path`
collides with the `path` descriptor on a provenance row); each must justify
itself and must still match live source, so a hole cannot outlive the code it
was granted for.

> ⚠️ **Do not build a "batch size" panel on `batch.in_flight`.**
> `onnx_genai_batch_size_current` is documented "Current generation batch size"
> but is `fetch_add(1)` in `GenerationMetrics::start()` and decremented in
> `Drop` (`metrics.rs:112`, `:145`) — it counts **generation requests in
> flight** at the HTTP layer and never consults the batch manager. With
> `max_batch` pinned at 4, 8 concurrent requests make it read 8 while the engine
> batches 4. The engine's real batch size is `batch.effective_size`, which is
> **unavailable** because nothing exposes it.
>
> This is the queueing-vs-batching distinction the spec requires you to narrate:
> `in_flight` is measured, everything above the batch limit is *queued*, and the
> batch itself is not observable. Say that, rather than implying a batch of 8.

> `metrics.prefix_cache_hits` is `measured` and is genuinely `0` on a
> static-cache server: the batching path bypasses the prefix trie entirely. That
> zero is a real finding about the architecture — caption it, don't debug it,
> and don't convert it to `unavailable`.

> Build your panel against the unavailable ones **now**. Unavailable is the
> correct first state, not a placeholder — when @d7cf9b84's plumbing lands, one
> `classification` flip in `telemetry-provenance.js` turns your panel live with
> no panel-side change.

---

## 3a. Field paths — the authoritative names

`telemetry-provenance.js` is the registry, and its keys are the paths
`store.field(path)` resolves. **An unknown path is safe**: the store returns a
real `unavailable` field whose reason names the missing key —

```js
store.field('kv.blocks_used')
// { state: 'unavailable', reason: 'No field named "kv.blocks_used" is published…' }
```

— so a typo degrades to an honest em-dash rather than `undefined`, a crash or a
zero. That is the correct failure direction, but it is not a good END state: the
reason explains our plumbing to a visitor who wants to know about the *server*.
Bind the real name where one exists, and where none does, pass a reason that
says what the server does not compute and why.

`demo-ux.md` §3.2 sketched a flat namespace that was never implemented. Where
the two differ, the registry wins. Renames:

| demo-ux.md §3.2 | registry key | note |
|---|---|---|
| `throughput.aggregate_tok_s` | `throughput.observed` | derived from the token counter. **Not** `throughput.tokens_per_second`, which the server hardcodes to zero |
| `scheduler.running` | `batch.in_flight` | |
| `scheduler.waiting` | `metrics.requests_waiting` | |
| `kv.blocks_used` / `_total` / `_shared` | `kv.pages_used` / `pages_total` / `pages_shared` | "pages" is the engine's own word |
| `resources.kv_bytes_limit` | `resources.kv_budget_bytes` | derived, not reported |
| `latency.*_p50/p95/max` | `metrics.ttft`, `metrics.e2e_latency` | **no percentiles exist.** The server exposes a single current value per metric, not a distribution |

Paths with **no registry key at all** — the server does not compute them, so
they cannot be bound, renamed or awaited: `scenario.makespan_ms`,
`scheduler.max_batch`, `scheduler.preemptions_total`, `queue.depth_peak`,
`kv.block_size`, `kv.slots_filled`, `kv.slot_capacity`, `kv.allocations`,
`kv.frees`, `kv.allocation_failures`, `kv.hot_evictions`,
`kv.prefix_evictions`, `kv.refcount_histogram`, `kv.tiers`,
`server.decode_backend`, `server.quantization`, `server.version`,
`server.uptime_ms`, `resources.kv_bytes_used`, `resources.host_ram_used`,
`resources.host_ram_limit`, `resources.disk_spill_bytes`.

`client.*` is not in the registry by design: the registry describes what the
SERVER publishes. A client measurement is built with `measuredField(value, {
sourceClass: 'client', origin })` at the point of measurement.

> ⚠️ **Do not bind any `prefix.*` / `prefix_cache.*` path.** Ruled final, and
> `prefix-counters-forbidden.test.js` enforces it as a shrinking ratchet. Those
> counters are not stubs — they are precisely computed and entirely false
> (~95%, because the counter increments on any nonzero token match and every
> chat request shares the template preamble), while a controlled measurement
> found reuse ABSENT. Every other safeguard here hunts fabricated *zeros*; a
> confident 95% invites no scrutiny at all, which is what makes it the most
> dangerous number in the tree.

**Absence is not one thing.** Before binding a missing path to
`unavailableField`, check the classification in the registry:
`NOT_PLUMBED` and `DOCUMENTED_ZERO` mean *unavailable* (plumbing would fix it);
`STRUCTURALLY_BYPASSED` means *not-applicable* (plumbing would not);
`MISATTRIBUTED` means *unavailable* for the reason no plumbing addresses — the
server computes a real, correct number that answers a different question than
the field name asks. Use `neverMeasuredField()` rather than deciding per call
site — the mapping was copied three times before and drifted in two of them.

`MISATTRIBUTED` is the only classification whose wire value looks perfect, so it
is the only one you cannot diagnose by looking at the number. It exists because
the vocabulary had no word for it: the three prefix-cache hit fields were
`MEASURED` because that was the only remaining option, not because anyone judged
them measured. **A missing word in a vocabulary does not read as a gap; it reads
as agreement.** If the true quantity is itself worth showing, RELABEL and keep it
`MEASURED` instead — `prefix_cache.lookups` genuinely counts completed
generations and is labelled that way.

---

## 3b. Rendering — `format.js`

**Do not branch on `field.state` in a panel.** Every `if (field.state === '…')`
is a place the next state gets missed, and that has already happened twice in
this codebase: one branch checking only `unavailable` silently swallowed
`not-applicable`, and a unit test asserting a single key passed while the page
displayed a fabricated zero.

```js
import { formatField, describeFieldText } from '../format.js';

const { text, badge, title, hasValue } = formatField(store.field('queue.depth'));
valueEl.textContent = text;      // '3 requests' | '—' | '···' | '41 · 12s old'
valueEl.title = title;           // AC7 provenance sentence
badgeEl.textContent = badge;     // ˢ | ᶜ | ᴰ | ᴱ
```

`formatField()` handles all five states in one place and gives you, free:

- a stark `0` for a measured zero — **never** hidden;
- `—` for absent vs `···` for pending;
- `~` + `ᴱ` for an estimate, visible **without hovering** (an estimate that
  looks like a measurement *is* a fabricated measurement — nobody hovers);
- `41 · 12s old` for stale — age in **words**, because colour alone fails
  grayscale and colourblind readers (AC25);
- the AC7 source-class badge from `SOURCE_CLASS_BADGES`.

Use `describeFieldText(name, field)` for prose and aria-labels.

**Rule for estimates:** if we cannot state the formula, we do not show the
number.

---

## 4. The panel contract

Every file in `dashboard/` provides **a named `meta` export and a named
`mount`** (shape ratified by @0837fdf9, who owns the panel seam):

```js
/** The shell reads this BEFORE mounting, to build the panel host and grid. */
export const meta = {
  id: 'kv-memory',
  title: 'Paged KV block table',
  group: 'memory',
  span: 2,               // grid columns
  cadence: 250,          // ms; how often this panel wants to repaint
  defaultOpen: true,
  acronyms: { KV: 'key/value attention cache' },  // AC30
};

/**
 * @param {HTMLElement} rootElement   Empty element the panel owns entirely.
 * @param {TelemetryStore} telemetryStore
 * @returns {{ unmount: () => void, describe: () => string }}
 */
export function mount(rootElement, telemetryStore) { /* … */ }
```

`describe()` returns a plain-English sentence describing the panel's **current**
state. It is **not optional**: it powers the chart `aria-label` (AC28) and the
"view as table" affordance. Build it from `describeFieldText()` in `format.js`
so phrasing cannot drift between panels.

### Lifecycle

1. **`mount(rootElement, telemetryStore)`** — called once. Build your DOM inside
   `rootElement` and call `telemetryStore.subscribe(...)`.
2. **update** — there is no `update()` function. Your subscriber callback *is*
   the update path. It fires immediately on subscribe and once per poll.
3. **`unmount()`** — you **must** return this and it **must** call your
   `unsubscribe`, cancel any `requestAnimationFrame`/timer, and release canvas
   references. A panel that leaks a subscription fails AC22 (no memory growth
   over a 60 s run).

   The name is `unmount`, not `destroy`. The shell calls `handle.unmount()`
   (`dashboard/index.js:176`); a handle returning `destroy` gets `undefined()`
   — except it never throws, because the shell wraps each teardown in a
   try/catch so one bad panel cannot strand the others' subscriptions. The
   leak is therefore **completely silent**, which is why this contract text
   being wrong sent four separate agents to build the wrong thing. Note that
   `destroy()` IS correct on two neighbouring objects — `roving.destroy()` and
   `adapter.destroy()` — so grepping `destroy` in the shell falsely confirms
   it.

### What a panel may rely on

- `rootElement` is empty, attached to the document, and yours alone. Nothing
  else writes into it.
- The shell sets `data-panel="<panel-id>"` on `rootElement`.
- CSS custom properties from the design token set are in scope. Use tokens —
  never a literal colour.
- The store is already started (or will be); your first callback may arrive
  before the first successful poll, with every field `unavailable`. Render that
  state correctly; it is the normal first frame.

### What a panel must NOT do

- No `fetch`. Ask the owner to add a field key.
- No writing outside `rootElement`.
- No mutating the snapshot.
- No rendering `field.value` without checking `field.state`.
- No colour-only encoding — pair every colour with a shape, pattern or label.
- No `innerHTML` with server-derived strings. Use `textContent`; server messages
  are shown verbatim and must not be interpretable as markup.

---

## 5. Worked example — one measured field, one unavailable field

```js
// dashboard/queue-panel.js
// Copyright (c) Microsoft Corporation.
import { formatFieldText, describeField, FIELD_STATES } from '../telemetry-field.js';

export default function mount(rootElement, telemetryStore) {
  rootElement.innerHTML = `
    <h3 class="panel__title">Queue</h3>
    <dl class="panel__stats">
      <dt>Queue depth</dt>       <dd data-field="queue.depth">—</dd>
      <dt>Batch utilization</dt> <dd data-field="batch.utilization">—</dd>
    </dl>
  `;

  const cells = new Map(
    [...rootElement.querySelectorAll('[data-field]')].map((el) => [el.dataset.field, el]),
  );

  const unsubscribe = telemetryStore.subscribe(() => {
    for (const [key, element] of cells) {
      renderField(element, telemetryStore.field(key));
    }
  });

  return { unmount: () => unsubscribe() };
}

/**
 * The whole contract in six lines. `data-state` is what the stylesheet hooks
 * to render the em-dash treatment and the stale de-emphasis, so no panel
 * invents its own visual language for absence.
 */
function renderField(element, field) {
  element.textContent = formatFieldText(field);        // '—' when unavailable
  element.dataset.state = field.state;                 // measured | unavailable | stale
  element.title = describeField(field);                // AC7: provenance on hover
  element.setAttribute('aria-label', describeField(field));
}
```

What the visitor sees against a live server today:

| Field | State | Rendered | Tooltip |
|---|---|---|---|
| `queue.depth` | `measured` | `3` | `3 requests — measured, source /v1/status` |
| `batch.utilization` | `unavailable` | `—` | `Unavailable — The server cannot compute this because the batch limit is not surfaced to the HTTP layer, so it sends a hardcoded 0.0. A zero here would be a fabricated measurement. (would come from /v1/status)` |

Note what did **not** happen: `batch.utilization` did not render `0.0`, and the
panel needed no special case to prevent it. The panel code is identical for both
fields. That is the point of the envelope — honesty is the default path, not the
disciplined path.

---

## 6. Connection state

`snapshot.connection.state` drives the two **blocking, full-stage** failure
states. They are deliberately distinct: different problems, different fixes.
Panels do not implement these — the shell does, and it replaces the stage
entirely. Panels only need to handle `stale` fields gracefully.

| State | Detection | Meaning |
|---|---|---|
| `connecting` | before the first cycle | — |
| `connected` | server answered | normal |
| `unreachable` | **transport** failure on `/health` and `/v1/status` | the process is not there |
| `no-model` | `/v1/status.healthy === false` | the process is there, with nothing loaded |

`connection.serverMessage` carries the server's **own** error text, verbatim,
extracted from its `{ error: { message } }` body
(`crates/onnx-genai-server/src/routes/mod.rs:444-449`). Never paraphrase it —
the server's messages are written in a what/why/how style that is better than
anything we would write, and a message the visitor can grep for is worth more
than one we made prettier.

---

## 7. Tests

```bash
./examples/serving-dashboard/run-tests.sh
```

The script resolves its own directory, so its *results* never depend on where
you stand. **This path does.** Run it from the repository root; from anywhere
else it fails loudly with `No such file or directory` -- which is the whole
point of the wrapper. The form it replaced answered a wrong `cwd` with
`0 tests, exit 0`: a normative document certifying an empty run as a pass.
A loud miss is recoverable; a green nothing is not. **This is the only
documented form.** The command it replaced was correct prose about a command
that had already drifted, and this section is kept as the reason the script
exists rather than as a second copy of it.

> **A TEST COMMAND WRITTEN IN A MARKDOWN FENCE IS AN UNTESTED COPY OF A FACT.**
> This command was documented in seven tracked files in five incompatible
> forms. Each of the following was a real, green, wrong run:
>
> - **No `cd`.** Globs resolve against the working directory, not against this
>   document. From the repository root the old command ran **279 of 582 tests
>   and exited 0**; from the directory this file lives in it ran **0 tests and
>   exited 0**.
> - **One glob.** `dashboard/` holds roughly half the suite. One glob silently
>   omitted all of it.
> - **Two globs.** Also wrong, and this is the instructive one: `'*.test.js'
>   'dashboard/*.test.js'` silently omits an entire test directory, `ui/`.
>   The size of the miss is the only stable part: **583 of 588 at `c71ed21e`,
>   595 of 600 at `10537446`** -- the totals drift with every commit and the
>   *gap* does not. Any absolute count written here is false within the hour,
>   which is why the runner asserts a floor and prints its own numbers instead
>   of this paragraph promising them. Four reviewers independently proposed
>   exactly that two-glob fix and **all four missed `ui/`** — so the script
>   *discovers* test files instead of listing them. A hardcoded list stops
>   covering whatever was added last, which is always the thing most likely
>   to be wrong.
> - **Unquoted globs.** The shell expands them first, and shell expansion does
>   not recurse.
>
> ⚠️ **`node --test` treats "no files matched" as success, not as an error**,
> and a file that fails to *import* contributes zero tests rather than failing
> ones — so a broken module makes the suite report **smaller**, not redder.
> Both are why the script reconciles the number of discovered files against the
> number of suites Node actually executed.

**Treat fewer than 500 tests as a FAILED run even if it exits 0.** A partial
suite reporting success is the one result that cannot be distinguished from a
good one by looking at it. The script asserts this floor for you and fails
loudly.

📌 The floor is a **liveness** assertion, not a completeness one: it proves the
instrument ran at all. It cannot see the silent loss of a single suite — only
the file-count reconciliation can. Do not read a passing floor as coverage.

📌 No exact expected count is written here on purpose. A number in prose is a
fact that must be remembered in N places; the script computes it, prints it
beside `pwd`, `HEAD` and the Node version, and asserts the properties that
actually matter.

Node's built-in runner. No dependencies, no install, consistent with the
demo's no-build-step rule. `telemetry-store.test.js` locks down the behaviours
above — most importantly that documented zeros can never surface as
measurements, and that a real `0` still can.
