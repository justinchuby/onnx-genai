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

`telemetry-provenance.js` enforces it: fields classified `DOCUMENTED_ZERO` or
`NOT_PLUMBED` are emitted as `unavailable` **even when the response carried a
parseable number for them.** You cannot accidentally bind a panel to a lie.

---

## 2. `TelemetryField`

```js
{
  value:        3,              // null whenever state === 'unavailable'
  state:        'measured',     // 'measured' | 'unavailable' | 'stale'
  source:       '/v1/status',   // or 'client' | 'derived' | endpoint path
  reason:       null,           // required sentence whenever state !== 'measured'
  unit:         'requests',     // or null
  observedAtMs: 1785390093123,  // null when unavailable
  derivedFrom:  null            // field keys, when source === 'derived'
}
```

| `state` | Meaning | Render as |
|---|---|---|
| `measured` | The server computed this, just now. | The number. |
| `unavailable` | No value exists — not measurable yet, endpoint disabled, or structurally inapplicable. | **Em-dash `—`** with `reason` in the tooltip. Never `0`, never blank, never `undefined`. |
| `stale` | Was measured; the latest poll did not refresh it. `value` is the last good reading. | The number, visibly de-emphasised, with its age. |

`observedAtMs` on a stale field is the **original** observation time, so age
keeps growing across repeated failed polls instead of resetting. That is what
lets a panel say "12s old" honestly.

Build fields with the helpers in `telemetry-field.js` — `measuredField`,
`unavailableField`, `staleField`, `derivedField`. Never construct the object
literal by hand; the helpers reject a `measuredField(null)` and an
`unavailableField()` with no reason, which is exactly where this would rot.

**Derivation is contagious.** `derivedField` returns `unavailable` if *any*
input is unavailable. A ratio computed from a documented zero is still a
fabricated number.

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
  before any poll completes. There is no "not loaded yet" hole to code around.

### Field keys

Keys are stable and defined in `telemetry-provenance.js` — reference keys, never
raw JSON paths. A server-side rename is then one edit in one file. Ask the owner
to add a key rather than reaching into `snapshot.raw`.

Measured today: `queue.depth`, `sessions.active`, `server.model_id`,
`server.healthy`, `server.node_id`, `server.context_length`, `server.pipeline`,
`batch.active_size`, `admission.slots_available`, `admission.rejections`,
`prefix_cache.hits`, `prefix_cache.lookups`.

Unavailable today (documented zeros / not plumbed): `kv.usage`, `kv.pages_used`,
`kv.pages_total`, `kv.pages_shared`, `kv.introspection`, `batch.utilization`,
`throughput.tokens_per_second`, `sessions.paused`, `prefix_cache.hashes`,
`server.model_path`, `server.execution_provider`.

> Build your panel against the unavailable ones **now**. Unavailable is the
> correct first state, not a placeholder — when @d7cf9b84's plumbing lands, one
> `classification` flip in `telemetry-provenance.js` turns your panel live with
> no panel-side change.

---

## 4. The panel contract

Every file in `dashboard/` default-exports **exactly this**:

```js
/**
 * @param {HTMLElement} rootElement   Empty element the panel owns entirely.
 * @param {TelemetryStore} telemetryStore
 * @returns {{ unmount: () => void }}
 */
export default function mount(rootElement, telemetryStore) { /* … */ }
```

### Lifecycle

1. **`mount(rootElement, telemetryStore)`** — called once. Build your DOM inside
   `rootElement` and call `telemetryStore.subscribe(...)`.
2. **update** — there is no `update()` function. Your subscriber callback *is*
   the update path. It fires immediately on subscribe and once per poll.
3. **`unmount()`** — you **must** return this and it **must** call your
   `unsubscribe`, cancel any `requestAnimationFrame`/timer, and release canvas
   references. A panel that leaks a subscription fails AC22 (no memory growth
   over a 60 s run).

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
node --test 'examples/serving-dashboard/*.test.js'
```

Node's built-in runner. No dependencies, no install, consistent with the
demo's no-build-step rule. `telemetry-store.test.js` locks down the behaviours
above — most importantly that documented zeros can never surface as
measurements, and that a real `0` still can.
