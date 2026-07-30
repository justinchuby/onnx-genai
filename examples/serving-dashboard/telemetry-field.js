// Copyright (c) Microsoft Corporation.
//
// The provenance envelope. Every telemetry value in this demo is wrapped in a
// TelemetryField before any UI code can see it.
//
// WHY THIS EXISTS: `GET /v1/status` returns hardcoded `0.0` for `kv_usage`,
// `tokens_per_second` and `batch_utilization`
// (crates/onnx-genai-server/src/routes/admin.rs:53-69 — each carries a
// `// not yet tracked` comment). Those are real HTTP responses containing
// numbers nobody measured. If the store held bare numbers, a panel could not
// tell "the server measured zero" apart from "the server cannot measure this",
// and the project's no-fabricated-numbers rule would be literally
// inexpressible. The envelope makes the distinction a type-level fact instead
// of a convention people have to remember.
//
// A panel therefore NEVER receives a number. It receives a field, and it must
// branch on `field.state` before rendering.

/**
 * @typedef {'measured' | 'pending' | 'stale' | 'unavailable'} FieldState
 *
 * - `measured`    — the server genuinely computed this value, just now.
 *                   Includes a genuine zero, which renders at full contrast.
 * - `pending`     — measurable, but no sample has arrived yet (the first poll
 *                   has not completed). `value` is `null`. Renders `···`.
 *                   Distinct from `unavailable` because pending resolves ON ITS
 *                   OWN and unavailable does not — telling a visitor to wait for
 *                   a number that will never arrive is its own small dishonesty.
 * - `stale`       — this WAS measured, but the most recent poll did not refresh
 *                   it (server unreachable, request in flight too long).
 *                   `value` is the last known good value; `observedAtMs` says
 *                   how old it is. Render it visibly de-emphasised with its age.
 * - `unavailable` — no value exists and none is coming without a server or
 *                   configuration change. Either the server cannot measure it
 *                   yet, the endpoint that carries it is disabled, or the field
 *                   is structurally inapplicable to the current decode path.
 *                   `value` is ALWAYS `null`. Render an em-dash, never a zero.
 */

/**
 * @typedef {object} TelemetryField
 * @property {number|string|boolean|Array|object|null} value
 *   The measurement. `null` whenever `state` is `unavailable` or `pending`.
 * @property {FieldState} state
 * @property {string} source
 *   Where the value came from, precisely enough to curl it. Examples:
 *   `'/v1/status'`, `'/v1/debug/kv'`, `'client'` (measured in the browser),
 *   `'derived'` (computed from other fields — `derivedFrom` says which).
 * @property {string|null} reason
 *   Required when `state !== 'measured'`. A complete sentence a visitor can
 *   read in a tooltip, ending with what they could do about it if anything.
 *   `null` when `state === 'measured'`.
 * @property {string|null} unit
 *   `'tokens/s'`, `'ms'`, `'ratio'`, `'pages'`, `'count'`, … or `null` for
 *   unitless/ordinal values. Panels must not invent units.
 * @property {number|null} observedAtMs
 *   `Date.now()` at the moment the value was read off the wire. `null` for
 *   unavailable and pending fields. For `stale` fields this is the ORIGINAL
 *   observation time, which is exactly what a panel needs to render an age.
 * @property {string[]|null} derivedFrom
 *   For `source === 'derived'`, the field keys this was computed from, so the
 *   footer provenance table can be generated rather than hand-maintained.
 */

/** Legal values of `TelemetryField.state`. */
export const FIELD_STATES = Object.freeze({
  MEASURED: 'measured',
  PENDING: 'pending',
  STALE: 'stale',
  UNAVAILABLE: 'unavailable',
});

/**
 * Build a genuinely-measured field.
 *
 * Do not call this for a value you did not read off a live response. If you
 * find yourself passing a literal, you want {@link unavailableField}.
 *
 * @param {number|string|boolean|Array|object} value
 * @param {object} options
 * @param {string} options.source        e.g. `'/v1/status'` or `'client'`.
 * @param {string|null} [options.unit]
 * @param {number} [options.observedAtMs] Defaults to now.
 * @returns {TelemetryField}
 */
export function measuredField(value, { source, unit = null, observedAtMs = Date.now(), derivedFrom = null }) {
  if (value === null || value === undefined) {
    throw new TypeError(
      'measuredField() was given a null/undefined value. A missing value is not a ' +
        'measurement — use unavailableField(reason) so the UI can render an em-dash.',
    );
  }
  if (!source) {
    throw new TypeError('measuredField() requires a `source` so the value can be audited.');
  }
  return Object.freeze({
    value,
    state: FIELD_STATES.MEASURED,
    source,
    reason: null,
    unit,
    observedAtMs,
    // Set when WE computed the value from other fields rather than reading it
    // off a response. Still a real measurement, but a panel showing it owes the
    // viewer the inputs — a number we derived needs more disclosure, not less.
    derivedFrom,
  });
}

/**
 * Build an unavailable field. This is the correct, honest first state for
 * anything not yet plumbed — it is not a placeholder and not a TODO.
 *
 * @param {string} reason
 *   A complete sentence explaining why, shown verbatim in the tooltip.
 * @param {object} [options]
 * @param {string} [options.source] The endpoint that WOULD carry it, if known.
 * @param {string|null} [options.unit]
 * @returns {TelemetryField}
 */
export function unavailableField(reason, { source = 'unavailable', unit = null } = {}) {
  if (!reason) {
    throw new TypeError(
      'unavailableField() requires a reason. "No data" with no explanation reads as a bug; ' +
        'an explained absence reads as honesty.',
    );
  }
  return Object.freeze({
    value: null,
    state: FIELD_STATES.UNAVAILABLE,
    source,
    reason,
    unit,
    observedAtMs: null,
    derivedFrom: null,
  });
}

/**
 * Build a pending field: this IS measurable, but no sample has arrived yet.
 *
 * Only correct before the first successful poll. Never use it for something the
 * server cannot measure — that is {@link unavailableField}. The distinction is
 * the whole point: pending resolves by itself, unavailable never will, and a
 * visitor waiting for a number that is never coming has been misled.
 *
 * @param {string} reason
 * @param {object} [options]
 * @param {string} [options.source]
 * @param {string|null} [options.unit]
 * @returns {TelemetryField}
 */
export function pendingField(reason, { source = 'unknown', unit = null } = {}) {
  if (!reason) {
    throw new TypeError('pendingField() requires a reason explaining what is being waited on.');
  }
  return Object.freeze({
    value: null,
    state: FIELD_STATES.PENDING,
    source,
    reason,
    unit,
    observedAtMs: null,
    derivedFrom: null,
  });
}

/**
 * Age a previously-measured field because the latest poll did not refresh it.
 *
 * Unavailable and pending fields are returned unchanged: absence has no age,
 * and a value that never arrived cannot go stale.
 *
 * An already-stale field keeps its ORIGINAL `observedAtMs`, so age keeps
 * growing across repeated failed polls instead of resetting each time.
 *
 * @param {TelemetryField} field
 * @param {string} reason
 * @returns {TelemetryField}
 */
export function staleField(field, reason) {
  if (field.state === FIELD_STATES.UNAVAILABLE || field.state === FIELD_STATES.PENDING) {
    return field;
  }
  return Object.freeze({
    ...field,
    state: FIELD_STATES.STALE,
    reason,
  });
}

/**
 * Build a field derived from other fields (a rate, a ratio, a difference).
 *
 * Derivation is contagious: if ANY input is not measured, the result is
 * unavailable. A ratio computed from a documented zero is still a fabricated
 * number, and this is where that mistake would otherwise slip in.
 *
 * @param {Record<string, TelemetryField>} inputs Keyed by field key.
 * @param {(values: Record<string, any>) => number|null} compute
 *   Receives the raw values keyed identically to `inputs`. Return `null` to
 *   signal "inputs were fine but the result is undefined" (e.g. divide by zero).
 * @param {object} options
 * @param {string|null} [options.unit]
 * @param {string} [options.undefinedReason] Reason used when `compute` returns null.
 * @returns {TelemetryField}
 */
export function derivedField(inputs, compute, { unit = null, undefinedReason } = {}) {
  const keys = Object.keys(inputs);
  const blocking = keys.filter((key) => inputs[key].state === FIELD_STATES.UNAVAILABLE);
  if (blocking.length > 0) {
    return unavailableField(
      `Cannot be derived because ${blocking.join(', ')} ${
        blocking.length === 1 ? 'is' : 'are'
      } unavailable: ${inputs[blocking[0]].reason}`,
      { source: 'derived', unit },
    );
  }

  // Pending is also contagious, but it resolves on its own, so the result is
  // pending rather than unavailable — the visitor should wait, not give up.
  const waiting = keys.filter((key) => inputs[key].state === FIELD_STATES.PENDING);
  if (waiting.length > 0) {
    return pendingField(
      `Waiting on ${waiting.join(', ')} before this can be derived.`,
      { source: 'derived', unit },
    );
  }

  const values = {};
  for (const key of keys) {
    values[key] = inputs[key].value;
  }
  const result = compute(values);
  if (result === null || result === undefined || Number.isNaN(result)) {
    return unavailableField(
      undefinedReason ?? 'The inputs are measured but the derived value is undefined for them.',
      { source: 'derived', unit },
    );
  }

  const anyStale = keys.some((key) => inputs[key].state === FIELD_STATES.STALE);
  const observedAtMs = Math.min(...keys.map((key) => inputs[key].observedAtMs ?? Date.now()));
  return Object.freeze({
    value: result,
    state: anyStale ? FIELD_STATES.STALE : FIELD_STATES.MEASURED,
    source: 'derived',
    reason: anyStale ? 'Derived from at least one stale input.' : null,
    unit,
    observedAtMs,
    derivedFrom: keys,
  });
}

/**
 * True when a panel may read `field.value` and render it.
 *
 * `stale` counts as renderable — but the panel must show it as aged.
 * `pending` and `unavailable` do not: both carry a `null` value.
 *
 * This is THE GUARD. Calling it IS reading the state, so any panel that reaches
 * a value through it is correct by construction, and reviewers can grep for a
 * `.value` access not preceded by one of these.
 *
 * @param {TelemetryField} field
 * @returns {boolean}
 */
export function hasValue(field) {
  return field.state === FIELD_STATES.MEASURED || field.state === FIELD_STATES.STALE;
}

/**
 * Read a numeric value, or `null` when the field is not renderable.
 *
 * Use this anywhere a panel does arithmetic. It makes the unavailable case
 * impossible to skip, because `null` does not silently behave like a number in
 * the comparisons panels actually write — whereas a bare `field.value` of
 * `null` coerces to 0 in `+` and `<`, which is precisely how a fabricated zero
 * would get onto the screen.
 *
 * @param {TelemetryField} field
 * @returns {number|null}
 */
export function numericValueOf(field) {
  if (!hasValue(field)) return null;
  const numeric = Number(field.value);
  return Number.isFinite(numeric) ? numeric : null;
}

/**
 * The one place the em-dash and the pending ellipsis live. Every panel formats
 * through this so absence looks identical everywhere in the page.
 *
 * @param {TelemetryField} field
 * @param {object} [options]
 * @param {(value: any) => string} [options.format] Formatter for the value.
 * @returns {string}
 */
export function formatFieldText(field, { format = defaultFormat } = {}) {
  if (field.state === FIELD_STATES.UNAVAILABLE) {
    return '—';
  }
  if (field.state === FIELD_STATES.PENDING) {
    return '···';
  }
  return format(field.value);
}

/**
 * The tooltip text for a field: what it is, where it came from, how fresh.
 * AC7 requires every metric to expose its source class on hover; this is the
 * single implementation of that.
 *
 * @param {TelemetryField} field
 * @param {number} [nowMs]
 * @returns {string}
 */
export function describeField(field, nowMs = Date.now()) {
  if (field.state === FIELD_STATES.UNAVAILABLE) {
    return `Unavailable — ${field.reason} (would come from ${field.source})`;
  }
  if (field.state === FIELD_STATES.PENDING) {
    return `Waiting for the first measurement — ${field.reason} (from ${field.source})`;
  }
  const unitSuffix = field.unit ? ` ${field.unit}` : '';
  const ageSeconds = Math.round((nowMs - (field.observedAtMs ?? nowMs)) / 1000);
  if (field.state === FIELD_STATES.STALE) {
    return `${field.value}${unitSuffix} — STALE, last measured ${ageSeconds}s ago from ${field.source}. ${field.reason}`;
  }
  const provenance = field.derivedFrom
    ? `derived from ${field.derivedFrom.join(', ')}`
    : `source ${field.source}`;
  return `${field.value}${unitSuffix} — measured, ${provenance}`;
}

/** @param {any} value */
function defaultFormat(value) {
  return typeof value === 'number' ? String(value) : String(value);
}
