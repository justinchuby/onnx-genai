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
 * @typedef {'measured' | 'unavailable' | 'stale'} FieldState
 *
 * - `measured`    — the server genuinely computed this value, just now.
 * - `unavailable` — no value exists. Either the server cannot measure it yet,
 *                   the endpoint that carries it is disabled, or the field is
 *                   structurally inapplicable to the current decode path.
 *                   `value` is ALWAYS `null`. Render an em-dash, never a zero.
 * - `stale`       — this WAS measured, but the most recent poll did not refresh
 *                   it (server unreachable, request in flight too long).
 *                   `value` is the last known good value; `observedAtMs` says
 *                   how old it is. Render it visibly de-emphasised with its age.
 */

/**
 * @typedef {object} TelemetryField
 * @property {number|string|boolean|Array|object|null} value
 *   The measurement. `null` whenever `state === 'unavailable'`.
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
 *   unavailable fields. For `stale` fields this is the ORIGINAL observation
 *   time, which is exactly what a panel needs to render an age.
 * @property {string[]|null} derivedFrom
 *   For `source === 'derived'`, the field keys this was computed from, so the
 *   footer provenance table can be generated rather than hand-maintained.
 */

/** Legal values of `TelemetryField.state`. */
export const FIELD_STATES = Object.freeze({
  MEASURED: 'measured',
  UNAVAILABLE: 'unavailable',
  STALE: 'stale',
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
export function measuredField(value, { source, unit = null, observedAtMs = Date.now() }) {
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
    derivedFrom: null,
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
 * Age a previously-measured field because the latest poll did not refresh it.
 *
 * An already-unavailable field stays unavailable — absence does not go stale.
 * An already-stale field keeps its ORIGINAL `observedAtMs`, so age keeps
 * growing across repeated failed polls instead of resetting each time.
 *
 * @param {TelemetryField} field
 * @param {string} reason
 * @returns {TelemetryField}
 */
export function staleField(field, reason) {
  if (field.state === FIELD_STATES.UNAVAILABLE) {
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
 * True when a panel may render `field.value` as a number/graphic.
 * `stale` counts as renderable — but the panel must show it as aged.
 *
 * @param {TelemetryField} field
 * @returns {boolean}
 */
export function hasValue(field) {
  return field.state !== FIELD_STATES.UNAVAILABLE;
}

/**
 * The one place the em-dash lives. Every panel formats through this so an
 * unavailable field looks identical everywhere in the page.
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
