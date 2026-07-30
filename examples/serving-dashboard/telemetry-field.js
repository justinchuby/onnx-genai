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
 * @typedef {'measured' | 'pending' | 'stale' | 'unavailable' | 'not-applicable'} FieldState
 *
 * These are the WIRE VALUES, which is what `field.state` is actually compared
 * against. Constant names and wire values now AGREE: `FIELD_STATES.MEASURED`
 * emits `'measured'`. They disagreed once, and it cost two separate bugs in
 * opposite directions — a CSS selector that matched nothing, and a documented
 * comparison that was never true — because a mismatch between a name and its
 * value has no symptom at the point of use. Always compare against
 * `FIELD_STATES.*` rather than a literal.
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
 *                   configuration change: the server cannot measure it yet, or
 *                   the endpoint that carries it is disabled. This is OUR GAP,
 *                   and it is a PROMISE that someone could do the work.
 *                   `value` is ALWAYS `null`. Render an em-dash, never a zero.
 * - `not-applicable` — the field is structurally inapplicable to this decode
 *                   path: that subsystem is never consulted here, so there is
 *                   nothing to measure and never will be. An ARCHITECTURAL
 *                   FACT, not a gap — do not render it as an apology.
 *                   `value` is ALWAYS `null`. Render `n/a`, never a zero.
 */

/**
 * @typedef {object} TelemetryField
 * @property {number|string|boolean|Array|object|null} value
 *   The measurement. `null` whenever `state` is `unavailable` or `pending`.
 * @property {FieldState} state
 * @property {string|null} source
 *   The ENDPOINT, precisely enough to curl: `'/v1/status'`, `'/v1/debug/kv'`.
 *   `null` when no endpoint applies — a client measurement, a derived value, or
 *   a field the server never exposes.
 *
 *   NEVER put a class or a state name here. It carried the sentinels
 *   `'unavailable'`, `'unknown'` and `'derived'` until D161, which is what
 *   forced panels to sniff `source.startsWith('/')` to tell an endpoint from a
 *   category — branching on a substring, and the exact thing we forbid for
 *   `reason`. The class lives in `sourceClass`, which is authoritative; ask it,
 *   never parse this.
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
 *   For `sourceClass === 'derived'`, the field keys this was computed from, so the
 *   footer provenance table can be generated rather than hand-maintained.
 */

/** Legal values of `TelemetryField.state`. */
/**
 * WHERE a value came from — a provenance CLASS, and a different axis from
 * `source`, which names the ENDPOINT.
 *
 * Both are needed and neither substitutes for the other: the class is what a
 * viewer needs on hover ("did you measure this or work it out?"), while the
 * endpoint is what makes a claim auditable ("`/v1/status`, and here is the
 * file:line where it is hardcoded"). Collapsing them would cost one of those.
 *
 * @typedef {'server' | 'client' | 'derived' | 'estimated'} SourceClass
 *
 * - `server`    — read verbatim off a server response.
 * - `client`    — measured in the browser (e.g. wall-clock time to first token).
 * - `derived`   — computed by us from other fields. Real, but owes its inputs.
 * - `estimated` — approximated, with error bounds. Must be labelled as such.
 */
export const SOURCE_CLASSES = Object.freeze({
  SERVER: 'server',
  CLIENT: 'client',
  DERIVED: 'derived',
  ESTIMATED: 'estimated',
});

export const FIELD_STATES = Object.freeze({
  /**
   * The state of a field the server computed, just now. Includes a genuine
   * zero.
   *
   * `MEASURED` and `'measured'` agree, which is the whole point. The constant
   * and its value disagreed once — `MEASURED: 'ok'` — and that is a landmine
   * with no symptom: `field.state === 'measured'` was false for every measured
   * field on the page, and because the formatter fell through to rendering a
   * plain number anyway, the comparison failed SILENTLY while the output still
   * looked correct.
   *
   * The name and the wire value must move together or not at all. Changing one
   * alone reproduces the original bug in the opposite direction, which is why
   * state-channel.test.js asserts BOTH halves — the enum value AND the
   * `[data-state='measured']` selector in shell.css — as a single atomic pair.
   *
   * Never global-replace the string `'ok'` to make this change: `status: 'ok'`
   * is the HTTP health payload, and renaming that fakes an unreachable server.
   */
  MEASURED: 'measured',
  PENDING: 'pending',
  STALE: 'stale',
  UNAVAILABLE: 'unavailable',
  NOT_APPLICABLE: 'not-applicable',
});

/**
 * THE THREE KINDS OF ZERO (demo-spec.md §3, binding).
 *
 * A `0` on the wire is byte-identical in all three cases below, so the wire
 * cannot tell them apart and neither can a panel. Only the provenance table
 * can, which is why this distinction lives in the type rather than in copy:
 *
 * 1. `ok` with value 0 — the question was asked and the answer really is zero.
 *    Renders as a stark `0`. This is DATA and must not be hidden.
 * 2. `unavailable` — the server hardcodes a stub and never computes it
 *    (e.g. tokens_per_second at routes/admin.rs:63). Em-dash, hover names the
 *    stub. It could be fixed by plumbing it.
 * 3. `not-applicable` — the subsystem exists but THIS code path never consults
 *    it, so the question is never asked (e.g. the prefix cache on a
 *    static-cache server, where the batching path bypasses the trie entirely).
 *    Em-dash, hover explains WHY. Plumbing would not fix it; it is a true
 *    statement about the architecture.
 *
 * Collapsing 2 and 3 into one bucket destroys a fact the visitor needs: one is
 * a gap in the server, the other is a property of it. The same wire value can
 * land in different buckets on different servers — prefix-cache hits are a
 * genuine measured 0 on the dynamic server and not-applicable on the scatter
 * server — so the classification depends on origin, not on the number.
 */

/**
 * Build a genuinely-measured field.
 *
 * Do not call this for a value you did not read off a live response. If you
 * find yourself passing a literal, you want {@link unavailableField}.
 *
 * @param {number|string|boolean|Array|object} value
 * @param {object} options
 * @param {string|null} options.source   The ENDPOINT, e.g. `'/v1/status'`, or
 *                                       `null` for a client or derived value.
 *                                       The CLASS goes in `sourceClass`.
 * @param {string|null} [options.unit]
 * @param {number} [options.observedAtMs] Defaults to now.
 * @returns {TelemetryField}
 */
export function measuredField(
  value,
  {
    source,
    sourceClass = SOURCE_CLASSES.SERVER,
    origin = null,
    originModelId = null,
    label = null,
    unit = null,
    observedAtMs = Date.now(),
    derivedFrom = null,
    provenanceWarning = null,
  },
) {
  if (value === null || value === undefined) {
    throw new TypeError(
      'measuredField() was given a null/undefined value. A missing value is not a ' +
        'measurement — use unavailableField(reason) so the UI can render an em-dash.',
    );
  }
  if (!source && sourceClass === SOURCE_CLASSES.SERVER) {
    throw new TypeError(
      'measuredField() requires a `source` endpoint for a server-sourced value, so the ' +
        'claim can be audited by curling it. If this was measured in the browser or ' +
        'computed from other fields, pass the matching `sourceClass` — a client or ' +
        'derived value legitimately has no endpoint, and `source: null` is how it says so.',
    );
  }
  return Object.freeze({
    value,
    state: FIELD_STATES.MEASURED,
    source,
    sourceClass,
    origin,
    originModelId,
    label,
    reason: null,
    unit,
    observedAtMs,
    // Set when WE computed the value from other fields rather than reading it
    // off a response. Still a real measurement, but a panel showing it owes the
    // viewer the inputs — a number we derived needs more disclosure, not less.
    derivedFrom,
    // Set when the provenance table said this field should be a placeholder
    // but the server sent something else. The value IS shown -- hiding a real
    // number is the exact failure this warns about -- but never silently.
    provenanceWarning,
  });
}

/**
 * Build a not-applicable field: the subsystem exists, but this code path never
 * consults it, so the question is never asked.
 *
 * This is NOT a degraded `unavailable`. `unavailable` says "the server does not
 * compute this yet", which implies it could. `not-applicable` says "asking this
 * of this server is meaningless", which is a permanent, true statement about
 * the architecture and is often the most interesting thing on the page.
 *
 * @param {string} reason Must explain WHY the path bypasses it.
 * @param {object} [options]
 * @returns {TelemetryField}
 */
export function notApplicableField(
  reason,
  {
    source = null,
    sourceClass = SOURCE_CLASSES.SERVER,
    origin = null,
    originModelId = null,
    label = null,
    unit = null,
  } = {},
) {
  if (!reason) {
    throw new TypeError(
      'notApplicableField() requires a reason explaining why this path never consults the ' +
        'subsystem. Without it the state is indistinguishable from a missing feature.',
    );
  }
  return Object.freeze({
    value: null,
    state: FIELD_STATES.NOT_APPLICABLE,
    source,
    sourceClass,
    origin,
    originModelId,
    label,
    reason,
    unit,
    observedAtMs: null,
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
 * @param {string|null} [options.source] The endpoint that WOULD carry it, if
 *                                       known. `null` when none would.
 * @param {string|null} [options.unit]
 * @returns {TelemetryField}
 */
export function unavailableField(
  reason,
  {
    source = null,
    sourceClass = SOURCE_CLASSES.SERVER,
    origin = null,
    originModelId = null,
    label = null,
    unit = null,
  } = {},
) {
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
    sourceClass,
    origin,
    originModelId,
    label,
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
export function pendingField(
  reason,
  {
    source = null,
    sourceClass = SOURCE_CLASSES.SERVER,
    origin = null,
    originModelId = null,
    label = null,
    unit = null,
  } = {},
) {
  if (!reason) {
    throw new TypeError('pendingField() requires a reason explaining what is being waited on.');
  }
  return Object.freeze({
    value: null,
    state: FIELD_STATES.PENDING,
    source,
    sourceClass,
    origin,
    originModelId,
    label,
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
  if (
    field.state === FIELD_STATES.UNAVAILABLE ||
    field.state === FIELD_STATES.PENDING ||
    field.state === FIELD_STATES.NOT_APPLICABLE
  ) {
    // Absence has no age, and a question never asked cannot become stale.
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
export function derivedField(inputs, compute, { unit = null, label = null, undefinedReason } = {}) {
  const keys = Object.keys(inputs);

  // NOT-APPLICABLE IS CHECKED FIRST, AND IT DOMINATES EVERY OTHER STATE.
  //
  // Without this branch a structurally-bypassed input falls through to the
  // compute path and the result is returned as `ok` at full contrast, with no
  // badge and no reason -- a confident number derived from a counter that is
  // never observed. On the batching profile `prefix_cache.*` is pinned to a
  // literal 0 in batched.rs, so a derived hit RATE over it renders as a
  // measurement of something nothing ever measured. That is precisely the
  // fabrication the five-state vocabulary exists to prevent, arriving through
  // the one path that bypasses it.
  //
  // It dominates rather than merely participating because it is the STRONGEST
  // claim available. `unavailable` and `pending` both leave the door open --
  // someone may plumb it, or the next poll may fill it -- so a derivation over
  // them may yet succeed. `not-applicable` says this execution path can never
  // consult that subsystem, so the derivation can never succeed either.
  // Reporting `unavailable` here would promise future work that will never
  // happen, which is the exact distinction between the two states.
  const inapplicable = keys.filter((key) => inputs[key].state === FIELD_STATES.NOT_APPLICABLE);
  if (inapplicable.length > 0) {
    return notApplicableField(
      `Cannot be derived here: ${inapplicable.join(', ')} ${
        inapplicable.length === 1 ? 'is' : 'are'
      } not applicable on this execution path. ${inputs[inapplicable[0]].reason}`,
      { source: null, sourceClass: SOURCE_CLASSES.DERIVED, label, unit },
    );
  }

  const blocking = keys.filter((key) => inputs[key].state === FIELD_STATES.UNAVAILABLE);
  if (blocking.length > 0) {
    return unavailableField(
      `Cannot be derived because ${blocking.join(', ')} ${
        blocking.length === 1 ? 'is' : 'are'
      } unavailable: ${inputs[blocking[0]].reason}`,
      { source: null, sourceClass: SOURCE_CLASSES.DERIVED, label, unit },
    );
  }

  // Pending is also contagious, but it resolves on its own, so the result is
  // pending rather than unavailable — the visitor should wait, not give up.
  const waiting = keys.filter((key) => inputs[key].state === FIELD_STATES.PENDING);
  if (waiting.length > 0) {
    return pendingField(
      `Waiting on ${waiting.join(', ')} before this can be derived.`,
      { source: null, sourceClass: SOURCE_CLASSES.DERIVED, label, unit },
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
      { source: null, sourceClass: SOURCE_CLASSES.DERIVED, label, unit },
    );
  }

  const anyStale = keys.some((key) => inputs[key].state === FIELD_STATES.STALE);
  const observedAtMs = Math.min(...keys.map((key) => inputs[key].observedAtMs ?? Date.now()));
  return Object.freeze({
    value: result,
    state: anyStale ? FIELD_STATES.STALE : FIELD_STATES.MEASURED,
    source: null,
    sourceClass: SOURCE_CLASSES.DERIVED,
    // All inputs must share an origin for the result to be attributable to one
    // server; mixing two servers into one number would make it unattributable.
    origin: keys.every((key) => inputs[key].origin === inputs[keys[0]].origin)
      ? inputs[keys[0]].origin
      : null,
    label,
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
 * REMOVED: formatFieldText.
 *
 * It lived here and had two defects @0837fdf9 caught, both structural rather
 * than cosmetic:
 *
 *  - it branched on `unavailable`, `not-applicable` and `pending`, then fell
 *    through to `return format(field.value)`. So ANY state it did not know --
 *    a typo, or a module written against an older spec -- rendered its value
 *    as though it were a measurement. A default branch that renders as fine is
 *    how AC6 dies quietly.
 *  - `stale` returned the bare number, so a dashboard whose server died 12
 *    seconds ago looked live. Staleness appeared only on hover, which defeats
 *    the reason the state exists.
 *
 * Both are fixed in `format.js`, which handles every state by name and has a
 * terminal branch for the unknown. Rather than fix them twice, this one is
 * gone: two rendering paths where one is safe and one is not is the actual
 * defect, and deleting the unsafe path is the only fix that stays fixed.
 *
 * Use `formatField()` from `format.js`.
 */

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
  const on = field.origin ? ` on the ${field.origin} server` : '';
  // `source` names an ENDPOINT and is null when none is known. It must never be
  // interpolated unguarded: it used to default to the sentinel 'unavailable',
  // which rendered "would come from unavailable" -- a state name in an
  // attribution slot, and the only place a screen-reader user meets provenance.
  const from = field.source ? ` (would come from ${field.source}${on})` : on ? ` (${on.trim()})` : '';
  if (field.state === FIELD_STATES.NOT_APPLICABLE) {
    // Deliberately NOT phrased as "unavailable": nothing is missing here. The
    // wording has to make clear that plumbing would not produce a value.
    return `Not applicable${on} — ${field.reason}`;
  }
  if (field.state === FIELD_STATES.UNAVAILABLE) {
    return `Unavailable — ${field.reason}${from}`;
  }
  if (field.state === FIELD_STATES.PENDING) {
    const waitingFrom = field.source ? ` (from ${field.source})` : '';
    return `Waiting for the first measurement — ${field.reason}${waitingFrom}`;
  }
  const unitSuffix = field.unit ? ` ${field.unit}` : '';
  const ageSeconds = Math.round((nowMs - (field.observedAtMs ?? nowMs)) / 1000);
  if (field.state === FIELD_STATES.STALE) {
    const lastFrom = field.source ? ` from ${field.source}` : '';
    return `${field.value}${unitSuffix} — STALE, last measured ${ageSeconds}s ago${lastFrom}. ${field.reason}`;
  }
  const provenance = field.derivedFrom
    ? `derived from ${field.derivedFrom.join(', ')}`
    : field.source
      ? `source ${field.source}`
      : `source ${field.sourceClass}`;
  return `${field.value}${unitSuffix} — measured${on}, ${provenance}`;
}

/** @param {any} value */
function defaultFormat(value) {
  return typeof value === 'number' ? String(value) : String(value);
}
