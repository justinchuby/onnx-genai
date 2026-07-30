// Copyright (c) Microsoft Corporation.
//
// The one place the dashboard decides what a field's `state` MEANS.
//
// WHY THIS FILE EXISTS — and why it should eventually be deleted:
//
// Two Field vocabularies exist in this project right now. `demo-ux.md` §3.2
// specifies `'ok' | 'pending' | 'unavailable'`; `telemetry-field.js` on disk
// implements `'measured' | 'unavailable' | 'stale'`. Both are defensible and
// each carries a state the other lacks, so a resolution is pending in the
// contract-team group (proposed: the four-state union below).
//
// Until that lands, EVERY dashboard panel routes its state check through this
// module. That is deliberate: the alternative is seven panels each guessing at
// the vocabulary, which is exactly how one of them ends up treating an unknown
// string as renderable and prints a documented zero. Here, an unrecognised
// state is UNAVAILABLE — the safe direction. An honest em-dash for a value we
// could have shown is a cosmetic bug; a number for a value we could not is a
// fabricated measurement.
//
// When the vocabulary is ratified: collapse this to the ratified set, delete
// the alias table, and every panel keeps working unchanged.

/**
 * The dashboard's internal render states. Panels branch on these, never on the
 * raw string off the wire.
 *
 * @typedef {'ok' | 'pending' | 'stale' | 'unavailable'} RenderState
 *
 * - `ok`          — a genuine measurement, including a genuine zero. Render it
 *                   at full contrast with no apology (demo-ux.md §4.1).
 * - `pending`     — measurable, but no sample has arrived yet. Renders `···`.
 *                   Distinct from `unavailable` because pending resolves on its
 *                   own and unavailable does not; telling a visitor to wait for
 *                   a number that will never come is its own small dishonesty.
 * - `stale`       — was measured, but the latest poll did not refresh it.
 *                   Renderable, but MUST be shown as aged. A frozen number
 *                   presented as live is the same class of error as a
 *                   fabricated one (demo-ux.md §5.1).
 * - `unavailable` — no value exists and none is coming without a server or
 *                   configuration change. Renders `—` (demo-ux.md §4.1).
 */

/** @type {Readonly<Record<string, RenderState>>} */
export const RENDER_STATES = Object.freeze({
  OK: 'ok',
  PENDING: 'pending',
  STALE: 'stale',
  UNAVAILABLE: 'unavailable',
});

/**
 * Wire state -> render state. Covers both live vocabularies.
 *
 * @type {Readonly<Record<string, RenderState>>}
 */
const STATE_ALIASES = Object.freeze({
  ok: RENDER_STATES.OK,
  measured: RENDER_STATES.OK,
  pending: RENDER_STATES.PENDING,
  awaiting: RENDER_STATES.PENDING,
  stale: RENDER_STATES.STALE,
  unavailable: RENDER_STATES.UNAVAILABLE,
});

/**
 * Normalise any field-like object to a render state.
 *
 * A null/undefined field is `unavailable`, not a crash: `store.field()` is
 * contractually total, but a panel must not be the thing that white-screens the
 * page if that contract is ever broken.
 *
 * @param {{state?: string, value?: unknown}|null|undefined} field
 * @returns {RenderState}
 */
export function renderStateOf(field) {
  if (!field || typeof field.state !== 'string') {
    return RENDER_STATES.UNAVAILABLE;
  }
  const mapped = STATE_ALIASES[field.state];
  if (mapped === undefined) {
    return RENDER_STATES.UNAVAILABLE;
  }
  // A field claiming to be measured while carrying no value is a store bug.
  // Render the absence rather than propagating `null` into a formatter, where
  // it would surface as "null" or, worse, coerce to 0 in arithmetic.
  if (
    (mapped === RENDER_STATES.OK || mapped === RENDER_STATES.STALE) &&
    (field.value === null || field.value === undefined)
  ) {
    return RENDER_STATES.UNAVAILABLE;
  }
  return mapped;
}

/**
 * True when `field.value` may be read and rendered as a value.
 *
 * THIS IS THE GUARD. The project's single most important rule is that
 * `field.state` is read before `field.value`; calling this function IS reading
 * the state, so any panel that reaches a value through it is correct by
 * construction. Reviewers grep for `.value` not preceded by one of these.
 *
 * `stale` is renderable — but see {@link isStale}: the caller still owes the
 * visitor an age indication.
 *
 * @param {{state?: string, value?: unknown}|null|undefined} field
 * @returns {boolean}
 */
export function isRenderable(field) {
  const state = renderStateOf(field);
  return state === RENDER_STATES.OK || state === RENDER_STATES.STALE;
}

/**
 * True when the value is real but was not refreshed by the latest poll.
 * @param {{state?: string, value?: unknown}|null|undefined} field
 * @returns {boolean}
 */
export function isStale(field) {
  return renderStateOf(field) === RENDER_STATES.STALE;
}

/**
 * True when no value exists and none is coming without a server change.
 * @param {{state?: string, value?: unknown}|null|undefined} field
 * @returns {boolean}
 */
export function isUnavailable(field) {
  return renderStateOf(field) === RENDER_STATES.UNAVAILABLE;
}

/**
 * True when the field is measurable but has produced no sample yet.
 * @param {{state?: string, value?: unknown}|null|undefined} field
 * @returns {boolean}
 */
export function isPending(field) {
  return renderStateOf(field) === RENDER_STATES.PENDING;
}

/**
 * Read a numeric value, or `null` if the field is not renderable.
 *
 * Use this anywhere a panel needs to do arithmetic. It makes the unavailable
 * case impossible to skip, because the result is `null` and `null` does not
 * silently behave like a number in the comparisons panels actually write.
 *
 * @param {{state?: string, value?: unknown}|null|undefined} field
 * @returns {number|null}
 */
export function numericValueOf(field) {
  if (!isRenderable(field)) {
    return null;
  }
  const value = Number(field.value);
  return Number.isFinite(value) ? value : null;
}

/**
 * Divide two fields into a ratio field-like object, refusing to invent one.
 *
 * This exists because of the specific trap named in demo-ux.md §5.3: batch
 * occupancy has a real numerator and an unavailable denominator, and the
 * tempting move is to substitute the `DEFAULT_MAX_BATCH = 4` literal read out
 * of `state.rs:25`. That would be a fabricated measurement wearing a division
 * sign. A ratio is unavailable unless BOTH inputs are renderable, and it is
 * unavailable — not zero, not infinity — when the denominator is zero.
 *
 * @param {{state?: string, value?: unknown}|null|undefined} numerator
 * @param {{state?: string, value?: unknown}|null|undefined} denominator
 * @param {object} [options]
 * @param {string} [options.unavailableReason] Shown when the ratio cannot be formed.
 * @param {string} [options.label]
 * @returns {{value: number|null, state: RenderState, source: string, unit: string, label: string, reason?: string}}
 */
export function ratioField(numerator, denominator, options = {}) {
  const {
    unavailableReason = 'This ratio needs both a numerator and a denominator, and one of them is not measured.',
    label = 'ratio',
  } = options;

  const top = numericValueOf(numerator);
  const bottom = numericValueOf(denominator);

  if (top === null || bottom === null || bottom === 0) {
    return {
      value: null,
      state: RENDER_STATES.UNAVAILABLE,
      source: 'derived',
      unit: '%',
      label,
      reason: unavailableReason,
    };
  }

  return {
    value: (top / bottom) * 100,
    // Staleness is contagious: a ratio is only as fresh as its stalest input.
    state: isStale(numerator) || isStale(denominator) ? RENDER_STATES.STALE : RENDER_STATES.OK,
    source: 'derived',
    unit: '%',
    label,
  };
}
