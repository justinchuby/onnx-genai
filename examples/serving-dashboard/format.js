// Copyright (c) Microsoft Corporation.
//
// The rendering vocabulary: the ONE place a provenance envelope becomes pixels.
//
// telemetry-field.js decides what is true. This file decides how that truth
// looks. Keeping them apart matters because the interesting bug is not a
// mis-formatted number, it is a field whose state was quietly ignored — and
// that bug is only preventable if there is exactly one function every panel
// must call to turn a field into text.
//
// THE RULE THIS FILE ENFORCES: absence is never formatted as a value, and a
// claim is never rendered without its strength attached. A stale reading shows
// its age in words, an estimate shows a tilde, a derived number is marked. All
// three are visible WITHOUT hovering, because a visitor scanning a dashboard
// does not hover, and AC25 forbids encoding meaning in colour alone.

import { FIELD_STATES, SOURCE_CLASSES, hasValue, describeField } from './telemetry-field.js';
import { isAbsolutePathValue } from './absolute-path.mjs';

export { describeField };

export const ABSOLUTE_PATH_REASON =
  'Hidden because the server reported an absolute filesystem path instead of a display-safe identifier.';

/**
 * Prevent a field from disclosing a filesystem path at the display boundary.
 *
 * Relative and namespaced strings, numbers, booleans, and absent values are
 * returned unchanged. Absolute paths become explicitly unavailable rather
 * than being silently truncated into a misleading identifier.
 *
 * @param {import('./telemetry-field.js').TelemetryField} field
 * @returns {import('./telemetry-field.js').TelemetryField}
 */
export function displaySafeField(field) {
  if (!isAbsolutePathValue(field?.value)) return field;
  return {
    ...field,
    value: null,
    state: FIELD_STATES.UNAVAILABLE,
    reason: ABSOLUTE_PATH_REASON,
  };
}

/**
 * The AC7 source-class badge: one glyph per provenance class, rendered beside
 * the value so the KIND of claim is legible at a glance.
 *
 * `source` on a field is an endpoint (`/v1/status`) because that is precise
 * enough to curl. `sourceClass` is the axis a reader needs, and this is the
 * documented mapping between them, so no panel invents its own classifier.
 */
export const SOURCE_CLASS_BADGES = Object.freeze({
  [SOURCE_CLASSES.SERVER]: Object.freeze({
    glyph: 'ˢ',
    name: 'Server-measured',
    description: 'Read directly from a server response.',
  }),
  [SOURCE_CLASSES.CLIENT]: Object.freeze({
    glyph: 'ᶜ',
    name: 'Client-measured',
    description: 'Timed in this browser, not reported by the server.',
  }),
  [SOURCE_CLASSES.DERIVED]: Object.freeze({
    glyph: 'ᴰ',
    name: 'Derived',
    description: 'Computed by arithmetic on other measured values.',
  }),
  [SOURCE_CLASSES.ESTIMATED]: Object.freeze({
    glyph: 'ᴱ',
    name: 'Estimated',
    description:
      'A model standing in for a measurement. The formula is stated wherever ' +
      'this appears; if we cannot state the formula, we do not show the number.',
  }),
});

/** Every state this file knows how to render. Anything else is refused. */
const KNOWN_STATES = new Set(Object.values(FIELD_STATES));

/** Shown instead of a value whenever there is no value. Never a 0. */
export const ABSENT_TEXT = '—';

/** Shown before the first sample arrives. Distinct from absence: this resolves. */
export const PENDING_TEXT = '···';

/**
 * Shown when a metric is meaningless on THIS execution path by design.
 *
 * Deliberately not `ABSENT_TEXT`. `—` is apologetic: it says we could not
 * measure something we intended to, and it resolves when someone does the
 * work. `n/a` says the question does not apply here — asking how often a cache
 * was reused on an engine that never consults that cache. Rendering them identically tells a
 * first-time visitor that half a correctly-working dashboard is broken, which
 * is why `not-applicable` was ratified as a state of its own rather than
 * folded into `unavailable`.
 */
export const NOT_APPLICABLE_TEXT = 'n/a';

/**
 * Shown when a reading is known to be stale but carries no timestamp.
 *
 * The alternative is worse than it looks: `nowMs - (observedAtMs ?? nowMs)` is
 * zero, which renders "0s old" — an undated value ASSERTING that it is
 * perfectly fresh, on the one code path that already knows it is not. An
 * unknown age must read as unknown.
 */
export const UNKNOWN_AGE_TEXT = 'age unknown';

/**
 * Human age for a stale reading. Seconds up to a minute, then minutes: a
 * dashboard that has been dead for six minutes should not say "374s old",
 * which reads as precision about something we are admitting we do not know.
 *
 * @param {number} ageMs
 * @returns {string}
 */
export function formatAge(ageMs) {
  const seconds = Math.max(0, Math.round(ageMs / 1000));
  if (seconds < 60) return `${seconds}s old`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m old`;
  return `${Math.floor(minutes / 60)}h old`;
}

/**
 * Turn a field into everything needed to render it, in one call.
 *
 * Panels should use this rather than reading `field.state` themselves. Every
 * `if (field.state === ...)` written in a panel is a place the next state we
 * add gets missed — which has already happened once in this codebase, when a
 * branch checking only `unavailable` silently swallowed `not-applicable`.
 *
 * @param {import('./telemetry-field.js').TelemetryField} field
 * @param {object} [options]
 * @param {(value: any) => string} [options.format] Formats the raw value.
 * @param {boolean} [options.withUnit] Append `field.unit` to the value.
 * @param {number} [options.nowMs]
 * @returns {{
 *   text: string,
 *   state: string,
 *   badge: string|null,
 *   badgeLabel: string|null,
 *   title: string,
 *   hasValue: boolean,
 *   isEstimate: boolean,
 *   ageText: string|null,
 *   provenanceWarning: string|null,
 * }}
 */
export function formatField(field, { format, withUnit, nowMs = Date.now() } = {}) {
  // A caller who supplies its own formatter owns the ENTIRE display string,
  // unit included -- `formatTokenCount` returns "32,768 tokens", and appending
  // field.unit on top of that produced "32,768 tokens tokens" on the live page
  // while every unit test passed. So the unit is only appended when we did the
  // formatting ourselves, unless the caller asks otherwise explicitly.
  const formatValue = format ?? defaultFormat;
  const appendUnit = withUnit ?? format === undefined;
  const badge = SOURCE_CLASS_BADGES[field.sourceClass] ?? null;
  const present = hasValue(field);
  const isEstimate = field.sourceClass === SOURCE_CLASSES.ESTIMATED;

  // THE TERMINAL BRANCH. Every state is handled BY NAME below; anything else
  // is a typo or a module written against an older spec, and the one thing it
  // must never do is render its value as though it were a measurement. A
  // default branch that renders as fine is how AC6 dies quietly.
  if (!KNOWN_STATES.has(field.state)) {
    console.error(
      `[format] unknown field state ${JSON.stringify(field.state)} for ${
        field.label ?? 'an unlabelled field'
      }. Refusing to render its value, because an unrecognised state is not a measurement.`,
    );
    return {
      text: ABSENT_TEXT,
      state: field.state,
      badge: badge?.glyph ?? null,
      badgeLabel: badge?.name ?? null,
      title: `This value cannot be displayed: its state ${JSON.stringify(field.state)} is not one this page knows how to render.`,
      hasValue: false,
      isEstimate: false,
      ageText: null,
      provenanceWarning: field.provenanceWarning ?? null,
    };
  }

  let text;
  let ageText = null;

  if (!present) {
    // Three distinguishable absences, distinguishable IN TEXT rather than in
    // colour, so the difference survives grayscale and a screen reader.
    if (field.state === FIELD_STATES.PENDING) text = PENDING_TEXT;
    else if (field.state === FIELD_STATES.NOT_APPLICABLE) text = NOT_APPLICABLE_TEXT;
    else text = ABSENT_TEXT;
  } else {
    const unit = appendUnit && field.unit ? ` ${field.unit}` : '';
    // The tilde is not decoration. An estimate that looks identical to a
    // measurement IS a fabricated measurement, however carefully the tooltip
    // is worded, because the tooltip is not what gets read.
    text = `${isEstimate ? '~' : ''}${formatValue(field.value)}${unit}`;

    if (field.state === FIELD_STATES.STALE) {
      // AC25: the age must be in WORDS. A colour shift alone disappears in
      // grayscale and for colourblind readers, and "this number is 12 seconds
      // out of date" is information, not styling.
      //
      // An undated stale field says so. Defaulting its age to `nowMs` renders
      // "0s old", which is the field asserting perfect freshness at the exact
      // moment we know it is not fresh — a stronger false claim than the one
      // the staleness treatment exists to prevent.
      ageText =
        typeof field.observedAtMs === 'number'
          ? formatAge(nowMs - field.observedAtMs)
          : UNKNOWN_AGE_TEXT;
      text = `${text} · ${ageText}`;
    }
  }

  return {
    text,
    state: field.state,
    badge: badge?.glyph ?? null,
    badgeLabel: badge?.name ?? null,
    title: describeField(field, nowMs),
    hasValue: present,
    isEstimate,
    ageText,
    provenanceWarning: field.provenanceWarning ?? null,
  };
}

/**
 * A complete sentence naming a field and its current state, for panel
 * `describe()` implementations and screen-reader text.
 *
 * Panels compose these into the chart's `aria-label` (AC28), so this must read
 * as prose rather than as a label-colon-value pair.
 *
 * @param {string} name How to refer to the field in the sentence.
 * @param {import('./telemetry-field.js').TelemetryField} field
 * @param {object} [options]
 * @returns {string}
 */
export function describeFieldText(name, field, options = {}) {
  const { text, hasValue: present, isEstimate } = formatField(field, options);
  if (!present) {
    if (field.state === FIELD_STATES.PENDING) return `${name} is still loading`;
    if (field.state === FIELD_STATES.NOT_APPLICABLE) {
      return `${name} does not apply here: ${field.reason}`;
    }
    return `${name} is unavailable: ${field.reason}`;
  }
  const qualifier = isEstimate ? 'estimated at' : 'is';
  return `${name} ${qualifier} ${text}`;
}

/** @param {any} value */
function defaultFormat(value) {
  if (typeof value !== 'number') return String(value);
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(2);
}
