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

export { describeField };

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

/** Shown instead of a value whenever there is no value. Never a 0. */
export const ABSENT_TEXT = '—';

/** Shown before the first sample arrives. Distinct from absence: this resolves. */
export const PENDING_TEXT = '···';

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
export function formatField(
  field,
  { format = defaultFormat, withUnit = true, nowMs = Date.now() } = {},
) {
  const badge = SOURCE_CLASS_BADGES[field.sourceClass] ?? null;
  const present = hasValue(field);
  const isEstimate = field.sourceClass === SOURCE_CLASSES.ESTIMATED;

  let text;
  let ageText = null;

  if (!present) {
    text = field.state === FIELD_STATES.PENDING ? PENDING_TEXT : ABSENT_TEXT;
  } else {
    const unit = withUnit && field.unit ? ` ${field.unit}` : '';
    // The tilde is not decoration. An estimate that looks identical to a
    // measurement IS a fabricated measurement, however carefully the tooltip
    // is worded, because the tooltip is not what gets read.
    text = `${isEstimate ? '~' : ''}${format(field.value)}${unit}`;

    if (field.state === FIELD_STATES.STALE) {
      // AC25: the age must be in WORDS. A colour shift alone disappears in
      // grayscale and for colourblind readers, and "this number is 12 seconds
      // out of date" is information, not styling.
      ageText = formatAge(nowMs - (field.observedAtMs ?? nowMs));
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
