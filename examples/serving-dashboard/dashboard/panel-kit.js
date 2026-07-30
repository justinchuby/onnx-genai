// Copyright (c) Microsoft Corporation.
//
// Shared building blocks for the dashboard panels.
//
// TWO JOBS, and the second one is the important one:
//
// 1. Small DOM constructors so seven panels look like one page instead of seven
//    people's opinions.
//
// 2. THE SINGLE IMPORT SEAM. demo-ux.md §3.1 puts `format.js` on the demo dev's
//    side of the ownership line, and §3.3 rule 5 forbids panels from rendering
//    a raw number outside it. Every dashboard import of that module is routed
//    through this file, so if the seam moves — and a proposal to move
//    `renderSeries` into `dashboard/` is open in contract-team — it is one edit
//    here rather than an edit in each of seven panels.
//
// While `format.js` is still landing, this module provides its own conforming
// implementations built strictly from the designer's tokens and the §4.1 markup
// in demo-ux.md. They are a temporary local implementation of someone else's
// contract, not a competing design: when format.js lands, the bodies below are
// replaced by re-exports and no panel changes.

import {
  IS_DEVELOPMENT,
  RENDER_STATES,
  ageMsOf,
  formatAge,
  isPastStaleCeiling,
  isRenderable,
  isStale,
  renderStateOf,
  resolveStaleCeilingMs,
} from './field-state.js';
import {
  describeSparkline,
  paintSparkline,
  planSparkline,
  tabulateSparkline,
} from './sparkline.js';
import { displaySafeField } from '../format.js';
import { DEBUG_GATED_ENDPOINTS, DEBUG_ENDPOINTS_FLAG } from '../telemetry-provenance.js';

/**
 * Source classes, as rendered by the AC7 provenance badge (demo-ux.md §4.5).
 *
 * @typedef {'server' | 'client' | 'derived' | 'estimated' | 'simulated'} SourceClass
 */

/**
 * The badge glyph and hover text for each source class.
 *
 * `estimated` and `simulated` escalate deliberately: a superscript is easy to
 * miss, and a claim that is modelled rather than measured must not be easy to
 * miss. Per the ratified no-simulated-baseline ruling nothing on this page
 * should ever need `simulated` — it exists so the language already has a loud,
 * ugly, honest word for it if something ever does.
 *
 * @type {Readonly<Record<SourceClass, {glyph: string, title: string}>>}
 */
export const SOURCE_BADGES = Object.freeze({
  server: { glyph: 'ˢ', title: 'Server counter — a real value read from the running server.' },
  client: { glyph: 'ᶜ', title: 'Measured in your browser from the response stream.' },
  derived: { glyph: 'ᴰ', title: 'Derived by arithmetic on measured inputs.' },
  estimated: { glyph: 'ᴱ', title: 'ESTIMATED, not measured — computed from a model, not observed.' },
  simulated: { glyph: 'SIM', title: 'SIMULATED — not measured at all.' },
});

/**
 * The badge for a value whose provenance the catalogue cannot identify.
 *
 * Deliberately NOT a member of `SOURCE_BADGES`. Two call sites in
 * `normaliseSourceClass` use `x in SOURCE_BADGES` as the validity predicate for
 * a source class, so adding a key here would make the literal string
 * `'unknown'` validate as a real, catalogued provenance — the opposite of what
 * it means.
 *
 * This exists because the previous fallback was `SOURCE_BADGES.derived`, which
 * answered "we do not know where this number came from" with "derived by
 * arithmetic on measured inputs". That is a positive claim about measurement,
 * awarded on the strength of no evidence at all. A fallback must make the
 * WEAKEST claim available, never a middle one: an unrecognised provenance has
 * to read as unrecognised, because the one thing we know about it is that we
 * do not know.
 */
export const UNKNOWN_SOURCE_BADGE = Object.freeze({
  glyph: '?',
  title:
    'UNKNOWN PROVENANCE — this value carries no source the dashboard recognises, ' +
    'so nothing is claimed about how it was obtained.',
});

/**
 * The source class used when no catalogued provenance can be identified.
 *
 * Kept as a named constant so the string is written once: it is deliberately
 * absent from `SOURCE_BADGES`, so a typo here would silently reintroduce the
 * fallback-to-`derived` behaviour this constant exists to remove.
 */
export const UNKNOWN_SOURCE_CLASS = 'unknown';

/** Reason copy reused across panels, taken verbatim from demo-ux.md §4.2. */
export const REASONS = Object.freeze({
  KV_NOT_EXPOSED:
    'KV page statistics are computed by the engine but not yet exposed over HTTP. ' +
    '`Engine::page_usage()` exists; the server does not call it.',
  DEBUG_GATED: 'Requires debug endpoints. Restart the server with `--enable-debug-endpoints`.',
  NO_MAX_BATCH:
    "Occupancy needs the engine's batch limit, which the server does not " +
    'surface. The sequence count is real; the denominator is not.',
  STATUS_DOCUMENTED_ZERO:
    '`/v1/status` reports this as a documented zero — the server records cumulative token ' +
    'totals only. This page derives throughput client-side instead.',
  CONTINUOUS_BATCH_OFF:
    "This model doesn't use a static KV cache, so the continuous batch driver is disabled and " +
    'requests run one at a time. Use a static-cache/scatter model to see batching.',
});

/**
 * Create an element with classes, text and attributes in one call.
 *
 * @param {string} tagName
 * @param {object} [spec]
 * @param {string|string[]} [spec.className]
 * @param {string} [spec.text]
 * @param {Record<string, string|number|boolean|null|undefined>} [spec.attrs]
 * @param {Array<Node|null|undefined>} [spec.children]
 * @returns {HTMLElement}
 */
export function element(tagName, spec = {}) {
  const node = document.createElement(tagName);
  if (spec.className) {
    // Accepts a string, a space-separated string, or an array. `classList.add`
    // throws on a token containing whitespace, so splitting here means no call
    // site has to remember which form is legal.
    const classes = (Array.isArray(spec.className) ? spec.className : [spec.className])
      .filter(Boolean)
      .flatMap((name) => String(name).split(/\s+/))
      .filter(Boolean);
    node.classList.add(...classes);
  }
  if (spec.text !== undefined) {
    node.textContent = spec.text;
  }
  for (const [name, value] of Object.entries(spec.attrs ?? {})) {
    if (value === null || value === undefined || value === false) {
      continue;
    }
    node.setAttribute(name, String(value));
  }
  for (const child of spec.children ?? []) {
    if (child) {
      node.append(child);
    }
  }
  return node;
}

/**
 * Format a number for display: tabular, unit-aware, sensibly precise.
 *
 * Precision is chosen by magnitude rather than by unit because the alternative
 * — a per-call `decimals` argument — guarantees that the same quantity is
 * rendered two ways in two panels.
 *
 * @param {number} value
 * @param {string} [unit]
 * @returns {string}
 */
export function formatNumber(value, unit = '') {
  if (!Number.isFinite(value)) {
    return '—';
  }
  const magnitude = Math.abs(value);
  let text;
  if (Number.isInteger(value) && magnitude < 1e6) {
    text = value.toLocaleString('en-US');
  } else if (magnitude >= 1000) {
    text = Math.round(value).toLocaleString('en-US');
  } else if (magnitude >= 100) {
    text = value.toFixed(0);
  } else if (magnitude >= 10) {
    text = value.toFixed(1);
  } else if (magnitude >= 1) {
    text = value.toFixed(2);
  } else if (magnitude === 0) {
    text = '0';
  } else {
    text = value.toFixed(3);
  }
  return unit === '%' ? `${text}%` : text;
}

/**
 * Format a millisecond duration at human resolution.
 *
 * @param {number} milliseconds
 * @returns {string}
 */
export function formatDuration(milliseconds) {
  if (!Number.isFinite(milliseconds)) {
    return '—';
  }
  if (milliseconds < 1) {
    return `${milliseconds.toFixed(2)} ms`;
  }
  if (milliseconds < 1000) {
    return `${Math.round(milliseconds)} ms`;
  }
  if (milliseconds < 60_000) {
    return `${(milliseconds / 1000).toFixed(2)} s`;
  }
  const minutes = Math.floor(milliseconds / 60_000);
  const seconds = (milliseconds % 60_000) / 1000;
  return `${minutes} m ${seconds.toFixed(0).padStart(2, '0')} s`;
}

/**
 * Build the AC7 provenance badge.
 *
 * @param {SourceClass} sourceClass
 * @param {string} [detail] Appended to the hover text, e.g. the endpoint path.
 * @returns {HTMLElement}
 */
export function sourceBadge(sourceClass, detail) {
  const badge = SOURCE_BADGES[sourceClass] ?? UNKNOWN_SOURCE_BADGE;
  const title = detail ? `${badge.title} · ${detail}` : badge.title;
  return element('abbr', {
    className: ['value__src', `value__src--${sourceClass}`],
    text: badge.glyph,
    attrs: { title },
  });
}

/**
 * The separator between a value and its age suffix.
 *
 * The ruled treatment for a stale value is literally `41 · 12s old`, and the
 * dot is not decoration. Without a real character the two spans are only held
 * apart by a flex gap, which means textContent reads "4112s old" — so the table
 * view, describe() and anyone who copies the number get a fused, wrong figure
 * while the screen looks correct. aria-hidden because the accessible name is
 * composed separately and should not contain "middle dot".
 */
function ageSeparator() {
  return element('span', {
    className: 'value__sep',
    text: '·',
    attrs: { 'aria-hidden': 'true' },
  });
}

/**
 * First sentence of a reason, for the one caption that renders on screen.
 *
 * The full `reason` stays in `title` and `aria-label`, so nothing is lost for
 * an auditor or a screen reader — this only decides what is shown unprompted.
 *
 * It has to be trimmed because these reasons carry their evidence inline (two
 * file:line citations each, ~350 characters), and a single panel can hold four
 * not-applicable fields. Rendered whole, that is well over a thousand
 * characters of near-identical prose stacked inside one card, which buries the
 * very sentence it exists to communicate. The first sentence is written to
 * stand alone in every case we author: it names the execution path and says
 * the question is never asked.
 *
 * Trimming is not the same as softening. The sentence still says IMPOSSIBLE
 * rather than ABSENT, which is the property demo-ux.md §20.2 actually
 * protects, and the citation is one hover away rather than deleted.
 *
 * @param {string} reason
 * @returns {string}
 */
function headlineSentence(reason) {
  const text = String(reason).trim();
  // Require a following space so a version number or a file:line citation
  // ("batched_static_decode.rs:53") is never mistaken for a sentence end.
  const match = /^(.+?[.!?])\s/s.exec(text);
  return match ? match[1] : text;
}

/**
 * Render a field as the §4.1 value element.
 *
 * THIS IS THE FUNCTION THAT MAKES THE HONESTY RULE MECHANICAL. It reads
 * `state` first and reaches `value` only on the branch where a value exists, so
 * a panel that renders through it cannot print a documented zero no matter how
 * carelessly it is written. Panels never format a value themselves.
 *
 * @param {{value?: unknown, state?: string, source?: string, unit?: string, label?: string, reason?: string, at?: number}|null|undefined} field
 * @param {object} [options]
 * @param {(value: any) => string} [options.format] Overrides the default formatter.
 * @param {boolean} [options.showUnit] Default true. The unit stays even when
 *   unavailable — WHICH thing is missing is itself information (§4.1).
 * @param {string} [options.label] Overrides `field.label` for the aria sentence.
 * @param {SourceClass} [options.sourceClass] Overrides `field.source`.
 * @returns {HTMLElement}
 */
export function renderField(field, options = {}) {
  field = displaySafeField(field);
  const state = renderStateOf(field, { strict: options.strict });
  const label = options.label ?? field?.label ?? 'value';
  const unit = field?.unit ?? '';
  const sourceClass = normaliseSourceClass(
    options.sourceClass ? { sourceClass: options.sourceClass } : field,
  );
  // '%' is excluded because `formatNumber` already appends it to the text.
  const unitIsSuffixable = Boolean(unit) && unit !== '%';
  const showUnit = options.showUnit !== false && unitIsSuffixable;

  // A custom formatter OWNS its unit, so the suffix span must not be appended
  // on top of it. `formatBytes` scales 5.35e9 to "5.35 GiB" and `formatDuration`
  // scales 192000 to "3 m 12 s"; appending the field's raw unit produced
  // "5.35 GiB bytes" and "3 m 12 s ms" on the served page.
  //
  // This is derived from `options.format` rather than left to each call site
  // to pass `showUnit: false`, because a caller who writes `format: formatBytes`
  // and forgets the flag reintroduces the defect silently — the number still
  // renders, just wrong. Deriving it makes the pairing structural.
  //
  // It applies ONLY to the branch that has a value. The placeholder branches
  // never run the formatter, so there is no unit to collide with, and they must
  // keep theirs: WHICH thing is missing is itself information (§4.1), and
  // "— bytes" says strictly more than "—".
  const showUnitWithValue = (options.showUnit ?? !options.format) && unitIsSuffixable;

  const ceilingMs = resolveStaleCeilingMs(options.staleCeilingMs);
  const nowMs = options.nowMs ?? Date.now();

  const wrapper = element('span', {
    className: 'value',
    attrs: { 'data-state': state, 'data-source': sourceClass },
  });

  if (state === RENDER_STATES.NOT_APPLICABLE) {
    // Deliberately not the same treatment as `unavailable`. Nothing is missing
    // here and nothing is coming: this code path structurally cannot reach the
    // subsystem. Saying "not measurable yet" would promise a value that will
    // never arrive.
    const reason = field?.reason ?? 'This server cannot produce this measurement.';
    wrapper.setAttribute(ROVING_ITEM_ATTR, '');
    wrapper.setAttribute('tabindex', '-1');
    wrapper.setAttribute('role', 'note');
    wrapper.setAttribute('aria-label', `${label}: not applicable here. ${reason}`);
    wrapper.setAttribute('title', reason);
    wrapper.append(
      element('span', {
        className: ['value__num', 'value__num--not-applicable'],
        text: 'n/a',
        attrs: { 'aria-hidden': 'true' },
      }),
      // THE ONLY STATE WHOSE EXPLANATION RENDERS ON SCREEN RATHER THAN BEHIND
      // A HOVER (demo-ux.md §17). Every other state can afford a tooltip; this
      // one cannot, for two reasons.
      //
      // First, a fact nobody hovers over is a fact nobody learns, and this is
      // the fact the demo most wants read: that continuous batching and the
      // paged KV cache are mutually exclusive execution paths. Behind a hover
      // it is invisible in a screenshot, on a projector, and on touch.
      //
      // Second, `n/a` alone is INDISTINGUISHABLE FROM A BROKEN PANEL. Under
      // two servers this state is the NORMAL case, not the exception, so a
      // visitor's first run would otherwise show a dashboard half-covered in
      // apparent breakage — rendering our single most interesting finding as
      // a bug.
      element('span', { className: 'value__na-caption', text: headlineSentence(reason) }),
    );
    return wrapper;
  }

  if (isPastStaleCeiling(field, ceilingMs, nowMs)) {
    // AC45(b). An unbounded age suffix is still a number on screen, and past
    // some point "last known" is indistinguishable from fiction. The number
    // goes away; the age stays, because WHY it went away is the useful part.
    const age = formatAge(ageMsOf(field, nowMs));
    const reason = `Last measured ${age}, which is past this panel's ${Math.round(
      ceilingMs / 1000,
    )}s limit for showing a number.`;
    wrapper.setAttribute('data-state', RENDER_STATES.UNAVAILABLE);
    wrapper.setAttribute('data-stale', 'expired');
    wrapper.setAttribute(ROVING_ITEM_ATTR, '');
    wrapper.setAttribute('tabindex', '-1');
    wrapper.setAttribute('role', 'note');
    wrapper.setAttribute('aria-label', `${label}: too old to show. ${reason}`);
    wrapper.setAttribute('title', reason);
    wrapper.append(
      element('span', {
        className: ['value__num', 'value__num--unavailable'],
        text: '—',
        attrs: { 'aria-hidden': 'true' },
      }),
      ageSeparator(),
      element('span', { className: 'value__stale', text: age }),
    );
    return wrapper;
  }

  if (state === RENDER_STATES.UNAVAILABLE) {
    const reason = field?.reason ?? 'This value is not measured by the server.';
    // Reachable and announced, but NOT its own tab stop: an em-dash with no
    // reachable reason is less honest than a zero because it withholds without
    // offering recourse, yet a panel full of them must still be one tab stop
    // with a roving cursor (AC29).
    wrapper.setAttribute(ROVING_ITEM_ATTR, '');
    wrapper.setAttribute('tabindex', '-1');
    wrapper.setAttribute('role', 'note');
    wrapper.setAttribute('aria-label', `${label}: not measurable yet. ${reason}`);
    wrapper.setAttribute('title', reason);
    wrapper.append(
      element('span', {
        className: ['value__num', 'value__num--unavailable'],
        text: '—',
        attrs: { 'aria-hidden': 'true' },
      }),
    );
    if (showUnit) {
      wrapper.append(element('span', { className: 'value__unit', text: ` ${unit}` }));
    }
    return wrapper;
  }

  if (state === RENDER_STATES.PENDING) {
    wrapper.setAttribute(ROVING_ITEM_ATTR, '');
    wrapper.setAttribute('tabindex', '-1');
    wrapper.setAttribute('role', 'note');
    wrapper.setAttribute('aria-label', `${label}: no samples yet. Run a scenario.`);
    wrapper.setAttribute('title', 'No samples yet in the 60 s window. Run a scenario.');
    wrapper.append(
      element('span', {
        className: ['value__num', 'value__num--pending'],
        text: '···',
        attrs: { 'aria-hidden': 'true' },
      }),
    );
    if (showUnit) {
      wrapper.append(element('span', { className: 'value__unit', text: ` ${unit}` }));
    }
    return wrapper;
  }

  // From here the state has been checked and a value provably exists.
  const format = options.format ?? ((value) => formatNumber(Number(value), unit));
  const rendered = typeof field.value === 'number' ? format(field.value) : String(field.value);
  const prefix = sourceClass === 'estimated' ? '~' : '';

  wrapper.append(element('span', { className: 'value__num', text: `${prefix}${rendered}` }));
  if (showUnitWithValue) {
    wrapper.append(element('span', { className: 'value__unit', text: ` ${unit}` }));
  }
  wrapper.append(sourceBadge(sourceClass, describeProvenance(field)));

  const staleAge = isStale(field) ? formatAge(ageMsOf(field, nowMs)) : null;

  if (staleAge !== null) {
    wrapper.setAttribute('data-stale', 'true');
    wrapper.append(
      ageSeparator(),
      element('span', {
        className: 'value__stale',
        text: staleAge,
        attrs: {
          title:
            field.reason ??
            'The latest poll did not refresh this value, so it is shown with its age rather than as current.',
        },
      }),
    );
  }

  // The age has to be in the ACCESSIBLE name, not only in the visual suffix.
  // Announcing "queue depth 41" while the screen says "41 · 12s old" hands the
  // number to a screen-reader user stripped of the one qualifier that makes it
  // honest — AC6 failing for exactly the visitors AC45 was written to protect.
  wrapper.setAttribute(
    'aria-label',
    // Same unit decision as the visual branch above. A screen reader announcing
    // "VRAM in use: 5.35 GiB bytes" is the same defect, heard instead of seen.
    `${label}: ${prefix}${rendered}${showUnitWithValue ? ` ${unit}` : ''}${staleAge ? `, stale, ${staleAge}` : ''}`,
  );
  return wrapper;
}

/**
 * A labelled metric row: label on the left, {@link renderField} on the right.
 *
 * @param {string} label
 * @param {any} field
 * @param {object} [options] Forwarded to {@link renderField}.
 * @returns {HTMLElement}
 */
export function metricRow(label, field, options = {}) {
  return element('div', {
    className: 'metric-row',
    children: [
      element('span', { className: 'metric-row__label', text: label }),
      renderField(field, { label, ...options }),
    ],
  });
}

/**
 * Bind a panel's stale ceiling (AC45(c)) to the two render helpers, so the
 * ceiling is declared once in `meta` rather than repeated at every call site
 * — where one omission silently reverts that metric to the global default.
 *
 * @param {{staleCeilingMs?: number|null}} panelMeta `null` opts out of the
 *   ceiling entirely; omitting the key inherits the global default. The two are
 *   NOT the same, and this used to collapse them.
 * @returns {{metricRow: typeof metricRow, renderField: typeof renderField}}
 */
export function bindPanel(panelMeta) {
  const staleCeilingMs = resolveStaleCeilingMs(panelMeta?.staleCeilingMs);
  return {
    metricRow: (label, field, options = {}) =>
      metricRow(label, field, { staleCeilingMs, ...options }),
    renderField: (field, options = {}) => renderField(field, { staleCeilingMs, ...options }),
  };
}

/**
 * A small-caps section divider inside a panel body.
 *
 * @param {string} text
 * @returns {HTMLElement}
 */
export function sectionLabel(text) {
  return element('h3', { className: 'panel-section', text });
}

/**
 * The §4.4 capability notice, used when an entire panel has no data.
 *
 * The closing "Everything else on this page still works." is not filler: it
 * converts "this dashboard is broken" into "this panel needs a flag", which is
 * the true state of the world (AC20).
 *
 * @param {object} spec
 * @param {string} spec.title
 * @param {string} spec.body
 * @param {string} [spec.command] A copy-pasteable fix, rendered with a copy button.
 * @returns {HTMLElement}
 */
export function capabilityNotice({ title, body, command }) {
  const notice = element('div', {
    className: 'capability-notice',
    attrs: { role: 'note' },
    children: [
      element('p', { className: 'capability-notice__title', text: title }),
      element('p', { className: 'capability-notice__body', text: body }),
    ],
  });

  if (command) {
    const code = element('code', { className: 'capability-notice__cmd', text: command });
    const copy = element('button', {
      className: 'capability-notice__copy',
      text: 'Copy',
      attrs: { type: 'button', 'aria-label': `Copy ${command} to the clipboard` },
    });
    copy.addEventListener('click', () => {
      navigator.clipboard?.writeText(command).then(
        () => {
          copy.textContent = 'Copied';
          setTimeout(() => {
            copy.textContent = 'Copy';
          }, 1200);
        },
        () => {
          // Clipboard access can be denied. Say so rather than silently doing
          // nothing — the command is still selectable by hand.
          copy.textContent = 'Select it';
        },
      );
    });
    notice.append(element('div', { className: 'capability-notice__cmdrow', children: [code, copy] }));
  }

  notice.append(
    element('p', {
      className: 'capability-notice__footer',
      text: 'Everything else on this page still works.',
    }),
  );
  return notice;
}

/**
 * Collect every field wrapper `renderField` produced inside a subtree.
 *
 * Walks `children` rather than calling `querySelectorAll`, because the test DOM
 * implements the tree but not the selector engine. A walk is the intersection
 * of the two, so this behaves identically under `node --test` and in a browser
 * — which is the whole point of having it in one place.
 *
 * @param {HTMLElement} rootElement
 * @returns {Array<HTMLElement>}
 */
function collectFieldWrappers(rootElement) {
  /** @type {Array<HTMLElement>} */
  const found = [];
  const visit = (node) => {
    if (!node || typeof node.getAttribute !== 'function') return;
    if (node.getAttribute('data-state') !== null && node.getAttribute('data-state') !== undefined) {
      found.push(node);
    }
    for (const child of node.children ?? []) visit(child);
  };
  visit(rootElement);
  return found;
}

/**
 * §31/D89 — the PANEL-level not-applicable treatment.
 *
 * When every field in a panel is `not-applicable`, the panel keeps its header
 * and frame and replaces its BODY with one explanation, instead of rendering a
 * column of `n/a` glyphs that each repeat the same sentence.
 *
 * WHY THIS IS A LAYOUT FIX AND NOT A DECORATION
 * `unavailable` is genuinely per-field: one stubbed metric beside working ones,
 * where an em-dash holds the alignment and admits a local gap. But
 * `not-applicable` is almost always per-PANEL, because its cause is structural
 * — a whole subsystem is bypassed, so every field goes not-applicable at once.
 * There is no mixed case to represent. Measured on this tree before the change,
 * a fully bypassed `throughput` rendered SEVENTEEN `n/a` glyphs carrying one
 * fact between them; `system` rendered fourteen and `scheduling` ten.
 *
 * Seventeen hole-glyphs do not read as seventeen honest admissions. They read
 * as a broken panel — and honesty that reads as malfunction has failed at its
 * only job. The collapse removes that failure rather than mitigating it: there
 * is no row of absences left to scan, so there is nothing to misread.
 *
 * WHY IT LIVES IN THE REPAINT PATH RATHER THAN IN EACH PANEL
 * Panels repaint on every store update, so anything applied once at mount is
 * erased by the next frame. Running it immediately after `paint()` means it
 * cannot fall out of sync, no panel author has to remember it, and a panel
 * added later inherits it without knowing it exists.
 *
 * WHAT IT DELIBERATELY DOES NOT DO
 * - With zero field wrappers it does nothing. A panel that binds no telemetry
 *   (prefix-cache) or that already renders its own capability notice
 *   (kv-memory) is not bypassed; it is speaking for itself, and overwriting
 *   that would delete the more specific message.
 * - With ANY field in another state it does nothing. A single structurally
 *   pinned field inside a live panel is the rare mixed case, and it stays an
 *   em-dash with a caption exactly as before.
 * - It never invents prose. Every sentence shown comes from a `reason` the
 *   server sent, and DISTINCT reasons are all preserved — collapsing three
 *   different explanations into one would trade a repetitive truth for a
 *   tidy falsehood.
 *
 * @param {HTMLElement} rootElement
 * @returns {boolean} True if the body was replaced.
 */
export function collapseNotApplicableBody(rootElement) {
  const wrappers = collectFieldWrappers(rootElement);
  if (wrappers.length === 0) return false;
  if (!wrappers.every((node) => node.getAttribute('data-state') === RENDER_STATES.NOT_APPLICABLE)) {
    return false;
  }

  // Distinct, order-preserving. `title` carries the untruncated reason;
  // the on-screen caption is only ever the first sentence.
  const reasons = [];
  for (const node of wrappers) {
    const reason = node.getAttribute('title');
    if (reason && !reasons.includes(reason)) reasons.push(reason);
  }
  if (reasons.length === 0) return false;

  const notice = element('div', {
    className: 'panel-bypass',
    attrs: { role: 'note', 'data-state': RENDER_STATES.NOT_APPLICABLE },
    children: [
      element('p', { className: 'panel-bypass__title', text: 'Not applicable here' }),
      ...reasons.map((reason) => element('p', { className: 'panel-bypass__reason', text: reason })),
    ],
  });

  replaceChildren(rootElement, [notice]);
  return true;
}

/**
 * Wrap first occurrences of known acronyms in `<abbr>` (AC30).
 *
 * @param {string} text
 * @param {Record<string, string>} acronyms
 * @returns {DocumentFragment}
 */
export function withAcronyms(text, acronyms) {
  const fragment = document.createDocumentFragment();
  const terms = Object.keys(acronyms).sort((a, b) => b.length - a.length);
  if (terms.length === 0) {
    fragment.append(document.createTextNode(text));
    return fragment;
  }
  const pattern = new RegExp(`\\b(${terms.map(escapeRegExp).join('|')})\\b`);
  let remaining = text;
  const seen = new Set();

  while (remaining.length > 0) {
    const match = pattern.exec(remaining);
    if (!match || seen.has(match[1])) {
      fragment.append(document.createTextNode(remaining));
      break;
    }
    seen.add(match[1]);
    fragment.append(document.createTextNode(remaining.slice(0, match.index)));
    fragment.append(
      element('abbr', {
        className: 'acronym',
        text: match[1],
        attrs: { title: acronyms[match[1]] },
      }),
    );
    remaining = remaining.slice(match.index + match[1].length);
  }
  return fragment;
}

/**
 * AC62. A debug-gated endpoint answers 404 when the server was launched without
 * `--enable-debug-endpoints`, and 404 is the cruellest status we could have been
 * given: 403 would say "you need permission" and diagnose itself, but 404 says
 * "this endpoint does not exist". A visitor reads that as a wrong URL, a stale
 * build, or a broken demo -- anything except "add a flag" -- and they will go
 * re-read the README, which will look correct, because it is.
 *
 * Without this, the panel renders em-dashes carrying the FIELD's reason, which
 * explains that the value is not exposed to the HTTP layer. That sentence is
 * true of the default build and it is the WRONG EXPLANATION here: it describes
 * an unfixable limitation when the actual fix is one flag. An honest absence
 * that misattributes its own cause is worse than a blank, because it forecloses
 * the fix.
 *
 * Shown only on panels that are actually degraded -- a panel with nothing
 * missing has nothing to explain, and repeating the notice on all five would
 * train the eye to skip it.
 *
 * @param {HTMLElement} rootElement
 * @param {Readonly<Record<string, string|null>>} [endpointErrors]
 * @returns {boolean} True if a notice was rendered.
 */
export function renderGatedEndpointNotice(rootElement, endpointErrors) {
  // Idempotent: the repaint scheduler calls this on every frame, so without
  // this the page would accumulate one notice per frame — four a second.
  // Membership is tested through classList, which both the browser and the
  // test DOM implement; `className` is a string in one and absent in the other,
  // and reading it is how this silently matched nothing the first time.
  for (const child of [...(rootElement.children ?? [])]) {
    if (child.classList?.contains('panel-gated-notice')) child.remove();
  }
  if (!endpointErrors) return false;

  const gated = DEBUG_GATED_ENDPOINTS.filter((path) => Boolean(endpointErrors[path]));
  if (gated.length === 0) return false;

  // Only speak up if this panel actually lost something. A panel with nothing
  // missing has nothing to explain, and repeating the notice on all five would
  // train the eye to skip it.
  const degraded = collectFieldWrappers(rootElement).some((node) =>
    [RENDER_STATES.UNAVAILABLE, RENDER_STATES.PENDING].includes(node.getAttribute('data-state')),
  );
  if (!degraded) return false;

  rootElement.append(
    element('div', {
      className: 'panel-gated-notice',
      attrs: { role: 'note' },
      children: [
        element('p', {
          className: 'panel-gated-notice__lead',
          text:
            `${gated.join(' and ')} returned 404 because this server was started without ` +
            `${DEBUG_ENDPOINTS_FLAG}. The endpoint is missing, not the data.`,
        }),
        element('p', {
          className: 'panel-gated-notice__command',
          text: gatedRestartCommand(),
        }),
      ],
    }),
  );
  return true;
}

/**
 * The full command, not just the flag name. A visitor who has to work out where
 * the flag goes has been handed a puzzle instead of a fix, and `--addr` is
 * already the flag people get wrong (`--port` is rejected outright).
 *
 * The address is read from the page's own origin rather than hardcoded: this
 * page is served BY the server it is describing, so that is the one part of the
 * command we can state as fact instead of guessing.
 *
 * @returns {string}
 */
export function gatedRestartCommand() {
  const host =
    typeof location === 'object' && location?.host ? location.host : '127.0.0.1:8123';
  return `onnx-genai-server --model <MODEL_DIR> --addr ${host} ${DEBUG_ENDPOINTS_FLAG}`;
}

/**
 * Coalesce repaint requests into a single `requestAnimationFrame`.
 *
 * demo-ux.md §3.3 rule 3: panels never set an interval and never paint per
 * event. At 32 concurrent streams, per-token DOM writes are thousands of layout
 * invalidations a second and AC23's 30 fps floor dies instantly.
 *
 * The returned scheduler also skips painting while the panel is hidden or
 * off-screen, which is most of AC23 on its own.
 *
 * @param {HTMLElement} rootElement
 * @param {() => void} paint
 * @returns {{request(): void, cancel(): void, setVisible(visible: boolean): void}}
 */
export function createRepaintScheduler(rootElement, paint, options = {}) {
  // A panel that forgets the store still paints perfectly and silently loses
  // its AC62 notice -- the failure would be invisible in exactly the situation
  // the notice exists for. Refusing here makes the omission impossible to ship
  // rather than merely discouraged, the same rule createRovingGroup applies to
  // an unnamed group.
  if (!options.telemetryStore && IS_DEVELOPMENT) {
    throw new Error(
      'createRepaintScheduler requires { telemetryStore } so the panel can render the ' +
        'AC62 gated-endpoint notice. Pass the same store the panel reads fields from.',
    );
  }
  let frameHandle = 0;
  let visible = true;
  let dirtyWhileHidden = false;
  // `null` means "never painted", NOT "painted at time zero". performance.now()
  // is measured from page load, so a zero sentinel makes the very first request
  // look like it arrived milliseconds after a previous paint — and the first
  // render of every panel would be deferred by up to a second at exactly the
  // moment the dashboard mounts.
  let lastPaintAtMs = null;
  let trailingHandle = 0;

  // AC30. The dashboard has no tweened animation, so the motion a visitor
  // actually perceives is the repaint cadence itself: numbers and sparklines
  // twitching four times a second. Under a reduced-motion preference that drops
  // to roughly 1 Hz.
  //
  // The DATA is never reduced and updates are never dropped — a trailing
  // repaint always lands, so the panel still ends up showing the newest value.
  // Withholding measurements from someone with a vestibular disorder would be a
  // strange reading of an accessibility preference; the fix is to move less
  // often, not to know less.
  const minIntervalMs = options.minIntervalMs ?? (prefersReducedMotion() ? 1000 : 0);
  const now = () => (typeof performance === 'object' ? performance.now() : Date.now());

  const run = () => {
    frameHandle = 0;
    if (!visible || rootElement.hidden) {
      dirtyWhileHidden = true;
      return;
    }
    lastPaintAtMs = now();
    paint();
    // §31/D89. After the paint, never instead of it: the panel builds its
    // fields exactly as it always has, and the collapse is a pure function of
    // what those fields turned out to be. That ordering is what lets a panel
    // become bypassed (or stop being bypassed) at runtime — when the visitor
    // switches servers — with no panel-specific code on either transition.
    collapseNotApplicableBody(rootElement);
    // AC62, after the collapse: if a panel is wholly not-applicable it has a
    // better explanation than a missing flag, and two notices would compete.
    renderGatedEndpointNotice(rootElement, options.telemetryStore?.getSnapshot?.()?.endpointErrors);
  };

  return {
    request() {
      if (frameHandle !== 0) {
        return;
      }
      const sinceLastMs = lastPaintAtMs === null ? Infinity : now() - lastPaintAtMs;
      if (minIntervalMs > 0 && sinceLastMs < minIntervalMs) {
        // Too soon. Schedule one trailing repaint for when the interval is up
        // rather than dropping this update on the floor.
        if (trailingHandle === 0) {
          trailingHandle = setTimeout(() => {
            trailingHandle = 0;
            this.request();
          }, minIntervalMs - sinceLastMs);
        }
        return;
      }
      frameHandle = requestAnimationFrame(run);
    },
    cancel() {
      if (frameHandle !== 0) {
        cancelAnimationFrame(frameHandle);
        frameHandle = 0;
      }
      if (trailingHandle !== 0) {
        clearTimeout(trailingHandle);
        trailingHandle = 0;
      }
    },
    setVisible(next) {
      visible = next;
      if (next && dirtyWhileHidden) {
        dirtyWhileHidden = false;
        this.request();
      }
    },
  };
}

/**
 * Observe whether the panel is on screen, so hidden panels stop painting.
 *
 * @param {HTMLElement} rootElement
 * @param {(visible: boolean) => void} onChange
 * @returns {() => void} Disconnect.
 */
export function observeVisibility(rootElement, onChange) {
  if (typeof IntersectionObserver !== 'function') {
    onChange(true);
    return () => {};
  }
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        onChange(entry.isIntersecting);
      }
    },
    { threshold: 0 },
  );
  observer.observe(rootElement);
  return () => observer.disconnect();
}

/**
 * Create a sparkline canvas plus the accessible description the chart needs.
 *
 * @param {object} spec
 * @param {string} spec.label
 * @param {number} [spec.width]
 * @param {number} [spec.height]
 * @returns {{root: HTMLElement, canvas: HTMLCanvasElement, setDescription(text: string): void}}
 */
export function createSparklineSlot({ label, width = 180, height = 28 }) {
  const canvas = element('canvas', {
    className: 'spark__canvas',
    attrs: { width, height, role: 'img', 'aria-label': `${label}: loading` },
  });

  // AC28: every canvas needs a view-as-table alternative. It is built here
  // rather than per panel so it cannot be forgotten by one of them, and it
  // lives in the DOM from the start rather than being constructed on toggle —
  // a table that only exists after a click is a table that only some visitors
  // can reach, and the ones who need it most are the least likely to click.
  const tableBody = element('tbody', { className: 'spark-table__body' });
  const table = element('table', {
    className: 'spark-table',
    attrs: { hidden: '' },
    children: [
      element('caption', { className: 'spark-table__caption', text: label }),
      element('thead', {
        children: [
          element('tr', {
            children: [
              element('th', { text: 'time', attrs: { scope: 'col' } }),
              element('th', { text: 'value', attrs: { scope: 'col' } }),
            ],
          }),
        ],
      }),
      tableBody,
    ],
  });

  const root = element('figure', {
    className: 'spark',
    attrs: { tabindex: '0', 'data-view': 'chart' },
    children: [canvas, table],
  });

  return {
    root,
    canvas: /** @type {HTMLCanvasElement} */ (canvas),
    setDescription(text) {
      canvas.setAttribute('aria-label', text);
      root.setAttribute('title', text);
      table.querySelector?.('caption')?.setAttribute('title', text);
    },

    /**
     * Populate the table alternative from the same plan the canvas paints, so
     * the two can never disagree about what was measured.
     *
     * @param {Array<{label: string, value: string}>} rows
     */
    setTableRows(rows) {
      replaceChildren(
        tableBody,
        rows.map((row) =>
          element('tr', {
            className: 'spark-table__row',
            children: [
              element('th', { text: row.label, attrs: { scope: 'row' } }),
              element('td', { text: row.value }),
            ],
          }),
        ),
      );
    },

    /**
     * @param {'chart'|'table'} view
     */
    setView(view) {
      const asTable = view === 'table';
      root.setAttribute('data-view', asTable ? 'table' : 'chart');
      if (asTable) {
        table.removeAttribute('hidden');
        canvas.setAttribute('hidden', '');
      } else {
        canvas.removeAttribute('hidden');
        table.setAttribute('hidden', '');
      }
    },
  };
}

/**
 * Whether the visitor has asked for reduced motion (AC30).
 *
 * Defaults to `false` when `matchMedia` is unavailable, which is the honest
 * default: we cannot claim to know a preference that was never expressed, and
 * assuming reduced motion would quietly degrade the charts for everyone in a
 * test environment.
 *
 * @returns {boolean}
 */
export function prefersReducedMotion() {
  if (typeof globalThis.matchMedia !== 'function') return false;
  try {
    return globalThis.matchMedia('(prefers-reduced-motion: reduce)').matches === true;
  } catch {
    return false;
  }
}

/**
 * Replace a container's contents in one operation.
 *
 * @param {HTMLElement} container
 * @param {Array<Node|null|undefined>} children
 * @returns {void}
 */
export function replaceChildren(container, children) {
  // Panels re-render on every poll — two to four times a second. A keyboard
  // user who has moved the roving cursor onto an unavailable value to read WHY
  // it is unavailable would otherwise have the focused element destroyed under
  // them within 250ms, dropping focus to <body>. The explanation would be
  // literally unreachable by keyboard: you could never finish reading it.
  //
  // Restoring by POSITION rather than identity is deliberate. The new element
  // at the same index is the same metric; the old node is gone no matter what
  // we do.
  const doc = container.ownerDocument ?? globalThis.document;
  const active = doc?.activeElement ?? null;
  const restoreIndex =
    active && active !== container && container.contains?.(active)
      ? rovingItems(container).indexOf(active)
      : -1;

  container.replaceChildren(...children.filter(Boolean));

  if (restoreIndex >= 0) {
    const items = rovingItems(container);
    const target = items[Math.min(restoreIndex, items.length - 1)];
    if (target) {
      setRovingCursor(items, target);
      target.focus?.();
    } else if (typeof container.focus === 'function') {
      // The metric it was on has gone away entirely. The enclosing group is
      // the nearest honest place to land — better than silently falling to
      // <body> with no announcement of what happened.
      container.focus();
    }
  }
}

/**
 * Switch every chart in a panel between the canvas and its table alternative.
 *
 * demo-ux.md §3 gives the shell a uniform "view as table" toggle, but the panel
 * contract only hands the shell `describe()` — a sentence. A sentence is not
 * tabular data, so the shell cannot build the real <table> with scoped headers
 * that §10 requires. The data lives here, so the toggle has to be driven from
 * here; this is the hook the shell calls.
 *
 * Walking the DOM rather than tracking slot handles means it works for any
 * panel, including ones written later that never heard of this function.
 *
 * @param {HTMLElement} rootElement The panel body.
 * @param {'chart'|'table'} view
 * @returns {number} How many charts were switched.
 */
export function setPanelView(rootElement, view) {
  const figures = collectSparkFigures(rootElement);
  for (const figure of figures) {
    const asTable = view === 'table';
    figure.setAttribute('data-view', asTable ? 'table' : 'chart');
    for (const child of figure.children ?? []) {
      const isTable = child.tagName === 'TABLE';
      const isCanvas = child.tagName === 'CANVAS';
      if (!isTable && !isCanvas) continue;
      if (isTable === asTable) {
        child.removeAttribute('hidden');
      } else {
        child.setAttribute('hidden', '');
      }
    }
  }
  return figures.length;
}

/**
 * @param {HTMLElement} container
 * @param {HTMLElement[]} [into]
 * @returns {HTMLElement[]}
 */
function collectSparkFigures(container, into = []) {
  for (const child of container.children ?? []) {
    if (child.tagName === 'FIGURE' && child.hasAttribute?.('data-view')) {
      into.push(child);
    }
    collectSparkFigures(child, into);
  }
  return into;
}

/** Attribute marking an element as a stop on a roving cursor (AC29). */
export const ROVING_ITEM_ATTR = 'data-roving-item';

/**
 * Collect roving stops in document order, without needing a selector engine.
 *
 * @param {HTMLElement} container
 * @param {HTMLElement[]} [into]
 * @returns {HTMLElement[]}
 */
export function rovingItems(container, into = []) {
  for (const child of container.children ?? []) {
    if (child.hasAttribute?.(ROVING_ITEM_ATTR)) {
      into.push(child);
    }
    rovingItems(child, into);
  }
  return into;
}

/**
 * @param {HTMLElement[]} items
 * @param {HTMLElement|null} current
 */
function setRovingCursor(items, current) {
  for (const item of items) {
    item.setAttribute('tabindex', item === current ? '0' : '-1');
  }
}

/**
 * Make a container of many read-only annotations into ONE tab stop with a
 * roving cursor (AC29).
 *
 * Without this, every em-dash carrying a "why is this not measured" note is its
 * own tab stop. On the continuous-batching server, where the whole KV group is
 * structurally unavailable, that is dozens of stops between a keyboard user and
 * the first thing they can actually operate — the spec calls this out as the
 * worst accessibility failure available to us.
 *
 * The listener lives on the container, which survives re-render, so the
 * behaviour does not need re-attaching on every poll.
 *
 * @param {HTMLElement} container
 * @param {{label?: string}} [options]
 * @returns {{refresh: () => void, destroy: () => void}}
 */
export function createRovingGroup(container, options = {}) {
  container.setAttribute('role', 'group');

  // A `role="group"` with no accessible name is worse than no group at all: a
  // screen reader announces "group" and the listener learns nothing, while the
  // page looks perfect to everyone else. This shipped for a while because the
  // shell passed `panel.title` and the registry had no `title` -- so `label`
  // was `undefined`, the `if` quietly skipped, and every panel on the dashboard
  // was an anonymous group.
  //
  // Refusing to build one is the only version of this that cannot recur. In a
  // browser we still set the group up and complain loudly, because dropping a
  // panel in front of a room is worse than an unnamed one.
  const label = typeof options.label === 'string' ? options.label.trim() : '';
  if (!label) {
    const message =
      'createRovingGroup requires a non-empty `label`: a role="group" with no ' +
      'accessible name announces "group" and nothing else. Pass the panel title.';
    if (IS_DEVELOPMENT) throw new Error(message);
    console.error(`[dashboard] ${message}`);
  } else {
    container.setAttribute('aria-label', label);
  }

  const move = (delta, absolute) => {
    const items = rovingItems(container);
    if (items.length === 0) return;
    const doc = container.ownerDocument ?? globalThis.document;
    const active = doc?.activeElement ?? null;
    const from = items.indexOf(active);
    let next;
    if (absolute === 'first') next = 0;
    else if (absolute === 'last') next = items.length - 1;
    else if (from === -1) next = 0;
    else next = Math.min(items.length - 1, Math.max(0, from + delta));
    setRovingCursor(items, items[next]);
    items[next].focus?.();
  };

  /** @param {KeyboardEvent} event */
  const onKeyDown = (event) => {
    switch (event.key) {
      case 'ArrowDown':
      case 'ArrowRight':
        move(1);
        break;
      case 'ArrowUp':
      case 'ArrowLeft':
        move(-1);
        break;
      case 'Home':
        move(0, 'first');
        break;
      case 'End':
        move(0, 'last');
        break;
      default:
        return;
    }
    event.preventDefault?.();
  };

  const refresh = () => {
    const items = rovingItems(container);
    if (items.length === 0) {
      // Nothing to rove over: do not leave an empty group as a tab stop that
      // announces itself and then offers nowhere to go.
      container.removeAttribute('tabindex');
      return;
    }
    container.setAttribute('tabindex', '0');
    const doc = container.ownerDocument ?? globalThis.document;
    const active = doc?.activeElement ?? null;
    setRovingCursor(items, items.includes(active) ? active : items[0]);
  };

  container.addEventListener('keydown', onKeyDown);
  refresh();

  return {
    refresh,
    destroy() {
      container.removeEventListener('keydown', onKeyDown);
      container.removeAttribute('tabindex');
      container.removeAttribute('role');
    },
  };
}

/**
 * Summarise a field for a `describe()` sentence (demo-ux.md §3.3 rule 6).
 *
 * @param {string} label
 * @param {any} field
 * @param {(value: number) => string} [format]
 * @returns {string}
 */
export function describeFieldText(label, field, format) {
  field = displaySafeField(field);
  if (!isRenderable(field)) {
    const state = renderStateOf(field);
    if (state === RENDER_STATES.PENDING) return `${label} has no samples yet`;
    // Not-applicable gets its own sentence, and it is not an apology. Saying
    // "not measurable yet" here would promise a value that is never coming and
    // would describe a correctly empty panel as a deficiency — which is the
    // whole reason the state was added. The reason is appended because for this
    // state the explanation IS the information, and a screen-reader user who
    // cannot hover has no other route to it.
    if (state === RENDER_STATES.NOT_APPLICABLE) {
      const because = field?.reason ? ` — ${field.reason}` : '';
      return `${label} is not applicable on this engine${because}`;
    }
    return `${label} is not measurable yet`;
  }
  // Same rule as `renderField`: a custom formatter owns its unit. This sentence
  // is the screen-reader and table-view route to the same number, so letting it
  // drift would fix the defect only for people who can see it.
  const unit = field.unit && !format ? ` ${field.unit}` : '';
  const value =
    typeof field.value === 'number'
      ? (format ?? ((raw) => formatNumber(raw, field.unit)))(field.value)
      : String(field.value);
  return `${label} ${value}${unit}`;
}

// ── internals ────────────────────────────────────────────────────────────────

/**
 * Resolve a field's provenance CLASS.
 *
 * `field.sourceClass` is authoritative and is asked first. Inference from
 * `source` remains only as a fallback for callers that predate the class key.
 *
 * This used to infer the class from the `source` string ALONE, treating
 * anything starting with `/` as a server value and everything else as derived.
 * That worked only because `source` carried category sentinels ('derived',
 * 'client', 'unavailable'). Those are gone (D161) — `source` is now an endpoint
 * or null — so inference alone would badge every stubbed server field as
 * DERIVED, claiming we computed a number the server never gave us.
 *
 * @param {{sourceClass?: string, source?: string}|string|undefined} field
 * @returns {SourceClass}
 */
function normaliseSourceClass(field) {
  const candidate = typeof field === 'string' ? undefined : field?.sourceClass;
  if (candidate && candidate in SOURCE_BADGES) {
    return /** @type {SourceClass} */ (candidate);
  }
  const source = typeof field === 'string' ? field : field?.source;
  if (source && source in SOURCE_BADGES) {
    return /** @type {SourceClass} */ (source);
  }
  if (source && source.startsWith('/')) {
    return 'server';
  }
  // Nothing above identified a provenance. `derived` is a CLAIM — arithmetic on
  // measured inputs — and it is only honest when there is evidence of the
  // arithmetic, which is what `derivedFrom` records. Awarding it to every
  // unidentified field, as this function used to, upgrades "we do not know"
  // into "we computed it from measurements" for free. Everything left is
  // genuinely unknown and must say so.
  const derivedFrom = typeof field === 'string' ? undefined : field?.derivedFrom;
  if (Array.isArray(derivedFrom) && derivedFrom.length > 0) {
    return 'derived';
  }
  return UNKNOWN_SOURCE_CLASS;
}

/** @param {{source?: string, at?: number, derivedFrom?: string[]}} field */
function describeProvenance(field) {
  const parts = [];
  if (field.source && field.source.startsWith('/')) {
    parts.push(field.source);
  }
  if (Array.isArray(field.derivedFrom) && field.derivedFrom.length > 0) {
    parts.push(`from ${field.derivedFrom.join(', ')}`);
  }
  return parts.join(' · ');
}

/** @param {string} text */
function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Read the client-observed request table, distinguishing "nothing has been sent
 * yet" from "nothing is feeding this panel".
 *
 * `null` and `[]` look identical if you coalesce them, and the resulting empty
 * table reads as "no traffic" — a claim about the server — when the truth is
 * "no scenario runner is wired up", a fact about the page. Same visual, opposite
 * meanings, and the wrong one blames the wrong component.
 *
 * @param {object} telemetryStore
 * @returns {{wired: boolean, rows: Array<object>}}
 */
export function readRequests(telemetryStore) {
  const rows = typeof telemetryStore.requests === 'function' ? telemetryStore.requests() : null;
  if (rows === null || rows === undefined) {
    return { wired: false, rows: [] };
  }
  return { wired: true, rows };
}

/** Copy for the two empty states, kept together so they cannot drift apart. */
export const REQUEST_TABLE_EMPTY = Object.freeze({
  unwired:
    'Per-request timing comes from the scenario runner. No scenario is connected to this page, ' +
    'so this table is empty for a reason that has nothing to do with the server.',
  idle: 'No requests in this scenario yet.',
});

/**
 * Paint a sparkline, describe it, and build its table alternative in one call.
 *
 * The three must happen together or AC28 fails silently: a canvas whose table
 * was not refreshed shows stale numbers to exactly the visitors who cannot see
 * that the chart beside it has moved on. Making it one function means a call
 * site cannot do two of the three.
 *
 * @param {object} slot A slot from {@link createSparklineSlot}.
 * @param {object} series
 * @param {object} options
 * @param {number} options.width
 * @param {number} options.height
 * @param {number} [options.windowMs]
 * @param {boolean} [options.zeroBaseline]
 * @param {string} options.label
 * @param {string} [options.unit]
 * @param {(value: number) => string} [options.format]
 * @returns {object} The plan, for callers that need it.
 */
export function renderSparkline(slot, series, options) {
  const plan = planSparkline(series, {
    width: options.width,
    height: options.height,
    windowMs: options.windowMs,
    zeroBaseline: options.zeroBaseline,
    nowMs: options.nowMs ?? Date.now(),
  });

  // The painter does a straight redraw with no tweening, so there is no
  // animation here to suppress for AC30. The motion a visitor actually
  // perceives is the REPAINT CADENCE — a chart twitching four times a second —
  // and that is throttled in createRepaintScheduler instead.
  paintSparkline(slot.canvas, plan);

  slot.setDescription(
    describeSparkline(plan, {
      label: options.label,
      unit: options.unit,
      windowSeconds: options.windowMs ? options.windowMs / 1000 : undefined,
      reason: series?.reason,
    }),
  );
  slot.setTableRows(tabulateSparkline(plan, { format: options.format }));
  return plan;
}
