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

import { RENDER_STATES, isRenderable, isStale, renderStateOf } from './field-state.js';

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

/** Reason copy reused across panels, taken verbatim from demo-ux.md §4.2. */
export const REASONS = Object.freeze({
  KV_NOT_EXPOSED:
    'KV page statistics are computed by the engine but not yet exposed over HTTP. ' +
    '`Engine::page_usage()` exists; the server does not call it.',
  DEBUG_GATED: 'Requires debug endpoints. Restart the server with `--enable-debug-endpoints`.',
  NO_PREEMPTION_COUNTER:
    'The scheduler performs preemption but keeps no counter for it. Not measurable today.',
  NO_MAX_BATCH:
    "Occupancy needs the server's max batch size, which isn't surfaced. " +
    "The current batch size is real; the denominator isn't.",
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
  const badge = SOURCE_BADGES[sourceClass] ?? SOURCE_BADGES.derived;
  const title = detail ? `${badge.title} · ${detail}` : badge.title;
  return element('abbr', {
    className: ['value__src', `value__src--${sourceClass}`],
    text: badge.glyph,
    attrs: { title },
  });
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
  const state = renderStateOf(field);
  const label = options.label ?? field?.label ?? 'value';
  const unit = field?.unit ?? '';
  const sourceClass = normaliseSourceClass(options.sourceClass ?? field?.source);
  const showUnit = options.showUnit !== false && Boolean(unit) && unit !== '%';

  const wrapper = element('span', {
    className: 'value',
    attrs: { 'data-state': state, 'data-source': sourceClass },
  });

  if (state === RENDER_STATES.UNAVAILABLE) {
    const reason = field?.reason ?? 'This value is not measured by the server.';
    // tabindex + aria-label, not just a title: these are the values whose
    // explanation matters most, and an em-dash with no reachable reason is less
    // honest than a zero because it withholds without offering recourse.
    wrapper.setAttribute('tabindex', '0');
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
      wrapper.append(element('span', { className: 'value__unit', text: unit }));
    }
    return wrapper;
  }

  if (state === RENDER_STATES.PENDING) {
    wrapper.setAttribute('tabindex', '0');
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
      wrapper.append(element('span', { className: 'value__unit', text: unit }));
    }
    return wrapper;
  }

  // From here the state has been checked and a value provably exists.
  const format = options.format ?? ((value) => formatNumber(Number(value), unit));
  const rendered = typeof field.value === 'number' ? format(field.value) : String(field.value);
  const prefix = sourceClass === 'estimated' ? '~' : '';

  wrapper.append(element('span', { className: 'value__num', text: `${prefix}${rendered}` }));
  if (showUnit) {
    wrapper.append(element('span', { className: 'value__unit', text: unit }));
  }
  wrapper.append(sourceBadge(sourceClass, describeProvenance(field)));

  if (isStale(field)) {
    const ageSeconds = Math.max(0, Math.round((Date.now() - (field.at ?? Date.now())) / 1000));
    wrapper.setAttribute('data-stale', 'true');
    wrapper.append(
      element('span', {
        className: 'value__stale',
        text: `${ageSeconds}s ago`,
        attrs: {
          title:
            field.reason ??
            'The latest poll did not refresh this value, so it is shown with its age rather than as current.',
        },
      }),
    );
  }

  wrapper.setAttribute('aria-label', `${label}: ${prefix}${rendered}${unit ? ` ${unit}` : ''}`);
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
export function createRepaintScheduler(rootElement, paint) {
  let frameHandle = 0;
  let visible = true;
  let dirtyWhileHidden = false;

  const run = () => {
    frameHandle = 0;
    if (!visible || rootElement.hidden) {
      dirtyWhileHidden = true;
      return;
    }
    paint();
  };

  return {
    request() {
      if (frameHandle !== 0) {
        return;
      }
      frameHandle = requestAnimationFrame(run);
    },
    cancel() {
      if (frameHandle !== 0) {
        cancelAnimationFrame(frameHandle);
        frameHandle = 0;
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
  const root = element('figure', {
    className: 'spark',
    attrs: { tabindex: '0' },
    children: [canvas],
  });
  return {
    root,
    canvas: /** @type {HTMLCanvasElement} */ (canvas),
    setDescription(text) {
      canvas.setAttribute('aria-label', text);
      root.setAttribute('title', text);
    },
  };
}

/**
 * Replace a container's contents in one operation.
 *
 * @param {HTMLElement} container
 * @param {Array<Node|null|undefined>} children
 * @returns {void}
 */
export function replaceChildren(container, children) {
  container.replaceChildren(...children.filter(Boolean));
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
  if (!isRenderable(field)) {
    return renderStateOf(field) === RENDER_STATES.PENDING
      ? `${label} has no samples yet`
      : `${label} is not measurable yet`;
  }
  const unit = field.unit ? ` ${field.unit}` : '';
  const value =
    typeof field.value === 'number'
      ? (format ?? ((raw) => formatNumber(raw, field.unit)))(field.value)
      : String(field.value);
  return `${label} ${value}${unit}`;
}

// ── internals ────────────────────────────────────────────────────────────────

/**
 * Map whatever the store put in `source` onto a badge class.
 *
 * The store may carry a source CLASS (`'server'`) or an endpoint PATH
 * (`'/v1/status'`) depending on which Field vocabulary wins in contract-team.
 * A path means the server produced it, so it maps to `server`.
 *
 * @param {string|undefined} source
 * @returns {SourceClass}
 */
function normaliseSourceClass(source) {
  if (!source) {
    return 'derived';
  }
  if (source in SOURCE_BADGES) {
    return /** @type {SourceClass} */ (source);
  }
  if (source.startsWith('/')) {
    return 'server';
  }
  return 'derived';
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
