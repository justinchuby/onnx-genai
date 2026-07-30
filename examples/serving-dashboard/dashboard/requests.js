// Copyright (c) Microsoft Corporation.
//
// Panel 5 — Requests. demo-ux.md §5.6.
//
// This is the audit surface. An engineer who does not believe a number on this
// page should be one click from the raw evidence, and this table is that click.
//
// THE STATE VOCABULARY IS DELIBERATELY NARROWER THAN THE SERVER'S.
// The server-side lifecycle a scheduler actually has — queued, admitted,
// prefilling, decoding, preempted, finishing, done — is not observable from a
// browser, and `recon-map.md` §7.5 confirms no such state enum exists in the
// server today anyway. A browser observes exactly this:
//
//     sent → streaming → done | error | cancelled
//
// Rendering the richer vocabulary by inferring it from client timing would be
// fabrication in the most invisible possible place: every state would be
// plausible, none would be measured, and no reviewer could tell by looking. If
// the server later reports a real state, this column gains a second badge and
// the richer states light up additively — never by inference.

import { isRenderable, numericValueOf } from './field-state.js';
import {
  createRepaintScheduler,
  bindPanel,
  element,
  formatDuration,
  formatNumber,
  observeVisibility,
  replaceChildren,
  readRequests,
  REQUEST_TABLE_EMPTY,
} from './panel-kit.js';

/** Above this many rows the table windows itself rather than laying out 32+ rows. */
const VIRTUALIZE_THRESHOLD = 32;

/**
 * The only request states a browser can actually observe.
 *
 * Each carries a glyph as well as a colour, so state is never conveyed by
 * colour alone (AC25) and a grayscale screenshot still parses.
 */
const CLIENT_STATES = Object.freeze({
  sent: { glyph: '◷', label: 'sent', hint: 'Request sent; no token has arrived yet' },
  streaming: { glyph: '▶', label: 'streaming', hint: 'Tokens are arriving' },
  done: { glyph: '✔', label: 'done', hint: 'The stream completed normally' },
  error: { glyph: '✕', label: 'error', hint: 'The stream ended with an error' },
  cancelled: { glyph: '⊘', label: 'cancelled', hint: 'The request was cancelled by the page' },
});

export const meta = Object.freeze({
  // Client-observed; every profile can produce it.
  requires: null,
  id: 'requests',
  title: 'Requests',
  group: 'scheduling',
  span: 2,
  cadence: 0,
  // The request table is client-observed and refreshes with the scenario runner.
  staleCeilingMs: 10000,
  defaultOpen: true,
  acronyms: {
    TTFT: 'Time to first token — how long until the first token of a response arrives',
    KV: 'Key/Value attention cache — the per-token state attention reads on every step',
  },
});

/** Panel-scoped renderers carrying this panel's stale ceiling (AC45(c)). */
const { renderField } = bindPanel(meta);

/**
 * @param {HTMLElement} rootElement
 * @param {any} telemetryStore
 * @returns {{destroy(): void, describe(): string}}
 */
export default function mount(rootElement, telemetryStore) {
  const table = element('table', {
    className: 'requests-table',
    attrs: { 'aria-label': 'Every request in the current scenario, as observed by this browser' },
  });
  const caption = element('p', { className: 'requests-table__caption' });
  rootElement.append(caption, table);

  let sortKey = 'sent';
  let sortAscending = true;
  let description = 'Requests: none in this scenario yet.';

  const paint = () => {
    const { wired, rows: requests } = readRequests(telemetryStore);

    if (requests.length === 0) {
      caption.textContent = wired
        ? REQUEST_TABLE_EMPTY.idle
        : REQUEST_TABLE_EMPTY.unwired;
      replaceChildren(table, []);
      description = wired
        ? 'Requests: none in this scenario yet.'
        : 'Requests: no scenario runner is connected to this page, so no request timing is available.';
      return;
    }

    const sorted = sortRequests(requests, sortKey, sortAscending);
    const visible = sorted.slice(0, VIRTUALIZE_THRESHOLD);

    caption.textContent =
      visible.length < sorted.length
        ? `Showing ${visible.length} of ${sorted.length} requests, newest first.`
        : `${sorted.length} request${sorted.length === 1 ? '' : 's'} in this scenario.`;

    replaceChildren(table, [
      buildHeader(sortKey, sortAscending, (key) => {
        if (key === sortKey) {
          sortAscending = !sortAscending;
        } else {
          sortKey = key;
          sortAscending = true;
        }
        scheduler.request();
      }),
      element('tbody', { children: visible.map((request) => buildRow(request)) }),
    ]);

    description = buildDescription(sorted);
  };

  const scheduler = createRepaintScheduler(rootElement, paint, { telemetryStore });
  const stopObserving = observeVisibility(rootElement, (visible) => scheduler.setVisible(visible));
  const unsubscribe = telemetryStore.subscribe(() => scheduler.request());
  const unsubscribeRequests = telemetryStore.subscribeRequests?.(() => scheduler.request()) ?? (() => {});
  scheduler.request();

  return {
    unmount() {
      unsubscribe();
      unsubscribeRequests();
      stopObserving();
      scheduler.cancel();
      rootElement.replaceChildren();
    },
    describe() {
      return description;
    },
  };
}

// ── rendering ────────────────────────────────────────────────────────────────

/** Column definitions, in display order. */
const COLUMNS = Object.freeze([
  { key: 'id', label: 'id', sortable: true },
  { key: 'state', label: 'state', sortable: true },
  { key: 'sent', label: 'sent', sortable: true },
  { key: 'ttft', label: 'TTFT', sortable: true },
  { key: 'tokens', label: 'in / out', sortable: false },
  { key: 'rate', label: 'tok/s', sortable: true },
  { key: 'kvBlocks', label: 'KV blocks', sortable: true },
  { key: 'reused', label: 'reused', sortable: true },
  { key: 'finish', label: 'finish', sortable: false },
]);

/**
 * @param {string} sortKey
 * @param {boolean} ascending
 * @param {(key: string) => void} onSort
 * @returns {HTMLElement}
 */
function buildHeader(sortKey, ascending, onSort) {
  const row = element('tr');
  for (const column of COLUMNS) {
    const cell = element('th', {
      attrs: {
        scope: 'col',
        'aria-sort': column.key === sortKey ? (ascending ? 'ascending' : 'descending') : 'none',
      },
    });
    if (column.sortable) {
      const button = element('button', {
        className: 'requests-table__sort',
        text: column.label,
        attrs: {
          type: 'button',
          'aria-label': `Sort by ${column.label}`,
        },
      });
      button.addEventListener('click', () => onSort(column.key));
      cell.append(button);
    } else {
      cell.textContent = column.label;
    }
    row.append(cell);
  }
  return element('thead', { children: [row] });
}

/**
 * @param {any} request
 * @returns {HTMLElement}
 */
function buildRow(request) {
  const marker = request.marker ?? '●';
  const slot = Number.isInteger(request.sequenceSlot) ? request.sequenceSlot : 0;

  const row = element('tr', {
    className: 'requests-table__row',
    attrs: { 'data-seq': String(slot % 8) },
  });

  row.append(
    element('th', {
      attrs: { scope: 'row' },
      children: [
        element('span', {
          className: 'requests-table__marker',
          text: marker,
          attrs: { 'aria-hidden': 'true' },
        }),
        element('span', { text: `#${request.id}` }),
      ],
    }),
  );

  row.append(element('td', { children: [renderState(request.state)] }));
  row.append(cell(request.sentAtOffsetMs, (value) => formatDuration(value), 'Sent at'));
  row.append(cell(request.ttftMs, (value) => formatDuration(value), 'Time to first token'));
  row.append(
    element('td', {
      className: 'requests-table__tokens',
      children: [
        renderField(request.promptTokens, { label: `Request ${request.id} prompt tokens` }),
        element('span', { text: '/' }),
        renderField(request.outputTokens, { label: `Request ${request.id} output tokens` }),
      ],
    }),
  );
  row.append(cell(request.tokensPerSecond, (value) => formatNumber(value), 'Tokens per second'));
  row.append(cell(request.kvBlocks, (value) => formatNumber(value), 'KV blocks held'));
  row.append(cell(request.reusedTokens, (value) => formatNumber(value), 'Tokens reused from cache'));
  row.append(
    element('td', {
      className: 'requests-table__finish',
      text: request.finishReason ?? '—',
      attrs: {
        title: request.finishReason
          ? `Finish reason reported by the server: ${request.finishReason}`
          : 'The request has not finished yet.',
      },
    }),
  );

  return row;
}

/**
 * @param {any} field
 * @param {(value: number) => string} format
 * @param {string} label
 * @returns {HTMLElement}
 */
function cell(field, format, label) {
  return element('td', { children: [renderField(field, { format, label })] });
}

/**
 * @param {string|undefined} state
 * @returns {HTMLElement}
 */
function renderState(state) {
  const known = CLIENT_STATES[state ?? ''] ?? null;
  if (!known) {
    return element('span', {
      className: 'request-state request-state--unknown',
      text: '—',
      attrs: {
        title: 'This request has no client-observed state yet.',
        'aria-label': 'State unknown',
      },
    });
  }
  return element('span', {
    className: ['request-state', `request-state--${state}`],
    attrs: { title: known.hint, 'aria-label': known.label },
    children: [
      element('span', {
        className: 'request-state__glyph',
        text: known.glyph,
        attrs: { 'aria-hidden': 'true' },
      }),
      element('span', { className: 'request-state__label', text: known.label }),
    ],
  });
}

/**
 * Sort requests, keeping unmeasured values at the end regardless of direction.
 *
 * An em-dash sorting into the middle of a numeric column reads as a value
 * between its neighbours, which is precisely the impression it must not give.
 *
 * @param {any[]} requests
 * @param {string} key
 * @param {boolean} ascending
 * @returns {any[]}
 */
export function sortRequests(requests, key, ascending) {
  const accessor = {
    id: (request) => request.id,
    state: (request) => request.state,
    sent: (request) => numericValueOf(request.sentAtOffsetMs),
    ttft: (request) => numericValueOf(request.ttftMs),
    rate: (request) => numericValueOf(request.tokensPerSecond),
    kvBlocks: (request) => numericValueOf(request.kvBlocks),
    reused: (request) => numericValueOf(request.reusedTokens),
  }[key];

  if (!accessor) {
    return [...requests];
  }

  return [...requests].sort((left, right) => {
    const a = accessor(left);
    const b = accessor(right);
    const aMissing = a === null || a === undefined;
    const bMissing = b === null || b === undefined;
    if (aMissing && bMissing) return 0;
    if (aMissing) return 1;
    if (bMissing) return -1;
    if (a === b) return 0;
    const order = a < b ? -1 : 1;
    return ascending ? order : -order;
  });
}

/**
 * @param {any[]} requests
 * @returns {string}
 */
function buildDescription(requests) {
  const counts = new Map();
  for (const request of requests) {
    counts.set(request.state, (counts.get(request.state) ?? 0) + 1);
  }
  const breakdown = [...counts.entries()]
    .map(([state, count]) => `${count} ${CLIENT_STATES[state]?.label ?? 'unknown'}`)
    .join(', ');

  const rates = requests.map((request) => numericValueOf(request.tokensPerSecond)).filter((rate) => rate !== null);
  const rateSentence =
    rates.length > 0
      ? ` Per-request throughput ranges from ${formatNumber(Math.min(...rates))} to ${formatNumber(
          Math.max(...rates),
        )} tokens per second.`
      : ' No per-request throughput has been measured yet.';

  const measuredTtft = requests.filter((request) => isRenderable(request.ttftMs)).length;
  return (
    `Requests: ${requests.length} in this scenario — ${breakdown}.${rateSentence} ` +
    `${measuredTtft} of ${requests.length} have a measured time to first token.`
  );
}

export { mount };
