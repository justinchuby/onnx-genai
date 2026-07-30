// Copyright (c) Microsoft Corporation.
//
// Panel 1 — Throughput & latency. demo-ux.md §5.2.
//
// TWO DESIGN DECISIONS WORTH DEFENDING:
//
// 1. Aggregate tokens/sec is DERIVED, never read from `/v1/status`.
//    `/v1/status.tokens_per_second` is a hardcoded 0.0 (`admin.rs:63`, comment
//    `// only cumulative token totals recorded`). The honest number is the
//    client-side derivative of the cumulative completion-token counter, and it
//    carries the `derived` badge to say so. Binding this panel's headline figure
//    to the status field would have rendered "0.0 tok/s" as a measurement of a
//    runtime that was, at that moment, generating tokens.
//
// 2. Client TTFT and server TTFT are shown as two rows, not reconciled into one.
//    They will diverge, and the divergence IS the network and serialization
//    overhead. A page that shows its own measurement disagreeing slightly with
//    the server's is more credible than one showing a single confident number,
//    and picking a winner silently would throw away the cross-check.

import { isRenderable, numericValueOf } from './field-state.js';
import {
  createRepaintScheduler,
  bindPanel,
  createSparklineSlot,
  describeFieldText,
  element,
  formatDuration,
  observeVisibility,
  replaceChildren,
  sectionLabel,
  readRequests,
  REQUEST_TABLE_EMPTY,
  renderSparkline,
} from './panel-kit.js';

const WINDOW_MS = 60_000;

/** Divergence above this fraction is called out rather than silently absorbed. */
const TTFT_DIVERGENCE_WARN_RATIO = 0.2;

export const meta = Object.freeze({
  id: 'throughput',
  title: 'Throughput & latency',
  group: 'throughput',
  span: 2,
  cadence: 250,
  // A tokens/sec figure three seconds old is describing a different moment of the run.
  staleCeilingMs: 3000,
  defaultOpen: true,
  acronyms: {
    TTFT: 'Time to first token — how long until the first token of a response arrives',
    ITL: 'Inter-token latency — the gap between consecutive tokens once streaming has begun',
    TPOT: 'Time per output token — total generation time divided by tokens produced',
    e2e: 'End to end — the full request duration, from arrival to completion',
    makespan: 'The wall-clock span from the first request being sent to the last one completing',
  },
});

/** Panel-scoped renderers carrying this panel's stale ceiling (AC45(c)). */
const { metricRow, renderField } = bindPanel(meta);

/**
 * @param {HTMLElement} rootElement
 * @param {any} telemetryStore
 * @returns {{destroy(): void, describe(): string}}
 */
export default function mount(rootElement, telemetryStore) {
  const hero = element('div', { className: 'panel-throughput__hero' });
  const perRequest = element('div', { className: 'panel-throughput__per-request' });
  const latency = element('table', {
    className: 'latency-table',
    attrs: { 'aria-label': 'Latency percentiles over the last 60 seconds' },
  });
  const makespan = element('div', { className: 'panel-throughput__makespan' });

  const heroSpark = createSparklineSlot({ label: 'Aggregate output tokens per second', width: 260, height: 40 });

  rootElement.append(
    hero,
    heroSpark.root,
    sectionLabel('per request'),
    perRequest,
    sectionLabel('latency'),
    latency,
    makespan,
  );

  let description = 'Throughput: waiting for the first sample.';

  const paint = () => {
    // The server publishes its own tokens-per-second field. We deliberately do
    // not bind it: it is classified DOCUMENTED_ZERO, meaning the server emits a
    // literal 0.0 because it cannot compute the rate. The honest number is the
    // client-side derivative of the token counter that IS measured, and it
    // arrives badged `derived` so the provenance is visible rather than implied.
    // A test asserts this module never reaches for that field.
    const aggregate = telemetryStore.rate('metrics.tokens_generated_total', {
      windowMs: WINDOW_MS,
      unit: 'tok/s',
    });

    replaceChildren(hero, [
      element('div', {
        className: 'hero-figure',
        children: [
          renderField(aggregate, { label: 'Aggregate output tokens per second' }),
          element('span', { className: 'hero-figure__caption', text: 'aggregate output' }),
        ],
      }),
    ]);

    replaceChildren(perRequest, [renderPerRequest(telemetryStore)]);
    replaceChildren(latency, buildLatencyRows(telemetryStore));
    replaceChildren(makespan, [
      metricRow('makespan', telemetryStore.field('scenario.makespan_ms'), {
        format: (value) => formatDuration(value),
      }),
      element('span', {
        className: 'panel-throughput__makespan-note',
        text: 'scenario start → last completion',
      }),
    ]);

    const heroSeries = telemetryStore.rateSeries('metrics.tokens_generated_total', WINDOW_MS);
    renderSparkline(heroSpark, heroSeries, {
      width: 260,
      height: 40,
      windowMs: WINDOW_MS,
      label: 'Aggregate output tokens per second',
      unit: 'tok/s',
    });

    description = buildDescription(telemetryStore, aggregate);
  };

  const scheduler = createRepaintScheduler(rootElement, paint);
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

/**
 * Per-request throughput, capped so the panel stays readable.
 *
 * Above eight requests the panel shows the fastest and slowest three plus a
 * count and defers the rest to the Requests panel. A 32-row list in a 340px
 * panel is a wall, and a wall is not information.
 *
 * @param {any} telemetryStore
 * @returns {HTMLElement}
 */
function renderPerRequest(telemetryStore) {
  const { wired, rows: requests } = readRequests(telemetryStore);
  const container = element('div', { className: 'per-request' });

  if (requests.length === 0) {
    const emptyText = wired ? REQUEST_TABLE_EMPTY.idle : REQUEST_TABLE_EMPTY.unwired;
    container.append(
      element('p', {
        className: 'per-request__empty',
        text: emptyText,
      }),
    );
    return container;
  }

  const withRates = requests.map((request) => ({
    request,
    rate: numericValueOf(request.tokensPerSecond),
  }));

  let shown = withRates;
  let elided = 0;
  if (withRates.length > 8) {
    const ranked = [...withRates].sort((a, b) => (b.rate ?? -1) - (a.rate ?? -1));
    shown = [...ranked.slice(0, 3), ...ranked.slice(-3)];
    elided = withRates.length - shown.length;
  }

  for (const { request } of shown) {
    container.append(renderRequestChip(request));
  }
  if (elided > 0) {
    container.append(
      element('span', {
        className: 'per-request__elided',
        text: `+${elided} more in the Requests panel`,
      }),
    );
  }
  return container;
}

/**
 * One request's throughput chip.
 *
 * The sequence marker glyph is carried alongside the colour so the identity
 * survives a grayscale screenshot and a colourblind reader (AC25) — the same
 * property, earned once and spent twice.
 *
 * @param {any} request
 * @returns {HTMLElement}
 */
function renderRequestChip(request) {
  const marker = request.marker ?? '●';
  const slot = Number.isInteger(request.sequenceSlot) ? request.sequenceSlot : 0;
  return element('span', {
    className: 'request-chip',
    attrs: { 'data-seq': String(slot % 8) },
    children: [
      element('span', { className: 'request-chip__marker', text: marker, attrs: { 'aria-hidden': 'true' } }),
      element('span', { className: 'request-chip__id', text: `#${request.id}` }),
      renderField(request.tokensPerSecond, { label: `Request ${request.id} tokens per second` }),
    ],
  });
}

/**
 * The latency table: one row per measure, one column per percentile.
 *
 * @param {any} telemetryStore
 * @returns {HTMLElement[]}
 */
function buildLatencyRows(telemetryStore) {
  const header = element('tr', {
    children: [
      element('th', { text: '', attrs: { scope: 'col' } }),
      element('th', { text: 'p50', attrs: { scope: 'col' } }),
      element('th', { text: 'p95', attrs: { scope: 'col' } }),
      element('th', { text: 'max', attrs: { scope: 'col' } }),
    ],
  });

  const definitions = [
    { label: 'TTFT', prefix: 'latency.ttft_client', suffix: 'client', note: 'measured in this browser' },
    { label: 'TTFT', prefix: 'latency.ttft_server', suffix: 'server', note: 'server histogram' },
    { label: 'ITL', prefix: 'latency.itl_client', suffix: 'client', note: 'measured in this browser' },
    { label: 'TPOT', prefix: 'latency.tpot_client', suffix: 'client', note: 'measured in this browser' },
    { label: 'e2e', prefix: 'latency.e2e_server', suffix: 'server', note: 'server histogram' },
  ];

  const rows = definitions.map((definition) => {
    const row = element('tr', { className: 'latency-table__row' });
    row.append(
      element('th', {
        className: 'latency-table__label',
        attrs: { scope: 'row', title: definition.note },
        children: [
          element('span', { text: definition.label }),
          element('span', { className: 'latency-table__origin', text: definition.suffix }),
        ],
      }),
    );
    for (const percentile of ['p50', 'p95', 'max']) {
      const field = telemetryStore.field(`${definition.prefix}_${percentile}`);
      row.append(
        element('td', {
          children: [
            renderField(field, {
              label: `${definition.label} ${definition.suffix} ${percentile}`,
              format: (value) => formatDuration(value),
            }),
          ],
        }),
      );
    }
    return row;
  });

  const divergence = ttftDivergenceNote(telemetryStore);
  if (divergence) {
    rows.push(divergence);
  }

  return [element('thead', { children: [header] }), element('tbody', { children: rows })];
}

/**
 * Flag a large gap between the client's TTFT and the server's.
 *
 * Showing the disagreement rather than resolving it is the point: it is a
 * public demonstration that the page cross-checks itself.
 *
 * @param {any} telemetryStore
 * @returns {HTMLElement|null}
 */
function ttftDivergenceNote(telemetryStore) {
  const clientP50 = numericValueOf(telemetryStore.field('latency.ttft_client_p50'));
  const serverP50 = numericValueOf(telemetryStore.field('latency.ttft_server_p50'));
  if (clientP50 === null || serverP50 === null || serverP50 === 0) {
    return null;
  }
  const divergence = Math.abs(clientP50 - serverP50) / serverP50;
  if (divergence < TTFT_DIVERGENCE_WARN_RATIO) {
    return null;
  }
  const cell = element('td', {
    className: 'latency-table__divergence',
    attrs: { colspan: '4' },
    text:
      `Client and server TTFT differ by ${(divergence * 100).toFixed(0)}%. ` +
      'The client measures network and streaming framing too; the server does not. ' +
      'Both numbers are real — neither is corrected against the other.',
  });
  return element('tr', { className: 'latency-table__row--warn', children: [cell] });
}

/**
 * @param {any} telemetryStore
 * @param {any} aggregate
 * @returns {string}
 */
function buildDescription(telemetryStore, aggregate) {
  const parts = ['Throughput and latency:'];
  parts.push(`${describeFieldText('aggregate output', aggregate)}.`);

  const clientTtft = telemetryStore.field('latency.ttft_client_p50');
  const serverTtft = telemetryStore.field('latency.ttft_server_p50');
  parts.push(
    `${describeFieldText('TTFT p50 measured in the browser', clientTtft, formatDuration)}; ` +
      `${describeFieldText('TTFT p50 reported by the server', serverTtft, formatDuration)}.`,
  );

  const makespan = telemetryStore.field('scenario.makespan_ms');
  if (isRenderable(makespan)) {
    parts.push(`Makespan ${formatDuration(Number(makespan.value))}.`);
  }

  const requestCount = readRequests(telemetryStore).rows.length;
  parts.push(`${requestCount} request${requestCount === 1 ? '' : 's'} in this scenario.`);
  return parts.join(' ');
}

export { mount };
