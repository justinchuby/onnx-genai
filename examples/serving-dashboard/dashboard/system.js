// Copyright (c) Microsoft Corporation.
//
// Panel 6 — System. demo-ux.md §5.7.
//
// Model identity, resource accounting, and — the self-aware part — the
// dashboard's own polling cost. Showing our own overhead is not vanity: a
// measurement tool that reports what it costs is making a statement about what
// kind of tool it is, and it is the honest counterpart to the <2% telemetry
// overhead claim the server side is proving separately.
//
// TWO LABELLING TRAPS FROM THE PROVENANCE AUDIT (§3), both of which would be
// fabrications of ATTRIBUTION rather than of value — the numbers are real, the
// obvious names for them are wrong:
//
//   • `/v1/resources.vram.*` is the scheduler's cross-session KV BYTE BUDGET
//     (governor.rs:548-554, from byte_budget.snapshot()), not an NVML device
//     query. Labelling it "GPU memory used" would be a lie about what was
//     measured. It is labelled "KV bytes reserved".
//
//   • `/v1/resources.host_ram.*` is a WHOLE-MACHINE capacity query
//     (governor.rs:575-579, total_bytes() - free_bytes()). It includes every
//     other process on the box. Labelling it "onnx-genai memory" would
//     attribute the whole machine's usage to this server.
//
//   • `active_sessions` counts PERSISTENT X-Session-Id sessions, not in-flight
//     requests. A dashboard driving 4 concurrent requests will show 0 here
//     unless the client opted into sessions, which looks like a bug and is not.

import { isRenderable, numericValueOf } from './field-state.js';
import {
  createRepaintScheduler,
  describeFieldText,
  bindPanel,
  element,
  formatDuration,
  observeVisibility,
  replaceChildren,
  sectionLabel,
} from './panel-kit.js';

/** Bytes per binary unit step. */
const BYTE_UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];

export const meta = Object.freeze({
  // Server identity and health exist on every profile.
  id: 'system',
  title: 'System',
  group: 'system',
  span: 1,
  cadence: 1000,
  // Model identity and configured ceilings are resolved at startup and do not drift.
  staleCeilingMs: 30000,
  defaultOpen: false,
  acronyms: {
    EP: 'Execution provider — the backend ONNX Runtime dispatches operators to',
    KV: 'Key/Value attention cache — the per-token state attention reads on every step',
    RTT: 'Round-trip time — how long one telemetry poll takes end to end',
    RSS: 'Resident set size — physical memory held by a process',
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
  const model = element('dl', { className: 'panel-system__model' });
  const resources = element('div', { className: 'panel-system__resources' });
  const selfReport = element('div', { className: 'panel-system__self' });

  rootElement.append(
    sectionLabel('model'),
    model,
    sectionLabel('resources'),
    resources,
    sectionLabel('this dashboard'),
    selfReport,
  );

  let description = 'System: waiting for the first sample.';

  const paint = () => {
    replaceChildren(model, [
      ...definition('model id', telemetryStore.field('server.model_id')),
      ...definition('context length', telemetryStore.field('server.context_length')),
      ...definition('execution provider', telemetryStore.field('server.execution_provider')),
      ...definition('decode backend', telemetryStore.field('server.decode_backend')),
      ...definition('quantization', telemetryStore.field('server.quantization')),
      ...definition('server version', telemetryStore.field('server.version')),
      ...definition('uptime', telemetryStore.field('server.uptime_ms'), (value) => formatDuration(value)),
    ]);

    replaceChildren(resources, [
      renderBudgetRow(
        'VRAM limit',
        telemetryStore.field('resources.vram_limit_bytes'),
        'The configured VRAM ceiling the scheduler plans against. It is a limit, not a ' +
          'reading: the server does not query the device, so this is what it was told it may ' +
          'use, not what it is using.',
      ),
      renderBudgetRow(
        'derived KV budget',
        telemetryStore.field('resources.kv_budget_bytes'),
        'The VRAM limit minus reserved bytes — the ceiling on cross-session KV. Also a budget ' +
          'rather than a measurement, and not the number nvidia-smi would show.',
      ),
      renderDiskSpill(telemetryStore.field('resources.disk_spill_bytes')),
      metricRow('persistent sessions', telemetryStore.field('sessions.active'), {
        label:
          'Persistent sessions \u2014 clients using an X-Session-Id header. This is not the number of ' +
          'in-flight requests, and it is legitimately 0 unless a client opted in.',
      }),
    ]);

    replaceChildren(selfReport, [
      metricRow('poll round-trip', telemetryStore.field('client.poll_rtt_ms'), {
        format: (value) => formatDuration(value),
      }),
      metricRow('poll interval', telemetryStore.field('client.poll_interval_ms'), {
        format: (value) => formatDuration(value),
      }),
      metricRow('dropped frames', telemetryStore.field('client.dropped_frames')),
      renderConnection(telemetryStore.connection?.()),
    ]);

    description = buildDescription(telemetryStore);
  };

  const scheduler = createRepaintScheduler(rootElement, paint, { telemetryStore });
  const stopObserving = observeVisibility(rootElement, (visible) => scheduler.setVisible(visible));
  const unsubscribe = telemetryStore.subscribe(() => scheduler.request());
  scheduler.request();

  return {
    unmount() {
      unsubscribe();
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
 * A `<dt>`/`<dd>` pair. Returned as an array so callers can spread it into a
 * definition list without an intervening wrapper, which would break `<dl>`
 * semantics for a screen reader.
 *
 * @param {string} label
 * @param {any} field
 * @param {(value: number) => string} [format]
 * @returns {HTMLElement[]}
 */
function definition(label, field, format) {
  return [
    element('dt', { text: label }),
    element('dd', { children: [renderField(field, { label, format })] }),
  ];
}

/**
 * A used/limit pair with an explicit accounting caveat.
 *
 * @param {string} label
 * @param {any} used
 * @param {any} limit
 * @param {string} caveat
 * @returns {HTMLElement}
 */
function renderBudgetRow(label, budget, caveat) {
  // Deliberately not a usage bar. /v1/resources publishes ceilings only — there
  // is no consumption figure behind them — so a fill bar would invent the very
  // number the visitor would read off it. The word "limit" carries the meaning
  // instead, and it is in the row, not only in the tooltip.
  return element('div', {
    className: 'resource-row',
    children: [
      element('span', {
        className: 'resource-row__label',
        text: label,
        attrs: { title: caveat, tabindex: '-1', 'data-roving-item': '' },
      }),
      renderField(budget, { label: `${label} (a configured ceiling, not a usage reading)`, format: formatBytes }),
      element('span', { className: 'resource-row__of', text: 'ceiling' }),
    ],
  });
}

/**
 * Disk spill is an `Option` server-side: absent means "not configured", which
 * is honest and must not render as 0 bytes. This is the one place the codebase
 * already gets absence right, and the UI should not undo that.
 *
 * @param {any} diskSpill
 * @returns {HTMLElement}
 */
function renderDiskSpill(diskSpill) {
  // NUMERIC OR NOTHING. The wire type is Option<u64> (ResolvedResourceLimits,
  // routes/mod.rs:454), so a string here is never a legitimate reading -- and
  // the store cannot filter it for us, because it promotes any unexpected value
  // to state="measured". Without this, a server that put a spill DIRECTORY in
  // the field would have this row paint an absolute filesystem path, which is
  // the model_path disclosure again under a different key.
  if (isRenderable(diskSpill) && typeof diskSpill.value === 'number') {
    return metricRow('disk spill', diskSpill, { format: formatBytes });
  }
  if (isRenderable(diskSpill)) {
    return element('div', {
      className: 'resource-row resource-row--absent',
      children: [
        element('span', { className: 'resource-row__label', text: 'disk spill' }),
        element('span', {
          className: 'resource-row__absent',
          text: 'unreadable',
          attrs: {
            tabindex: '-1',
            'data-roving-item': '',
            title:
              'The server sent a disk-spill value that is not a byte count. It is not shown ' +
              'because it cannot be read as one, and showing it verbatim could disclose a ' +
              'filesystem path.',
          },
        }),
      ],
    });
  }
  return element('div', {
    className: 'resource-row resource-row--absent',
    children: [
      element('span', { className: 'resource-row__label', text: 'disk spill' }),
      element('span', {
        className: 'resource-row__absent',
        text: 'not configured',
        attrs: {
          tabindex: '-1',
          'data-roving-item': '',
          title:
            'No disk-spill tier is configured on this server. This is a configuration state, ' +
            'not a measurement of zero.',
        },
      }),
    ],
  });
}

/**
 * @param {{state?: string, rttMs?: number, retryInMs?: number, attempt?: number}|undefined} connection
 * @returns {HTMLElement}
 */
function renderConnection(connection) {
  const state = connection?.state ?? 'unknown';
  const labels = {
    live: 'live',
    slow: 'slow — polls are taking longer than the interval',
    retrying: 'retrying',
    offline: 'offline',
    unknown: 'unknown',
  };
  return element('div', {
    className: ['connection', `connection--${state}`],
    attrs: { role: 'status', 'aria-label': `Telemetry connection: ${labels[state] ?? state}` },
    children: [
      // A glyph as well as a colour: connection state is exactly the kind of
      // status that gets encoded in a coloured dot and nothing else (AC25).
      element('span', {
        className: 'connection__glyph',
        text: state === 'live' ? '●' : state === 'offline' ? '○' : '◐',
        attrs: { 'aria-hidden': 'true' },
      }),
      element('span', { className: 'connection__label', text: labels[state] ?? state }),
    ],
  });
}

/**
 * Format a byte count in binary units.
 *
 * @param {number} bytes
 * @returns {string}
 */
export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) {
    return '—';
  }
  if (bytes === 0) {
    return '0 B';
  }
  const exponent = Math.min(BYTE_UNITS.length - 1, Math.floor(Math.log2(Math.abs(bytes)) / 10));
  const scaled = bytes / 1024 ** exponent;
  const digits = scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2;
  return `${scaled.toFixed(digits)} ${BYTE_UNITS[exponent]}`;
}

/**
 * @param {any} telemetryStore
 * @returns {string}
 */
function buildDescription(telemetryStore) {
  const parts = ['System:'];
  parts.push(
    `${describeFieldText('model', telemetryStore.field('server.model_id'))}.`,
  );
  parts.push(`${describeFieldText('Context length', telemetryStore.field('server.context_length'))}.`);
  parts.push(
    `${describeFieldText('Execution provider', telemetryStore.field('server.execution_provider'))}.`,
  );
  parts.push(
    `${describeFieldText('KV bytes reserved', telemetryStore.field('resources.vram_limit_bytes'), formatBytes)}.`,
  );
  parts.push(
    `${describeFieldText('Telemetry poll round-trip', telemetryStore.field('client.poll_rtt_ms'), formatDuration)}.`,
  );
  return parts.join(' ');
}

export { mount };
