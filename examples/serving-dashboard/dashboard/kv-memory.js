// Copyright (c) Microsoft Corporation.
//
// Panel 3 — KV memory. demo-ux.md §5.4.
//
// The numeric companion to the Paged KV block table scenario: it carries the
// numbers a grid cannot show precisely.
//
// NAMING, deliberately: this panel is about the paged KV ALLOCATOR. Paged
// ATTENTION kernels are not implemented in this runtime — attention still runs
// over materialized KV. The allocator is real, and the allocator is what we
// show. The panel never uses the phrase "paged attention".
//
// SLOT FILL IS SHOWN ON PURPOSE, with its explanation inline. It is the honest
// cost of paging — partially filled blocks — and volunteering an imperfection
// is the strongest available signal that this visualization is a real
// measurement rather than a marketing render. Hiding it would be the tell.

import { isRenderable, numericValueOf, ratioField } from './field-state.js';
import {
  REASONS,
  capabilityNotice,
  createRepaintScheduler,
  describeFieldText,
  bindPanel,
  element,
  formatNumber,
  observeVisibility,
  replaceChildren,
  sectionLabel,
} from './panel-kit.js';

export const meta = Object.freeze({
  // §13(d): deliberately null, NOT 'paged-kv'. The panel stays mounted on a
  // static-cache model so the server's written not-applicable reason remains
  // visible; it does not reinterpret admitted HTTP generations as decode rows.
  id: 'kv-memory',
  title: 'KV memory',
  group: 'memory',
  span: 2,
  // 1 Hz, deliberately NOT the 4 Hz the gauges run at. Block-table detail is
  // served from its own endpoint at a lower cadence because a 4096-block grid
  // at 4 Hz is roughly 1 MB/s of traffic to animate something that changes on
  // allocation, not on every decode step. Asking for frames the server will not
  // send would render as permanent staleness on a perfectly healthy panel.
  // Nothing here interpolates between samples: the sparkline infers each
  // series' own median interval, so a slower feed draws correctly rather than
  // being shredded against a faster panel's assumed spacing.
  cadence: 1000,
  // Block accounting moves with allocation, not with every decode step, so it stays meaningful longer.
  staleCeilingMs: 15000,
  defaultOpen: true,
  acronyms: {
    KV: 'Key/Value attention cache — the per-token state attention reads on every step',
    COW: 'Copy-on-write — sharing a block between sequences until one of them needs to diverge',
    refcount: 'How many sequences currently share a given block',
    eviction: 'Reclaiming a block so its memory can serve another sequence',
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
  let description = 'KV memory: waiting for the first sample.';

  const paint = () => {
    const capability = telemetryStore.capability?.('kv-introspection') ?? { available: true };

    if (!capability.available) {
      // A grid of hatch marks for an entire absent feature is noise. A notice
      // that names the flag turns "this dashboard is broken" into "this panel
      // needs a flag", which is the true state of the world (AC20).
      replaceChildren(rootElement, [
        capabilityNotice({
          title: 'KV block introspection is off',
          body: capability.reason ?? REASONS.KV_NOT_EXPOSED,
          command: capability.fix ?? '--enable-debug-endpoints',
        }),
      ]);
      description = `KV memory: not available. ${capability.reason ?? REASONS.KV_NOT_EXPOSED}`;
      return;
    }

    const blocksUsed = telemetryStore.field('kv.pages_used');
    const blocksTotal = telemetryStore.field('kv.pages_total');
    const blockSize = telemetryStore.field('kv.block_size');
    const shared = telemetryStore.field('kv.pages_shared');
    const slotsFilled = telemetryStore.field('kv.slots_filled');
    const slotCapacity = telemetryStore.field('kv.slot_capacity');

    const utilization = ratioField(blocksUsed, blocksTotal, {
      label: 'KV page utilization',
      unavailableReason: REASONS.KV_NOT_EXPOSED,
    });
    const sharedShare = ratioField(shared, blocksUsed, {
      label: 'Share of blocks that are shared',
      unavailableReason: 'Sharing is a proportion of blocks in use, and one of those is not measured.',
    });
    const slotFill = ratioField(slotsFilled, slotCapacity, {
      label: 'Slot fill efficiency',
      unavailableReason: 'Slot fill needs both filled slots and slot capacity from the engine.',
    });

    replaceChildren(rootElement, [
      element('div', {
        className: 'panel-kv-memory__hero',
        children: [
          renderField(utilization),
          element('span', { className: 'hero-figure__caption', text: 'utilization' }),
          renderBlockFraction(blocksUsed, blocksTotal),
        ],
      }),
      renderUtilizationBar(utilization, blocksUsed, blocksTotal),
      element('div', {
        className: 'panel-kv-memory__grid',
        children: [
          metricRow('block size', blockSize),
          metricRow('shared', shared),
          metricRow('shared share', sharedShare),
          metricRow('slot fill', slotFill),
        ],
      }),
      renderSlotFillNote(slotsFilled, slotCapacity),
      sectionLabel('refcount distribution'),
      renderRefcounts(telemetryStore.field('kv.refcount_histogram')),
      sectionLabel('tiers'),
      renderTiers(telemetryStore.field('kv.tiers')),
      sectionLabel('lifetime counters'),
      element('div', {
        className: 'panel-kv-memory__counters',
        children: [
          metricRow('allocations', telemetryStore.field('kv.allocations')),
          metricRow('frees', telemetryStore.field('kv.frees')),
          metricRow('alloc failures', telemetryStore.field('kv.allocation_failures')),
          metricRow('hot evictions', telemetryStore.field('kv.hot_evictions')),
          metricRow('prefix evictions', telemetryStore.field('kv.prefix_evictions')),
        ],
      }),
    ]);

    description = buildDescription({ utilization, blocksUsed, blocksTotal, shared, slotFill });
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
 * "318 / 512 blocks", with each half independently honest.
 *
 * @param {any} blocksUsed
 * @param {any} blocksTotal
 * @returns {HTMLElement}
 */
function renderBlockFraction(blocksUsed, blocksTotal) {
  // NO `label` OVERRIDE, DELIBERATELY. These two carried 'KV blocks in use'
  // and 'KV blocks total', and the visible unit beside them read 'blocks'.
  // The server does not have blocks. The wire fields are `kv_pages_used` and
  // `kv_pages_total` (crates/onnx-genai-server/src/routes/mod.rs), the
  // catalogue says 'KV pages in use' and 'KV pages total', and the
  // provenance entry cites admin.rs by name. The noun was invented here.
  //
  // It survived because renderField resolves `options.label ?? field.label`,
  // so a caller override silently wins over the catalogue and no amount of
  // fixing the catalogue could reach this widget. Deleting the overrides is
  // the fix; caption-catalogue.test.js is what stops it coming back.
  return element('span', {
    className: 'block-fraction',
    children: [
      renderField(blocksUsed, {}),
      element('span', { className: 'block-fraction__slash', text: '/' }),
      renderField(blocksTotal, {}),
      element('span', { className: 'block-fraction__unit', text: 'pages' }),
    ],
  });
}

/**
 * The utilization bar. Drawn only when the ratio is real — an empty bar would
 * read as 0% utilization, which is a measurement we do not have.
 *
 * @param {any} utilization
 * @param {any} blocksUsed
 * @param {any} blocksTotal
 * @returns {HTMLElement}
 */
function renderUtilizationBar(utilization, blocksUsed, blocksTotal) {
  // `fraction`, not `value`: a discrete ratio carries its NUMERATOR in `value`
  // (`3 of 4`), so measuring the bar from `value` would draw 3% and quietly
  // contradict the text beside it.
  const percentage =
    typeof utilization?.fraction === 'number' ? utilization.fraction * 100 : numericValueOf(utilization);
  if (percentage === null) {
    return element('div', {
      className: ['utilization-bar', 'utilization-bar--unavailable'],
      attrs: {
        tabindex: '-1',
        'data-roving-item': '',
        role: 'note',
        'aria-label': `KV block utilization: not measurable yet. ${utilization.reason ?? ''}`,
        title: utilization.reason ?? REASONS.KV_NOT_EXPOSED,
      },
    });
  }

  const bar = element('div', {
    className: 'utilization-bar',
    attrs: {
      role: 'meter',
      'aria-valuenow': percentage.toFixed(1),
      'aria-valuemin': '0',
      'aria-valuemax': '100',
      'aria-label': `KV block utilization ${percentage.toFixed(1)} percent, ${
        numericValueOf(blocksUsed) ?? '?'
      } of ${numericValueOf(blocksTotal) ?? '?'} blocks`,
    },
  });
  bar.append(
    element('span', {
      className: 'utilization-bar__fill',
      attrs: { style: `width:${Math.min(100, percentage).toFixed(2)}%` },
    }),
  );
  return bar;
}

/**
 * The self-critical note. Only shown when there is a real gap to explain —
 * otherwise it is a disclaimer with nothing to disclaim.
 *
 * @param {any} slotsFilled
 * @param {any} slotCapacity
 * @returns {HTMLElement|null}
 */
function renderSlotFillNote(slotsFilled, slotCapacity) {
  const filled = numericValueOf(slotsFilled);
  const capacity = numericValueOf(slotCapacity);
  if (filled === null || capacity === null || capacity === 0) {
    return null;
  }
  return element('p', {
    className: 'panel-kv-memory__note',
    text:
      `${formatNumber(filled)} of ${formatNumber(capacity)} token slots hold data. ` +
      'The gap is the cost of paging: blocks are allocated whole, so the last block of ' +
      'each sequence is usually partly empty. This is real and we show it.',
  });
}

/**
 * Refcount distribution as a bar list.
 *
 * Three to five discrete integers do not need an axis, and a bar list is
 * exactly readable — each row states its own count in text, so nothing depends
 * on comparing bar lengths by eye.
 *
 * @param {any} histogram
 * @returns {HTMLElement}
 */
function renderRefcounts(histogram) {
  if (!isRenderable(histogram) || !Array.isArray(histogram.value)) {
    return renderField(histogram);
  }

  const entries = histogram.value
    .map((entry) => ({ refcount: Number(entry.refcount), blocks: Number(entry.blocks) }))
    .filter((entry) => Number.isFinite(entry.refcount) && Number.isFinite(entry.blocks))
    .sort((a, b) => a.refcount - b.refcount);

  if (entries.length === 0) {
    return element('p', {
      className: 'refcounts__empty',
      text: 'No blocks are allocated, so there is no sharing to report.',
    });
  }

  const largest = Math.max(...entries.map((entry) => entry.blocks));
  const list = element('ul', {
    className: 'refcounts',
    attrs: { 'aria-label': 'Blocks grouped by how many sequences share them' },
  });

  for (const entry of entries) {
    const width = largest > 0 ? (entry.blocks / largest) * 100 : 0;
    list.append(
      element('li', {
        className: 'refcounts__row',
        children: [
          element('span', {
            className: 'refcounts__key',
            text: String(entry.refcount),
            attrs: {
              title:
                entry.refcount === 1
                  ? 'Held by one sequence only'
                  : `Shared by ${entry.refcount} sequences (copy-on-write)`,
            },
          }),
          element('span', {
            className: 'refcounts__bar',
            attrs: { style: `width:${width.toFixed(1)}%` },
          }),
          element('span', { className: 'refcounts__count', text: formatNumber(entry.blocks) }),
        ],
      }),
    );
  }
  return list;
}

/**
 * @param {any} tiers
 * @returns {HTMLElement}
 */
function renderTiers(tiers) {
  if (!isRenderable(tiers) || !Array.isArray(tiers.value)) {
    return renderField(tiers);
  }
  const list = element('ul', { className: 'kv-tiers' });
  for (const tier of tiers.value) {
    list.append(
      element('li', {
        className: 'kv-tiers__row',
        children: [
          element('span', { className: 'kv-tiers__name', text: String(tier.name) }),
          element('span', { className: 'kv-tiers__pages', text: formatNumber(Number(tier.pages)) }),
        ],
      }),
    );
  }
  return list;
}

/**
 * @param {Record<string, any>} fields
 * @returns {string}
 */
function buildDescription(fields) {
  const parts = ['KV memory:'];

  if (isRenderable(fields.blocksUsed) && isRenderable(fields.blocksTotal)) {
    parts.push(`${fields.blocksUsed.value} of ${fields.blocksTotal.value} blocks in use.`);
  } else {
    parts.push('Block counts are not measurable yet.');
  }

  parts.push(`${describeFieldText('Utilization', fields.utilization)}.`);
  parts.push(`${describeFieldText('Blocks shared', fields.shared)}.`);
  parts.push(`${describeFieldText('Slot fill', fields.slotFill)}.`);
  return parts.join(' ');
}

export { mount };
