// Copyright (c) Microsoft Corporation.
//
// Panel 2 — Scheduling & batching. demo-ux.md §5.3.
//
// THE LESSON THIS PANEL TEACHES: a visitor watching numbers move needs to know
// whether they are watching a BATCH or a QUEUE. Those are different phenomena
// with different implications, and conflating them is how a demo accidentally
// claims a scheduling property it does not have.
//
// THE TRAP THIS PANEL AVOIDS: batch occupancy is a ratio. Its numerator
// (`active_batch_size`) is genuinely measured; its denominator (max batch size)
// is not surfaced by the server today. The tempting move is to read
// `DEFAULT_MAX_BATCH = 4` out of `state.rs:25` and divide by it. That would be
// a fabricated measurement wearing a division sign, and it is the subtlest AC6
// violation available anywhere on this page — the server may have been built or
// configured differently, and nothing in the UI would reveal it. So: the
// absolute count renders as a real number with its sparkline, and the
// PERCENTAGE renders as an em-dash until the server states its own limit.
//
// Adjacent to it, `rejections: 0` (a real, good zero at full contrast) sits
// beside `preemptions: —` (an honest absence). That single line is the clearest
// teaching example of the unavailable-data language anywhere in the UI.

import { isRenderable, numericValueOf, ratioField } from './field-state.js';
import {
  REASONS,
  createRepaintScheduler,
  createSparklineSlot,
  describeFieldText,
  element,
  metricRow,
  observeVisibility,
  renderField,
  replaceChildren,
  sectionLabel,
  renderSparkline,
} from './panel-kit.js';

/** History window shared by every sparkline on the page. */
const WINDOW_MS = 60_000;

export const meta = Object.freeze({
  id: 'scheduling',
  title: 'Scheduling & batching',
  group: 'scheduling',
  span: 2,
  cadence: 250,
  defaultOpen: true,
  acronyms: {
    batch: 'A group of sequences decoded together in one forward pass',
    queue: 'Requests admitted but waiting for a decode slot',
    occupancy: 'How much of the available batch capacity is in use',
    preemption: 'Evicting a running sequence to free capacity for another',
  },
});

/**
 * Mount the scheduling panel.
 *
 * @param {HTMLElement} rootElement The shell's empty `.panel__body`.
 * @param {any} telemetryStore
 * @returns {{destroy(): void, describe(): string}}
 */
export default function mount(rootElement, telemetryStore) {
  const counters = element('div', { className: 'panel-scheduling__counters' });
  const occupancy = element('div', { className: 'panel-scheduling__occupancy' });
  const queue = element('div', { className: 'panel-scheduling__queue' });
  const footer = element('div', { className: 'panel-scheduling__footer' });

  const occupancySpark = createSparklineSlot({ label: 'Batch occupancy', width: 320, height: 34 });
  const queueSpark = createSparklineSlot({ label: 'Queue depth', width: 320, height: 34 });

  rootElement.append(
    counters,
    sectionLabel('batch occupancy'),
    occupancy,
    occupancySpark.root,
    sectionLabel('queue depth'),
    queue,
    queueSpark.root,
    footer,
  );

  let description = 'Scheduling: waiting for the first sample.';

  const paint = () => {
    const running = telemetryStore.field('scheduler.running');
    const waiting = telemetryStore.field('scheduler.waiting');
    const admissionSlots = telemetryStore.field('admission.slots_available');
    const batchSize = telemetryStore.field('batch.active_size');
    const maxBatch = telemetryStore.field('scheduler.max_batch');
    const queueDepth = telemetryStore.field('queue.depth');
    const preemptions = telemetryStore.field('scheduler.preemptions_total');
    const rejections = telemetryStore.field('admission.rejections');
    const allocationFailures = telemetryStore.field('kv.allocation_failures');

    replaceChildren(counters, [
      metricRow('running', running),
      metricRow('waiting', waiting),
      metricRow('admission slots', admissionSlots),
    ]);

    replaceChildren(occupancy, [renderOccupancy(batchSize, maxBatch)]);
    replaceChildren(queue, [renderQueueSummary(telemetryStore, queueDepth)]);

    replaceChildren(footer, [
      metricRow('preemptions', withReason(preemptions, REASONS.NO_PREEMPTION_COUNTER)),
      metricRow('rejections', rejections),
      renderAllocationFailures(allocationFailures),
    ]);

    paintSpark(occupancySpark, telemetryStore.series('batch.active_size', WINDOW_MS), {
      label: 'Batch occupancy',
      unit: 'sequences',
    });
    paintSpark(queueSpark, telemetryStore.series('queue.depth', WINDOW_MS), {
      label: 'Queue depth',
      unit: 'requests',
    });

    description = buildDescription({
      running,
      waiting,
      batchSize,
      maxBatch,
      queueDepth,
      preemptions,
      rejections,
    });
  };

  const scheduler = createRepaintScheduler(rootElement, paint);
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
 * Render batch occupancy as an absolute count plus a percentage that is only
 * shown when the server states its own limit.
 *
 * The wording changes with availability, and that is the point: "6 sequences"
 * makes no claim about capacity, while "6 of 4 max" does. Saying the second
 * when we only know the first is the fabrication.
 *
 * @param {any} batchSize
 * @param {any} maxBatch
 * @returns {HTMLElement}
 */
function renderOccupancy(batchSize, maxBatch) {
  const percentage = ratioField(batchSize, maxBatch, {
    unavailableReason: REASONS.NO_MAX_BATCH,
    label: 'Batch occupancy',
  });

  const row = element('div', { className: 'occupancy' });

  row.append(
    element('span', {
      className: 'occupancy__count',
      children: [renderField(batchSize, { label: 'Sequences in the current batch' })],
    }),
  );

  if (isRenderable(maxBatch)) {
    row.append(element('span', { className: 'occupancy__of', text: 'of' }));
    row.append(renderField(maxBatch, { label: 'Maximum batch size' }));
    row.append(element('span', { className: 'occupancy__max-label', text: 'max' }));
    row.append(renderField(percentage, { label: 'Batch occupancy' }));
    row.append(renderCapacityBar(batchSize, maxBatch));
  } else {
    // No denominator: state what IS known, name what is not, and make it
    // explicit that the visitor is watching a queue rather than a batch.
    row.append(element('span', { className: 'occupancy__of', text: 'sequences' }));
    row.append(renderField(percentage, { label: 'Batch occupancy' }));
    row.append(
      element('p', {
        className: 'occupancy__note',
        text:
          "The count is real; the limit isn't reported, so there is no percentage to show. " +
          'Without a limit you are watching a queue length, not batch occupancy.',
      }),
    );
  }

  return row;
}

/**
 * A capacity bar, drawn only when a real denominator exists.
 *
 * @param {any} batchSize
 * @param {any} maxBatch
 * @returns {HTMLElement}
 */
function renderCapacityBar(batchSize, maxBatch) {
  const used = numericValueOf(batchSize) ?? 0;
  const limit = numericValueOf(maxBatch) ?? 0;
  const filled = limit > 0 ? Math.min(100, (used / limit) * 100) : 0;

  const bar = element('div', {
    className: 'capacity-bar',
    attrs: {
      role: 'meter',
      'aria-valuenow': String(used),
      'aria-valuemin': '0',
      'aria-valuemax': String(limit),
      'aria-label': `${used} of ${limit} batch slots in use`,
    },
  });
  // Slot ticks, so the bar is readable as "3 of 4" without relying on colour or
  // on the visitor estimating a proportion by eye.
  for (let slot = 0; slot < limit && slot < 64; slot += 1) {
    bar.append(
      element('span', {
        className: ['capacity-bar__slot', slot < used ? 'capacity-bar__slot--used' : ''],
      }),
    );
  }
  bar.setAttribute('data-fill', filled.toFixed(0));
  return bar;
}

/**
 * @param {any} telemetryStore
 * @param {any} queueDepth
 * @returns {HTMLElement}
 */
function renderQueueSummary(telemetryStore, queueDepth) {
  const peak = telemetryStore.field('queue.depth_peak');
  return element('div', {
    className: 'queue-summary',
    children: [
      metricRow('now', queueDepth),
      metricRow('peak (60 s)', peak),
    ],
  });
}

/**
 * Allocation failures get an alarm treatment above zero.
 *
 * The KV crate's own doc comment says a run with allocation failures is
 * thrashing. If the server says so, the page should say so too rather than
 * letting a bad number sit quietly in a row of good ones.
 *
 * @param {any} allocationFailures
 * @returns {HTMLElement}
 */
function renderAllocationFailures(allocationFailures) {
  const count = numericValueOf(allocationFailures);
  const row = metricRow('alloc failures', allocationFailures);
  if (count !== null && count > 0) {
    row.classList.add('metric-row--alarm');
    row.append(
      element('span', {
        className: 'metric-row__alarm-note',
        // Not colour alone (AC25): the word "thrashing" carries the meaning.
        text: 'thrashing — KV allocation is failing under this load',
      }),
    );
  }
  return row;
}

/**
 * @param {{root: HTMLElement, canvas: HTMLCanvasElement, setDescription(text: string): void}} slot
 * @param {any} series
 * @param {{label: string, unit: string}} options
 */
function paintSpark(slot, series, options) {
  renderSparkline(slot, series, {
    width: slot.canvas.width || 320,
    height: slot.canvas.height || 34,
    windowMs: WINDOW_MS,
    label: options.label,
    unit: options.unit,
  });
}

/**
 * Attach a fallback reason to a field that arrived without one.
 *
 * An unavailable field with no explanation is the one shape the design language
 * has no answer for, so this guarantees every em-dash has something to say.
 *
 * @param {any} field
 * @param {string} reason
 * @returns {any}
 */
function withReason(field, reason) {
  if (!field || isRenderable(field) || field.reason) {
    return field;
  }
  return { ...field, reason };
}

/**
 * The plain-English state of the panel, for `aria-label` and the table view.
 *
 * @param {Record<string, any>} fields
 * @returns {string}
 */
function buildDescription(fields) {
  const parts = [
    'Scheduling:',
    `${describeFieldText('running', fields.running)},`,
    `${describeFieldText('waiting', fields.waiting)}.`,
  ];

  if (isRenderable(fields.batchSize) && isRenderable(fields.maxBatch)) {
    parts.push(`Batch occupancy ${fields.batchSize.value} of ${fields.maxBatch.value} slots.`);
  } else if (isRenderable(fields.batchSize)) {
    parts.push(
      `${fields.batchSize.value} sequences in the current batch; the server does not report a ` +
        'batch limit, so occupancy as a percentage is not available.',
    );
  } else {
    parts.push('Batch size is not measurable yet.');
  }

  parts.push(`${describeFieldText('Queue depth', fields.queueDepth)}.`);
  parts.push(`${describeFieldText('Rejections', fields.rejections)}.`);
  parts.push(`${describeFieldText('Preemptions', fields.preemptions)}.`);
  return parts.join(' ');
}

export { mount };
