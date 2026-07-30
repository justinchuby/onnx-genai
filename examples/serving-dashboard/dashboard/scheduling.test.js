// Copyright (c) Microsoft Corporation.
//
// Tests for the scheduling panel.
//
// The occupancy-denominator cases are the reason this file exists. Everything
// else here is ordinary panel behaviour; those tests guard the subtlest AC6
// violation available on this page — dividing a real numerator by a denominator
// nobody reported.

import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';

import { flushAnimationFrames, installFakeDom } from './testing/fake-dom.js';
import { createFakeStore, measured, series, unavailable } from './testing/fake-store.js';

let uninstallDom;
before(() => {
  uninstallDom = installFakeDom();
});
after(() => uninstallDom());

const { meta, mount } = await import('./scheduling.js');

/** @param {object} [overrides] */
function storeWith(overrides = {}) {
  const now = Date.now();
  return createFakeStore({
    fields: {
      'scheduler.running': measured(6, { unit: 'requests', label: 'Running' }),
      'scheduler.waiting': measured(2, { unit: 'requests', label: 'Waiting' }),
      'admission.slots_available': measured(248, { unit: 'slots' }),
      'batch.active_size': measured(6, { unit: 'sequences' }),
      'scheduler.max_batch': unavailable(
        "Occupancy needs the server's max batch size, which isn't surfaced.",
      ),
      'queue.depth': measured(2, { unit: 'requests' }),
      'queue.depth_peak': measured(8, { unit: 'requests' }),
      'scheduler.preemptions_total': unavailable(
        'The scheduler performs preemption but keeps no counter for it.',
      ),
      'admission.rejections': measured(0, { unit: 'count' }),
      'kv.allocation_failures': measured(0, { unit: 'count' }),
      ...overrides.fields,
    },
    series: {
      'batch.active_size': series([
        [now - 2000, 4],
        [now - 1000, 6],
      ]),
      'queue.depth': series([
        [now - 2000, 1],
        [now - 1000, 2],
      ]),
      ...overrides.series,
    },
  });
}

/** Mount into a fresh root and flush the first paint. */
function mountPanel(store) {
  const root = document.createElement('div');
  const handle = mount(root, store);
  flushAnimationFrames();
  return { root, handle };
}

describe('scheduling panel — the batch occupancy denominator', () => {
  it('refuses to show a percentage when the server does not report its batch limit', () => {
    const { root, handle } = mountPanel(storeWith());

    const occupancy = root.findByClass('occupancy');
    assert.ok(occupancy, 'the occupancy block must render');
    assert.match(
      occupancy.textContent,
      /—/,
      'the percentage must be an em-dash, never a number derived from an assumed limit',
    );
    assert.doesNotMatch(
      occupancy.textContent,
      /of 4 max/,
      'DEFAULT_MAX_BATCH = 4 from state.rs:25 must never be substituted for a reported limit',
    );
    handle.unmount();
  });

  it('tells the visitor they are watching a queue, not a batch, when the limit is unknown', () => {
    // This is the lesson of the scenario. Without a limit, the number moving on
    // screen is a queue length; calling it occupancy would teach the wrong idea.
    const { root, handle } = mountPanel(storeWith());

    assert.match(root.textContent, /watching a queue length, not batch occupancy/);
    handle.unmount();
  });

  it('still shows the real numerator, because the count itself is genuinely measured', () => {
    const { root, handle } = mountPanel(storeWith());

    assert.match(root.findByClass('occupancy__count').textContent, /6/);
    handle.unmount();
  });

  it('shows the percentage and a slot bar once the server reports --max-batch', () => {
    const { root, handle } = mountPanel(
      storeWith({
        fields: {
          'batch.active_size': measured(3, { unit: 'sequences' }),
          'scheduler.max_batch': measured(4, { unit: 'sequences' }),
        },
      }),
    );

    const occupancy = root.findByClass('occupancy');
    assert.match(occupancy.textContent, /75/, 'a real denominator yields a real percentage');

    const bar = root.findByClass('capacity-bar');
    assert.equal(bar.getAttribute('aria-valuemax'), '4');
    assert.equal(bar.getAttribute('aria-valuenow'), '3');
    assert.equal(
      bar.children.filter((slot) => slot.classList.contains('capacity-bar__slot--used')).length,
      3,
      'discrete slot ticks make "3 of 4" readable without estimating a proportion by eye',
    );
    handle.unmount();
  });
});

describe('scheduling panel — a real zero next to an honest absence', () => {
  it('renders rejections 0 as a number and preemptions as an em-dash, in the same row group', () => {
    // demo-ux.md §5.3 calls this the clearest teaching example of the
    // unavailable-data language anywhere in the UI. It must not regress.
    const { root, handle } = mountPanel(storeWith());

    const footer = root.findByClass('panel-scheduling__footer');
    const rows = footer.children;
    const rejectionsRow = rows.find((row) => row.textContent.includes('rejections'));
    const preemptionsRow = rows.find((row) => row.textContent.includes('preemptions'));

    assert.equal(rejectionsRow.findByClass('value').getAttribute('data-state'), 'ok');
    assert.match(rejectionsRow.findByClass('value__num').textContent, /^0$/);
    assert.equal(preemptionsRow.findByClass('value').getAttribute('data-state'), 'unavailable');
    assert.equal(preemptionsRow.findByClass('value__num--unavailable').textContent, '—');
    handle.unmount();
  });

  it('supplies a reason for preemptions even if the store forgot to attach one', () => {
    const { root, handle } = mountPanel(
      storeWith({
        fields: {
          'scheduler.preemptions_total': { value: null, state: 'unavailable', source: 'server' },
        },
      }),
    );

    const footer = root.findByClass('panel-scheduling__footer');
    const row = footer.children.find((child) => child.textContent.includes('preemptions'));
    assert.match(
      row.findByClass('value').getAttribute('title'),
      /keeps no counter for it/,
      'every em-dash must have something to say when a visitor reaches for it',
    );
    handle.unmount();
  });

  it('escalates allocation failures with a word, not only a colour (AC25)', () => {
    const { root, handle } = mountPanel(
      storeWith({ fields: { 'kv.allocation_failures': measured(7, { unit: 'count' }) } }),
    );

    assert.match(root.textContent, /thrashing/);
    handle.unmount();
  });

  it('does not escalate a genuine zero allocation-failure count', () => {
    const { root, handle } = mountPanel(storeWith());

    assert.doesNotMatch(root.textContent, /thrashing/);
    handle.unmount();
  });
});

describe('scheduling panel — lifecycle', () => {
  it('paints only on store ticks, never on a timer of its own', () => {
    const store = storeWith();
    const { handle } = mountPanel(store);

    assert.equal(store.subscriberCount(), 1, 'exactly one subscription, and no setInterval');
    handle.unmount();
  });

  it('destroy() unsubscribes and empties the root, so a remount cannot double-render', () => {
    const store = storeWith();
    const { root, handle } = mountPanel(store);

    handle.unmount();

    assert.equal(store.subscriberCount(), 0, 'AC22: a leaked subscription is a memory leak');
    assert.equal(root.children.length, 0);
  });

  it('re-renders when the store ticks with new values', () => {
    const store = storeWith();
    const { root, handle } = mountPanel(store);

    store.setField('queue.depth', measured(9, { unit: 'requests' }));
    store.tick();
    flushAnimationFrames();

    assert.match(root.findByClass('queue-summary').textContent, /9/);
    handle.unmount();
  });

  it('renders em-dashes rather than crashing when the store knows nothing at all', () => {
    // The real store contract says field() is total. This proves the panel does
    // not quietly depend on that being true.
    const { root, handle } = mountPanel(createFakeStore());

    assert.match(root.textContent, /—/);
    assert.doesNotMatch(root.textContent, /undefined|NaN|null/);
    handle.unmount();
  });
});

describe('scheduling panel — describe() for AC28', () => {
  it('states the occupancy caveat in plain English for a screen-reader user', () => {
    const { handle } = mountPanel(storeWith());

    const text = handle.describe();
    assert.match(text, /does not report a batch limit/);
    assert.match(text, /Preemptions is not measurable yet/);
    handle.unmount();
  });

  it('states occupancy as a fraction once the limit is known', () => {
    const { handle } = mountPanel(
      storeWith({
        fields: {
          'batch.active_size': measured(3),
          'scheduler.max_batch': measured(4),
        },
      }),
    );

    assert.match(handle.describe(), /Batch occupancy 3 of 4 slots/);
    handle.unmount();
  });
});

describe('scheduling panel — metadata', () => {
  it('declares acronym definitions the shell can surface (AC30)', () => {
    assert.equal(meta.id, 'scheduling');
    assert.match(meta.acronyms.occupancy, /batch capacity/);
    assert.match(meta.acronyms.preemption, /Evicting/);
  });
});
