// Copyright (c) Microsoft Corporation.
//
// Accessibility contract — AC28 (view-as-table) and AC30 (reduced motion).
//
// demo-spec.md §Accessibility calls these binding, not advisory, so they are
// tested like any other behaviour. The theme running through both: an
// accessibility preference changes HOW something is presented, never WHETHER
// the measurement is available. A reduced-motion visitor who gets stale numbers,
// or a screen-reader user who gets a chart with no readable alternative, has
// been given less truth rather than less motion.

import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';

import { flushAnimationFrames, installFakeDom } from './testing/fake-dom.js';
import { createFakeStore, measured, series } from './testing/fake-store.js';

let uninstallDom;
before(() => {
  uninstallDom = installFakeDom();
});
after(() => {
  delete globalThis.matchMedia;
  uninstallDom();
});

const { createRepaintScheduler, createSparklineSlot, renderSparkline } = await import(
  './panel-kit.js'
);
const { planSparkline, tabulateSparkline } = await import('./sparkline.js');
const throughput = await import('./throughput.js');
const scheduling = await import('./scheduling.js');

/** @param {boolean} reduce */
function setReducedMotion(reduce) {
  globalThis.matchMedia = (query) => ({
    matches: reduce && query.includes('prefers-reduced-motion'),
    media: query,
  });
}

const samples = (count) =>
  series(Array.from({ length: count }, (_, index) => [index * 250, 10 + index]));

describe('AC28 — every canvas has a table alternative', () => {
  it('builds a table from the same samples the canvas painted', () => {
    const slot = createSparklineSlot({ label: 'Queue depth', width: 200, height: 30 });
    renderSparkline(slot, samples(6), {
      width: 200,
      height: 30,
      windowMs: 60_000,
      label: 'Queue depth',
      unit: 'requests',
      nowMs: 1500,
    });

    const table = slot.root.findByClass('spark-table');
    assert.ok(table, 'no table alternative was rendered');
    const rows = slot.root.findByClass('spark-table__body').children;
    assert.ok(rows.length > 0, 'the table alternative is empty');
    // The newest reading is the one a visitor most wants and the one fixed-step
    // downsampling is most likely to drop.
    assert.match(slot.root.textContent, /15/);
  });

  it('downsamples rather than emitting one row per poll', () => {
    // A table of twelve hundred polls is the same inaccessibility in a
    // different format.
    const plan = planSparkline(samples(400), {
      width: 200,
      height: 30,
      windowMs: 10 ** 9,
      nowMs: 400 * 250,
    });
    const rows = tabulateSparkline(plan, { maxRows: 12 });
    assert.ok(rows.length <= 13, `expected a downsampled table, got ${rows.length} rows`);
    assert.ok(rows.length >= 5, 'downsampled so far that the shape is lost');
  });

  it('explains itself rather than showing an empty grid when there are no samples', () => {
    const plan = planSparkline(
      { state: 'unavailable', t: [], v: [], gaps: [], reason: 'not plumbed' },
      { width: 200, height: 30, windowMs: 60_000, nowMs: 1000 },
    );
    const rows = tabulateSparkline(plan);
    assert.equal(rows.length, 1);
    assert.ok(rows[0].value.length > 0, 'an empty table reads as a rendering failure');
  });

  it('keeps the table in the DOM from the start, not built on toggle', () => {
    // A table that only exists after a click is one only some visitors reach,
    // and the ones who need it most are the least likely to click.
    const slot = createSparklineSlot({ label: 'Batch occupancy', width: 200, height: 30 });
    assert.ok(slot.root.findByClass('spark-table'), 'table absent before any render');

    slot.setView('table');
    assert.equal(slot.root.getAttribute('data-view'), 'table');
    assert.equal(slot.root.findByClass('spark-table').hasAttribute('hidden'), false);

    slot.setView('chart');
    assert.equal(slot.root.findByClass('spark-table').hasAttribute('hidden'), true);
  });

  it('gives every panel canvas an aria-label that is a real sentence', () => {
    setReducedMotion(false);
    const store = createFakeStore({ series: { 'queue.depth': samples(5) } });
    const root = document.createElement('div');
    const handle = scheduling.mount(root, store);
    flushAnimationFrames();

    for (const canvas of collectByTag(root, 'CANVAS')) {
      const label = canvas.getAttribute('aria-label');
      assert.ok(label && label.length > 12, `canvas aria-label is not a sentence: ${label}`);
    }
    handle.unmount();
  });
});

describe('AC30 — reduced motion moves less, never knows less', () => {
  it('throttles the repaint cadence when reduced motion is requested', () => {
    setReducedMotion(true);
    const root = document.createElement('div');
    let paints = 0;
    const scheduler = createRepaintScheduler(root, () => {
      paints += 1;
    });

    scheduler.request();
    flushAnimationFrames();
    assert.equal(paints, 1, 'the first paint must be immediate');

    // Four more poll ticks inside the throttle window.
    for (let index = 0; index < 4; index += 1) {
      scheduler.request();
      flushAnimationFrames();
    }
    assert.equal(paints, 1, 'reduced motion should not repaint on every poll');
    scheduler.cancel();
  });

  it('does not throttle when reduced motion is not requested', () => {
    setReducedMotion(false);
    const root = document.createElement('div');
    let paints = 0;
    const scheduler = createRepaintScheduler(root, () => {
      paints += 1;
    });
    for (let index = 0; index < 3; index += 1) {
      scheduler.request();
      flushAnimationFrames();
    }
    assert.equal(paints, 3);
    scheduler.cancel();
  });

  it('still renders the current measurements under reduced motion', () => {
    // The preference reduces motion, not truth. A panel that withheld numbers
    // here would be a worse accessibility failure than the motion it avoided.
    setReducedMotion(true);
    const store = createFakeStore({
      rates: {
        'metrics.tokens_generated_total': measured(42.5, { source: 'derived', unit: 'tok/s' }),
      },
      requests: [],
    });
    const root = document.createElement('div');
    const handle = throughput.mount(root, store);
    flushAnimationFrames();

    assert.match(root.textContent, /42\.5/, 'reduced motion suppressed the measurement itself');
    handle.unmount();
  });

  it('schedules a trailing repaint so a throttled update is delayed, not dropped', async () => {
    setReducedMotion(true);
    const root = document.createElement('div');
    let paints = 0;
    const scheduler = createRepaintScheduler(root, () => {
      paints += 1;
    }, { minIntervalMs: 20 });

    scheduler.request();
    flushAnimationFrames();
    assert.equal(paints, 1);

    scheduler.request();
    flushAnimationFrames();
    assert.equal(paints, 1, 'second update should be deferred, not painted immediately');

    await new Promise((resolve) => setTimeout(resolve, 40));
    flushAnimationFrames();
    assert.equal(paints, 2, 'the deferred update never landed — it was dropped, not delayed');
    scheduler.cancel();
  });

  it('assumes motion is fine when the preference cannot be read', () => {
    // We cannot claim to know a preference nobody expressed, and defaulting to
    // reduced would quietly degrade every chart in an environment without
    // matchMedia.
    delete globalThis.matchMedia;
    const root = document.createElement('div');
    let paints = 0;
    const scheduler = createRepaintScheduler(root, () => {
      paints += 1;
    });
    scheduler.request();
    flushAnimationFrames();
    scheduler.request();
    flushAnimationFrames();
    assert.equal(paints, 2);
    scheduler.cancel();
  });
});

/**
 * @param {any} node
 * @param {string} tagName
 * @param {any[]} [into]
 */
function collectByTag(node, tagName, into = []) {
  if (node.tagName === tagName) into.push(node);
  for (const child of node.children ?? []) collectByTag(child, tagName, into);
  return into;
}
