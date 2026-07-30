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

const {
  createRepaintScheduler,
  createRovingGroup,
  createSparklineSlot,
  element,
  renderField,
  renderSparkline,
  replaceChildren,
  rovingItems,
  setPanelView,
} = await import('./panel-kit.js');
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

/**
 * A container holding `count` unavailable values — the shape the KV panel takes
 * on a continuous-batching server, where nothing in the group can be measured.
 *
 * @param {number} count
 */
function groupOfUnavailableValues(count) {
  const container = document.createElement('div');
  replaceChildren(
    container,
    Array.from({ length: count }, (_, index) =>
      element('div', {
        className: 'metric-row',
        children: [
          renderField({
            value: null,
            state: 'unavailable',
            label: `metric ${index}`,
            reason: 'Not exposed over HTTP.',
          }),
        ],
      }),
    ),
  );
  return container;
}

describe('AC29 — composite widgets are one tab stop with a roving cursor', () => {
  it('costs a keyboard user one tab stop, not one per metric', () => {
    const container = groupOfUnavailableValues(12);
    const roving = createRovingGroup(container, { label: 'KV cache' });

    const items = rovingItems(container);
    assert.equal(items.length, 12, 'fixture did not produce twelve roving stops');
    const tabStops = items.filter((item) => item.getAttribute('tabindex') === '0');
    assert.equal(tabStops.length, 1, `expected one tab stop, found ${tabStops.length}`);
    roving.destroy();
  });

  it('moves the cursor with arrow keys and clamps at the ends', () => {
    const container = groupOfUnavailableValues(3);
    const roving = createRovingGroup(container, { label: 'KV cache' });
    const items = rovingItems(container);

    items[0].focus();
    container.dispatchEvent({ type: 'keydown', key: 'ArrowDown' });
    assert.equal(document.activeElement, items[1]);
    assert.equal(items[1].getAttribute('tabindex'), '0');
    assert.equal(items[0].getAttribute('tabindex'), '-1', 'the old stop must stand down');

    container.dispatchEvent({ type: 'keydown', key: 'End' });
    assert.equal(document.activeElement, items[2]);
    // Clamping, not wrapping: wrapping past the end of a list of measurements
    // makes it impossible to tell by feel how long the list is.
    container.dispatchEvent({ type: 'keydown', key: 'ArrowDown' });
    assert.equal(document.activeElement, items[2]);

    container.dispatchEvent({ type: 'keydown', key: 'Home' });
    assert.equal(document.activeElement, items[0]);
    roving.destroy();
  });

  it('leaves keys it does not handle alone', () => {
    const container = groupOfUnavailableValues(3);
    const roving = createRovingGroup(container, { label: 'KV cache' });
    const notPrevented = container.dispatchEvent({ type: 'keydown', key: 'Tab' });
    assert.equal(notPrevented, true, 'Tab must still leave the group');
    roving.destroy();
  });

  it('keeps focus on the same metric across a re-render', () => {
    // Panels replaceChildren on every poll. Without this, a keyboard user
    // reading why a value is unavailable loses focus to <body> within 250ms and
    // can never finish reading the explanation.
    const container = groupOfUnavailableValues(5);
    const roving = createRovingGroup(container, { label: 'KV cache' });
    rovingItems(container)[3].focus();

    const before = document.activeElement;
    replaceChildren(
      container,
      Array.from({ length: 5 }, (_, index) =>
        element('div', {
          className: 'metric-row',
          children: [
            renderField({
              value: null,
              state: 'unavailable',
              label: `metric ${index}`,
              reason: 'Still not exposed.',
            }),
          ],
        }),
      ),
    );

    const after = document.activeElement;
    assert.notEqual(after, null, 'focus was dropped to the document by a poll');
    assert.notEqual(after, before, 'fixture did not actually replace the node');
    assert.equal(after, rovingItems(container)[3], 'focus landed on a different metric');
    assert.equal(after.getAttribute('tabindex'), '0');
    roving.destroy();
  });

  it('falls back to the group when the focused metric disappears entirely', () => {
    const container = groupOfUnavailableValues(4);
    const roving = createRovingGroup(container, { label: 'KV cache' });
    rovingItems(container)[3].focus();

    replaceChildren(container, []);
    assert.equal(document.activeElement, container, 'focus vanished with the metric');
    roving.destroy();
  });

  it('does not advertise an empty group as a tab stop', () => {
    const container = document.createElement('div');
    const roving = createRovingGroup(container, { label: 'KV cache' });
    assert.equal(
      container.hasAttribute('tabindex'),
      false,
      'an empty group announces itself and then offers nowhere to go',
    );
    roving.destroy();
  });

  it('restores the container to inert markup on destroy', () => {
    const container = groupOfUnavailableValues(2);
    const roving = createRovingGroup(container, { label: 'KV cache' });
    roving.destroy();
    assert.equal(container.hasAttribute('tabindex'), false);
    assert.equal(container.hasAttribute('role'), false);
  });
});

describe('the shell can drive the table toggle it owns', () => {
  it('switches every chart in a panel and reports how many', () => {
    // demo-ux.md §3 gives the shell the toggle but hands it only describe(),
    // a sentence. The data lives in the panel, so the shell needs a hook.
    setReducedMotion(false);
    const store = createFakeStore({ series: { 'queue.depth': samples(5) } });
    const root = document.createElement('div');
    const handle = scheduling.mount(root, store);
    flushAnimationFrames();

    const switched = setPanelView(root, 'table');
    assert.ok(switched > 0, 'the panel exposed no charts to switch');

    for (const figure of collectByTag(root, 'FIGURE')) {
      assert.equal(figure.getAttribute('data-view'), 'table');
      const [canvas] = collectByTag(figure, 'CANVAS');
      const [table] = collectByTag(figure, 'TABLE');
      assert.equal(canvas.hasAttribute('hidden'), true, 'canvas still shown in table view');
      assert.equal(table.hasAttribute('hidden'), false, 'table still hidden in table view');
    }

    setPanelView(root, 'chart');
    for (const figure of collectByTag(root, 'FIGURE')) {
      assert.equal(figure.getAttribute('data-view'), 'chart');
      assert.equal(collectByTag(figure, 'CANVAS')[0].hasAttribute('hidden'), false);
      assert.equal(collectByTag(figure, 'TABLE')[0].hasAttribute('hidden'), true);
    }
    handle.unmount();
  });

  it('reports zero for a panel with no charts, so the shell can hide the toggle', () => {
    const root = document.createElement('div');
    assert.equal(setPanelView(root, 'table'), 0);
  });

  it('keeps the table view across a re-render', () => {
    // The toggle is a user decision. A poll arriving 250ms later must not
    // silently throw them back to the chart they chose to leave.
    setReducedMotion(false);
    const store = createFakeStore({ series: { 'queue.depth': samples(5) } });
    const root = document.createElement('div');
    const handle = scheduling.mount(root, store);
    flushAnimationFrames();

    const switched = setPanelView(root, 'table');
    store.tick();
    flushAnimationFrames();

    const figures = collectByTag(root, 'FIGURE');
    assert.equal(figures.length, switched);
    assert.ok(figures.length > 0, 'no figures: this test would pass vacuously');
    for (const figure of figures) {
      assert.equal(figure.getAttribute('data-view'), 'table', 'a poll reverted the chosen view');
      assert.equal(collectByTag(figure, 'TABLE')[0].hasAttribute('hidden'), false);
    }
    handle.unmount();
  });
});
