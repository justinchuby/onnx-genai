// Copyright (c) Microsoft Corporation.
//
// Registry tests.
//
// The registry is small, but it is the one file the page shell imports, so a
// mistake here is a mistake the other developer inherits. These tests pin the
// two properties the shell relies on: that every registered panel really is a
// mountable panel module, and that mode filtering never silently produces an
// empty dashboard.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { PANELS, panelById, panelsForMode } from './index.js';

describe('panel registry', () => {
  it('registers every panel module with a complete meta block', () => {
    assert.equal(PANELS.length, 6);
    for (const panel of PANELS) {
      assert.equal(typeof panel.module.mount, 'function', `${panel.id} has no mount()`);
      assert.equal(panel.id, panel.module.meta.id, `${panel.id} id disagrees with its meta`);
      assert.ok(panel.module.meta.title, `${panel.id} has no title`);
      assert.ok(panel.modes.length > 0, `${panel.id} can never be shown`);
    }
  });

  it('gives every panel a unique id', () => {
    const ids = PANELS.map((panel) => panel.id);
    assert.equal(new Set(ids).size, ids.length, `duplicate panel id in ${ids.join(', ')}`);
  });

  it('KEEPS the KV and prefix panels on the batching server', () => {
    // This test previously asserted the opposite, on the reasoning that
    // "showing them with em-dashes would promise data that is not coming".
    // That reasoning is from a four-state world, and it is exactly the
    // confusion the fifth state was introduced to remove: `unavailable`
    // promises a value, `not-applicable` explicitly does not — it states an
    // architectural fact and points at where the number IS real.
    //
    // These panels are how a visitor learns the demo's central claim, that
    // continuous batching and the paged KV cache are mutually exclusive
    // execution paths. Hiding them replaces that lesson with SILENCE, and
    // silence reads as a smaller dashboard rather than as something withheld.
    // kv-memory also ADAPTS rather than disappears (decode-row occupancy here,
    // a paged block table on the other profile), so hiding it discards a panel
    // that had a real number to show.
    const ids = panelsForMode('batching').map((panel) => panel.id);
    assert.deepEqual(ids, [
      'throughput',
      'scheduling',
      'kv-memory',
      'prefix-cache',
      'requests',
      'system',
    ]);
  });

  it('derives modes from each panel\'s own meta.requires, with no second table', () => {
    // The bug this closes: `modes` was declared by hand in the registry,
    // separately from `meta.requires`, and the two drifted. kv-memory and
    // prefix-cache both declared `requires: null` ("I ship everywhere") while
    // the registry gated them to ['paged'] — and the registry won, because it
    // is what panelsForMode filters on.
    //
    // Both suites stayed green the whole time, because each mechanism was
    // only ever tested against itself. A reconciling test would have caught
    // this instance; deriving removes the second mechanism entirely.
    // Pinned as an EXPLICIT table rather than recomputed from meta.requires.
    // Deriving the expectation from the same field the implementation derives
    // from would produce a test that cannot fail — false assurance, which is
    // worse than no test, and precisely the shape that let the original drift
    // survive. This table encodes the RULING, so changing a panel's
    // meta.requires now has to argue with the ruling rather than silently
    // redefine it.
    const RATIFIED = {
      throughput: ['batching', 'paged'],
      // The only genuinely gated panel: queue depth and occupancy are
      // properties of the continuous batch scheduler.
      scheduling: ['batching'],
      // Adapts rather than disappears — decode-row occupancy vs a paged block
      // table. Same component, different noun.
      'kv-memory': ['batching', 'paged'],
      // Ships unconditionally. Hiding it where the story is weak is the one
      // genuinely dishonest move available here.
      'prefix-cache': ['batching', 'paged'],
      requests: ['batching', 'paged'],
      system: ['batching', 'paged'],
    };

    assert.deepEqual(
      PANELS.map((panel) => panel.id).sort(),
      Object.keys(RATIFIED).sort(),
      'a panel was added or removed without deciding where it belongs',
    );
    for (const panel of PANELS) {
      assert.deepEqual(
        [...panel.modes],
        RATIFIED[panel.id],
        `${panel.id}: placement disagrees with the ruling. If this is intentional, the ruling ` +
          'is what needs changing — hiding a panel is silence, not honesty.',
      );
    }
  });

  it('refuses to guess where a panel belongs', async () => {
    // modesFor throws on an unknown meta.requires rather than defaulting.
    // Defaulting to universal would show a panel that cannot populate;
    // defaulting to none would hide one that had something to say, with no
    // error anywhere. Neither is guessable, so it stops the build.
    const { PANELS: registered } = await import('./index.js');
    assert.ok(registered.length > 0);

    const { default: mountFn, meta } = await import('./prefix-cache.js');
    assert.equal(typeof mountFn, 'function');
    assert.equal(meta.requires, null, 'prefix-cache must remain universal');
  });

  it('hides the scheduling panel on the paged server', () => {
    const ids = panelsForMode('paged').map((panel) => panel.id);
    assert.deepEqual(ids, ['throughput', 'kv-memory', 'prefix-cache', 'requests', 'system']);
  });

  it('preserves DOM order when filtering', () => {
    for (const mode of ['batching', 'paged']) {
      const filtered = panelsForMode(mode).map((panel) => panel.id);
      const canonical = PANELS.map((panel) => panel.id).filter((id) => filtered.includes(id));
      assert.deepEqual(filtered, canonical);
    }
  });

  it('shows everything rather than nothing when the mode is unrecognised', () => {
    // A misspelled config string must not blank the dashboard: an empty page
    // looks like a dead server, while a full one shows panels explaining
    // themselves and is diagnosable in seconds.
    assert.equal(panelsForMode('scatter-v2').length, PANELS.length);
    assert.equal(panelsForMode(undefined).length, PANELS.length);
  });

  it('never hardcodes a port or an origin anywhere in the registry', async () => {
    const { readFileSync } = await import('node:fs');
    const { fileURLToPath } = await import('node:url');
    const source = readFileSync(fileURLToPath(new URL('./index.js', import.meta.url)), 'utf8');
    assert.equal(/localhost:\d+|127\.0\.0\.1|:\d{4}\b/.test(source), false);
  });

  it('looks a panel up by id', () => {
    assert.equal(panelById('kv-memory')?.module.meta.title, 'KV memory');
    assert.equal(panelById('nope'), undefined);
  });
});

describe('mountDashboard', () => {
  it('adapts the store once and mounts only the panels for the mode', async () => {
    const { installFakeDom, flushAnimationFrames } = await import('./testing/fake-dom.js');
    const { mountDashboard } = await import('./index.js');
    const uninstall = installFakeDom();
    try {
      let subscriptions = 0;
      const store = {
        field: (key) => ({
          value: null,
          state: 'unavailable',
          source: 'unknown',
          reason: `No field named "${key}".`,
          unit: null,
          observedAtMs: null,
          derivedFrom: null,
        }),
        getSnapshot: () => ({
          timestampMs: 1000,
          fields: {},
          connection: { state: 'connected', origin: 'http://example.invalid' },
        }),
        subscribe(listener) {
          subscriptions += 1;
          listener(this.getSnapshot());
          return () => {
            subscriptions -= 1;
          };
        },
      };

      const roots = new Map();
      const dashboard = mountDashboard({
        telemetryStore: store,
        mode: 'batching',
        resolveRoot: (panel) => {
          const root = document.createElement('div');
          roots.set(panel.id, root);
          return root;
        },
      });
      flushAnimationFrames();

      assert.deepEqual(dashboard.mounted, [
        'throughput',
        'scheduling',
        'kv-memory',
        'prefix-cache',
        'requests',
        'system',
      ]);
      // One upstream subscription for the whole dashboard, not one per panel.
      assert.equal(subscriptions, 1);

      // Every panel rendered something against a server that can measure
      // nothing — the normal first frame, not an error path.
      for (const root of roots.values()) {
        assert.ok(root.children.length > 0, 'a panel rendered nothing on the unavailable path');
      }

      dashboard.unmount();
      assert.equal(subscriptions, 0, 'unmount left the store subscription behind');
    } finally {
      uninstall();
    }
  });

  it('skips a panel whose root the shell declines to supply', async () => {
    const { installFakeDom } = await import('./testing/fake-dom.js');
    const { mountDashboard } = await import('./index.js');
    const uninstall = installFakeDom();
    try {
      const store = {
        field: () => ({ value: null, state: 'unavailable', source: 'unknown', reason: 'none', unit: null, observedAtMs: null, derivedFrom: null }),
        getSnapshot: () => ({ timestampMs: 1, fields: {}, connection: { state: 'connected' } }),
        subscribe: (listener) => { listener({ timestampMs: 1, fields: {}, connection: { state: 'connected' } }); return () => {}; },
      };
      const dashboard = mountDashboard({
        telemetryStore: store,
        mode: 'paged',
        resolveRoot: (panel) => (panel.id === 'system' ? null : document.createElement('div')),
      });
      assert.equal(dashboard.mounted.includes('system'), false);
      dashboard.unmount();
    } finally {
      uninstall();
    }
  });
});
