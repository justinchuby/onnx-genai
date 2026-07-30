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

  it('hides the KV and prefix panels on the batching server', () => {
    // Not a styling choice. ContinuousBatchManager never touches engine.kv_cache,
    // so these panels cannot populate there — showing them with em-dashes would
    // promise data that is not coming.
    const ids = panelsForMode('batching').map((panel) => panel.id);
    assert.deepEqual(ids, ['throughput', 'scheduling', 'requests', 'system']);
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

      assert.deepEqual(dashboard.mounted, ['throughput', 'scheduling', 'requests', 'system']);
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
