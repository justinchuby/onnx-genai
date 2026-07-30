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

import { PANELS, panelById } from './index.js';

describe('panel registry', () => {
  it('registers every panel module with a complete meta block', () => {
    // Six. The prefix-cache panel is the sixth and binds no telemetry at all --
    // it renders the null result rather than a counter, per demo-ux.md D279.
    // Pinned deliberately -- adding or removing a panel should require saying
    // so here, because that is a decision about what the demo claims.
    assert.equal(PANELS.length, 6);
    for (const panel of PANELS) {
      assert.equal(typeof panel.module.mount, 'function', `${panel.id} has no mount()`);
      assert.equal(panel.id, panel.module.meta.id, `${panel.id} id disagrees with its meta`);
      assert.ok(panel.module.meta.title, `${panel.id} has no title`);
      assert.equal('modes' in panel, false, `${panel.id} carries a mode gate again`);
    }
  });

  it('ships the prefix-cache panel, bound to zero telemetry fields', async () => {
    // THIS ASSERTION USED TO RUN THE OTHER WAY, and inverting it is the
    // required move rather than a loosening. It asserted the module must not
    // exist on disk, citing a ruling -- and it said in as many words that
    // re-adding the panel "needs a new ruling, not a merge". That ruling
    // arrived: demo-ux.md D279 ratifies the panel as SHIPPING, bound to zero
    // fields, on the reasoning that deleting it is the one option that is
    // definitely dishonest -- a visitor who sees no prefix panel concludes
    // prefix caching works, because nothing on the page says otherwise.
    //
    // The old test was not wrong when written. It went stale, silently and
    // greenly, the moment the ruling it encoded was overturned, and a green
    // test enforcing a dead requirement is indistinguishable from a green test
    // enforcing a live one. That is why the citation is in the comment.
    //
    // WHAT THE RATCHET GUARDS NOW IS THE THING THE RULING ACTUALLY FORBADE.
    // Every denial in the design record is scoped to a prefix COUNTER FIELD --
    // `prefix_cache.hits`, `_lookups`, `hit_rate` -- and never to the panel.
    // FIELD and PANEL are different nouns. The counter is what cannot come
    // back, so the counter is what this pins.
    const { existsSync, readFileSync } = await import('node:fs');
    const { fileURLToPath } = await import('node:url');
    const modulePath = fileURLToPath(new URL('./prefix-cache.js', import.meta.url));

    assert.equal(existsSync(modulePath), true, 'prefix-cache.js is missing from disk');
    assert.equal(
      PANELS.some((panel) => panel.id === 'prefix-cache'),
      true,
      'the prefix-cache panel is not registered, so no visitor can ever reach the finding',
    );

    // Read the SOURCE, not the module object. A binding added later would be a
    // string in this file long before it were observable through any export.
    const source = readFileSync(modulePath, 'utf8');
    for (const banned of ['prefix_cache.hits', 'prefix_cache.lookups', 'prefix.hit_rate']) {
      assert.equal(
        source.includes(`'${banned}'`),
        false,
        `prefix-cache.js binds ${banned}. The panel ships precisely because it binds nothing.`,
      );
    }
  });

  it('renders the prefix finding without any withdrawn timing figure', async () => {
    // The panel's original argument rested on a latency A/B that was later
    // withdrawn by the engineer who ran it. The conclusion survived because it
    // never rested on the timing -- counter arithmetic and the engine source
    // are both timing-free -- but the prose had to be rebuilt, and prose drifts
    // back. These are the exact figures that were retracted.
    const { readFileSync } = await import('node:fs');
    const { fileURLToPath } = await import('node:url');
    const source = readFileSync(fileURLToPath(new URL('./prefix-cache.js', import.meta.url)), 'utf8');

    for (const figure of ['1341', '1254', '7.0%', '7% slower', '1380 ms', '140 ms']) {
      assert.equal(
        source.includes(figure),
        false,
        `prefix-cache.js quotes the withdrawn figure ${figure}. No prefix timing number ships.`,
      );
    }

    // Positive control: the surviving argument must actually be present, or
    // this test would pass just as happily against an empty file.
    assert.ok(
      source.includes('control request'),
      'the counter-behaviour argument is missing, so the checks above are vacuous',
    );
  });

  it('gives every panel a unique id', () => {
    const ids = PANELS.map((panel) => panel.id);
    assert.equal(new Set(ids).size, ids.length, `duplicate panel id in ${ids.join(', ')}`);
  });

  it('mounts every registered panel, on every server, with no mode gate', () => {
    // This replaces four tests that asserted a filter: "KEEPS the KV panel on
    // the batching server", "hides the scheduling panel on the paged server",
    // "preserves DOM order when filtering", and a RATIFIED placement table.
    //
    // The filter is gone on a ruling AND on measurement. The measurement is the
    // half that was never taken: `/v1/status` on the paged origin returns the
    // SAME served keys and the SAME server-declared absences as the batching
    // origin. `scheduling` was therefore not hidden because it had nothing to
    // show -- it was hidden while the server was measuring its numbers, and the
    // provenance table classified those fields MEASURED on both origins the
    // whole time.
    //
    // Asserted as an explicit list rather than derived from PANELS, so that
    // adding a panel is a decision someone has to write down here rather than a
    // silent membership change.
    assert.deepEqual(
      PANELS.map((panel) => panel.id),
      ['throughput', 'scheduling', 'kv-memory', 'prefix-cache', 'requests', 'system'],
    );
  });

  it('exposes no mode-filtering API at all', async () => {
    // The mechanism is REMOVED, not neutralised. A `panelsForMode` that returns
    // everything is a dormant gate: the next reader restores the filter inside
    // it and every caller silently starts hiding panels again. A missing export
    // fails loudly at import time instead.
    const registry = await import('./index.js');
    for (const gone of ['panelsForMode', 'modesFor']) {
      assert.equal(
        gone in registry,
        false,
        `${gone} is exported again — the mode gate is back. Panels are not hidden by `
          + 'server mode; both servers serve an identical field set.',
      );
    }
  });

  it('ignores a mode argument instead of quietly filtering on it', async () => {
    // The shell still passes `mode:` while app.js catches up. That must be
    // INERT, not partially honoured: a half-removed gate that filters for some
    // values and not others is the divergence this change exists to end.
    const { installFakeDom } = await import('./testing/fake-dom.js');
    const { mountDashboard } = await import('./index.js');
    const uninstall = installFakeDom();
    try {
      // A RAW store, not the panel-facing fake: mountDashboard adapts what it
      // is given, so handing it an already-adapted store tests the wrong seam.
      const rawStore = () => ({
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
          listener(this.getSnapshot());
          return () => {};
        },
      });

      const mountedFor = (mode) =>
        mountDashboard({
          telemetryStore: rawStore(),
          mode,
          resolveRoot: () => document.createElement('div'),
        }).mounted;

      const everything = PANELS.map((panel) => panel.id);
      for (const mode of ['batching', 'paged', 'nonsense', undefined]) {
        assert.deepEqual(
          [...mountedFor(mode)],
          everything,
          `mode ${JSON.stringify(mode)} changed what mounted`,
        );
      }
    } finally {
      uninstall();
    }
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
  it('adapts the store exactly once and mounts every panel', async () => {
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

      // AC29. Every panel is a roving group, and a group without an accessible
      // name announces "group" and nothing else. This read `panel.title` while
      // the registry only carried `{id, module, modes}`, so the label was
      // `undefined` for every panel -- invisible on screen, and audible only to
      // the people the group exists for.
      for (const [id, root] of roots) {
        assert.equal(root.getAttribute('role'), 'group', `${id} is not a group`);
        const label = root.getAttribute('aria-label');
        assert.ok(label, `${id}: roving group has no accessible name`);
        assert.equal(
          label,
          panelById(id).title,
          `${id}: the group is named something other than the panel's own title`,
        );
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
