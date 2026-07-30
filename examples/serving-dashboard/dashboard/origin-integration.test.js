// Copyright (c) Microsoft Corporation.
//
// End-to-end honesty against the REAL telemetry store, not a fake.
//
// Every other suite here mounts panels against a hand-written fake, which can
// only prove the panels are self-consistent. These tests drive the real store
// through a real poll, so they also prove the provenance table, the store's
// classification handling and the panels agree about what a given server can
// actually measure. That seam is where a fabricated number would appear, and it
// is invisible to both sides individually.

import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';

import { flushAnimationFrames, installFakeDom } from './testing/fake-dom.js';

let uninstallDom;
before(() => {
  uninstallDom = installFakeDom();
});
after(() => uninstallDom());

const { createTelemetryStore } = await import('../telemetry-store.js');
const { adaptStore } = await import('./store-adapter.js');

/**
 * A server that answers everything successfully, reporting the values the real
 * scatter server reports today: hardcoded 0.0 placeholders alongside genuine
 * counters. Telling those apart is the whole point.
 */
function respondingServer(overrides = {}) {
  const bodies = {
    '/health': { status: 'ok' },
    '/v1/models': { data: [{ id: 'phi-3' }] },
    '/v1/status': {
      healthy: true,
      model_id: 'phi-3',
      node_id: 'node-1',
      context_length: 4096,
      queue_depth: 3,
      active_sessions: 2,
      batch: { active_size: 4 },
      // The three fabricated fields, exactly as routes/admin.rs emits them.
      kv_usage: 0.0,
      tokens_per_second: 0.0,
      batch_utilization: 0.0,
    },
    '/v1/debug/kv': { prefix_cache_hits: 0, prefix_cache_lookups: 17 },
    '/v1/debug/config': {},
    '/v1/resources': {},
    '/metrics': '',
    ...overrides,
  };
  return async (url) => {
    const path = Object.keys(bodies).find((candidate) => url.endsWith(candidate));
    const body = bodies[path];
    return {
      ok: body !== undefined,
      status: body === undefined ? 404 : 200,
      json: async () => body,
      text: async () => (typeof body === 'string' ? body : JSON.stringify(body)),
    };
  };
}

/**
 * @param {string} origin
 * @param {string} moduleName
 */
async function mountAgainstRealStore(origin, moduleName) {
  const store = createTelemetryStore({ origin, fetchImpl: respondingServer() });
  await store.pollOnce();
  const adapter = adaptStore(store);
  const panel = await import(`./${moduleName}.js`);
  const root = document.createElement('div');
  const handle = panel.mount(root, adapter);
  flushAnimationFrames();
  return {
    root,
    text: root.textContent,
    release() {
      handle.unmount();
      adapter.destroy();
      store.stop();
    },
  };
}

describe('a structurally bypassed counter never renders as a measurement', () => {
  it('shows n/a, not 0%, for prefix cache on the batching server', async () => {
    // prefix_cache_hit_len is a hardcoded 0 on this decode path
    // (batched.rs:262/:486), and the engine tests assert it. A "0%" here would
    // describe a cache that tried and missed. It never tried.
    const panel = await mountAgainstRealStore('scatter', 'prefix-cache');

    assert.doesNotMatch(panel.text, /\b0(\.0)?\s*%/, 'rendered a hit rate the server never measured');
    assert.match(panel.text, /n\/a/i, 'expected the not-applicable treatment');
    panel.release();
  });

  it('never says "not measurable yet" about something structurally impossible', async () => {
    // That wording promises a value that will arrive later. On this origin it
    // cannot arrive at all, so the promise is false — including for the
    // DERIVED hit rate, whose inputs are both bypassed.
    const panel = await mountAgainstRealStore('scatter', 'prefix-cache');
    const hero = panel.root.findByClass('panel-prefix-cache__hero');
    assert.doesNotMatch(
      hero.textContent,
      /not measurable yet/i,
      'the derived hit rate softened its inputs into a promise',
    );
    panel.release();
  });

  it('does not call the counter "lookups", because it counts generations', async () => {
    // metrics.rs:130-132 increments per completed generation regardless of any
    // cache activity. The upstream name is simply wrong.
    const panel = await mountAgainstRealStore('scatter', 'prefix-cache');
    assert.doesNotMatch(panel.text, /lookups/i);
    panel.release();
  });

  it('reports the same counter as a real measurement on the paging server', async () => {
    // The mirror image, and the reason this cannot be a blanket suppression:
    // on the dynamic server the cache IS consulted, so 0 is genuine data and
    // must render as a stark zero rather than being hidden.
    const store = createTelemetryStore({ origin: 'dynamic', fetchImpl: respondingServer() });
    await store.pollOnce();
    assert.equal(store.field('prefix_cache.hits').state, 'ok');
    assert.equal(store.field('prefix_cache.hits').value, 0);
    store.stop();
  });
});

describe('the fabricated zeros never reach the screen', () => {
  for (const [moduleName, key] of [
    ['throughput', 'throughput.tokens_per_second'],
    ['scheduling', 'batch.utilization'],
    ['kv-memory', 'kv.usage'],
  ]) {
    it(`${moduleName} does not render ${key}'s hardcoded 0.0 as a measurement`, async () => {
      const store = createTelemetryStore({ origin: 'scatter', fetchImpl: respondingServer() });
      await store.pollOnce();

      // The server really did send 0.0 on the wire for this key...
      const field = store.field(key);
      assert.notEqual(field.state, 'ok', `${key} is being presented as a live measurement`);
      assert.notEqual(field.state, 'measured', `${key} is being presented as a live measurement`);
      store.stop();
    });
  }

  it('renders an em-dash rather than a zero for an unavailable field', async () => {
    // The check the lead asked for, run against the real store: a field the
    // server cannot measure must show the em-dash treatment, never a 0.
    const panel = await mountAgainstRealStore('scatter', 'kv-memory');
    const unavailable = panel.root.findByClass('value__num--unavailable');
    assert.ok(unavailable, 'no unavailable treatment found on a panel full of unmeasurable fields');
    assert.equal(unavailable.textContent, '—');
    assert.doesNotMatch(
      panel.text,
      /\b0(\.0)?\s*%\s*utilization/i,
      'KV utilization rendered as a zero percentage',
    );
    panel.release();
  });
});

describe('genuinely measured fields still render as numbers', () => {
  it('shows queue depth and active sessions, so honesty is not just suppression', async () => {
    // A dashboard that renders nothing is trivially honest and useless. These
    // are MEASURED today and must appear.
    const store = createTelemetryStore({ origin: 'scatter', fetchImpl: respondingServer() });
    await store.pollOnce();

    assert.equal(store.field('queue.depth').value, 3);
    assert.equal(store.field('queue.depth').state, 'ok');
    assert.equal(store.field('sessions.active').value, 2);

    const adapter = adaptStore(store);
    const scheduling = await import('./scheduling.js');
    const root = document.createElement('div');
    const handle = scheduling.mount(root, adapter);
    flushAnimationFrames();

    assert.match(root.textContent, /3/, 'a genuinely measured queue depth did not reach the screen');
    handle.unmount();
    adapter.destroy();
    store.stop();
  });
});

describe('a structurally absent metric is explained, not apologised for', () => {
  // `unavailable` and `not-applicable` were rendering identically on this
  // panel: the headline hit rate said "—" with a NOT MEASURABLE YET chart while
  // its own supporting rows correctly said n/a. Same panel, two voices, and the
  // apologetic one was wrong — nothing is missing on the scatter server, the
  // cache is simply never consulted. A first-time visitor reading "not
  // measurable yet" concludes the demo is half broken, which is exactly the
  // misreading the fifth state exists to prevent.

  it('says not-applicable, never "not measurable yet", on the batching server', async () => {
    const panel = await mountAgainstRealStore('scatter', 'prefix-cache');

    assert.match(panel.text, /n\/a/i);
    assert.doesNotMatch(
      panel.text,
      /NOT MEASURABLE YET/i,
      'apologised for a metric that is absent by design, not by omission',
    );
    assert.match(panel.text, /NOT APPLICABLE HERE/i, 'the chart well lost its explanation');
    panel.release();
  });

  it('explains WHY in the accessible description, not only on hover', async () => {
    // For not-applicable the explanation IS the information. A tooltip-only
    // reason is unreachable by keyboard, by touch and by screen reader.
    const store = createTelemetryStore({ origin: 'scatter', fetchImpl: respondingServer() });
    await store.pollOnce();
    const adapter = adaptStore(store);
    const panel = await import('./prefix-cache.js');
    const root = document.createElement('div');
    const handle = panel.mount(root, adapter);
    flushAnimationFrames();

    const description = handle.describe();
    assert.match(description, /not applicable|never consulted/i);
    assert.doesNotMatch(description, /not measurable yet/i);

    handle.unmount();
    adapter.destroy();
    store.stop();
  });

  it('gives the paging server a different explanation, not the scatter one', async () => {
    // The opposite treatment of the same field. On scatter the capability is
    // structurally absent; here it exists but its counters were disproved. Two
    // distinct facts that must not collapse into one message — and neither is
    // an apology.
    const panel = await mountAgainstRealStore('dynamic', 'prefix-cache');

    assert.doesNotMatch(panel.text, /0\s*%/, 'a rate from broken counters must not be printed');
    assert.doesNotMatch(panel.text, /NOT APPLICABLE/i, 'the capability is present on this path');
    assert.match(
      panel.text,
      /denominator|share no prefix|measure prefix reuse/i,
      'the panel must say why the rate is withheld',
    );
    panel.release();
  });

  it('reports the capability as structurally absent rather than unplumbed', async () => {
    const store = createTelemetryStore({ origin: 'scatter', fetchImpl: respondingServer() });
    await store.pollOnce();
    const adapter = adaptStore(store);

    const capability = adapter.capability('prefix-cache');
    assert.equal(capability.available, false);
    assert.equal(capability.state, 'not-applicable');
    assert.ok(capability.reason, 'a not-applicable capability must carry its explanation');

    adapter.destroy();
    store.stop();
  });
});
