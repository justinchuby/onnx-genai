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
      // routes/admin.rs OMITS every field it cannot compute -- it sends `None`,
      // so the key is absent from the payload rather than present-and-zero.
      // These two were zeros here until the telemetry landed; sending them as
      // zeros now would test a server that no longer exists.
      // batch_utilization is deliberately present and NON-zero: it is a real
      // computation over in-flight and capacity today, not a stub.
      batch_utilization: 0.5,
      batch_in_flight: 2,
      batch_capacity: 4,
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

describe('the omitted fields never reach the screen as zeros', () => {
  // batch.utilization was in this list and has been REMOVED from it, which is
  // the whole point of the change: it is a genuine computation now, so
  // asserting it is not measured would encode a denial of our own working
  // telemetry -- the exact defect this sweep existed to remove, one layer up.
  // Its positive assertion lives in 'genuinely measured fields' below.
  for (const [moduleName, key] of [
    ['throughput', 'throughput.tokens_per_second'],
    ['kv-memory', 'kv.usage'],
  ]) {
    it(`${moduleName} does not render ${key} as a measurement when the server omits it`, async () => {
      const store = createTelemetryStore({ origin: 'scatter', fetchImpl: respondingServer() });
      await store.pollOnce();

      // The server sends no value at all for this key, so nothing about it is
      // measurable here and the panel must not claim otherwise.
      const field = store.field(key);
      assert.notEqual(field.state, 'measured', `${key} is being presented as a live measurement`);
      assert.notEqual(field.value, 0, `${key} materialised a 0 the server never sent`);
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
    assert.equal(store.field('queue.depth').state, 'measured');
    assert.equal(store.field('sessions.active').value, 2);

    // batch.utilization moved out of the suppressed set when it became a real
    // computation. Asserting it POSITIVELY here is what stops that move from
    // being a silent loss of coverage: a field dropped from a "must not render"
    // list and added to no other list is a field nothing checks at all.
    assert.equal(store.field('batch.utilization').state, 'measured');
    assert.equal(store.field('batch.utilization').value, 0.5);

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

