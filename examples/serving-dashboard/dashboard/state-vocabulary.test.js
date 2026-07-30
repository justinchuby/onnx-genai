// Copyright (c) Microsoft Corporation.
//
// The safety net for deleting the state bridge.
//
// field-state.js used to translate 'measured' -> 'ok'. The vocabulary is now
// ruled and final, so that bridge is gone and anything unrecognised resolves to
// `unavailable` instead. That is the right failure direction — it admits
// ignorance rather than asserting false confidence — but it has a cost the
// bridge was hiding: if a producer ever drifts back to an unratified spelling,
// a REAL MEASUREMENT renders as an em-dash and nobody is told. That is the same
// failure the lead called out when retiring the client-side classification
// table, just pointed the other way.
//
// So the bridge is replaced with a loud check rather than simply removed. These
// tests drive the real store through real polls and assert that every state it
// can actually produce is one of the five. A drift fails the build with an
// explanation instead of quietly blanking a panel.

import assert from 'node:assert/strict';

import { describe, it } from 'node:test';

import { FIELD_STATES } from '../telemetry-field.js';
import { createTelemetryStore } from '../telemetry-store.js';
import { RENDER_STATES, renderStateOf } from './field-state.js';

/** The five ruled states. Changing this list is a project-level decision. */
const RULED_STATES = Object.freeze(['ok', 'pending', 'stale', 'unavailable', 'not-applicable']);

function statesIn(snapshot) {
  return Object.entries(snapshot.fields).map(([key, field]) => [key, field?.state]);
}

function assertRuled(pairs, context) {
  for (const [key, state] of pairs) {
    assert.ok(
      RULED_STATES.includes(state),
      `${context}: field "${key}" has state "${state}", which is not one of the five ruled ` +
        `states (${RULED_STATES.join(', ')}). field-state.js no longer bridges spellings, so ` +
        'this field now renders as an em-dash even if the server measured it correctly. ' +
        'Either fix the producer or take the vocabulary change through the lead.',
    );
  }
}

/** A server answering everything, including the three hardcoded 0.0 placeholders. */
function respondingServer() {
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
      kv_usage: 0.0,
      tokens_per_second: 0.0,
      batch_utilization: 0.0,
    },
    '/v1/debug/kv': { prefix_cache_hits: 0, prefix_cache_lookups: 17 },
    '/v1/debug/config': {},
    '/v1/resources': {},
    '/metrics': '',
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

describe('every state the real store can emit is one of the five ruled states', () => {
  for (const origin of ['scatter', 'dynamic']) {
    it(`holds for a healthy ${origin} server`, async () => {
      const store = createTelemetryStore({ origin, fetchImpl: respondingServer() });
      await store.pollOnce();

      const pairs = statesIn(store.getSnapshot());
      assert.ok(pairs.length > 0, 'a poll produced no fields at all');
      assertRuled(pairs, `healthy ${origin}`);

      store.stop();
    });
  }

  it('holds before the first poll, when everything is pending', () => {
    const store = createTelemetryStore({ origin: 'scatter', fetchImpl: respondingServer() });

    assertRuled(statesIn(store.getSnapshot()), 'pre-poll');

    store.stop();
  });

  it('holds when the server is unreachable', async () => {
    const store = createTelemetryStore({
      origin: 'scatter',
      fetchImpl: async () => {
        throw new Error('ECONNREFUSED');
      },
    });
    await store.pollOnce();

    // This is the path that produces the whole-origin stall, so it is the most
    // likely place for an ad-hoc state string to be invented under pressure.
    assertRuled(statesIn(store.getSnapshot()), 'unreachable');

    store.stop();
  });

  it('holds after a good poll is followed by a failed one, which is the stale path', async () => {
    let healthy = true;
    const good = respondingServer();
    const store = createTelemetryStore({
      origin: 'scatter',
      fetchImpl: async (url) => {
        if (!healthy) throw new Error('ECONNREFUSED');
        return good(url);
      },
    });

    await store.pollOnce();
    healthy = false;
    await store.pollOnce();

    assertRuled(statesIn(store.getSnapshot()), 'stale after dropped poll');

    store.stop();
  });
});

describe('the exported state constants agree with the ruling', () => {
  it('exposes no state value outside the five', () => {
    for (const [name, value] of Object.entries(FIELD_STATES)) {
      assert.ok(
        RULED_STATES.includes(value),
        `FIELD_STATES.${name} is "${value}", which is not a ruled state`,
      );
    }
  });

  it('still exports a constant for every ruled state, so panels need no literals', () => {
    const exported = new Set(Object.values(FIELD_STATES));
    for (const state of RULED_STATES) {
      assert.ok(exported.has(state), `no FIELD_STATES constant carries the ruled state "${state}"`);
    }
  });

  it('maps every ruled state to a distinct render state', () => {
    // A collision here would make two states look identical on screen, which is
    // precisely the complaint that added `not-applicable`: an intentional gap
    // that renders like breakage reads as breakage.
    const rendered = RULED_STATES.map((state) => renderStateOf({ state, value: 1 }));

    assert.equal(new Set(rendered).size, RULED_STATES.length, `states collapsed: ${rendered}`);
    assert.deepEqual(
      new Set(rendered),
      new Set(Object.values(RENDER_STATES)),
      'render states and ruled states have drifted apart',
    );
  });
});
