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
const RULED_STATES = Object.freeze(['measured', 'pending', 'stale', 'unavailable', 'not-applicable']);

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

describe('the field shape the panels actually read', () => {
  // A proposed shape is in circulation that renames `source` to `provenance`
  // and `endpoint`, and adds `classification`. None of it has landed. This
  // matters because a rename here fails SILENTLY in the worst way: reading a
  // property that no longer exists yields undefined, so the source badge and
  // the origin attribution simply stop rendering. No error, no crash — the
  // provenance affordances just quietly disappear, which is the one category
  // of bug this dashboard exists to prevent.
  //
  // If the rename does land, this test fails and names the property instead of
  // letting six panels lose their badges unnoticed.
  it('exposes source, origin and observedAtMs on a real polled field', async () => {
    const store = createTelemetryStore({ origin: 'scatter', fetchImpl: respondingServer() });
    await store.pollOnce();

    const field = store.field('queue.depth');
    for (const key of ['value', 'state', 'source', 'origin', 'observedAtMs', 'reason']) {
      assert.ok(key in field, `TelemetryField lost the "${key}" property the panels read`);
    }

    store.stop();
  });

  it('attributes a field to an engine, not to a URL', async () => {
    // `origin` answers WHICH ENGINE, and it is carried on client-measured
    // fields too. It must never be inferred from the base URL that was
    // fetched, or the derived metrics end up unattributed.
    const store = createTelemetryStore({ origin: 'dynamic', fetchImpl: respondingServer() });
    await store.pollOnce();

    assert.equal(store.field('queue.depth').origin, 'dynamic');

    store.stop();
  });
});

describe('no two of the five states render identically', () => {
  // The lead's standing invitation: if any two would look the same, say so and
  // they get collapsed. This answers it with evidence rather than opinion, and
  // it answers it in TEXT — the comparison is on textContent, so a pair that is
  // distinguished only by colour counts as identical here. That is the same bar
  // as the grayscale-screenshot test, applied automatically on every run.
  const RENDERABLE = { value: 41, unit: 'requests', label: 'Queue depth', source: 'server' };

  it('produces five distinct renderings without relying on colour', async () => {
    const { installFakeDom } = await import('./testing/fake-dom.js');
    const uninstall = installFakeDom();
    const { renderField } = await import('./panel-kit.js');
    const now = Date.now();

    const rendered = new Map();
    for (const state of RULED_STATES) {
      const field = {
        ...RENDERABLE,
        state,
        reason: 'Prefix cache is never consulted on this execution path.',
        observedAtMs: state === 'stale' ? now - 12_000 : now,
      };
      rendered.set(state, renderField(field, { staleCeilingMs: 30_000 }).textContent);
    }

    const seen = new Map();
    for (const [state, text] of rendered) {
      const clash = seen.get(text);
      assert.equal(
        clash,
        undefined,
        `"${state}" and "${clash}" both render as ${JSON.stringify(text)} — they are ` +
          'indistinguishable without colour, so either the treatments must differ or the ' +
          'two states should be collapsed.',
      );
      seen.set(text, state);
    }

    uninstall();
  });

  it('never renders a zero for a state that carries no value', async () => {
    const { installFakeDom } = await import('./testing/fake-dom.js');
    const uninstall = installFakeDom();
    const { renderField } = await import('./panel-kit.js');

    for (const state of ['pending', 'unavailable', 'not-applicable']) {
      const text = renderField({ ...RENDERABLE, value: null, state, reason: 'n' }).textContent;
      assert.doesNotMatch(text, /0/, `"${state}" rendered a zero it never measured`);
    }

    uninstall();
  });
});
