// Copyright (c) Microsoft Corporation.
//
// A CENSUS over every snapshot the telemetry store can build.
//
// WHY THIS FILE EXISTS, and why it is a census rather than more behavioural
// tests: `catalogueMeta()` was introduced to stop the pre-response frames from
// hand-rolling `{source, unit}` and dropping `label`. It was applied to three
// call sites. A FOURTH -- `allUnavailable()`, which builds the entire NO_MODEL
// frame -- kept its hand-rolled object and shipped 35 of 35 fields with no
// label at all, plus one DERIVED field defaulted to sourceClass SERVER.
//
// Every behavioural test in the suite passed the whole time, because each one
// exercises a frame that was already correct. Nothing asserted that the frames
// are the SAME on this axis. That is the shape of the bug:
//
//   a fix that lands is not a fix that travelled, and only a census
//   distinguishes the two.
//
// So this file does not test a frame. It enumerates them, and any new frame
// added to the store must be added here or the coverage floor below fails.

import test from 'node:test';
import assert from 'node:assert/strict';

import { createTelemetryStore, CONNECTION_STATES } from './telemetry-store.js';
import { ENDPOINTS, allFieldKeys, PROVENANCE, resolveForOrigin } from './telemetry-provenance.js';

const BASE_URL = 'http://127.0.0.1:8123';

/** The caption `renderField` invents when a field carries no label. */
const FALLBACK_CAPTION = 'value';

function jsonResponse(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    async json() {
      return body;
    },
    async text() {
      return JSON.stringify(body);
    },
  };
}

function textResponse(status, text) {
  return {
    ok: status >= 200 && status < 300,
    status,
    async json() {
      throw new SyntaxError('not JSON');
    },
    async text() {
      return text;
    },
  };
}

function fakeFetch(routes) {
  return async (url) => {
    const route = routes[url.replace(BASE_URL, '')];
    if (route === undefined) {
      return jsonResponse(404, { error: { message: 'not found', type: 'server_error' } });
    }
    if (route instanceof Error) {
      throw route;
    }
    if (typeof route.text === 'string') {
      return textResponse(route.status ?? 200, route.text);
    }
    return jsonResponse(route.status ?? 200, route.body);
  };
}

const METRICS_BODY = [
  '# TYPE onnx_genai_tokens_generated_total counter',
  'onnx_genai_tokens_generated_total 5048',
].join('\n');

function healthyRoutes(overrides = {}) {
  return {
    [ENDPOINTS.HEALTH]: { body: { status: 'ok', model: 'qwen-scatter' } },
    [ENDPOINTS.STATUS]: {
      body: {
        node_id: 'node-0',
        healthy: true,
        queue_depth: 3,
        active_sessions: 2,
        batch_utilization: 0.5,
        batch_in_flight: 2,
        sessions: [],
      },
    },
    [ENDPOINTS.DEBUG_KV]: { body: { prefix_cache_hits: 0, prefix_cache_lookups: 0 } },
    [ENDPOINTS.DEBUG_CONFIG]: {
      body: {
        model_id: 'qwen-scatter',
        pipeline: false,
        max_output_tokens: 512,
        max_sessions: 256,
        max_queue_depth: 64,
        model_max_context: 32768,
      },
    },
    [ENDPOINTS.METRICS]: { text: METRICS_BODY },
    [ENDPOINTS.RESOURCES]: {
      body: {
        derived_kv_budget: { bytes: 5746050801 },
        vram: { used: 0, limit: 5746050801, headroom: 5746050801 },
      },
    },
    ...overrides,
  };
}

function unreachableRoutes() {
  const routes = {};
  for (const path of Object.values(ENDPOINTS)) {
    routes[path] = new TypeError('Failed to fetch');
  }
  return routes;
}

/**
 * Every distinct snapshot the store can produce, each built through the real
 * polling path rather than by constructing fields directly -- a hand-built
 * field would test this file's idea of a frame, not the store's.
 *
 * @returns {Promise<Array<{name: string, expectedState: string, snapshot: object}>>}
 */
async function everyFrame() {
  const frames = [];

  // 1. The first frame. No server has answered yet; it is the frame a visitor
  //    always sees, and for never-measurable fields it is the only one.
  const fresh = createTelemetryStore({ baseUrl: BASE_URL, fetchImpl: fakeFetch(healthyRoutes()) });
  frames.push({
    name: 'first frame (pre-poll)',
    expectedState: CONNECTION_STATES.CONNECTING,
    snapshot: fresh.getSnapshot(),
  });

  // 2. The healthy frame, which routes through the `fieldMeta` closure.
  const healthy = createTelemetryStore({ baseUrl: BASE_URL, fetchImpl: fakeFetch(healthyRoutes()) });
  frames.push({
    name: 'connected frame',
    expectedState: CONNECTION_STATES.CONNECTED,
    snapshot: await healthy.pollOnce(),
  });

  // 3. Reachable, but the server has no model loaded. THE REGRESSION FRAME.
  const noModel = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: fakeFetch(
      healthyRoutes({
        [ENDPOINTS.STATUS]: {
          body: { node_id: 'node-0', healthy: false, queue_depth: 0, active_sessions: 0, sessions: [] },
        },
        [ENDPOINTS.DEBUG_CONFIG]: {
          status: 500,
          body: { error: { message: 'no model loaded', type: 'server_error' } },
        },
      }),
    ),
  });
  frames.push({
    name: 'no-model frame',
    expectedState: CONNECTION_STATES.NO_MODEL,
    snapshot: await noModel.pollOnce(),
  });

  // 4. Never reached the server at all.
  const dead = createTelemetryStore({ baseUrl: BASE_URL, fetchImpl: fakeFetch(unreachableRoutes()) });
  frames.push({
    name: 'unreachable frame (never connected)',
    expectedState: CONNECTION_STATES.UNREACHABLE,
    snapshot: await dead.pollOnce(),
  });

  // 5. Measured, then the server disappeared -- the ageing path, which keeps
  //    real prior readings and must not lose their captions while doing so.
  let reachable = true;
  const flaky = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: async (url) => {
      if (!reachable) {
        throw new TypeError('Failed to fetch');
      }
      return fakeFetch(healthyRoutes())(url);
    },
  });
  await flaky.pollOnce();
  reachable = false;
  frames.push({
    name: 'offline frame (measured, then server vanished)',
    expectedState: CONNECTION_STATES.UNREACHABLE,
    snapshot: await flaky.pollOnce(),
  });

  return frames;
}

test('every frame the store can build is enumerated here', async () => {
  const frames = await everyFrame();

  // The anti-vacuity floor. A census that enumerates nothing and a store with
  // no defects are byte-identical from here, so the denominator is asserted
  // before anything is concluded from the numerator.
  assert.ok(
    frames.length >= 5,
    `only ${frames.length} frames enumerated; this census is BROKEN, not the store clean`,
  );
  for (const { name, expectedState, snapshot } of frames) {
    assert.equal(snapshot.connection.state, expectedState, `${name} did not reach its state`);
    assert.ok(Object.keys(snapshot.fields).length > 0, `${name} carries no fields at all`);
  }
});

test('no field on any frame falls back to the caption "value"', async () => {
  const frames = await everyFrame();
  const offenders = [];

  for (const { name, snapshot } of frames) {
    for (const key of allFieldKeys()) {
      const field = snapshot.fields[key];
      assert.ok(field, `${name} is missing the field ${key} entirely`);
      if (typeof field.label !== 'string' || field.label.length === 0) {
        offenders.push(`${name}: ${key} has no label, so it renders as "${FALLBACK_CAPTION}"`);
      }
    }
  }

  assert.deepEqual(
    offenders,
    [],
    `${offenders.length} field(s) would render as the literal word "${FALLBACK_CAPTION}":\n  ` +
      offenders.slice(0, 8).join('\n  '),
  );
});

test('every field carries the catalogue caption, not merely some caption', async () => {
  // Presence is exactly what an overwrite preserves: a frame could satisfy the
  // test above by inventing a caption of its own. The frames must AGREE.
  const frames = await everyFrame();
  const disagreements = [];

  for (const { name, snapshot } of frames) {
    for (const key of allFieldKeys()) {
      const expected = resolveForOrigin(PROVENANCE[key], 'scatter').label;
      const actual = snapshot.fields[key].label;
      if (actual !== expected) {
        disagreements.push(`${name}: ${key} reads "${actual}", catalogue says "${expected}"`);
      }
    }
  }

  assert.deepEqual(disagreements, [], disagreements.slice(0, 8).join('\n  '));
});

test('a derived field never claims on any frame that the server reported it', async () => {
  // The half of this bug that is not cosmetic. `unavailableField` defaults
  // sourceClass to SERVER, so a hand-rolled meta silently converts a value the
  // DASHBOARD computed into one the server is blamed for.
  const frames = await everyFrame();
  const lies = [];

  for (const { name, snapshot } of frames) {
    for (const key of allFieldKeys()) {
      const entry = resolveForOrigin(PROVENANCE[key], 'scatter');
      const expected = entry.derived ? 'derived' : 'server';
      const actual = snapshot.fields[key].sourceClass;
      if (actual !== expected) {
        lies.push(`${name}: ${key} is ${expected} but announces ${actual}`);
      }
    }
  }

  // The positive control: this assertion is only meaningful if the catalogue
  // actually contains derived fields to get wrong.
  const derivedCount = allFieldKeys().filter(
    (key) => resolveForOrigin(PROVENANCE[key], 'scatter').derived,
  ).length;
  assert.ok(
    derivedCount > 0,
    'the catalogue declares no derived fields, so this test proves nothing',
  );

  assert.deepEqual(lies, [], lies.slice(0, 8).join('\n  '));
});
