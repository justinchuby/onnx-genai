// Copyright (c) Microsoft Corporation.
//
// Tests for the telemetry store. Run with the Node built-in test runner — no
// dependencies, no install, consistent with the demo's no-build-step rule:
//
//   node --test examples/serving-dashboard/
//
// The behaviours locked down here are the ones the demo's credibility rests on:
// a documented zero must never become a rendered measurement, the two blocking
// failure states must stay distinct, and a slow server must not queue polls.

import test from 'node:test';
import assert from 'node:assert/strict';

import { createTelemetryStore, CONNECTION_STATES } from './telemetry-store.js';
import { FIELD_STATES } from './telemetry-field.js';
import { ENDPOINTS } from './telemetry-provenance.js';

const BASE_URL = 'http://127.0.0.1:8123';

/** A /v1/status body shaped exactly like the real server's, documented zeros and all. */
function statusBody(overrides = {}) {
  return {
    node_id: 'node-0',
    healthy: true,
    kv_usage: 0.0,
    kv_pages_used: 0,
    kv_pages_total: 0,
    kv_pages_shared: 0,
    queue_depth: 3,
    active_sessions: 2,
    paused_sessions: 0,
    tokens_per_second: 0.0,
    batch_utilization: 0.0,
    sessions: [],
    prefix_hashes: [],
    ...overrides,
  };
}

function debugKvBody(overrides = {}) {
  return {
    prefix_cache_hits: 4,
    prefix_cache_lookups: 5,
    prefix_cache_hit_rate: 0.8,
    active_batch_size: 3,
    pending_queue_depth: 3,
    available_admission_slots: 253,
    rejected_requests: 0,
    engine_kv_introspection: 'unavailable: engine does not yet expose KV page statistics',
    ...overrides,
  };
}

/**
 * Build a fake fetch from a path -> response-descriptor map.
 * A descriptor is `{ status, body }`, or an Error to simulate a transport failure.
 */
function fakeFetch(routes) {
  return async (url) => {
    const path = url.replace(BASE_URL, '');
    const route = routes[path];
    if (route === undefined) {
      return jsonResponse(404, { error: { message: 'not found', type: 'server_error' } });
    }
    if (route instanceof Error) {
      throw route;
    }
    return jsonResponse(route.status ?? 200, route.body);
  };
}

function jsonResponse(status, body) {
  const text = JSON.stringify(body);
  return {
    ok: status >= 200 && status < 300,
    status,
    async json() {
      return JSON.parse(text);
    },
    async text() {
      return text;
    },
  };
}

function healthyRoutes(overrides = {}) {
  return {
    [ENDPOINTS.HEALTH]: { body: { status: 'ok', model: 'qwen-scatter' } },
    [ENDPOINTS.STATUS]: { body: statusBody() },
    [ENDPOINTS.DEBUG_KV]: { body: debugKvBody() },
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
    ...overrides,
  };
}

function storeWith(routes) {
  return createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: fakeFetch(routes),
  });
}

test('a genuinely measured field is reported as measured with its source', async () => {
  const store = storeWith(healthyRoutes());
  await store.pollOnce();

  const queueDepth = store.field('queue.depth');
  assert.equal(queueDepth.state, FIELD_STATES.MEASURED);
  assert.equal(queueDepth.value, 3);
  assert.equal(queueDepth.source, ENDPOINTS.STATUS);
  assert.equal(queueDepth.unit, 'requests');
  assert.equal(queueDepth.reason, null);
});

test('documented zeros from /v1/status NEVER become measured values', async () => {
  // This is the single most important test in the demo. The server sends
  // literal 0.0 for these; rendering them as measurements would be a lie.
  const store = storeWith(healthyRoutes());
  await store.pollOnce();

  for (const key of [
    'kv.usage',
    'kv.pages_used',
    'kv.pages_total',
    'kv.pages_shared',
    'throughput.tokens_per_second',
    'batch.utilization',
    'sessions.paused',
  ]) {
    const field = store.field(key);
    assert.equal(field.state, FIELD_STATES.UNAVAILABLE, `${key} must be unavailable`);
    assert.equal(field.value, null, `${key} must carry no value`);
    assert.ok(field.reason && field.reason.length > 20, `${key} must explain itself`);
  }
});

test('a real zero from a measured field is still reported as a measurement', async () => {
  // The mirror image: `queue_depth: 0` is a genuine "nothing is queued".
  const routes = healthyRoutes({
    [ENDPOINTS.STATUS]: { body: statusBody({ queue_depth: 0 }) },
  });
  const store = storeWith(routes);
  await store.pollOnce();

  const field = store.field('queue.depth');
  assert.equal(field.state, FIELD_STATES.MEASURED);
  assert.equal(field.value, 0);
});

test('an unknown field key returns an explained unavailable field, never undefined', async () => {
  const store = storeWith(healthyRoutes());
  await store.pollOnce();

  const field = store.field('does.not.exist');
  assert.equal(field.state, FIELD_STATES.UNAVAILABLE);
  assert.match(field.reason, /No field named/);
});

test('transport failure produces the UNREACHABLE blocking state', async () => {
  const store = storeWith({
    [ENDPOINTS.HEALTH]: new TypeError('Failed to fetch'),
    [ENDPOINTS.STATUS]: new TypeError('Failed to fetch'),
    [ENDPOINTS.DEBUG_KV]: new TypeError('Failed to fetch'),
    [ENDPOINTS.DEBUG_CONFIG]: new TypeError('Failed to fetch'),
  });
  const snapshot = await store.pollOnce();

  assert.equal(snapshot.connection.state, CONNECTION_STATES.UNREACHABLE);
  assert.equal(snapshot.connection.transportError, 'Failed to fetch');
  assert.equal(snapshot.connection.origin, BASE_URL);
  assert.equal(snapshot.connection.consecutiveFailures, 1);
});

test('reachable-but-no-model is a DISTINCT state from unreachable', async () => {
  const store = storeWith(
    healthyRoutes({
      [ENDPOINTS.STATUS]: { body: statusBody({ healthy: false }) },
      [ENDPOINTS.DEBUG_CONFIG]: {
        status: 500,
        body: { error: { message: 'no model loaded', type: 'server_error' } },
      },
    }),
  );
  const snapshot = await store.pollOnce();

  assert.equal(snapshot.connection.state, CONNECTION_STATES.NO_MODEL);
  // The server's own words, verbatim — we do not paraphrase server errors.
  assert.equal(snapshot.connection.serverMessage, 'no model loaded');
  assert.equal(snapshot.connection.transportError, null);
});

test('measured fields go stale (not unavailable) when the server disappears', async () => {
  let reachable = true;
  const routes = healthyRoutes();
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: async (url) => {
      if (!reachable) throw new TypeError('Failed to fetch');
      return fakeFetch(routes)(url);
    },
  });

  await store.pollOnce();
  const before = store.field('queue.depth');
  assert.equal(before.state, FIELD_STATES.MEASURED);

  reachable = false;
  await store.pollOnce();
  const after = store.field('queue.depth');
  assert.equal(after.state, FIELD_STATES.STALE);
  assert.equal(after.value, 3, 'a stale field keeps its last good value');
  assert.equal(after.observedAtMs, before.observedAtMs, 'staleness must preserve the original age');
});

test('an unavailable field does not become stale — absence has no age', async () => {
  let reachable = true;
  const routes = healthyRoutes();
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: async (url) => {
      if (!reachable) throw new TypeError('Failed to fetch');
      return fakeFetch(routes)(url);
    },
  });

  await store.pollOnce();
  reachable = false;
  await store.pollOnce();

  assert.equal(store.field('kv.usage').state, FIELD_STATES.UNAVAILABLE);
});

test('a disabled debug endpoint degrades only its own fields, with the exact fix', async () => {
  const store = storeWith(
    healthyRoutes({
      [ENDPOINTS.DEBUG_KV]: {
        status: 404,
        body: { error: { message: 'not found', type: 'server_error' } },
      },
    }),
  );
  const snapshot = await store.pollOnce();

  assert.equal(snapshot.connection.state, CONNECTION_STATES.CONNECTED);
  assert.equal(store.field('queue.depth').state, FIELD_STATES.MEASURED, '/v1/status still works');

  const gated = store.field('prefix_cache.hits');
  assert.equal(gated.state, FIELD_STATES.UNAVAILABLE);
  assert.match(gated.reason, /--enable-debug-endpoints/);
});

test('subscribers receive the current snapshot immediately on subscribe', async () => {
  const store = storeWith(healthyRoutes());
  await store.pollOnce();

  let received = null;
  const unsubscribe = store.subscribe((snapshot) => {
    received = snapshot;
  });

  assert.ok(received, 'subscribe must deliver the current snapshot synchronously');
  assert.equal(received.connection.state, CONNECTION_STATES.CONNECTED);
  unsubscribe();
});

test('unsubscribe stops delivery', async () => {
  const store = storeWith(healthyRoutes());
  let calls = 0;
  const unsubscribe = store.subscribe(() => {
    calls += 1;
  });
  assert.equal(calls, 1);
  unsubscribe();
  await store.pollOnce();
  assert.equal(calls, 1, 'no delivery after unsubscribe');
});

test('one panel throwing does not stop other panels updating', async () => {
  const store = storeWith(healthyRoutes());
  let goodPanelUpdates = 0;
  store.subscribe(() => {
    throw new Error('panel bug');
  });
  store.subscribe(() => {
    goodPanelUpdates += 1;
  });

  await store.pollOnce();
  assert.ok(goodPanelUpdates >= 2, 'the healthy panel kept receiving snapshots');
});

test('at most one poll cycle is in flight at a time', async () => {
  // A slow server must not cause polls to queue up behind each other.
  let inFlight = 0;
  let maxInFlight = 0;
  const routes = healthyRoutes();
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: async (url) => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((resolve) => setTimeout(resolve, 10));
      inFlight -= 1;
      return fakeFetch(routes)(url);
    },
  });

  await Promise.all([store.pollOnce(), store.pollOnce(), store.pollOnce()]);
  // 4 endpoints fetched concurrently within ONE cycle is expected; 8+ would
  // mean a second cycle started before the first finished.
  assert.ok(maxInFlight <= 4, `expected <= 4 concurrent requests, saw ${maxInFlight}`);
});

test('a failing endpoint is not re-requested on every poll (no console flood)', async () => {
  // A disabled debug endpoint or a model-less server returns the same error
  // forever. At 250ms that is 4 failed requests/second — a flood that buries
  // real errors. The failure must persist for panels without the network noise.
  let debugKvRequests = 0;
  const routes = healthyRoutes();
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: async (url) => {
      if (url.endsWith(ENDPOINTS.DEBUG_KV)) {
        debugKvRequests += 1;
        return jsonResponse(404, { error: { message: 'not found', type: 'server_error' } });
      }
      return fakeFetch(routes)(url);
    },
  });

  await store.pollOnce();
  await store.pollOnce();
  await store.pollOnce();
  await store.pollOnce();

  assert.equal(debugKvRequests, 1, 'the failing endpoint was requested exactly once');
  // The explanation must survive suppression — panels still need it.
  const field = store.field('prefix_cache.hits');
  assert.equal(field.state, FIELD_STATES.UNAVAILABLE);
  assert.match(field.reason, /--enable-debug-endpoints/);
  // Healthy endpoints keep polling normally.
  assert.equal(store.field('queue.depth').state, FIELD_STATES.MEASURED);
});

test('a suppressed endpoint recovers once its retry window elapses', async () => {
  // The failure states promise the page recovers on its own. It must.
  let clock = 1_000_000;
  let failing = true;
  let requests = 0;
  const routes = healthyRoutes();
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    now: () => clock,
    fetchImpl: async (url) => {
      if (url.endsWith(ENDPOINTS.DEBUG_KV)) {
        requests += 1;
        if (failing) {
          return jsonResponse(404, { error: { message: 'not found', type: 'server_error' } });
        }
      }
      return fakeFetch(routes)(url);
    },
  });

  await store.pollOnce();
  assert.equal(requests, 1);

  await store.pollOnce();
  assert.equal(requests, 1, 'still suppressed inside the retry window');

  failing = false;
  clock += 11_000;
  await store.pollOnce();
  assert.equal(requests, 2, 'retried after the window elapsed');
  assert.equal(store.field('prefix_cache.hits').state, FIELD_STATES.MEASURED);
});

test('an unavailable field never inherits an explanation from an unrelated earlier state', async () => {
  // Caught in the browser: after no-model -> unreachable, hovering queue.depth
  // said "the server has no model loaded". A confident, specific, WRONG answer
  // is worse than no answer — the reason must be re-derived, not carried over.
  let mode = 'nomodel';
  const routes = healthyRoutes();
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: async (url) => {
      if (mode === 'dead') throw new TypeError('Failed to fetch');
      if (url.endsWith(ENDPOINTS.STATUS)) {
        return jsonResponse(200, statusBody({ healthy: false }));
      }
      return fakeFetch(routes)(url);
    },
  });

  await store.pollOnce();
  assert.match(store.field('queue.depth').reason, /no model loaded/);

  mode = 'dead';
  await store.pollOnce();

  const snapshot = store.getSnapshot();
  assert.equal(snapshot.connection.state, CONNECTION_STATES.UNREACHABLE);
  const field = store.field('queue.depth');
  assert.doesNotMatch(
    field.reason,
    /no model loaded/,
    'the no-model explanation must not survive into the unreachable state',
  );
  assert.match(field.reason, /not responding/);

  // A permanently-unmeasurable field keeps its own true explanation throughout.
  assert.match(store.field('kv.usage').reason, /hardcoded 0\.0/);
});

test('the store is inert until start() is called', () => {
  const store = storeWith(healthyRoutes());
  assert.equal(store.isRunning, false);
  assert.equal(store.getSnapshot().connection.state, CONNECTION_STATES.CONNECTING);
});

test('before the first poll, measurable fields are PENDING and documented zeros are UNAVAILABLE', () => {
  // The distinction matters to the visitor: pending resolves on its own,
  // unavailable never will. Showing a spinner for `kv.usage` would promise a
  // number that is never coming.
  const store = storeWith(healthyRoutes());

  assert.equal(store.field('queue.depth').state, FIELD_STATES.PENDING);
  assert.equal(store.field('queue.depth').value, null);
  assert.equal(store.field('kv.usage').state, FIELD_STATES.UNAVAILABLE);
  assert.equal(store.field('throughput.tokens_per_second').state, FIELD_STATES.UNAVAILABLE);
});

test('a pending field does not become stale — a value that never arrived cannot age', async () => {
  const store = storeWith({
    [ENDPOINTS.HEALTH]: new TypeError('Failed to fetch'),
    [ENDPOINTS.STATUS]: new TypeError('Failed to fetch'),
    [ENDPOINTS.DEBUG_KV]: new TypeError('Failed to fetch'),
    [ENDPOINTS.DEBUG_CONFIG]: new TypeError('Failed to fetch'),
  });
  await store.pollOnce();

  assert.equal(store.field('queue.depth').state, FIELD_STATES.PENDING);
});

test('every field is present in the very first snapshot, before any poll', () => {
  const store = storeWith(healthyRoutes());
  const snapshot = store.getSnapshot();
  for (const field of Object.values(snapshot.fields)) {
    assert.ok(
      field.state === FIELD_STATES.PENDING || field.state === FIELD_STATES.UNAVAILABLE,
      'no field may claim to be measured before the first poll',
    );
    assert.equal(field.value, null);
    assert.ok(field.reason);
  }
});
