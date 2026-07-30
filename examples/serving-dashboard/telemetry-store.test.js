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
import {
  ENDPOINTS,
  allFieldKeys,
  provenanceFor,
  PROVENANCE,
  NEVER_BIND,
  resolveForOrigin,
  NEVER_MEASURED_CLASSIFICATIONS,
} from './telemetry-provenance.js';

const BASE_URL = 'http://127.0.0.1:8123';

/**
 * A /v1/status body shaped exactly like the real server's.
 *
 * THE OMISSIONS ARE THE POINT, and they used to be zeros. The server now sends
 * `None` for everything it cannot compute (routes/admin.rs:155-164), so those
 * keys are ABSENT from the payload rather than present-and-zero. That is the
 * fix this whole demo argued for: a documented zero is byte-identical to a
 * measured zero, and only the server can tell them apart -- by omitting.
 *
 * So this fixture must OMIT them too. Sending `kv_usage: 0` here would test a
 * server that no longer exists, and it would do it invisibly: the store would
 * read 0, see it no longer matches a declared stub, and correctly announce
 * that the plumbing had landed. The test would fail for a reason that has
 * nothing to do with the behaviour it names.
 */
function statusBody(overrides = {}) {
  return {
    node_id: 'node-0',
    healthy: true,
    queue_depth: 3,
    active_sessions: 2,
    // Genuinely computed now: in-flight over assemblable batch capacity
    // (routes/admin.rs:169-172). Not a stub, so it carries a real value.
    batch_utilization: 0.5,
    batch_in_flight: 2,
    sessions: [],
    ...overrides,
  };
}

// Models the SCATTER server, which is what storeWith() defaults to. The
// prefix-cache counters are 0 here because that is the only thing a real
// scatter server can send: engine/batched.rs:262 passes a hardcoded literal 0
// as prefix_cache_hit_len, and ContinuousBatchManager holds no reference to
// engine.prefix_cache at all. A fixture that emitted 4 hits here -- as this one
// did until the stale-provenance check caught it -- describes a server that
// cannot exist, and any test built on it is verifying an imaginary world.
// Tests that want live cache numbers pass DYNAMIC and override explicitly.
function debugKvBody(overrides = {}) {
  return {
    prefix_cache_hits: 0,
    prefix_cache_lookups: 0,
    prefix_cache_hit_rate: 0,
    active_batch_size: 3,
    pending_queue_depth: 3,
    available_admission_slots: 253,
    rejected_requests: 0,
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
    // `/metrics` is Prometheus text, not JSON. Routes carrying `text` are
    // served verbatim so the store exercises its real parsing path.
    if (typeof route.text === 'string') {
      return textResponse(route.status ?? 200, route.text);
    }
    return jsonResponse(route.status ?? 200, route.body);
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

/**
 * A Prometheus body shaped exactly like the live server's, with the values the
 * tests need. `batch_size_current` is deliberately larger than the hardcoded
 * max_batch of 4, which is the case that exposes the naming trap.
 */
function metricsBody({ inFlight = 3, tokens = 5048, ttftSum = 20.7, ttftCount = 10, lookups = 11 } = {}) {
  return `# HELP onnx_genai_tokens_generated_total Total prompt and completion tokens processed.
# TYPE onnx_genai_tokens_generated_total counter
onnx_genai_tokens_generated_total ${tokens}
# HELP onnx_genai_batch_size_current Current generation batch size.
# TYPE onnx_genai_batch_size_current gauge
onnx_genai_batch_size_current ${inFlight}
# HELP onnx_genai_time_to_first_token_seconds Time to first generated token.
# TYPE onnx_genai_time_to_first_token_seconds histogram
onnx_genai_time_to_first_token_seconds_bucket{le="+Inf"} ${ttftCount}
onnx_genai_time_to_first_token_seconds_sum ${ttftSum}
onnx_genai_time_to_first_token_seconds_count ${ttftCount}
# HELP onnx_genai_prefix_cache_hits_total Generation requests with a prefix-cache hit.
# TYPE onnx_genai_prefix_cache_hits_total counter
onnx_genai_prefix_cache_hits_total 0
# HELP onnx_genai_prefix_cache_lookups_total Generation requests checked for prefix-cache reuse.
# TYPE onnx_genai_prefix_cache_lookups_total counter
onnx_genai_prefix_cache_lookups_total ${lookups}
`;
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
    [ENDPOINTS.METRICS]: { text: metricsBody() },
    [ENDPOINTS.RESOURCES]: {
      body: {
        derived_kv_budget: { bytes: 5746050801 },
        vram: { used: 0, limit: 5746050801, headroom: 5746050801 },
      },
    },
    ...overrides,
  };
}

function storeWith(routes, options = {}) {
  return createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: fakeFetch(routes),
    ...options,
  });
}

function jsonPathRows() {
  return Object.entries(PROVENANCE).filter(
    ([, entry]) => !entry.derived && !entry.metric && entry.path,
  );
}

function setDottedPath(body, path, value) {
  const parts = path.split('.');
  let cursor = body;
  for (const part of parts.slice(0, -1)) {
    cursor[part] = {};
    cursor = cursor[part];
  }
  cursor[parts.at(-1)] = value;
  return body;
}

test('wrong JSON wire types are rejected as unavailable across the full census', async () => {
  const rows = jsonPathRows();
  assert.equal(rows.length, 30, 'the store enforcement census drifted from the declaration guard');

  const numericSentinels = [
    '',
    '/Users/operator/models/qwen',
    'node-operator',
    { unexpected: true },
    ['unexpected'],
    'NaN',
  ];
  const wrongValueFor = (key, wireType, index) => {
    if (wireType === 'number') return numericSentinels[index % numericSentinels.length];
    if (wireType === 'boolean') return key === 'server.healthy' ? 'false' : 1;
    if (key === 'server.model_id') return 17;
    if (key === 'server.node_id') return ['node-0'];
    return { provider: 'CPU' };
  };

  for (const [index, [key, entry]] of rows.entries()) {
    const wrongValue = wrongValueFor(key, entry.wireType, index);
    const requestPath =
      entry.source === ENDPOINTS.DEBUG_KV_BLOCKS
        ? `${entry.source}?start=0&count=1024`
        : entry.source;
    const routes = healthyRoutes({
      [requestPath]: { body: setDottedPath({}, entry.path, wrongValue) },
    });
    const warnings = [];
    const originalWarn = console.warn;
    console.warn = (...args) => warnings.push(args.join(' '));
    let store;
    try {
      store = storeWith(routes);
      await store.pollOnce();
    } finally {
      console.warn = originalWarn;
    }

    const field = store.field(key);
    assert.equal(field.state, FIELD_STATES.UNAVAILABLE, `${key} accepted ${JSON.stringify(wrongValue)}`);
    assert.equal(field.value, null, `${key} retained a mismatched wire value`);
    assert.ok(
      Object.hasOwn(store.provenanceWarnings(), key),
      `${key} rejected a mismatch without a developer-visible provenance warning`,
    );
    assert.ok(
      warnings.some((warning) => warning.includes('wire type mismatch')),
      `${key} did not emit a visible wire-type warning`,
    );
    assert.ok(
      !warnings.some((warning) => warning.includes('/Users/operator')),
      `${key} echoed a rejected sensitive value into the warning`,
    );
  }
});

test('a missing wireType declaration fails closed instead of permitting the value', async () => {
  const entry = PROVENANCE['queue.depth'];
  const declaredType = entry.wireType;
  const warnings = [];
  const originalWarn = console.warn;
  delete entry.wireType;
  console.warn = (...args) => warnings.push(args.join(' '));
  let store;
  try {
    store = storeWith(healthyRoutes());
    await store.pollOnce();
  } finally {
    entry.wireType = declaredType;
    console.warn = originalWarn;
  }

  assert.equal(store.field('queue.depth').state, FIELD_STATES.UNAVAILABLE);
  assert.ok(Object.hasOwn(store.provenanceWarnings(), 'queue.depth'));
  assert.ok(warnings.some((warning) => warning.includes('wire type mismatch')));
});

test('valid numeric zero and boolean false still promote as measurements', async () => {
  const store = storeWith(healthyRoutes());
  await store.pollOnce();

  assert.equal(store.field('admission.rejections').state, FIELD_STATES.MEASURED);
  assert.equal(store.field('admission.rejections').value, 0);
  assert.equal(store.field('server.pipeline').state, FIELD_STATES.MEASURED);
  assert.equal(store.field('server.pipeline').value, false);
});

// Endpoint GATING and structural bypass are different facts, and on the scatter
// server the structural one is deeper: enabling --enable-debug-endpoints there
// would still yield nothing, because the batching path never consults the prefix
// cache. Tests about gating therefore use the dynamic server, where the gate
// really is the only thing in the way.
const DYNAMIC = { origin: 'dynamic' };

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

  // NOT_PLUMBED: the server omits these entirely, so they are unavailable --
  // "we cannot measure this", with a fix that exists and is someone's job.
  // The four kv.* keys that used to head this list are gone: they are now
  // MEASURED off /v1/debug/kv/blocks, an endpoint that had shipped while this
  // table still called them unplumbed. What remains is the set the server
  // genuinely does not compute.
  for (const key of [
    'throughput.tokens_per_second',
    'batch.effective_size',
    'server.execution_provider',
  ]) {
    const field = store.field(key);
    assert.equal(field.state, FIELD_STATES.UNAVAILABLE, `${key} must be unavailable`);
    assert.equal(field.value, null, `${key} must carry no value`);
    assert.ok(field.reason && field.reason.length > 20, `${key} must explain itself`);
  }

  // STRUCTURALLY_BYPASSED is a DIFFERENT claim and must not collapse into the
  // one above: this path never pauses a session, so the question is never
  // asked. No plumbing changes that, which is exactly why it renders n/a.
  const paused = store.field('sessions.paused');
  assert.equal(paused.state, FIELD_STATES.NOT_APPLICABLE, 'sessions.paused is bypassed, not missing');
  assert.equal(paused.value, null);

  // And the one that became REAL. batch_utilization is now computed from
  // in-flight over assemblable capacity, so asserting it unavailable would
  // deny working telemetry -- the inverse defect, and the harder one to see.
  const utilisation = store.field('batch.utilization');
  assert.equal(utilisation.state, FIELD_STATES.MEASURED, 'batch.utilization is measured now');
  assert.equal(utilisation.value, 0.5);
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

  assert.equal(store.field('throughput.tokens_per_second').state, FIELD_STATES.UNAVAILABLE);
});

test('a disabled debug endpoint degrades only its own fields, with the exact fix', async () => {
  const store = storeWith(
    healthyRoutes({
      [ENDPOINTS.DEBUG_KV]: {
        status: 404,
        body: { error: { message: 'not found', type: 'server_error' } },
      },
    }),
    DYNAMIC,
  );
  const snapshot = await store.pollOnce();

  assert.equal(snapshot.connection.state, CONNECTION_STATES.CONNECTED);
  assert.equal(store.field('queue.depth').state, FIELD_STATES.MEASURED, '/v1/status still works');

  // batch.active_size, NOT a prefix-cache field: those are MISATTRIBUTED and so
  // are unavailable whatever the endpoint does, which would make this assertion
  // pass for the wrong reason and keep passing if the gating broke entirely.
  const gated = store.field('batch.active_size');
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
  // One cycle fetches every polled endpoint concurrently, so the bound is the
  // endpoint count itself. Derived rather than hardcoded: adding an endpoint
  // should not look like a concurrency regression. Anything above this means a
  // second cycle started before the first finished.
  const endpointsPerCycle = Object.keys(ENDPOINTS).length;
  assert.ok(
    maxInFlight <= endpointsPerCycle,
    `expected <= ${endpointsPerCycle} concurrent requests (one cycle), saw ${maxInFlight}`,
  );
});

test('a failing endpoint is not re-requested on every poll (no console flood)', async () => {
  // A disabled debug endpoint or a model-less server returns the same error
  // forever. At 250ms that is 4 failed requests/second — a flood that buries
  // real errors. The failure must persist for panels without the network noise.
  let debugKvRequests = 0;
  const routes = healthyRoutes();
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    ...DYNAMIC,
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
  const field = store.field('batch.active_size');
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
    ...DYNAMIC,
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
  assert.equal(store.field('batch.active_size').state, FIELD_STATES.MEASURED);
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
  // The server stopped sending a hardcoded 0.0 and now omits the field, so the
  // reason names the OMISSION. Pinning the old prose would assert a server
  // behaviour that no longer exists.
  assert.match(store.field('throughput.tokens_per_second').reason, /Omitted, not zeroed/);
});

test('the store is inert until start() is called', () => {
  const store = storeWith(healthyRoutes());
  assert.equal(store.isRunning, false);
  assert.equal(store.getSnapshot().connection.state, CONNECTION_STATES.CONNECTING);
});

test('before the first poll, measurable fields are PENDING and documented zeros are UNAVAILABLE', () => {
  // The distinction matters to the visitor: pending resolves on its own,
  // unavailable never will. Showing a spinner for a field the server does not
  // compute would promise a number that is never coming.
  const store = storeWith(healthyRoutes());

  assert.equal(store.field('queue.depth').state, FIELD_STATES.PENDING);
  assert.equal(store.field('queue.depth').value, null);
  assert.equal(store.field('batch.effective_size').state, FIELD_STATES.UNAVAILABLE);
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
    // Stated as what must NOT be true, rather than an allow-list of permitted
    // states. The invariant is "nothing claims a measurement before one has
    // happened"; an allow-list expresses that only accidentally, and has to be
    // edited every time a non-measuring state is added -- which quietly turns a
    // real invariant into a list someone maintains.
    assert.ok(
      field.state !== FIELD_STATES.MEASURED && field.state !== FIELD_STATES.STALE,
      `no field may claim to be measured before the first poll (saw ${field.state})`,
    );
    assert.equal(field.value, null);
    assert.ok(field.reason);
  }
});

// ── /metrics: the honest counterpart to /v1/status's fabricated zeros ───────

test('Prometheus metrics become measured fields', async () => {
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: fakeFetch(healthyRoutes()),
  });
  await store.pollOnce();
  const { fields } = store.getSnapshot();

  assert.equal(fields['metrics.tokens_generated_total'].state, FIELD_STATES.MEASURED);
  assert.equal(fields['metrics.tokens_generated_total'].value, 5048);
  // TTFT arrives as a histogram and must be reduced to sum/count, not read raw.
  assert.equal(fields['metrics.ttft'].state, FIELD_STATES.MEASURED);
  assert.ok(Math.abs(fields['metrics.ttft'].value - 2.07) < 0.001);
});

test('the in-flight gauge is NEVER exposed as the engine batch size', async () => {
  // The single most dangerous quantity the server publishes. It is serialised as
  // /v1/status.batch_in_flight from snapshot.current_batch_size, and it is
  // incremented per HTTP generation and decremented on drop -- so it counts
  // requests in flight, NOT the width the engine actually batched. On a busy
  // server it reads 8 while the engine batched only 4.
  //
  // This field used to be read from /metrics (onnx_genai_batch_size_current) and
  // was rebound to /v1/status. Inject through the endpoint the catalogue actually
  // reads: feeding the old route does not error, it silently supplies nothing and
  // the store returns the other endpoint's value instead.
  //
  // What actually defends this test's title is the `source` binding, and that is
  // mutation-proven: flip 'batch.in_flight' to ENDPOINTS.METRICS in
  // telemetry-provenance.js and this goes red. Note it fails with 'unavailable'
  // rather than with a gauge value -- nothing parses onnx_genai_batch_size_current
  // into this field, so also feeding /metrics here adds no discriminating power.
  // That was measured all four ways, not assumed.
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: fakeFetch(healthyRoutes({ [ENDPOINTS.STATUS]: { body: statusBody({ batch_in_flight: 8 }) } })),
  });
  await store.pollOnce();
  const { fields } = store.getSnapshot();

  // It is a real measurement -- of in-flight generations.
  assert.equal(fields['batch.in_flight'].state, FIELD_STATES.MEASURED);
  assert.equal(fields['batch.in_flight'].value, 8);

  // The number a viewer would assume "batch size" means is NOT available, and
  // must not be silently backfilled from the gauge above.
  assert.equal(fields['batch.effective_size'].state, FIELD_STATES.UNAVAILABLE);
  assert.equal(fields['batch.effective_size'].value, null);
  assert.match(fields['batch.effective_size'].reason, /does not report/i);
});

test('the same zero means opposite things on the two servers', async () => {
  // THE THREE KINDS OF ZERO, on one wire value. prefix_cache_lookups is 0 on
  // both servers and byte-identical in the response. On the dynamic server the
  // counter DOES run, so 0 is real data -- zero completed generations. On the
  // scatter server the batching path never consults the cache at all, so 0 is
  // not an observation and rendering it would imply a cache that tried.
  //
  // THIS TEST USED TO ASSERT THE PAIR ON prefix_cache_hits AND ITS COMMENT SAID
  // "on the dynamic server the cache WAS consulted and hit nothing, so 0 is real
  // data". That sentence was copied from the docstring at the top of
  // telemetry-provenance.js, and it was FALSE: the hits counter never reads 0 on
  // a server doing work -- it scores a hit for ANY nonzero token match, and every
  // chat request shares the template preamble. A false premise in a doc comment
  // was ratified here as a test, which is how it survived review. The hits
  // counter is now MISATTRIBUTED; the pair moved to the counter that is honest.
  const routes = healthyRoutes({
    [ENDPOINTS.METRICS]: { text: metricsBody({ lookups: 0 }) },
  });
  const dynamic = createTelemetryStore({
    baseUrl: BASE_URL,
    origin: 'dynamic',
    fetchImpl: fakeFetch(routes),
  });
  const scatter = createTelemetryStore({
    baseUrl: BASE_URL,
    origin: 'scatter',
    fetchImpl: fakeFetch(routes),
  });
  await Promise.all([dynamic.pollOnce(), scatter.pollOnce()]);

  const onDynamic = dynamic.getSnapshot().fields['metrics.prefix_cache_lookups'];
  assert.equal(onDynamic.state, FIELD_STATES.MEASURED);
  assert.equal(onDynamic.value, 0, 'a real zero must survive as zero, not be hidden');

  const onScatter = scatter.getSnapshot().fields['metrics.prefix_cache_lookups'];
  assert.equal(onScatter.state, FIELD_STATES.NOT_APPLICABLE);
  assert.equal(onScatter.value, null);
  // The reason must explain WHY the path bypasses it, not merely that it did.
  assert.match(onScatter.reason, /never consults/i);

  // THE THIRD KIND, and the one that has no zero at all: the hits counter is
  // live on dynamic and counts the wrong quantity, so it is suppressed on BOTH
  // servers -- for two DIFFERENT reasons, which must not collapse into one.
  const hitsDynamic = dynamic.getSnapshot().fields['metrics.prefix_cache_hits'];
  const hitsScatter = scatter.getSnapshot().fields['metrics.prefix_cache_hits'];
  assert.equal(hitsDynamic.state, FIELD_STATES.UNAVAILABLE);
  assert.equal(hitsScatter.state, FIELD_STATES.NOT_APPLICABLE);
  assert.notEqual(
    hitsDynamic.state,
    hitsScatter.state,
    'suppressed for different reasons must not render as the same state',
  );
  assert.match(hitsDynamic.reason, /not cache hits|matching token/i);
});
test('not-applicable is distinct from unavailable, not a synonym', async () => {
  // One says "the server does not compute this yet" (fixable by plumbing); the
  // other says "asking this of this server is meaningless" (permanent, and a
  // true statement about the architecture). Collapsing them loses a fact.
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    origin: 'scatter',
    fetchImpl: fakeFetch(healthyRoutes()),
  });
  await store.pollOnce();
  const { fields } = store.getSnapshot();

  assert.equal(fields['metrics.prefix_cache_hits'].state, FIELD_STATES.NOT_APPLICABLE);
  // A hardcoded stub is unavailable, NOT not-applicable.
  assert.equal(fields['throughput.tokens_per_second'].state, FIELD_STATES.UNAVAILABLE);
  assert.notEqual(
    fields['metrics.prefix_cache_hits'].state,
    fields['throughput.tokens_per_second'].state,
  );
});

test('an idle server reports no latency rather than zero latency', async () => {
  // Zero observations means sum/count is 0/0. Reporting "0s time to first
  // token" for a server that has generated nothing is a fabricated number.
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: fakeFetch(
      healthyRoutes({
        [ENDPOINTS.METRICS]: { text: metricsBody({ ttftSum: 0, ttftCount: 0 }) },
      }),
    ),
  });
  await store.pollOnce();
  const { fields } = store.getSnapshot();

  assert.equal(fields['metrics.ttft'].state, FIELD_STATES.UNAVAILABLE);
  assert.equal(fields['metrics.ttft'].value, null);
});

test('a server built without the metrics feature explains a rebuild, not a flag', async () => {
  // /metrics is compiled in by default, so a 404 means --no-default-features.
  // Telling the visitor to pass a runtime flag would send them down a dead end.
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: fakeFetch(healthyRoutes({ [ENDPOINTS.METRICS]: undefined })),
  });
  await store.pollOnce();
  const { fields } = store.getSnapshot();

  const field = fields['metrics.tokens_generated_total'];
  assert.equal(field.state, FIELD_STATES.UNAVAILABLE);
  assert.match(field.reason, /rebuild/i);
  assert.doesNotMatch(field.reason, /--enable-debug-endpoints/);
});

test('metrics degrade independently of the JSON endpoints', async () => {
  // Losing /metrics must not blank fields that /v1/status still serves.
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: fakeFetch(healthyRoutes({ [ENDPOINTS.METRICS]: undefined })),
  });
  await store.pollOnce();
  const { fields } = store.getSnapshot();

  assert.equal(fields['queue.depth'].state, FIELD_STATES.MEASURED);
  assert.equal(fields['metrics.ttft'].state, FIELD_STATES.UNAVAILABLE);
});

test('a stalled slow endpoint cannot block the fast lane or retain stale values', async () => {
  const routes = healthyRoutes();
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    requestTimeoutMs: 30,
    pollIntervalMs: 100,
    fetchImpl: (url, options = {}) => {
      if (new URL(url).pathname !== ENDPOINTS.METRICS) {
        return fakeFetch(routes)(url);
      }
      return new Promise((_resolve, reject) => {
        options.signal.addEventListener('abort', () => reject(new Error('aborted')));
      });
    },
  });

  const fastSnapshot = new Promise((resolve) => {
    const unsubscribe = store.subscribe((candidate) => {
      if (candidate.fields['queue.depth'].state !== FIELD_STATES.MEASURED) return;
      unsubscribe();
      resolve(candidate);
    });
  });
  const failedSlowSnapshot = new Promise((resolve) => {
    const unsubscribe = store.subscribe((candidate) => {
      const field = candidate.fields['metrics.ttft'];
      if (!/no response within 30 ms/.test(field.reason ?? '')) return;
      unsubscribe();
      resolve(candidate);
    });
  });

  store.start();
  const beforeSlowTimeout = await fastSnapshot;
  const afterSlowTimeout = await failedSlowSnapshot;
  store.stop();

  assert.equal(beforeSlowTimeout.fields['queue.depth'].state, FIELD_STATES.MEASURED);
  assert.equal(afterSlowTimeout.fields['queue.depth'].state, FIELD_STATES.MEASURED);
  assert.equal(afterSlowTimeout.fields['metrics.ttft'].state, FIELD_STATES.UNAVAILABLE);
});

test('an HTML error page served at /metrics does not crash the store', async () => {
  // A proxy or a wrong port realistically returns HTML with a 200.
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: fakeFetch(
      healthyRoutes({ [ENDPOINTS.METRICS]: { text: '<html><body>hello</body></html>' } }),
    ),
  });
  await store.pollOnce();
  const { fields } = store.getSnapshot();

  assert.equal(fields['metrics.ttft'].state, FIELD_STATES.UNAVAILABLE);
  assert.equal(fields['queue.depth'].state, FIELD_STATES.MEASURED);
});

test('/v1/resources fields are measured', async () => {
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: fakeFetch(healthyRoutes()),
  });
  await store.pollOnce();
  const { fields } = store.getSnapshot();

  assert.equal(fields['resources.kv_budget_bytes'].state, FIELD_STATES.MEASURED);
  assert.equal(fields['resources.kv_budget_bytes'].value, 5746050801);
});

// ── derived throughput: recovering a rate the server refuses to compute ─────

test('throughput is PENDING on the first poll, not unavailable', async () => {
  // A rate needs two samples. `pending` promises a number that is genuinely
  // coming; `unavailable` would wrongly say it never will.
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: fakeFetch(healthyRoutes()),
  });
  await store.pollOnce();

  assert.equal(store.getSnapshot().fields['throughput.observed'].state, FIELD_STATES.PENDING);
});

test('throughput is derived from the delta of the cumulative token counter', async () => {
  let tokens = 1000;
  let clock = 10_000;
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    now: () => clock,
    fetchImpl: async (url) =>
      fakeFetch(healthyRoutes({ [ENDPOINTS.METRICS]: { text: metricsBody({ tokens }) } }))(url),
  });

  await store.pollOnce();
  tokens = 1200;
  clock = 12_000;
  await store.pollOnce();

  // 200 tokens over 2 seconds.
  const field = store.getSnapshot().fields['throughput.observed'];
  assert.equal(field.state, FIELD_STATES.MEASURED);
  assert.equal(field.value, 100);
  // It must disclose that we computed it rather than read it.
  assert.deepEqual(field.derivedFrom, ['metrics.tokens_generated_total']);
});

test('a counter reset re-measures instead of reporting a negative rate', async () => {
  let tokens = 5000;
  let clock = 10_000;
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    now: () => clock,
    fetchImpl: async (url) =>
      fakeFetch(healthyRoutes({ [ENDPOINTS.METRICS]: { text: metricsBody({ tokens }) } }))(url),
  });

  await store.pollOnce();
  clock = 12_000;
  tokens = 12; // server restarted; counter went backwards
  await store.pollOnce();

  const field = store.getSnapshot().fields['throughput.observed'];
  assert.equal(field.state, FIELD_STATES.PENDING);
  assert.match(field.reason, /reset/i);
});

test('throughput is unavailable when /metrics is, and does not go stale silently', async () => {
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: fakeFetch(healthyRoutes({ [ENDPOINTS.METRICS]: undefined })),
  });
  await store.pollOnce();

  assert.equal(store.getSnapshot().fields['throughput.observed'].state, FIELD_STATES.UNAVAILABLE);
});

test('every provenance entry appears in the snapshot, including derived ones', () => {
  // Guards the footer table: a key documented in PROVENANCE but never emitted
  // (or vice versa) would make the "what's real" table lie by omission.
  const store = createTelemetryStore({ baseUrl: BASE_URL, fetchImpl: fakeFetch({}) });
  const keys = Object.keys(store.getSnapshot().fields).sort();

  assert.deepEqual(keys, allFieldKeys().sort());
});

// ── provenance axes and two-server attribution ─────────────────────────────

test('source CLASS and source ENDPOINT are separate, independent axes', async () => {
  // The designer's contract needs a provenance class for the hover; the spec
  // needs the endpoint so a claim can be audited against file:line evidence.
  // Collapsing them into one string would lose one of those uses.
  const store = createTelemetryStore({ baseUrl: BASE_URL, fetchImpl: fakeFetch(healthyRoutes()) });
  await store.pollOnce();
  const { fields } = store.getSnapshot();

  const readOffServer = fields['queue.depth'];
  assert.equal(readOffServer.sourceClass, 'server');
  assert.equal(readOffServer.source, ENDPOINTS.STATUS);

  // Same endpoint family, but WE computed this one -- different class.
  await new Promise((r) => setTimeout(r, 5));
  await store.pollOnce();
  const computed = store.getSnapshot().fields['throughput.observed'];
  assert.equal(computed.sourceClass, 'derived');
  assert.equal(computed.source, ENDPOINTS.METRICS);
});

test('an unavailable field still states which endpoint and server it would come from', async () => {
  // Without this, "unavailable" is unfalsifiable -- the audit table could not
  // tell a reader where to go and check.
  const store = createTelemetryStore({ baseUrl: BASE_URL, fetchImpl: fakeFetch(healthyRoutes()) });
  await store.pollOnce();

  const field = store.getSnapshot().fields['throughput.tokens_per_second'];
  assert.equal(field.state, FIELD_STATES.UNAVAILABLE);
  assert.equal(field.source, ENDPOINTS.STATUS);
  assert.equal(field.sourceClass, 'server');
  assert.ok(field.label, 'an unavailable field still needs a human label');
});

test('every field is attributed to the server it came from', async () => {
  // The demo runs two servers because batching and paged KV are mutually
  // exclusive. A number shown without saying which server produced it is its
  // own kind of fabrication.
  const scatter = createTelemetryStore({
    baseUrl: BASE_URL,
    origin: 'scatter',
    fetchImpl: fakeFetch(healthyRoutes()),
  });
  const dynamic = createTelemetryStore({
    baseUrl: BASE_URL,
    origin: 'dynamic',
    fetchImpl: fakeFetch(healthyRoutes()),
  });
  await Promise.all([scatter.pollOnce(), dynamic.pollOnce()]);

  const a = scatter.getSnapshot().fields;
  const b = dynamic.getSnapshot().fields;
  assert.equal(a['queue.depth'].origin, 'scatter');
  assert.equal(b['queue.depth'].origin, 'dynamic');
  // Including values that are absent -- attribution is not only for numbers.
  assert.equal(a['kv.usage'].origin, 'scatter');
  assert.equal(b['kv.usage'].origin, 'dynamic');

  // Two stores must not share state; that would silently mix the servers.
  assert.notEqual(a, b);
});

test('the two stores poll independently and do not share a snapshot', async () => {
  const scatter = createTelemetryStore({
    baseUrl: BASE_URL,
    origin: 'scatter',
    fetchImpl: fakeFetch(healthyRoutes()),
  });
  const dynamic = createTelemetryStore({
    baseUrl: BASE_URL,
    origin: 'dynamic',
    // The whole server is down, not just one endpoint: the store only reports
    // UNREACHABLE when no ungated endpoint answers.
    fetchImpl: async () => {
      throw new Error('offline');
    },
  });
  await Promise.all([scatter.pollOnce(), dynamic.pollOnce()]);

  // One server going down must not blank the other's panels.
  assert.equal(scatter.getSnapshot().connection.state, CONNECTION_STATES.CONNECTED);
  assert.equal(dynamic.getSnapshot().connection.state, CONNECTION_STATES.UNREACHABLE);
  assert.equal(scatter.getSnapshot().fields['queue.depth'].state, FIELD_STATES.MEASURED);
});

test('the prefix-reuse rate is not published, whatever the server sends', async () => {
  // WAS: "the hit rate is suppressed whatever the denominator does" -- two arms
  // asserting the field RESOLVED to UNAVAILABLE at 0 and at 11 lookups. That
  // test guarded a CLASSIFICATION. The field no longer has one: it is cut at
  // the register and banned in NEVER_BIND, so the correct assertion is now
  // ABSENCE -- which is precisely the thing a suppressed-value test cannot see,
  // because a suppressed field and an absent one both render no number.
  //
  // The wire is made HOSTILE on purpose. It sends a rate under BOTH the name
  // the register used to bind AND the name the server actually ships today. A
  // cut that removed the catalogue row while some other path still read the
  // body would surface one of them and fail here.
  for (const lookups of [0, 11]) {
    const store = createTelemetryStore({
      baseUrl: BASE_URL,
      origin: 'dynamic',
      fetchImpl: fakeFetch(
        healthyRoutes({
          [ENDPOINTS.DEBUG_KV]: {
            body: {
              ...debugKvBody(),
              prefix_cache_hit_rate: 0.94,
              generation_prefix_reuse_rate: 0.94,
            },
          },
          [ENDPOINTS.METRICS]: { text: metricsBody({ lookups }) },
        }),
      ),
    });
    await store.pollOnce();
    const fields = store.getSnapshot().fields;

    // POSITIVE CONTROL FIRST. Every assertion below is an absence, and an empty
    // snapshot satisfies all of them. This proves the poll ran and built fields.
    assert.equal(
      fields['queue.depth'].state,
      FIELD_STATES.MEASURED,
      `the store built no fields at lookups=${lookups}, so the absences below prove nothing`,
    );

    assert.equal(fields['prefix_cache.hit_rate'], undefined, `lookups=${lookups}`);

    // The property rather than the one key: any prefix rate, under any
    // spelling, reaching the snapshot is the defect.
    const rateish = Object.keys(fields).filter((key) => /prefix/i.test(key) && /rate/i.test(key));
    assert.deepEqual(rateish, [], `a prefix rate reached the snapshot at lookups=${lookups}`);
  }
});

test('the cut is enforced by a ban on the name that actually ships', () => {
  // WAS: "the zero-denominator correction is retained as a dormant second
  // line", which asserted suppressUndefinedHitRate() stayed present-but-dormant
  // so that a future reclassification could not silently re-open the 0/0
  // defect. That guard is now deleted and this replaces it with a STRONGER
  // invariant: a banned field cannot be reclassified at all, so there is no
  // reclassification for a dormant guard to catch.
  //
  // AND THE POINT OF THIS TEST IS THE SPELLING. The register used to bind
  // `prefix_cache_hit_rate`. That name is on NO json route any more -- the
  // server renamed it -- so a ban inherited against the dead spelling would
  // pass every check in this suite while protecting nothing. The ban has to
  // name the field the server actually sends today.
  assert.equal(
    PROVENANCE['prefix_cache.hit_rate'],
    undefined,
    'the rate is back in the register; it was cut at the registry, not merely reclassified',
  );

  const banned = NEVER_BIND.filter((entry) => entry.field === 'generation_prefix_reuse_rate');
  assert.equal(banned.length, 1, 'the LIVE wire name must be the banned one, not the dead spelling');
  assert.equal(banned[0].endpoint, ENDPOINTS.DEBUG_KV);
  assert.match(
    banned[0].why,
    /runtime\.rs:\d+/,
    'the ban must cite the engine branch where a counted reuse materialises no KV, ' +
      'because that is the reason the rate survives its own rename as a false number',
  );
});

test('no prefix-cache hit field is MEASURED on any origin', async () => {
  // The property, not the three named keys. The last time this defect was
  // fixed per-key, the override existed on the /metrics copy and panels read
  // the /v1/debug/kv copy -- the test passed while the page lied. A new hit
  // field added tomorrow, from either endpoint, fails here without edits.
  const offenders = [];
  for (const [key, raw] of Object.entries(PROVENANCE)) {
    if (!/hit/i.test(key) || !/prefix/i.test(key)) continue;
    for (const origin of ['dynamic', 'scatter']) {
      const entry = provenanceFor(key, origin);
      if (entry.classification === 'MEASURED') offenders.push(`${key} (${origin})`);
    }
  }
  assert.deepEqual(
    offenders,
    [],
    'a prefix-cache hit field is classified MEASURED. The counter scores one hit ' +
      'for ANY nonzero token match (metrics.rs:232-237), so it reads the same with ' +
      'and without reuse:\n' + offenders.join('\n'),
  );
});

// ---------------------------------------------------------------------------
// Structural guard, added after a browser check found what these tests missed.
//
// `prefix_cache.hits` rendered as a measured 0 on the SCATTER server. The
// byOrigin override existed -- but only on the /metrics copy of the metric, and
// panels read the /v1/debug/kv copy. The per-key test passed while the page
// lied, because it asserted the wrong key.
//
// So these assert a PROPERTY over the whole table rather than named keys: any
// prefix-cache field, wherever it is sourced from and whenever it is added,
// must be structurally bypassed on the batching server.
// ---------------------------------------------------------------------------

test('EVERY prefix-cache field is not-applicable on the scatter server', () => {
  const keys = allFieldKeys().filter((key) => key.includes('prefix_cache'));
  assert.ok(keys.length >= 4, 'expected the table to carry several prefix-cache fields');

  for (const key of keys) {
    const entry = provenanceFor(key, 'scatter');
    assert.notEqual(
      entry.classification,
      'MEASURED',
      `${key} is MEASURED on the scatter server, whose batching path never consults the ` +
        'prefix cache. Its zero would render as a measurement.',
    );
    assert.ok(
      entry.reason && entry.reason.length > 20,
      `${key} must explain WHY it is unavailable on the scatter server`,
    );
  }
});

test('no field claims to count prefix cache lookups', () => {
  // metrics.rs:132-134 increments that counter on every completed generation,
  // consulted cache or not. The upstream name is wrong; ours must not repeat it.
  for (const key of allFieldKeys()) {
    const label = provenanceFor(key, 'dynamic').label ?? '';
    assert.equal(
      /lookup/i.test(label),
      false,
      `${key} is labelled "${label}", but the underlying counter counts generations, ` +
        'not cache lookups.',
    );
  }
});

test('slow endpoints respect their declared cadences', async () => {
  // This test measures request frequency only. It does NOT establish AC33's
  // "<2% decode overhead" requirement, which remains UNMEASURED and requires
  // repeated telemetry-on/off runs under load with an equivalence CI.
  let clock = 1_000_000;
  const counts = {};
  const routes = healthyRoutes();
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    now: () => clock,
    fetchImpl: async (url) => {
      const path = new URL(url).pathname;
      counts[path] = (counts[path] ?? 0) + 1;
      return fakeFetch(routes)(url);
    },
  });

  // Four polls, one second of wall clock at the dashboard's 250 ms cadence.
  for (let i = 0; i < 4; i += 1) {
    await store.pollOnce();
    clock += 250;
  }

  assert.equal(counts[ENDPOINTS.STATUS], 4, 'live counters poll every cycle');
  assert.equal(counts[ENDPOINTS.DEBUG_KV], 4, 'batch/KV occupancy is the live signal');
  assert.equal(counts[ENDPOINTS.METRICS], 2, '/metrics is the largest payload; 500 ms');
  assert.equal(counts[ENDPOINTS.DEBUG_CONFIG], 1, 'model config cannot change');
  assert.equal(counts[ENDPOINTS.RESOURCES], 1, 'resource limits are configuration');
  assert.equal(counts[ENDPOINTS.MODELS], 1, 'the loaded model set changes rarely');
  // The block table is the largest payload the page can ask for -- a window of
  // up to 1024 pages across four parallel arrays -- and it describes a grid
  // that does not turn over between frames. Pinned explicitly rather than
  // absorbed into the total, so a future cadence change to it is a visible
  // diff here instead of a number that quietly moved.
  assert.equal(counts[ENDPOINTS.DEBUG_KV_BLOCKS], 1, 'the block table is the largest payload; 1 s');

});

test('a reused response never claims to be fresher than it is', async () => {
  // If a cached body took this poll's timestamp, a value fetched 30 s ago would
  // look as fresh as one fetched now, and staleness would become
  // unrepresentable for precisely the fields most likely to be stale.
  let clock = 1_000_000;
  const routes = healthyRoutes();
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    now: () => clock,
    fetchImpl: fakeFetch(routes),
  });

  await store.pollOnce();
  const firstObservedAt = store.field('server.context_length').observedAtMs;

  clock += 250;
  await store.pollOnce();

  const field = store.field('server.context_length');
  assert.equal(field.state, FIELD_STATES.MEASURED);
  assert.equal(
    field.observedAtMs,
    firstObservedAt,
    'a reused /v1/debug/config body must keep the timestamp of the request that produced it',
  );
});

test('every field is attributed to the model the SERVER named, not the one we assumed', async () => {
  // The lead requires per-panel server attribution. The honest version is one
  // the server asserts: `origin` is the client's belief about which server it
  // is talking to, and if two servers are started with their ports swapped that
  // belief becomes a confident lie while the reported model id stays true.
  const store = storeWith(healthyRoutes(), DYNAMIC);
  await store.pollOnce();

  const field = store.field('queue.depth');
  assert.equal(field.state, FIELD_STATES.MEASURED);
  assert.equal(field.originModelId, 'qwen-scatter', 'attribution comes from the server');
  assert.equal(field.sourceClass, 'server');
});

test('an unattributed number is preferred over a wrongly attributed one', async () => {
  // If the server has not told us what it is running, the correct answer is
  // "unknown", not the client's guess.
  const routes = healthyRoutes({
    [ENDPOINTS.MODELS]: { status: 404, body: { error: { message: 'nope', type: 'server_error' } } },
    [ENDPOINTS.HEALTH]: { status: 200, body: { status: 'ok' } },
  });
  const store = storeWith(routes, DYNAMIC);
  await store.pollOnce();

  assert.equal(store.field('queue.depth').originModelId, null);
});

test('a store pointed at the wrong kind of server says so', async () => {
  // The failure this catches: two servers started with their ports swapped, or
  // a shared URL whose origin parameters no longer match reality. Per-server
  // classification then INVERTS -- a structurally impossible prefix-cache zero
  // renders as a real measurement -- and nothing else in the system notices,
  // because every individual response is perfectly well formed.
  const warnings = [];
  const originalWarn = console.warn;
  console.warn = (...args) => warnings.push(args.join(' '));
  try {
    // healthyRoutes reports model "qwen-scatter"; we claim to be the dynamic one.
    const store = storeWith(healthyRoutes(), DYNAMIC);
    await store.pollOnce();
    await store.pollOnce();
  } finally {
    console.warn = originalWarn;
  }

  assert.equal(warnings.length, 1, 'warned exactly once, not once per poll');
  assert.match(warnings[0], /qwen-scatter/);
  assert.match(warnings[0], /dynamic/);
});

test('a correctly configured store is silent', async () => {
  const warnings = [];
  const originalWarn = console.warn;
  console.warn = (...args) => warnings.push(args.join(' '));
  try {
    const store = storeWith(healthyRoutes()); // defaults to the scatter origin
    await store.pollOnce();
  } finally {
    console.warn = originalWarn;
  }
  assert.deepEqual(warnings, []);
});

// --------------------------------------------------------------- stale audit
// telemetry-provenance.js is a snapshot of Rust source read at one commit, and
// the server team's entire job is to invalidate it. These lock the behaviour
// for the day that happens.

test('a stub that starts returning real data is shown, not hidden', async () => {
  // THE MIRRORED FABRICATION. Printing a hardcoded 0 as a measurement is the
  // failure this codebase is built to prevent -- but em-dashing a number that
  // has become real is the same lie inverted, and far harder to catch: it
  // looks like caution, it survives review, and nobody files a bug reporting a
  // number that is absent. They just conclude the feature does not work.
  const warnings = [];
  const originalWarn = console.warn;
  console.warn = (...args) => warnings.push(args.join(' '));
  let store;
  try {
    // tokens_per_second is NOT_PLUMBED with a declared stub of 0 (older
    // binaries send a literal 0.0). Pretend the plumbing landed.
    store = storeWith(
      healthyRoutes({
        [ENDPOINTS.STATUS]: { body: { ...statusBody(), tokens_per_second: 42.5 } },
      }),
    );
    await store.pollOnce();
  } finally {
    console.warn = originalWarn;
  }

  const field = store.field('throughput.tokens_per_second');
  assert.equal(field.state, FIELD_STATES.MEASURED, 'a real measurement must not be suppressed');
  assert.equal(field.value, 42.5);

  // ...but it must never pass as an ordinary measurement while the table
  // disagrees. The disagreement travels with the value.
  assert.ok(field.provenanceWarning, 'the value carries the contradiction');
  assert.match(field.provenanceWarning, /out of date/);
  assert.equal(warnings.length, 1, 'warns once per field, not once per poll');
  assert.deepEqual(Object.keys(store.provenanceWarnings()), ['throughput.tokens_per_second']);
});

test('a stub still returning its stub value stays suppressed and silent', async () => {
  const warnings = [];
  const originalWarn = console.warn;
  console.warn = (...args) => warnings.push(args.join(' '));
  let store;
  try {
    store = storeWith(healthyRoutes());
    await store.pollOnce();
  } finally {
    console.warn = originalWarn;
  }
  assert.equal(store.field('throughput.tokens_per_second').state, FIELD_STATES.UNAVAILABLE);
  assert.deepEqual(warnings, [], 'no false alarm at HEAD');
  assert.deepEqual(store.provenanceWarnings(), {});
});

test('a counter that legitimately rises is not mistaken for a stale audit', async () => {
  // prefix_cache_lookups is suppressed because it is MISNAMED -- metrics.rs:132-134
  // increments it on every completed generation -- not because it is pinned.
  // It rises on a healthy scatter server, and that proves nothing either way.
  const warnings = [];
  const originalWarn = console.warn;
  console.warn = (...args) => warnings.push(args.join(' '));
  let store;
  try {
    store = storeWith(
      healthyRoutes({
        [ENDPOINTS.DEBUG_KV]: { body: { ...debugKvBody(), prefix_cache_lookups: 91 } },
      }),
    );
    await store.pollOnce();
  } finally {
    console.warn = originalWarn;
  }
  assert.equal(store.field('prefix_cache.lookups').state, FIELD_STATES.NOT_APPLICABLE);
  assert.deepEqual(warnings, []);
});

test('every suppressed field can be checked against the wire, or says why not', () => {
  // The guard on the guard. A new suppressed entry added without a stubValue
  // would silently opt out of staleness detection -- reintroducing exactly the
  // blind spot this mechanism removes, one entry at a time.
  const suppressed = [];
  for (const [key, raw] of Object.entries(PROVENANCE)) {
    if (raw.derived) continue;
    for (const origin of ['scatter', 'dynamic']) {
      const entry = resolveForOrigin(raw, origin);
      const isSuppressed =
        entry.classification === 'STRUCTURALLY_BYPASSED' ||
        NEVER_MEASURED_CLASSIFICATIONS.includes(entry.classification);
      if (!isSuppressed) continue;
      // NOT_PLUMBED needs no declared stub: "the path carries nothing" is
      // already a checkable claim, and the day it carries something is exactly
      // the day the classification became false.
      //
      // A row with NO SOURCE is the one case this test cannot decide. It claims
      // no endpoint serves the field and none could, so there is no path to
      // read and no body that could ever contradict it -- `matchesStub()` has
      // nothing to compare. The check does not vanish, it MOVES: every
      // sourceless row is held against the Rust sources by
      // check-unplumbed-claims.test.js ("a row that names no endpoint is still
      // held against the server"), which fails the day the server grows a name
      // that would serve it. Deleting that file does not silently widen this
      // exemption -- the sourceless rows would then be checked by nothing, and
      // that is a deletion a reviewer can see.
      const checkable =
        entry.classification === 'NOT_PLUMBED' ||
        entry.source === null ||
        entry.source === undefined ||
        'stubValue' in entry ||
        typeof entry.isStub === 'function' ||
        Boolean(entry.unfalsifiable);
      if (!checkable) suppressed.push(`${key} (${origin})`);
    }
  }
  assert.deepEqual(
    suppressed,
    [],
    'these entries hide a value with no way to notice when it becomes real:\n' +
      suppressed.join('\n'),
  );
});

test('no reason string promises an endpoint that does not exist', () => {
  // A reason is shown to a VISITOR, so it is a promise the project has to keep.
  // /v1/debug/live was designed, referenced here, and then retired in favour of
  // /v1/status -- leaving the honesty layer directing people to a 404. An
  // honesty mechanism that misleads is worse than none, because it is trusted.
  const live = new Set(Object.values(ENDPOINTS));
  const offenders = [];
  for (const [key, entry] of Object.entries(PROVENANCE)) {
    for (const text of [entry.reason, entry.caveat]) {
      if (!text) continue;
      for (const mentioned of text.match(/\/v1\/[a-z0-9/_-]+|\/metrics|\/health/gi) ?? []) {
        if (!live.has(mentioned)) offenders.push(`${key}: "${mentioned}"`);
      }
    }
  }
  assert.deepEqual(
    offenders,
    [],
    `these reasons name endpoints this demo does not poll:\n${offenders.join('\n')}`,
  );
});

// ---------------------------------------------------------------------------
// The served-model projection.
//
// /v1/models returns a LIST, so no dotted path can express "the entry for the
// model this page is watching". Index 0 would be a guess, and the implicit
// default is the ALPHABETICALLY FIRST id -- in the two-model demo that is the
// dynamic model, so a scatter-server panel reading index 0 would describe the
// wrong model while looking entirely correct.
// ---------------------------------------------------------------------------

// THESE THREE TESTS USED TO DRIVE `server.model_path` THROUGH THE STORE. That
// row is gone -- the directory is TRUE and still unshowable, so it is a ban in
// NEVER_BIND rather than a classification -- and `projectServedModel()` went
// with it. Two of the three went red when it did. THE THIRD WENT GREEN, which
// is the one worth remembering: 'an unidentifiable served model yields no
// value rather than a guess' asserted the value was null-or-undefined, and a
// field that no longer exists is undefined. IT PASSED BECAUSE ITS SUBJECT HAD
// BEEN DELETED. A test whose assertion is satisfied by absence cannot tell you
// the difference between "correctly declined to guess" and "nothing here at
// all", so it is replaced below by one that must first prove the store is
// answering.

test('the absolute model directory is not addressable through the store at all', async () => {
  // The real shape, from a live probe of all four demo origins: on loopback
  // the server sends an operator's absolute path, username included.
  const HOME_PATH = '/Users/someone/GitHub/onnx-genai-demo/../onnx-genai/models/qwen2.5-0.5b';
  const store = storeWith(
    healthyRoutes({
      [ENDPOINTS.MODELS]: {
        body: {
          object: 'list',
          data: [
            { id: 'qwen2.5-0.5b', object: 'model', path: '/models/dynamic', is_default: true },
            { id: 'qwen-scatter', object: 'model', path: HOME_PATH, is_default: false },
          ],
        },
      },
    }),
  );

  await store.pollOnce();
  const fields = store.getSnapshot().fields;

  // POSITIVE CONTROL, DELIBERATELY FIRST. Without it this test passes on a
  // store that answered nothing at all, which is the failure mode that looks
  // most like success -- and is exactly how the test this one replaces went
  // green while its subject was being deleted.
  assert.equal(fields['server.model_id'].value, 'qwen-scatter');
  assert.equal(fields['server.model_id'].state, FIELD_STATES.MEASURED);

  // Not "no field named model_path" -- no field carrying the VALUE, whatever
  // it is called. The ban is on the bytes reaching a visitor, and renaming the
  // row must not be a way to satisfy it.
  const leaking = Object.entries(fields)
    .filter(([, field]) => typeof field.value === 'string' && field.value.includes('/'))
    .map(([key]) => key);
  assert.deepEqual(
    leaking,
    [],
    'no field may carry a filesystem path: on loopback that is the operator home directory',
  );
});

// @73e77d95's finding. The staleness branch promotes any value the table did
// not expect, and its `present` test was `!== undefined && !== null` -- which
// an empty string passes. The result would be a field rendering its label
// followed by nothing, marked `measured`, i.e. an ABSENCE WEARING A TRUSTED
// STATE. It is reachable from real server code: model_path_for_display ends in
// `unwrap_or_default()`, so a path whose file_name() is None emits exactly "".
test('an empty string is treated as an absence, not as a stale-table contradiction', async () => {
  const store = storeWith(
    healthyRoutes({
      [ENDPOINTS.STATUS]: { body: statusBody({ server: { execution_provider: '' } }) },
    }),
  );

  await store.pollOnce();
  const field = store.field('server.execution_provider');

  assert.notEqual(
    field.state,
    FIELD_STATES.MEASURED,
    'an empty string must never render as a trusted reading',
  );
  assert.ok(
    !Object.hasOwn(store.provenanceWarnings(), 'server.execution_provider'),
    'an empty string is not evidence the table went stale, so it must not accuse it',
  );
});

// A value can contradict a not-plumbed row only after it satisfies that row's
// declared wire contract. This path is string-valued; a numeric zero is not
// evidence that plumbing landed, and promoting it would reopen the type hole.
test('a wrong-typed zero from a not-plumbed string field is rejected', async () => {
  const store = storeWith(
    healthyRoutes({
      [ENDPOINTS.STATUS]: { body: statusBody({ server: { execution_provider: 0 } }) },
    }),
  );

  await store.pollOnce();

  assert.equal(store.field('server.execution_provider').state, FIELD_STATES.UNAVAILABLE);
  assert.equal(store.field('server.execution_provider').value, null);
  assert.ok(
    Object.hasOwn(store.provenanceWarnings(), 'server.execution_provider'),
    'the type mismatch must be visible to a developer',
  );
});

test('a provenance row that has gone stale is detected rather than em-dashed forever', async () => {
  // Vehicle swapped to another NOT_PLUMBED row. The property under test is the
  // STALENESS MACHINERY, never the model directory: a row that asserts its
  // path carries NOTHING must notice the day the server starts sending
  // something, instead of hiding a real measurement behind an em-dash forever.
  const store = storeWith(
    healthyRoutes({
      [ENDPOINTS.STATUS]: {
        body: statusBody({ server: { execution_provider: 'CUDAExecutionProvider' } }),
      },
    }),
  );

  await store.pollOnce();

  const warnings = store.provenanceWarnings();
  assert.ok(
    Object.hasOwn(warnings, 'server.execution_provider'),
    'arrival of a real value must be flagged',
  );
  assert.match(warnings['server.execution_provider'], /out of date/);
  assert.equal(store.field('server.execution_provider').value, 'CUDAExecutionProvider');
});

// --- Bypassed fields are never PENDING -------------------------------------
//
// These three tests exist because the rule "a bypassed subsystem can never
// produce a number" was enforced at ONE of its three call sites. The poll path
// special-cased STRUCTURALLY_BYPASSED inline; the first-frame path and the
// server-down path consulted only NEVER_MEASURED_CLASSIFICATIONS, which omitted
// it. So on the scatter server the prefix-cache fields rendered a spinner
// promising a measurement that could never arrive on that server.
//
// The distinction being protected: PENDING is a promise that resolves itself,
// UNAVAILABLE never will. Showing a spinner for a subsystem that is not in this
// configuration's execution path tells the visitor to wait for something that
// is not coming -- and unlike a wrong number, a spinner never looks wrong.

test('a structurally bypassed field is not-applicable on the very first frame, not pending', () => {
  const store = storeWith(healthyRoutes(), { origin: 'scatter' });

  // No poll: this is the frame every visitor sees before the network answers.
  // prefix_cache.lookups, not hit_rate: the rate is MISATTRIBUTED on every
  // origin now, so it would be suppressed here for a reason that has nothing to
  // do with the structural bypass this test exists to pin.
  const field = store.field('prefix_cache.lookups');

  assert.equal(
    field.state,
    FIELD_STATES.NOT_APPLICABLE,
    'the batching path never consults the prefix cache, so no poll can fill this in',
  );
  assert.notEqual(field.state, FIELD_STATES.PENDING);
});

test('the same field IS pending on the first frame of the dynamic server', () => {
  const store = storeWith(healthyRoutes(), DYNAMIC);

  // Same field, same instant, opposite treatment -- decided by which model
  // loaded. This is the pair that a table keyed on field name alone cannot
  // represent, so it is asserted as a pair.
  assert.equal(store.field('prefix_cache.lookups').state, FIELD_STATES.PENDING);
});

test('a misattributed field is unavailable on the first frame, never pending', () => {
  // The distinction is the whole point of the five-state vocabulary: PENDING
  // promises a number is coming. A misattributed field HAS its number already
  // -- it arrives on the very first poll, correct and useless -- so promising
  // one is a lie in the one state a visitor is guaranteed to see.
  const store = storeWith(healthyRoutes(), DYNAMIC);

  const field = store.field('prefix_cache.hit_rate');
  assert.equal(field.state, FIELD_STATES.UNAVAILABLE);
  assert.notEqual(field.state, FIELD_STATES.PENDING);
  // NOT not-applicable either: this server DOES run the code path. Collapsing
  // it into the bypass state would claim the dynamic server never asks.
  assert.notEqual(field.state, FIELD_STATES.NOT_APPLICABLE);
});

test('a bypassed field keeps one state across every path that can build it', async () => {
  // The bug was three paths disagreeing about one unchanging architectural
  // fact: not-applicable after a poll, unavailable on the first frame, pending
  // when the server was down -- the answer depended only on WHEN you looked.
  //
  // The exemplar used to be `prefix_cache.hit_rate`, which has since been cut
  // at the register. It is deliberately NOT replaced with another prefix
  // counter: those are themselves under a standing ruling and would take this
  // invariant down with them the day they go. `sessions.paused` is bypassed on
  // the scatter server for an architectural reason that is not in dispute, so
  // the property outlives the argument about the prefix cache.
  const store = storeWith({}, { origin: 'scatter' });

  const firstFrame = store.field('sessions.paused').state;
  await store.pollOnce();
  const afterFailedPoll = store.field('sessions.paused');

  assert.equal(
    afterFailedPoll.state,
    FIELD_STATES.NOT_APPLICABLE,
    'a dead server does not make a bypassed subsystem start existing',
  );
  assert.equal(firstFrame, afterFailedPoll.state, 'every path must agree on one state');
  assert.doesNotMatch(
    afterFailedPoll.reason ?? '',
    /will fill in when the server returns/,
    'the down-server explanation promises arrival, which is false for a bypassed field',
  );
});

test('every non-measured classification is listed in NEVER_MEASURED_CLASSIFICATIONS', () => {
  // Guards the direction this bug came from: a classification was added to the
  // vocabulary and not to the list, so it silently defaulted to "measurable".
  // Adding a classification must be a deliberate choice about this list, and
  // forgetting must fail loudly here rather than render a spinner on stage.
  const seen = new Set();
  for (const key of allFieldKeys()) {
    for (const origin of ['scatter', 'dynamic']) {
      seen.add(resolveForOrigin(PROVENANCE[key], origin).classification);
    }
  }

  for (const classification of seen) {
    if (classification === 'MEASURED') continue;
    assert.ok(
      NEVER_MEASURED_CLASSIFICATIONS.includes(classification),
      `${classification} is in use but absent from NEVER_MEASURED_CLASSIFICATIONS, so fields ` +
        'carrying it will render as pending -- a promise of a number that is never coming',
    );
  }
});

test('the FIRST frame announces every field by its catalogue label', async () => {
  // THE FRAME NOBODY TESTED. Every assertion about labels in this suite runs
  // after `pollOnce()`, so they all describe the SECOND frame onward. The first
  // one -- the seconds between mount and the server's first answer -- built its
  // pending fields without a label at all, and `renderField` falls back to the
  // literal word "value". A screen-reader user heard "value: no samples yet"
  // for every field on the page, and heard it FIRST, before any correct label
  // existed to replace it.
  //
  // It survived because it is invisible to sighted review (the label is the
  // aria sentence, not the visible caption) and because the panel that first
  // exposed it passed an explicit `label:` override that happened to paper over
  // the gap -- while naming the wrong quantity, which is how it was found.
  //
  // Derived from the catalogue rather than enumerated: a new field added
  // without a label, or a store path that drops one, fails here.
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: async () => {
      throw new Error('the first frame must exist before any request is made');
    },
  });

  const missing = [];
  for (const key of allFieldKeys()) {
    const field = store.field(key);
    const expected = resolveForOrigin(PROVENANCE[key], null).label;
    if (field.label !== expected) {
      missing.push(`${key}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(field.label)}`);
    }
  }
  assert.deepEqual(
    missing,
    [],
    'these fields announce themselves as the bare word "value" on the first ' +
      `frame:\n  ${missing.join('\n  ')}`,
  );
  // Vacuity guard: allFieldKeys() returning nothing would pass the above.
  assert.ok(allFieldKeys().length > 30, `only ${allFieldKeys().length} keys checked`);
});

test('the OFFLINE frame keeps every catalogue label too', async () => {
  // The same defect, one frame over, and it is the frame most likely to be on
  // screen when something has gone wrong -- the moment labels matter most.
  // `agedFields` hand-rolled `{source, unit}` exactly as the first frame did,
  // so a server disappearing silently renamed every non-measured field to
  // "value" until it came back.
  const store = createTelemetryStore({
    baseUrl: BASE_URL,
    fetchImpl: async () => {
      throw new TypeError('Failed to fetch');
    },
  });

  const snapshot = await store.pollOnce();
  assert.equal(snapshot.connection.state, CONNECTION_STATES.UNREACHABLE);

  const missing = [];
  for (const key of allFieldKeys()) {
    const expected = resolveForOrigin(PROVENANCE[key], null).label;
    if (store.field(key).label !== expected) {
      missing.push(`${key}: got ${JSON.stringify(store.field(key).label)}`);
    }
  }
  assert.deepEqual(missing, [], `unlabelled while offline:\n  ${missing.join('\n  ')}`);
  assert.ok(allFieldKeys().length > 30, 'vacuity guard');
});
