// Copyright (c) Microsoft Corporation.
//
// Tests for the request deadline.
//
// THE FAILURE THIS GUARDS IS THE ONLY ONE THAT RENDERS AS A LIE. Every other
// absence we handle produces an honest state: a refused socket becomes a
// transport error, a 500 becomes an endpoint error, an unpublished field
// becomes an em-dash. A server that ACCEPTS the connection and never replies
// produced none of those — the promise never settled, the in-flight guard never
// cleared, and the page kept painting its last good numbers with their original
// timestamps forever. Not stale, not unavailable: STOPPED, while looking
// perfectly healthy.
//
// The fake below stalls the way a real hung server stalls: it accepts the
// request, returns a promise, and settles ONLY when the abort signal fires.
// A fake that ignored the signal would hang this test file rather than fail it,
// which is worth stating because that is exactly what the browser does when the
// timeout is removed.

import test from 'node:test';
import assert from 'node:assert/strict';

import { createTelemetryStore, CONNECTION_STATES, DEFAULT_REQUEST_TIMEOUT_MS } from './telemetry-store.js';

/**
 * A server that never answers. Resolves nothing; rejects only when aborted,
 * which is precisely what `fetch` does with a signal attached.
 *
 * @param {{onRequest?: () => void}} [hooks]
 */
function stallingFetch({ onRequest } = {}) {
  return (_url, options = {}) => {
    onRequest?.();
    return new Promise((_resolve, reject) => {
      const { signal } = options;
      if (!signal) {
        // No signal means the production code forgot to pass one. Fail loudly
        // instead of hanging the suite, so the defect is reported rather than
        // waited on.
        reject(new Error('fetch was called with no abort signal — the request has no deadline'));
        return;
      }
      signal.addEventListener('abort', () => {
        reject(new Error('aborted'));
      });
    });
  };
}

/** Wait for the store to publish a snapshot matching `predicate`. */
function nextMatchingSnapshot(store, predicate, timeoutMs = 3_000) {
  return new Promise((resolve, reject) => {
    const giveUp = setTimeout(() => {
      unsubscribe();
      reject(new Error('the store never published a matching snapshot — it is hung'));
    }, timeoutMs);
    const unsubscribe = store.subscribe((snapshot) => {
      if (!predicate(snapshot)) return;
      clearTimeout(giveUp);
      unsubscribe();
      resolve(snapshot);
    });
  });
}

test('a stalled server becomes a transport error instead of a frozen page', async () => {
  const store = createTelemetryStore({
    fetchImpl: stallingFetch(),
    requestTimeoutMs: 40,
    pollIntervalMs: 100,
  });

  store.start();
  const snapshot = await nextMatchingSnapshot(
    store,
    (candidate) => candidate.connection.transportError !== null,
  );
  store.stop();

  assert.match(
    snapshot.connection.transportError,
    /no response within 40 ms/,
    'the timeout did not produce its own message',
  );
  assert.match(
    snapshot.connection.transportError,
    /accepted the connection but never replied/,
    'the message does not distinguish a stall from a dead port',
  );
});

test('a stalled server does not wedge the poll loop — later cycles still run', async () => {
  // The in-flight guard is the thing that used to latch. If it never clears,
  // exactly one request is ever issued no matter how long the page is left
  // open, and the dashboard is dead with no signal.
  let requests = 0;
  const store = createTelemetryStore({
    fetchImpl: stallingFetch({ onRequest: () => (requests += 1) }),
    requestTimeoutMs: 20,
    pollIntervalMs: 100,
  });

  store.start();
  await nextMatchingSnapshot(store, (candidate) => candidate.connection.transportError !== null);
  const afterFirstCycle = requests;
  await nextMatchingSnapshot(store, () => requests > afterFirstCycle);
  store.stop();

  assert.ok(
    requests > afterFirstCycle,
    `the poll loop issued ${requests} requests and then stopped — the in-flight guard latched`,
  );
});

test('a stalled server is reported as disconnected, not as healthy stale data', async () => {
  const store = createTelemetryStore({
    fetchImpl: stallingFetch(),
    requestTimeoutMs: 20,
    pollIntervalMs: 100,
  });

  store.start();
  const snapshot = await nextMatchingSnapshot(
    store,
    (candidate) => candidate.connection.transportError !== null,
  );
  store.stop();

  assert.notEqual(
    snapshot.connection.state,
    CONNECTION_STATES.CONNECTED,
    'a server that never replies is still being reported as connected',
  );
});

test('the default deadline is many multiples of the poll interval', () => {
  // This exists to break a hang, not to police a slow server. If the two ever
  // converge, a merely-slow server starts being reported as a stalled one,
  // which is a fabrication in the opposite direction.
  assert.ok(
    DEFAULT_REQUEST_TIMEOUT_MS >= 1_000,
    `a ${DEFAULT_REQUEST_TIMEOUT_MS} ms deadline is too aggressive for a busy inference server`,
  );
});

test('a responsive server is never aborted by its own deadline', async () => {
  // The anti-vacuity control. Every assertion above is satisfied by a store
  // that fails all requests, so this pins that the deadline does not fire on
  // the happy path.
  //
  // It waits for CONNECTED specifically. `subscribe` delivers the current
  // snapshot synchronously on registration, and that first snapshot is the
  // pre-poll one with `transportError: null` already — so asserting on
  // whatever arrives first would pass without a single request being made.
  const store = createTelemetryStore({
    fetchImpl: async () => ({
      ok: true,
      status: 200,
      json: async () => ({ status: 'ok', model: 'qwen-scatter' }),
      text: async () => '',
    }),
    requestTimeoutMs: 1_000,
    pollIntervalMs: 100,
  });

  store.start();
  const snapshot = await nextMatchingSnapshot(
    store,
    (candidate) => candidate.connection.state === CONNECTION_STATES.CONNECTED,
  );
  store.stop();

  assert.equal(
    snapshot.connection.transportError,
    null,
    'a server that answered immediately was reported as a transport failure',
  );
});
