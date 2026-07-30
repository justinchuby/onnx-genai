// Copyright (c) Microsoft Corporation.
//
// Adapter tests, run against a store that behaves like the real one.
//
// The adapter is the only place in the dashboard that manufactures numbers the
// server never sent — rates and series are computed here — so it is the place
// where a fabricated measurement is most likely to originate. These tests are
// mostly about what it REFUSES to produce.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  FIELD_STATES,
  measuredField,
  pendingField,
  staleField,
  unavailableField,
} from '../telemetry-field.js';
import { adaptStore } from './store-adapter.js';

/**
 * A stand-in with the real store's surface: field(), subscribe() delivering
 * immediately, getSnapshot(). Frozen snapshots, like the real one.
 * @param {object} [initial]
 */
function createStoreDouble(initial = {}) {
  let snapshot = freezeSnapshot(initial.fields ?? {}, initial.timestampMs ?? 1000);
  const listeners = new Set();
  return {
    field(key) {
      return snapshot.fields[key] ?? unavailableField(`No field named "${key}".`, { source: 'unknown' });
    },
    getSnapshot() {
      return snapshot;
    },
    subscribe(listener) {
      listeners.add(listener);
      listener(snapshot);
      return () => listeners.delete(listener);
    },
    /** Publish a new poll cycle. */
    poll(fields, timestampMs) {
      snapshot = freezeSnapshot(fields, timestampMs);
      for (const listener of listeners) listener(snapshot);
    },
    listenerCount: () => listeners.size,
  };
}

function freezeSnapshot(fields, timestampMs) {
  return Object.freeze({
    timestampMs,
    fields: Object.freeze({ ...fields }),
    connection: Object.freeze({ state: 'connected', origin: 'http://example.invalid' }),
  });
}

const counter = (value, at) => measuredField(value, { source: '/metrics', observedAtMs: at });

describe('store adapter — history', () => {
  it('subscribes to the store exactly once however many panels mount', () => {
    // Independent subscriptions would each record their own history and the
    // panels would disagree about what time it is.
    const store = createStoreDouble();
    const adapter = adaptStore(store);
    adapter.subscribe(() => {});
    adapter.subscribe(() => {});
    adapter.subscribe(() => {});
    assert.equal(store.listenerCount(), 1);
    adapter.destroy();
    assert.equal(store.listenerCount(), 0);
  });

  it('records a gap, not a value, when a field is unavailable', () => {
    const store = createStoreDouble({ fields: { 'kv.pages_used': counter(10, 1000) } });
    const adapter = adaptStore(store);
    store.poll({ 'kv.pages_used': unavailableField('not plumbed', { source: '/v1/status' }) }, 1250);
    store.poll({ 'kv.pages_used': counter(14, 1500) }, 1500);

    const result = adapter.series('kv.pages_used');
    assert.deepEqual(result.v, [10, 14]);
    assert.ok(result.gaps.length >= 1, 'the unavailable poll must leave a hole in the series');
    adapter.destroy();
  });

  it('records a gap for a stale field rather than replotting the old reading', () => {
    // Replotting would draw a flat line asserting nothing changed, when the
    // truth is that we stopped hearing.
    const measured = counter(10, 1000);
    const store = createStoreDouble({ fields: { 'queue.depth': measured } });
    const adapter = adaptStore(store);
    store.poll({ 'queue.depth': staleField(measured, 'poll failed') }, 1250);

    const result = adapter.series('queue.depth');
    assert.deepEqual(result.v, [10], 'the stale repeat must not appear as a second sample');
    adapter.destroy();
  });

  it('merges a long outage into one gap instead of one per poll', () => {
    const store = createStoreDouble({ fields: { 'queue.depth': counter(3, 1000) } });
    const adapter = adaptStore(store);
    for (let index = 1; index <= 10; index += 1) {
      store.poll({ 'queue.depth': unavailableField('server down', { source: '/v1/status' }) }, 1000 + index * 250);
    }
    assert.equal(adapter.series('queue.depth').gaps.length, 1);
    adapter.destroy();
  });

  it('reports an unavailable series before any sample exists, with the field reason', () => {
    const store = createStoreDouble({
      fields: { 'kv.usage': unavailableField('The server sends a hardcoded 0.0.', { source: '/v1/status' }) },
    });
    const adapter = adaptStore(store);
    const result = adapter.series('kv.usage');
    assert.equal(result.state, 'unavailable');
    assert.match(result.reason, /hardcoded 0\.0/);
    assert.deepEqual(result.v, []);
    adapter.destroy();
  });

  it('drops samples older than the requested window', () => {
    const store = createStoreDouble({ fields: { 'queue.depth': counter(1, 1000) } });
    const adapter = adaptStore(store);
    store.poll({ 'queue.depth': counter(2, 5000) }, 5000);
    store.poll({ 'queue.depth': counter(3, 9000) }, 9000);
    assert.deepEqual(adapter.series('queue.depth', 5000).v, [2, 3]);
    adapter.destroy();
  });
});

describe('store adapter — rate', () => {
  const tokens = (value, at) => ({ 'metrics.tokens_generated_total': counter(value, at) });

  it('is pending, never zero, before two samples exist', () => {
    // Zero here would be a claim that nothing is being generated.
    const store = createStoreDouble({ fields: tokens(100, 1000) });
    const adapter = adaptStore(store);
    const rate = adapter.rate('metrics.tokens_generated_total');
    assert.equal(rate.state, 'pending');
    assert.equal(rate.value, null);
    adapter.destroy();
  });

  it('differentiates a counter into a per-second rate', () => {
    const store = createStoreDouble({ fields: tokens(100, 1000) });
    const adapter = adaptStore(store);
    store.poll(tokens(150, 2000), 2000);
    const rate = adapter.rate('metrics.tokens_generated_total', { unit: 'tok/s' });
    // Bound to the exported symbol, not the literal: the measured state's wire
    // value is still being settled between CONTRACT.md and telemetry-field.js,
    // and a test that pins the spelling would break on a rename that changes
    // nothing about the behaviour it is checking.
    assert.equal(rate.state, FIELD_STATES.MEASURED);
    assert.equal(rate.value, 50);
    // The CLASS is what says "we computed this"; `source` is the ENDPOINT and is
    // null for a derived value, because no endpoint carries it. It used to hold
    // the sentinel 'derived', which is what forced panels to sniff the string.
    assert.equal(rate.sourceClass, 'derived');
    assert.equal(rate.source, null);
    assert.deepEqual(rate.derivedFrom, ['metrics.tokens_generated_total']);
    adapter.destroy();
  });

  it('declines to answer when the counter goes backwards', () => {
    // A restarted server. A negative rate is nonsense and clamping to zero
    // would claim an idleness we never observed.
    const store = createStoreDouble({ fields: tokens(100, 1000) });
    const adapter = adaptStore(store);
    store.poll(tokens(5, 2000), 2000);
    const rate = adapter.rate('metrics.tokens_generated_total');
    assert.equal(rate.state, 'pending');
    assert.notEqual(rate.value, 0);
    adapter.destroy();
  });

  it('inherits unavailability from the counter it would differentiate', () => {
    const store = createStoreDouble({
      fields: {
        'metrics.tokens_generated_total': unavailableField('not exported by this build', {
          source: '/metrics',
        }),
      },
    });
    const adapter = adaptStore(store);
    const rate = adapter.rate('metrics.tokens_generated_total');
    assert.equal(rate.state, 'unavailable');
    assert.match(rate.reason, /not exported by this build/);
    adapter.destroy();
  });

  it('inherits pending from the counter rather than reporting a rate of zero', () => {
    const store = createStoreDouble({
      fields: { 'metrics.tokens_generated_total': pendingField('first poll has not landed') },
    });
    const adapter = adaptStore(store);
    assert.equal(adapter.rate('metrics.tokens_generated_total').state, 'pending');
    adapter.destroy();
  });

  it('plots the rate, not the counter, in rateSeries', () => {
    // A monotonically rising line under a tok/s heading is a mislabelled chart,
    // and the eye trusts the shape long before it reads the axis.
    const store = createStoreDouble({ fields: tokens(0, 1000) });
    const adapter = adaptStore(store);
    store.poll(tokens(50, 2000), 2000);
    store.poll(tokens(130, 3000), 3000);

    const result = adapter.rateSeries('metrics.tokens_generated_total');
    assert.equal(result.state, 'ok');
    assert.deepEqual(result.v, [50, 80]);
    adapter.destroy();
  });

  it('reports an unavailable rate series rather than a single fabricated point', () => {
    const store = createStoreDouble({ fields: tokens(10, 1000) });
    const adapter = adaptStore(store);
    const result = adapter.rateSeries('metrics.tokens_generated_total');
    assert.equal(result.state, 'unavailable');
    assert.deepEqual(result.v, []);
    adapter.destroy();
  });
});

describe('store adapter — capability', () => {
  it('treats pending as available so panels do not flicker into existence', () => {
    const store = createStoreDouble({
      fields: { 'prefix_cache.hits': pendingField('awaiting first poll') },
    });
    const adapter = adaptStore(store);
    assert.equal(adapter.capability('prefix-cache').available, true);
    adapter.destroy();
  });

  it('is unavailable, with the field reason, when every backing key is unavailable', () => {
    const store = createStoreDouble({
      fields: {
        'kv.pages_used': unavailableField('The server sends a documented zero.', { source: '/v1/status' }),
        'kv.pages_total': unavailableField('The server sends a documented zero.', { source: '/v1/status' }),
      },
    });
    const adapter = adaptStore(store);
    const capability = adapter.capability('kv-pages');
    assert.equal(capability.available, false);
    assert.match(capability.reason, /documented zero/);
    adapter.destroy();
  });

  it('is available when any one backing key works', () => {
    const store = createStoreDouble({
      fields: {
        'kv.pages_used': unavailableField('documented zero', { source: '/v1/status' }),
        'kv.pages_total': measuredField(512, { source: '/v1/debug/kv' }),
      },
    });
    const adapter = adaptStore(store);
    assert.equal(adapter.capability('kv-pages').available, true);
    adapter.destroy();
  });

  it('assumes available for an unknown capability name', () => {
    // Failing open is right here: a typo should not silently hide a panel that
    // works, and a panel that appears and explains itself is diagnosable.
    const adapter = adaptStore(createStoreDouble());
    assert.equal(adapter.capability('not-a-capability').available, true);
    adapter.destroy();
  });
});

describe('store adapter — panel isolation', () => {
  it('keeps delivering to other panels when one subscriber throws', () => {
    const store = createStoreDouble({ fields: { 'queue.depth': counter(1, 1000) } });
    const adapter = adaptStore(store);
    const seen = [];
    const originalError = console.error;
    console.error = () => {};
    try {
      adapter.subscribe(() => {
        throw new Error('panel bug');
      });
      adapter.subscribe(() => seen.push('ok'));
      store.poll({ 'queue.depth': counter(2, 1250) }, 1250);
    } finally {
      console.error = originalError;
    }
    assert.ok(seen.length >= 2, 'the healthy panel stopped updating when its neighbour threw');
    adapter.destroy();
  });

  it('returns null requests when no scenario runner is wired up', () => {
    // Not an empty array: an empty table reads as "no traffic" when the truth
    // is "not connected to anything".
    const adapter = adaptStore(createStoreDouble());
    assert.equal(adapter.requests(), null);
    adapter.destroy();
  });

  it('exposes the connection state from the live snapshot', () => {
    const adapter = adaptStore(createStoreDouble());
    assert.equal(adapter.connection().state, 'connected');
    adapter.destroy();
  });
});

describe('derivation contagion — not-applicable must dominate', () => {
  it('never manufactures a number from a structurally-bypassed input', async () => {
    // THE BUG THIS PINS: derivedField propagated unavailable, pending and
    // stale, but had no branch for not-applicable at all. A bypassed input
    // fell through to the compute path and came back `ok` at full contrast,
    // with no badge and no reason — a confident number derived from a counter
    // nothing ever observed.
    //
    // It is not hypothetical. On the batching profile prefix_cache.* is pinned
    // to a literal 0 in batched.rs, so a derived hit RATE over it rendered as
    // a measurement of something that was never measured — the exact
    // fabrication the five-state vocabulary exists to prevent, arriving
    // through the one path that bypassed the vocabulary entirely.
    const { derivedField, notApplicableField } = await import('../telemetry-field.js');

    const bypassed = notApplicableField('This path never consults the prefix cache.', {
      label: 'Prefix cache hits',
    });
    const derived = derivedField({ 'prefix_cache.hits': bypassed }, () => 42, { unit: 'per second' });

    assert.equal(derived.state, 'not-applicable');
    assert.equal(derived.value, null, 'a bypassed derivation must carry no number at all');
    assert.match(derived.reason, /not applicable on this execution path/);
    // The upstream sentence must survive, or the panel explains nothing.
    assert.match(derived.reason, /never consults the prefix cache/);
  });

  it('lets not-applicable beat unavailable, because only one of them is final', async () => {
    // Ordering matters and is not arbitrary. `unavailable` and `pending` both
    // leave the door open — someone may plumb it, the next poll may fill it —
    // so a derivation over them may yet succeed. `not-applicable` says this
    // execution path can NEVER consult that subsystem, so the derivation can
    // never succeed either. Reporting `unavailable` here would promise future
    // work that will never happen, collapsing the one distinction that carries
    // the demo's central technical claim.
    const { derivedField, notApplicableField, unavailableField } = await import(
      '../telemetry-field.js'
    );

    const mixed = derivedField(
      {
        'prefix_cache.hits': notApplicableField('Never consulted on this path.'),
        'queue.depth': unavailableField('Not plumbed through yet.'),
      },
      () => 7,
      {},
    );

    assert.equal(mixed.state, 'not-applicable');
    assert.equal(mixed.value, null);
  });
});
