// Copyright (c) Microsoft Corporation.
//
// The 48-path reconciliation, as a test rather than as a one-off audit.
//
// Panels ask the store for fields by string key. The store is contractually
// TOTAL: an unknown key returns an explained `unavailable` field rather than
// undefined or a throw. That is the right design — a typo must not white-screen
// the dashboard — but it has a consequence that is easy to miss:
//
//   A MISTYPED KEY AND A GENUINELY UNPLUMBED METRIC RENDER IDENTICALLY.
//
// Both produce a calm em-dash with a plausible explanation, forever, and both
// look completely correct in review and in the browser. There is no error, no
// warning and no visual difference. It is the single cheapest way for this
// dashboard to silently show nothing where it promised something.
//
// So the set of keys the panels request but the store does not publish is
// pinned here as an explicit inventory. Two failure directions, both wanted:
//
//   • A NEW unlisted key appears  -> almost certainly a typo, caught immediately
//     rather than at QA, which is where the DAG would otherwise surface it.
//   • A listed key starts being published -> the entry is stale, and the panel
//     is now rendering an em-dash over REAL DATA. That is the failure the lead
//     named when retiring the client-side classification table, and it is the
//     more dangerous of the two because everything still looks fine.

import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';

import { createTelemetryStore } from '../telemetry-store.js';

const DASHBOARD_DIR = fileURLToPath(new URL('./', import.meta.url));

/**
 * Keys the panels are known to request before anything publishes them, each
 * with the reason it is not a typo. Anything here renders an em-dash today.
 *
 * Keep this list SHRINKING. An entry that never shrinks is a panel promising a
 * metric nobody is building.
 */
const NOT_YET_PUBLISHED = Object.freeze({
  // Paged-KV block detail. Arrives on its own endpoint at a lower cadence than
  // the 4Hz gauges, because a 4096-block grid at 4Hz is roughly 1MB/s.
  'kv.allocation_failures': 'block-table endpoint, not yet landed',
  'kv.allocations': 'block-table endpoint, not yet landed',
  'kv.block_size': 'block-table endpoint, not yet landed',
  'kv.frees': 'block-table endpoint, not yet landed',
  'kv.hot_evictions': 'block-table endpoint, not yet landed',
  'kv.prefix_evictions': 'block-table endpoint, not yet landed',
  'kv.refcount_histogram': 'block-table endpoint, not yet landed',
  'kv.slot_capacity': 'block-table endpoint, not yet landed',
  'kv.slots_filled': 'block-table endpoint, not yet landed',
  'kv.tiers': 'block-table endpoint, not yet landed',

  // Prefix-cache savings. Namespaced prefix_cache.* to match the four prefix
  // metrics the store already publishes, so these bind on the day they ship
  // instead of em-dashing against a namespace nobody uses.
  'prefix_cache.evictions': 'savings metrics not yet plumbed server-side',
  'prefix_cache.prefill_tokens_skipped': 'savings metrics not yet plumbed server-side',
  'prefix_cache.time_saved_ms': 'savings metrics not yet plumbed server-side',
  'prefix_cache.tokens_reused': 'savings metrics not yet plumbed server-side',

  // Scheduler detail beyond queue.depth.
  'queue.depth_peak': 'peak tracking not yet plumbed server-side',
  'scheduler.max_batch': 'scheduler introspection not yet plumbed',
  'scheduler.preemptions_total': 'scheduler introspection not yet plumbed',
  'scheduler.running': 'scheduler introspection not yet plumbed',
  'scheduler.waiting': 'scheduler introspection not yet plumbed',

  // Latency percentiles. Client and server are deliberately separate keys: the
  // difference between them IS the finding, so they must never be merged.
  'latency.ttft_client_p50': 'percentile aggregation not yet plumbed',
  'latency.ttft_server_p50': 'percentile aggregation not yet plumbed',

  // Build and host facts.
  'resources.disk_spill_bytes': 'spill accounting not yet plumbed',
  'scenario.makespan_ms': 'supplied by the scenario runner, not the server',
  'server.decode_backend': 'build metadata not yet exposed',
  'server.quantization': 'build metadata not yet exposed',
  'server.uptime_ms': 'build metadata not yet exposed',
  'server.version': 'build metadata not yet exposed',
});

/**
 * Comments are stripped first. A comment explaining why a panel deliberately
 * does NOT read `prefix.hit_rate` would otherwise be scraped as a real binding
 * — the audit would flag the documentation of a decision as a defect.
 *
 * @param {string} source
 */
function stripComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1');
}

/** Every field/series key any panel actually requests. */
function requestedKeys() {
  const keys = new Map();
  const files = readdirSync(DASHBOARD_DIR).filter(
    (name) => name.endsWith('.js') && !name.endsWith('.test.js'),
  );

  for (const name of files) {
    const source = stripComments(readFileSync(`${DASHBOARD_DIR}${name}`, 'utf8'));
    for (const match of source.matchAll(/\.(?:field|series)\(\s*'([a-z0-9_.]+)'/g)) {
      if (!keys.has(match[1])) keys.set(match[1], name);
    }
  }
  return keys;
}

/** A server answering every endpoint, so the store publishes its full key set. */
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

/** The union of keys published across both engines. */
async function publishedKeys() {
  const published = new Set();
  for (const origin of ['scatter', 'dynamic']) {
    const store = createTelemetryStore({ origin, fetchImpl: respondingServer() });
    await store.pollOnce();
    for (const key of Object.keys(store.getSnapshot().fields)) published.add(key);
    store.stop();
  }
  return published;
}

describe('every field key a panel requests is reconciled against the store', () => {
  it('has no unexplained key — an unlisted one is almost certainly a typo', async () => {
    const published = await publishedKeys();
    const unexplained = [];

    for (const [key, file] of requestedKeys()) {
      // client.* is measured in the browser and answered by the adapter, so the
      // store is not expected to publish it and must never be blamed for it.
      if (key.startsWith('client.')) continue;
      if (published.has(key)) continue;
      if (key in NOT_YET_PUBLISHED) continue;
      unexplained.push(`${file} requests "${key}", which no server publishes`);
    }

    assert.deepEqual(
      unexplained,
      [],
      `${unexplained.join('\n')}\n\nThe store answers an unknown key with an explained ` +
        'unavailable field, so a typo renders exactly like an unplumbed metric — a calm ' +
        'em-dash, forever, with nothing to distinguish it. Either fix the key or add it to ' +
        'NOT_YET_PUBLISHED with the reason it is not a typo.',
    );
  });

  it('has no stale entry — a listed key that is now published is an em-dash over real data', async () => {
    const published = await publishedKeys();
    const arrived = Object.keys(NOT_YET_PUBLISHED).filter((key) => published.has(key));

    assert.deepEqual(
      arrived,
      [],
      `These keys are now published but are still listed as unplumbed: ${arrived.join(', ')}. ` +
        'The data has ARRIVED and the panel is still rendering an em-dash over it, which is ' +
        'the worst failure available here: it looks correct, reports nothing, and understates ' +
        'a server that got better. Remove them from NOT_YET_PUBLISHED.',
    );
  });

  it('never asks for a key under a namespace the store does not use', async () => {
    // prefix.* vs prefix_cache.* is the trap this catches. Every prefix metric
    // the store publishes is prefix_cache.*, so a panel binding prefix.foo
    // would still be em-dashing on the day prefix_cache.foo ships — and nobody
    // would notice, because it had always shown an em-dash.
    const published = await publishedKeys();
    const namespaces = new Set([...published].map((key) => key.split('.')[0]));
    namespaces.add('client');
    namespaces.add('latency');
    namespaces.add('scheduler');
    namespaces.add('scenario');

    const strays = [];
    for (const [key] of requestedKeys()) {
      const namespace = key.split('.')[0];
      if (!namespaces.has(namespace)) strays.push(`"${key}" uses unknown namespace "${namespace}"`);
    }

    assert.deepEqual(strays, [], strays.join('\n'));
  });
});
