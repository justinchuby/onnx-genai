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

function syntheticBlockTable({
  pageSize = 16,
  pagesInUse,
  pagesShared,
  refCounts,
  filledSlots,
  tiers = refCounts.map((value) => (value === null ? null : 0)),
  start = 0,
  total = pagesInUse,
  poolTotal = pagesInUse,
  pagesFree = poolTotal - pagesInUse,
  utilization = poolTotal > 0 ? pagesInUse / poolTotal : null,
  truncated = poolTotal > total,
}) {
  assert.equal(
    pagesFree,
    poolTotal - pagesInUse,
    'synthetic whole-pool free pages must agree with pool total and pages in use',
  );
  assert.equal(
    utilization,
    poolTotal > 0 ? pagesInUse / poolTotal : null,
    'synthetic whole-pool utilization must agree with pool total and pages in use',
  );
  const scanned = refCounts.length;
  return {
    model_id: 'phi-3',
    applicable: true,
    page_size: pageSize,
    pages_in_use: pagesInUse,
    pages_shared: pagesShared,
    hot_capacity: poolTotal,
    hot_evictions: 0,
    allocation_failures: 0,
    tiers: { 0: 'hot', 1: 'cold' },
    window: {
      start,
      scanned,
      observed: refCounts.filter((value) => value !== null).length,
      total,
      pool_total: poolTotal,
      truncated,
    },
    blocks: {
      page_ids: Array.from({ length: scanned }, (_, index) => start + index),
      ref_counts: refCounts,
      filled_slots: filledSlots,
      tiers,
    },
  };
}

function blockTableServer(baseServer, blockTable, requestedUrls = []) {
  return async (url) => {
    requestedUrls.push(url);
    if (new URL(url).pathname === '/v1/debug/kv/blocks') {
      return {
        ok: true,
        status: 200,
        json: async () => blockTable,
        text: async () => JSON.stringify(blockTable),
      };
    }
    return baseServer(url);
  };
}

async function pollBlockTable(blockTable) {
  const requestedUrls = [];
  const store = createTelemetryStore({
    origin: 'dynamic',
    fetchImpl: blockTableServer(respondingServer(), blockTable, requestedUrls),
  });
  await store.pollOnce();
  return { store, requestedUrls };
}

describe('KV block-window aggregation', () => {
  it('requests the maximum window and renders a complete 300-page pool as 100%', async () => {
    // SYNTHETIC: a complete global view of a fully allocated 300-page pool.
    const poolPages = 300;
    const blockTable = syntheticBlockTable({
      pagesInUse: poolPages,
      pagesShared: 0,
      refCounts: Array(poolPages).fill(1),
      filledSlots: Array(poolPages).fill(16),
    });
    const { store, requestedUrls } = await pollBlockTable(blockTable);

    const slotsFilled = store.field('kv.slots_filled');
    const slotCapacity = store.field('kv.slot_capacity');
    assert.equal(
      slotsFilled.state,
      'measured',
      `synthetic block table did not publish kv.slots_filled: ${JSON.stringify(slotsFilled)}`,
    );
    assert.equal(
      slotCapacity.state,
      'measured',
      `synthetic block table did not publish kv.slot_capacity: ${JSON.stringify(slotCapacity)}`,
    );
    assert.equal(slotsFilled.value, 4800);
    assert.equal(slotCapacity.value, 4800);
    assert.equal(slotsFilled.sourceClass, 'derived');
    assert.equal(slotCapacity.sourceClass, 'derived');
    assert.ok(
      requestedUrls.some((url) => new URL(url).searchParams.get('count') === '1024'),
      `block-table request did not ask for the 1024-page maximum: ${requestedUrls.join(', ')}`,
    );
    const computedPercent = (slotsFilled.value / slotCapacity.value) * 100;
    const adapter = adaptStore(store);
    const panel = await import('./kv-memory.js');
    const root = document.createElement('div');
    const handle = panel.mount(root, adapter);
    flushAnimationFrames();

    assert.match(root.textContent, /100%/);
    assert.equal(computedPercent, 100, 'a complete fully filled pool must compute to exactly 100%');

    handle.unmount();
    adapter.destroy();
    store.stop();
  });

  it('withholds both slot fields when only 256 of 300 pool pages were scanned', async () => {
    // SYNTHETIC: the reviewer's exact original failure shape. The first 256
    // pages contain no shared pages, while the whole pool reports 280 pages in
    // use, 37 shared, 20 free, and 93.3% utilization. The disagreement is
    // deliberate: global scalars must not be recomputed from this partial scan.
    const scannedPages = 256;
    const blockTable = syntheticBlockTable({
      pagesInUse: 280,
      pagesShared: 37,
      pagesFree: 20,
      utilization: 280 / 300,
      refCounts: Array(scannedPages).fill(1),
      filledSlots: Array(scannedPages).fill(16),
      total: 300,
      poolTotal: 300,
    });
    assert.equal(
      blockTable.blocks.ref_counts.filter((refCount) => refCount > 1).length,
      0,
      'fixture must make scanned shared pages disagree with the whole-pool scalar',
    );
    const { store } = await pollBlockTable(blockTable);

    for (const [key, expectedValue] of [
      ['kv.pages_used', 280],
      ['kv.pages_total', 300],
      ['kv.pages_shared', 37],
      ['kv.usage', 280 / 300],
    ]) {
      const field = store.field(key);
      assert.equal(field.state, 'measured', `${key} must remain a whole-pool measurement`);
      assert.equal(field.value, expectedValue, `${key} was recomputed from the partial window`);
    }
    assert.equal(
      store.field('kv.pages_total').value - store.field('kv.pages_used').value,
      20,
      'free pages must remain the whole-pool total minus whole-pool pages in use',
    );

    for (const key of ['kv.slots_filled', 'kv.slot_capacity']) {
      const field = store.field(key);
      assert.equal(field.state, 'unavailable', `${key} published a partial global view`);
      assert.match(field.reason, /256 of 300 pool pages/);
    }
    store.stop();
  });

  it('does not trust truncated=false when only 256 of 1024 pool pages were scanned', async () => {
    // SYNTHETIC: reproduces the live server's misleading flag exactly.
    // `truncated` compares pool_total with mirror total, not with scanned.
    const scannedPages = 256;
    const blockTable = syntheticBlockTable({
      pagesInUse: 400,
      pagesShared: 0,
      refCounts: Array(scannedPages).fill(1),
      filledSlots: Array(scannedPages).fill(16),
      total: 1024,
      poolTotal: 1024,
      truncated: false,
    });
    assert.equal(blockTable.window.truncated, false, 'fixture must preserve the wire lie');
    const { store } = await pollBlockTable(blockTable);

    for (const key of ['kv.slots_filled', 'kv.slot_capacity']) {
      const field = store.field(key);
      assert.equal(field.state, 'unavailable', `${key} trusted truncated=false`);
      assert.match(field.reason, /256 of 1024 pool pages/);
    }
    store.stop();
  });

  it('withholds both slot fields when a full-length window starts after page zero', async () => {
    const blockTable = syntheticBlockTable({
      pagesInUse: 4,
      pagesShared: 0,
      refCounts: [1, 1, 1, 1],
      filledSlots: [16, 16, 16, 16],
      start: 1,
      total: 5,
      poolTotal: 4,
      truncated: false,
    });
    const { store } = await pollBlockTable(blockTable);

    for (const key of ['kv.slots_filled', 'kv.slot_capacity']) {
      const field = store.field(key);
      assert.equal(field.state, 'unavailable', `${key} published a nonzero-start window`);
      assert.match(field.reason, /starts at page 1/);
    }
    store.stop();
  });

  it('withholds both slot fields when the pool exceeds the server maximum window', async () => {
    // SYNTHETIC: the server clamps count=1024 while the pool holds 1200 pages.
    const scannedPages = 1024;
    const blockTable = syntheticBlockTable({
      pagesInUse: 1200,
      pagesShared: 0,
      refCounts: Array(scannedPages).fill(1),
      filledSlots: Array(scannedPages).fill(16),
      total: 1024,
      poolTotal: 1200,
      truncated: true,
    });
    const { store } = await pollBlockTable(blockTable);

    for (const key of ['kv.slots_filled', 'kv.slot_capacity']) {
      const field = store.field(key);
      assert.equal(field.state, 'unavailable', `${key} published a pool larger than MAX_WINDOW`);
      assert.match(field.reason, /1024 of 1200 pool pages/);
    }
    store.stop();
  });

  it('withholds both slot fields when any dense block array is misaligned', async () => {
    const blockTable = syntheticBlockTable({
      pagesInUse: 4,
      pagesShared: 0,
      refCounts: [1, 1, 1, 1],
      filledSlots: [16, 16, 16, 16],
      total: 4,
      poolTotal: 4,
    });
    blockTable.blocks.tiers = [0, 0, 0];
    const { store } = await pollBlockTable(blockTable);

    for (const key of ['kv.slots_filled', 'kv.slot_capacity']) {
      const field = store.field(key);
      assert.equal(field.state, 'unavailable', `${key} published misaligned dense arrays`);
      assert.match(field.reason, /block arrays do not all contain window\.scanned/);
    }
    store.stop();
  });

  it('keeps null, released-zero, and non-zero pages distinct in a complete view', async () => {
    // SYNTHETIC: imitates the dense parallel-array contract in routes/mod.rs:
    // null = never written, 0 = observed and released, positive = allocated.
    const blockTable = syntheticBlockTable({
      pagesInUse: 2,
      pagesShared: 1,
      refCounts: [null, 0, 1, 2],
      filledSlots: [null, 0, 5, 16],
      tiers: [null, 0, 0, 1],
      total: 4,
      poolTotal: 4,
    });
    const { store } = await pollBlockTable(blockTable);

    assert.equal(store.field('kv.slots_filled').value, 21);
    assert.equal(store.field('kv.slot_capacity').value, 32);
    assert.equal(store.field('kv.slots_filled').sourceClass, 'derived');
    assert.equal(store.field('kv.slot_capacity').sourceClass, 'derived');
    assert.deepEqual(store.field('kv.refcount_histogram').value, [
      { refcount: 0, blocks: 1 },
      { refcount: 1, blocks: 1 },
      { refcount: 2, blocks: 1 },
    ]);
    assert.deepEqual(store.field('kv.tiers').value, [
      { name: 'hot', pages: 2 },
      { name: 'cold', pages: 1 },
    ]);
    store.stop();
  });
});

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


// A SERVER THAT NO LONGER EXISTS IS STILL RUNNING ON SOMEONE'S LAPTOP.
//
// The suite above correctly stopped sending zeros for fields the current
// server omits. But "the current server" is not what the demo necessarily
// runs against: builds in circulation tonight predate the telemetry work, and
// those binaries send `tokens_per_second: 0.0` and `kv_usage: 0.0` as literal
// values on the wire.
//
// That case is the dangerous one, and removing it from the fixture above left
// it uncovered. When a NOT_PLUMBED field arrives carrying a value, the store
// reads the disagreement as "the table is stale, this field became real" and
// DISPLAYS the number — failing open, in the one direction this project
// cannot afford. The provenance entries declare `stubValue: 0` so a zero is
// recognised as the documented placeholder rather than as news.
//
// A non-zero value must still raise the staleness warning: that branch is how
// we learn the server team shipped something, and this must not gut it.
describe('an older binary that still sends the fabricated zeros', () => {
  const legacyServer = () =>
    respondingServer({
      '/v1/status': {
        healthy: true,
        model_id: 'phi-3',
        node_id: 'node-1',
        context_length: 4096,
        queue_depth: 3,
        active_sessions: 2,
        batch: { active_size: 4 },
        // The literal zeros an older build puts on the wire.
        tokens_per_second: 0.0,
        kv_usage: 0.0,
      },
    });

  for (const key of ['throughput.tokens_per_second', 'kv.usage']) {
    it(`${key} sent as a literal 0.0 is not promoted to a measurement`, async () => {
      const store = createTelemetryStore({ origin: 'scatter', fetchImpl: legacyServer() });
      await store.pollOnce();

      const field = store.field(key);
      assert.notEqual(
        field.state,
        'measured',
        `${key} arrived as a hardcoded 0.0 and was shown as a live reading`,
      );
      store.stop();
    });
  }

  it('still flags a NON-zero value, so the staleness warning keeps its teeth', async () => {
    const store = createTelemetryStore({
      origin: 'scatter',
      fetchImpl: respondingServer({
        '/v1/status': {
          healthy: true,
          model_id: 'phi-3',
          node_id: 'node-1',
          context_length: 4096,
          queue_depth: 3,
          active_sessions: 2,
          batch: { active_size: 4 },
          // A real rate would mean the server started computing this.
          tokens_per_second: 20.7,
        },
      }),
    });
    await store.pollOnce();

    const field = store.field('throughput.tokens_per_second');
    assert.equal(
      field.state,
      'measured',
      'a genuine measurement must never be suppressed by the stub declaration',
    );
    assert.equal(field.value, 20.7);
    store.stop();
  });
});
