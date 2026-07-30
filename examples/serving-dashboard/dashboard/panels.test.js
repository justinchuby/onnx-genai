// Copyright (c) Microsoft Corporation.
//
// Tests for the throughput, KV memory, requests and system panels.
//
// One file rather than five because the assertions worth writing are the same
// shape in each: does this panel refuse to render a number it was not given,
// and does it label a real number with what was actually measured. Keeping them
// together makes it obvious when one panel is missing a case the others have.

import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';

import { flushAnimationFrames, installFakeDom } from './testing/fake-dom.js';
import { createFakeStore, measured, series, unavailable } from './testing/fake-store.js';

let uninstallDom;
before(() => {
  uninstallDom = installFakeDom();
});
after(() => uninstallDom());

const throughput = await import('./throughput.js');
const kvMemory = await import('./kv-memory.js');
const requests = await import('./requests.js');
const system = await import('./system.js');

/**
 * @param {{mount: Function}} panel
 * @param {any} store
 */
function mountPanel(panel, store) {
  const root = document.createElement('div');
  const handle = panel.mount(root, store);
  flushAnimationFrames();
  return { root, handle };
}

describe('throughput panel', () => {
  it('renders the aggregate rate as DERIVED, never as the /v1/status documented zero', () => {
    // /v1/status.tokens_per_second is a hardcoded 0.0 (admin.rs:63). The honest
    // number is the client-side derivative, and its badge must say so.
    const store = createFakeStore({
      // Supplied as a RATE, not a field: the panel differentiates the measured
      // counter rather than reading the server's own tokens_per_second, which
      // is a documented zero.
      rates: {
        'metrics.tokens_generated_total': measured(98.7, { source: 'derived', unit: 'tok/s' }),
      },
      fields: {
        'throughput.tokens_per_second': measured(0, { source: '/v1/status', unit: 'tok/s' }),
      },
    });
    const { root, handle } = mountPanel(throughput, store);

    const hero = root.findByClass('hero-figure');
    assert.match(hero.textContent, /98\.7/);
    assert.equal(hero.findByClass('value__src').textContent, 'ᴰ');
    // Even offered a plausible-looking 0.0 on the documented-zero field, the
    // panel must not have reached for it.
    assert.doesNotMatch(hero.textContent, /\b0\.0\b/);
    handle.unmount();
  });

  it('never reads the server tokens_per_second field at all', async () => {
    const { readFileSync } = await import('node:fs');
    const { fileURLToPath } = await import('node:url');
    const source = readFileSync(fileURLToPath(new URL('./throughput.js', import.meta.url)), 'utf8');
    assert.equal(
      /field\('throughput\.tokens_per_second'\)/.test(source),
      false,
      'throughput.tokens_per_second is a hardcoded 0.0 — binding it would render a fabrication',
    );
  });

  it('shows client and server TTFT as separate rows rather than reconciling them', () => {
    const store = createFakeStore({
      fields: {
        'latency.ttft_client_p50': measured(310, { source: 'client', unit: 'ms' }),
        'latency.ttft_server_p50': measured(298, { source: 'server', unit: 'ms' }),
      },
    });
    const { root, handle } = mountPanel(throughput, store);

    const table = root.findByClass('latency-table');
    assert.match(table.textContent, /310 ms/);
    assert.match(table.textContent, /298 ms/);
    handle.unmount();
  });

  it('calls out a large client/server TTFT divergence instead of silently picking a winner', () => {
    const store = createFakeStore({
      fields: {
        'latency.ttft_client_p50': measured(600, { source: 'client', unit: 'ms' }),
        'latency.ttft_server_p50': measured(300, { source: 'server', unit: 'ms' }),
      },
    });
    const { root, handle } = mountPanel(throughput, store);

    assert.match(root.textContent, /differ by 100%/);
    assert.match(root.textContent, /Both numbers are real/);
    handle.unmount();
  });

  it('does not nag about divergence when the two measurements broadly agree', () => {
    const store = createFakeStore({
      fields: {
        'latency.ttft_client_p50': measured(310, { source: 'client', unit: 'ms' }),
        'latency.ttft_server_p50': measured(298, { source: 'server', unit: 'ms' }),
      },
    });
    const { root, handle } = mountPanel(throughput, store);

    assert.doesNotMatch(root.textContent, /differ by/);
    handle.unmount();
  });

  it('summarises rather than listing when there are more requests than fit', () => {
    const many = Array.from({ length: 20 }, (_, index) => ({
      id: index + 1,
      sequenceSlot: index,
      marker: '●',
      tokensPerSecond: measured(10 + index, { unit: 'tok/s' }),
    }));
    const { root, handle } = mountPanel(throughput, createFakeStore({ requests: many }));

    assert.match(root.findByClass('per-request').textContent, /\+14 more in the Requests panel/);
    handle.unmount();
  });

  it('says so plainly when no requests have run', () => {
    const { root, handle } = mountPanel(throughput, createFakeStore({ requests: [] }));

    assert.match(root.textContent, /No requests in this scenario yet/);
    handle.unmount();
  });

  it('blames the wiring, not the server, when no scenario runner is connected', () => {
    const { root, handle } = mountPanel(throughput, createFakeStore());

    assert.match(root.textContent, /No scenario is connected to this page/);
    handle.unmount();
  });
});

describe('kv-memory panel', () => {
  const kvStore = (overrides = {}) =>
    createFakeStore({
      fields: {
        'kv.pages_used': measured(318, { unit: 'blocks' }),
        'kv.pages_total': measured(512, { unit: 'blocks' }),
        'kv.block_size': measured(16, { unit: 'tokens' }),
        'kv.pages_shared': measured(96, { unit: 'blocks' }),
        'kv.slots_filled': measured(4602, { unit: 'slots' }),
        'kv.slot_capacity': measured(5088, { unit: 'slots' }),
        'kv.refcount_histogram': measured([
          { refcount: 1, blocks: 222 },
          { refcount: 2, blocks: 60 },
          { refcount: 3, blocks: 36 },
        ]),
        'kv.tiers': measured([{ name: 'cpu', pages: 512 }]),
        'kv.allocations': measured(1204, { unit: 'count' }),
        'kv.frees': measured(886, { unit: 'count' }),
        'kv.allocation_failures': measured(0, { unit: 'count' }),
        'kv.hot_evictions': measured(12, { unit: 'count' }),
        'kv.prefix_evictions': measured(2, { unit: 'count' }),
        ...overrides.fields,
      },
      capabilities: overrides.capabilities,
    });

  it('replaces the body with an actionable capability notice when introspection is off', () => {
    const store = kvStore({
      capabilities: {
        'kv-introspection': {
          available: false,
          reason: "The engine computes page statistics; this server build doesn't expose them.",
          fix: '--enable-debug-endpoints',
        },
      },
    });
    const { root, handle } = mountPanel(kvMemory, store);

    assert.ok(root.findByClass('capability-notice'), 'a hatched grid for an absent feature is noise');
    assert.match(root.textContent, /--enable-debug-endpoints/);
    assert.match(root.textContent, /Everything else on this page still works/);
    handle.unmount();
  });

  it('derives utilization from real block counts and badges it as derived', () => {
    const { root, handle } = mountPanel(kvMemory, kvStore());

    const hero = root.findByClass('panel-kv-memory__hero');
    assert.match(hero.textContent, /62\.1%/);
    assert.equal(hero.findByClass('value__src').textContent, 'ᴰ');
    handle.unmount();
  });

  it('em-dashes utilization and draws no bar when the block total is unknown', () => {
    // An empty utilization bar would read as 0% — a measurement we do not have.
    const store = kvStore({
      fields: { 'kv.pages_total': unavailable('KV page statistics are not plumbed yet.') },
    });
    const { root, handle } = mountPanel(kvMemory, store);

    const bar = root.findByClass('utilization-bar');
    assert.ok(bar.classList.contains('utilization-bar--unavailable'));
    assert.equal(bar.findByClass('utilization-bar__fill'), null, 'no fill may be drawn from no data');
    handle.unmount();
  });

  it('volunteers the cost of paging instead of hiding partial block fill', () => {
    const { root, handle } = mountPanel(kvMemory, kvStore());

    assert.match(root.textContent, /The gap is the cost of paging/);
    assert.match(root.textContent, /This is real and we show it/);
    handle.unmount();
  });

  it('states each refcount bucket in text so nothing depends on comparing bar lengths', () => {
    const { root, handle } = mountPanel(kvMemory, kvStore());

    const list = root.findByClass('refcounts');
    assert.match(list.textContent, /222/);
    assert.match(list.textContent, /60/);
    assert.match(list.textContent, /36/);
    handle.unmount();
  });

  it('never uses the phrase "paged attention", because those kernels do not exist', () => {
    const { root, handle } = mountPanel(kvMemory, kvStore());

    assert.doesNotMatch(root.textContent, /paged attention/i);
    handle.unmount();
  });
});

describe('requests panel', () => {
  const requestOf = (overrides) => ({
    id: 1,
    sequenceSlot: 0,
    marker: '●',
    state: 'done',
    sentAtOffsetMs: measured(0, { unit: 'ms', source: 'client' }),
    ttftMs: measured(310, { unit: 'ms', source: 'client' }),
    promptTokens: measured(128, { unit: 'tokens' }),
    outputTokens: measured(64, { unit: 'tokens' }),
    tokensPerSecond: measured(12.4, { unit: 'tok/s', source: 'client' }),
    kvBlocks: unavailable('KV page statistics are not plumbed yet.'),
    reusedTokens: unavailable('Prefix reuse is not reported per request.'),
    finishReason: 'stop',
    ...overrides,
  });

  it('uses only the five states a browser can actually observe', () => {
    const { root, handle } = mountPanel(
      requests,
      createFakeStore({ requests: [requestOf({ state: 'streaming' })] }),
    );

    assert.match(root.textContent, /streaming/);
    assert.doesNotMatch(
      root.textContent,
      /prefilling|decoding|admitted|queued/,
      'inferring server-side states from client timing would be invisible fabrication',
    );
    handle.unmount();
  });

  it('carries a glyph alongside the state, so state is never colour alone (AC25)', () => {
    const { root, handle } = mountPanel(requests, createFakeStore({ requests: [requestOf({})] }));

    const state = root.findByClass('request-state');
    assert.equal(state.findByClass('request-state__glyph').textContent, '✔');
    assert.equal(state.getAttribute('aria-label'), 'done');
    handle.unmount();
  });

  it('renders unmeasured per-request columns as em-dashes, not zeros', () => {
    const { root, handle } = mountPanel(requests, createFakeStore({ requests: [requestOf({})] }));

    const row = root.findByClass('requests-table__row');
    const unavailableCells = row.findAllByTag('span').filter((node) => node.textContent === '—');
    assert.ok(unavailableCells.length >= 2, 'KV blocks and reused tokens are not plumbed per request');
    handle.unmount();
  });

  it('distinguishes "no traffic yet" from "nothing is feeding this panel"', () => {
    // Both render an empty table, and coalescing them would blame the server
    // for a wiring problem on the page. A scenario runner that is connected but
    // idle is a fact about the server; one that is absent is a fact about us.
    const idle = mountPanel(requests, createFakeStore({ requests: [] }));
    assert.match(idle.root.textContent, /No requests in this scenario yet/);
    assert.doesNotMatch(idle.root.textContent, /No scenario is connected/);
    idle.handle.unmount();

    // The fake store returns null for requests when the spec omits them, which
    // is what the adapter does with no scenario runner wired up.
    const unwired = mountPanel(requests, createFakeStore({}));
    assert.match(unwired.root.textContent, /No scenario is connected to this page/);
    assert.match(
      unwired.root.textContent,
      /nothing to do with the server/,
      'the empty state must not read as an accusation against the server',
    );
    unwired.handle.unmount();
  });

  it('sorts unmeasured values to the end in both directions', () => {
    // An em-dash sorting into the middle of a numeric column reads as a value
    // between its neighbours, which is exactly the impression it must not give.
    const rows = [
      { id: 1, tokensPerSecond: measured(10) },
      { id: 2, tokensPerSecond: unavailable('not measured') },
      { id: 3, tokensPerSecond: measured(30) },
    ];

    assert.deepEqual(
      requests.sortRequests(rows, 'rate', true).map((row) => row.id),
      [1, 3, 2],
    );
    assert.deepEqual(
      requests.sortRequests(rows, 'rate', false).map((row) => row.id),
      [3, 1, 2],
    );
  });
});

describe('system panel', () => {
  const systemStore = (overrides = {}) =>
    createFakeStore({
      fields: {
        'server.model_id': measured('qwen2.5-0.5b-scatter-v2'),
        'server.context_length': measured(32_768, { unit: 'tokens' }),
        'server.execution_provider': measured('CPUExecutionProvider'),
        'server.quantization': unavailable(
          "Quantization is recorded in the model's inference metadata but not exposed by the server.",
        ),
        'resources.vram_limit_bytes': measured(1_073_741_824, { unit: 'bytes' }),
        'resources.kv_budget_bytes': measured(4_294_967_296, { unit: 'bytes' }),
        'sessions.active': measured(0, { unit: 'count' }),
        'client.poll_rtt_ms': measured(4.2, { source: 'client', unit: 'ms' }),
        ...overrides.fields,
      },
      connection: overrides.connection,
    });

  it('presents the VRAM figure as a configured ceiling, never as memory in use', () => {
    // /v1/resources publishes vram.limit and derived_kv_budget.bytes — both
    // ceilings the scheduler plans against, resolved from configuration rather
    // than queried from the device. There is no consumption figure behind them,
    // so the panel must not imply one.
    const { root, handle } = mountPanel(system, systemStore());

    assert.match(root.textContent, /VRAM limit/);
    assert.match(root.textContent, /ceiling/);
    assert.doesNotMatch(root.textContent, /GPU memory used/i);
    assert.doesNotMatch(root.textContent, /\bused\b/i);
    handle.unmount();
  });

  it('never draws a utilization bar against a budget with no measured consumption', () => {
    // A fill bar is a claim about how much of the ceiling is occupied. Nothing
    // on the wire measures that, so the bar would invent the exact number a
    // visitor would read off it — the most persuasive kind of fabrication.
    const { root, handle } = mountPanel(system, systemStore());

    const resources = root.findByClass('panel-system__resources');
    assert.equal(
      resources.findByClass('utilization-bar'),
      null,
      'the system panel drew a usage bar against a figure that measures no usage',
    );
    handle.unmount();
  });

  it('says the KV budget is derived from the VRAM limit rather than measured', () => {
    const { root, handle } = mountPanel(system, systemStore());

    const row = root
      .findByClass('panel-system__resources')
      .children.find((child) => child.textContent.includes('derived KV budget'));
    assert.ok(row, 'expected a derived KV budget row');
    assert.match(
      row.findByClass('resource-row__label').getAttribute('title'),
      /not the number nvidia-smi would show/,
    );
    handle.unmount();
  });

  it('explains that zero persistent sessions is not zero traffic', () => {
    const { root, handle } = mountPanel(system, systemStore());

    const row = root
      .findByClass('panel-system__resources')
      .children.find((child) => child.textContent.includes('persistent sessions'));
    assert.match(row.findByClass('value').getAttribute('aria-label'), /not the number of in-flight requests/);
    handle.unmount();
  });

  it('renders an unconfigured disk-spill tier as "not configured", never as 0 bytes', () => {
    const { root, handle } = mountPanel(system, systemStore());

    assert.match(root.textContent, /not configured/);
    handle.unmount();
  });

  it('reports the dashboard\u2019s own polling cost', () => {
    const { root, handle } = mountPanel(system, systemStore());

    assert.match(root.findByClass('panel-system__self').textContent, /4 ms/);
    handle.unmount();
  });

  it('shows connection state with a glyph as well as a colour', () => {
    const { root, handle } = mountPanel(
      system,
      systemStore({ connection: { state: 'offline', rttMs: 0, attempt: 3 } }),
    );

    const connection = root.findByClass('connection');
    assert.equal(connection.findByClass('connection__glyph').textContent, '○');
    assert.match(connection.getAttribute('aria-label'), /offline/);
    handle.unmount();
  });

  it('formats bytes in binary units', () => {
    assert.equal(system.formatBytes(0), '0 B');
    assert.equal(system.formatBytes(1024), '1.00 KiB');
    assert.equal(system.formatBytes(1_073_741_824), '1.00 GiB');
    assert.equal(system.formatBytes(NaN), '—');
  });
});

describe('every panel — the contract the shell relies on', () => {
  const panels = [
    ['throughput', throughput],
    ['kv-memory', kvMemory],
    ['requests', requests],
    ['system', system],
  ];

  for (const [name, panel] of panels) {
    it(`${name} declares complete meta`, () => {
      assert.equal(typeof panel.meta.id, 'string');
      assert.equal(typeof panel.meta.title, 'string');
      assert.ok(['throughput', 'scheduling', 'memory', 'cache', 'system'].includes(panel.meta.group));
      assert.ok([1, 2].includes(panel.meta.span));
      assert.ok(Object.keys(panel.meta.acronyms).length > 0, 'AC30 needs definitions to show');
    });

    it(`${name} survives a store that knows nothing, rendering em-dashes rather than crashing`, () => {
      const { root, handle } = mountPanel(panel, createFakeStore());

      assert.doesNotMatch(root.textContent, /undefined|NaN|\[object Object\]/);
      handle.unmount();
    });

    it(`${name} destroy() releases its subscription`, () => {
      const store = createFakeStore();
      const { handle } = mountPanel(panel, store);

      handle.unmount();

      assert.equal(store.subscriberCount(), 0, 'AC22: a leaked subscription is a memory leak');
    });

    it(`${name} describe() returns a sentence, not a label`, () => {
      const { handle } = mountPanel(panel, createFakeStore());

      const text = handle.describe();
      assert.ok(text.length > 20, 'describe() feeds the chart aria-label and the table view');
      assert.doesNotMatch(text, /undefined|NaN/);
      handle.unmount();
    });
  }
});
