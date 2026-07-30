// Copyright (c) Microsoft Corporation.
//
// The stylesheet contract.
//
// Unstyled markup is the one dashboard bug that unit tests normally cannot see:
// every assertion about text and structure passes, and the defect only surfaces
// when a human opens a browser. Since nothing here is browser-verifiable until
// GET /demo lands, this file closes that gap by mounting every panel in every
// state we can construct, collecting the class names it actually emits, and
// requiring styles/panels.css to have a rule for each one.
//
// It also enforces the two rules that keep the visual language from drifting:
// panels may only spend design tokens the designer defined, and panels size
// themselves with @container rather than @media, because a panel is ~340px in a
// one-column layout and ~700px in a two-column one and a media query cannot
// tell those apart.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { after, before, describe, it } from 'node:test';

import { flushAnimationFrames, installFakeDom } from './testing/fake-dom.js';
import { createFakeStore, measured, series } from './testing/fake-store.js';

const CSS_PATH = fileURLToPath(new URL('../styles/panels.css', import.meta.url));
const css = readFileSync(CSS_PATH, 'utf8');

let uninstallDom;
before(() => {
  uninstallDom = installFakeDom();
});
after(() => uninstallDom());

const PANEL_FILES = [
  'throughput.js',
  'scheduling.js',
  'kv-memory.js',
  'prefix-cache.js',
  'requests.js',
  'system.js',
];

const panels = [
  await import('./throughput.js'),
  await import('./scheduling.js'),
  await import('./kv-memory.js'),
  await import('./prefix-cache.js'),
  await import('./requests.js'),
  await import('./system.js'),
];

/**
 * Classes the shell owns. Panels are allowed to emit them — they are part of
 * the DOM contract in demo-ux.md §3.4 — but they are styled in the shell's
 * stylesheet, not ours, so this file must not demand a rule for them.
 * @type {ReadonlySet<string>}
 */
const SHELL_OWNED = new Set(['panel__body', 'visually-hidden', 'sr-only']);

/**
 * Every class name emitted anywhere in a rendered subtree.
 * @param {any} node
 * @param {Set<string>} into
 * @returns {Set<string>}
 */
function collectClasses(node, into = new Set()) {
  if (node && node.classList && typeof node.classList.values === 'function') {
    for (const name of node.classList.values()) {
      into.add(name);
    }
  }
  for (const child of node?.children ?? []) {
    collectClasses(child, into);
  }
  return into;
}

/**
 * A store rich enough to drive every panel down its measured path.
 *
 * These keys are not decorative: they are the exact paths the panels request,
 * extracted from the sources. A fixture that misses a path still passes the
 * class-coverage assertion — by silently rendering the unavailable branch — so
 * a drifted fixture would quietly stop testing the measured markup it claims
 * to test. `assertFixtureCoversPanels` below pins them together.
 *
 * @returns {any}
 */
function fullStore() {
  const fields = {};
  for (const [path, field] of Object.entries(MEASURED_FIELDS)) {
    fields[path] = field;
  }
  return createFakeStore({
    fields,
    series: {
      'throughput.aggregate_tok_s': series([[0, 90], [400, 94], [800, 98.7]]),
      'queue.depth': series([[0, 2], [400, 4], [800, 5]]),
      'batch.active_size': series([[0, 1], [400, 2], [800, 3]]),
      'kv.pages_used': series([[0, 200], [400, 260], [800, 318]]),
      'prefix.hit_rate': series([[0, 0.1], [400, 0.25], [800, 0.3]]),
    },
    requests: [
      { id: 'r-1', seq: 0, state: 'streaming', sentAtMs: 0, ttftMs: 310, tokens: 40 },
      { id: 'r-2', seq: 1, state: 'done', sentAtMs: 10, ttftMs: 290, tokens: 128 },
      { id: 'r-3', seq: 2, state: 'error', sentAtMs: 20, error: 'upstream closed' },
      { id: 'r-4', seq: 3, state: 'cancelled', sentAtMs: 30 },
      { id: 'r-5', seq: 4, state: 'sent', sentAtMs: 40 },
    ],
  });
}

/** Latency rows are built from prefix × percentile, so the fixture is too. */
const LATENCY_PREFIXES = [
  'latency.ttft_client',
  'latency.ttft_server',
  'latency.itl_client',
  'latency.tpot_client',
  'latency.e2e_server',
];

/** @type {Record<string, object>} */
const MEASURED_FIELDS = (() => {
  const fields = {
    'throughput.aggregate_tok_s': measured(98.7, { source: 'derived', unit: 'tok/s' }),
    'scenario.makespan_ms': measured(7400, { source: 'client', unit: 'ms' }),

    'scheduler.running': measured(3, { source: 'server' }),
    'scheduler.waiting': measured(5, { source: 'server' }),
    'scheduler.max_batch': measured(4, { source: 'server' }),
    'scheduler.preemptions_total': measured(1, { source: 'server' }),
    'queue.depth': measured(5, { source: 'server' }),
    'queue.depth_peak': measured(9, { source: 'server' }),
    'batch.active_size': measured(3, { source: 'server' }),
    'admission.slots_available': measured(1, { source: 'server' }),
    'admission.rejections': measured(0, { source: 'server' }),

    'kv.pages_used': measured(318, { source: 'server' }),
    'kv.pages_total': measured(512, { source: 'server' }),
    'kv.pages_shared': measured(46, { source: 'server' }),
    'kv.block_size': measured(16, { source: 'server' }),
    'kv.slots_filled': measured(4900, { source: 'server' }),
    'kv.slot_capacity': measured(8192, { source: 'server' }),
    'kv.allocations': measured(1204, { source: 'server' }),
    'kv.frees': measured(886, { source: 'server' }),
    'kv.allocation_failures': measured(2, { source: 'server' }),
    'kv.hot_evictions': measured(7, { source: 'server' }),
    'kv.prefix_evictions': measured(3, { source: 'server' }),
    'kv.refcount_histogram': measured({ 1: 260, 2: 44, 3: 2 }, { source: 'server' }),
    'kv.tiers': measured([{ name: 'hot', pages: 300 }, { name: 'cold', pages: 18 }], {
      source: 'server',
    }),

    'prefix_cache.hits': measured(12, { source: 'server' }),
    'prefix_cache.lookups': measured(40, { source: 'server' }),
    'prefix.hit_rate': measured(0.3, { source: 'derived' }),
    'prefix.tokens_reused': measured(880, { source: 'server' }),
    'prefix.prefill_tokens_skipped': measured(880, { source: 'server' }),
    'prefix.time_saved_ms': measured(1200, { source: 'derived', unit: 'ms' }),
    'prefix.evictions': measured(3, { source: 'server' }),

    'server.model_id': measured('qwen2.5-0.5b-scatter-v2', { source: 'server' }),
    'server.model_path': measured('models/qwen2.5-0.5b-scatter-v2', { source: 'server' }),
    'server.context_length': measured(32768, { source: 'server' }),
    'server.execution_provider': measured('CPU', { source: 'server' }),
    'server.decode_backend': measured('scatter', { source: 'server' }),
    'server.quantization': measured('none', { source: 'server' }),
    'server.version': measured('0.1.0', { source: 'server' }),
    'server.uptime_ms': measured(612_000, { source: 'server', unit: 'ms' }),
    'sessions.active': measured(0, { source: 'server' }),

    'resources.vram_limit_bytes': measured(1_073_741_824, { source: 'server' }),
    'resources.kv_budget_bytes': measured(4_294_967_296, { source: 'server' }),
    'resources.host_ram_used': measured(9_000_000_000, { source: 'server' }),
    'resources.host_ram_limit': measured(34_359_738_368, { source: 'server' }),
    'resources.disk_spill_bytes': measured(0, { source: 'server' }),

    'client.poll_rtt_ms': measured(9, { source: 'client', unit: 'ms' }),
    'client.poll_interval_ms': measured(250, { source: 'client', unit: 'ms' }),
    'client.dropped_frames': measured(0, { source: 'client' }),
  };
  for (const prefix of LATENCY_PREFIXES) {
    fields[`${prefix}_p50`] = measured(310, { source: 'server', unit: 'ms' });
    fields[`${prefix}_p95`] = measured(880, { source: 'server', unit: 'ms' });
    fields[`${prefix}_max`] = measured(1400, { source: 'server', unit: 'ms' });
  }
  return fields;
})();

/**
 * A store where nothing is measurable — the state a first-time visitor is most
 * likely to see, and the one that exercises every unavailable treatment. The
 * fake store already answers unknown paths with an unavailable field, so an
 * empty spec is exactly a server that can measure nothing.
 * @returns {any}
 */
function barrenStore() {
  return createFakeStore({ requests: [] });
}

describe('stylesheet contract', () => {
  it('exercises every field path the panels actually request', () => {
    // The class-coverage tests below are only meaningful if the fixture drives
    // the measured branch. A missing path renders as unavailable and would let
    // those tests pass while silently testing the wrong markup, so the fixture
    // is checked against the paths extracted from the panel sources.
    const requested = new Set();
    for (const file of PANEL_FILES) {
      const source = readFileSync(fileURLToPath(new URL(`./${file}`, import.meta.url)), 'utf8');
      for (const match of source.matchAll(/\bfield\('([a-z0-9_.]+)'\)/g)) {
        requested.add(match[1]);
      }
      // Latency rows are assembled as `${prefix}_${percentile}`; expand them.
      for (const match of source.matchAll(/prefix: '([a-z0-9_.]+)'/g)) {
        for (const percentile of ['p50', 'p95', 'max']) {
          requested.add(`${match[1]}_${percentile}`);
        }
      }
    }

    const missing = [...requested].filter((path) => !(path in MEASURED_FIELDS)).sort();
    assert.ok(requested.size > 30, `extracted only ${requested.size} paths — the regex has drifted`);
    assert.deepEqual(
      missing,
      [],
      `the fixture never supplies these paths, so they render unavailable: ${missing.join(', ')}`,
    );
  });

  it('styles every class the panels emit when data is flowing', () => {
    const emitted = new Set();
    for (const panel of panels) {
      const root = document.createElement('div');
      root.classList.add('panel__body');
      const handle = panel.mount(root, fullStore());
      flushAnimationFrames();
      collectClasses(root, emitted);
      handle.unmount();
    }

    const unstyled = [...emitted]
      .filter((name) => !SHELL_OWNED.has(name))
      .filter((name) => !css.includes(`.${name}`))
      .sort();

    assert.deepEqual(
      unstyled,
      [],
      `panels emit classes with no rule in styles/panels.css: ${unstyled.join(', ')}`,
    );
  });

  it('styles every class the panels emit when nothing is measurable', () => {
    // The unavailable path emits markup the measured path never does, so it
    // needs its own sweep. This is the state the demo is most likely to be seen
    // in, which makes unstyled markup here more costly, not less.
    const emitted = new Set();
    for (const panel of panels) {
      const root = document.createElement('div');
      root.classList.add('panel__body');
      const handle = panel.mount(root, barrenStore());
      flushAnimationFrames();
      collectClasses(root, emitted);
      handle.unmount();
    }

    const unstyled = [...emitted]
      .filter((name) => !SHELL_OWNED.has(name))
      .filter((name) => !css.includes(`.${name}`))
      .sort();

    assert.deepEqual(
      unstyled,
      [],
      `unavailable-state markup is unstyled: ${unstyled.join(', ')}`,
    );
  });

  it('spends only design tokens the designer defined', () => {
    // `--og-value-slot` is deliberately undefined: demo-ux.md §4.1 introduces it
    // with a `5ch` fallback so a panel can widen a reserved slot locally without
    // the designer having to anticipate every value width.
    const declared = new Set(
      [...css.matchAll(/--og-[a-z0-9-]+/g)].map((match) => match[0]),
    );
    assert.ok(declared.size > 30, 'expected the stylesheet to actually use tokens');

    const tokensFile = readFileSync(
      fileURLToPath(new URL('../styles/tokens.css', import.meta.url)),
      'utf8',
    );
    const defined = new Set(
      [...tokensFile.matchAll(/^\s*(--og-[a-z0-9-]+)\s*:/gm)].map((match) => match[1]),
    );

    const invented = [...declared]
      .filter((token) => token !== '--og-value-slot')
      .filter((token) => !defined.has(token))
      .sort();

    assert.deepEqual(
      invented,
      [],
      `panels.css invents tokens the designer never defined: ${invented.join(', ')}`,
    );
  });

  it('sizes panels with @container, never @media', () => {
    assert.ok(css.includes('@container'), 'expected container queries');
    assert.equal(
      /@media\s*\((min|max)-width/.test(css),
      false,
      'a width media query cannot see the panel column it was placed in — use @container',
    );
  });

  it('never encodes a request state in colour alone', () => {
    // Every `.request-state--x` colour rule must be matched by a glyph in the
    // markup. Checking the rendered text is the honest version of this test:
    // asserting on CSS alone would pass even if the glyphs were removed.
    const root = document.createElement('div');
    const handle = panels[4].mount(root, fullStore());
    flushAnimationFrames();
    for (const state of ['streaming', 'done', 'error', 'cancelled', 'sent']) {
      const cell = root.findByClass(`request-state--${state}`);
      assert.ok(cell, `expected a rendered ${state} row`);
      const text = cell.textContent.trim();
      assert.ok(
        text.length > state.length,
        `${state} is styled by colour but carries no glyph or word beyond its name`,
      );
    }
    handle.unmount();
  });
});
