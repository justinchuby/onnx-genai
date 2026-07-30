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
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { after, before, describe, it } from 'node:test';

import { flushAnimationFrames, installFakeDom } from './testing/fake-dom.js';
import { createFakeStore, measured, series } from './testing/fake-store.js';

const CSS_PATH = fileURLToPath(new URL('../styles/panels.css', import.meta.url));
const css = readFileSync(CSS_PATH, 'utf8');

const PAGE_PATH = fileURLToPath(new URL('../index.html', import.meta.url));
const page = readFileSync(PAGE_PATH, 'utf8');


let uninstallDom;
before(() => {
  uninstallDom = installFakeDom();
});
after(() => uninstallDom());

const PANEL_FILES = [
  'throughput.js',
  'scheduling.js',
  'kv-memory.js',
  'requests.js',
  'system.js',
  // Renders into a panel body and requests `server.model_id`, but sits in `ui/`
  // rather than `dashboard/`. Its CLASSES were always styled; its FIELD PATH was
  // reconciled by nothing until the corpus check below derived this list from the
  // code instead of trusting it.
  '../ui/model-card.js',
];

/**
 * Comments are stripped before any path is scraped. A comment explaining why a
 * panel deliberately does NOT read a field would otherwise be scraped as a real
 * binding — the audit would flag the documentation of a decision as a defect.
 *
 * Shares its shape with the same helper in `field-keys.test.js`; the two audits
 * scrape the same corpus and must not disagree about what counts as code.
 *
 * @param {string} source
 * @returns {string}
 */
function stripJsComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1');
}

/**
 * Panels may BUILD a field path instead of writing it, but only if this audit
 * knows the rule that reconstructs it.
 *
 * A template literal is opaque to a literal-scanner, and a scanner that shrugs
 * at what it cannot read reports "nothing wrong here" and "I could not look
 * here" in the identical green. `throughput.js` has built its latency paths
 * this way the whole time; the expansion below existed but was keyed to a
 * COINCIDENCE — it happened to also scrape `prefix:`. Nothing failed if the
 * template changed shape.
 */
const DYNAMIC_PATH_SITES = new Map([
  [
    'throughput.js',
    {
      rule:
        'latency rows build `${definition.prefix}_${percentile}`; expanded from each ' +
        '`prefix:` crossed with the percentile suffixes',
      expand: (source) => {
        const paths = [];
        for (const match of source.matchAll(/prefix: '([A-Za-z0-9_.-]+)'/g)) {
          for (const percentile of ['p50', 'p95', 'max']) paths.push(`${match[1]}_${percentile}`);
        }
        return paths;
      },
    },
  ],
  [
    'model-card.js',
    {
      rule:
        'CARD_FIELDS rows are relayed to `field(key)` by the render loop; expanded ' +
        'from each `key:` entry in that table',
      expand: (source) => [...source.matchAll(/key: '([A-Za-z0-9_.-]+)'/g)].map((m) => m[1]),
    },
  ],
]);

/** A forwarder takes a path from its caller and relays it; it originates none. */
const PATH_FORWARDERS = new Map([
  ['store-adapter.js', 'generic adapter; every `key` is a parameter supplied by a panel'],
]);

/**
 * Look a panel up by its own `meta.id` rather than by position.
 *
 * This read `panels[4]` until the prefix-cache panel was cut, at which point
 * index 4 silently became a different panel and the assertion failed with
 * "expected a rendered streaming row" -- a message pointing at the requests
 * panel, which was not the thing that changed. Positional lookup into a list
 * that panels are added to and removed from is the same defect as the `modes`
 * table: a second place that has to be updated in step, with nothing to say so.
 *
 * @param {string} id
 */
function panelById(id) {
  const found = panels.find((panel) => panel.meta?.id === id);
  assert.ok(found, `no panel module with meta.id "${id}" -- was it renamed or removed?`);
  return found;
}

const panels = [
  await import('./throughput.js'),
  await import('./scheduling.js'),
  await import('./kv-memory.js'),
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
    'batch.capacity': measured(4, { source: 'server' }),
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
    'prefix_cache.tokens_reused': measured(880, { source: 'server' }),
    'prefix_cache.prefill_tokens_skipped': measured(880, { source: 'server' }),
    'prefix_cache.time_saved_ms': measured(1200, { source: 'derived', unit: 'ms' }),
    'prefix_cache.evictions': measured(3, { source: 'server' }),

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

/**
 * Remove CSS comments so a lint reads declarations rather than prose.
 *
 * @param {string} source
 * @returns {string}
 */
function stripCssComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, '');
}

describe('stylesheet contract', () => {
  it('exercises every field path the panels actually request', () => {
    // The class-coverage tests below are only meaningful if the fixture drives
    // the measured branch. A missing path renders as unavailable and would let
    // those tests pass while silently testing the wrong markup, so the fixture
    // is checked against the paths extracted from the panel sources.
    const requested = new Set();
    for (const file of PANEL_FILES) {
      const source = stripJsComments(
        readFileSync(fileURLToPath(new URL(`./${file}`, import.meta.url)), 'utf8'),
      );
      // Character class DELIBERATELY WIDER than the keys we ship: a narrow
      // `[a-z0-9_.]` SKIPS a malformed key rather than flagging it, and a
      // skipped key is scored as a clean panel.
      //
      // The trailing `\)` is gone for the same reason it was wrong to be there:
      // it made `field('x', opts)` invisible, and an invisible path is scored
      // as a clean panel exactly like a malformed one.
      for (const match of source.matchAll(/\bfield\(\s*'([A-Za-z0-9_.-]+)'/g)) {
        requested.add(match[1]);
      }
      // Paths this file BUILDS rather than writes, expanded by its DECLARED rule.
      // Keyed to the rule, not to whichever scrape happened to work: the guard
      // below goes red the day a panel builds a path no rule here explains.
      const site = DYNAMIC_PATH_SITES.get(file.replace('../ui/', ''));
      for (const path of site ? site.expand(source) : []) requested.add(path);
      // Latency rows are assembled as `${prefix}_${percentile}`; that expansion now
      // lives in DYNAMIC_PATH_SITES above, keyed to throughput.js by a stated rule.
    }

    const missing = [...requested].filter((path) => !(path in MEASURED_FIELDS)).sort();
    assert.ok(requested.size > 30, `extracted only ${requested.size} paths — the regex has drifted`);
    assert.deepEqual(
      missing,
      [],
      `the fixture never supplies these paths, so they render unavailable: ${missing.join(', ')}`,
    );
  });

  it('every declared build rule actually reconstructs paths', () => {
    // A NON-ZERO FLOOR PER RULE, not just over the total. Measured: breaking
    // throughput.js's expansion so it returned NOTHING left all thirteen tests
    // green -- the aggregate floor of 30 had enough slack to absorb fifteen
    // vanished latency paths. An expansion that silently yields nothing is the
    // same defect as a scanner that silently skips: the panel is scored clean
    // because it was never read.
    const empty = [];
    for (const file of PANEL_FILES) {
      const name = file.replace('../ui/', '');
      const site = DYNAMIC_PATH_SITES.get(name);
      if (!site) continue;
      const source = stripJsComments(
        readFileSync(fileURLToPath(new URL(`./${file}`, import.meta.url)), 'utf8'),
      );
      if (site.expand(source).length === 0) empty.push(`${name} (${site.rule})`);
    }
    assert.deepEqual(
      empty,
      [],
      `${empty.join('; ')} — the declared rule reconstructed NO path. Either the panel ` +
        'stopped building paths, or the rule no longer matches how it builds them. ' +
        'Until this is fixed those paths are reconciled against nothing.',
    );
  });

  it('fails loudly on a panel that builds a field path by a rule this audit does not know', () => {
    // The defect this reconciliation exists to prevent, aimed at the reconciliation
    // itself. The check above scrapes LITERALS. A path built at runtime is invisible
    // to it, and an invisible path is indistinguishable from an absent one -- so the
    // fixture check scores a panel it never read as a panel with nothing to fix.
    //
    // Tests for ANY non-literal first argument, not just a backtick. A bare
    // `field(key)` is exactly as opaque to a literal-scanner as `field(`${x}`)`,
    // and for the same reason; matching only the one opaque shape somebody
    // happened to think of is how the previous version stayed green.
    const unexplained = [];
    for (const file of PANEL_FILES) {
      const name = file.replace('../ui/', '');
      if (DYNAMIC_PATH_SITES.has(name) || PATH_FORWARDERS.has(name)) continue;
      const source = stripJsComments(
        readFileSync(fileURLToPath(new URL(`./${file}`, import.meta.url)), 'utf8'),
      );
      const opaque = [...source.matchAll(/\.(?:field|series)\(\s*([^'"\s)][^,)]*)/g)];
      if (opaque.length > 0) unexplained.push(`${file} (${opaque[0][1].trim()})`);
    }
    assert.deepEqual(
      unexplained,
      [],
      `${unexplained.join(', ')} build a field path this audit cannot read. Declare the ` +
        'rule in DYNAMIC_PATH_SITES so the expansion above is keyed to a STATED rule ' +
        'rather than to a coincidence, or add it to PATH_FORWARDERS if it only relays ' +
        'a caller-supplied path. Otherwise those paths are reconciled by nothing.',
    );
  });

  it('reads every file that requests a field path', () => {
    // ANTI-VACUITY, ONE LEVEL UP. Everything above is only as wide as PANEL_FILES,
    // which is hand-maintained. This requirement is derived from the CODE and never
    // from PANEL_FILES: a list checked against itself is a mirror, not an inventory,
    // and deleting an entry would delete the assertion that notices.
    //
    // This found `ui/model-card.js` -- styled, rendered, requesting a real path, and
    // outside this audit for as long as the audit has existed.
    const inCorpus = new Set(PANEL_FILES.map((file) => file.replace('../ui/', '')));
    const unread = [];
    for (const [dir, prefix] of [
      [fileURLToPath(new URL('./', import.meta.url)), ''],
      [fileURLToPath(new URL('../ui/', import.meta.url)), 'ui/'],
    ]) {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (!entry.isFile() || !entry.name.endsWith('.js') || entry.name.endsWith('.test.js')) {
          continue;
        }
        const source = stripJsComments(readFileSync(`${dir}${entry.name}`, 'utf8'));
        if (!/\.(?:field|series)\(/.test(source)) continue;
        if (inCorpus.has(entry.name) || PATH_FORWARDERS.has(entry.name)) continue;
        unread.push(`${prefix}${entry.name}`);
      }
    }
    assert.deepEqual(
      unread,
      [],
      `${unread.join(', ')} request field paths but are outside this audit's corpus, so ` +
        'the fixture is never checked against them. Add them to PANEL_FILES, or to ' +
        'PATH_FORWARDERS with the reason they originate no path.',
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
    // Scan only what the BROWSER parses. This file's comments discuss token
    // FAMILIES as prose ("the --og-na-* set"), and a raw scan reads those as
    // literal token names, so the lint fails naming tokens nobody wrote. That
    // false positive is worse than no lint: the fix that makes it pass is to
    // reword a comment, which teaches everyone to treat it as noise, and the
    // next real invented token gets waved through with it.
    const styleSource = stripCssComments(css);
    const declared = new Set(
      // Case-INSENSITIVE on purpose. A lowercase-only scan still fails on an
      // uppercase typo, but it truncates the name at the first bad character
      // and reports "--og-na-" — sending the reader hunting a token that is
      // fine. An error message that names the wrong thing is worse than a
      // terse one, because it is confidently wrong.
      [...styleSource.matchAll(/var\(\s*(--og-[a-z0-9-]+)/gi)].map((match) => match[1]),
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
    const handle = panelById('requests').mount(root, fullStore());
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

describe('the panel stylesheet is actually reachable from the page', () => {
  it('is linked by index.html', () => {
    // This is the only failure mode in the whole dashboard that produces NO
    // signal at all: an unlinked stylesheet is not a 404, not a console
    // warning, not a DevTools entry. The file is present and correct and
    // simply never requested, so every panel renders unstyled while looking
    // like a CSS bug. Hatch fills, the stale age suffix and the unavailable
    // treatment are all carried by this file, which means the honesty
    // affordances are the first thing to disappear.
    //
    // index.html belongs to the shell owner, not to me; this test does not
    // edit it, it just refuses to let the link go missing quietly.
    assert.match(
      page,
      /<link[^>]+href=["'][^"']*styles\/panels\.css["']/,
      'index.html does not link styles/panels.css, so every panel renders ' +
        'completely unstyled with no error anywhere. Fix is one line beside ' +
        'the existing tokens.css link: <link rel="stylesheet" href="./styles/panels.css" />',
    );
  });

  it('links tokens.css too, since panels.css only consumes --og-* variables', () => {
    // panels.css defines no colours of its own by design. Linked without
    // tokens.css it would render worse than unstyled: every custom property
    // resolves to nothing, so text and hatches lose their colour entirely.
    assert.match(page, /<link[^>]+href=["'][^"']*styles\/tokens\.css["']/);
  });
});

describe('page/stylesheet wiring is complete in BOTH directions', () => {
  // The two tests above name `panels.css` and `tokens.css` explicitly, which
  // fixes the orphan we actually hit and nothing else. The next orphan will be
  // a file that does not exist yet, and it will fail exactly as silently: an
  // unlinked stylesheet is not a 404, not a console warning, not a DevTools
  // entry — the file is simply never requested.
  //
  // Checking both directions closes the class rather than the instance.

  const STYLES_DIR = fileURLToPath(new URL('../styles/', import.meta.url));

  /** Every href the page links, in order, so duplicates stay visible. */
  function linkedHrefs() {
    return [...page.matchAll(/<link[^>]+href=["']([^"']+\.css)["']/g)].map((match) => match[1]);
  }

  it('links no stylesheet that is missing from disk', () => {
    // A typo'd href IS a 404, but it is a 404 for a stylesheet — the page
    // still renders, just unstyled, and nobody reads the network tab of a page
    // that looks like it loaded.
    const missing = linkedHrefs()
      .map((href) => href.replace(/^\.\//, ''))
      .filter((href) => href.startsWith('styles/'))
      .filter((href) => !existsSync(fileURLToPath(new URL(`../${href}`, import.meta.url))));

    assert.deepEqual(missing, [], `index.html links stylesheets that do not exist: ${missing}`);
  });

  it('leaves no stylesheet on disk unlinked', () => {
    const onDisk = readdirSync(STYLES_DIR).filter((name) => name.endsWith('.css'));
    assert.ok(onDisk.length >= 3, 'expected the styles directory to hold the real stylesheets');

    const linked = new Set(linkedHrefs().map((href) => href.split('/').pop()));
    const orphans = onDisk.filter((name) => !linked.has(name)).sort();

    assert.deepEqual(
      orphans,
      [],
      `these stylesheets exist but nothing loads them, so their rules silently do ` +
        `nothing: ${orphans.join(', ')}. A file that EXISTS and a file that is USED are ` +
        `different claims, and only one of them reaches the screen.`,
    );
  });

  it('links each stylesheet exactly once, so cascade order is unambiguous', () => {
    // A stylesheet linked twice wins over anything declared between its two
    // links. That is invisible in every tool and reorders the cascade in a way
    // no one would think to look for.
    const names = linkedHrefs().map((href) => href.split('/').pop());
    const duplicated = [...new Set(names.filter((n, i) => names.indexOf(n) !== i))];

    assert.deepEqual(duplicated, [], `linked more than once: ${duplicated.join(', ')}`);
  });
});

// The maps above suppress findings. That makes them the only part of this audit
// that can be edited to make it pass, and therefore the only place a reason is
// load-bearing. Same guard as `field-keys.test.js`, over this file's own lists,
// because the defect it prevents is not specific to either audit: an entry added
// without a reason is indistinguishable from a decision somebody made.
describe('this audit\'s own exemptions are decisions, not residue', () => {
  const PLACEHOLDER = /^(?:todo|tbd|fixme|n\/?a|none|unknown|\?+|-+|x+|\.+)$/i;
  const MIN_REASON = 12;

  it('every build rule and forwarder states its reason', () => {
    const bad = [];
    for (const [key, value] of [...DYNAMIC_PATH_SITES, ...PATH_FORWARDERS]) {
      const reason = typeof value === 'string' ? value : value?.rule;
      if (typeof reason !== 'string' || reason.trim() === '') bad.push(`${key} (no reason)`);
      else if (PLACEHOLDER.test(reason.trim())) bad.push(`${key} (placeholder)`);
      else if (reason.trim().length < MIN_REASON) bad.push(`${key} (reason too short)`);
    }
    assert.deepEqual(
      bad,
      [],
      `${bad.join(', ')} — an entry here suppresses a finding, so it must say WHY. ` +
        'An entry without a reason is not an exemption, it is a silenced assertion.',
    );
  });

  it('is auditing a non-empty set of exemptions', () => {
    // Without this, the assertion above is green over a map somebody emptied.
    assert.ok(DYNAMIC_PATH_SITES.size >= 2, `only ${DYNAMIC_PATH_SITES.size} build rules`);
    assert.ok(PATH_FORWARDERS.size >= 1);
  });
});
