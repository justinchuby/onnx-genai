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

import { declaredKeys as scanKeys, duplicatesAmong, findLiteralOpener } from './testing/object-keys.js';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';

import { createTelemetryStore } from '../telemetry-store.js';

const DASHBOARD_DIR = fileURLToPath(new URL('./', import.meta.url));

// `ui/` requests field keys too. It was outside this audit's corpus until
// 2026-07-30, which is the failure mode this file exists to prevent, one level
// up: not a vacuous ASSERTION but a vacuous CORPUS. Every reconciliation below
// was green over a directory it never opened, and a directory nobody reads is
// indistinguishable from a directory with nothing to say.
const UI_DIR = fileURLToPath(new URL('../ui/', import.meta.url));
const PACKAGE_DIR = fileURLToPath(new URL('../', import.meta.url));

const SOURCE_DIRS = [
  ['dashboard/', DASHBOARD_DIR],
  ['ui/', UI_DIR],
];

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
  'scheduler.preemptions_total': 'scheduler introspection not yet plumbed',
  'scheduler.running': 'scheduler introspection not yet plumbed',
  'scheduler.waiting': 'scheduler introspection not yet plumbed',

  // Latency percentiles. Client and server are deliberately separate keys: the
  // difference between them IS the finding, so they must never be merged.
  //
  // ALL FIFTEEN ARE LISTED, AND THAT IS THE POINT. Until the extractor learned
  // to read `throughput.js`'s template literals, only the two `_p50` entries
  // below were visible here -- not because anyone judged the other thirteen,
  // but because a literal-scanner could not see them. Two hand-written entries
  // and thirteen invisible ones are INDISTINGUISHABLE from a reviewed
  // inventory, so this list read as a decision when it was a coincidence.
  //
  // Listing them by hand is deliberate: an exemption list generated from the
  // panel sources would exempt whatever the panels ask for, and could never go
  // red. Every line here has to be typed by someone.
  'latency.ttft_client_p50': 'percentile aggregation not yet plumbed',
  'latency.ttft_client_p95': 'percentile aggregation not yet plumbed',
  'latency.ttft_client_max': 'percentile aggregation not yet plumbed',
  'latency.ttft_server_p50': 'percentile aggregation not yet plumbed',
  'latency.ttft_server_p95': 'percentile aggregation not yet plumbed',
  'latency.ttft_server_max': 'percentile aggregation not yet plumbed',
  'latency.itl_client_p50': 'percentile aggregation not yet plumbed',
  'latency.itl_client_p95': 'percentile aggregation not yet plumbed',
  'latency.itl_client_max': 'percentile aggregation not yet plumbed',
  'latency.tpot_client_p50': 'percentile aggregation not yet plumbed',
  'latency.tpot_client_p95': 'percentile aggregation not yet plumbed',
  'latency.tpot_client_max': 'percentile aggregation not yet plumbed',
  'latency.e2e_server_p50': 'percentile aggregation not yet plumbed',
  'latency.e2e_server_p95': 'percentile aggregation not yet plumbed',
  'latency.e2e_server_max': 'percentile aggregation not yet plumbed',

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

// Percentile suffixes appended to every latency-row `prefix:`.
const LATENCY_PERCENTILES = ['p50', 'p95', 'max'];

// Panels are allowed to BUILD a key instead of writing it, but only if this
// audit knows the rule that reconstructs it. A template literal is opaque to a
// literal-scanner, and a scanner that shrugs at what it cannot read reports
// "nothing wrong here" and "I could not look here" with the identical green.
const DYNAMIC_KEY_SITES = new Map([
  [
    'dashboard/throughput.js',
    {
      rule:
        'latency rows build `${definition.prefix}_${percentile}`; reconstructed here ' +
        'from each `prefix:` crossed with LATENCY_PERCENTILES',
      extract: (source) => {
        const built = [];
        for (const match of source.matchAll(/prefix: '([A-Za-z0-9_.-]+)'/g)) {
          for (const percentile of LATENCY_PERCENTILES) built.push(`${match[1]}_${percentile}`);
        }
        return built;
      },
    },
  ],
  [
    'ui/model-card.js',
    {
      rule:
        'CARD_FIELDS rows are forwarded to `field(key)` by the render loop; ' +
        'reconstructed here from each `key:` entry in the table',
      extract: (source) => [...source.matchAll(/key: '([A-Za-z0-9_.-]+)'/g)].map((m) => m[1]),
    },
  ],
]);

// A FORWARDER takes a key from its caller and hands it to the store unchanged.
// It originates no key, so there is nothing here for this audit to enumerate --
// the keys it passes are written as literals at the CALL sites, which are in the
// corpus. This is a genuinely different category from a site that BUILDS a key,
// and collapsing the two would redden a file that is behaving correctly. A guard
// that reddens on correct work is a guard somebody switches off.
const KEY_FORWARDERS = new Map([
  ['dashboard/store-adapter.js', 'generic adapter; every `key` is a parameter supplied by a panel'],
]);

// DELIBERATELY WIDER THAN THE KEYS WE SHIP. A narrow `[a-z0-9_.]` does not
// merely miss `latency.TTFT_p50` -- it SKIPS it silently, and a skipped key is
// scored as a clean file. The malformed key is the exact input this audit
// exists to catch, and it was the one input invisible to it.
const KEY_LITERAL = /\.(?:field|series)\(\s*'([A-Za-z0-9_.-]+)'/g;

/** Panel sources, excluding tests. Names in `ui/` carry their directory. */
function panelSources() {
  const sources = [];
  for (const [prefix, dir] of SOURCE_DIRS) {
    for (const name of readdirSync(dir)) {
      if (!name.endsWith('.js') || name.endsWith('.test.js')) continue;
      sources.push([`${prefix}${name}`, stripComments(readFileSync(`${dir}${name}`, 'utf8'))]);
    }
  }
  return sources;
}

// Files that call field()/series() but request no key from the real store. Each
// needs a REASON, because "outside the corpus" and "audited and found clean" are
// otherwise written down identically.
const CORPUS_EXEMPT = new Map([
  [
    'dashboard/testing/fake-store.js',
    'test double; `this.series(...)` is a call into its own implementation, ' +
      'not a request for a published key',
  ],
]);

/** Every non-test source in the package, comment-stripped, for corpus checks. */
function packageSources() {
  const sources = [];
  const walk = (dir, prefix) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue;
      if (entry.isDirectory()) {
        walk(`${dir}${entry.name}/`, `${prefix}${entry.name}/`);
      } else if (entry.name.endsWith('.js') && !entry.name.endsWith('.test.js')) {
        sources.push([
          `${prefix}${entry.name}`,
          stripComments(readFileSync(`${dir}${entry.name}`, 'utf8')),
        ]);
      }
    }
  };
  walk(PACKAGE_DIR, '');
  return sources;
}

/** Every field/series key any panel actually requests. */
function requestedKeys() {
  const keys = new Map();

  for (const [name, source] of panelSources()) {
    for (const match of source.matchAll(KEY_LITERAL)) {
      if (!keys.has(match[1])) keys.set(match[1], name);
    }

    // Reconstruct the keys this file builds rather than writes.
    const site = DYNAMIC_KEY_SITES.get(name);
    if (!site) continue;
    for (const key of site.extract(source)) {
      if (!keys.has(key)) keys.set(key, name);
    }
  }
  return keys;
}

/** A server answering every endpoint, so the store publishes its full key set. */
function respondingServer() {
  const bodies = {
    '/health': { status: 'ok' },
    '/v1/models': { data: [{ id: 'phi-3' }] },
    // FIELD NAMES HERE ARE RECONCILED AGAINST THE RUST HANDLERS BY THE TEST
    // BELOW. This payload decides what publishedKeys() believes the server
    // emits, so a field missing HERE makes a real server field look unplumbed
    // -- which is exactly how `batch_capacity` stayed invisible while a panel
    // bound a key that never existed.
    '/v1/status': {
      healthy: true,
      model_id: 'phi-3',
      node_id: 'node-1',
      queue_depth: 3,
      active_sessions: 2,
      batch: { active_size: 4 },
      kv_usage: 0.0,
      tokens_per_second: 0.0,
      batch_utilization: 0.0,
      batch_in_flight: 4,
      batch_capacity: 8,
    },
    '/v1/debug/kv': {
      prefix_cache_hits: 0,
      prefix_cache_lookups: 17,
      kv_pages_used: 12,
      kv_pages_total: 64,
      kv_pages_shared: 3,
      prefix_hashes: 5,
    },
    // Was `{}`, which starved every server.* key sourced from this endpoint --
    // so `server.context_length` looked unplumbed while the server has served
    // it as `model_max_context` all along. The fixture had ALSO put a
    // `context_length` field on /v1/status, a name no handler emits.
    '/v1/debug/config': {
      model_id: 'phi-3',
      pipeline: 'scatter',
      max_output_tokens: 512,
      max_sessions: 32,
      max_queue_depth: 16,
      model_max_context: 4096,
    },
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

describe('the audit can see what it claims to audit', () => {
  // ANTI-VACUITY. Every reconciliation below is scored against requestedKeys().
  // A regex that drifted and matched nothing would report a spotless dashboard,
  // byte-identical to a genuinely clean run. A zero is not a measurement until
  // the instrument is proven able to return non-zero.
  it('extracts a non-zero, non-trivial set of keys', () => {
    const keys = requestedKeys();
    assert.ok(
      keys.size >= 40,
      `extracted only ${keys.size} keys — the extraction pattern has drifted and ` +
        'every reconciliation in this file is now scoring an empty set as clean',
    );
  });

  it('reconstructs the keys that panels BUILD rather than write', () => {
    const keys = requestedKeys();
    const built = [...keys].filter(([key]) => key.startsWith('latency.'));
    assert.ok(
      built.length >= 15,
      `only ${built.length} latency keys reconstructed — throughput.js builds them ` +
        'from a template literal, which no literal-scanner can see',
    );
  });

  it('fails loudly on a panel that builds keys by a rule this audit does not know', () => {
    // The defect this whole file exists to prevent, aimed at the file itself:
    // a key the scanner cannot parse must never be silently skipped. If a new
    // panel starts composing keys, this goes red until someone teaches the
    // extractor the rule -- rather than quietly auditing a smaller dashboard.
    //
    // This tests for ANY non-literal argument, not just a template literal. A
    // bare `field(key)` is exactly as opaque to a literal-scanner as `field(`${x}`)`
    // is, and for the same reason; testing only for a backtick would have caught
    // the one opaque shape somebody happened to think of.
    const unexplained = [];
    for (const [name, source] of panelSources()) {
      if (DYNAMIC_KEY_SITES.has(name) || KEY_FORWARDERS.has(name)) continue;
      const opaque = [...source.matchAll(/\.(?:field|series)\(\s*([^'"\s)][^,)]*)/g)];
      if (opaque.length > 0) unexplained.push(`${name} (${opaque[0][1].trim()})`);
    }
    assert.deepEqual(
      unexplained,
      [],
      `${unexplained.join(', ')} pass a non-literal argument to field()/series(), which ` +
        'this audit cannot read. Enumerate the generated keys: add the file to ' +
        'DYNAMIC_KEY_SITES with the rule that reconstructs them, or to KEY_FORWARDERS ' +
        'if it only relays a caller-supplied key. Otherwise those keys are audited by nothing.',
    );
  });

  // ANTI-VACUITY, ONE LEVEL UP. The assertions above are only as wide as the
  // corpus panelSources() returns. `ui/` sat outside it and every one of them
  // was green over a directory it never opened.
  //
  // This requirement is derived from the CODE, never from SOURCE_DIRS. A corpus
  // checked against its own definition is a mirror, not an inventory: deleting a
  // directory from SOURCE_DIRS would delete the assertion that notices. The
  // first version of this test did exactly that and passed its own mutation.
  it('reads every file that calls field() or series()', () => {
    const inCorpus = new Set(panelSources().map(([name]) => name));
    const unread = [];
    for (const [rel, source] of packageSources()) {
      if (!/\.(?:field|series)\(/.test(source)) continue;
      if (inCorpus.has(rel) || CORPUS_EXEMPT.has(rel)) continue;
      unread.push(rel);
    }
    assert.deepEqual(
      unread,
      [],
      `${unread.join(', ')} call field()/series() but are outside this audit's corpus, ` +
        'so the keys they request are reconciled by nothing. Add the directory to ' +
        'SOURCE_DIRS, or the file to CORPUS_EXEMPT with the reason it requests no key.',
    );
  });
});

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

// THE CHECK THAT WOULD HAVE CAUGHT TONIGHT'S DEAD BINDING, AIMED AT THE ONE
// ARTEFACT NOTHING ELSE IN THIS FILE VERIFIES: THE FAKE SERVER ITSELF.
//
// Every other test here reconciles panels against `publishedKeys()`. But
// publishedKeys() polls `respondingServer()`, which is a payload WE wrote. So
// the entire reconciliation rests on a fixture, and a fixture cannot testify
// about a server. If the payload omits a field the real server sends, that
// field looks unplumbed to every check above -- and a panel binding a
// NONEXISTENT key alongside it looks equally unplumbed, so the two become
// indistinguishable.
//
// That is precisely what happened. The payload omitted `batch_capacity`, which
// the server has been serving all along, while a panel bound
// `scheduler.max_batch`, which nothing has ever served. Both rendered a calm
// em-dash. The "this key has ARRIVED, stop em-dashing it" check could never
// fire, because the fixture never let it arrive.
//
// So this test reads the Rust handler and asserts our fake speaks the same
// vocabulary as the real one. It is the only check here whose inputs are not
// both written by us.
describe('the fake server is reconciled against the Rust handler it imitates', () => {
  it('invents no field the real /v1/status and /v1/debug/kv do not serve', async () => {
    const { readFile } = await import('node:fs/promises');
    // Read every handler + state file, not just admin.rs: the response structs
    // are split across routes/mod.rs and state.rs, and checking one file would
    // report honest fields as invented -- a false positive, which teaches the
    // team to ignore this test and is worse than not having it.
    const crate = new URL('../../../crates/onnx-genai-server/src/', import.meta.url);
    const sources = ['routes/admin.rs', 'routes/mod.rs', 'state.rs', 'metrics.rs'];
    const handler = (
      await Promise.all(sources.map((f) => readFile(new URL(f, crate), 'utf8')))
    ).join('\n');

    const fixture = respondingServer();
    const invented = [];
    for (const path of ['/v1/status', '/v1/debug/kv']) {
      const response = await fixture(`http://127.0.0.1:8123${path}`);
      for (const name of Object.keys(await response.json())) {
        // `batch` is a client-side nesting the store flattens, not a wire name.
        if (name === 'batch') continue;
        if (new RegExp(`\\b${name}\\b`).test(handler)) continue;
        invented.push(`${path} fixture sends "${name}", absent from the handler sources`);
      }
    }

    assert.deepEqual(
      invented,
      [],
      `${invented.join('\n')}\n\nThe fake server sends a field the real one does not. Every ` +
        'key check in this file polls that fake, so an invented field makes a panel binding ' +
        'it look healthy in CI and em-dash forever in front of an audience.',
    );
  });
});

describe('the provenance catalogue defines every field exactly once', () => {
  // WHY THIS EXISTS.
  //
  // `'batch.capacity'` was defined TWICE in the catalogue. A duplicate key in a
  // JS object literal is not an error: no syntax error, no warning, no lint.
  // The last definition silently wins and the first becomes dead code that
  // still reads perfectly in the file. The catalogue stated one field's
  // provenance twice, the program believed one of them, and nothing anywhere
  // said which.
  //
  // That is absent-vs-zero -- the defect this whole product refuses -- sitting
  // inside the provenance table itself. And it is invisible to every check we
  // already own, because `Object.keys()` DEDUPLICATES: each one sees a tidy 37
  // and cannot distinguish it from a tidy 37 where two entries collided.
  //
  // Which survives is not neutral. The duplicate that won cited `admin.rs:178`;
  // the one it silently killed cited the SYMBOL. So the catalogue preferred the
  // fragile citation and discarded the durable one -- the exact inverse of the
  // rule this repo runs on -- and a reader who scrolls to the good entry and
  // stops reading believes it is in force. It is not.
  //
  // The only instrument that can see this is the SOURCE, not the object.
  const PROVENANCE_PATH = fileURLToPath(new URL('../telemetry-provenance.js', import.meta.url));

  /** Top-level catalogue keys as WRITTEN, duplicates preserved. */
  function declaredKeys() {
    const source = readFileSync(PROVENANCE_PATH, 'utf8');
    // Was a regex: /^ {2}'([A-Za-z0-9_.]+)': \{/gm. It matched the shape this
    // file happens to use rather than the syntax JavaScript defines, and it was
    // BLIND to `"batch.capacity"` -- the identical key, double quoted. Proven by
    // mutation: injecting that form left this suite fully green while the
    // catalogue silently lost an entry, which is the exact defect this check
    // exists to catch, surviving inside the check itself. The reconciliation
    // control below could not catch it either, because a line the regex does
    // not match never enters the count being reconciled: both halves agreed,
    // and both were reading the same blind spot.
    //
    // The character-class history is kept because it is the same lesson one
    // size smaller: an earlier version used [a-z_.] and skipped
    // `metrics.e2e_latency`, the one key containing a digit. Each fix taught
    // the pattern one more shape. The scanner reads the grammar instead.
    return scanKeys(source, findLiteralOpener(source, 'export const PROVENANCE')).map((key) => key.name);
  }

  it('parses exactly the key set the module actually exports', async () => {
    // The anti-vacuity control, and it is not optional. A parser that matches
    // nothing finds no duplicates and reports success, which is byte-identical
    // to a clean catalogue. Reconciling against the runtime keys means the
    // detector cannot go blind without going RED.
    const { PROVENANCE } = await import('../telemetry-provenance.js');
    const runtime = Object.keys(PROVENANCE).sort();
    const declared = [...new Set(declaredKeys())].sort();

    assert.deepEqual(
      declared,
      runtime,
      'the source parser and the exported object disagree about which keys exist, so the '
        + 'duplicate check below is scanning something other than the catalogue.',
    );
  });

  it('declares no field key twice', () => {
    const source = readFileSync(PROVENANCE_PATH, 'utf8');
    const keys = scanKeys(source, findLiteralOpener(source, 'export const PROVENANCE'));
    const duplicates = duplicatesAmong(keys).map((d) => `${d.name} (lines ${d.lines.join(' and ')})`);

    assert.deepEqual(
      duplicates,
      [],
      `${duplicates.join(', ')} is defined more than once in telemetry-provenance.js. `
        + 'JS keeps the LAST definition and silently discards the earlier one, so the file '
        + 'states a provenance the program does not use. Delete the duplicate, keeping the '
        + 'entry whose evidence cites a SYMBOL rather than a line number.',
    );
  });
});
