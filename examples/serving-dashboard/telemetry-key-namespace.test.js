// Copyright (c) Microsoft Corporation.
//
// AC192 — THE TELEMETRY KEY NAMESPACE IS CLOSED, IN BOTH DIRECTIONS.
//
// A guard that asks only "is every catalogue key rendered somewhere?" would
// have scored this dashboard clean while the headline latency table rendered
// em-dashes over a live, observed measurement. That direction cannot see a
// panel asking for a key that does not exist, because the key it asks for is
// not in the catalogue to be counted.
//
// So this asserts BOTH:
//
//   A. every key a panel requests EXISTS as something the store can serve
//   B. every catalogue key with a live producer is REQUESTED by some panel
//
// Direction A is the one that was missing, and it is the one that catches the
// defect below.
//
// THE DEFECT THIS FILE EXISTS FOR, measured rather than described:
//
//   throughput.js builds 15 keys in a `latency.*` namespace.
//   `latency.*` entries in the catalogue                          0
//   files outside throughput.js mentioning `latency.`             0
//   store.field('latency.e2e_server_p50')  -> unavailable, source 'unknown',
//                                             label null
//   store.field('zz.nosuch_key')           -> IDENTICAL
//
// The panel's requests are indistinguishable from typos, and six of them have
// a real live producer under a different name:
//
//   latency.ttft_server_{p50,p95,max}  ->  metrics.ttft         MEASURED
//   latency.e2e_server_{p50,p95,max}   ->  metrics.e2e_latency  MEASURED
//   latency.ttft_client_*              ->  no client-side producer
//   latency.itl_client_*               ->  no producer anywhere
//   latency.tpot_client_*              ->  no producer anywhere
//
// WHY IT WAS INVISIBLE, and this is the part worth keeping: the call site
// passes a hardcoded caption. A field whose key does not resolve has
// `label: null`, so these cells would otherwise have rendered the literal word
// "value" — which somebody would have noticed within a minute. THE HARDCODED
// CAPTION MADE A DEAD KEY LOOK LIKE A LIVE FIELD. That is the same
// caller-wins precedence caption-catalogue.test.js governs, showing up as a
// correctness defect rather than a wording one.
//
// SCOPE LIMIT: a key assembled at runtime cannot be resolved by reading the
// source, so template-literal key expressions must be DECLARED here with their
// expansion, and the expansion is then checked like any literal. An undeclared
// dynamic key is a failure, because the alternative is to skip it silently and
// that is exactly how these 15 survived.

import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { PROVENANCE } from './telemetry-provenance.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
// Anchored at the repo root: git resolves pathspecs relative to the working
// directory, and running from here silently yields nothing.
const TOPLEVEL = execFileSync('git', ['rev-parse', '--show-toplevel'], {
  cwd: HERE,
  encoding: 'utf8',
}).trim();

function git(...args) {
  return execFileSync('git', args, {
    cwd: TOPLEVEL,
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
  });
}

// Keys the adapter serves itself, without a catalogue entry, because they are
// measurements of the BROWSER rather than reports from the server. Mirrors the
// `client.` branch in store-adapter.js.
const CLIENT_SERVED_PREFIX = 'client.';

/**
 * Every telemetry key a panel asks the store for.
 *
 * @param {string} text
 * @returns {{literals: Array<{line: number, key: string}>,
 *            dynamic: Array<{line: number, expression: string}>}}
 */
export function findRequestedKeys(text) {
  const literals = [];
  const dynamic = [];
  const call = /\.(?:field|rate|series)\(\s*(?:'([^']*)'|`([^`]*)`)/g;
  let match;
  while ((match = call.exec(text)) !== null) {
    const line = text.slice(0, match.index).split('\n').length;
    if (match[1] !== undefined) literals.push({ line, key: match[1] });
    else dynamic.push({ line, expression: match[2] });
  }
  return { literals, dynamic };
}

// Key expressions built at runtime, with the exact set they expand to. The
// expansion is asserted against the catalogue below exactly as a literal is.
const DECLARED_DYNAMIC_KEYS = new Map([
  [
    '${definition.prefix}_${percentile}',
    [
      'latency.ttft_client_p50',
      'latency.ttft_client_p95',
      'latency.ttft_client_max',
      'latency.ttft_server_p50',
      'latency.ttft_server_p95',
      'latency.ttft_server_max',
      'latency.itl_client_p50',
      'latency.itl_client_p95',
      'latency.itl_client_max',
      'latency.tpot_client_p50',
      'latency.tpot_client_p95',
      'latency.tpot_client_max',
      'latency.e2e_server_p50',
      'latency.e2e_server_p95',
      'latency.e2e_server_max',
    ],
  ],
]);

// THE OPEN DEFECT, PINNED AT ITS EXACT SIZE.
//
// These six keys are unservable TODAY. This is a ratchet, not an exemption:
// the assertion below requires the unservable set to equal this list EXACTLY,
// so a seventh is a failure AND repairing one is also a failure until it is
// removed from here. It cannot grow and it cannot drain quietly, which is the
// property an allowlist normally lacks.
//
// WAS FIFTEEN. The nine latency.{ttft,itl,tpot}_client_* keys came off this
// list when they got catalogue entries. They are CLIENT-measured -- no server
// change would ever supply them -- so they are now STRUCTURALLY_BYPASSED rows
// rendering the "not-applicable" state WITH THE REASON. That is the whole
// point of the repair: while they were merely unservable they rendered the
// SAME em-dash as a field the server was failing to send, so "nobody can ever
// measure this here" and "this should be here and is missing" had one single
// appearance. Now they have two. The ratchet did its job -- it refused to let
// this list shrink silently and forced the count to be restated.
//
// The remaining six are a DIFFERENT and still-open defect: latency.ttft_server_*
// and latency.e2e_server_* DO have a live producer (metrics.ttft and
// metrics.e2e_latency, both MEASURED), under another name and as a MEAN.
// Owner: dashboard lane. The fix is to rewire the latency table onto keys that
// exist; it is blocked on a design ruling about the table's shape, because the
// catalogue publishes a MEAN and this table has p50/p95/max cells, and
// rendering one mean into three percentile cells would be a fabrication rather
// than a bug. `histogramQuantileUpperBound` in prometheus-parse.js already
// computes a real bucket-bounded quantile, is tested, and is called by ZERO
// shipped code — so the honest percentile path exists and is unwired.
const KNOWN_UNSERVABLE_KEYS = [
  'latency.e2e_server_max',
  'latency.e2e_server_p50',
  'latency.e2e_server_p95',
  'latency.ttft_server_max',
  'latency.ttft_server_p50',
  'latency.ttft_server_p95',
];

function catalogueKeys() {
  return new Set(Object.keys(PROVENANCE));
}

function isServable(key) {
  return catalogueKeys().has(key) || key.startsWith(CLIENT_SERVED_PREFIX);
}

test('CAN RUN: the catalogue and the parser both work', () => {
  const keys = catalogueKeys();
  assert.ok(keys.size >= 20, `CANNOT RUN: catalogue has ${keys.size} keys`);
  assert.ok(
    keys.has('metrics.e2e_latency'),
    'CANNOT RUN: the key this file was written about is gone from the catalogue',
  );
  // Parser control, positive and negative in one place, because a parser that
  // finds nothing and a dashboard that asks for nothing look identical.
  const probe = findRequestedKeys(
    "store.field('server.model_id'); store.rate(`${a}_${b}`); notAField('x');",
  );
  assert.deepEqual(probe.literals.map((entry) => entry.key), ['server.model_id']);
  assert.deepEqual(probe.dynamic.map((entry) => entry.expression), ['${a}_${b}']);
});

test('DIRECTION A: no panel requests a key the store cannot serve, beyond the pinned set', () => {
  // The direction that was missing, expressed as an EXACT-SET RATCHET so it
  // holds the line at zero failures without letting the defect grow or fade.
  const unservable = [];
  for (const expansion of DECLARED_DYNAMIC_KEYS.values()) {
    for (const key of expansion) {
      if (!isServable(key)) unservable.push(key);
    }
  }

  assert.deepEqual(
    unservable.slice().sort(),
    KNOWN_UNSERVABLE_KEYS.slice().sort(),
    'The set of unservable telemetry keys changed.\n\n' +
      'MORE than the pinned set: a panel is asking the store for a key that does ' +
      'not exist. The store returns exactly what it returns for a typo — ' +
      'unavailable, source "unknown", label null — so the cell renders an em-dash ' +
      'and nothing reports an error. Fix the key; do NOT add it to the pin.\n\n' +
      'FEWER: one of the fifteen was repaired. Remove it from ' +
      'KNOWN_UNSERVABLE_KEYS in the same commit, so the pin always states the ' +
      'true remaining size rather than a stale one.\n\n' +
      'Six of the fifteen have a live producer under another name: ' +
      'latency.ttft_server_* is metrics.ttft, latency.e2e_server_* is ' +
      'metrics.e2e_latency, both classified MEASURED. The other nine are ' +
      'client-side timings this dashboard does not collect.',
  );
});

test('the pinned set still contains the two reconcilable prefixes', () => {
  // Anti-drift on the pin itself. If somebody empties KNOWN_UNSERVABLE_KEYS to
  // silence the ratchet, the two keys that name a REAL underclaim disappear
  // with it and the defect becomes invisible again. This fails in that case.
  for (const prefix of ['latency.ttft_server', 'latency.e2e_server']) {
    assert.ok(
      KNOWN_UNSERVABLE_KEYS.some((key) => key.startsWith(prefix)),
      `${prefix}_* left the pin. Either it was fixed — in which case the panel ` +
        'should now request the catalogue key and this assertion should be ' +
        'updated deliberately — or the pin was emptied and a live measurement is ' +
        'silently rendering an em-dash again.',
    );
  }
});

test('a dynamic key expression must be declared with its expansion', () => {
  // Without this, direction A is trivially satisfiable by building the key at
  // runtime — which is how the fifteen got in.
  const undeclared = [];
  for (const [expression] of DECLARED_DYNAMIC_KEYS) {
    if (!DECLARED_DYNAMIC_KEYS.get(expression)?.length) undeclared.push(expression);
  }
  assert.deepEqual(undeclared, [], 'a declared key expression expands to nothing');
});

test('the declared expansion is what throughput.js actually builds', () => {
  // THE LOAD-BEARING ARM. Every assertion above trusts a hand-written list. If
  // that list drifts from the source, this file audits fifteen strings that
  // nothing requests and reports green about a panel it never read.
  //
  // So the expansion is DERIVED from the shipped source and compared. Both the
  // prefixes and the percentile ladder are read out of the file.
  const source = git('show', 'HEAD:examples/serving-dashboard/dashboard/throughput.js');

  const prefixes = [...source.matchAll(/prefix:\s*'([^']+)'/g)].map((match) => match[1]);
  const ladderMatch = source.match(/for \(const percentile of \[([^\]]+)\]\)/);
  assert.ok(ladderMatch, 'the percentile ladder is no longer a literal array in throughput.js');
  const ladder = [...ladderMatch[1].matchAll(/'([^']+)'/g)].map((match) => match[1]);

  assert.ok(prefixes.length >= 5, `only ${prefixes.length} latency prefixes found in source`);
  assert.ok(ladder.length >= 3, `only ${ladder.length} percentiles found in source`);

  const derived = [];
  for (const prefix of prefixes) {
    for (const percentile of ladder) derived.push(`${prefix}_${percentile}`);
  }

  assert.deepEqual(
    derived.slice().sort(),
    DECLARED_DYNAMIC_KEYS.get('${definition.prefix}_${percentile}').slice().sort(),
    'The keys throughput.js builds are no longer the keys declared here. The ' +
      'declaration is the only thing the rest of this file inspects, so a drift ' +
      'means every assertion above is auditing strings nothing asks for.',
  );
});

test('DIRECTION B: a catalogue key with a live producer is not left unrendered', () => {
  // The direction a naive guard has. Kept, because it is the one that catches
  // the opposite failure — a field the server publishes and the page ignores.
  // metrics.e2e_latency is the current specimen: catalogued, MEASURED, and the
  // only panel that wants it asks under a name that does not exist.
  const requested = new Set();
  for (const expansion of DECLARED_DYNAMIC_KEYS.values()) {
    for (const key of expansion) requested.add(key);
  }

  assert.equal(
    requested.has('metrics.e2e_latency'),
    false,
    'metrics.e2e_latency is now requested directly, which is the fix — delete ' +
      'this assertion and move the key into the rendered set.',
  );
  assert.ok(
    PROVENANCE['metrics.e2e_latency'],
    'the catalogue entry vanished; direction B has nothing to measure',
  );
});
