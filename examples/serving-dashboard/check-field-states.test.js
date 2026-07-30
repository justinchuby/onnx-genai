// Binds the README's description of the telemetry envelope to the envelope the
// code actually produces.
//
// This exists because of a specific, live landmine: the constant is named
// `MEASURED` but its value is the string `'ok'`. A rename to `'measured'` has
// been ratified and has not landed. Both the README and any panel author can
// therefore write `field.state === 'measured'`, get `false` for every measured
// field on the page, and see an interface that renders without error and
// without data.
//
// Every assertion below was verified by breaking the thing it protects and
// watching it go red. The mutation is stated on each test.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { FIELD_STATES, measuredField } from './telemetry-field.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const README = readFileSync(join(HERE, 'README.md'), 'utf8');

// The states the README names to the reader, as concepts.
const DOCUMENTED_STATES = ['measured', 'pending', 'stale', 'unavailable', 'not-applicable'];

test('every state the README names exists in FIELD_STATES', () => {
  // MUTATION: added 'preempted' to DOCUMENTED_STATES -> red. That is the
  // dropped-counter state, and documenting it would promise a lane we cut.
  const actual = new Set(Object.values(FIELD_STATES));
  assert.ok(actual.size > 0, 'FIELD_STATES is empty — the import resolved to nothing');

  for (const state of DOCUMENTED_STATES) {
    // 'measured' is the concept; its wire value is asserted separately below,
    // because that is precisely the pair that is currently mismatched.
    if (state === 'measured') continue;
    assert.ok(
      actual.has(state),
      `README documents state '${state}' but no FIELD_STATES value equals it. ` +
        `Actual values: ${[...actual].join(', ')}`,
    );
  }
});

test('the README states the CURRENT wire value of the measured state', () => {
  // This is the drift guard. When the ratified rename to 'measured' lands,
  // FIELD_STATES.OK becomes 'measured', this test goes red, and the README's
  // warning block must be updated in the SAME commit. That is the intent:
  // the doc cannot quietly outlive the landmine it warns about.
  //
  // MUTATION: changed the asserted literal to 'measured' -> red today, which
  // is exactly the state of the world this test encodes.
  const wire = FIELD_STATES.OK;

  assert.equal(
    wire,
    'ok',
    'FIELD_STATES.OK is no longer \'ok\'. If the ratified rename has landed, ' +
      'delete the warning block in README.md that says the wire value is \'ok\' ' +
      'and says the rename has not landed — it is now false.',
  );

  assert.ok(
    /wire value is the string `'ok'`, not `'measured'`/.test(README),
    'README no longer carries the warning that the measured state is on the wire ' +
      "as 'ok'. The code still emits 'ok', so removing the warning strands every " +
      "reader who compares against 'measured'.",
  );
});

test('the README does not tell a reader to compare against a string literal', () => {
  // MUTATION: added "check `field.state === 'measured'`" as advice -> red.
  // Literal comparison is the exact bug the warning exists to prevent, so the
  // README must never model it except as the counter-example it already is.
  const advice = README.match(/^[^\n>|]*field\.state === '(\w[\w-]*)'/gm) ?? [];
  const endorsed = advice.filter((line) => !/false|never|not|wrong|bug/i.test(line));
  assert.deepEqual(
    endorsed,
    [],
    `README appears to endorse comparing field.state to a string literal: ${endorsed.join(' | ')}`,
  );
});

test('the envelope documented in the README matches what measuredField returns', () => {
  // MUTATION: removed `origin` from the README's js block -> red. That key is
  // load-bearing: it is the only one answering "which server produced this",
  // and an endpoint path is byte-identical across both servers.
  const field = measuredField(1, { source: '/v1/status' });
  const actualKeys = Object.keys(field).sort();

  const block = README.match(/```js\n(\{[^`]*?\})\n```/);
  assert.ok(block, 'Could not find the envelope js block in README.md');

  const documentedKeys = [...block[1].matchAll(/[{,]\s*(\w+)/g)].map((m) => m[1]).sort();
  assert.ok(documentedKeys.length > 0, 'Parsed zero keys — the regex matched nothing (false green)');

  const missing = actualKeys.filter((k) => !documentedKeys.includes(k));
  const invented = documentedKeys.filter((k) => !actualKeys.includes(k));

  assert.deepEqual(
    { missing, invented },
    { missing: [], invented: [] },
    `README envelope is out of sync with measuredField().\n` +
      `  Keys the code returns but the README omits: ${missing.join(', ') || '(none)'}\n` +
      `  Keys the README claims but the code does not return: ${invented.join(', ') || '(none)'}`,
  );
});
