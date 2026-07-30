// Binds the README's description of the telemetry envelope to the envelope the
// code actually produces.
//
// This exists because of a specific landmine that has now been defused: the
// constant was named `MEASURED` while its value was the string `'ok'`. Both the
// README and any panel author could therefore write `field.state === 'measured'`,
// get `false` for every measured field on the page, and see an interface that
// renders without error and without data.
//
// The rename has landed — name and value now both read `measured`. This file
// stays because it does not hardcode a spelling: it reads the wire value off
// the constant and requires the README to name that exact value, so the next
// rename cannot land without the docs moving in the same commit.
//
// Every assertion below was verified by breaking the thing it protects and
// watching it go red. The mutation is stated on each test.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { FIELD_STATES, measuredField } from './telemetry-field.js';
import { assertShippingTree } from './shipping-tree.mjs';

// Provenance before content. Every path below is resolved from import.meta.url,
// so this file would read a parked worktree self-consistently and pass. Assert
// which tree we are in BEFORE asserting anything about what is in it.
assertShippingTree();

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
  // This test deliberately does NOT hardcode a spelling. The value has been
  // 'ok' and is being renamed to 'measured'; hardcoding either one makes this
  // test wrong in the other world, and the point is that the README tracks the
  // CODE rather than tracking a ruling about the code.
  //
  // So: read the wire value from the constant, then require the README to name
  // that exact value. Whenever the rename lands, this goes red until the README
  // is updated in the same commit.
  //
  // MUTATION: pointed the README sentence at the other spelling while leaving
  // the constant alone -> red, naming both sides.
  const wire = FIELD_STATES.MEASURED ?? FIELD_STATES.OK;

  assert.ok(
    typeof wire === 'string' && wire.length > 0,
    'Neither FIELD_STATES.MEASURED nor FIELD_STATES.OK exists. The measured ' +
      'state was renamed again and this test can no longer find it.',
  );

  const declared = README.match(/measured state's wire value is the string `'([^']+)'`/);
  assert.ok(
    declared,
    "README.md no longer contains the sentence declaring the measured state's " +
      'wire value. That sentence is what stops a reader comparing field.state ' +
      'against a literal that is false for every field on the page.',
  );

  assert.equal(
    declared[1],
    wire,
    `README.md says the measured state is on the wire as '${declared[1]}', but ` +
      `the constant emits '${wire}'. Update the README in the same commit as ` +
      `the rename -- a stale spelling here is worse than none, because it is ` +
      `precise and wrong.`,
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

// ---------------------------------------------------------------------------
// The README prints the GLYPH for each state, and a glyph is the one part of a
// state machine a reader copies into their own mental model. It was wrong here
// for over an hour: the table said `not-applicable` renders an em-dash, while
// panel-kit.js has rendered `n/a` and collapsed whole panel bodies since the
// design ruling that superseded it.
//
// Nothing could catch that. The wire VALUES were bound to the constants by the
// tests above, and they stayed green throughout, because the value and the glyph
// are different facts and only one of them was pinned. A state's identity on the
// wire and its appearance on screen drift independently.
test('the glyph the README prints for not-applicable is the one the code renders', () => {
  const panelKit = readFileSync(join(HERE, 'dashboard', 'panel-kit.js'), 'utf8');

  const rendered = panelKit.match(/value__num--not-applicable'\],\s*\n\s*text: '([^']+)'/);
  assert.ok(
    rendered,
    'Could not find the not-applicable field text in dashboard/panel-kit.js. ' +
      'If that render path moved, this test must follow it rather than be deleted.',
  );

  const row = README.match(/\|\s*\*\*`([^`]+)`[^|]*\*\*\s*\*not applicable\*/);
  assert.ok(
    row,
    'README.md no longer has a "not applicable" row in its state table naming a ' +
      'glyph in backticks.',
  );

  assert.equal(
    row[1],
    rendered[1],
    `README.md shows '${row[1]}' for not-applicable; panel-kit.js renders ` +
      `'${rendered[1]}'. The em-dash was the ratified treatment once and stopped ` +
      `being it, and the README kept printing it for an hour after the code changed.`,
  );

  // MUTATION: README row set back to `—` -> red, naming both sides. Render text
  // changed to 'N/A' in panel-kit.js -> red from the other direction.
  assert.ok(
    /collapseNotApplicableBody/.test(README),
    'README.md no longer mentions collapseNotApplicableBody. The panel-level ' +
      'collapse is what makes `unavailable` and `not-applicable` impossible to ' +
      'confuse -- they render at different SCALES -- and a reader who does not ' +
      'know the panel body gets replaced will read a collapsed panel as broken.',
  );
});
