// Copyright (c) Microsoft Corporation.
//
// The provenance register is the page's court of last resort: a sceptical
// visitor consults it BECAUSE they distrust the panels. These are the checks
// that the register agrees with the table it claims to be generated from.
//
// WHY THIS FILE EXISTS. Two defects shipped to a live page at once, and both
// were invisible to every other guard in the tree.
//
//  1. The register rendered the BASE classification for every row. It was the
//     one consumer of the provenance table that never called resolveForOrigin,
//     so on the batching server it certified the three prefix-cache counters as
//     "Measured by the server" while the panels beside it — reading the SAME
//     table, correctly resolved — showed them structurally bypassed. The two
//     halves of the page disagreed about the same fact, and the half that was
//     wrong was the half that exists to be trusted.
//
//  2. The classification-to-English map had no entry for
//     STRUCTURALLY_BYPASSED, and the lookup fell back to `?? entry.classification`.
//     So the register printed the raw constant — an internal enum token, in
//     screaming snake case, in a table cell, beside nine carefully written
//     English sentences. A `??` fallback on a display string cannot fail; it
//     renders something, and something is indistinguishable from working.
//
// Neither is visible from either artefact alone. app.js is internally
// consistent, telemetry-provenance.js is internally consistent, and the defect
// is in the relationship. app.js self-starts on import (it calls main() at the
// bottom), so the map is read off disk rather than imported, the same way
// state-treatments.test.js reads shell.css.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { PROVENANCE, allFieldKeys, resolveForOrigin } from './telemetry-provenance.js';

const appSource = readFileSync(new URL('./app.js', import.meta.url), 'utf8');
const provenanceSource = readFileSync(
  new URL('./telemetry-provenance.js', import.meta.url),
  'utf8',
);

/** Every origin the register can be rendered for, plus the unresolved case. */
const ORIGINS = Object.freeze([null, 'scatter', 'dynamic']);

/** The keys of CLASSIFICATION_TEXT, read from app.js rather than restated. */
function labelledClassifications(source) {
  const block = source.match(/const CLASSIFICATION_TEXT = Object\.freeze\(\{([\s\S]*?)\n\}\);/);
  assert.ok(
    block,
    'CLASSIFICATION_TEXT was not found in app.js. If it was renamed or reshaped, ' +
      'update this matcher — do NOT delete the test. A parser that silently ' +
      'matches nothing reports a clean bill of health, which is the one failure ' +
      'mode indistinguishable from success.',
  );
  const keys = [...block[1].matchAll(/^\s{2}([A-Z_]+):/gm)].map((m) => m[1]);
  assert.ok(
    keys.length > 0,
    'Parsed CLASSIFICATION_TEXT but found zero keys. See above: an under-matching ' +
      'checker passes for the wrong reason.',
  );
  return new Set(keys);
}

/** The `Classification` union, read from its typedef rather than restated. */
function declaredClassifications() {
  const union = provenanceSource.match(/@typedef \{([^}]*)\} Classification/);
  assert.ok(
    union,
    'The Classification typedef was not found in telemetry-provenance.js. If it ' +
      'was renamed, update this matcher — do NOT delete the test. A parser that ' +
      'silently matches nothing reports a clean bill of health, which is the one ' +
      'failure mode indistinguishable from success.',
  );
  const names = [...union[1].matchAll(/'([A-Z_]+)'/g)].map((m) => m[1]);
  assert.ok(names.length > 0, 'Parsed the Classification typedef and found zero members.');
  return new Set(names);
}

/** Every classification actually reachable, at any origin, after resolution. */
function reachableClassifications() {
  const reached = new Set();
  for (const key of allFieldKeys()) {
    for (const origin of ORIGINS) {
      reached.add(resolveForOrigin(PROVENANCE[key], origin).classification);
    }
  }
  return reached;
}

test('every declared classification has English text for the visitor', () => {
  // Checked against the DECLARED vocabulary, not against what entries happen to
  // use today. DOCUMENTED_ZERO is currently declared and documented but unused;
  // requiring its text now means the leak cannot reappear the day someone
  // classifies a field that way. Waiting for a member to be reachable before
  // requiring its label is waiting for the defect to ship.
  const labelled = labelledClassifications(appSource);
  const unlabelled = [...declaredClassifications()].filter((c) => !labelled.has(c));

  assert.deepEqual(
    unlabelled,
    [],
    `The register can render ${unlabelled.join(', ')} and has no sentence for it. ` +
      'The visitor sees the raw constant — an internal identifier presented as a ' +
      'status, in the table that exists to prove we are not hiding anything.',
  );
});

test('the register invents no status of its own', () => {
  const declared = declaredClassifications();
  const invented = [...labelledClassifications(appSource)].filter((c) => !declared.has(c));

  assert.deepEqual(
    invented,
    [],
    `CLASSIFICATION_TEXT describes ${invented.join(', ')}, which is not in the ` +
      'Classification union. Either the vocabulary gained a member without its ' +
      'documentation, or the register is describing a status that does not exist.',
  );
});

test('no entry carries a classification outside the vocabulary', () => {
  // Classifications are bare strings in a large hand-edited object literal, so a
  // typo produces a value that is in no list — and being in no list means being
  // in neither NEVER_MEASURED_CLASSIFICATIONS nor any suppression check, so a
  // misspelled classification FAILS OPEN and the field renders as measured.
  const declared = declaredClassifications();
  const undeclared = [...reachableClassifications()].filter((c) => !declared.has(c));

  assert.deepEqual(
    undeclared,
    [],
    `${undeclared.join(', ')} is used as a classification but is not declared. A ` +
      'classification in no list is suppressed by nothing: the field renders as a ' +
      'measured value, which is the exact failure this table exists to prevent.',
  );
});

test('the register resolves per origin, so it cannot contradict the panels', () => {
  // The regression that motivated the file. The bug was a MISSING ARGUMENT, so
  // asserting the call is not pedantry: the previous code read `PROVENANCE[key]`
  // directly and every count, key and label in the table was still correct.
  assert.match(
    appSource,
    /resolveForOrigin\(PROVENANCE\[key\], origin\)/,
    'The provenance register must resolve each entry for the origin that served ' +
      'the page. Reading the base entry makes it disagree with the panels on any ' +
      'field whose classification is per-origin — silently, and only on one server.',
  );
  assert.ok(
    !/CLASSIFICATION_TEXT\[[^\]]+\]\s*\?\?/.test(appSource),
    'A `??` fallback on the classification text renders the raw enum token to a ' +
      'visitor. classificationText() throws instead: unlabelled is a bug, not a ' +
      'display state.',
  );
});

test('the prefix-cache counters are not certified as measured on the batching server', () => {
  // The specific live defect, pinned so it cannot come back by another route.
  // This server never consults the prefix cache, so a register row claiming the
  // counters are "measured by the server" certifies a feature this project
  // cut — on the one surface a sceptic reads precisely because they doubt us.
  const counters = allFieldKeys().filter((key) => /prefix_cache/.test(key));
  assert.ok(counters.length >= 3, 'expected the prefix-cache counter family in the register');

  for (const key of counters) {
    const entry = resolveForOrigin(PROVENANCE[key], 'scatter');
    assert.notEqual(
      entry.classification,
      'MEASURED',
      `${key} resolves to MEASURED on the batching origin, which never runs that ` +
        'code path. The register would print "Measured by the server" beside ' +
        'evidence explaining that nothing measured it.',
    );
  }
});
