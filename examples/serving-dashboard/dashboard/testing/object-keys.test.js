// Copyright (c) Microsoft Corporation.
//
// Tests for the duplicate-key scanner. A guard that reads source text is
// itself source text, and the two failures below were both REAL: they were
// produced by earlier drafts of this scanner against this repository, not
// invented as hypotheticals. They are pinned by name so a future simplification
// reintroduces them loudly.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { declaredKeys, duplicatesAmong, findLiteralOpener } from './object-keys.js';

function keysOf(source, marker = 'const T =') {
  return declaredKeys(source, findLiteralOpener(source, marker));
}

describe('the duplicate-key scanner reads keys, not shapes', () => {
  it('sees a duplicate however the key is quoted', () => {
    // The defect that shipped: the regex guard matched single-quoted keys at a
    // fixed indent, so this exact file scored green while the catalogue lost an
    // entry. All three spellings are the SAME key to JavaScript.
    const source = `const T = {
  'batch.capacity': { a: 1 },
  "batch.capacity": { a: 2 },
};`;
    const duplicates = duplicatesAmong(keysOf(source));
    assert.deepEqual(duplicates.map((d) => d.name), ['batch.capacity']);
    assert.deepEqual(duplicates[0].lines, [2, 3], 'both definitions must be located, not just counted');
  });

  it('sees a duplicate written as a bare identifier', () => {
    const source = `const T = {
  alpha: 1,
  alpha: 2,
};`;
    assert.deepEqual(duplicatesAmong(keysOf(source)).map((d) => d.name), ['alpha']);
  });

  it('is not fooled by indentation', () => {
    const source = `const T = {
  'a.b': { x: 1 },
      'a.b': { x: 2 },
};`;
    assert.deepEqual(duplicatesAmong(keysOf(source)).map((d) => d.name), ['a.b']);
  });

  it('does not report a ternary colon as a key', () => {
    // FALSE POSITIVE, OBSERVED: an earlier draft flagged `top` twice in
    // dashboard/field-state.js, from `value: discrete ? top : x` and
    // `numerator: discrete ? top : null`. A ternary colon reads exactly like a
    // key colon unless you track what precedes it.
    const source = `const T = {
  value: discrete ? top : (top / bottom) * 100,
  numerator: discrete ? top : null,
};`;
    assert.deepEqual(duplicatesAmong(keysOf(source)), []);
    assert.deepEqual(keysOf(source).map((k) => k.name), ['value', 'numerator']);
  });

  it('does not read inside a regex literal', () => {
    // FALSE POSITIVE, OBSERVED: an earlier draft flagged `ModelDecodePath`
    // twice from a single regex in check-perf-claims.test.js, whose `{` also
    // desynchronised the depth counter for the rest of the file.
    const source = `const T = {
  refusal: /ModelDecodePath::PastPresent\\s*\\{\\s*\\}\\s*\\|\\s*ModelDecodePath::Legacy/,
  other: 1,
};`;
    assert.deepEqual(duplicatesAmong(keysOf(source)), []);
    assert.deepEqual(keysOf(source).map((k) => k.name), ['refusal', 'other']);
  });

  it('does not read keys out of strings, comments or nested objects', () => {
    const source = `const T = {
  // dupe: 1,
  /* dupe: 2, */
  note: 'dupe: 3,',
  nested: { dupe: 4, dupe: 5 },
  dupe: 6,
};`;
    // The nested duplicate is a different scope; reporting it here would be a
    // lie about WHICH object is broken.
    assert.deepEqual(duplicatesAmong(keysOf(source)), []);
    assert.deepEqual(keysOf(source).map((k) => k.name), ['note', 'nested', 'dupe']);
  });

  it('refuses to answer when it cannot find the literal', () => {
    // A scanner aimed at nothing finds no duplicates, which is byte-identical
    // to a clean file. It must throw rather than reassure.
    assert.throws(() => keysOf('const OTHER = { a: 1 };'), /does not appear in the source/);
  });

  it('refuses to answer when it loses sync', () => {
    assert.throws(() => keysOf('const T = { a: 1,'), /never closed/);
  });
});
