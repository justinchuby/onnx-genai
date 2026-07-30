// Copyright (c) Microsoft Corporation.
//
// Tests for the command/prose discriminator.
//
// The regression fixture for every case below is the same real defect: this
// repository's guards must be able to DOCUMENT the command shapes they ban.
// Each test names the artefact that would have broken without it.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { fencedLines, isBlockquote, isCommandLine } from './markdown-scan.js';

describe('fencedLines', () => {
  it('returns only the lines inside a fence', () => {
    const source = ['before', '```', 'inside', '```', 'after'].join('\n');
    assert.deepEqual(
      fencedLines(source).map((entry) => entry.line),
      ['inside'],
    );
  });

  it('reports 1-based line numbers so a finding can cite an address', () => {
    const source = ['before', '```', 'inside', '```'].join('\n');
    assert.equal(fencedLines(source)[0].lineNumber, 3);
  });

  it('handles tilde fences and indented fences', () => {
    const source = ['~~~', 'a', '~~~', '  ```', '  b', '  ```'].join('\n');
    assert.deepEqual(
      fencedLines(source).map((entry) => entry.line.trim()),
      ['a', 'b'],
    );
  });

  it('excludes prose outside the fence even when it names a command', () => {
    // THE DEFECT THIS EXISTS FOR. CONTRACT.md documents the superseded two-glob
    // form in a bullet explaining that it reaches only 583 of 588 tests. Under a
    // whole-file scan that bullet SATISFIED the launch-command check, so
    // deleting the real documented command left the suite green.
    const source = ['We used to run `node --test *.test.js`, which was wrong.'].join('\n');
    assert.deepEqual(fencedLines(source), []);
  });

  it('is not confused by an unterminated fence', () => {
    // A truncated document must not silently classify its whole tail as code.
    // It will over-include here, which is the SAFE direction: a guard that sees
    // too much goes red and gets read. One that sees too little goes green.
    const source = ['```', 'a', 'b'].join('\n');
    assert.equal(fencedLines(source).length, 2);
  });
});

describe('isBlockquote', () => {
  it('recognises a quoted line, indented or not', () => {
    assert.equal(isBlockquote('> ./run-tests.sh'), true);
    assert.equal(isBlockquote('   > ./run-tests.sh'), true);
  });

  it('does not treat a shell redirect as a quote', () => {
    assert.equal(isBlockquote('./run-tests.sh > out.txt'), false);
  });
});

describe('isCommandLine', () => {
  it('accepts the shapes a reader can paste', () => {
    assert.equal(isCommandLine('./run-tests.sh'), true);
    assert.equal(isCommandLine('node --test dashboard/'), true);
    assert.equal(isCommandLine('cd examples/serving-dashboard'), true);
    assert.equal(isCommandLine('$ ./run-tests.sh'), true, 'a $ prompt is still a command');
    assert.equal(isCommandLine('  ./run-tests.sh  '), true, 'indentation is not meaning');
  });

  it('rejects prose that merely NAMES a command', () => {
    // BAN THE CLAIM, NOT THE TOPIC. Without this, a guard cannot explain itself:
    // documenting the defect trips the guard against the defect.
    assert.equal(isCommandLine('Run the suite with ./run-tests.sh before pushing.'), false);
    assert.equal(isCommandLine('The old form was `node --test *.test.js`.'), false);
  });

  it('rejects a blockquoted command — via the prefix rule, which is the whole mechanism', () => {
    // THIS TEST USED TO CLAIM THE REJECTION WAS "EXPLICIT, NOT BY ACCIDENT", and
    // the mutation that should have proved it stayed GREEN: deleting the
    // `isBlockquote()` call changed nothing, because `>` is not an accepted
    // prefix and never can be. The claim was false and the test could not tell.
    //
    // So this now pins what is actually true. If someone ever adds a prefix that
    // a blockquote could satisfy, this goes red and the second check earns its
    // place back.
    assert.equal(isCommandLine('> ./run-tests.sh'), false);
    assert.equal(isCommandLine('> node --test *.test.js'), false);
    assert.equal(isCommandLine('   > cd examples'), false);
  });

  it('does not accept an arbitrary line just because it is long', () => {
    // A control that must stay FALSE. A classifier that returned true for
    // everything would satisfy every acceptance case above and nothing else.
    assert.equal(isCommandLine('This sentence is not a command at all.'), false);
    assert.equal(isCommandLine(''), false);
  });
});
