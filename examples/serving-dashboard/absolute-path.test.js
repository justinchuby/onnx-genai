import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  findAbsolutePaths,
  isAbsolutePathValue,
  knownPosixRoots,
} from './absolute-path.mjs';

/**
 * The detector behind every model-path disclosure guard.
 *
 * This file exists because the previous detector was `text.includes('/Users/')`
 * and it was certified by a mutation that injected `/Users/presenter/...` --
 * a constant defined inside the test file itself. The probe was drawn from the
 * detector's own literal, so the proof could only ever succeed. Measured at
 * b63f0a82: /home, C:\ and /var disclosures all rendered with the suite at 5/5.
 *
 * A detector must therefore be tested in BOTH directions against values it did
 * not supply: things it must catch, and things it must leave alone.
 */

/** Must be flagged. Each row is a real disclosure on some operating system. */
const MUST_DETECT = Object.freeze([
  ['macOS home', '/Users/presenter/Documents/GitHub/onnx-genai/models/qwen2.5-0.5b'],
  ['Linux home', '/home/presenter/models/qwen2.5-0.5b'],
  ['Linux service dir', '/var/lib/onnx-genai/models/qwen2.5-0.5b'],
  ['Linux root home', '/root/models/qwen'],
  ['macOS external volume', '/Volumes/Scratch/models/qwen'],
  ['Windows drive, backslash', 'C:\\Users\\presenter\\models\\qwen'],
  ['Windows drive, forward slash', 'D:/models/qwen2.5-0.5b'],
  ['Windows UNC share', '\\\\fileserver\\models\\qwen'],
]);

/**
 * Must NOT be flagged. The first two are this repository's own defaults --
 * `--model-id` takes an operator string with no character validation, so
 * flagging these turns the suite red on a legal invocation with a message
 * accusing the operator of leaking their home directory.
 */
const MUST_IGNORE = Object.freeze([
  ['namespaced model id', 'Qwen/Qwen2.5-0.5B-Instruct'],
  ['namespaced model id, 2', 'roneneldan/TinyStories-33M'],
  ['plain model id', 'qwen-scatter'],
  ['a rate label', '128 req/s'],
  ['a relative path', 'models/qwen2.5-0.5b'],
  ['a bare origin', 'http://127.0.0.1:8080'],
  ['empty', ''],
]);

describe('absolute path detection', () => {
  it('flags an absolute path on every operating system, not just this desk', () => {
    for (const [label, value] of MUST_DETECT) {
      assert.ok(
        isAbsolutePathValue(value),
        `${label}: isAbsolutePathValue missed ${value}`,
      );
      assert.ok(
        findAbsolutePaths(`Telemetry connection from ${value} is live`).length > 0,
        `${label}: findAbsolutePaths missed ${value} embedded in a sentence`,
      );
    }
  });

  it('leaves relative and namespaced identifiers alone', () => {
    for (const [label, value] of MUST_IGNORE) {
      assert.equal(
        isAbsolutePathValue(value),
        false,
        `${label}: isAbsolutePathValue falsely flagged ${value}`,
      );
    }
  });

  it('does not flag a namespaced model id inside rendered text', () => {
    // The false positive is the delivery mechanism for the regression: a guard
    // that reddens on `--model-id Qwen/Qwen2.5-0.5B-Instruct` gets loosened by
    // whoever hits it, and the loosening nearest to hand is weakening the ban.
    const rendered = 'model Qwen/Qwen2.5-0.5B-Instruct · 128 req/s · GET /v1/models';
    assert.deepEqual(findAbsolutePaths(rendered), []);
  });

  it('the two predicates disagree on a URL path, deliberately', () => {
    // `/v1/models` IS absolute, so the value-level predicate says so -- it
    // answers the question its name asks and nothing more. The text scanner
    // says no, because scanning rendered prose for a bare leading slash would
    // flag every endpoint and every `req/s` axis label, and a guard that
    // reddens on legal output gets loosened by whoever hits it.
    //
    // Recorded as a designed boundary rather than tuned away: the two helpers
    // have different scopes, and a disclosure guard reading free text must use
    // findAbsolutePaths.
    assert.equal(isAbsolutePathValue('/v1/models'), true);
    assert.deepEqual(findAbsolutePaths('GET /v1/models'), []);
  });

  it('reports the match, so a failure names what leaked', () => {    const found = findAbsolutePaths('label: /home/presenter/models/qwen, ok');
    assert.deepEqual(found, ['/home/presenter/models/qwen']);
  });

  it('ignores non-strings rather than throwing on them', () => {
    for (const value of [null, undefined, 42, {}, []]) {
      assert.equal(isAbsolutePathValue(value), false);
      assert.deepEqual(findAbsolutePaths(value), []);
    }
  });

  it('knows about more than one root, so the denylist cannot rot to empty', () => {
    const roots = knownPosixRoots();
    assert.ok(roots.length >= 8, `only ${roots.length} roots; the list was gutted`);
    assert.ok(roots.includes('Users') && roots.includes('home'), 'the two commonest homes must be listed');
  });
});
