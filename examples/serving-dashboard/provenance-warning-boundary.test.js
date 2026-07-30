// Copyright (c) Microsoft Corporation.
//
// The provenanceWarning boundary: a warning may never carry filesystem bytes,
// whatever the field's VALUE happens to look like.
//
// WHY THIS FILE EXISTS
// --------------------
// `displaySafeField` neutralises a path-shaped field VALUE. It used to clear
// `provenanceWarning` too -- but only from inside that branch:
//
//     if (!isAbsolutePathValue(field?.value)) return field;   // early return
//     return { ...field, value: null, provenanceWarning: null };
//
// So the warning was cleaned only when the VALUE was already a path. Give it a
// value of `42` and a warning naming `/Users/operator/secret/provider`, and the
// early return hands back the field untouched. Measured before the fix:
//
//     safeSame                : true
//     safeWarningHasPath      : true
//     formattedWarningHasPath : true
//
// AND WHY IT WAS INVISIBLE, which is the more useful half. The existing test in
// format.test.js builds its fixture by putting the SAME path in the value and in
// the warning. That field always satisfies `isAbsolutePathValue`, so every run
// enters the cleanup branch and the leaking state is unreachable BY
// CONSTRUCTION. The suite was green because of how the fixture was written, not
// because the code was safe.
//
// The lesson generalises past this bug: when a fixture uses one poisoned value
// for two independent inputs, it silently tests the conjunction and reports it
// as coverage of each. Every case below therefore DECOUPLES them -- non-path
// value, path-bearing warning -- which is the only shape that can observe the
// bypass.
//
// WHERE THE FIX LIVES, and why not here. Neutralisation is intrinsic to
// `measuredField`, the single constructor through which a provenanceWarning
// becomes field metadata. That placement is deliberate: it holds regardless of
// the field's value, and regardless of whether `displaySafeField` ever ran. A
// second cleanup that a caller must remember to invoke is what got the previous
// attempt rejected, and correctly so -- the failure mode of caller-coordinated
// hygiene is the caller who does not know the rule exists.
//
// SURFACES NOT ASSERTED HERE, stated rather than quietly omitted. The store's
// `provenanceWarnings()` map and its `console.warn` carry only STORE-BUILT
// strings, which describe a value's TYPE via `describeWireValue`
// (telemetry-store.js:797) and never interpolate the value itself. A
// caller-supplied warning cannot reach that map, so an assertion here that
// "the stored warning has no path" could not fail for this defect. Writing one
// anyway would be a third test that cannot reach the state it claims to cover,
// which is the exact fault this file was opened to repair. That surface is
// covered live at telemetry-store.test.js:1834.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { measuredField } from './telemetry-field.js';
import { displaySafeField, formatField } from './format.js';
import { findAbsolutePaths } from './absolute-path.mjs';

/** 6c28526f's reproduction, verbatim. */
const SECRET = '/Users/operator/secret/provider';

/** A value that is NOT path-shaped, so the displaySafeField branch is skipped. */
const PLAIN_VALUE = 42;

function poisonedField(value = PLAIN_VALUE) {
  return measuredField(value, {
    source: '/v1/status',
    label: 'provider',
    provenanceWarning:
      `"server.execution_provider" is classified DOCUMENTED_ZERO in ` +
      `telemetry-provenance.js, but the server sent ${SECRET} instead.`,
  });
}

/**
 * Assert one surface, named, with the offending bytes reported.
 *
 * Deliberately not folded into a single aggregate assertion. An aggregate says
 * "something leaked" and leaves the next reader to bisect five surfaces by
 * hand; worse, it goes green the moment the LAST surface is fixed, hiding that
 * four others were only ever incidentally clean.
 */
function assertNoPath(surfaceName, text) {
  const found = findAbsolutePaths(String(text ?? ''));
  assert.deepEqual(
    found,
    [],
    `${surfaceName} disclosed ${found.length} absolute path(s): ${found.join(', ')}\n` +
      `  full text: ${JSON.stringify(text)}\n` +
      'A provenance warning explains WHICH field disagreed with the table and WHY. ' +
      'It never needs the operator\'s directory layout to do that.',
  );
}

describe('the provenanceWarning boundary neutralises paths intrinsically', () => {
  it('field metadata carries no path when the value is not path-shaped', () => {
    assertNoPath('field.provenanceWarning', poisonedField().provenanceWarning);
  });

  it('the display-safe envelope carries no path, even though it never entered the cleanup branch', () => {
    const field = poisonedField();
    const safe = displaySafeField(field);

    // The bypass, pinned as a property rather than described in a comment: this
    // field is returned UNCHANGED by displaySafeField, because its value is not
    // a path. That is correct behaviour and is not what is being fixed. The
    // warning must already be clean by the time it gets here.
    assert.equal(
      safe,
      field,
      'Fixture no longer exercises the bypass: displaySafeField modified this ' +
        'field, so the value must have become path-shaped and the cleanup branch ' +
        'ran after all. The decoupling that makes this test meaningful is gone.',
    );
    assertNoPath('displaySafeField(field).provenanceWarning', safe.provenanceWarning);
  });

  it('formatted output carries no path', () => {
    assertNoPath('formatField(field).provenanceWarning', formatField(poisonedField()).provenanceWarning);
  });

  it('the unknown-state branch of formatField carries no path either', () => {
    // format.js has TWO returns that pass provenanceWarning through: the normal
    // one, and the terminal branch for an unrecognised state (format.js:184).
    // A fix applied to only one of them would leave the other leaking on
    // precisely the fields nobody is watching.
    const field = { ...poisonedField(), state: 'not-a-real-state' };
    const originalError = console.error;
    const logged = [];
    console.error = (...args) => logged.push(args.join(' '));
    let out;
    try {
      out = formatField(field);
    } finally {
      console.error = originalError;
    }

    assert.equal(out.state, 'not-a-real-state', 'fixture did not reach the terminal branch');
    assertNoPath('formatField(unknown-state).provenanceWarning', out.provenanceWarning);
    assertNoPath('console.error during formatField', logged.join('\n'));
  });

  it('keeps the warning diagnostic rather than deleting it', () => {
    // Neutralisation must not degrade into suppression. A developer reading
    // this still has to learn WHICH field disagreed with the provenance table
    // and WHY -- that is the entire purpose of the warning, and a null teaches
    // nobody anything. Redaction removes the bytes, not the diagnosis.
    const warning = poisonedField().provenanceWarning;

    assert.ok(warning, 'the warning was deleted rather than redacted');
    assert.match(warning, /server\.execution_provider/, 'lost WHICH field disagreed');
    assert.match(warning, /DOCUMENTED_ZERO/, 'lost WHY it disagreed');
    assert.match(
      warning,
      /withheld/i,
      'redaction must be VISIBLE -- silently deleting the bytes leaves a sentence ' +
        'that reads as though the server sent nothing at all',
    );
  });

  it('leaves a legitimate namespaced identifier in a warning untouched', () => {
    // ANTI-OVER-MATCHING. `Qwen/Qwen2.5-0.5B-Instruct` is a legal --model-id for
    // this repository. A redactor that eats it produces a warning that misreads
    // as a disclosure incident, and the next person to hit that loosens the
    // guard -- the false positive is the delivery mechanism for the regression.
    const warning = measuredField(PLAIN_VALUE, {
      source: '/v1/status',
      label: 'model',
      provenanceWarning: 'the server sent Qwen/Qwen2.5-0.5B-Instruct, which is fine.',
    }).provenanceWarning;

    assert.match(warning, /Qwen\/Qwen2\.5-0\.5B-Instruct/);
  });

  it('redacts a Windows path and a POSIX home path, not just /Users', () => {
    // The predicate must not encode the desk it was written on. absolute-path.mjs
    // exists because `text.includes('/Users/')` let three real disclosures through.
    for (const path of ['C:\\Users\\presenter\\models\\qwen', '/home/presenter/models/qwen']) {
      const warning = measuredField(PLAIN_VALUE, {
        source: '/v1/status',
        label: 'model',
        provenanceWarning: `the server sent ${path} instead.`,
      }).provenanceWarning;

      assertNoPath(`warning containing ${path}`, warning);
    }
  });
});
