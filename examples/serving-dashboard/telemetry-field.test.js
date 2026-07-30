// Copyright (c) Microsoft Corporation.
//
// Tests for the field BUILDERS -- the constructors every telemetry value passes
// through. These guard the call site: a field that is built wrong is frozen,
// looks well-formed, and renders confidently, so the only place to catch the
// mistake is the moment it is made.

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  measuredField,
  unavailableField,
  notApplicableField,
  pendingField,
  describeField,
  FIELD_STATES,
  SOURCE_CLASSES,
} from './telemetry-field.js';

test('an absence builder refuses an options object passed as the reason', () => {
  // Found by making this exact mistake while probing describeField. Every
  // absence builder is (reason, options) positionally, but `{ reason }` is the
  // shape used almost everywhere else in this codebase, so the slip is natural.
  //
  // `if (!reason)` accepted it, because an object is truthy. The field was
  // built, froze clean, and rendered the literal "[object Object]" into the
  // tooltip AND into describeField -- the one sentence a screen-reader user
  // gets INSTEAD of the number. Nothing threw and nothing logged.
  for (const build of [unavailableField, notApplicableField, pendingField]) {
    assert.throws(
      () => build({ reason: 'a reason in the wrong position' }),
      /requires a non-empty string reason/,
      `${build.name} accepted an options object as its reason`,
    );
    assert.throws(
      () => build('   '),
      /requires a non-empty string reason/,
      `${build.name} accepted a blank reason`,
    );
  }
});

test('the refusal names the positional mistake, because that is the actual slip', () => {
  assert.throws(
    () => unavailableField({ reason: 'x' }),
    /reason` is the FIRST positional argument/,
    'the message must say how to fix it, not merely that it is wrong',
  );
});

test('`source` is an ENDPOINT or null, never a state or a class name', () => {
  // It used to default to the sentinels 'unavailable' / 'unknown' / 'derived'.
  // A state name in an attribution slot is unparseable as either an endpoint or
  // a class, and it is what forced panels to sniff source.startsWith('/').
  const absences = [
    unavailableField('the server hardcodes this to zero'),
    notApplicableField('this path never consults the allocator'),
    pendingField('waiting for the first poll'),
  ];
  for (const field of absences) {
    assert.equal(field.source, null, `${field.state} put a sentinel in source`);
    assert.ok(
      field.sourceClass,
      'sourceClass is the axis that answers "how do we know", and must survive',
    );
  }
});

test('describeField never interpolates a null source into its sentence', () => {
  // This is the accessible description, so a placeholder here is read aloud.
  const field = unavailableField('the server hardcodes this to zero', {
    label: 'Preemptions',
    origin: 'scatter',
  });
  const text = describeField(field);
  assert.ok(!/null|undefined/.test(text), `leaked a placeholder: ${text}`);
  assert.ok(
    !/would come from unavailable/.test(text),
    'the old sentinel is back in the accessible description',
  );
  assert.match(text, /the server hardcodes this to zero/);
});

test('a server measurement must name the endpoint that proves it', () => {
  assert.throws(
    () => measuredField(3, { sourceClass: SOURCE_CLASSES.SERVER, label: 'Queue depth' }),
    /requires a `source` endpoint/,
    'a server value with no endpoint is an unauditable claim',
  );
});

test('a client or derived measurement may legitimately have no endpoint', () => {
  // The counterpart to the rule above: requiring an endpoint from EVERY caller
  // is what pushed the sentinels into `source` in the first place.
  const client = measuredField(140, {
    source: null,
    sourceClass: SOURCE_CLASSES.CLIENT,
    label: 'TTFT',
    origin: 'scatter',
  });
  assert.equal(client.state, FIELD_STATES.MEASURED);
  assert.equal(client.source, null);

  // A client measurement still has an ORIGIN. The browser measured it, but it
  // measured it against a particular engine, and these are the hero metrics --
  // if origin were set only on server-sourced fields they would be unattributed.
  assert.equal(client.origin, 'scatter');
});
