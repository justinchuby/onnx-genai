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
  staleField,
  derivedField,
  describeField,
  FIELD_STATES,
  SOURCE_CLASSES,
  withoutSourceCitations,
} from './telemetry-field.js';
import { readFileSync } from 'node:fs';

import { PROVENANCE } from './telemetry-provenance.js';

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

test('not-applicable dominates EVERY other state through derivation', () => {
  // The precedence question, asked explicitly by @c0de4c2e and @c7a654ed: when
  // a derivation mixes states, which one describes the result?
  //
  // not-applicable wins over all of them, and the reason is not symmetry --
  // unavailable, pending and stale are all statements about our MEASUREMENT
  // PIPELINE ('not plumbed', 'not polled yet', 'poll failed'), and every one of
  // them implies the number could still arrive. not-applicable is a statement
  // about the EXECUTION PATH: the question is not being asked at all, so no
  // amount of waiting or plumbing changes it.
  //
  // Reporting `pending` for a quantity that can never exist is a small lie with
  // a spinner on it -- the same argument that won `pending` its place.
  const na = () => notApplicableField('This path never consults the allocator.');
  const others = {
    unavailable: unavailableField('Not plumbed through yet.'),
    pending: pendingField('Waiting for the first poll.'),
    stale: staleField(
      measuredField(41, { source: '/v1/status', label: 'Queue depth' }),
      'The last poll did not refresh this.',
    ),
  };

  for (const [name, other] of Object.entries(others)) {
    const derived = derivedField({ 'kv.pages_used': na(), 'queue.depth': other }, () => 7, {});
    assert.equal(
      derived.state,
      FIELD_STATES.NOT_APPLICABLE,
      `a not-applicable input lost to ${name}; the result would promise a value that cannot exist`,
    );
    assert.equal(derived.value, null, `${name}: a dominated derivation must not carry a number`);
    assert.ok(derived.reason, `${name}: the result must still explain itself`);
  }
});

test('a derivation over healthy inputs is unaffected by the precedence rule', () => {
  // The meta-guard. Every assertion above passes if derivedField simply always
  // returned not-applicable, so this pins that the dominance is conditional.
  const derived = derivedField(
    {
      a: measuredField(10, { source: '/v1/status', label: 'a' }),
      b: measuredField(2, { source: '/v1/status', label: 'b' }),
    },
    ({ a, b }) => a / b,
    { unit: 'ratio' },
  );
  assert.equal(derived.state, FIELD_STATES.MEASURED);
  assert.equal(derived.value, 5);
});

test('an undated stale field is never described as freshly measured', () => {
  // F3. `describeField` used to compute its age as `nowMs - (observedAtMs ??
  // nowMs)`, so a field with NO timestamp rendered as "last measured 0s ago" --
  // perfect freshness asserted at the exact moment we know the field is stale.
  const field = {
    state: FIELD_STATES.STALE,
    value: 42,
    unit: 'req/s',
    reason: 'poll failed',
    observedAtMs: null,
    source: '/v1/status',
  };

  const text = describeField(field, 1_000_000);

  assert.match(text, /age unknown/, 'an undateable field must say its age is unknown');
  assert.doesNotMatch(
    text,
    /0s ago/,
    'an undated stale field is being spoken as if it had just been measured',
  );
});

test('a dated stale field still reports its real age', () => {
  // The anti-vacuity control. Deleting the age arithmetic entirely would
  // satisfy the assertion above perfectly.
  const field = {
    state: FIELD_STATES.STALE,
    value: 42,
    unit: 'req/s',
    reason: 'poll failed',
    observedAtMs: 1_000_000 - 12_000,
    source: '/v1/status',
  };

  assert.match(describeField(field, 1_000_000), /last measured 12s ago/);
});

test('the spoken and the visible channel agree about age', async () => {
  // THE GUARD THAT WAS MISSING. Both stacks were individually tested and
  // neither test could see the other channel, so they disagreed for as long as
  // they both existed. This asserts the PAIR, which is the only level at which
  // the defect was visible.
  const { formatField } = await import('./format.js');
  const nowMs = 1_000_000;

  for (const observedAtMs of [null, undefined, nowMs - 30_000]) {
    const field = {
      state: FIELD_STATES.STALE,
      value: 7,
      unit: 'req/s',
      reason: 'poll failed',
      observedAtMs,
      source: '/v1/status',
    };
    const spoken = describeField(field, nowMs);
    const visible = formatField(field, { nowMs }).text;
    const visibleSaysUnknown = visible.includes('age unknown');
    const spokenSaysUnknown = spoken.includes('age unknown');

    assert.equal(
      spokenSaysUnknown,
      visibleSaysUnknown,
      `the two channels disagree for observedAtMs=${String(observedAtMs)}: ` +
        `visible ${JSON.stringify(visible)} vs spoken ${JSON.stringify(spoken)}`,
    );
  }
});

// ---------------------------------------------------------------------------
// SOURCE CITATIONS REACH A VISITOR THROUGH TWO CHANNELS. ONE WAS SANITISED.
//
// app.js's provenance footer runs `entry.evidence` through a stripper before
// putting it in the Evidence column. Nothing ran `reason` through anything --
// and reason is rendered by format.js:227, telemetry-field.js:587 and
// model-card.js:98. Eight byOrigin reasons carried a raw `file.rs:232-237` all
// the way to the page.
//
// A line number is a coordinate into a tree that moves ~45 seconds per commit.
// It is useless on a projector and wrong by morning. The FILE is the argument
// and it stays; the LINE is the half that rots and it goes.
//
// Sanitised at the FAN-IN -- the four field constructors -- rather than at the
// six render sites, because a rule every renderer must remember is discipline
// and doing it once is construction.

const CITATION = /[A-Za-z0-9_\-/.]+\.(?:rs|js|toml|md):\d+(?:-\d+)?/;

test('the stripper removes the line and keeps the file', () => {
  assert.equal(
    withoutSourceCitations('see crates/onnx-genai-server/src/metrics.rs:232-237 for why'),
    'see crates/onnx-genai-server/src/metrics.rs for why',
  );
  // The half that must SURVIVE. Deleting the citation entirely would be the
  // over-correction: the reason must still say why, and the file is most of it.
  assert.match(withoutSourceCitations('metrics.rs:232'), /metrics\.rs/);
});

test('every field constructor that accepts a reason sanitises it', () => {
  const dirty = 'the counter is wrong, see crates/onnx-genai-server/src/metrics.rs:232-237';
  const built = {
    unavailableField: unavailableField(dirty),
    notApplicableField: notApplicableField(dirty),
    pendingField: pendingField(dirty),
    staleField: staleField(measuredField(1, { label: 'x', source: '/v1/status' }), dirty),
  };

  // Derived, not hardcoded: if a fifth reason-bearing constructor is exported,
  // this fails until somebody decides whether it sanitises.
  assert.deepEqual(
    Object.keys(built).sort(),
    ['notApplicableField', 'pendingField', 'staleField', 'unavailableField'],
    'a reason-bearing constructor is missing from this check',
  );

  for (const [name, field] of Object.entries(built)) {
    assert.ok(field.reason, `${name} produced no reason at all`);
    assert.doesNotMatch(
      field.reason,
      CITATION,
      `${name}() puts a source line number on the page`,
    );
    assert.match(field.reason, /metrics\.rs/, `${name}() deleted the citation instead of trimming it`);
  }
});

test('no reason in the shipped catalogue carries a line number to a visitor', () => {
  const offenders = [];
  let reasons = 0;
  for (const [key, entry] of Object.entries(PROVENANCE)) {
    for (const [origin, resolved] of Object.entries(entry.byOrigin ?? {})) {
      if (typeof resolved.reason !== 'string') continue;
      reasons += 1;
      const rendered = unavailableField(resolved.reason).reason;
      if (CITATION.test(rendered)) offenders.push(`${key} [${origin}]`);
    }
    if (typeof entry.reason === 'string') {
      reasons += 1;
      if (CITATION.test(unavailableField(entry.reason).reason)) offenders.push(key);
    }
  }

  // ANTI-VACUITY. A catalogue with no reasons cannot leak one, and this test
  // would be permanently green over a deleted byOrigin layer.
  assert.ok(reasons >= 8, `CANNOT RUN: only ${reasons} reasons inspected, expected >= 8`);
  assert.deepEqual(offenders, [], 'a source line number reaches the rendered reason text');
});

test('the Evidence column and the reason channel share ONE stripper', () => {
  // The drift guard. app.js had its own copy of this regex; two sanitisers with
  // the same job diverge the moment one is fixed, and the divergence is silent
  // because both still look like they work.
  const appSource = readFileSync(new URL('./app.js', import.meta.url), 'utf8');
  assert.match(
    appSource,
    /citationForVisitor\s*=\s*withoutSourceCitations/,
    'app.js has re-grown its own citation stripper instead of importing the shared one',
  );
  assert.doesNotMatch(
    appSource,
    /replace\(\s*\/\(\[A-Za-z0-9_/,
    'app.js carries a second copy of the citation regex',
  );
});
