// Copyright (c) Microsoft Corporation.
//
// Tests for the rendering vocabulary. Each one locks a property that, if
// broken, would put a claim on screen stronger than the evidence behind it.

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  formatField,
  formatAge,
  describeFieldText,
  SOURCE_CLASS_BADGES,
  ABSENT_TEXT,
  PENDING_TEXT,
} from './format.js';
import {
  measuredField,
  unavailableField,
  notApplicableField,
  pendingField,
  staleField,
  FIELD_STATES,
  SOURCE_CLASSES,
} from './telemetry-field.js';

const NOW = 1_700_000_000_000;

test('a measured value renders with its unit and a server badge', () => {
  const out = formatField(measuredField(41, { source: '/v1/status', unit: 'req' }), { nowMs: NOW });
  assert.equal(out.text, '41 req');
  assert.equal(out.badge, 'ˢ');
  assert.equal(out.hasValue, true);
});

test('a measured zero renders as a stark 0, never as absence', () => {
  // The single most important line in this file. A real zero is DATA, and
  // hiding it is the mirrored fabrication of printing a stub.
  const out = formatField(measuredField(0, { source: '/v1/debug/kv', unit: 'hits' }));
  assert.equal(out.text, '0 hits');
  assert.notEqual(out.text, ABSENT_TEXT);
});

test('unavailable and not-applicable both render an em-dash but read differently', () => {
  const unavailable = formatField(unavailableField('The server hardcodes 0.0.'), { nowMs: NOW });
  const notApplicable = formatField(notApplicableField('This path never consults the cache.'), {
    nowMs: NOW,
  });

  // Identical on the surface: absence looks the same everywhere in the page.
  assert.equal(unavailable.text, ABSENT_TEXT);
  assert.equal(notApplicable.text, ABSENT_TEXT);

  // ...but the hover must not collapse the distinction between "not built yet"
  // and "meaningless to ask here". One is a gap, the other is architecture.
  assert.match(unavailable.title, /Unavailable/);
  assert.match(notApplicable.title, /Not applicable/);
  assert.notEqual(unavailable.title, notApplicable.title);
});

test('pending is visually distinct from absence', () => {
  // Pending resolves by itself; unavailable never will. A visitor waiting for
  // a number that is never coming has been misled.
  const out = formatField(pendingField('Waiting for the first poll.'));
  assert.equal(out.text, PENDING_TEXT);
  assert.notEqual(out.text, ABSENT_TEXT);
});

test('a stale value states its age in words, not in colour alone', () => {
  // AC25. A colour shift vanishes in grayscale and for colourblind readers,
  // and "12 seconds out of date" is information the visitor needs.
  const measured = measuredField(41, { source: '/v1/status', observedAtMs: NOW - 12_000 });
  const out = formatField(staleField(measured, 'The server stopped responding.'), { nowMs: NOW });

  assert.equal(out.text, '41 · 12s old');
  assert.equal(out.ageText, '12s old');
  assert.equal(out.state, FIELD_STATES.STALE);
  assert.equal(out.hasValue, true, 'the last good value is still shown');
});

test('stale age degrades to coarser units rather than faking precision', () => {
  assert.equal(formatAge(0), '0s old');
  assert.equal(formatAge(59_000), '59s old');
  assert.equal(formatAge(374_000), '6m old');
  assert.equal(formatAge(7_200_000), '2h old');
});

test('an estimate is marked with a tilde AND a badge, visible without hovering', () => {
  // An estimate that looks identical to a measurement IS a fabricated
  // measurement, however carefully the tooltip is worded -- nobody hovers.
  const out = formatField(
    measuredField(0.8, {
      source: 'derived',
      sourceClass: SOURCE_CLASSES.ESTIMATED,
      unit: 's',
    }),
  );
  assert.equal(out.text, '~0.80 s');
  assert.equal(out.badge, 'ᴱ');
  assert.equal(out.isEstimate, true);
});

test('every source class has a distinct badge glyph', () => {
  const glyphs = Object.values(SOURCE_CLASS_BADGES).map((b) => b.glyph);
  assert.equal(new Set(glyphs).size, glyphs.length, 'badges must not collide');
  for (const cls of Object.values(SOURCE_CLASSES)) {
    assert.ok(SOURCE_CLASS_BADGES[cls], `${cls} has no badge, so AC7 cannot be satisfied for it`);
  }
});

test('a derived value is marked as derived, not as a server reading', () => {
  const out = formatField(
    measuredField(120, {
      source: 'derived',
      sourceClass: SOURCE_CLASSES.DERIVED,
      unit: 'tok/s',
      derivedFrom: ['metrics.tokens_total'],
    }),
  );
  assert.equal(out.badge, 'ᴰ');
  assert.match(out.title, /derived from/);
});

test('a field contradicting the provenance audit carries the warning through', () => {
  const out = formatField(
    measuredField(0.42, {
      source: '/v1/status',
      provenanceWarning: 'kv.usage is classified DOCUMENTED_ZERO but the server sent 0.42.',
    }),
  );
  assert.equal(out.text, '0.42');
  assert.match(out.provenanceWarning, /DOCUMENTED_ZERO/);
});

test('describeFieldText reads as prose for every state', () => {
  const source = '/v1/status';
  assert.equal(
    describeFieldText('Queue depth', measuredField(3, { source, unit: 'requests' })),
    'Queue depth is 3 requests',
  );
  assert.equal(
    describeFieldText('Prefix cache hits', notApplicableField('this path never consults it')),
    'Prefix cache hits does not apply here: this path never consults it',
  );
  assert.equal(
    describeFieldText('KV usage', unavailableField('the server hardcodes 0.0')),
    'KV usage is unavailable: the server hardcodes 0.0',
  );
  assert.equal(
    describeFieldText('Throughput', pendingField('waiting for two samples')),
    'Throughput is still loading',
  );
});

test('describeFieldText says "estimated at" rather than "is"', () => {
  // "Time saved is 0.8s" asserts a measurement. "estimated at ~0.8s" does not.
  const field = measuredField(0.8, {
    source: 'derived',
    sourceClass: SOURCE_CLASSES.ESTIMATED,
    unit: 's',
  });
  assert.equal(describeFieldText('Time saved', field), 'Time saved estimated at ~0.80 s');
});

test('no state falls through to a blank or an undefined', () => {
  // The guard on the guard: a state added later must not render as empty.
  const fields = [
    measuredField(1, { source: '/x' }),
    unavailableField('r'),
    notApplicableField('r'),
    pendingField('r'),
    staleField(measuredField(1, { source: '/x' }), 'r'),
  ];
  assert.equal(fields.length, Object.keys(FIELD_STATES).length, 'one sample per declared state');
  for (const field of fields) {
    const out = formatField(field, { nowMs: NOW });
    assert.ok(out.text && out.text.trim().length > 0, `${field.state} rendered blank`);
    assert.ok(!out.text.includes('undefined'), `${field.state} leaked undefined`);
    assert.ok(out.title && out.title.length > 0, `${field.state} has no hover text`);
  }
});

test('an unknown state is refused, never rendered as a measurement', () => {
  // @0837fdf9 caught this in the old formatFieldText: it branched on the
  // states it knew and then fell through to `return format(field.value)`, so a
  // typo or a module written against an older spec rendered its value as
  // though it had been measured. A default branch that renders as fine is how
  // AC6 dies quietly, so this one refuses.
  const errors = [];
  const originalError = console.error;
  console.error = (...args) => errors.push(args.join(' '));
  let out;
  try {
    out = formatField({
      value: 999,
      state: 'measured', // the value a stale spec would produce
      source: '/v1/status',
      sourceClass: SOURCE_CLASSES.SERVER,
      label: 'Queue depth',
      unit: 'requests',
      reason: null,
      observedAtMs: NOW,
    });
  } finally {
    console.error = originalError;
  }

  assert.equal(out.text, ABSENT_TEXT, 'must not render the value');
  assert.ok(!out.text.includes('999'), 'the unverified value must not reach the screen');
  assert.equal(out.hasValue, false);
  assert.equal(errors.length, 1, 'and it must be loud, not silent');
  assert.match(errors[0], /unknown field state/);
});

test('a custom formatter owns the whole string; the unit is not appended twice', () => {
  // Found in a browser, not in a test: model-card's formatTokenCount returns
  // "32,768 tokens" and formatField appended field.unit on top, rendering
  // "32,768 tokens tokens" on the live page while every unit test passed.
  const field = measuredField(32768, { source: '/v1/debug/config', unit: 'tokens' });
  const out = formatField(field, { format: (v) => `${Number(v).toLocaleString()} tokens` });
  assert.equal(out.text, '32,768 tokens');

  // The default formatter still appends the unit, as every panel expects.
  assert.equal(formatField(field).text, '32768 tokens');

  // ...and an explicit request still wins, in either direction.
  assert.equal(formatField(field, { withUnit: false }).text, '32768');
});
