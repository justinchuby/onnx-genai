// Copyright (c) Microsoft Corporation.
//
// Tests for the rendering vocabulary. Each one locks a property that, if
// broken, would put a claim on screen stronger than the evidence behind it.

import test, { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  ABSOLUTE_PATH_REASON,
  displaySafeField,
  formatField,
  formatAge,
  describeFieldText,
  SOURCE_CLASS_BADGES,
  ABSENT_TEXT,
  NOT_APPLICABLE_TEXT,
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

test('a measured false renders as literal false, never as absence', () => {
  const out = formatField(measuredField(false, { source: '/v1/status' }));
  assert.equal(out.text, 'false');
  assert.notEqual(out.text, ABSENT_TEXT);
  assert.doesNotMatch(out.text, /—/);
});

test('unavailable and not-applicable differ on the surface, not just in the hover', () => {
  const unavailable = formatField(unavailableField('The server hardcodes 0.0.'), { nowMs: NOW });
  const notApplicable = formatField(notApplicableField('This path never consults the cache.'), {
    nowMs: NOW,
  });

  // This test previously asserted both rendered `—`, with the distinction
  // carried entirely by `title`. That was the pre-ruling design and it fails
  // the bar the rest of this file is held to: a hover is invisible to a
  // visitor scanning the page, invisible on touch, invisible in a grayscale
  // screenshot, and absent from the text a table view or clipboard copy emits.
  // Putting the ONLY copy of a distinction there is the same error class as
  // encoding it in colour.
  assert.equal(unavailable.text, ABSENT_TEXT);
  assert.equal(notApplicable.text, NOT_APPLICABLE_TEXT);
  assert.notEqual(unavailable.text, notApplicable.text);

  // The hover still carries the full prose — surfacing the distinction did not
  // remove the explanation, it just stopped the explanation being the only copy.
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
  // Counted against DISTINCT wire values, not key count: `MEASURED` is a
  // transitional alias for `OK` and is the same state, so a key-count check
  // would demand a sixth sample that does not exist.
  assert.equal(
    fields.length,
    new Set(Object.values(FIELD_STATES)).size,
    'one sample per declared state',
  );
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
      // The RETIRED spelling of the measured state. Before D160 this test used
      // 'measured' as its unknown state, because the wire value was 'ok'; the
      // rename made that string valid and quietly turned this guard green while
      // testing nothing. 'ok' is now the realistic stale-producer case: a module
      // written against the old spec must render an em-dash and shout, never 999.
      state: 'ok',
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

// ---------------------------------------------------------------------------
// The five ratified states must be distinguishable IN TEXT.
//
// The bar is the grayscale screenshot rule, automated: these assertions read
// `.text` only, so a pair of states separated by nothing but a colour token
// fails here even though it would look fine on a developer's monitor.
// ---------------------------------------------------------------------------

test('not-applicable does not render as unavailable', () => {
  const base = { value: null, unit: 'reqs', label: 'prefix reuse' };

  const notApplicable = formatField({ ...base, state: 'not-applicable' });
  const unavailable = formatField({ ...base, state: 'unavailable' });

  // `—` is an apology for a metric we failed to collect. `n/a` is the statement
  // that the metric is meaningless on this execution path BY DESIGN — the
  // prefix cache is never consulted on the scatter engine, so a hit rate there
  // is not missing, it is inapplicable. Collapsing them makes a correctly
  // empty panel read as a broken one.
  assert.equal(notApplicable.text, 'n/a');
  assert.equal(unavailable.text, '—');
  assert.notEqual(notApplicable.text, unavailable.text);
});

test('all five states render distinct text, so none relies on colour', () => {
  // Derived from FIELD_STATES rather than spelled out. This test hardcoded
  // 'ok' and went red on the D160 rename -- correctly, but for the wrong
  // reason: 'ok' had become an UNKNOWN state, so it rendered an em-dash and
  // collided with unavailable. Reading the enum means the guard tracks the
  // vocabulary instead of a snapshot of it.
  const carriesValue = new Set([FIELD_STATES.MEASURED, FIELD_STATES.STALE]);
  const states = Object.values(FIELD_STATES);
  const texts = states.map(
    (state) =>
      formatField(
        {
          value: carriesValue.has(state) ? 41 : null,
          state,
          unit: 'reqs',
          label: 'queue depth',
          observedAtMs: 0,
        },
        { nowMs: 12_000 },
      ).text,
  );

  assert.equal(states.length, 5, 'FIELD_STATES no longer has five states — update this test');
  assert.equal(new Set(texts).size, texts.length, `states collapsed: ${texts.join(' | ')}`);
});

test('an undated stale reading admits its age is unknown rather than claiming 0s', () => {
  // The old default aged an undated field against `now`, producing "0s old":
  // a value asserting perfect freshness on the one path that already knows it
  // is stale. That is a stronger false claim than showing no age at all.
  const undated = formatField(
    { value: 41, state: 'stale', unit: 'reqs', label: 'queue depth' },
    { nowMs: 900_000 },
  );

  assert.equal(undated.text, '41 reqs · age unknown');
  assert.doesNotMatch(undated.text, /0s/);

  // A dated one still reports real age, so this did not disable the treatment.
  const dated = formatField(
    { value: 41, state: 'stale', unit: 'reqs', label: 'queue depth', observedAtMs: 888_000 },
    { nowMs: 900_000 },
  );
  assert.equal(dated.text, '41 reqs · 12s old');
});

describe('display-safe fields', () => {
  const measured = (value) => measuredField(value, { source: '/v1/status' });

  it('rejects absolute path shapes without retaining the sensitive value', () => {
    for (const value of [
      '/Users/presenter/models/qwen',
      '/home/presenter/models/qwen',
      'C:\\Users\\presenter\\models\\qwen',
      '\\\\fileserver\\models\\qwen',
    ]) {
      const field = measured(value);
      assert.deepEqual(displaySafeField(field), {
        ...field,
        value: null,
        state: FIELD_STATES.UNAVAILABLE,
        reason: ABSOLUTE_PATH_REASON,
      });
    }
  });

  it('removes sensitive warning metadata with the rejected path', () => {
    const value = '/Users/operator/secret/provider';
    const field = measuredField(value, {
      source: '/v1/status',
      provenanceWarning: `The server sent ${value}.`,
    });

    const safe = displaySafeField(field);
    assert.equal(safe.value, null);
    assert.equal(safe.provenanceWarning, null);
    assert.ok(!JSON.stringify(safe).includes(value), 'display-safe field retained path residue');
    assert.ok(
      !JSON.stringify(formatField(field)).includes(value),
      'formatted output retained path residue',
    );
  });

  it('preserves legitimate model identifiers exactly', () => {
    for (const value of [
      'Qwen/Qwen2.5-0.5B-Instruct',
      'models/qwen2.5-0.5b',
      'qwen-scatter',
    ]) {
      const field = measured(value);
      assert.equal(displaySafeField(field), field);
      assert.equal(displaySafeField(field).value, value);
    }
  });

  it('leaves non-string values unchanged', () => {
    for (const value of [0, 42, false]) {
      const field = measured(value);
      assert.equal(displaySafeField(field), field);
    }
    const unavailable = unavailableField('not reported');
    assert.equal(displaySafeField(unavailable), unavailable);
  });
});
