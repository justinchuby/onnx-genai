// Copyright (c) Microsoft Corporation.
//
// The fake store is a fixture, and a fixture that disagrees with the shipping
// code is worse than no fixture at all: it is a green test asserting against a
// value the real page would refuse to render.
//
// That is not hypothetical. `measured()` used to hand-write `state: 'ok'`,
// which was the enum's wire value BEFORE it was renamed (see the landmine note
// in telemetry-field.js, where `MEASURED: 'ok'` is recorded as a mistake that
// was corrected). The rename landed; the fixture did not follow. So the fixture
// kept minting fields carrying a retired spelling, and `formatField` — which
// validates against the CURRENT enum — refused every one of them, returning an
// em-dash and `hasValue: false`.
//
// It stayed invisible because the panel render path (`renderField`) normalises
// the state before formatting, and normalisation still accepts the old
// spelling. So the panels looked fine and only the direct-format path was
// affected. A defect that is real, silent, and reachable only through one of
// two doors is exactly the kind that survives review.
//
// These tests pin the invariant the rename broke: A FIELD THE FIXTURE MINTS
// MUST BE A FIELD THE FORMATTER HONOURS. The negative control exists because
// the positive arm alone would pass just as happily if `formatField` had
// stopped validating anything at all.

import test from 'node:test';
import assert from 'node:assert/strict';

import { formatField } from '../../format.js';
import { FIELD_STATES } from '../../telemetry-field.js';
import { measured } from './fake-store.js';

// The spelling the enum used to carry. Kept here, and ONLY here, so the
// negative control has something concrete to refuse. It is deliberately a
// literal: importing it from anywhere would mean it still exists somewhere as
// a supported value, which is the situation this file exists to prevent.
const RETIRED_MEASURED_SPELLING = 'ok';

test('a field minted by the fixture is one the formatter will render', () => {
  const formatted = formatField(measured(42, { unit: 'ms' }));

  assert.equal(
    formatted.hasValue,
    true,
    'formatField refused a field built by the fake store. The fixture and the ' +
      'formatter disagree about how a measured field is spelled, so every test ' +
      'that formats a fixture field is asserting against an em-dash.',
  );
  assert.match(
    formatted.text,
    /42/,
    'the formatted text should contain the value the fixture was given',
  );
});

test('the fixture emits the current enum value, not a string that resembles it', () => {
  assert.equal(
    measured(42).state,
    FIELD_STATES.MEASURED,
    'the fake store should use the FIELD_STATES constant rather than hand-writing ' +
      'the string. A hand-written state is a second, silent implementation of the ' +
      'enum, and it does not move when the enum does.',
  );
});

test('the formatter still refuses the retired spelling, so the check above is not vacuous', () => {
  const stale = formatField({
    value: 42,
    state: RETIRED_MEASURED_SPELLING,
    unit: 'ms',
    source: 'server',
    label: '',
  });

  assert.equal(
    stale.hasValue,
    false,
    'formatField accepted the retired state spelling. That means it is no longer ' +
      'validating states, and the passing test above proves nothing — it would ' +
      'pass for any string at all.',
  );
});

test('the two spellings produce different results, which is what makes this measurable', () => {
  const current = formatField(measured(42, { unit: 'ms' }));
  const retired = formatField({
    value: 42,
    state: RETIRED_MEASURED_SPELLING,
    unit: 'ms',
    source: 'server',
    label: '',
  });

  assert.notEqual(
    current.text,
    retired.text,
    'the current and retired spellings format identically. Either the formatter ' +
      'has stopped discriminating between them or the fixture has regressed to ' +
      'the retired one; in both cases this file can no longer detect the defect ' +
      'it was written to catch.',
  );
});
