// Copyright (c) Microsoft Corporation.
//
// Tests for the state-before-value guard.
//
// Every one of these encodes a way the dashboard could print a number nobody
// measured. They are cheap to run and they are the difference between a rule
// people remember and a rule the code enforces.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  RENDER_STATES,
  isPending,
  isRenderable,
  isStale,
  isUnavailable,
  numericValueOf,
  ratioField,
  renderStateOf,
} from './field-state.js';

describe('renderStateOf', () => {
  it('renders the five ruled states and nothing else', () => {
    assert.equal(renderStateOf({ state: 'ok', value: 5 }), RENDER_STATES.OK);
    assert.equal(renderStateOf({ state: 'pending' }), RENDER_STATES.PENDING);
    assert.equal(renderStateOf({ state: 'stale', value: 5 }), RENDER_STATES.STALE);
    assert.equal(renderStateOf({ state: 'unavailable' }), RENDER_STATES.UNAVAILABLE);
    assert.equal(renderStateOf({ state: 'not-applicable' }), RENDER_STATES.NOT_APPLICABLE);
  });

  it('renders a measured value under EITHER ruled spelling', () => {
    // The wire spelling of this one state has been ruled three times in a
    // single session, in a file this dashboard does not own. Accepting both is
    // the only option that cannot blank the page: if the panels recognised
    // only the spelling that did NOT ship, every field would resolve through
    // the unknown-state path and the whole dashboard would render em-dashes
    // over a server answering perfectly, with nothing logged anywhere.
    //
    // It fabricates nothing, because the two spellings make the identical
    // claim -- neither promotes a value the server did not vouch for. Drift is
    // caught by state-vocabulary.test.js, which drives the real store.
    assert.equal(renderStateOf({ state: 'ok', value: 5 }), RENDER_STATES.OK);
    assert.equal(renderStateOf({ state: 'measured', value: 5 }), RENDER_STATES.OK);
  });

  it('THROWS on an unrecognised state under test, naming the ruled vocabulary', () => {
    // Under `node --test` an unknown state means a producer has drifted from
    // the ruled five. Degrading it to an em-dash is how a real measurement
    // vanishes for an hour with nothing in any log; stopping the build is the
    // only thing that reliably gets it fixed.
    assert.throws(
      () => renderStateOf({ state: 'live', value: 5 }),
      (error) => {
        assert.match(error.message, /Unrecognised field state "live"/);
        // The message must name the legal values, or the reader's next step is
        // to go source-diving for an allow-list they did not know existed.
        assert.match(error.message, /not-applicable/);
        return true;
      },
    );
  });

  it('degrades to an em-dash instead of throwing when NOT strict (the browser)', () => {
    // The branch that runs in front of an audience. A throw here white-screens
    // a panel mid-demo; an em-dash is the honest rendering of "we do not know
    // what this is". Tested explicitly because the default under Node is the
    // other branch, which would otherwise leave this one never executed.
    assert.equal(
      renderStateOf({ state: 'live', value: 5 }, { strict: false }),
      RENDER_STATES.UNAVAILABLE,
    );
    assert.equal(
      renderStateOf({ state: '', value: 5 }, { strict: false }),
      RENDER_STATES.UNAVAILABLE,
    );
  });

  it('treats a missing or malformed field as unavailable rather than throwing', () => {
    assert.equal(renderStateOf(null), RENDER_STATES.UNAVAILABLE);
    assert.equal(renderStateOf(undefined), RENDER_STATES.UNAVAILABLE);
    assert.equal(renderStateOf({}), RENDER_STATES.UNAVAILABLE);
  });

  it('refuses to call a valueless field measured, even when the store says so', () => {
    // A store bug must not become a rendered "null" or a silent coercion to 0.
    assert.equal(renderStateOf({ state: 'ok', value: null }), RENDER_STATES.UNAVAILABLE);
    assert.equal(renderStateOf({ state: 'measured', value: undefined }), RENDER_STATES.UNAVAILABLE);
  });

  it('preserves a genuine zero as a measurement', () => {
    // This is the other half of the honesty rule and it is just as important:
    // `rejections: 0` is a real, good zero and must render at full contrast.
    assert.equal(renderStateOf({ state: 'ok', value: 0 }), RENDER_STATES.OK);
    assert.equal(isRenderable({ state: 'ok', value: 0 }), true);
    assert.equal(numericValueOf({ state: 'ok', value: 0 }), 0);
  });
});

describe('isRenderable — the guard panels reach values through', () => {
  it('permits measured and stale, and refuses unavailable and pending', () => {
    assert.equal(isRenderable({ state: 'ok', value: 1 }), true);
    assert.equal(isRenderable({ state: 'stale', value: 1 }), true);
    assert.equal(isRenderable({ state: 'unavailable', value: null }), false);
    assert.equal(isRenderable({ state: 'pending' }), false);
  });

  it('separates pending from unavailable, because only one of them resolves itself', () => {
    assert.equal(isPending({ state: 'pending' }), true);
    assert.equal(isUnavailable({ state: 'pending' }), false);
    assert.equal(isUnavailable({ state: 'unavailable', value: null }), true);
    assert.equal(isPending({ state: 'unavailable', value: null }), false);
  });

  it('flags stale so a frozen number is never presented as current', () => {
    assert.equal(isStale({ state: 'stale', value: 42 }), true);
    assert.equal(isStale({ state: 'ok', value: 42 }), false);
  });
});

describe('numericValueOf', () => {
  it('returns null rather than a number for anything not renderable', () => {
    assert.equal(numericValueOf({ state: 'unavailable', value: null }), null);
    assert.equal(numericValueOf({ state: 'pending' }), null);
    assert.equal(numericValueOf(null), null);
  });

  it('returns null for a non-numeric value instead of NaN', () => {
    // NaN propagates silently through arithmetic and surfaces as a broken chart
    // far from its cause; null fails where it is used.
    assert.equal(numericValueOf({ state: 'ok', value: 'unknown' }), null);
    assert.equal(numericValueOf({ state: 'ok', value: Infinity }), null);
  });

  it('reads a real number through', () => {
    assert.equal(numericValueOf({ state: 'ok', value: 12.5 }), 12.5);
    assert.equal(numericValueOf({ state: 'stale', value: 3 }), 3);
  });
});

describe('ratioField — the batch-occupancy trap (demo-ux.md §5.3)', () => {
  const measured = (value) => ({ state: 'ok', value });

  it('refuses to form a ratio when the denominator is unavailable', () => {
    // This is the exact scenario: active_batch_size is REAL, max_batch is not
    // surfaced. Substituting the DEFAULT_MAX_BATCH = 4 literal from state.rs:25
    // would be a fabricated measurement wearing a division sign.
    const occupancy = ratioField(measured(6), { state: 'unavailable', value: null });

    assert.equal(occupancy.state, RENDER_STATES.UNAVAILABLE);
    assert.equal(occupancy.value, null, 'no percentage may be invented from a real numerator alone');
  });

  it('refuses to divide by a measured zero instead of returning Infinity', () => {
    const occupancy = ratioField(measured(6), measured(0));

    assert.equal(occupancy.state, RENDER_STATES.UNAVAILABLE);
    assert.equal(occupancy.value, null);
  });

  it('renders a small-denominator ratio as n of m, never as a percentage', () => {
    // D116. `75%` is arithmetically correct and epistemically false: it invents
    // resolution the quantity does not have. It invites the reader to expect
    // `76%`, and makes 50->75 read as a smooth 25-point move rather than as ONE
    // sequence entering the batch. It also silently means something different
    // if `--max-batch` changes, while `3 of 4` stays honest.
    const occupancy = ratioField(measured(3), measured(4));

    assert.equal(occupancy.state, RENDER_STATES.OK);
    assert.equal(occupancy.value, 3);
    assert.equal(occupancy.numerator, 3);
    assert.equal(occupancy.denominator, 4);
    assert.equal(occupancy.unit, 'of 4');
    assert.notEqual(occupancy.value, 75);
    assert.equal(occupancy.source, 'derived', 'a computed ratio must be badged as derived, not server');

    // Geometry still gets a true fraction, so a bar drawn from this ratio
    // cannot contradict the text printed beside it.
    assert.equal(occupancy.fraction, 0.75);
  });

  it('still uses a percentage where the denominator is a genuine continuum', () => {
    // The block grid is ~14,612 pages: there, a percentage is the honest
    // format and `9137 of 14612` is the unreadable one.
    const utilization = ratioField(measured(7306), measured(14_612));

    assert.equal(utilization.unit, '%');
    assert.equal(utilization.value, 50);
    assert.equal(utilization.numerator, null, 'a continuum ratio has no n-of-m form');
  });

  it('propagates staleness, because a ratio is only as fresh as its stalest input', () => {
    const occupancy = ratioField({ state: 'stale', value: 2 }, measured(4));

    assert.equal(occupancy.state, RENDER_STATES.STALE);
    assert.equal(occupancy.value, 2);
    assert.equal(occupancy.denominator, 4);
  });

  it('carries a reason a visitor can act on when it declines', () => {
    const occupancy = ratioField(measured(6), { state: 'unavailable', value: null }, {
      unavailableReason: "Occupancy needs the server's max batch size, which isn't surfaced.",
    });

    assert.match(occupancy.reason, /max batch size/);
  });
});
