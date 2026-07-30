// Copyright (c) Microsoft Corporation.
//
// Tests for the sparkline geometry engine.
//
// Run with:  node --test examples/serving-dashboard/dashboard/
// No npm install, no bundler, no test framework — `node:test` ships with Node.
//
// These tests exist to make the demo's honesty rules mechanically enforced
// rather than remembered. The important cases are not "does the line go up" —
// they are "does an unavailable series refuse to draw a line at zero" and "does
// a gap stay unbridged". Those are the two ways this page could lie in a chart.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  CAPTION_PENDING,
  CAPTION_UNAVAILABLE,
  describeSparkline,
  planSparkline,
} from './sparkline.js';

const NOW = 1_000_000;
const WINDOW = 60_000;
const GEOMETRY = { width: 200, height: 40, windowMs: WINDOW, nowMs: NOW };

/**
 * @param {Array<[number, number]>} pairs [msBeforeNow, value]
 * @param {object} [extra]
 */
function seriesOf(pairs, extra = {}) {
  return {
    state: 'ok',
    t: pairs.map(([ago]) => NOW - ago),
    v: pairs.map(([, value]) => value),
    ...extra,
  };
}

describe('planSparkline — the honesty rules', () => {
  it('draws NO line for an unavailable series, because a flat zero line implies a duration we never observed', () => {
    const plan = planSparkline({ state: 'unavailable', t: [], v: [], reason: 'why' }, GEOMETRY);

    assert.equal(plan.mode, 'unavailable');
    assert.deepEqual(plan.polylines, [], 'an unavailable series must produce zero polylines');
    assert.equal(plan.caption, CAPTION_UNAVAILABLE);
    assert.equal(plan.lastValue, null);
    assert.equal(plan.minValue, null, 'no axis range may be implied for data we do not have');
  });

  it('treats an unrecognised state as unavailable rather than renderable', () => {
    // The safe direction. An em-dash for a value we could have shown is
    // cosmetic; a line for a value we could not is a fabricated measurement.
    const series = { state: 'totally-new-vocabulary', t: [NOW], v: [5] };
    const plan = planSparkline(series, { ...GEOMETRY, strict: false });

    assert.equal(plan.mode, 'unavailable');
    assert.deepEqual(plan.polylines, []);

    // Under test the same drift stops the build rather than quietly hatching.
    assert.throws(() => planSparkline(series, GEOMETRY));
  });

  it('plots a series only for the ruled "ok" state', () => {
    const plan = planSparkline(seriesOf([[1000, 5], [500, 7]], { state: 'ok' }), GEOMETRY);
    assert.equal(plan.mode, 'data');

    // Both ruled spellings of the measured state draw the line. Recognising
    // only one would silently hatch a live series as NOT MEASURABLE YET the
    // moment the enum flipped -- a working metric drawn as unmeasurable.
    const alternate = planSparkline(
      seriesOf([[1000, 5], [500, 7]], { state: 'measured' }),
      GEOMETRY,
    );
    assert.equal(alternate.mode, 'data');

    // A genuinely unknown state still must not draw. That guard is the point;
    // it is only the two ratified spellings that are treated as the same word.
    const unknown = planSparkline(
      seriesOf([[1000, 5], [500, 7]], { state: 'live' }),
      { ...GEOMETRY, strict: false },
    );
    assert.notEqual(unknown.mode, 'data');
  });

  it('distinguishes pending (no samples yet) from unavailable (never coming)', () => {
    const pending = planSparkline({ state: 'ok', t: [], v: [] }, GEOMETRY);

    assert.equal(pending.mode, 'pending');
    assert.equal(pending.caption, CAPTION_PENDING);
    assert.notEqual(
      pending.caption,
      CAPTION_UNAVAILABLE,
      'a series that will fill in must not read as one that never will',
    );
  });

  it('never bridges a declared gap with a line segment', () => {
    const series = seriesOf(
      [
        [50_000, 10],
        [45_000, 12],
        [10_000, 14],
        [5_000, 15],
      ],
      { gaps: [[NOW - 44_000, NOW - 11_000]] },
    );

    const plan = planSparkline(series, GEOMETRY);

    assert.equal(plan.polylines.length, 2, 'the gap must split the line into two runs');
    assert.equal(plan.polylines[0].length, 2);
    assert.equal(plan.polylines[1].length, 2);
    assert.equal(plan.gapBands.length, 1, 'the gap must be hatched, not merely skipped');
    assert.ok(plan.gapBands[0].x1 > plan.gapBands[0].x0);
  });

  it('splits on an undeclared stall longer than three cadences, as a second line of defence', () => {
    const series = seriesOf([
      [40_000, 1],
      [39_750, 2],
      [20_000, 3],
      [19_750, 4],
    ]);

    const plan = planSparkline(series, { ...GEOMETRY, cadenceMs: 250 });

    assert.equal(
      plan.polylines.length,
      2,
      'a 19-second hole at a 250 ms cadence is a stall, and bridging it would draw data nobody measured',
    );
  });

  it('drops non-finite samples rather than charting NaN as zero', () => {
    const series = { state: 'ok', t: [NOW - 3000, NOW - 2000, NOW - 1000], v: [5, NaN, 7] };

    const plan = planSparkline(series, GEOMETRY);

    const pointCount = plan.polylines.reduce((total, line) => total + line.length, 0);
    assert.equal(pointCount, 2, 'the NaN sample must be omitted, not coerced');
  });

  it('renders a single sample as a point, not a line implying duration', () => {
    const plan = planSparkline(seriesOf([[1000, 42]]), GEOMETRY);

    assert.equal(plan.mode, 'data');
    assert.equal(plan.polylines.length, 1);
    assert.equal(plan.polylines[0].length, 1);
  });

  it('marks a stale series so a frozen number is never painted as live', () => {
    const plan = planSparkline(seriesOf([[2000, 3], [1000, 4]], { state: 'stale' }), GEOMETRY);

    assert.equal(plan.mode, 'data', 'stale data is real data and must still be shown');
    assert.equal(plan.stale, true, 'but it must be flagged so the painter can de-emphasise it');
  });
});

describe('planSparkline — geometry', () => {
  it('maps the window onto the well width with now at the right edge', () => {
    const plan = planSparkline(seriesOf([[WINDOW, 0], [0, 10]]), GEOMETRY);

    const [first, last] = plan.polylines[0];
    assert.equal(Math.round(first.x), 0);
    assert.equal(Math.round(last.x), GEOMETRY.width);
  });

  it('puts larger values higher on the canvas', () => {
    const plan = planSparkline(seriesOf([[2000, 1], [1000, 9]]), GEOMETRY);

    const [low, high] = plan.polylines[0];
    assert.ok(high.y < low.y, 'canvas y grows downward, so the larger value must have the smaller y');
  });

  it('keeps a genuinely flat series visible instead of dividing by a zero range', () => {
    const plan = planSparkline(seriesOf([[3000, 7], [2000, 7], [1000, 7]]), {
      ...GEOMETRY,
      zeroBaseline: false,
    });

    for (const point of plan.polylines[0]) {
      assert.ok(Number.isFinite(point.y), 'a flat line must not produce NaN coordinates');
    }
    assert.equal(plan.lastValue, 7);
  });

  it('excludes samples outside the window', () => {
    const plan = planSparkline(seriesOf([[WINDOW + 10_000, 99], [1000, 5]]), GEOMETRY);

    const pointCount = plan.polylines.reduce((total, line) => total + line.length, 0);
    assert.equal(pointCount, 1);
    assert.equal(plan.maxValue, 5, 'an out-of-window outlier must not inflate the axis');
  });

  it('anchors the baseline at zero for rates so noise is not exaggerated into drama', () => {
    const plan = planSparkline(seriesOf([[2000, 100], [1000, 102]]), {
      ...GEOMETRY,
      zeroBaseline: true,
    });

    assert.equal(plan.minValue, 0);
  });
});

describe('describeSparkline — AC28 chart accessibility', () => {
  it('explains WHY an unavailable chart is empty rather than saying "chart"', () => {
    const plan = planSparkline({ state: 'unavailable', t: [], v: [] }, GEOMETRY);

    const text = describeSparkline(plan, {
      label: 'Preemptions',
      reason: 'The scheduler performs preemption but keeps no counter for it.',
    });

    assert.match(text, /not measurable yet/i);
    assert.match(text, /no counter for it/i);
  });

  it('gives a sighted reader and a screen-reader user the same information', () => {
    const plan = planSparkline(seriesOf([[3000, 10], [2000, 30], [1000, 20]]), GEOMETRY);

    const text = describeSparkline(plan, { label: 'Queue depth', unit: 'requests' });

    assert.match(text, /Queue depth/);
    assert.match(text, /now 20 requests/);
    assert.match(text, /range 0 to 30 requests/);
  });

  it('announces gaps, because an unmentioned gap reads as continuous data', () => {
    const series = seriesOf([[50_000, 1], [45_000, 2], [5_000, 3]], {
      gaps: [[NOW - 44_000, NOW - 6_000]],
    });

    const text = describeSparkline(planSparkline(series, GEOMETRY), { label: 'Tokens/s' });

    assert.match(text, /1 gap where no data was collected/);
  });

  it('says a pending chart is waiting, not that it is zero', () => {
    const plan = planSparkline({ state: 'ok', t: [], v: [] }, GEOMETRY);

    const text = describeSparkline(plan, { label: 'Hit rate' });

    assert.match(text, /no samples yet/i);
    assert.doesNotMatch(text, /\b0\b/, 'a pending chart must never mention a zero value');
  });
});
