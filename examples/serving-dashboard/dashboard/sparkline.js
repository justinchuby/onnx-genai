// Copyright (c) Microsoft Corporation.
//
// Sparkline rendering for the dashboard panels.
//
// This module implements demo-ux.md §4.3 — "Unavailable in charts — the hard
// case" — which is the sharpest honesty requirement in the whole design:
//
//   "A flat line at zero is the most dangerous single mark this page could
//    render, because unlike a 0 in a table it also implies DURATION — it says
//    'we watched this for sixty seconds and it was zero the whole time.'"
//
// So this renderer refuses to draw one. An unavailable series gets a hatched
// well and a NOT MEASURABLE YET caption; a pending series gets a faint baseline
// and AWAITING DATA; and a gap in an otherwise real series is never bridged,
// because interpolating across a gap fabricates the most convincing kind of
// false data — data that looks continuous.
//
// STRUCTURE, and why: the geometry is computed by pure functions
// ({@link planSparkline}) and only then handed to a painter
// ({@link paintSparkline}) that touches the canvas. That split is not
// ceremony — it means every honesty rule above is unit-testable under plain
// `node --test` with no DOM, no canvas, no bundler and no npm install. A rule
// that cannot be tested is a rule that erodes.

import { RENDER_STATES, renderStateOf } from './field-state.js';

/**
 * How a series well should be painted.
 *
 * @typedef {'data' | 'unavailable' | 'pending'} SparklineMode
 */

/**
 * A series as handed over by the telemetry store.
 *
 * @typedef {object} Series
 * @property {string} state             Field state vocabulary; see field-state.js.
 * @property {ArrayLike<number>} t      Sample timestamps, ms epoch, ascending.
 * @property {ArrayLike<number>} v      Sample values, parallel to `t`.
 * @property {Array<[number, number]>} [gaps]
 *   Half-open [startMs, endMs) ranges the store knows it has no data for.
 * @property {string} [unit]
 * @property {string} [label]
 * @property {string} [reason]          Required when the series is not `ok`.
 */

/**
 * The pure geometry result. Everything the painter needs; nothing it doesn't.
 *
 * @typedef {object} SparklinePlan
 * @property {SparklineMode} mode
 * @property {number} width
 * @property {number} height
 * @property {Array<Array<{x: number, y: number}>>} polylines
 *   One entry per contiguous run of samples. NEVER joined across a gap.
 * @property {Array<{x0: number, x1: number}>} gapBands
 *   Pixel-space bands to hatch inside an otherwise-real series.
 * @property {number|null} minValue
 * @property {number|null} maxValue
 * @property {number|null} lastValue
 * @property {string|null} caption      Well caption, already upper-cased.
 * @property {boolean} stale
 */

/** Caption for a series the server cannot measure. demo-ux.md §4.3, verbatim. */
export const CAPTION_UNAVAILABLE = 'NOT MEASURABLE YET';

/** Caption for a measurable series with no samples yet. demo-ux.md §4.3, verbatim. */
export const CAPTION_PENDING = 'AWAITING DATA';

/**
 * Never bridge a sample interval longer than this multiple of the expected
 * cadence, even if the store did not declare a gap.
 *
 * The store owns gap declaration and normally gets it right. This is a second
 * line of defence for the one case that matters: a poll stall the store did not
 * classify. Bridging it would draw a smooth line across seconds in which we
 * measured nothing — a fabrication with no author, which is the hardest kind to
 * catch in review.
 */
const IMPLICIT_GAP_CADENCE_MULTIPLE = 3;

/**
 * Compute sparkline geometry. Pure: no DOM, no canvas, no clock.
 *
 * @param {Series|null|undefined} series
 * @param {object} options
 * @param {number} options.width         CSS pixels of the plot well.
 * @param {number} options.height        CSS pixels of the plot well.
 * @param {number} options.windowMs      Time window the well represents, e.g. 60_000.
 * @param {number} options.nowMs         Right edge of the window.
 * @param {number} [options.cadenceMs]
 *   Expected sample spacing, for the undeclared-stall defence. When omitted it
 *   is inferred from the median observed interval, which is more robust than a
 *   default: series on this page run at wildly different rates (4 Hz polled
 *   metrics, per-request events, 1 Hz system stats) and a single hardcoded
 *   cadence would shred the slow ones into disconnected fragments.
 * @param {number} [options.padY]        Vertical inset so the stroke isn't clipped.
 * @param {boolean} [options.zeroBaseline]
 *   Anchor the value axis at zero. Correct for counts and rates, where a
 *   floating baseline exaggerates noise into drama. Wrong for latencies, where
 *   the interesting variation lives far above zero.
 * @returns {SparklinePlan}
 */
export function planSparkline(series, options) {
  const { width, height, windowMs, nowMs, cadenceMs, padY = 2, zeroBaseline = true } = options;

  const state = renderStateOf(seriesAsField(series));

  if (state === RENDER_STATES.UNAVAILABLE) {
    return emptyPlan('unavailable', width, height, CAPTION_UNAVAILABLE, false);
  }

  const times = series?.t ?? [];
  const values = series?.v ?? [];
  const sampleCount = Math.min(times.length, values.length);

  if (state === RENDER_STATES.PENDING || sampleCount === 0) {
    return emptyPlan('pending', width, height, CAPTION_PENDING, false);
  }

  const windowStartMs = nowMs - windowMs;
  const visible = collectVisibleSamples(times, values, sampleCount, windowStartMs, nowMs);

  if (visible.length === 0) {
    return emptyPlan('pending', width, height, CAPTION_PENDING, state === RENDER_STATES.STALE);
  }

  const { minValue, maxValue } = valueBounds(visible, zeroBaseline);
  const toX = (timeMs) => ((timeMs - windowStartMs) / windowMs) * width;
  const toY = makeValueToY(minValue, maxValue, height, padY);

  const declaredGaps = normaliseGaps(series?.gaps);
  const effectiveCadenceMs = cadenceMs ?? medianInterval(visible);
  const implicitGapMs = effectiveCadenceMs * IMPLICIT_GAP_CADENCE_MULTIPLE;

  const polylines = [];
  let current = [];
  for (let index = 0; index < visible.length; index += 1) {
    const sample = visible[index];
    if (index > 0) {
      const previous = visible[index - 1];
      const interrupted =
        sample.t - previous.t > implicitGapMs ||
        spansDeclaredGap(declaredGaps, previous.t, sample.t);
      if (interrupted && current.length > 0) {
        polylines.push(current);
        current = [];
      }
    }
    current.push({ x: toX(sample.t), y: toY(sample.v) });
  }
  if (current.length > 0) {
    polylines.push(current);
  }

  const gapBands = declaredGaps
    .filter((gap) => gap.endMs > windowStartMs && gap.startMs < nowMs)
    .map((gap) => ({
      x0: toX(Math.max(gap.startMs, windowStartMs)),
      x1: toX(Math.min(gap.endMs, nowMs)),
    }))
    .filter((band) => band.x1 > band.x0);

  return {
    mode: 'data',
    width,
    height,
    polylines,
    gapBands,
    minValue,
    maxValue,
    lastValue: visible[visible.length - 1].v,
    caption: null,
    stale: state === RENDER_STATES.STALE,
    // Retained so the AC28 table alternative is built from the SAME windowed
    // samples the canvas paints. Deriving the table from the raw series
    // separately would let the two disagree about the window, and a table that
    // contradicts the chart beside it is worse than no table.
    samples: visible,
  };
}

/**
 * Paint a plan onto a canvas.
 *
 * Handles device-pixel-ratio scaling itself so panels never have to think about
 * it — a sparkline that is soft on a retina projector reads as sloppiness and
 * undermines the "these are real measurements" claim more than it should.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {SparklinePlan} plan
 * @param {object} [options]
 * @param {string} [options.stroke]      Line colour. Default `--og-info`.
 * @param {number} [options.lineWidth]
 * @param {(token: string) => string} [options.readToken]
 *   Resolve a CSS custom property. Injected so the painter can be exercised
 *   without a live stylesheet.
 * @returns {void}
 */
export function paintSparkline(canvas, plan, options = {}) {
  const context = canvas.getContext('2d');
  if (!context) {
    return;
  }
  const readToken = options.readToken ?? makeTokenReader(canvas);
  const ratio = canvas.ownerDocument?.defaultView?.devicePixelRatio ?? 1;

  canvas.width = Math.max(1, Math.round(plan.width * ratio));
  canvas.height = Math.max(1, Math.round(plan.height * ratio));
  canvas.style.width = `${plan.width}px`;
  canvas.style.height = `${plan.height}px`;

  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, plan.width, plan.height);

  if (plan.mode === 'unavailable') {
    paintHatchedWell(context, 0, plan.width, plan.height, readToken);
    paintCaption(context, plan, readToken('--og-unavail-label') || '#6e7d8c');
    return;
  }

  if (plan.mode === 'pending') {
    // A faint baseline, NOT a zero line: it sits at the well's floor as a ruler,
    // is drawn in the pending ink rather than the data ink, and carries the
    // AWAITING DATA caption so it cannot be read as "the value is zero".
    context.strokeStyle = readToken('--og-fg-faint') || '#3d4855';
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(0, plan.height - 0.5);
    context.lineTo(plan.width, plan.height - 0.5);
    context.stroke();
    paintCaption(context, plan, readToken('--og-pending-fg') || '#4a5560');
    return;
  }

  for (const band of plan.gapBands) {
    paintHatchedWell(context, band.x0, band.x1 - band.x0, plan.height, readToken);
    // The vertical dashed rules are what tell a reader the line STOPPED rather
    // than the data being flat there (demo-ux.md §4.3).
    paintGapRule(context, band.x0, plan.height, readToken);
    paintGapRule(context, band.x1, plan.height, readToken);
  }

  context.strokeStyle = options.stroke ?? readToken('--og-info') ?? '#56b4e9';
  context.lineWidth = options.lineWidth ?? 1.5;
  context.lineJoin = 'round';
  context.lineCap = 'round';
  context.globalAlpha = plan.stale ? 0.45 : 1;

  for (const polyline of plan.polylines) {
    if (polyline.length === 1) {
      // A single sample is a point, not a line. Drawing a 1px line for it would
      // imply a duration we did not observe.
      const point = polyline[0];
      context.beginPath();
      context.arc(point.x, point.y, (options.lineWidth ?? 1.5) / 1.5, 0, Math.PI * 2);
      context.fillStyle = context.strokeStyle;
      context.fill();
      continue;
    }
    context.beginPath();
    context.moveTo(polyline[0].x, polyline[0].y);
    for (let index = 1; index < polyline.length; index += 1) {
      context.lineTo(polyline[index].x, polyline[index].y);
    }
    context.stroke();
  }
  context.globalAlpha = 1;
}

/**
 * The screen-reader description of a sparkline (AC28).
 *
 * Charts are the least accessible thing on this page, and an `aria-label` of
 * "chart" is compliance theatre. This produces a sentence with the same
 * information a sighted reader takes from the shape.
 *
 * @param {SparklinePlan} plan
 * @param {object} context
 * @param {string} context.label
 * @param {string} [context.unit]
 * @param {number} [context.windowSeconds]
 * @param {string} [context.reason] Required when the plan is not `data`.
 * @param {(value: number) => string} [context.format]
 * @returns {string}
 */
export function describeSparkline(plan, context) {
  const { label, unit = '', windowSeconds = 60, reason, format = defaultFormat } = context;
  const unitSuffix = unit ? ` ${unit}` : '';

  if (plan.mode === 'unavailable') {
    return `${label}: not measurable yet. ${reason ?? ''}`.trim();
  }
  if (plan.mode === 'pending') {
    return `${label}: no samples yet in the last ${windowSeconds} seconds.`;
  }

  const parts = [
    `${label} over the last ${windowSeconds} seconds:`,
    `now ${format(plan.lastValue ?? 0)}${unitSuffix},`,
    `range ${format(plan.minValue ?? 0)} to ${format(plan.maxValue ?? 0)}${unitSuffix}.`,
  ];
  if (plan.gapBands.length > 0) {
    parts.push(
      `${plan.gapBands.length} gap${plan.gapBands.length === 1 ? '' : 's'} where no data was collected.`,
    );
  }
  if (plan.stale) {
    parts.push('The most recent poll did not refresh this series.');
  }
  return parts.join(' ');
}

// ── internals ────────────────────────────────────────────────────────────────

/** @param {Series|null|undefined} series */
function seriesAsField(series) {
  if (!series) {
    return null;
  }
  // A series carries no scalar `value`, so give the state normaliser a sentinel
  // that cannot be mistaken for missing data.
  return { state: series.state, value: series.t ?? [] };
}

/**
 * @param {SparklineMode} mode
 * @param {number} width
 * @param {number} height
 * @param {string} caption
 * @param {boolean} stale
 * @returns {SparklinePlan}
 */
function emptyPlan(mode, width, height, caption, stale) {
  return {
    mode,
    width,
    height,
    polylines: [],
    gapBands: [],
    minValue: null,
    maxValue: null,
    lastValue: null,
    caption,
    stale,
    samples: [],
  };
}

/**
 * @param {ArrayLike<number>} times
 * @param {ArrayLike<number>} values
 * @param {number} count
 * @param {number} windowStartMs
 * @param {number} nowMs
 */
function collectVisibleSamples(times, values, count, windowStartMs, nowMs) {
  const visible = [];
  for (let index = 0; index < count; index += 1) {
    const t = times[index];
    const v = values[index];
    if (t < windowStartMs || t > nowMs) {
      continue;
    }
    if (!Number.isFinite(v)) {
      // A non-finite sample is not a value. Skipping it opens a gap, which is
      // the honest rendering; charting NaN as 0 is not.
      continue;
    }
    visible.push({ t, v });
  }
  return visible;
}

/**
 * The median gap between consecutive samples — the series' own natural cadence.
 *
 * Median rather than mean precisely because the thing we are trying to detect
 * is an outlier interval, and a mean is dragged toward the outlier it is
 * supposed to expose.
 *
 * @param {Array<{t: number, v: number}>} samples
 * @returns {number} Milliseconds. Falls back to 250 when fewer than 2 samples.
 */
function medianInterval(samples) {
  if (samples.length < 2) {
    return 250;
  }
  const intervals = [];
  for (let index = 1; index < samples.length; index += 1) {
    intervals.push(samples[index].t - samples[index - 1].t);
  }
  intervals.sort((a, b) => a - b);
  const middle = Math.floor(intervals.length / 2);
  const median =
    intervals.length % 2 === 0
      ? (intervals[middle - 1] + intervals[middle]) / 2
      : intervals[middle];
  return median > 0 ? median : 250;
}

/**
 * @param {Array<{t: number, v: number}>} samples
 * @param {boolean} zeroBaseline
 */
function valueBounds(samples, zeroBaseline) {
  let minValue = Infinity;
  let maxValue = -Infinity;
  for (const sample of samples) {
    if (sample.v < minValue) minValue = sample.v;
    if (sample.v > maxValue) maxValue = sample.v;
  }
  if (zeroBaseline) {
    minValue = Math.min(0, minValue);
  }
  if (minValue === maxValue) {
    // A genuinely flat series is real data and must still be visible. Give it a
    // symmetric band so it renders as a centred flat line rather than being
    // divided by a zero range.
    const padding = Math.abs(maxValue) > 0 ? Math.abs(maxValue) * 0.5 : 1;
    minValue -= padding;
    maxValue += padding;
  }
  return { minValue, maxValue };
}

/**
 * @param {number} minValue
 * @param {number} maxValue
 * @param {number} height
 * @param {number} padY
 */
function makeValueToY(minValue, maxValue, height, padY) {
  const span = maxValue - minValue;
  const usable = Math.max(1, height - padY * 2);
  return (value) => padY + usable - ((value - minValue) / span) * usable;
}

/** @param {Array<[number, number]>|undefined} gaps */
function normaliseGaps(gaps) {
  if (!Array.isArray(gaps)) {
    return [];
  }
  return gaps
    .filter((gap) => Array.isArray(gap) && gap.length === 2 && Number.isFinite(gap[0]) && Number.isFinite(gap[1]))
    .map(([startMs, endMs]) => ({ startMs, endMs }))
    .filter((gap) => gap.endMs > gap.startMs);
}

/**
 * @param {Array<{startMs: number, endMs: number}>} gaps
 * @param {number} fromMs
 * @param {number} toMs
 */
function spansDeclaredGap(gaps, fromMs, toMs) {
  return gaps.some((gap) => gap.startMs < toMs && gap.endMs > fromMs);
}

/**
 * @param {CanvasRenderingContext2D} context
 * @param {number} x
 * @param {number} width
 * @param {number} height
 * @param {(token: string) => string} readToken
 */
function paintHatchedWell(context, x, width, height, readToken) {
  context.save();
  context.beginPath();
  context.rect(x, 0, width, height);
  context.clip();

  context.fillStyle = readToken('--og-unavail-bg') || '#131920';
  context.fillRect(x, 0, width, height);

  // 45° hatch, 1px lines at 6px pitch — demo-ux.md §4.3. Deliberately low
  // contrast: present, obviously non-data, never mistakable for a chart.
  context.strokeStyle = readToken('--og-unavail-hatch') || '#212932';
  context.lineWidth = 1;
  const pitch = 6;
  for (let offset = -height; offset < width + height; offset += pitch) {
    context.beginPath();
    context.moveTo(x + offset, height);
    context.lineTo(x + offset + height, 0);
    context.stroke();
  }
  context.restore();
}

/**
 * @param {CanvasRenderingContext2D} context
 * @param {number} x
 * @param {number} height
 * @param {(token: string) => string} readToken
 */
function paintGapRule(context, x, height, readToken) {
  context.save();
  context.strokeStyle = readToken('--og-unavail-rule') || '#3d4855';
  context.lineWidth = 1;
  context.setLineDash([2, 2]);
  context.beginPath();
  context.moveTo(Math.round(x) + 0.5, 0);
  context.lineTo(Math.round(x) + 0.5, height);
  context.stroke();
  context.restore();
}

/**
 * @param {CanvasRenderingContext2D} context
 * @param {SparklinePlan} plan
 * @param {string} colour
 */
function paintCaption(context, plan, colour) {
  if (!plan.caption || plan.height < 14) {
    // Below ~14px the caption would be illegible. The hatch still carries the
    // meaning, and the well's `aria-label` carries the full sentence, so
    // dropping the glyphs loses nothing a visitor could have read anyway.
    return;
  }
  context.save();
  context.fillStyle = colour;
  context.font = '9px ui-monospace, SF Mono, Menlo, monospace';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  const spaced = plan.caption.split('').join('\u2009');
  context.fillText(spaced, plan.width / 2, plan.height / 2);
  context.restore();
}

/** @param {HTMLCanvasElement} canvas */
function makeTokenReader(canvas) {
  const view = canvas.ownerDocument?.defaultView;
  if (!view) {
    return () => '';
  }
  const styles = view.getComputedStyle(canvas);
  return (token) => styles.getPropertyValue(token).trim();
}

/** @param {number} value */
function defaultFormat(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

/**
 * The AC28 view-as-table alternative for a sparkline.
 *
 * Built from the plan rather than the series so it shows exactly the samples
 * the canvas drew. Downsampled to `maxRows`, because a table of twelve hundred
 * polls is not an accessible alternative — it is the same inaccessibility in a
 * different format, and a screen-reader user would have to page through five
 * minutes of noise to learn what the chart says at a glance.
 *
 * @param {object} plan A plan from {@link planSparkline}.
 * @param {object} [options]
 * @param {number} [options.maxRows]
 * @param {(value: number) => string} [options.format]
 * @returns {Array<{label: string, value: string}>}
 */
export function tabulateSparkline(plan, options = {}) {
  const maxRows = options.maxRows ?? 12;
  const format = options.format ?? ((value) => String(Math.round(value * 100) / 100));
  const samples = plan?.samples ?? [];

  if (samples.length === 0) {
    // The caption already says why — unavailable and pending have different
    // ones — so the table repeats it rather than showing an empty grid that
    // looks like a rendering failure.
    return [{ label: '—', value: plan?.caption ?? 'No samples.' }];
  }

  const step = Math.max(1, Math.ceil(samples.length / maxRows));
  const rows = [];
  for (let index = 0; index < samples.length; index += step) {
    const sample = samples[index];
    rows.push({ label: formatClockTime(sample.t), value: format(sample.v) });
  }

  // The most recent sample is the one a visitor is most likely to want, and
  // fixed-step downsampling drops it whenever the count is not a multiple of
  // the step.
  const last = samples[samples.length - 1];
  if (rows[rows.length - 1]?.label !== formatClockTime(last.t)) {
    rows.push({ label: formatClockTime(last.t), value: format(last.v) });
  }
  return rows;
}

/**
 * @param {number} timestampMs
 * @returns {string}
 */
function formatClockTime(timestampMs) {
  const date = new Date(timestampMs);
  const pad = (part) => String(part).padStart(2, '0');
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}
