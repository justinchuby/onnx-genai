// Copyright (c) Microsoft Corporation.
//
// A minimal parser for the Prometheus text exposition format, covering exactly
// the subset `GET /metrics` emits: counters, gauges, and histograms.
//
// Why parse this at all, when /v1/status returns tidy JSON? Because several of
// the numbers this demo needs are HONEST here and FABRICATED there.
// `/v1/status` writes literal `0.0` for tokens_per_second and batch_utilization
// (see telemetry-provenance.js). `/metrics` carries genuine observations for
// the same concepts, recorded at the point where the work actually happens.
// Reaching for the harder format is the difference between a real number and a
// plausible one.
//
// This is deliberately NOT a general Prometheus client. It rejects what it does
// not understand rather than guessing, because a parser that silently returns
// the wrong number is worse here than one that returns nothing: the whole page
// is built on being able to say "we did not measure that".

/**
 * One parsed metric family.
 *
 * @typedef {object} MetricFamily
 * @property {string} name
 * @property {'counter'|'gauge'|'histogram'|'summary'|'untyped'} type
 * @property {string} help
 * @property {Array<{labels: Record<string,string>, value: number}>} samples
 *   For histograms this holds only the `_bucket` samples.
 * @property {number} [sum]    Histogram only: the `_sum` sample.
 * @property {number} [count]  Histogram only: the `_count` sample.
 * @property {Array<{le: number, cumulative: number}>} [buckets]
 *   Histogram only, sorted ascending by `le`.
 */

/** Prometheus spells these three floats in a way `Number()` gets wrong. */
const SPECIAL_VALUES = Object.freeze({
  NaN: Number.NaN,
  '+Inf': Number.POSITIVE_INFINITY,
  '-Inf': Number.NEGATIVE_INFINITY,
  Inf: Number.POSITIVE_INFINITY,
});

/**
 * Parse a metric value. Returns NaN for anything unrecognised so callers using
 * `Number.isFinite` reject it; we never coerce a bad value to 0, which would be
 * indistinguishable from a real zero.
 *
 * @param {string} raw
 * @returns {number}
 */
function parseValue(raw) {
  if (Object.prototype.hasOwnProperty.call(SPECIAL_VALUES, raw)) {
    return SPECIAL_VALUES[raw];
  }
  // Number('') is 0 and Number(' ') is 0 — both must be rejected, not zeroed.
  if (raw.trim() === '') return Number.NaN;
  return Number(raw);
}

/**
 * Split a sample line into its metric name, label set, and value.
 *
 * Label values may contain spaces, `}` and escaped quotes, so the braces cannot
 * be found by naive splitting — we scan for the closing brace while tracking
 * whether we are inside a quoted string.
 *
 * @param {string} line
 * @returns {{name: string, labels: Record<string,string>, value: number}|null}
 */
function parseSampleLine(line) {
  const braceStart = line.indexOf('{');
  // A sample may legally carry a trailing timestamp, which we ignore.
  if (braceStart === -1) {
    const parts = line.split(/\s+/);
    if (parts.length < 2) return null;
    return { name: parts[0], labels: {}, value: parseValue(parts[1]) };
  }

  let inQuotes = false;
  let escaped = false;
  let braceEnd = -1;
  for (let i = braceStart + 1; i < line.length; i += 1) {
    const ch = line[i];
    if (escaped) {
      escaped = false;
    } else if (ch === '\\') {
      escaped = true;
    } else if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === '}' && !inQuotes) {
      braceEnd = i;
      break;
    }
  }
  if (braceEnd === -1) return null;

  const name = line.slice(0, braceStart).trim();
  const labels = parseLabels(line.slice(braceStart + 1, braceEnd));
  const rest = line.slice(braceEnd + 1).trim().split(/\s+/);
  if (rest.length === 0 || rest[0] === '') return null;
  return { name, labels, value: parseValue(rest[0]) };
}

/**
 * @param {string} body Text between the braces.
 * @returns {Record<string,string>}
 */
function parseLabels(body) {
  /** @type {Record<string,string>} */
  const labels = {};
  let i = 0;
  while (i < body.length) {
    const eq = body.indexOf('=', i);
    if (eq === -1) break;
    const key = body.slice(i, eq).trim();
    const quoteStart = body.indexOf('"', eq);
    if (quoteStart === -1) break;

    let value = '';
    let escaped = false;
    let j = quoteStart + 1;
    for (; j < body.length; j += 1) {
      const ch = body[j];
      if (escaped) {
        // The exposition format defines exactly three escapes.
        value += ch === 'n' ? '\n' : ch;
        escaped = false;
      } else if (ch === '\\') {
        escaped = true;
      } else if (ch === '"') {
        break;
      } else {
        value += ch;
      }
    }
    if (key) labels[key] = value;
    const comma = body.indexOf(',', j);
    if (comma === -1) break;
    i = comma + 1;
  }
  return labels;
}

/**
 * Parse a Prometheus text exposition document.
 *
 * @param {string} text
 * @returns {Map<string, MetricFamily>} Keyed by family name.
 */
export function parsePrometheusText(text) {
  /** @type {Map<string, MetricFamily>} */
  const families = new Map();
  if (typeof text !== 'string') return families;

  /** @type {Map<string, {sum?: number, count?: number, buckets: Array<{le:number,cumulative:number}>}>} */
  const histogramParts = new Map();

  const family = (name) => {
    let f = families.get(name);
    if (!f) {
      f = { name, type: 'untyped', help: '', samples: [] };
      families.set(name, f);
    }
    return f;
  };

  for (const rawLine of text.split('\n')) {
    const line = rawLine.trim();
    if (line === '') continue;

    if (line.startsWith('#')) {
      const meta = /^#\s+(HELP|TYPE)\s+(\S+)\s*(.*)$/.exec(line);
      // Any other comment is a plain comment and is ignored by the format.
      if (!meta) continue;
      const [, kind, name, rest] = meta;
      if (kind === 'HELP') family(name).help = rest;
      else family(name).type = /** @type {MetricFamily['type']} */ (rest.trim() || 'untyped');
      continue;
    }

    const sample = parseSampleLine(line);
    if (!sample) continue;

    // Histogram sub-samples carry a suffix and belong to the base family.
    for (const suffix of ['_bucket', '_sum', '_count']) {
      if (!sample.name.endsWith(suffix)) continue;
      const base = sample.name.slice(0, -suffix.length);
      // Only fold into a base family that is actually declared a histogram;
      // otherwise a counter legitimately named `..._count` would be swallowed.
      const declared = families.get(base);
      if (!declared || (declared.type !== 'histogram' && declared.type !== 'summary')) continue;

      let parts = histogramParts.get(base);
      if (!parts) {
        parts = { buckets: [] };
        histogramParts.set(base, parts);
      }
      if (suffix === '_sum') parts.sum = sample.value;
      else if (suffix === '_count') parts.count = sample.value;
      else {
        parts.buckets.push({ le: parseValue(sample.labels.le ?? ''), cumulative: sample.value });
        declared.samples.push({ labels: sample.labels, value: sample.value });
      }
      sample.name = '';
      break;
    }
    if (sample.name === '') continue;

    family(sample.name).samples.push({ labels: sample.labels, value: sample.value });
  }

  for (const [base, parts] of histogramParts) {
    const f = families.get(base);
    if (!f) continue;
    f.sum = parts.sum;
    f.count = parts.count;
    f.buckets = parts.buckets.slice().sort((a, b) => a.le - b.le);
  }

  return families;
}

/**
 * Read a single unlabelled scalar (counter or gauge).
 *
 * Returns `null` — never 0 — when the metric is absent or unparseable, so the
 * caller can tell "the server did not report this" apart from "the server
 * reported zero". That distinction is the entire point of this demo.
 *
 * @param {Map<string, MetricFamily>} families
 * @param {string} name
 * @returns {number|null}
 */
export function scalarOf(families, name) {
  const f = families.get(name);
  if (!f || f.samples.length === 0) return null;
  const { value } = f.samples[0];
  return Number.isFinite(value) ? value : null;
}

/**
 * Mean of a histogram, as `sum / count`.
 *
 * Returns `null` when no observations have been recorded. A histogram with
 * count 0 has a sum of 0, and reporting "0 ms average latency" for a server
 * that has served nothing is precisely the fabricated number this demo exists
 * to avoid.
 *
 * @param {Map<string, MetricFamily>} families
 * @param {string} name
 * @returns {{mean: number, count: number}|null}
 */
export function histogramMean(families, name) {
  const f = families.get(name);
  if (!f || typeof f.sum !== 'number' || typeof f.count !== 'number') return null;
  if (!Number.isFinite(f.sum) || !Number.isFinite(f.count) || f.count <= 0) return null;
  return { mean: f.sum / f.count, count: f.count };
}

/**
 * Approximate a quantile from cumulative histogram buckets.
 *
 * This is an APPROXIMATION bounded by bucket width, and callers must label it
 * as such — the server's coarse buckets (…, 1, 2.5, 5, +Inf) mean a p95 landing
 * in the top finite bucket could be anywhere in a 2.5s-wide band. Returns the
 * bucket's upper bound, and `null` if the quantile falls in the `+Inf` bucket,
 * where there is no upper bound to honestly report.
 *
 * @param {Map<string, MetricFamily>} families
 * @param {string} name
 * @param {number} quantile Between 0 and 1.
 * @returns {{upperBound: number, count: number}|null}
 */
export function histogramQuantileUpperBound(families, name, quantile) {
  const f = families.get(name);
  if (!f || !Array.isArray(f.buckets) || typeof f.count !== 'number') return null;
  if (!Number.isFinite(f.count) || f.count <= 0) return null;
  if (!(quantile > 0 && quantile < 1)) return null;

  const target = quantile * f.count;
  for (const bucket of f.buckets) {
    if (bucket.cumulative >= target) {
      if (!Number.isFinite(bucket.le)) return null;
      return { upperBound: bucket.le, count: f.count };
    }
  }
  return null;
}
