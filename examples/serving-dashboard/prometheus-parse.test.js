// Copyright (c) Microsoft Corporation.
//
// Tests for the Prometheus text parser.
//
// The properties under test are mostly about REFUSING to produce a number.
// A parser that returns 0 for a missing metric would quietly reintroduce the
// exact failure mode the provenance table exists to prevent, so several tests
// below assert `null` rather than asserting a value.

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  parsePrometheusText,
  scalarOf,
  histogramMean,
  histogramQuantileUpperBound,
} from './prometheus-parse.js';

/**
 * A verbatim excerpt of `GET /metrics` from a live scatter server on :8123.
 * Kept exact — including the label ordering and the `+Inf` bucket — so that a
 * change in the server's output shape fails here rather than in the browser.
 */
const LIVE_SAMPLE = `# HELP onnx_genai_requests_total Total HTTP requests.
# TYPE onnx_genai_requests_total counter
onnx_genai_requests_total{endpoint="/health",status="200"} 1
onnx_genai_requests_total{endpoint="/v1/chat/completions",status="200"} 10
# HELP onnx_genai_tokens_generated_total Total prompt and completion tokens processed.
# TYPE onnx_genai_tokens_generated_total counter
onnx_genai_tokens_generated_total 5048
# HELP onnx_genai_time_to_first_token_seconds Time to first generated token.
# TYPE onnx_genai_time_to_first_token_seconds histogram
onnx_genai_time_to_first_token_seconds_bucket{le="0.005"} 0
onnx_genai_time_to_first_token_seconds_bucket{le="1"} 0
onnx_genai_time_to_first_token_seconds_bucket{le="2.5"} 8
onnx_genai_time_to_first_token_seconds_bucket{le="5"} 10
onnx_genai_time_to_first_token_seconds_bucket{le="+Inf"} 10
onnx_genai_time_to_first_token_seconds_sum 20.737784291
onnx_genai_time_to_first_token_seconds_count 10
# HELP onnx_genai_batch_size_current Current generation batch size.
# TYPE onnx_genai_batch_size_current gauge
onnx_genai_batch_size_current 1
# HELP onnx_genai_prefix_cache_hit_rate Prefix-cache hit ratio.
# TYPE onnx_genai_prefix_cache_hit_rate gauge
onnx_genai_prefix_cache_hit_rate 0
`;

test('parses counters, gauges and histograms from real server output', () => {
  const families = parsePrometheusText(LIVE_SAMPLE);

  assert.equal(scalarOf(families, 'onnx_genai_tokens_generated_total'), 5048);
  assert.equal(scalarOf(families, 'onnx_genai_batch_size_current'), 1);
  assert.equal(families.get('onnx_genai_time_to_first_token_seconds').type, 'histogram');
  assert.equal(families.get('onnx_genai_time_to_first_token_seconds').count, 10);
});

test('labelled samples keep their labels and do not collide', () => {
  const families = parsePrometheusText(LIVE_SAMPLE);
  const f = families.get('onnx_genai_requests_total');

  assert.equal(f.samples.length, 2);
  assert.deepEqual(f.samples[0].labels, { endpoint: '/health', status: '200' });
  assert.equal(f.samples[1].labels.endpoint, '/v1/chat/completions');
  assert.equal(f.samples[1].value, 10);
});

test('a missing metric reads as null, NEVER as zero', () => {
  const families = parsePrometheusText(LIVE_SAMPLE);

  // This is the property that keeps a disabled `metrics` feature from being
  // rendered as a server that measured zero of everything.
  assert.equal(scalarOf(families, 'onnx_genai_not_a_real_metric'), null);
  assert.equal(histogramMean(families, 'onnx_genai_not_a_real_metric'), null);
});

test('a genuine zero is still reported as zero', () => {
  const families = parsePrometheusText(LIVE_SAMPLE);

  // The mirror of the test above: `prefix_cache_hit_rate 0` is a real
  // observation and must survive as 0, not be flattened into "missing".
  assert.equal(scalarOf(families, 'onnx_genai_prefix_cache_hit_rate'), 0);
});

test('histogram mean is sum/count', () => {
  const families = parsePrometheusText(LIVE_SAMPLE);
  const ttft = histogramMean(families, 'onnx_genai_time_to_first_token_seconds');

  assert.equal(ttft.count, 10);
  assert.ok(Math.abs(ttft.mean - 2.0737784291) < 1e-9);
});

test('an empty histogram yields null, not a zero average', () => {
  const families = parsePrometheusText(`# HELP h Nothing observed yet.
# TYPE h histogram
h_bucket{le="+Inf"} 0
h_sum 0
h_count 0
`);

  // sum/count would be 0/0 = NaN, and a naive guard would report 0. Reporting
  // "average latency: 0s" for a server that has served nothing is a lie.
  assert.equal(histogramMean(families, 'h'), null);
});

test('quantile returns the bucket upper bound, and null when it lands in +Inf', () => {
  const families = parsePrometheusText(LIVE_SAMPLE);

  // 0.5 * 10 = 5 observations; the first bucket reaching 5 is le="2.5".
  assert.deepEqual(histogramQuantileUpperBound(families, 'onnx_genai_time_to_first_token_seconds', 0.5), {
    upperBound: 2.5,
    count: 10,
  });

  const openTop = parsePrometheusText(`# TYPE h histogram
h_bucket{le="1"} 1
h_bucket{le="+Inf"} 10
h_sum 50
h_count 10
`);
  // p95 falls in the unbounded bucket, where no honest upper bound exists.
  assert.equal(histogramQuantileUpperBound(openTop, 'h', 0.95), null);
});

test('label values containing braces and escaped quotes parse correctly', () => {
  const families = parsePrometheusText(`# TYPE m counter
m{path="/a{b}c",note="he said \\"hi\\""} 7
`);
  const [sample] = families.get('m').samples;

  // Naive splitting on `}` would truncate this label and mis-key the sample.
  assert.equal(sample.labels.path, '/a{b}c');
  assert.equal(sample.labels.note, 'he said "hi"');
  assert.equal(sample.value, 7);
});

test('+Inf, -Inf and NaN values parse to the right floats', () => {
  const families = parsePrometheusText(`# TYPE g gauge
g{k="pos"} +Inf
g{k="neg"} -Inf
g{k="nan"} NaN
`);
  const values = families.get('g').samples.map((s) => s.value);

  assert.equal(values[0], Number.POSITIVE_INFINITY);
  assert.equal(values[1], Number.NEGATIVE_INFINITY);
  assert.ok(Number.isNaN(values[2]));
  // scalarOf rejects non-finite values rather than passing Infinity to a panel.
  assert.equal(scalarOf(families, 'g'), null);
});

test('a counter whose name ends in _count is not swallowed by histogram folding', () => {
  const families = parsePrometheusText(`# TYPE my_count counter
my_count 42
`);

  // `_count` is a histogram suffix, but only for a family declared a histogram.
  assert.equal(scalarOf(families, 'my_count'), 42);
});

test('garbage input yields no metrics instead of throwing', () => {
  // The store must survive an HTML error page served where /metrics was
  // expected — a proxy or a wrong port is a realistic failure.
  assert.equal(parsePrometheusText('<html><body>404</body></html>').size, 0);
  assert.equal(parsePrometheusText('').size, 0);
  assert.equal(parsePrometheusText(null).size, 0);
  assert.equal(parsePrometheusText(undefined).size, 0);
});

test('a sample line with no value is skipped rather than read as zero', () => {
  const families = parsePrometheusText(`# TYPE g gauge
g
`);

  assert.equal(scalarOf(families, 'g'), null);
});
