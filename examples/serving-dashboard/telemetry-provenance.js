// Copyright (c) Microsoft Corporation.
//
// The provenance table: for every field this demo can read off the server, a
// binding decision about whether it is a genuine measurement.
//
// This file is the mechanism behind the project's "a number that was never
// measured is omitted rather than printed as 0" rule. `GET /v1/status` is the
// trap: it is ungated, it returns a well-formed JSON document, and several of
// its numbers are literal `0.0` written into the response body with a
// `// not yet tracked` comment beside them. A developer binding a panel to
// `status.tokens_per_second` in good faith would render `0.0 tok/s` as a
// measurement. The classification below prevents that mechanically: fields
// marked DOCUMENTED_ZERO can never become a `measured` field, no matter what
// the server sends.
//
// EVERY entry cites file:line evidence. When @d7cf9b84's server plumbing lands
// and a field becomes genuinely measured, flip its `classification` here and
// update the citation — one edit, and every panel in the page starts telling
// the truth about it at once.
//
// Line citations were read from the worktree at branch feat/genai-demo-dashboard,
// commit 54c0bc98. That SHA is not decoration: this table is a snapshot of a
// tree the server work is actively invalidating, so `matchesStub` below checks
// each placeholder against the wire on every poll and reports when a row here
// has gone stale. Update the citation in the same commit that makes a row false.

/**
 * How a server field is classified. This is not a UI state — it is a statement
 * about the server's code.
 *
 * @typedef {'MEASURED' | 'DOCUMENTED_ZERO' | 'NOT_PLUMBED' | 'STRUCTURALLY_BYPASSED'} Classification
 *
 * - `MEASURED`        — the server computes this from real runtime state.
 * - `DOCUMENTED_ZERO` — the server writes a constant `0` / `""` / `[]` into the
 *                       response and documents that it is not tracked. NEVER
 *                       render this. Always `unavailable`.
 * - `NOT_PLUMBED`     — the data exists inside the process but no endpoint
 *                       returns it yet. Also `unavailable`, but the reason
 *                       differs and so does the fix.
 * - `STRUCTURALLY_BYPASSED`
 *                       — the subsystem exists and works, but THIS server's
 *                       code path never consults it, so the question is never
 *                       asked. Renders `not-applicable`. No amount of plumbing
 *                       would fix it; it is a true statement about the
 *                       architecture, not a gap in it.
 *
 * CLASSIFICATION CAN DEPEND ON WHICH SERVER WE ASKED. The demo runs two, and
 * they differ structurally: the scatter server batches (bypassing the page
 * table and prefix trie) while the dynamic server pages KV (disabling
 * continuous batching). An entry may therefore carry `byOrigin` to override
 * its classification per server. `prefix_cache_hits: 0` is a genuine measured
 * zero on the dynamic server and not-applicable on the scatter server — the
 * SAME wire value, opposite treatments, and only this table knows which.
 */

/**
 * @typedef {object} ProvenanceEntry
 * @property {string} source     Endpoint that carries (or would carry) the field.
 * @property {string} path       Dotted path into that endpoint's JSON body.
 * @property {Classification} classification
 * @property {string|null} unit
 * @property {string} evidence   `file:line` backing the classification.
 * @property {string} label      Human name for tooltips and the footer table.
 * @property {string} [reason]   Required unless MEASURED. Shown to the visitor.
 */

/** Server endpoints this demo polls. Gated ones degrade panel-by-panel (AC20). */
export const ENDPOINTS = Object.freeze({
  HEALTH: '/health',
  MODELS: '/v1/models',
  STATUS: '/v1/status',
  DEBUG_KV: '/v1/debug/kv',
  DEBUG_CONFIG: '/v1/debug/config',
  RESOURCES: '/v1/resources',
  METRICS: '/metrics',
});

/**
 * Endpoints whose body is Prometheus text rather than JSON. The store parses
 * these with prometheus-parse.js and reads fields via `metric`, not `path`.
 */
export const TEXT_ENDPOINTS = Object.freeze([ENDPOINTS.METRICS]);

/**
 * Endpoints gated behind a cargo feature rather than a CLI flag. A 404 is a
 * build-time choice, so the fix we show the visitor is a rebuild, not a flag.
 */
export const FEATURE_GATED_ENDPOINTS = Object.freeze({
  [ENDPOINTS.METRICS]: 'metrics',
});

/** Endpoints behind `--enable-debug-endpoints`. A 404 here is configuration, not breakage. */
export const DEBUG_GATED_ENDPOINTS = Object.freeze([
  ENDPOINTS.DEBUG_KV,
  ENDPOINTS.DEBUG_CONFIG,
]);

/** The exact flag a visitor must add. Used verbatim in every "how to fix" message. */
export const DEBUG_ENDPOINTS_FLAG = '--enable-debug-endpoints';

/**
 * The table. Keys are the stable field keys panels use with
 * `telemetryStore.field(key)`. Panels reference keys, never raw JSON paths, so
 * a server-side rename is one edit here.
 *
 * @type {Readonly<Record<string, ProvenanceEntry>>}
 */
export const PROVENANCE = Object.freeze({
  // ---------------------------------------------------------------- server identity
  'server.model_id': {
    source: ENDPOINTS.HEALTH,
    path: 'model',
    classification: 'MEASURED',
    unit: null,
    evidence: 'crates/onnx-genai-server/src/routes/mod.rs:105-108 (HealthResponse.model)',
    label: 'Model id',
  },
  'server.healthy': {
    source: ENDPOINTS.STATUS,
    path: 'healthy',
    classification: 'MEASURED',
    unit: null,
    evidence: 'crates/onnx-genai-server/src/routes/admin.rs:47-51 (registry default_id().is_some())',
    label: 'Server healthy',
  },
  'server.node_id': {
    source: ENDPOINTS.STATUS,
    path: 'node_id',
    classification: 'MEASURED',
    unit: null,
    evidence: 'crates/onnx-genai-server/src/routes/admin.rs:45 (config.node_id)',
    label: 'Node id',
  },
  'server.context_length': {
    source: ENDPOINTS.DEBUG_CONFIG,
    path: 'model_max_context',
    classification: 'MEASURED',
    unit: 'tokens',
    evidence: 'crates/onnx-genai-server/src/routes/admin.rs:99 (handle.model_max_context)',
    label: 'Context length',
  },
  'server.pipeline': {
    source: ENDPOINTS.DEBUG_CONFIG,
    path: 'pipeline',
    classification: 'MEASURED',
    unit: null,
    evidence: 'crates/onnx-genai-server/src/routes/admin.rs:95 (handle.pipeline)',
    label: 'Pipeline model',
  },
  'server.model_path': {
    source: ENDPOINTS.MODELS,
    path: 'served.path',
    classification: 'NOT_PLUMBED',
    unit: null,
    evidence:
      'No route returns the model directory today. /v1/debug/config exposes id, pipeline ' +
      'and context only — crates/onnx-genai-server/src/routes/admin.rs:93-100, and the ' +
      '/v1/models entry carries id/object/created/owned_by — routes/mod.rs:96-101.',
    label: 'Model directory',
    reason:
      'The server does not expose the model directory path on any endpoint yet. @d7cf9b84 is ' +
      'adding it to /v1/models, which is ungated and already polled, so it will appear ' +
      'here without a flag on a visitor\'s first run.',
    // POINTED AT THE ENDPOINT IT WILL ARRIVE ON, DELIBERATELY. This entry was
    // previously aimed at /v1/status, where `path` is never going to appear.
    // A NOT_PLUMBED entry is checked by asserting its path carries NOTHING, so
    // an entry aimed at the wrong endpoint can never notice the field going
    // live -- it would em-dash a real measurement forever, silently. Aimed
    // here, the staleness check fires the moment the server ships it.
  },
  'server.execution_provider': {
    source: ENDPOINTS.STATUS,
    path: 'server.execution_provider',
    classification: 'NOT_PLUMBED',
    unit: null,
    evidence:
      'The EP is chosen from the ONNX_GENAI_EP env var at startup and is not echoed by ' +
      'any ungated route (no `execution_provider` field exists in routes/mod.rs response types).',
    label: 'Execution provider',
    reason:
      'The server does not report which execution provider it loaded. Until it does, this ' +
      'demo will not guess — the EP changes how every latency number should be read.',
  },

  // ---------------------------------------------------------------- scheduling (real)
  'queue.depth': {
    source: ENDPOINTS.STATUS,
    path: 'queue_depth',
    classification: 'MEASURED',
    unit: 'requests',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:59 (snapshot.pending_requests) — ' +
      'a real admission/backpressure counter.',
    label: 'Queue depth',
  },
  'sessions.active': {
    source: ENDPOINTS.STATUS,
    path: 'active_sessions',
    classification: 'MEASURED',
    unit: 'sessions',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:61 (snapshot.active_sessions). The counter ' +
      'is driven ONLY by the X-Session-Id registry: session.rs:73 is the single caller of ' +
      'metrics::active_sessions_added, and it fires when a client_id is inserted into the ' +
      'session map. Nothing about a generation touches it.',
    // NOT "Active sessions". It does not count in-flight work, and during
    // Scenario A's concurrent requests it reads 0 -- a visitor watching four
    // lanes stream next to "Active sessions: 0" concludes the page is broken.
    //
    // It is also mutually exclusive with the headline claim: driver.rs:500-508
    // routes X-Session-Id requests down the per-request engine path, BYPASSING
    // ContinuousBatchManager. Making this number non-zero would switch off the
    // continuous batching Scenario A exists to demonstrate.
    //
    // Use batch.active_size for concurrency; that one is genuinely in-flight.
    label: 'Persistent sessions',
    caveat:
      'Counts long-lived X-Session-Id sessions, not in-flight requests. Stateless requests ' +
      'never increment it, and session requests bypass continuous batching entirely.',
  },
  'sessions.paused': {
    source: ENDPOINTS.STATUS,
    path: 'paused_sessions',
    stubValue: 0,
    classification: 'DOCUMENTED_ZERO',
    unit: 'sessions',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:62 — `paused_sessions: 0, ' +
      '// not yet tracked (no preemption/pause state exposed)`',
    label: 'Paused sessions',
    reason:
      'The server has no preemption/pause state to report, so it sends a documented zero. ' +
      'This demo will not render that zero as a measurement.',
  },
  'batch.utilization': {
    source: ENDPOINTS.STATUS,
    path: 'batch_utilization',
    stubValue: 0,
    classification: 'DOCUMENTED_ZERO',
    unit: 'ratio',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:64 — `batch_utilization: 0.0, ' +
      '// not yet tracked (max batch size not surfaced to the server)`',
    label: 'Batch utilization',
    reason:
      'The server cannot compute this because the batch limit is not surfaced to the HTTP ' +
      'layer, so it sends a hardcoded 0.0. A zero here would be a fabricated measurement.',
  },
  'throughput.tokens_per_second': {
    source: ENDPOINTS.STATUS,
    path: 'tokens_per_second',
    stubValue: 0,
    classification: 'DOCUMENTED_ZERO',
    unit: 'tokens/s',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:63 — `tokens_per_second: 0.0, ' +
      '// not yet tracked (only cumulative token totals recorded)`',
    label: 'Server-reported tokens/sec',
    reason:
      'The server records cumulative token totals but no rate, so it sends a hardcoded 0.0. ' +
      'This demo measures throughput client-side from streamed tokens instead, and labels it ' +
      'as client-measured.',
  },

  // ---------------------------------------------------------------- paged KV (all zeros today)
  'kv.usage': {
    source: ENDPOINTS.STATUS,
    path: 'kv_usage',
    stubValue: 0,
    classification: 'DOCUMENTED_ZERO',
    unit: 'ratio',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:54 — `kv_usage: 0.0, // not yet tracked`',
    label: 'KV utilization',
    reason:
      'The engine does not expose paged-KV introspection to the HTTP layer yet, so /v1/status ' +
      'sends a hardcoded 0.0.',
  },
  'kv.pages_used': {
    source: ENDPOINTS.STATUS,
    path: 'kv_pages_used',
    stubValue: 0,
    classification: 'DOCUMENTED_ZERO',
    unit: 'pages',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:55 — `kv_pages_used: 0, // not yet tracked`',
    label: 'KV pages in use',
    reason: 'Not exposed to the HTTP layer yet; /v1/status sends a hardcoded 0.',
  },
  'kv.pages_total': {
    source: ENDPOINTS.STATUS,
    path: 'kv_pages_total',
    stubValue: 0,
    classification: 'DOCUMENTED_ZERO',
    unit: 'pages',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:56 — `kv_pages_total: 0, // not yet tracked`',
    label: 'KV pages total',
    reason: 'Not exposed to the HTTP layer yet; /v1/status sends a hardcoded 0.',
  },
  'kv.pages_shared': {
    source: ENDPOINTS.STATUS,
    path: 'kv_pages_shared',
    stubValue: 0,
    classification: 'DOCUMENTED_ZERO',
    unit: 'pages',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:57 — `kv_pages_shared: 0, // not yet tracked`',
    label: 'KV pages shared',
    reason: 'Not exposed to the HTTP layer yet; /v1/status sends a hardcoded 0.',
  },
  'kv.introspection': {
    source: ENDPOINTS.DEBUG_KV,
    path: 'engine_kv_introspection',
    // This one carries a sentence rather than a number, and the exact wording is
    // not worth pinning: what makes it a placeholder is the "unavailable:" prefix.
    isStub: (value) => typeof value === 'string' && value.startsWith('unavailable'),
    classification: 'NOT_PLUMBED',
    unit: null,
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:140 returns the string ' +
      '"unavailable: engine does not yet expose KV page statistics", while ' +
      'Engine::page_usage()/page_stats() already compute it ' +
      '(crates/onnx-genai-kv/src/page_table.rs:564-613).',
    label: 'KV page table',
    reason:
      'The engine computes full page statistics, but no driver command carries them to the ' +
      'HTTP layer yet. The block table stays empty rather than showing invented blocks.',
  },

  // ---------------------------------------------------------------- prefix cache
  'prefix_cache.hits': {
    source: ENDPOINTS.DEBUG_KV,
    path: 'prefix_cache_hits',
    classification: 'MEASURED',
    unit: 'count',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:132 (snapshot.prefix_cache_hits) — a real ' +
      'counter. NOTE: it reports 0 on the static-cache decode path because prefix caching ' +
      'lives in the paged-KV manager, which the continuous-batch path does not use. The ' +
      'counter is honest; the path simply never hits it.',
    label: 'Prefix cache hits',
    byOrigin: {
      scatter: {
        // Pinned at literal 0 by the bypass, so 0 is what "still true" looks like.
        stubValue: 0,
        classification: 'STRUCTURALLY_BYPASSED',
        reason:
          'This server uses static-cache batching, and that path never consults the prefix ' +
          'cache, so the question is never asked. The engine asserts it: ' +
          'crates/onnx-genai-engine/tests/batched_static_decode.rs:53 and ' +
          'engine_continuous_batch_scheduled.rs:82 require prefix_cache_hit_len == 0 for ' +
          'every batched result. A 0 here would imply a cache that tried and missed.',
      },
      // On the dynamic server the cache IS consulted, so 0 is genuine data.
      dynamic: { classification: 'MEASURED' },
    },
  },
  'prefix_cache.lookups': {
    source: ENDPOINTS.DEBUG_KV,
    path: 'prefix_cache_lookups',
    classification: 'MEASURED',
    unit: 'count',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:133 reads the SAME counter that ' +
      'metrics.rs:132-134 increments unconditionally on every completed generation, whether ' +
      'or not a cache was consulted. The upstream name is wrong: it counts generations.',
    // Deliberately NOT "Prefix cache lookups". Labelling it that would report a
    // cache being consulted on a server that never consults one.
    label: 'Completed generations',
    byOrigin: {
      scatter: {
        unfalsifiable:
          'This counter is not pinned -- it rises on every completed generation. It is ' +
          'suppressed because it does not count what its name claims, and no value on ' +
          'the wire can confirm or deny that.',
        classification: 'STRUCTURALLY_BYPASSED',
        reason:
          'This server uses static-cache batching, and that path never consults the prefix ' +
          'cache, so the question is never asked. The engine asserts it: ' +
          'crates/onnx-genai-engine/tests/batched_static_decode.rs:53 and ' +
          'engine_continuous_batch_scheduled.rs:82 require prefix_cache_hit_len == 0 for ' +
          'every batched result. A 0 here would imply a cache that tried and missed.',
      },
      dynamic: { classification: 'MEASURED' },
    },
  },
  'prefix_cache.hashes': {
    source: ENDPOINTS.STATUS,
    path: 'prefix_hashes',
    stubValue: [],
    classification: 'DOCUMENTED_ZERO',
    unit: null,
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:80-81 — `// System-prompt prefix hashes ' +
      'are not yet surfaced by the engine.` then `prefix_hashes: Vec::new()`',
    label: 'Prefix hashes',
    reason: 'The engine does not surface prefix hashes, so /v1/status always sends an empty list.',
  },

  // ---------------------------------------------------------------- batching / admission
  'batch.active_size': {
    source: ENDPOINTS.DEBUG_KV,
    path: 'active_batch_size',
    classification: 'MEASURED',
    unit: 'requests',
    evidence: 'crates/onnx-genai-server/src/routes/admin.rs:136 (snapshot.current_batch_size)',
    label: 'Active batch size',
  },
  'admission.slots_available': {
    source: ENDPOINTS.DEBUG_KV,
    path: 'available_admission_slots',
    classification: 'MEASURED',
    unit: 'slots',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:138 ' +
      '(handle.engine.generation_capacity.available_permits())',
    label: 'Admission slots free',
  },
  'admission.rejections': {
    source: ENDPOINTS.DEBUG_KV,
    path: 'rejected_requests',
    classification: 'MEASURED',
    unit: 'count',
    evidence: 'crates/onnx-genai-server/src/routes/admin.rs:139 (snapshot.rejections)',
    label: 'Rejected requests',
  },

  // ── GET /metrics ────────────────────────────────────────────────────────
  //
  // The honest counterpart to the fabricated half of /v1/status. These are
  // recorded at the point the work happens (metrics.rs), not assembled for a
  // response, and several of them are the genuine version of a number that
  // /v1/status hardcodes to 0.0. Read via `metric` + `kind` rather than
  // `path`, because the payload is Prometheus text, not JSON.
  //
  // /metrics is behind the `metrics` cargo feature. It is ON BY DEFAULT
  // (crates/onnx-genai-server/Cargo.toml: `default = ["metrics"]`), so a 404
  // here means someone built with --no-default-features, which is a
  // configuration fact worth stating rather than a breakage.
  'metrics.ttft': {
    source: ENDPOINTS.METRICS,
    metric: 'onnx_genai_time_to_first_token_seconds',
    kind: 'histogram_mean',
    classification: 'MEASURED',
    unit: 's',
    evidence:
      'crates/onnx-genai-server/src/metrics.rs:119-123 — GenerationMetrics::token() observes ' +
      'started.elapsed() on the first token of each generation.',
    label: 'Time to first token (mean)',
  },
  'metrics.e2e_latency': {
    source: ENDPOINTS.METRICS,
    metric: 'onnx_genai_e2e_request_latency_seconds',
    kind: 'histogram_mean',
    classification: 'MEASURED',
    unit: 's',
    evidence:
      'crates/onnx-genai-server/src/metrics.rs:141-144 — observed in Drop for GenerationMetrics, ' +
      'so it covers the full generation lifetime.',
    label: 'End-to-end latency (mean)',
  },
  'metrics.tokens_generated_total': {
    source: ENDPOINTS.METRICS,
    metric: 'onnx_genai_tokens_generated_total',
    kind: 'scalar',
    classification: 'MEASURED',
    unit: 'tokens',
    evidence:
      'crates/onnx-genai-server/src/metrics.rs — cumulative prompt + completion token counter, ' +
      'incremented per generation result.',
    label: 'Tokens generated (cumulative)',
  },
  'metrics.completion_tokens_total': {
    source: ENDPOINTS.METRICS,
    metric: 'onnx_genai_completion_tokens_total',
    kind: 'scalar',
    classification: 'MEASURED',
    unit: 'tokens',
    evidence:
      'crates/onnx-genai-server/src/metrics.rs:128-130 — fetch_add(completion_tokens) in ' +
      'GenerationMetrics::result().',
    label: 'Completion tokens (cumulative)',
  },

  // ⚠️ THE MOST MISLEADING METRIC ON THE SERVER.
  //
  // `onnx_genai_batch_size_current` is named "batch size" and is documented
  // "Current generation batch size." It is neither. It is fetch_add(1) in
  // GenerationMetrics::start() and decremented in Drop — so it counts
  // GENERATION REQUESTS IN FLIGHT at the HTTP layer, and never consults the
  // ContinuousBatchManager. With max_batch pinned at 4, firing 8 concurrent
  // requests makes this gauge read 8 while the engine's actual batch is 4.
  //
  // Rendering it as "batch size" would be a fabricated measurement wearing a
  // plausible label — worse than a hardcoded zero, because nothing about the
  // value looks wrong. It is exposed here under a name that says what it
  // actually counts.
  'batch.in_flight': {
    source: ENDPOINTS.METRICS,
    metric: 'onnx_genai_batch_size_current',
    kind: 'scalar',
    classification: 'MEASURED',
    unit: 'requests',
    evidence:
      'crates/onnx-genai-server/src/metrics.rs:112 (fetch_add on generation start) and :145 ' +
      '(decrement in Drop). Counts in-flight generations, NOT the engine batch.',
    label: 'Generations in flight',
  },
  // The number a viewer actually wants — how many sequences the engine stepped
  // together — is not exposed by anything. ContinuousBatchManager does not
  // report its batch, so it must stay unavailable rather than be approximated
  // by the gauge above.
  'batch.effective_size': {
    source: ENDPOINTS.METRICS,
    metric: null,
    kind: 'scalar',
    classification: 'NOT_PLUMBED',
    unit: 'sequences',
    evidence:
      'crates/onnx-genai-engine/src/engine/batched.rs:101-110 — ContinuousBatchManager steps a ' +
      'batch but exposes no counter for it; onnx_genai_batch_size_current counts HTTP-layer ' +
      'in-flight generations instead.',
    label: 'Engine batch size',
    reason:
      'The engine does not report how many sequences it actually stepped together. The ' +
      'available gauge counts requests in flight, which is a different number whenever ' +
      'concurrency exceeds the batch limit — so this demo shows in-flight and queued ' +
      'separately rather than guessing at the batch.',
  },
  'metrics.requests_waiting': {
    source: ENDPOINTS.METRICS,
    metric: 'onnx_genai_requests_waiting',
    kind: 'scalar',
    classification: 'MEASURED',
    unit: 'requests',
    evidence:
      'crates/onnx-genai-server/src/metrics.rs:273-277 — gauge fed from snapshot.pending_requests.',
    label: 'Requests waiting',
  },
  'metrics.prefix_cache_hits': {
    source: ENDPOINTS.METRICS,
    metric: 'onnx_genai_prefix_cache_hits_total',
    kind: 'scalar',
    classification: 'MEASURED',
    unit: 'count',
    evidence:
      'crates/onnx-genai-server/src/metrics.rs:136-138 — incremented when prefix_cache_hit_len > 0.',
    label: 'Prefix-cache hits',
    // The same 0 means two opposite things depending on which server answered.
    byOrigin: {
      scatter: {
        // Pinned at literal 0 by the bypass, so 0 is what "still true" looks like.
        stubValue: 0,
        classification: 'STRUCTURALLY_BYPASSED',
        reason:
          'This server uses static-cache batching, and that path never consults the prefix ' +
          'cache — so the question is never asked. The engine asserts this: ' +
          'crates/onnx-genai-engine/tests/batched_static_decode.rs:53 and ' +
          'engine_continuous_batch_scheduled.rs:82 require prefix_cache_hit_len == 0 for every ' +
          'batched result. Showing 0% here would imply a cache that tried and failed.',
      },
      // On the dynamic server the cache IS consulted, so 0 is real data.
      dynamic: { classification: 'MEASURED' },
    },
  },
  'metrics.prefix_cache_lookups': {    source: ENDPOINTS.METRICS,
    metric: 'onnx_genai_prefix_cache_lookups_total',
    kind: 'scalar',
    classification: 'MEASURED',
    unit: 'count',
    evidence:
      'crates/onnx-genai-server/src/metrics.rs:132-134 — fetch_add(1) runs unconditionally in ' +
      'GenerationMetrics::result(), whether or not any cache was consulted.',
    // ⚠️ THE METRIC NAME IS A LIE, and it is upstream of us. This counter is
    // called `prefix_cache_lookups_total`, but it increments once per COMPLETED
    // GENERATION regardless of whether a lookup happened. Labelling it "cache
    // lookups" would report a cache being consulted on a server whose code path
    // never consults one. It is labelled for what it actually counts.
    label: 'Completed generations',
    byOrigin: {
      // Even though the counter itself is unconditional, it lives in the
      // prefix-cache family and a visitor reads it in that context. On the
      // batching server there is no cache activity to contextualise it.
      scatter: {
        unfalsifiable:
          'This counter is not pinned -- it rises on every completed generation. It is ' +
          'suppressed because it does not count what its name claims, and no value on ' +
          'the wire can confirm or deny that.',
        classification: 'STRUCTURALLY_BYPASSED',
        reason:
          'This server batches with a static cache and never consults the prefix cache, so ' +
          'this counter describes generations that bypassed it entirely. The engine asserts ' +
          'the bypass: crates/onnx-genai-engine/tests/batched_static_decode.rs:53.',
      },
      dynamic: { classification: 'MEASURED' },
    },
  },
  'prefix_cache.hit_rate': {
    source: ENDPOINTS.DEBUG_KV,
    path: 'prefix_cache_hit_rate',
    classification: 'MEASURED',
    unit: 'ratio',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:126-130 — hits/lookups, but emits a literal ' +
      '0.0 when lookups == 0, so an undefined rate and a genuine 0% are the same bytes. The ' +
      'store corrects this where the denominator is still in scope.',
    label: 'Prefix-cache hit rate',
    byOrigin: {
      scatter: {
        // Pinned at literal 0 by the bypass, so 0 is what "still true" looks like.
        stubValue: 0,
        classification: 'STRUCTURALLY_BYPASSED',
        reason:
          'This server batches with a static cache, and that path never consults the prefix ' +
          'cache, so there is no rate to report. A 0% here would imply a cache that tried.',
      },
      dynamic: { classification: 'MEASURED' },
    },
  },

  // ── GET /v1/resources ───────────────────────────────────────────────────
  // Ungated and genuinely computed from the configured limits.
  'resources.kv_budget_bytes': {
    source: ENDPOINTS.RESOURCES,
    path: 'derived_kv_budget.bytes',
    classification: 'MEASURED',
    unit: 'bytes',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs — derived_kv_budget is computed from the ' +
      'resolved VRAM limit minus reserved bytes.',
    label: 'Derived KV budget',
  },
  'resources.vram_limit_bytes': {
    source: ENDPOINTS.RESOURCES,
    path: 'vram.limit',
    classification: 'MEASURED',
    unit: 'bytes',
    evidence: 'crates/onnx-genai-server/src/routes/admin.rs — resolved from configured vram limit.',
    label: 'VRAM limit',
  },

  // ── Derived in the browser ──────────────────────────────────────────────
  // Computed by telemetry-store.js rather than read from a response, so
  // buildFields skips it (see `derived`). Listed here so it still appears in
  // the footer's provenance table: a number we computed ourselves needs MORE
  // disclosure than one the server reported, not less.
  'throughput.observed': {
    source: ENDPOINTS.METRICS,
    derived: true,
    classification: 'MEASURED',
    unit: 'tokens/s',
    evidence:
      'Derived by differentiating onnx_genai_tokens_generated_total between two polls. ' +
      'The server hardcodes tokens_per_second: 0.0 (routes/admin.rs:63) because it records ' +
      'totals but no rate; the totals are genuine, so the rate is recoverable client-side.',
    label: 'Throughput (derived from token counter)',
  },
});

/** Field keys whose classification means they can never render as a number. */
export const NEVER_MEASURED_CLASSIFICATIONS = Object.freeze(['DOCUMENTED_ZERO', 'NOT_PLUMBED']);

/**
 * Wire fields that must NEVER be bound to a panel, whatever they are named.
 *
 * These are not stubs. A stub is discoverable: someone greps, finds the
 * hardcoded literal, and fixes it. These are the more dangerous kind the Lead
 * made a standing rule — a REAL, CORRECTLY-COMPUTED value under a name that
 * describes a different quantity. Nothing about the wire looks wrong, so the
 * error survives every review that checks "is this field computed?".
 *
 * Listing them here rather than trusting nobody binds them is the same
 * principle as the Field envelope itself: enforce it in the shape, not in
 * developer discipline. `never-bind.test.js` fails the build if one appears.
 *
 * @type {ReadonlyArray<{endpoint: string, field: string, why: string}>}
 */
export const NEVER_BIND = Object.freeze([
  Object.freeze({
    endpoint: ENDPOINTS.MODELS,
    field: 'created',
    why:
      'It is now_unix() computed inside the per-model map at ' +
      'crates/onnx-genai-server/src/routes/admin.rs:30, so it is the CURRENT TIME, ' +
      'recomputed on every request. It is not a creation date and it ticks if polled. ' +
      'Rendered as "created", it would be a confident, precise, wrong fact about when ' +
      'the model was built.',
  }),
]);

/**
 * Look up a field's provenance entry.
 *
 * @param {string} key
 * @returns {ProvenanceEntry|null}
 */
export function provenanceFor(key, origin = null) {
  if (!Object.prototype.hasOwnProperty.call(PROVENANCE, key)) return null;
  return resolveForOrigin(PROVENANCE[key], origin);
}

/**
 * Does a value observed on the wire still match the placeholder this table was
 * written against?
 *
 * WHY THIS EXISTS. Every classification in this file is a snapshot of server
 * source read at one commit, and the server team's whole job is to invalidate
 * it: when the telemetry work lands, these placeholders become real numbers.
 * A stale table then fails in the direction nobody catches — a genuine
 * measurement rendered as an em-dash. That reads as caution, survives review,
 * and no visitor ever files a bug saying "this number is missing"; they just
 * conclude the feature does not work. It is the same fabrication as printing a
 * stub, only mirrored, and it is strictly harder to see.
 *
 * So the table records the exact literal the server writes today. If the wire
 * value ever differs, this file is provably out of date and says so, loudly,
 * instead of quietly suppressing a real measurement.
 *
 * LIMIT, STATED HONESTLY: when a field becomes real and its true value happens
 * to equal the placeholder, this cannot tell the difference — a documented zero
 * is byte-identical to a measured zero. Only the server can close that gap, by
 * omitting or nulling what it does not compute. This catches every other case.
 *
 * @param {ProvenanceEntry} entry
 * @param {unknown} value Raw value read at `entry.path`, or `undefined`.
 * @returns {boolean} True while the entry's classification is still credible.
 */
export function matchesStub(entry, value) {
  // Some fields are suppressed on SEMANTIC grounds, not because they hold a
  // placeholder: the counter moves, it is simply not counting what its name
  // says. No wire value can confirm or refute that, so there is nothing to
  // check and a rising number is expected rather than suspicious. Entries in
  // that position must say so, and say why.
  if (entry.unfalsifiable) return true;
  if (typeof entry.isStub === 'function') return entry.isStub(value);
  // No declared stub means the path should carry nothing at all.
  if (!('stubValue' in entry)) return value === undefined || value === null;
  const stub = entry.stubValue;
  if (typeof stub === 'object' && stub !== null) {
    return JSON.stringify(value) === JSON.stringify(stub);
  }
  return value === stub;
}

/**
 * Collapse an entry's per-origin overrides against the server we actually
 * asked, yielding a plain entry the store can use without knowing about
 * origins.
 *
 * @param {object} entry
 * @param {string|null} origin
 */
export function resolveForOrigin(entry, origin) {
  const override = entry.byOrigin && origin ? entry.byOrigin[origin] : null;
  return override ? Object.freeze({ ...entry, ...override, byOrigin: undefined }) : entry;
}

/** Every field key, for the footer "What's real, what's not" table (AC10). */
export function allFieldKeys() {
  return Object.keys(PROVENANCE);
}
