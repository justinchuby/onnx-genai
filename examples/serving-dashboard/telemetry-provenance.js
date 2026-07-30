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
 * @typedef {'MEASURED' | 'DOCUMENTED_ZERO' | 'NOT_PLUMBED' | 'STRUCTURALLY_BYPASSED'
 *   | 'MISATTRIBUTED'} Classification
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
 * - `MISATTRIBUTED`   — the server DOES run this code path and DOES compute a
 *                       real number from real runtime state, but the quantity
 *                       it computes is not the quantity its name denotes.
 *                       Renders `unavailable`. This is the only classification
 *                       where the wire value is both live and correct and the
 *                       field is still dishonest, so it is the only one that
 *                       cannot be diagnosed by looking at the number.
 *
 *                       WHY IT EXISTS AS A WORD. Before it, the table had
 *                       `DOCUMENTED_ZERO` (a constant), `NOT_PLUMBED` (absent)
 *                       and `STRUCTURALLY_BYPASSED` (never asked) — and nothing
 *                       for ASKED, ANSWERED, ANSWERING SOMETHING ELSE. Three
 *                       prefix-cache fields were therefore `MEASURED` because
 *                       it was the only remaining option, not because anyone
 *                       judged them measured. A missing word in a vocabulary
 *                       does not read as a gap; it reads as agreement.
 *
 *                       NOT EVERY WRONG NAME LANDS HERE. If the true quantity
 *                       is itself worth showing, the honest fix is to RELABEL
 *                       and keep it `MEASURED` — `prefix_cache.lookups` counts
 *                       completed generations, so it is labelled "Completed
 *                       generations" and still renders. `MISATTRIBUTED` is for
 *                       the ones with no honest label worth rendering.
 *
 * CLASSIFICATION CAN DEPEND ON WHICH SERVER WE ASKED. The demo runs two, and
 * they differ structurally: the scatter server batches (bypassing the page
 * table and prefix trie) while the dynamic server pages KV (disabling
 * continuous batching). An entry may therefore carry `byOrigin` to override
 * its classification per server. `prefix_cache_lookups` is a genuine measured
 * count of completed generations on the dynamic server and not-applicable on
 * the scatter server — the SAME wire value, opposite treatments, and only this
 * table knows which.
 *
 * ⚠️ THIS PARAGRAPH USED TO SAY `prefix_cache_hits: 0` IS A GENUINE MEASURED
 * ZERO ON THE DYNAMIC SERVER. That was false in a way worth recording, because
 * a doc comment was the PRODUCER of a false classification on the rendered
 * page rather than a stale description of one: it justified `MEASURED` on the
 * premise that the value would be 0, and the value is not 0. It is ~0.95. The
 * premise was never checked against the wire because a sentence about a zero
 * invites no arithmetic.
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
    classification: 'STRUCTURALLY_BYPASSED',
    unit: 'sessions',
    // The wire carries nothing for this field, and that absence is itself the
    // checkable claim: the day a number arrives, "no session can be paused" has
    // become false and the staleness detector must fire. Declared explicitly
    // rather than exempting STRUCTURALLY_BYPASSED as a class -- most entries in
    // that class DO carry a plausible value, which is the whole reason they are
    // dangerous, so a blanket exemption would gut the guard to fix one entry.
    isStub: (value) => value === undefined || value === null,
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs — `paused_sessions: None`, registered in ' +
      'the handler unavailable map via `FieldUnavailable::not_applicable`, reason ' +
      '`the driver runs generations to completion without preemption, so no session can be ' +
      'paused`.',
    label: 'Paused sessions',
    reason:
      'Not a missing number — a concept that does not apply to this scheduler. The driver ' +
      'runs each generation inline to completion, so no session can ever be paused. The ' +
      'server states the distinction itself: not-applicable, not unavailable.',
  },
  'batch.utilization': {
    source: ENDPOINTS.STATUS,
    path: 'batch_utilization',
    classification: 'MEASURED',
    unit: 'ratio',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs — `fn batch_utilization(in_flight: u64, ' +
      'capacity: usize) -> f32` divides the live in-flight count by the capacity the server ' +
      'admitted against, guards the zero-capacity case, and clamps to 1.0.',
    label: 'Batch utilization',
    reason:
      'This was a hardcoded 0.0 earlier in the project and this table said so. It is now a ' +
      'real computation over two live values, and its denominator is published beside it as ' +
      '`batch_capacity`, so the client never has to assume a capacity no endpoint confirms.',
  },
  'throughput.tokens_per_second': {
    source: ENDPOINTS.STATUS,
    path: 'tokens_per_second',
    classification: 'NOT_PLUMBED',
    // OLDER BINARIES SEND A LITERAL 0.0 HERE. The current server omits the
    // field, so absence is the normal case and is handled by the caller's
    // presence check. But several builds in circulation tonight predate that
    // change, and without this declaration a 0 from one of them reads as a
    // value CONTRADICTING the table -- which makes the store display it as a
    // live measurement. The fabricated zero would arrive wearing the one badge
    // this project exists to withhold. A non-zero value still raises the
    // staleness warning, so the branch keeps its purpose.
    stubValue: 0,
    unit: 'tokens/s',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs — `tokens_per_second: None`, registered ' +
      'in the handler unavailable map via `FieldUnavailable::unavailable`, reason ' +
      '`only a cumulative token count is recorded; a lifetime average would misreport as a ' +
      'current rate -- see the latency histograms on /metrics`.',
    label: 'Server-reported tokens/sec',
    reason:
      'Omitted, not zeroed. Only a cumulative total is recorded; dividing it by uptime gives ' +
      'a lifetime average that reads as a current rate — lowest exactly when the node has ' +
      'been idle longest. This demo measures throughput client-side from streamed tokens and ' +
      'labels it as client-measured.',
  },

  // ---------------------------------------------------------------- paged KV (omitted today, not zeroed)
  'kv.usage': {
    source: ENDPOINTS.STATUS,
    path: 'kv_usage',
    classification: 'NOT_PLUMBED',
    // OLDER BINARIES SEND A LITERAL 0.0 HERE. The current server omits the
    // field, so absence is the normal case and is handled by the caller's
    // presence check. But several builds in circulation tonight predate that
    // change, and without this declaration a 0 from one of them reads as a
    // value CONTRADICTING the table -- which makes the store display it as a
    // live measurement. The fabricated zero would arrive wearing the one badge
    // this project exists to withhold. A non-zero value still raises the
    // staleness warning, so the branch keeps its purpose.
    stubValue: 0,
    unit: 'ratio',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs — `kv_usage: None`. The four KV fields ' +
      'travel together and are omitted from the payload, not sent as zeros.',
    label: 'KV utilization',
    reason:
      'The status handler holds no engine reference, so it cannot see the paged-KV pool and ' +
      'omits the field rather than sending 0. An absent value renders as "not measured here", ' +
      'which is a different and stronger claim than a zero. Live paged-KV data comes from the ' +
      'KV block-table endpoint instead.',
  },
  'kv.pages_used': {
    source: ENDPOINTS.STATUS,
    path: 'kv_pages_used',
    classification: 'NOT_PLUMBED',
    unit: 'pages',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs — `kv_pages_used: None`, omitted from the ' +
      'payload rather than sent as a zero.',
    label: 'KV pages in use',
    reason:
      'Not exposed to the HTTP layer; the field is omitted, not zeroed. Absent says "not ' +
      'measured here"; a zero would say "measured, and the pool is empty".',
  },
  'kv.pages_total': {
    source: ENDPOINTS.STATUS,
    path: 'kv_pages_total',
    classification: 'NOT_PLUMBED',
    unit: 'pages',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs — `kv_pages_total: None`, omitted from the ' +
      'payload rather than sent as a zero.',
    label: 'KV pages total',
    reason:
      'Not exposed to the HTTP layer; the field is omitted, not zeroed. This is the trap of ' +
      'the group: it reads a real structure, so a non-zero value would survive any ' +
      '"is this hardcoded?" audit while describing a pool the decoder never uses. A non-zero ' +
      'value is not evidence that a mechanism is in play.',
  },
  'kv.pages_shared': {
    source: ENDPOINTS.STATUS,
    path: 'kv_pages_shared',
    classification: 'NOT_PLUMBED',
    unit: 'pages',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs — `kv_pages_shared: None`, omitted from ' +
      'the payload rather than sent as a zero.',
    label: 'KV pages shared',
    reason:
      'Not exposed to the HTTP layer; the field is omitted, not zeroed. Sharing is ' +
      'page-granular, so even when this is plumbed a zero will not mean "no reuse happened".',
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
  //
  // ⚖️ RULED AND CLOSED, @12e42da8 — THE THREE HIT-DERIVED FIELDS ARE
  // `MISATTRIBUTED` ON THE DYNAMIC ORIGIN. The two `lookups` counters are NOT,
  // and the difference is the whole point of the new word.
  //
  // WHAT IS ACTUALLY WRONG. Not a stub, and not a dead counter. On dynamic the
  // cache IS consulted and the counter IS live -- it is precisely computed,
  // beautifully behaved, and answering a different question than its name:
  //
  //   metrics.rs:232-237  prefix_reuse_increments(len) -> (0,0) if len == 0
  //                                                       (1, len) OTHERWISE
  //
  // One hit for ANY nonzero token match. Every /v1/chat/completions request
  // shares the ~24-token chat-template preamble, so every request scores a hit
  // whether or not a single prompt was reused. Measured, not reasoned: twelve
  // requests with six DELIBERATELY UNIQUE prompts produced +12 hits -- one per
  // completed generation, unique prompts included. A COUNTER THAT READS THE
  // SAME WITH AND WITHOUT REUSE IS NOT MEASURING REUSE.
  //
  // THE DELTA IS THE EVIDENCE, NEVER A RATE (@12e42da8, ruled; reasoning in
  // prefix-counters-forbidden.test.js). Both counters are CUMULATIVE SINCE
  // BOOT, so their ratio is a property of the process rather than of the
  // experiment -- diluted by warm-up and tunable to any value by sending more
  // traffic. Four different rates for this one finding appear across our
  // documents, every one honestly transcribed and not one of them evidence.
  //
  // The server's own docstring concedes it at metrics.rs:230-231 -- "the hit
  // count cannot express value: reusing 8 tokens of a 900-token prompt and
  // reusing 890 are both exactly one hit". The defect was documented upstream
  // and classified as MEASURED downstream, which is the entire failure in one
  // sentence.
  //
  // WHY THIS IS THE DANGEROUS DIRECTION. Every other safeguard in this tree
  // hunts fabricated ZEROS -- suppressUndefinedHitRate, the stubValue pins, the
  // 0/0 correction. A confident ~95% trips none of them and invites no scrutiny
  // at all. The honesty apparatus is asymmetric and this is the side it misses.
  //
  // WHY NOT NEVER_BIND, WHICH IS WHERE THIS WAS HEADED. NEVER_BIND is the
  // accurate description, but never-bind.test.js forbids any PROVENANCE entry
  // from naming a never-bind field, so promoting these DELETES their table
  // entries -- and takes 28 tests across three agents' files with them,
  // including the ratified per-origin rule and the zero-denominator guard.
  // Measured, not guessed: I ran it. Under freeze that is a demolition, not a
  // fix. MISATTRIBUTED keeps the entry, keeps the evidence, keeps the tests,
  // and still refuses to render. The ban list is for fields with no entry to
  // hang the reason on; this table can hold its own reason.
  //
  // WHY THE lookups PAIR IS UNTOUCHED. `prefix_cache_lookups` increments once
  // per completed generation unconditionally -- so it is an honest count of
  // COMPLETED GENERATIONS, and it is already labelled that. A true number with
  // a fixable name is a RELABEL, not a reclassification. Reclassifying it would
  // suppress a working measurement and teach the next reader that the word
  // means "the name is wrong" rather than "there is nothing honest to show".
  //
  // CONTAINMENT WAS ALREADY LOAD-BEARING AND STAYS: prefix-counters-forbidden.test.js
  // fails any NEW module naming these in either the underscored wire spelling or
  // the dotted store key. That test is the executable half and the only half
  // worth citing -- an earlier version of this comment argued safety from
  // "dashboard/prefix-cache.js does not exist", which was true when written and
  // would have become a false all-clear the moment anyone re-added that module.
  // A safety argument premised on a file's absence rots; one premised on a test
  // goes red instead.
  //
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
      dynamic: {
        unfalsifiable:
          'This counter is not pinned -- it rises on every completed generation, so no '
          + 'value on the wire can confirm or deny that it is counting the wrong thing. '
          + 'The disqualifying evidence is in the SERVER SOURCE, not in the reading: '
          + 'metrics.rs:232-237 returns one hit for any nonzero match. If that function '
          + 'ever weights a hit by reuse length, this classification is stale.',
        classification: 'MISATTRIBUTED',
        reason:
          'The counter is live and correct on this server -- and it counts GENERATIONS WITH AT LEAST ONE MATCHING TOKEN, not cache hits. crates/onnx-genai-server/src/metrics.rs:232-237 scores exactly one hit for any prefix_cache_hit_len > 0, and every /v1/chat/completions request shares the ~24-token chat-template preamble, so it reads the same with and without reuse (measured: 12 requests, 6 deliberately unique prompts, +12 hits). metrics.rs:230-231 concedes it upstream: reusing 8 tokens of a 900-token prompt and reusing 890 are both exactly one hit.',
      },
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
    classification: 'NOT_PLUMBED',
    unit: null,
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs — `prefix_hashes: None`, registered in ' +
      'the handler unavailable map rather than sent as an empty list.',
    label: 'Prefix hashes',
    reason:
      'The engine does not surface prefix hashes to the HTTP layer, so the field is omitted. ' +
      'It previously shipped as an empty list, which is the worst of the three options: an ' +
      'empty collection reads as "measured, and there were none" rather than "not measured". ' +
      'Absent says the second thing and cannot be mistaken for the first.',
  },

  // ---------------------------------------------------------------- batching / admission
  'batch.capacity': {
    source: ENDPOINTS.STATUS,
    path: 'batch_capacity',
    classification: 'MEASURED',
    unit: 'requests',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs — `batch_capacity` is serialised from ' +
      'AppConfig::effective_batch_capacity(), which state.rs defines as ' +
      'max_batch.min(max_queue_depth). Genuinely computed from configuration; no stub.',
    // The denominator of the "N of M" the scheduling panel renders. It is
    // deliberately NOT max_batch: state.rs documents that max_batch alone
    // overstates the ceiling whenever admission is the tighter constraint --
    // with max_batch 4 and max_queue_depth 1 the batch can never exceed one, so
    // a max_batch denominator would draw a saturated server as 25% busy. A
    // denominator that overstates capacity is the one error direction that
    // makes our headline look WORSE than reality, which is why the server
    // clamps it and why the client must not "helpfully" un-clamp it.
    //
    // ⚠️ THIS KEY WAS DECLARED TWICE, and the duplicate is why the label reads
    // the way it does. Two commits added it independently; JS keeps the LAST
    // definition and discards the first with no error, so the surviving entry
    // was the terse one labelled "Batch limit" -- and "batch limit" is the name
    // of max_batch, the RAW value, not of the clamped minimum actually served.
    // The paragraph above warning against exactly that confusion was sitting in
    // the DISCARDED half. A duplicate key does not merely make provenance
    // ambiguous; it picks a winner silently, and here it picked the misnomer.
    // provenance-expiry.test.js now fails on any duplicate key.
    label: 'Effective batch capacity',
  },
  'batch.active_size': {
    source: ENDPOINTS.DEBUG_KV,
    path: 'active_batch_size',
    classification: 'MEASURED',
    unit: 'requests',
    evidence: 'crates/onnx-genai-server/src/routes/admin.rs:136 (snapshot.current_batch_size)',
    label: 'Sequences in the current batch',
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
  // AND IT NO LONGER COSTS A STALL, WHICH IS WHY AC63 AND D88 ARE NO LONGER IN
  // CONFLICT. This used to be read off /metrics, and D87/D88 banned polling
  // that endpoint (14,784 ms during a generation vs 0.8 ms idle -- it stalls
  // precisely when concurrency is worth showing). The server now emits the same
  // count directly on /v1/status at ~1.8 ms, so the field the ruling asked us
  // to bind is reachable without the stall. The caveat above is unchanged: this
  // counts REQUESTS IN FLIGHT, not the engine batch.
  'batch.in_flight': {
    source: ENDPOINTS.STATUS,
    path: 'batch_in_flight',
    classification: 'MEASURED',
    unit: 'requests',
    evidence:
      'crates/onnx-genai-server/src/routes/admin.rs:175 (batch_in_flight, from ' +
      'snapshot.current_batch_size). Counts in-flight generations, NOT the engine batch.',
    label: 'Generations in flight',
  },
  // The denominator, SERVED rather than assumed. It is
  // `effective_batch_capacity()` = min(max_batch, max_queue_depth), NOT
  // max_batch: max_batch alone overstates the ceiling whenever the queue is the
  // binding constraint, so occupancy would read low against a limit the server
  // would never reach. A panel previously bound `scheduler.max_batch`, which no
  // server has ever emitted -- four test fixtures supplied it, so the suite was
  // green while the panel was permanently degraded live.
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
      'crates/onnx-genai-engine/src/batched.rs:101-110 — ContinuousBatchManager steps a ' +
      'batch but exposes no counter for it; onnx_genai_batch_size_current counts HTTP-layer ' +
      'in-flight generations instead.',
    label: 'Sequences stepped together',
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
      dynamic: {
        unfalsifiable:
          'This counter is not pinned -- it rises on every completed generation, so no '
          + 'value on the wire can confirm or deny that it is counting the wrong thing. '
          + 'The disqualifying evidence is in the SERVER SOURCE, not in the reading: '
          + 'metrics.rs:232-237 returns one hit for any nonzero match. If that function '
          + 'ever weights a hit by reuse length, this classification is stale.',
        classification: 'MISATTRIBUTED',
        reason:
          'The counter is live and correct on this server -- and it counts GENERATIONS WITH AT LEAST ONE MATCHING TOKEN, not cache hits. crates/onnx-genai-server/src/metrics.rs:232-237 scores exactly one hit for any prefix_cache_hit_len > 0, and every /v1/chat/completions request shares the ~24-token chat-template preamble, so it reads the same with and without reuse (measured: 12 requests, 6 deliberately unique prompts, +12 hits). metrics.rs:230-231 concedes it upstream: reusing 8 tokens of a 900-token prompt and reusing 890 are both exactly one hit.',
      },
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
          'cache, so there is no rate to report. A 0% here would imply a cache that tried. ' +
          'The engine asserts the bypass: ' +
          'crates/onnx-genai-engine/tests/batched_static_decode.rs:53 requires ' +
          'prefix_cache_hit_len == 0 for every batched result.',
      },
      dynamic: {
        unfalsifiable:
          'This counter is not pinned -- it rises on every completed generation, so no '
          + 'value on the wire can confirm or deny that it is counting the wrong thing. '
          + 'The disqualifying evidence is in the SERVER SOURCE, not in the reading: '
          + 'metrics.rs:232-237 returns one hit for any nonzero match. If that function '
          + 'ever weights a hit by reuse length, this classification is stale.',
        classification: 'MISATTRIBUTED',
        reason:
          'The counter is live and correct on this server -- and it counts GENERATIONS WITH AT LEAST ONE MATCHING TOKEN, not cache hits. crates/onnx-genai-server/src/metrics.rs:232-237 scores exactly one hit for any prefix_cache_hit_len > 0, and every /v1/chat/completions request shares the ~24-token chat-template preamble, so it reads the same with and without reuse (measured: 12 requests, 6 deliberately unique prompts, +12 hits). metrics.rs:230-231 concedes it upstream: reusing 8 tokens of a 900-token prompt and reusing 890 are both exactly one hit.',
      },
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

/**
 * Classifications that can never yield a number, so a field carrying one is
 * UNAVAILABLE from the first frame rather than pending.
 *
 * STRUCTURALLY_BYPASSED belongs here for the reason @0837fdf9 ratified about
 * `pending`: pending resolves on its own, unavailable never will, and telling
 * a visitor to wait for a number that is never coming is its own small
 * dishonesty. A bypassed subsystem is not slow — it is not in this
 * configuration's execution path at all, so no amount of waiting produces a
 * value on this server.
 *
 * It was previously omitted here and special-cased at ONE of the three call
 * sites, which meant the other two treated a bypassed field as merely
 * un-arrived: on the scatter server the prefix-cache fields rendered a spinner
 * promising a number that could never arrive. Keeping the rule in one list is
 * what stops the three sites from disagreeing again.
 *
 * MISATTRIBUTED belongs here for a DIFFERENT reason than the other three, and
 * the difference matters if anyone ever tries to render it. The other three
 * have no number, or a fake one. This one has a real, live, correct number —
 * it is suppressed because the number answers a question nobody asked. So it
 * is the only member of this list where the wire value looks perfect, and the
 * only one whose suppression cannot be justified by pointing at the value.
 */
export const NEVER_MEASURED_CLASSIFICATIONS = Object.freeze([
  'DOCUMENTED_ZERO',
  'NOT_PLUMBED',
  'STRUCTURALLY_BYPASSED',
  'MISATTRIBUTED',
]);

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
