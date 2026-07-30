// The single registry of keys the dashboard asks for and nothing publishes.
//
// WHY THIS MODULE EXISTS
// ----------------------
// This inventory used to live in three places at once:
//
//   dashboard/field-keys.test.js   NOT_YET_PUBLISHED -- key -> one-line reason,
//                                  the allowlist that stops a real typo from
//                                  being reported as a deferral.
//   check-unplumbed-claims.test.js UNPLUMBED_CLAIMS -- key -> {reason,
//                                  absentWireNames}, the falsifiable half.
//   check-binding-liveness.test.js a readFileSync + regex that LIFTED the first
//                                  one out of the other file's source text.
//
// Keyset drift was already guarded, so the keys could not disagree. What was
// never guarded is that each key carried TWO INDEPENDENTLY WORDED REASONS, and
// nothing on earth reconciled them. One could be corrected and the other left
// stale, and every suite would stay green -- the same "two statements of one
// fact, only one of which can go red" defect this branch has been paying for
// all evening, sitting inside the very files built to catch it.
//
// WHY THE THIRD COPY WAS A readFileSync AND NOT AN IMPORT
// ------------------------------------------------------
// Not carelessness, and worth preserving as a note: importing a `.test.js` file
// for one export RUNS that file's tests inside the importer's run, inflating
// the suite total and reporting another file's failures under this one's name.
// check-binding-liveness.test.js reached for the source text to dodge that.
//
// The regex was the workaround; the missing registry was the disease. A plain
// `.mjs` module has no tests to trigger, so every consumer can now just import
// it, and the by-name file read is deleted rather than merely guarded.
//
// WHAT AN ENTRY MUST CARRY, AND WHY EACH FIELD IS NOT OPTIONAL
// -----------------------------------------------------------
//   class            One of CLASS below. "Not published" is not one situation,
//                    and flattening five situations into one list is how six
//                    client-measured rows once sat among server gaps promising
//                    a server change that could not possibly deliver them.
//   summary          The one-line form the allowlist shows. DERIVED FROM HERE,
//                    never retyped.
//   reason           The full argument, for a reader deciding whether the claim
//                    still holds.
//   absentWireNames  The falsifier. A free-text reason cannot be checked; a
//                    wire name can. These are the spellings a server WOULD use,
//                    not our key names -- our vocabulary proves nothing about
//                    theirs. Landing the feature turns the guard red.
//   evidence         Structured citations, `{file, symbol, note}`. Required for
//                    any class that claims something IS present but unusable,
//                    because those claims cannot be falsified by a name scan --
//                    if the thing exists, "the name is absent" says nothing.
//                    Checked to resolve against real files.

/**
 * Why a key is unpublished. These are NOT interchangeable severities; they say
 * who could fix it, and three of them can never be fixed by the server.
 *
 * @type {Readonly<Record<string, string>>}
 */
export const CLASS = Object.freeze({
  /** No such field exists yet. A server release could supply it. */
  SERVER_GAP: 'SERVER_GAP',
  /**
   * The server DOES measure it, but publishes a shape that cannot answer the
   * question. Deriving the answer anyway would be a fabricated measurement in a
   * plausible costume -- the p95-from-14-buckets trap.
   */
  SHAPE_MISMATCH: 'SHAPE_MISMATCH',
  /** Nothing the server could ever send would supply it; it is client-side. */
  NEVER_SERVER_SUPPLIED: 'NEVER_SERVER_SUPPLIED',
  /** On the wire and deliberately NOT bound. Our choice, not the server's. */
  DELIBERATE_BAN: 'DELIBERATE_BAN',
  /** The event it counts cannot occur in this build, so zero would mislead. */
  EVENT_CANNOT_OCCUR: 'EVENT_CANNOT_OCCUR',
});

/** Classes whose claim is about something PRESENT, so a name scan cannot falsify them. */
export const CLASSES_REQUIRING_EVIDENCE = Object.freeze([
  CLASS.SHAPE_MISMATCH,
  CLASS.DELIBERATE_BAN,
  CLASS.EVENT_CANNOT_OCCUR,
]);

/**
 * @typedef {object} Citation
 * @property {string} file   Repo-relative path. Never a bare basename -- two
 *                           crates ship a metrics.rs.
 * @property {string} symbol Identifier or declaration to look for in that file.
 * @property {string} note   What the citation establishes.
 */

/**
 * @typedef {object} UnplumbedEntry
 * @property {string} class
 * @property {string} summary
 * @property {string} reason
 * @property {readonly string[]} absentWireNames
 * @property {readonly Citation[]} evidence
 */

/** @type {Readonly<Record<string, UnplumbedEntry>>} */
export const UNPLUMBED = Object.freeze({
  'kv.allocations': {
    class: CLASS.SERVER_GAP,
    summary: 'the pool keeps no cumulative allocation ledger',
    reason:
      'The pool keeps no cumulative allocation ledger. BlockTableResponse serves occupancy ' +
      '(pages_in_use) and pressure (hot_evictions, allocation_failures), never a lifetime ' +
      'allocation count.',
    absentWireNames: ['allocations_total', 'total_allocations'],
    evidence: [],
  },
  'kv.frees': {
    class: CLASS.SERVER_GAP,
    summary: 'no free events are counted; a released page reports ref_count 0',
    reason:
      'The counterpart to kv.allocations, and absent for the same reason: a released page ' +
      'reports ref_count 0, which is an occupancy fact, not a free event anyone counts.',
    absentWireNames: ['frees', 'frees_total', 'total_frees'],
    evidence: [],
  },
  'kv.prefix_evictions': {
    class: CLASS.SERVER_GAP,
    summary: 'evictions are counted by tier, not by cause',
    reason:
      'Evictions are counted by TIER, not by cause. `hot_evictions` is the only eviction ' +
      'counter on the wire, and it does not distinguish a prefix-driven eviction from any ' +
      'other. Splitting it here would invent a breakdown the server never computed.',
    absentWireNames: ['prefix_evictions', 'prefix_evictions_total'],
    evidence: [],
  },
  'queue.depth_peak': {
    class: CLASS.SERVER_GAP,
    summary: 'peak tracking not yet plumbed server-side',
    reason:
      'The server reports instantaneous depth (pending_queue_depth, queue_depth) and keeps ' +
      'no high-water mark. A peak computed client-side would be the peak SINCE THIS TAB ' +
      'OPENED, which is a different quantity wearing the same label.',
    absentWireNames: ['depth_peak', 'peak_queue_depth', 'queue_depth_peak', 'high_water'],
    evidence: [],
  },
  'scheduler.preemptions_total': {
    class: CLASS.EVENT_CANNOT_OCCUR,
    summary: 'scheduler introspection not yet plumbed',
    reason:
      'The driver runs each generation to completion without preemption, so the event this ' +
      'counter would count cannot occur. Consistent with `paused_sessions: None`, which the ' +
      'server itself registers as not-applicable rather than unavailable.',
    absentWireNames: ['preemptions', 'preemptions_total', 'preempted'],
    evidence: [
      { file: 'crates/onnx-genai-server/src/routes/admin.rs', symbol: 'paused_sessions: None', note: 'server itself reports not-applicable rather than zero' },
    ],
  },
  'prefix_cache.evictions': {
    class: CLASS.SERVER_GAP,
    summary: 'savings metrics not yet plumbed server-side',
    reason:
      'No prefix-cache eviction counter exists in metrics.rs or any route.',
    absentWireNames: ['prefix_cache_evictions', 'prefix_cache_evictions_total'],
    evidence: [],
  },
  'prefix_cache.prefill_tokens_skipped': {
    class: CLASS.SERVER_GAP,
    summary: 'savings metrics not yet plumbed server-side',
    reason:
      'The server counts tokens MATCHED (prefix_tokens_reused / ' +
      'prefix_cache_hit_tokens_total), not tokens whose prefill was actually skipped. On ' +
      'the token-prefix branch a counted reuse skips no prefill at all (engine/runtime.rs ' +
      'computes an LCP without retaining a page), so binding matched tokens to this key ' +
      'would overstate the saving.',
    absentWireNames: ['prefill_tokens_skipped', 'tokens_skipped', 'prefill_skipped'],
    evidence: [],
  },
  'prefix_cache.time_saved_ms': {
    class: CLASS.SERVER_GAP,
    summary: 'savings metrics not yet plumbed server-side',
    reason:
      'No timer measures prefill time avoided. It could only be inferred from skipped ' +
      'tokens, which are themselves not measured — an estimate stacked on an estimate.',
    absentWireNames: ['time_saved', 'time_saved_ms', 'prefill_time_saved'],
    evidence: [],
  },
  'prefix_cache.tokens_reused': {
    class: CLASS.DELIBERATE_BAN,
    summary: 'savings metrics not yet plumbed server-side',
    reason:
      'The count IS on the wire, under `prefix_tokens_reused`, and is deliberately NOT ' +
      'bound: prefix-counters-forbidden.test.js and the NEVER_BIND register ban this family ' +
      'because the numerator scores one for any nonzero match. This is a BAN, not a gap — ' +
      'the only entry here whose absence is our choice rather than the server\'s.',
    absentWireNames: [],
    evidence: [
      { file: 'crates/onnx-genai-server/src/routes/mod.rs', symbol: 'prefix_tokens_reused', note: 'the count IS served; this entry is a ban, not an absence' },
      { file: 'examples/serving-dashboard/never-bind.test.js', symbol: 'NEVER_BIND', note: 'the register that bans binding it' },
    ],
  },
  'latency.ttft_server_p50': {
    class: CLASS.SHAPE_MISMATCH,
    summary: 'percentile aggregation not yet plumbed',
    reason:
      'The server exports a bucketed histogram, not percentiles: metrics.rs histogram() ' +
      'writes _bucket{le=}/_sum/_count over 14 fixed bounds and no quantile label exists in ' +
      'the crate. The mean is derivable and is bound as metrics.ttft / metrics.e2e_latency; ' +
      'a percentile is not.',
    absentWireNames: ['quantile'],
    evidence: [
      { file: 'crates/onnx-genai-server/src/metrics.rs', symbol: 'LATENCY_BUCKETS_NS', note: '14 fixed bounds; the only latency shape on the wire' },
      { file: 'crates/onnx-genai-server/src/metrics.rs', symbol: 'fn histogram', note: 'writes _bucket{le=}/_sum/_count; emits no quantile label' },
    ],
  },
  'latency.ttft_server_p95': {
    class: CLASS.SHAPE_MISMATCH,
    summary: 'percentile aggregation not yet plumbed',
    reason:
      'The server exports a bucketed histogram, not percentiles: metrics.rs histogram() ' +
      'writes _bucket{le=}/_sum/_count over 14 fixed bounds and no quantile label exists in ' +
      'the crate. The mean is derivable and is bound as metrics.ttft / metrics.e2e_latency; ' +
      'a percentile is not.',
    absentWireNames: ['quantile'],
    evidence: [
      { file: 'crates/onnx-genai-server/src/metrics.rs', symbol: 'LATENCY_BUCKETS_NS', note: '14 fixed bounds; the only latency shape on the wire' },
      { file: 'crates/onnx-genai-server/src/metrics.rs', symbol: 'fn histogram', note: 'writes _bucket{le=}/_sum/_count; emits no quantile label' },
    ],
  },
  'latency.ttft_server_max': {
    class: CLASS.SHAPE_MISMATCH,
    summary: 'percentile aggregation not yet plumbed',
    reason:
      'The server exports a bucketed histogram, not percentiles: metrics.rs histogram() ' +
      'writes _bucket{le=}/_sum/_count over 14 fixed bounds and no quantile label exists in ' +
      'the crate. The mean is derivable and is bound as metrics.ttft / metrics.e2e_latency; ' +
      'a percentile is not.',
    absentWireNames: ['quantile'],
    evidence: [
      { file: 'crates/onnx-genai-server/src/metrics.rs', symbol: 'LATENCY_BUCKETS_NS', note: '14 fixed bounds; the only latency shape on the wire' },
      { file: 'crates/onnx-genai-server/src/metrics.rs', symbol: 'fn histogram', note: 'writes _bucket{le=}/_sum/_count; emits no quantile label' },
    ],
  },
  'latency.e2e_server_p50': {
    class: CLASS.SHAPE_MISMATCH,
    summary: 'percentile aggregation not yet plumbed',
    reason:
      'The server exports a bucketed histogram, not percentiles: metrics.rs histogram() ' +
      'writes _bucket{le=}/_sum/_count over 14 fixed bounds and no quantile label exists in ' +
      'the crate. The mean is derivable and is bound as metrics.ttft / metrics.e2e_latency; ' +
      'a percentile is not.',
    absentWireNames: ['quantile'],
    evidence: [
      { file: 'crates/onnx-genai-server/src/metrics.rs', symbol: 'LATENCY_BUCKETS_NS', note: '14 fixed bounds; the only latency shape on the wire' },
      { file: 'crates/onnx-genai-server/src/metrics.rs', symbol: 'fn histogram', note: 'writes _bucket{le=}/_sum/_count; emits no quantile label' },
    ],
  },
  'latency.e2e_server_p95': {
    class: CLASS.SHAPE_MISMATCH,
    summary: 'percentile aggregation not yet plumbed',
    reason:
      'The server exports a bucketed histogram, not percentiles: metrics.rs histogram() ' +
      'writes _bucket{le=}/_sum/_count over 14 fixed bounds and no quantile label exists in ' +
      'the crate. The mean is derivable and is bound as metrics.ttft / metrics.e2e_latency; ' +
      'a percentile is not.',
    absentWireNames: ['quantile'],
    evidence: [
      { file: 'crates/onnx-genai-server/src/metrics.rs', symbol: 'LATENCY_BUCKETS_NS', note: '14 fixed bounds; the only latency shape on the wire' },
      { file: 'crates/onnx-genai-server/src/metrics.rs', symbol: 'fn histogram', note: 'writes _bucket{le=}/_sum/_count; emits no quantile label' },
    ],
  },
  'latency.e2e_server_max': {
    class: CLASS.SHAPE_MISMATCH,
    summary: 'percentile aggregation not yet plumbed',
    reason:
      'The server exports a bucketed histogram, not percentiles: metrics.rs histogram() ' +
      'writes _bucket{le=}/_sum/_count over 14 fixed bounds and no quantile label exists in ' +
      'the crate. The mean is derivable and is bound as metrics.ttft / metrics.e2e_latency; ' +
      'a percentile is not.',
    absentWireNames: ['quantile'],
    evidence: [
      { file: 'crates/onnx-genai-server/src/metrics.rs', symbol: 'LATENCY_BUCKETS_NS', note: '14 fixed bounds; the only latency shape on the wire' },
      { file: 'crates/onnx-genai-server/src/metrics.rs', symbol: 'fn histogram', note: 'writes _bucket{le=}/_sum/_count; emits no quantile label' },
    ],
  },
  'scenario.makespan_ms': {
    class: CLASS.NEVER_SERVER_SUPPLIED,
    summary: 'supplied by the scenario runner, not the server',
    reason:
      'Supplied by the scenario runner in the browser, never by the server. Listed so the ' +
      'key is not mistaken for a typo; it is not a server gap and no server change would ' +
      'supply it.',
    absentWireNames: ['makespan', 'makespan_ms'],
    evidence: [],
  },
  'server.decode_backend': {
    class: CLASS.SERVER_GAP,
    summary: 'build metadata not yet exposed',
    reason:
      'The decode path is chosen at runtime from the model config and is not echoed by any ' +
      'route. /v1/debug/config serves `pipeline`, which names the model family, not the ' +
      'decode backend.',
    absentWireNames: ['decode_backend', 'decoder_backend'],
    evidence: [],
  },
  'server.quantization': {
    class: CLASS.SERVER_GAP,
    summary: 'build metadata not yet exposed',
    reason:
      'No route reports the weight quantization of the loaded model.',
    absentWireNames: ['quantization', 'quant_type', 'weight_dtype'],
    evidence: [],
  },
  'server.uptime_ms': {
    class: CLASS.SERVER_GAP,
    summary: 'build metadata not yet exposed',
    reason:
      'No endpoint exposes uptime, start time or build id. The capture manifest records the ' +
      'same finding independently: "no endpoint exposes uptime, start time or build id, so ' +
      'nothing on the wire can date this capture".',
    absentWireNames: ['uptime', 'uptime_ms', 'started_at', 'start_time'],
    evidence: [],
  },
  'server.version': {
    class: CLASS.SERVER_GAP,
    summary: 'build metadata not yet exposed',
    reason:
      'No binary carries its commit: no vergen, no env!("GIT_*"), no stamping build.rs and ' +
      'no version endpoint. Recorded independently in fixtures/captures/manifest.json.',
    absentWireNames: ['build_version', 'server_version', 'git_sha', 'commit_sha'],
    evidence: [],
  },
});

/** Keys in the registry, sorted. */
export function unplumbedKeys() {
  return Object.keys(UNPLUMBED).sort();
}

/**
 * The allowlist shape `dashboard/field-keys.test.js` needs: key -> one-line
 * reason. DERIVED, so the short and long forms cannot drift apart.
 *
 * @returns {Readonly<Record<string, string>>}
 */
export function summaryAllowlist() {
  return Object.freeze(
    Object.fromEntries(Object.entries(UNPLUMBED).map(([key, e]) => [key, e.summary])),
  );
}

/** Entries of a given class. */
export function keysOfClass(className) {
  return Object.entries(UNPLUMBED)
    .filter(([, e]) => e.class === className)
    .map(([key]) => key)
    .sort();
}
