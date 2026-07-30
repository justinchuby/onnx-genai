// Copyright (c) Microsoft Corporation.
//
// The unplumbed-claims guard.
//
// A CLAIM OF ABSENCE IS A CLAIM ABOUT THE SERVER, AND NOTHING IN THIS
// REPOSITORY EVER HELD ONE AGAINST THE SERVER.
//
// `dashboard/field-keys.test.js` reconciles every key a panel requests against
// the keys the store publishes, and it is good: it reconstructs template-built
// keys, it proves its own corpus non-empty, and it fails when a listed key
// starts being published. But it has an escape hatch — `NOT_YET_PUBLISHED` —
// and that hatch is where this class of defect lives. An entry there is a
// free-text sentence. `'block-table endpoint, not yet landed'` is a statement
// about somebody else's source tree, written once, by hand, and then never
// evaluated by anything ever again.
//
// COMPARE THE HATCH NEXT DOOR. `check-binding-liveness.test.js` has the same
// shape of list and treats it as expensive: `reason` AND `evidence` are both
// required, a stale entry FAILS, and `MAX_DECLARED_ABSENT` caps the whole list
// so growing it is a diff a reviewer must approve. `NOT_YET_PUBLISHED` has
// none of that. It is uncapped, unevidenced, and unchecked, and it had grown
// to forty entries.
//
// WHY THAT MATTERS MORE THAN IT SOUNDS. The stale-entry check in
// field-keys.test.js can only fire once a key becomes PUBLISHED BY OUR STORE —
// which happens when somebody adds a row to `telemetry-provenance.js`. So the
// trigger for noticing that the server grew a feature is... us noticing that
// the server grew a feature. The loop is closed on our own artefacts. Nothing
// reads the Rust.
//
// AND IT HAD ALREADY FAILED. Ten `kv.*` keys were allowlisted with the reason
// "block-table endpoint, not yet landed". The block-table endpoint HAD landed:
// `/v1/debug/kv/blocks` is a registered route (routes/mod.rs, routes/admin.rs),
// and the already-polled `/v1/debug/kv` advertises its own URL on the wire as
// `block_table_endpoint`. The panel rendered an em-dash over live data, which
// field-keys.test.js:200 itself names as "the worst failure available here: it
// looks correct, reports nothing, and understates a server that got better".
// Every test in this package was green throughout.
//
// SO THIS FILE READS THE PRODUCER. For every key still claimed unplumbed, the
// claim must name the WIRE NAMES the server would serve it under, and none of
// those names may appear anywhere in the server's route sources. If one does,
// the feature shipped and the claim is stale — red, by name, with the file.
//
// WHAT IT DELIBERATELY CANNOT DO. A name present in the Rust is not proof the
// field reaches the wire on THIS build, and a name absent is not proof no
// other spelling serves it. Both directions fail LOUD (a false red costs a
// reviewer five minutes and a corrected `absentWireNames`), never silent,
// which is the only trade this package accepts.

import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';

import { NOT_YET_PUBLISHED } from './dashboard/field-keys.test.js';
import { PROVENANCE } from './telemetry-provenance.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROUTES_DIR = join(HERE, '..', '..', 'crates', 'onnx-genai-server', 'src', 'routes');
const METRICS_FILE = join(HERE, '..', '..', 'crates', 'onnx-genai-server', 'src', 'metrics.rs');

/**
 * Wire names that DO exist, used to prove this scanner can return "present".
 *
 * Without this, a scanner whose corpus failed to load would report every claim
 * of absence as confirmed — the guard would be at its greenest exactly when it
 * had read nothing. A zero is not a measurement until the instrument is proven
 * able to return non-zero, and these are the proof.
 */
const PRESENT_CONTROLS = Object.freeze([
  'pages_in_use',
  'block_table_endpoint',
  'active_batch_size',
  'onnx_genai_requests_waiting',
]);

/**
 * A name no server has ever served, used to prove the scanner can return
 * "absent" rather than matching everything it is handed.
 */
const ABSENT_CONTROL = 'kv_flux_capacitor_discharge_total';

/**
 * A name that appears ONLY inside a Rust comment, never in code.
 *
 * `uptime` occurs once in the whole crate's routes: in a doc comment at
 * routes/admin.rs:169 explaining why a rate is NOT derived from it. This is a
 * regression control for the comment-stripping in `serverSources()` — without
 * it, prose explaining an absence is scored as evidence of a presence, and
 * this guard reports a landed feature that never landed. It did exactly that
 * on its first run.
 */
const COMMENT_ONLY_CONTROL = 'uptime';

/**
 * Every key still claimed unplumbed, with the names the server WOULD serve it
 * under and the evidence that it does not.
 *
 * `absentWireNames` is the whole mechanism. A free-text reason cannot be
 * checked against anything; a wire name can. Entries must name the spelling a
 * server would ACTUALLY use — not the dashboard's key — because the dashboard
 * key is our vocabulary and proves nothing about theirs.
 *
 * @type {Readonly<Record<string, {reason: string, absentWireNames: readonly string[]}>>}
 */
const UNPLUMBED_CLAIMS = Object.freeze({
  // ── Paged-KV lifetime counters the block table does NOT carry ──────────
  //
  // These three are the residue after `/v1/debug/kv/blocks` was wired up.
  // BlockTableResponse serves pressure signals (`hot_evictions`,
  // `allocation_failures`) but keeps no cumulative alloc/free ledger and no
  // separate prefix-eviction counter, so these remain genuinely absent.
  'kv.allocations': {
    reason:
      'The pool keeps no cumulative allocation ledger. BlockTableResponse serves occupancy ' +
      '(pages_in_use) and pressure (hot_evictions, allocation_failures), never a lifetime ' +
      'allocation count.',
    absentWireNames: ['allocations_total', 'total_allocations'],
  },
  'kv.frees': {
    reason:
      'The counterpart to kv.allocations, and absent for the same reason: a released page ' +
      'reports ref_count 0, which is an occupancy fact, not a free event anyone counts.',
    absentWireNames: ['frees', 'frees_total', 'total_frees'],
  },
  'kv.prefix_evictions': {
    reason:
      'Evictions are counted by TIER, not by cause. `hot_evictions` is the only eviction ' +
      'counter on the wire, and it does not distinguish a prefix-driven eviction from any ' +
      'other. Splitting it here would invent a breakdown the server never computed.',
    absentWireNames: ['prefix_evictions', 'prefix_evictions_total'],
  },

  // ── Scheduler detail beyond what /v1/debug/kv serves ────────────────────
  'queue.depth_peak': {
    reason:
      'The server reports instantaneous depth (pending_queue_depth, queue_depth) and keeps ' +
      'no high-water mark. A peak computed client-side would be the peak SINCE THIS TAB ' +
      'OPENED, which is a different quantity wearing the same label.',
    absentWireNames: ['depth_peak', 'peak_queue_depth', 'queue_depth_peak', 'high_water'],
  },
  'scheduler.preemptions_total': {
    reason:
      'The driver runs each generation to completion without preemption, so the event this ' +
      'counter would count cannot occur. Consistent with `paused_sessions: None`, which the ' +
      'server itself registers as not-applicable rather than unavailable.',
    absentWireNames: ['preemptions', 'preemptions_total', 'preempted'],
  },

  // ── Prefix-cache savings ────────────────────────────────────────────────
  'prefix_cache.evictions': {
    reason: 'No prefix-cache eviction counter exists in metrics.rs or any route.',
    absentWireNames: ['prefix_cache_evictions', 'prefix_cache_evictions_total'],
  },
  'prefix_cache.prefill_tokens_skipped': {
    reason:
      'The server counts tokens MATCHED (prefix_tokens_reused / prefix_cache_hit_tokens_total), ' +
      'not tokens whose prefill was actually skipped. On the token-prefix branch a counted ' +
      'reuse skips no prefill at all (engine/runtime.rs computes an LCP without retaining a ' +
      'page), so binding matched tokens to this key would overstate the saving.',
    absentWireNames: ['prefill_tokens_skipped', 'tokens_skipped', 'prefill_skipped'],
  },
  'prefix_cache.time_saved_ms': {
    reason:
      'No timer measures prefill time avoided. It could only be inferred from skipped tokens, ' +
      'which are themselves not measured — an estimate stacked on an estimate.',
    absentWireNames: ['time_saved', 'time_saved_ms', 'prefill_time_saved'],
  },
  'prefix_cache.tokens_reused': {
    reason:
      'The count IS on the wire, under `prefix_tokens_reused`, and is deliberately NOT bound: ' +
      'prefix-counters-forbidden.test.js and the NEVER_BIND register ban this family because ' +
      'the numerator scores one for any nonzero match. This is a BAN, not a gap — the only ' +
      'entry here whose absence is our choice rather than the server\'s.',
    // Named as a ban rather than an absence, so no wire name is claimed absent.
    // A claim we know to be false must not be written down as true.
    absentWireNames: [],
  },

  // ── Latency percentiles ─────────────────────────────────────────────────
  //
  // THE WIRE CARRIES HISTOGRAMS, NOT PERCENTILES. metrics.rs emits
  // `_bucket{le=...}`, `_sum` and `_count` over 14 fixed LATENCY_BUCKETS_NS.
  // There is no `quantile=` label anywhere in the crate. A p95 interpolated
  // from 14 coarse buckets is an estimate with an error bar the width of a
  // bucket, and rendering it as "p95" is a fabricated measurement wearing a
  // plausible label — the exact defect telemetry-provenance.js exists to
  // prevent. The MEAN is genuinely derivable (_sum / _count) and IS bound, as
  // `metrics.ttft` and `metrics.e2e_latency`.
  //
  // The client rows are a different absence again and are handled in the
  // provenance table as STRUCTURALLY_BYPASSED, not here: they need a scenario
  // runner in the browser, and no server change could ever supply them.
  'latency.ttft_server_p50': {
    reason:
      'The server exports a bucketed histogram, not percentiles: metrics.rs histogram() ' +
      'writes _bucket{le=}/_sum/_count over 14 fixed bounds and no quantile label exists ' +
      'in the crate. The mean is derivable and is bound as metrics.ttft / ' +
      'metrics.e2e_latency; a percentile is not.',
    absentWireNames: ['quantile'],
  },
  'latency.ttft_server_p95': {
    reason:
      'The server exports a bucketed histogram, not percentiles: metrics.rs histogram() ' +
      'writes _bucket{le=}/_sum/_count over 14 fixed bounds and no quantile label exists ' +
      'in the crate. The mean is derivable and is bound as metrics.ttft / ' +
      'metrics.e2e_latency; a percentile is not.',
    absentWireNames: ['quantile'],
  },
  'latency.ttft_server_max': {
    reason:
      'The server exports a bucketed histogram, not percentiles: metrics.rs histogram() ' +
      'writes _bucket{le=}/_sum/_count over 14 fixed bounds and no quantile label exists ' +
      'in the crate. The mean is derivable and is bound as metrics.ttft / ' +
      'metrics.e2e_latency; a percentile is not.',
    absentWireNames: ['quantile'],
  },
  'latency.e2e_server_p50': {
    reason:
      'The server exports a bucketed histogram, not percentiles: metrics.rs histogram() ' +
      'writes _bucket{le=}/_sum/_count over 14 fixed bounds and no quantile label exists ' +
      'in the crate. The mean is derivable and is bound as metrics.ttft / ' +
      'metrics.e2e_latency; a percentile is not.',
    absentWireNames: ['quantile'],
  },
  'latency.e2e_server_p95': {
    reason:
      'The server exports a bucketed histogram, not percentiles: metrics.rs histogram() ' +
      'writes _bucket{le=}/_sum/_count over 14 fixed bounds and no quantile label exists ' +
      'in the crate. The mean is derivable and is bound as metrics.ttft / ' +
      'metrics.e2e_latency; a percentile is not.',
    absentWireNames: ['quantile'],
  },
  'latency.e2e_server_max': {
    reason:
      'The server exports a bucketed histogram, not percentiles: metrics.rs histogram() ' +
      'writes _bucket{le=}/_sum/_count over 14 fixed bounds and no quantile label exists ' +
      'in the crate. The mean is derivable and is bound as metrics.ttft / ' +
      'metrics.e2e_latency; a percentile is not.',
    absentWireNames: ['quantile'],
  },

  // ── Build and host facts ────────────────────────────────────────────────
  'scenario.makespan_ms': {
    reason:
      'Supplied by the scenario runner in the browser, never by the server. Listed so the key ' +
      'is not mistaken for a typo; it is not a server gap and no server change would supply it.',
    absentWireNames: ['makespan', 'makespan_ms'],
  },
  'server.decode_backend': {
    reason:
      'The decode path is chosen at runtime from the model config and is not echoed by any ' +
      'route. /v1/debug/config serves `pipeline`, which names the model family, not the ' +
      'decode backend.',
    absentWireNames: ['decode_backend', 'decoder_backend'],
  },
  'server.quantization': {
    reason: 'No route reports the weight quantization of the loaded model.',
    absentWireNames: ['quantization', 'quant_type', 'weight_dtype'],
  },
  'server.uptime_ms': {
    reason:
      'No endpoint exposes uptime, start time or build id. The capture manifest records the ' +
      'same finding independently: "no endpoint exposes uptime, start time or build id, so ' +
      'nothing on the wire can date this capture".',
    absentWireNames: ['uptime', 'uptime_ms', 'started_at', 'start_time'],
  },
  'server.version': {
    reason:
      'No binary carries its commit: no vergen, no env!("GIT_*"), no stamping build.rs and no ' +
      'version endpoint. Recorded independently in fixtures/captures/manifest.json.',
    absentWireNames: ['build_version', 'server_version', 'git_sha', 'commit_sha'],
  },
});

/**
 * Provenance rows that name NO endpoint at all, and the wire names that would
 * prove them wrong.
 *
 * WHY THESE NEED THEIR OWN LIST. A row with `source: null` is making the
 * strongest claim in the catalogue -- NO SERVER SERVES THIS, AND NONE COULD --
 * and it is the one claim `matchesStub()` cannot evaluate, because there is no
 * path to read and therefore no observation that could ever contradict it.
 * telemetry-store.js's own guard ("every suppressed field can be checked
 * against the wire, or says why not") would otherwise have to treat these as
 * unfalsifiable, which is precisely the blind spot it exists to remove.
 *
 * So the falsifiability MOVES HERE rather than disappearing: the claim is
 * checked against the server SOURCES instead of against a response body. If
 * the server ever grows a client-latency percentile under any of these names,
 * this goes red and the row must be rewritten.
 *
 * @type {Readonly<Record<string, readonly string[]>>}
 */
const SOURCELESS_CLAIMS = Object.freeze({
  'latency.ttft_client_p50': ['ttft_client', 'client_latency', 'percentile'],
  'latency.ttft_client_p95': ['ttft_client', 'client_latency', 'percentile'],
  'latency.ttft_client_max': ['ttft_client', 'client_latency', 'percentile'],
  'latency.itl_client_p50': ['itl_client', 'inter_token_latency', 'percentile'],
  'latency.itl_client_p95': ['itl_client', 'inter_token_latency', 'percentile'],
  'latency.itl_client_max': ['itl_client', 'inter_token_latency', 'percentile'],
  'latency.tpot_client_p50': ['tpot_client', 'time_per_output_token', 'percentile'],
  'latency.tpot_client_p95': ['tpot_client', 'time_per_output_token', 'percentile'],
  'latency.tpot_client_max': ['tpot_client', 'time_per_output_token', 'percentile'],
});

/**
 * Every Rust source that can put a name on the wire, with comments removed.
 *
 * STRIPPING COMMENTS IS NOT COSMETIC. The first run of this guard reported
 * `server.uptime_ms` as a landed feature because `uptime` appears in a PROSE
 * SENTENCE at routes/admin.rs:169 — "Dividing it by uptime yields a lifetime
 * average". A doc comment EXPLAINING WHY A FIELD IS ABSENT was scored as
 * evidence that it is present, which is the precise inversion of its meaning.
 * `dashboard/field-keys.test.js` strips comments before scanning JS for the
 * same reason and in the same direction.
 */
function serverSources() {
  const sources = [['metrics.rs', readFileSync(METRICS_FILE, 'utf8')]];
  for (const name of readdirSync(ROUTES_DIR)) {
    if (!name.endsWith('.rs')) continue;
    sources.push([`routes/${name}`, readFileSync(join(ROUTES_DIR, name), 'utf8')]);
  }
  return sources.map(([name, source]) => [name, stripRustComments(source)]);
}

/**
 * Remove `//`, `///` and `/* *\/` comments from Rust source.
 *
 * String literals are NOT protected, deliberately: a `//` inside a string is
 * vanishingly rare in these files, and the failure direction if it happened
 * would be to strip too much — a MISSED landing, which this guard's own
 * `PRESENT_CONTROLS` would catch the moment it touched a control name.
 *
 * @param {string} source
 */
function stripRustComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1');
}

/**
 * Which server sources mention a wire name, as a whole word.
 *
 * Whole-word matching on purpose: a substring test would score `frees` as
 * present inside `frees_total` (fine) but also inside unrelated identifiers,
 * and a guard that reddens on noise is a guard somebody switches off.
 *
 * @param {string} wireName
 * @returns {string[]} Names of the files that mention it.
 */
function sourcesMentioning(wireName) {
  const pattern = new RegExp(`\\b${wireName}\\b`);
  return serverSources()
    .filter(([, source]) => pattern.test(source))
    .map(([name]) => name);
}

describe('the unplumbed-claims scanner can see what it claims to scan', () => {
  it('reads a non-empty corpus of server sources', () => {
    const sources = serverSources();
    assert.ok(
      sources.length >= 3,
      `read only ${sources.length} server sources from ${ROUTES_DIR} — every claim of ` +
        'absence below is now being confirmed against an empty corpus, which is the one ' +
        'way this guard can be green and worthless',
    );
    for (const [name, source] of sources) {
      assert.ok(source.length > 0, `${name} is empty`);
    }
  });

  it('finds names that ARE served — proving it can return "present"', () => {
    // The positive control. If the scanner cannot find a name we know the
    // server serves, it cannot find a name the server GREW either, and every
    // stale claim below passes for the wrong reason.
    const missed = PRESENT_CONTROLS.filter((name) => sourcesMentioning(name).length === 0);
    assert.deepEqual(
      missed,
      [],
      `the scanner could not find ${missed.join(', ')} in the server sources, but these are ` +
        'known to be served. The scanner is broken or the corpus moved; either way every ' +
        '"still absent" verdict below is unfounded.',
    );
  });

  it('reports a fabricated name as absent — proving it can return "absent"', () => {
    // The negative control, against the opposite failure: a scanner that
    // matched everything would redden honestly-absent claims and teach the
    // team to delete the guard.
    assert.deepEqual(
      sourcesMentioning(ABSENT_CONTROL),
      [],
      `the scanner found "${ABSENT_CONTROL}" in the server sources. No server serves it, so ` +
        'the matcher is over-matching and every red below is noise.',
    );
  });

  it('does not read prose as wire evidence', () => {
    // Regression control. Comments are the one place a name appears BECAUSE it
    // is absent, so counting them inverts the reading of every doc comment
    // that explains a gap.
    const raw = readFileSync(join(ROUTES_DIR, 'admin.rs'), 'utf8');
    assert.ok(
      new RegExp(`\\b${COMMENT_ONLY_CONTROL}\\b`).test(raw),
      `"${COMMENT_ONLY_CONTROL}" no longer appears in routes/admin.rs at all, so this control ` +
        'proves nothing. Pick another word that occurs only inside a comment.',
    );
    assert.deepEqual(
      sourcesMentioning(COMMENT_ONLY_CONTROL),
      [],
      `the scanner found "${COMMENT_ONLY_CONTROL}" in the server sources, but it occurs only ` +
        'inside a comment explaining why the value is NOT derived. Comment stripping in ' +
        'serverSources() has regressed, and prose about an absence is now being scored as ' +
        'evidence of a presence.',
    );
  });
});

describe('every claim of absence is evidenced and still true', () => {
  it('covers exactly the allowlist, with no drift in either direction', () => {
    // Keyset equality rather than a subset test. A claim here for a key nobody
    // allowlists is dead evidence that looks like diligence; an allowlisted key
    // with no claim here is the bare unevidenced sentence this file exists to
    // abolish, and a subset test in either direction would permit one of them.
    const claimed = Object.keys(UNPLUMBED_CLAIMS).sort();
    const allowlisted = Object.keys(NOT_YET_PUBLISHED).sort();
    assert.deepEqual(
      claimed,
      allowlisted,
      'UNPLUMBED_CLAIMS and NOT_YET_PUBLISHED have drifted apart.\n' +
        `  claimed here but not allowlisted: ${claimed.filter((k) => !allowlisted.includes(k)).join(', ') || '(none)'}\n` +
        `  allowlisted but unevidenced here: ${allowlisted.filter((k) => !claimed.includes(k)).join(', ') || '(none)'}\n` +
        'Every key deferred as "not yet plumbed" must name the wire names it is absent under, ' +
        'so the claim can be re-evaluated against the server instead of trusted forever.',
    );
  });

  it('states a non-trivial reason for every claim', () => {
    const thin = Object.entries(UNPLUMBED_CLAIMS)
      .filter(([, claim]) => !claim.reason || claim.reason.length < 40)
      .map(([key]) => key);
    assert.deepEqual(thin, [], `${thin.join(', ')} have no substantive reason.`);
  });

  it('has no stale claim — a name the server now serves means the feature landed', () => {
    // THE CHECK. Everything above exists to make this one trustworthy.
    const stale = [];
    for (const [key, claim] of Object.entries(UNPLUMBED_CLAIMS)) {
      for (const wireName of claim.absentWireNames) {
        const found = sourcesMentioning(wireName);
        if (found.length > 0) {
          stale.push(`"${key}" is claimed unplumbed, but "${wireName}" is served by ${found.join(', ')}`);
        }
      }
    }

    assert.deepEqual(
      stale,
      [],
      `${stale.join('\n')}\n\nThe server grew this field and the dashboard is still rendering ` +
        'an em-dash over it. That failure looks exactly like caution, nobody reports it, and ' +
        'it understates a server that got better. Bind the field in telemetry-provenance.js ' +
        'and delete the key from NOT_YET_PUBLISHED and from UNPLUMBED_CLAIMS.',
    );
  });
});

describe('a row that names no endpoint is still held against the server', () => {
  it('covers every sourceless provenance row', () => {
    // Derived from PROVENANCE, never from SOURCELESS_CLAIMS. A list checked
    // against its own definition is a mirror: adding a tenth sourceless row
    // would extend the catalogue while this guard kept reporting nine covered
    // out of nine.
    const sourceless = Object.entries(PROVENANCE)
      .filter(([, entry]) => entry.source === null || entry.source === undefined)
      .map(([key]) => key)
      .sort();
    const covered = Object.keys(SOURCELESS_CLAIMS).sort();
    assert.deepEqual(
      sourceless,
      covered,
      'A provenance row with no `source` claims that NO endpoint serves it and none could. ' +
        'That is the only claim in the catalogue no response body can refute, so it must be ' +
        'refutable HERE instead — name the wire names that would prove it wrong in ' +
        `SOURCELESS_CLAIMS.\n  sourceless rows: ${sourceless.join(', ') || '(none)'}\n` +
        `  covered here:    ${covered.join(', ') || '(none)'}`,
    );
  });

  it('finds none of those names in the server sources', () => {
    const landed = [];
    for (const [key, wireNames] of Object.entries(SOURCELESS_CLAIMS)) {
      for (const wireName of wireNames) {
        const found = sourcesMentioning(wireName);
        if (found.length === 0) continue;
        landed.push(
          `"${key}" claims no server could serve it, but "${wireName}" is in ${found.join(', ')}`,
        );
      }
    }
    assert.deepEqual(
      landed,
      [],
      `${landed.join('\n')}\n\nThese rows render "not-applicable" — a claim that the question ` +
        'is not the server\'s to answer. If the server started answering it, that claim became ' +
        'false and the row must be reclassified, not left telling visitors the number cannot ' +
        'exist.',
    );
  });
});
