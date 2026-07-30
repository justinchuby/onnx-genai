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
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';

import { CLASS, CLASSES_REQUIRING_EVIDENCE, UNPLUMBED } from './unplumbed-registry.mjs';
import { REPO_ROOT, SHIPPING_REF, shipped, shippedPaths } from './shipping-tree.mjs';
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
 * Every key still claimed unplumbed. IMPORTED, not declared here.
 *
 * This was a second hand-written copy of the deferral inventory: same keys as
 * dashboard/field-keys.test.js's allowlist, but a separate, longer, separately
 * worded reason for each. Keyset equality was asserted, so the KEYS could not
 * drift -- but nothing compared the two reason texts, so one could be corrected
 * and the other left stale while every suite stayed green.
 *
 * Both now derive from ../unplumbed-registry.mjs, which also carries the class
 * of absence and structured evidence. See the single-source guard below.
 */
const UNPLUMBED_CLAIMS = UNPLUMBED;

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
  it('the inventory has exactly one definition — no second copy anywhere', () => {
    // WHAT THIS REPLACED, AND WHY THE OLD TEST HAD TO GO.
    //
    // This was a keyset-equality assertion between UNPLUMBED_CLAIMS and
    // NOT_YET_PUBLISHED, back when both were hand-written literals. It was a
    // real guard then. It is VACUOUS now: both derive from the same registry,
    // so it compares a list with itself and can never fail.
    //
    // Leaving it would have been worse than deleting it -- a green that reads
    // like drift protection while proving nothing is exactly the false comfort
    // this file exists to abolish. So it is replaced by the invariant that
    // actually matters once the copies are collapsed: that no SECOND copy
    // comes back.
    //
    // Both names are still asserted to agree, but the point is now structural:
    // the only way to reintroduce drift is to redeclare one of them as its own
    // literal, and that is what this scans for.
    // The keyset comparison this replaced imported NOT_YET_PUBLISHED out of
    // dashboard/field-keys.test.js. That import is now DELETED, for the same
    // reason the readFileSync lift in check-binding-liveness was deleted:
    // pulling one export out of a `.test.js` runs that file's whole suite
    // inside this run, inflating the total by 13 and reporting its failures
    // under this file's name. Fixing that in one consumer and leaving it in
    // the other would have been a half-migration.
    //
    // Nothing is lost. The invariant it enforced -- that field-keys does not
    // keep its own copy -- is enforced BELOW, structurally, against the
    // shipped source rather than against a runtime value.
    const REGISTRY = 'unplumbed-registry.mjs';
    const INVENTORY_NAMES = ['NOT_YET_PUBLISHED', 'UNPLUMBED_CLAIMS', 'UNPLUMBED'];

    // A declaration BOUND TO AN OBJECT LITERAL is a second inventory. Binding
    // to a call or an identifier (`= summaryAllowlist();`, `= UNPLUMBED;`) is a
    // derivation and is exactly what we want.
    const inventoryLiteralsIn = (source) =>
      INVENTORY_NAMES.filter((name) =>
        new RegExp(
          `(?:const|let|var|export const)\\s+${name}\\s*=\\s*(?:Object\\.freeze\\(\\s*)?\\{`,
        ).test(source),
      );

    const registryPath = shippedPaths().find((p) => p.endsWith(REGISTRY));
    assert.ok(registryPath, `${REGISTRY} is not at the shipping ref at all.`);

    // ANTI-VACUITY, both directions, and BOTH CONTROLS ARE REAL SHIPPED FILES
    // rather than fixtures.
    //
    // The corpus scan below passes if `inventoryLiteralsIn` never matches
    // anything -- a broken regex, a reformatted declaration style, an empty
    // corpus -- and it would then report a confident green having read nothing.
    // Zero redeclarations is also the permanent goal state, so there is no real
    // positive occurrence left to anchor on. That is the trap I walked into on
    // the by-name guard: an anti-vacuity control anchored on a real DEFECT
    // breaks the moment the defect is legitimately repaired.
    //
    // The way out here is that ONE legitimate occurrence must exist forever, by
    // definition: the registry itself declares the literal. So the detector is
    // proved against the file it deliberately skips. That control cannot rot,
    // because the day it stops holding, the single definition is gone.
    //
    // Using real files also sidesteps the second trap: a synthetic fixture
    // containing an inventory literal would sit in THIS file's source, which is
    // itself in the scanned corpus, and the guard would flag its own sample data.
    assert.deepEqual(
      inventoryLiteralsIn(stripRustComments(shipped(registryPath))),
      ['UNPLUMBED'],
      `The detector could not find the inventory literal in ${REGISTRY}, which is ` +
        'the one file that definitely contains it. The scan below is therefore ' +
        'reading a corpus it cannot interpret, and its empty result means nothing.',
    );

    // NEGATIVE CONTROL: this file derives (`const UNPLUMBED_CLAIMS = UNPLUMBED;`)
    // and must NOT be read as a declaration. A detector that cannot tell a
    // derivation from a definition would flag every correct consumer, and a
    // guard that reddens on correct code gets deleted by whoever hits it.
    assert.deepEqual(
      inventoryLiteralsIn('const UNPLUMBED_CLAIMS = UNPLUMBED;\nexport const NOT_YET_PUBLISHED = summaryAllowlist();'),
      [],
      'The detector treats a derivation as a second inventory, so every correctly ' +
        'migrated consumer would be reported as a duplicate.',
    );

    const redeclared = [];

    for (const path of shippedPaths()) {
      if (!/\.(?:js|mjs)$/.test(path)) continue;
      if (path === registryPath) continue;
      for (const name of inventoryLiteralsIn(stripRustComments(shipped(path)))) {
        redeclared.push(`${path} declares ${name} as its own object literal`);
      }
    }

    assert.deepEqual(
      redeclared,
      [],
      'The deferral inventory has been copied again:\n' +
        `${redeclared.map((r) => `  ${r}`).join('\n')}\n\n` +
        `There must be exactly one definition, in ${REGISTRY}. Every other site ` +
        'derives from it. Two copies of one fact is how this inventory spent the ' +
        'evening carrying two independently worded reasons per key with nothing ' +
        'on earth reconciling them -- keyset drift was guarded, wording drift was ' +
        'not, and a corrected reason beside a stale one is invisible.',
    );

  });

  it('every entry is classified, and the taxonomy is not decorative', () => {
    const valid = new Set(Object.values(CLASS));
    const unclassified = Object.entries(UNPLUMBED_CLAIMS)
      .filter(([, e]) => !valid.has(e.class))
      .map(([key, e]) => `${key} has class ${JSON.stringify(e.class)}`);

    assert.deepEqual(
      unclassified,
      [],
      `Entries carry a class that is not in CLASS:\n${unclassified.join('\n')}\n\n` +
        '"Not published" is not one situation. Flattening the classes is how six ' +
        'client-measured rows once sat among server gaps, promising a server ' +
        'change that could not possibly have delivered them.',
    );

    // ANTI-DECORATION. A taxonomy where everything lands in one bucket is a
    // constant wearing a type's clothing. Deliberately NOT "every class is
    // used" -- retiring the last DELIBERATE_BAN is a REPAIR and must not go red.
    const inUse = new Set(Object.values(UNPLUMBED_CLAIMS).map((e) => e.class));
    assert.ok(
      inUse.size >= 3,
      `All ${Object.keys(UNPLUMBED_CLAIMS).length} entries fall into only ` +
        `${inUse.size} class(es): ${[...inUse].join(', ')}. Either the taxonomy ` +
        'stopped being applied, or it is no longer earning its complexity.',
    );
  });

  it('a claim about something PRESENT cites where it is present', () => {
    // WHY ONLY SOME CLASSES NEED EVIDENCE.
    //
    // SERVER_GAP claims a name is ABSENT, and `absentWireNames` falsifies that
    // mechanically -- the scan below reads the Rust and goes red when the name
    // appears. That is the strongest evidence available and needs no citation.
    //
    // SHAPE_MISMATCH, DELIBERATE_BAN and EVENT_CANNOT_OCCUR are the opposite
    // shape: they concede the thing EXISTS and argue it cannot or must not be
    // used. A name scan cannot falsify those -- of course the name is there,
    // that is the premise -- so the falsifiability has to MOVE to a citation.
    // Without this rule they would be the one class of claim in this file that
    // nothing can ever check, which is precisely where a wrong claim would hide.
    const missing = [];
    const unresolved = [];
    const tracked = new Set(
      execFileSync('git', ['ls-tree', '-r', '--name-only', SHIPPING_REF], {
        cwd: REPO_ROOT,
        maxBuffer: 16 * 1024 * 1024,
      })
        .toString()
        .split('\n')
        .filter(Boolean),
    );

    for (const [key, entry] of Object.entries(UNPLUMBED_CLAIMS)) {
      if (!CLASSES_REQUIRING_EVIDENCE.includes(entry.class)) continue;

      if (entry.evidence.length === 0) {
        missing.push(`${key} (${entry.class}) cites nothing`);
        continue;
      }

      for (const citation of entry.evidence) {
        if (!tracked.has(citation.file)) {
          unresolved.push(`${key}: ${citation.file} is not tracked at the shipping ref`);
          continue;
        }
        const body = readFileSync(join(REPO_ROOT, citation.file), 'utf8');
        if (!body.includes(citation.symbol)) {
          unresolved.push(`${key}: ${citation.file} no longer contains '${citation.symbol}'`);
        }
      }
    }

    assert.deepEqual(
      missing,
      [],
      `These entries claim something exists but is unusable, and cite nothing:\n` +
        `${missing.map((m) => `  ${m}`).join('\n')}\n\n` +
        'absentWireNames cannot falsify a claim about a field that IS present. ' +
        'Cite the file and symbol, or reclassify.',
    );

    assert.deepEqual(
      unresolved,
      [],
      `Evidence citations no longer resolve:\n${unresolved.map((u) => `  ${u}`).join('\n')}\n\n` +
        'The server moved and the claim did not. Re-read the cited source and ' +
        'either update the citation or retire the entry -- a citation that points ' +
        'nowhere is how a claim outlives the thing that justified it.',
    );

    // ANTI-VACUITY: the rule must actually be exercised by real entries.
    const audited = Object.values(UNPLUMBED_CLAIMS).filter((e) =>
      CLASSES_REQUIRING_EVIDENCE.includes(e.class),
    );
    assert.ok(
      audited.length >= 3,
      `Only ${audited.length} entries require evidence, so this check is nearly ` +
        'vacuous. It audited 8 when written.',
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
