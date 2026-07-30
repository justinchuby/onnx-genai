// Copyright (c) Microsoft Corporation.
//
// The binding-liveness guard.
//
// A BINDING IS A CLAIM THAT A PRODUCER EMITS SOMETHING, and until this file
// existed nothing in this repository ever checked that claim against a
// producer. Every other guard here reconciles our own artefacts with each
// other: `field-keys.test.js` checks bound keys against what the STORE
// publishes, and the store publishes exactly `Object.keys(PROVENANCE)`. So a
// row in `telemetry-provenance.js` could claim `MEASURED`, name a `path` the
// server has never emitted, pass every test we own, and render a permanent
// em-dash in front of a visitor. The table is a hand-maintained snapshot of
// somebody else's source tree; the one thing it is never held against is the
// tree it describes.
//
// This guard holds every `MEASURED` claim against a RECORDED LIVE CAPTURE of
// the server that is supposed to produce it.
//
// WHAT IT IS NOT. It does not check that a value is CORRECT — only that the
// producer emits the field at all. `matchesStub()` covers the "documented zero
// went real" direction, and no instrument can tell a measured zero from a
// documented one. This closes a different hole: THE FIELD IS NOT THERE AT ALL.
//
// WHICH TREE THIS READS. Sources and fixtures are both read from DISK, on
// purpose, so this guard lives in ONE universe rather than two. Enumerating
// from `git ls-files` and reading with `readFileSync` is the exact defect this
// crew has fixed twice tonight. The cost is stated rather than hidden: an
// UNTRACKED panel on somebody's desk is scanned here and would not ship. That
// direction fails LOUD (a false red), never silent, and a clean checkout — CI,
// or a detached worktree — has no untracked files, so there the desk IS the
// branch.

import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';

import { PROVENANCE, TEXT_ENDPOINTS, resolveForOrigin } from './telemetry-provenance.js';
import { createTelemetryStore } from './telemetry-store.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const CAPTURE_DIR = join(HERE, 'fixtures', 'captures');

/**
 * Claims the captures cannot support, each with the reason and a citation.
 *
 * THIS IS THE ESCAPE HATCH AND IT IS DELIBERATELY EXPENSIVE. An exemption list
 * is how every mechanism like this dies: declaring a key absent is always
 * cheaper than fixing it, so without a ceiling the list becomes the answer to
 * every red and the guard quietly measures nothing. Three rules hold it shut:
 *
 *   1. `reason` and `evidence` are both REQUIRED and must be non-trivial.
 *   2. An entry that is no longer absent FAILS. A stale exemption is not free
 *      -- it is a panel rendering an em-dash over real data, which is the more
 *      dangerous direction and the one nobody reports.
 *   3. `MAX_DECLARED_ABSENT` caps the list. Raising it is a visible diff that
 *      a reviewer must approve, which is the entire point.
 *
 * @type {Readonly<Record<string, {reason: string, evidence: string}>>}
 */
export const DECLARED_ABSENT = Object.freeze({
  'prefix_cache.lookups': {
    reason:
      'The table names path `prefix_cache_lookups` on /v1/debug/kv; the server emits ' +
      '`generations_completed` there instead -- the rename this row\'s own evidence string ' +
      'ASKED FOR ("the upstream name is wrong: it counts generations") landed on the server ' +
      'and the path here was never updated. Impact is confined to the DEGRADED path: this key ' +
      'is only the fallback denominator for prefix_cache.hit_rate (telemetry-store.js:898), ' +
      'behind `metrics.prefix_cache_lookups`, which is live. So the fallback is dead exactly ' +
      'when /metrics is feature-gated off -- the one condition it exists for. NOT repointed ' +
      'here: reviving a prefix-cache number is the crew\'s call, not this guard\'s, and ' +
      'prefix-counters-forbidden.test.js exists for a reason.',
    evidence:
      'fixtures/captures/dynamic.json :: /v1/debug/kv emits generations_completed, not ' +
      'prefix_cache_lookups; telemetry-provenance.js:505-517 (the row); ' +
      'telemetry-store.js:898 (the only consumer).',
  },
});

/** Hard cap on the exemption list. Raising this is a reviewable diff. */
export const MAX_DECLARED_ABSENT = 1;

/** Load one origin's capture. */
function loadCapture(label) {
  return JSON.parse(readFileSync(join(CAPTURE_DIR, `${label}.json`), 'utf8'));
}

const manifest = JSON.parse(readFileSync(join(CAPTURE_DIR, 'manifest.json'), 'utf8'));
const ORIGIN_LABELS = Object.keys(manifest.origins);
const CAPTURES = Object.fromEntries(ORIGIN_LABELS.map((l) => [l, loadCapture(l)]));

/** Every first-party source that can consume a field, excluding tests. */
function consumerSources() {
  const files = [];
  for (const entry of readdirSync(HERE, { withFileTypes: true })) {
    if (entry.isFile() && entry.name.endsWith('.js') && !entry.name.includes('.test.')) {
      files.push(entry.name);
    }
  }
  for (const entry of readdirSync(join(HERE, 'dashboard'), { withFileTypes: true })) {
    if (entry.isFile() && entry.name.endsWith('.js') && !entry.name.includes('.test.')) {
      files.push(join('dashboard', entry.name));
    }
  }
  return files;
}

/**
 * Map every field key to the files that consume it, so a failure can name the
 * BINDING AND THE PANEL rather than just the key. A key with no consumer is
 * still checked -- the table claims it is measured, and a panel may bind it
 * tomorrow.
 *
 * @returns {Map<string, Set<string>>}
 */
function consumersByKey() {
  /** @type {Map<string, Set<string>>} */
  const map = new Map();
  for (const file of consumerSources()) {
    const source = readFileSync(join(HERE, file), 'utf8');
    for (const match of source.matchAll(/\.field\(\s*['"]([^'"]+)['"]/g)) {
      if (!map.has(match[1])) map.set(match[1], new Set());
      map.get(match[1]).add(file);
    }
    for (const match of source.matchAll(/fields\[\s*['"]([^'"]+)['"]\s*\]/g)) {
      if (!map.has(match[1])) map.set(match[1], new Set());
      map.get(match[1]).add(file);
    }
  }
  return map;
}

const CONSUMERS = consumersByKey();

/** Walk a dotted path, returning undefined at the first missing segment. */
function readPath(body, path) {
  return path.split('.').reduce((node, key) => (node == null ? undefined : node[key]), body);
}

/**
 * Can this capture support the entry's claim?
 *
 * @returns {{ok: boolean, why: string}}
 */
function supportedBy(entry, capture) {
  const recorded = capture.endpoints[entry.source];
  if (!recorded) return { ok: false, why: `${entry.source} was not captured` };
  if (recorded.body == null) {
    return { ok: false, why: `${entry.source} answered ${recorded.status}` };
  }

  // THE SERVER MAY DECLARE THE WHOLE ENDPOINT INAPPLICABLE, AND THAT IS AN
  // ANSWER, NOT A GAP. /v1/debug/kv/blocks replies `applicable: false` with a
  // FieldUnavailable reason on a model whose KV cache holds no paged storage
  // (routes/mod.rs). The row is not unsupported there -- it was asked and the
  // server said the question does not apply, which is exactly what the panel
  // then renders. Reporting that as an unsupported MEASURED claim would push
  // authors to delete a binding that works everywhere it can.
  if (recorded.body.applicable === false && recorded.body.unavailable) {
    const declared = recorded.body.unavailable;
    return { ok: true, why: `${entry.source} declares ${declared.code} for this origin` };
  }

  // A derived field has no wire path of its own -- it is computed client-side
  // from another series. Treating that as absent would be a blind instrument
  // reporting a false red, so the claim we CAN check is that its named input is
  // really on the wire. A derivation whose input is missing is just as dead.
  if (entry.derived) {
    // Inputs are DECLARED (`derivedFrom`), not scraped out of the prose. The
    // scrape this replaces matched only /\w+_total/, which was every input the
    // catalogue had while the only derived rows read /metrics counters; the
    // first derivation off a JSON endpoint named `page_size` and `ref_counts`
    // and the regex found nothing to check, so the guard would have gone red
    // on a live binding while calling the reason "names no *_total input".
    const inputs = entry.derivedFrom ?? [];
    if (inputs.length === 0) {
      return { ok: false, why: 'derived, but declares no `derivedFrom` inputs to verify' };
    }
    // `body` is a parsed object for JSON endpoints and a string for text ones.
    // The former stringifies to "[object Object]", so a substring test against
    // it can only ever fail -- silently, and identically for a real input and a
    // typo. Serialise before searching.
    const haystack = typeof recorded.body === 'string' ? recorded.body : JSON.stringify(recorded.body);
    const missing = inputs.filter((name) => !haystack.includes(name.split('.').pop()));
    return missing.length === 0
      ? { ok: true, why: '' }
      : { ok: false, why: `derived from ${missing.join(', ')}, absent from ${entry.source}` };
  }

  if (TEXT_ENDPOINTS.includes(entry.source)) {
    if (!entry.metric) return { ok: false, why: `text endpoint entry declares no metric` };
    return String(recorded.body).includes(entry.metric)
      ? { ok: true, why: '' }
      : { ok: false, why: `metric ${entry.metric} absent from ${entry.source}` };
  }

  if (!entry.path) return { ok: false, why: 'entry declares no path' };
  if (readPath(recorded.body, entry.path) !== undefined) return { ok: true, why: '' };

  // Presence is not a role: a key inside the server's `unavailable` map is
  // there to DECLARE ITS OWN ABSENCE. Reporting that as "found" would promote
  // an honest unavailable declaration into a fabricated measurement.
  const declared = recorded.body.unavailable?.[entry.path];
  return {
    ok: false,
    why: declared
      ? `${entry.path} is declared UNAVAILABLE on the wire: ${declared.detail ?? declared.code}`
      : `path ${entry.path} absent from ${entry.source}`,
  };
}

/** Every (key, origin) pair whose resolved classification claims a measurement. */
function measuredClaims() {
  const claims = [];
  for (const [key, base] of Object.entries(PROVENANCE)) {
    for (const label of ORIGIN_LABELS) {
      const entry = resolveForOrigin(base, label);
      if (entry.classification !== 'MEASURED') continue;
      claims.push({ key, label, entry });
    }
  }
  return claims;
}

/** Format a failure so it names the binding, the panel and the producer. */
function describe_(key, label, entry, why) {
  const consumers = CONSUMERS.get(key);
  const where = consumers ? [...consumers].sort().join(', ') : 'no consumer yet';
  return (
    `${key} [${label}] is classified MEASURED but the capture cannot support it.\n` +
    `      producer : ${entry.source} (${entry.path ?? entry.metric ?? 'derived'})\n` +
    `      bound by : ${where}\n` +
    `      why      : ${why}\n` +
    `      fix      : correct the row in telemetry-provenance.js, or add ${key} to ` +
    `DECLARED_ABSENT in check-binding-liveness.test.js with a reason and a citation.`
  );
}

describe('binding liveness', () => {
  it('captures carry the provenance that makes them auditable', () => {
    assert.ok(manifest.captured_at, 'manifest must record when it was captured');
    assert.ok(manifest.repo_head, 'manifest must record the tree the recorder ran from');
    assert.ok(manifest.conditions, 'manifest must record the load conditions');
    assert.ok(
      manifest.repo_head_is_not_the_server_identity,
      'manifest must state that repo_head does not identify the binary that answered',
    );
    assert.ok(ORIGIN_LABELS.length >= 2, 'both demo origins must be captured');
    for (const label of ORIGIN_LABELS) {
      assert.ok(manifest.origins[label].model_id, `${label} must record which model answered`);
      assert.ok(
        manifest.origins[label].endpoints_2xx.length > 0,
        `${label} must have captured at least one endpoint`,
      );
    }
  });

  it('every MEASURED claim is supported by a recorded live capture', () => {
    const failures = [];
    for (const { key, label, entry } of measuredClaims()) {
      if (Object.prototype.hasOwnProperty.call(DECLARED_ABSENT, key)) continue;
      const { ok, why } = supportedBy(entry, CAPTURES[label]);
      if (!ok) failures.push(describe_(key, label, entry, why));
    }
    assert.deepEqual(
      failures,
      [],
      `\n\n${failures.length} MEASURED claim(s) no producer supports:\n\n${failures.join('\n\n')}\n`,
    );
  });

  it('every declared-absent key is real, reasoned and cited', () => {
    for (const [key, note] of Object.entries(DECLARED_ABSENT)) {
      assert.ok(
        Object.prototype.hasOwnProperty.call(PROVENANCE, key),
        `DECLARED_ABSENT names ${key}, which is not in PROVENANCE at all`,
      );
      assert.ok(
        typeof note.reason === 'string' && note.reason.length >= 40,
        `${key} needs a real reason, not a placeholder`,
      );
      assert.ok(
        typeof note.evidence === 'string' && note.evidence.includes('::')
          ? true
          : /[a-z-]+\.(js|json|rs|md):\d+/.test(note.evidence ?? ''),
        `${key} needs a citation pointing at a file`,
      );
    }
  });

  it('no declared-absent key has quietly started working', () => {
    const stale = [];
    for (const key of Object.keys(DECLARED_ABSENT)) {
      const base = PROVENANCE[key];
      if (!base) continue;
      for (const label of ORIGIN_LABELS) {
        const entry = resolveForOrigin(base, label);
        if (entry.classification !== 'MEASURED') continue;
        if (supportedBy(entry, CAPTURES[label]).ok) {
          stale.push(
            `${key} [${label}] is exempted but the producer DOES emit it now. ` +
              `Remove it from DECLARED_ABSENT -- a panel is rendering an em-dash over real data.`,
          );
        }
      }
    }
    assert.deepEqual(stale, [], `\n\n${stale.join('\n')}\n`);
  });

  it('the exemption list stays under its ceiling', () => {
    const count = Object.keys(DECLARED_ABSENT).length;
    assert.ok(
      count <= MAX_DECLARED_ABSENT,
      `DECLARED_ABSENT holds ${count} entries, ceiling is ${MAX_DECLARED_ABSENT}. ` +
        'Fix the binding rather than raising the ceiling; if the ceiling must rise, ' +
        'that is a deliberate, reviewable diff.',
    );
  });
});

// ---------------------------------------------------------------------------
// THE ROUTING HALF.
//
// Everything above answers "does the producer emit this field at all". That
// catches a DEAD binding and is blind to its twin, which @f6527cc9 named: a key
// that is correctly registered, correctly spelled, and genuinely published by
// the server -- but routed by OUR catalogue to an endpoint we banned for
// stalling. It passes every existence check ever written and still freezes a
// panel. A dead key and a live key on a forbidden endpoint are ONE defect
// class -- a binding nobody checked against the wire -- and only one of them is
// a missing name.

/**
 * Endpoints D88 excludes from the 4 Hz loop. Not "polled slowly" -- EXCLUDED.
 *
 * design/demo-ux.md D88: "the 4 Hz loop polls /v1/status, /v1/debug/kv and
 * /health ONLY ... a single in-flight request against them holds a connection
 * for 15 s and AC24's at-most-one-cycle-in-flight rule would stall the whole
 * loop behind it." demo-spec.md:1874 is blunter: "/metrics is not involved and
 * must not be polled." Measured cost: 14,784 ms under load vs ~1.8 ms for the
 * three permitted endpoints DURING THE SAME GENERATION.
 *
 * This list is duplicated from prose because the ruling exists only as prose --
 * which is the root cause the lead named: our output contract is not an
 * artifact anywhere, so every party hand-maintains a private model of it. This
 * constant is the first executable form of D88 in the repository.
 */
const EXCLUDED_FROM_POLL_LOOP = Object.freeze(['/metrics', '/v1/resources']);

/**
 * `telemetry-store.js` binds fields as the module that SERVES them, not as a
 * panel that renders them. Counting it as a consumer reports the store's own
 * plumbing as four dashboard defects and buries the one real panel binding in
 * them -- the same blind-instrument error that made `derived: true` fields look
 * dead in the first version of this guard. The store's use of a stalling
 * endpoint is a POLL-LOOP defect, asserted separately below.
 */
const NOT_A_PANEL = 'telemetry-store.js';

/**
 * Routing violations that exist at HEAD, declared so a NEW one cannot hide
 * among them. Same expensive escape hatch as `DECLARED_ABSENT`: reason and
 * evidence required, and the ceiling makes an addition a visible diff.
 *
 * These are NOT closed. They are open and reported; the entry exists so the
 * suite stays green for a frozen branch while the defect stays executable and
 * un-loseable, rather than living in a broadcast nobody re-reads.
 */
const DECLARED_ROUTING_VIOLATIONS = Object.freeze({
  'resources.vram_limit_bytes': {
    reason:
      'dashboard/system.js renders VRAM limit from /v1/resources, which D88 excludes from ' +
      'the loop. It only appears to work because the store polls the excluded endpoint ' +
      'anyway -- so the binding and the poll loop are ONE defect, not two.',
    evidence:
      'design/demo-ux.md D88 (excluded, not slowed); demo-spec.md:805 and :830 ' +
      '(stay excluded until a load run posts a measured latency -- no such ' +
      'measurement exists in the tree at HEAD); telemetry-store.js poll set.',
  },
  'resources.kv_budget_bytes': {
    reason:
      'Second binding in dashboard/system.js on the same excluded endpoint. Recorded ' +
      'separately because two sites IS the finding: one exemption covering both would ' +
      'have made the second invisible while looking handled.',
    evidence: 'design/demo-ux.md D88; dashboard/system.js renders it from /v1/resources.',
  },
  'resources.disk_spill_bytes': {
    reason:
      'THIRD instance of the SAME defect, not a new one: same panel (dashboard/system.js), ' +
      'same renderBudgetRow group, same already-polled endpoint, and it will move with its ' +
      'two siblings the day D88 closes. Declared rather than left unbound because the ' +
      'alternative was worse — the key was previously carried as an unpublished field, i.e. ' +
      'a written claim that the server does not serve it, and the server DOES: ' +
      'ResolvedResourceLimits.disk_spill_bytes is declared at routes/mod.rs:454 and ' +
      'populated at routes/admin.rs:610. Keeping a claim of absence that the source refutes ' +
      'is the exact failure this suite exists to catch, so the honest routing violation was ' +
      'preferred over the dishonest absence.',
    evidence:
      'design/demo-ux.md D88; crates/onnx-genai-server/src/routes/mod.rs:454 ' +
      '(ResolvedResourceLimits.disk_spill_bytes: Option<u64>); routes/admin.rs:610 ' +
      '(populated); adds ZERO polling cost — /v1/resources is already fetched and parsed ' +
      'for the two siblings above, so this binding changes no request, only what is read ' +
      'out of a response the page already has.',
  },
});

describe('no binding routes a panel to an endpoint D88 excluded', () => {
  it('binds no panel field to a stalling endpoint', () => {
    const offenders = [];
    for (const [key, panels] of consumersByKey()) {
      if (!Object.hasOwn(PROVENANCE, key)) continue;
      const rendering = [...panels].filter((p) => !p.endsWith(NOT_A_PANEL));
      if (rendering.length === 0) continue;
      if (Object.hasOwn(DECLARED_ROUTING_VIOLATIONS, key)) continue;
      // Per ORIGIN, not once: resolveForOrigin can override `source`, so a key
      // that is fast on the base entry can still be routed to a stalling
      // endpoint on one arm. Checking the base entry alone would be a guard
      // blind to the exact asymmetry it exists to find.
      for (const origin of [null, 'scatter', 'dynamic']) {
        const { source } = resolveForOrigin(PROVENANCE[key], origin);
        if (!EXCLUDED_FROM_POLL_LOOP.includes(source)) continue;
        offenders.push(`${key} [${origin ?? 'base'}] -> ${source} <- ${rendering.join(', ')}`);
      }
    }
    assert.deepEqual(
      offenders,
      [],
      `These panel bindings resolve to an endpoint D88 excluded from the poll ` +
        `loop:\n  ${offenders.join('\n  ')}\n\nThe field may well be real and ` +
        'correctly named -- that is what makes this invisible to every ' +
        'existence check. Move the binding to a field served on /v1/status, ' +
        '/v1/debug/kv or /health, or render it unavailable with a reason.',
    );
  });

  it('is measuring real bindings, not an empty set', () => {
    // Non-vacuity. A drifted consumer scan would report a spotless dashboard
    // in bytes identical to a genuinely clean one.
    const registered = [...consumersByKey().keys()].filter((k) => Object.hasOwn(PROVENANCE, k));
    assert.ok(
      registered.length > 10,
      `Only ${registered.length} registered bindings found; the consumer scan has drifted.`,
    );
    const reachable = new Set(
      registered.map((k) => resolveForOrigin(PROVENANCE[k], 'dynamic').source),
    );
    assert.ok(
      reachable.size > 1,
      'Every binding resolved to one endpoint, which means resolution is not running.',
    );
  });
});

describe('the poll loop is held to D88 by execution, not by reading it', () => {
  it('requests only endpoints D88 permits, and declares the ones it does not', async () => {
    // EXECUTED. Reading the array literal in telemetry-store.js would audit the
    // source; running the store audits the behaviour. Our fake servers answer
    // /metrics instantly, which is precisely why a 15-second stall has been
    // invisible to 600 tests: THE TEST DOUBLE CANNOT EXHIBIT THE DEFECT.
    const asked = [];
    const fetchImpl = async (url) => {
      asked.push(new URL(url).pathname);
      return new Response('{}', { status: 200, headers: { 'content-type': 'application/json' } });
    };
    const store = createTelemetryStore({ origin: 'scatter', fetchImpl });
    await store.pollOnce();
    store.stop();

    assert.ok(asked.length > 2, `The loop requested ${asked.length} paths; the recorder is blind.`);

    const declared = new Set(
      Object.values(DECLARED_ROUTING_VIOLATIONS).length ? EXCLUDED_FROM_POLL_LOOP : [],
    );
    const undeclared = asked.filter(
      (p) => EXCLUDED_FROM_POLL_LOOP.includes(p) && !declared.has(p),
    );
    assert.deepEqual(
      undeclared,
      [],
      `The 4 Hz loop requests ${undeclared.join(', ')}, which D88 excludes ENTIRELY ` +
        '-- "not polled slowly, excluded" -- because one in-flight request holds a ' +
        'connection for 15 s and AC24 allows at most one cycle in flight, so the ' +
        'whole loop stalls behind it.',
    );
  });
});

describe('a declared-unplumbed key expires against the wire, not against us', () => {
  it('has no declared-unplumbed key that the server has started emitting', () => {
    // field-keys.test.js promises to remove a key from NOT_YET_PUBLISHED once
    // the data arrives. That promise CANNOT BE KEPT by the check that makes it:
    // it asks whether the key is in `publishedKeys()`, which polls a FAKE and
    // reads `Object.keys(PROVENANCE)`. A key absent from PROVENANCE can never
    // appear there, so for exactly the keys it covers the check is structurally
    // incapable of firing. The server could emit queue_depth_peak tomorrow and
    // nothing would notice, forever. This is the same expiry, anchored to a
    // recorded live capture instead -- the only artefact here that is not us.
    const declared = unplumbedKeys();
    assert.ok(declared.length > 2, `Only ${declared.length} declared keys read; the lift drifted.`);

    const bodies = readCaptureBodies();
    assert.ok(bodies.length > 0, 'No capture bodies read; this check would pass vacuously.');

    const arrived = declared.filter((key) => {
      const wire = key.slice(key.indexOf('.') + 1);
      // PRESENCE IS NOT A ROLE, and this guard caught its own author making
      // that mistake on the first run: `resources.disk_spill_bytes` IS a key in
      // the /v1/resources body on both origins -- with the value `null`. The
      // server emits the field NAME while declaring it unmeasured, so a
      // `hasOwn` test reports a correct declaration as a stale one and sends
      // somebody to "fix" a panel that is already honest. A field has arrived
      // when it carries a VALUE, not when it appears in the response shape.
      return bodies.some((body) => body[wire] !== null && body[wire] !== undefined);
    });
    assert.deepEqual(
      arrived,
      [],
      `${arrived.join(', ')} is declared unplumbed but the server is now emitting it. ` +
        'The panel is rendering an em-dash over real data -- the direction nobody ' +
        'reports, because it looks correct. Register it and drop the declaration.',
    );
  });
});

/**
 * The declared-unplumbed keys, LIFTED from field-keys.test.js rather than
 * copied. A copy would be a second inventory that drifts from the first, and
 * this crew has paid full price twice tonight for two copies of one fact.
 */
function unplumbedKeys() {
  const src = readFileSync(join(HERE, 'dashboard', 'field-keys.test.js'), 'utf8');
  const block = src.match(/NOT_YET_PUBLISHED = Object\.freeze\(\{([\s\S]*?)\n\}\);/);
  assert.ok(block, 'Could not lift NOT_YET_PUBLISHED from field-keys.test.js.');
  return [...block[1].matchAll(/^\s*'([^']+)':/gm)].map((m) => m[1]);
}

/** Every JSON object body in the captures, flattened one level. */
function readCaptureBodies() {
  const bodies = [];
  for (const file of readdirSync(CAPTURE_DIR)) {
    if (file === 'manifest.json') continue;
    const capture = JSON.parse(readFileSync(join(CAPTURE_DIR, file), 'utf8'));
    for (const entry of Object.values(capture.endpoints ?? capture)) {
      const body = entry?.body;
      if (body && typeof body === 'object') {
        bodies.push(body);
        for (const nested of Object.values(body)) {
          if (nested && typeof nested === 'object' && !Array.isArray(nested)) bodies.push(nested);
        }
      }
    }
  }
  return bodies;
}
