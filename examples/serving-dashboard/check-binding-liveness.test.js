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

  // A derived field has no wire path of its own -- it is computed client-side
  // from another series. Treating that as absent would be a blind instrument
  // reporting a false red, so the claim we CAN check is that its named input is
  // really on the wire. A derivation whose input is missing is just as dead.
  if (entry.derived) {
    const inputs = [...String(entry.evidence ?? '').matchAll(/\b([a-z][a-z0-9_]*_total)\b/g)].map(
      (m) => m[1],
    );
    if (inputs.length === 0) {
      return { ok: false, why: 'derived, but its evidence names no *_total input to verify' };
    }
    const missing = inputs.filter((name) => !String(recorded.body).includes(name));
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
