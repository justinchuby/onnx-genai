// Copyright (c) Microsoft Corporation.
//
// The never-bind guard.
//
// NEVER_BIND lists wire fields that are real, correctly computed, and named
// after a quantity they do not hold. They are more dangerous than stubs: a
// stub is discoverable by grep, but a correct number under a wrong name looks
// right forever and passes any review that asks "is this field computed?".
//
// This test is the enforcement. Without it NEVER_BIND is a comment.

import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import { ENDPOINTS, NEVER_BIND, PROVENANCE } from './telemetry-provenance.js';
import { createTelemetryStore } from './telemetry-store.js';
import { FIELD_STATES } from './telemetry-field.js';

const HERE = new URL('./', import.meta.url);

/** Every first-party source file, excluding tests and other agents' modules. */
function shellSources() {
  const files = [];
  const walk = (dir, prefix = '') => {
    for (const entry of readdirSync(fileURLToPath(new URL(dir, HERE)), { withFileTypes: true })) {
      const rel = `${prefix}${entry.name}`;
      if (entry.name.startsWith('.') || entry.name === 'node_modules') continue;
      if (entry.isDirectory()) {
        walk(`${dir}${entry.name}/`, `${rel}/`);
      } else if (entry.name.endsWith('.js') && !entry.name.endsWith('.test.js')) {
        files.push({ rel, text: readFileSync(fileURLToPath(new URL(`${dir}${entry.name}`, HERE)), 'utf8') });
      }
    }
  };
  walk('./');
  return files;
}

test('NEVER_BIND entries each cite a file:line so the claim is checkable', () => {
  assert.ok(NEVER_BIND.length > 0, 'the list should not be silently empty');

  for (const entry of NEVER_BIND) {
    assert.match(entry.why, /crates\/[^\s]+:\d+/, `${entry.field} must cite evidence`);
    assert.ok(entry.endpoint.startsWith('/'), `${entry.field} must name an endpoint`);
  }
});

test('no provenance entry reads a never-bind field', () => {
  for (const { endpoint, field } of NEVER_BIND) {
    for (const [key, entry] of Object.entries(PROVENANCE)) {
      if (entry.source !== endpoint) continue;
      const paths = [entry.path, ...Object.values(entry.byOrigin ?? {}).map((o) => o.path)];
      for (const path of paths) {
        if (typeof path !== 'string') continue;
        assert.notEqual(
          path.split('.').at(-1),
          field,
          `"${key}" binds ${endpoint}.${field}, which must never be displayed`,
        );
      }
    }
  }
});

// The broad net. A panel could read the wire directly rather than going
// through PROVENANCE, so this scans the source for the field name appearing as
// a property access or key on a parsed body.
test('no shell module reads a never-bind field off a response body', () => {
  const sources = shellSources();
  assert.ok(sources.length > 5, `expected to scan several files, found ${sources.length}`);

  for (const { field, why, exemptions = [] } of NEVER_BIND) {
    const patterns = [
      new RegExp(`\\.${field}\\b`),
      new RegExp(`\\[['"]${field}['"]\\]`),
      new RegExp(`['"]${field}['"]\\s*:`),
    ];
    for (const { rel, text } of sources) {
      // The declaration itself names the field, which is the point of it.
      if (rel === 'telemetry-provenance.js') continue;
      // Remove the declared safe tokens FIRST, so any OTHER occurrence of the
      // field still fires. This is a subtraction of exact strings, never of
      // whole files -- excusing a file would excuse the one place most likely
      // to bind the field.
      let scanned = text;
      for (const { token } of exemptions) scanned = scanned.split(token).join('');
      for (const pattern of patterns) {
        assert.ok(
          !pattern.test(scanned),
          `${rel} appears to read "${field}", which must never be bound. ${why}`,
        );
      }
    }
  }
});

// An exemption is a HOLE in the ban above, and a hole nobody can see is how a
// guard quietly stops guarding. Two ways that happens: the reason goes
// unwritten, or the code the exemption was granted for is deleted and the hole
// outlives it -- still subtracting a token, now from files it was never
// examined against. So every exemption must justify itself AND still be load
// bearing. When the last exempted expression disappears, this goes red and the
// exemption gets deleted rather than inherited.
test('every never-bind exemption is justified and still earning its keep', () => {
  const sources = shellSources();
  const granted = NEVER_BIND.flatMap((entry) =>
    (entry.exemptions ?? []).map((exemption) => ({ ...exemption, field: entry.field })),
  );

  for (const { field, token, why } of granted) {
    assert.ok(
      typeof why === 'string' && why.length > 40,
      `the "${token}" exemption on "${field}" must say why it is safe`,
    );
    assert.ok(
      token.includes(field),
      `the "${token}" exemption on "${field}" does not mention the field it excuses`,
    );

    const users = sources.filter(
      ({ rel, text }) => rel !== 'telemetry-provenance.js' && text.includes(token),
    );
    assert.ok(
      users.length > 0,
      `the "${token}" exemption on "${field}" no longer matches any shipping source. ` +
        'Delete it: an exemption for code that no longer exists is a permanent hole in ' +
        'the ban, granted for a reason nobody can check.',
    );
  }
});

// AN EXEMPTION'S PREMISE MUST BE EXECUTABLE, NOT MERELY WRITTEN DOWN.
//
// Every exemption above carries a `why` in prose, and prose is not checked by
// anything. The premise under all of them is the same sentence: "this spelling
// can only mean the catalogue addressing itself, never a panel reading the
// field off a response body." If that sentence is false, the exemption is not
// a narrow hole -- it is an open door positioned exactly where the defect walks
// in, and the guard reports green while holding it.
//
// IT WAS FALSE HERE. The `path` ban once exempted the bare identifier
// `entry.path`, justified as "this table addressing itself, not a panel reading
// `path` off a /v1/models body". But telemetry-store.js binds `entry` to a
// /v1/models WIRE OBJECT when it picks the primary model -- `entries.map((entry)
// => entry?.id)` over `models.body.data`. A future `entry.path` at that site
// reads the operator's home directory straight off the wire, and the exemption
// subtracted it before the scan ever saw it.
//
// So this test does not trust the prose. It builds the wire-read spelling of
// every banned field, subtracts the real granted exemptions from it exactly as
// the scan does, and asserts the ban still fires. It is the ONE assertion here
// that fails if an exemption is ever widened back into an identifier.
test('no exemption subtracts a spelling the defect could use', () => {
  const granted = NEVER_BIND.flatMap((entry) => entry.exemptions ?? []);
  assert.ok(
    granted.length > 0,
    'this test is vacuous with no exemptions granted -- it would pass by having nothing ' +
      'to subtract. If the last exemption was deleted, delete this test with it.',
  );

  for (const { field, exemptions = [] } of NEVER_BIND) {
    // How a panel would actually read the field off a parsed body. `entry` is
    // used deliberately: it is the identifier telemetry-store.js already binds
    // to a /v1/models element, so it is the likeliest spelling of the defect.
    const wireReads = [
      `const leaked = entry.${field};`,
      `const leaked = body.data[0].${field};`,
      `const leaked = entry['${field}'];`,
    ];

    for (const wireRead of wireReads) {
      let scanned = wireRead;
      for (const { token } of exemptions) scanned = scanned.split(token).join('');

      const patterns = [
        new RegExp(`\\.${field}\\b`),
        new RegExp(`\\[['"]${field}['"]\\]`),
        new RegExp(`['"]${field}['"]\\s*:`),
      ];
      assert.ok(
        patterns.some((pattern) => pattern.test(scanned)),
        `the exemptions on "${field}" swallow ${JSON.stringify(wireRead)}, which is a read ` +
          'of the banned field off a response body. An exemption must subtract the smallest ' +
          'unique EXPRESSION that covers the legitimate use, never a bare identifier the ' +
          'defect could spell the same way.',
      );
    }
  }
});

// ---------------------------------------------------------------------------
// THE RUNTIME HALF.
//
// Everything above this line is a file read plus a regular expression -- a
// grep wearing a test's clothing. It answers "does any source contain this
// string", and the question is "does any field a visitor can see carry this
// value when the code RUNS". Those differ exactly where it matters: a computed
// key, an alias, a spread, or an adapter mapping is invisible to source
// reading, and nobody types a banned name on purpose. The `exemptions`
// mechanism above makes the gap slightly wider, since it subtracts tokens
// before matching.
//
// So this half never looks at source. It puts a unique sentinel into the
// banned field on the wire, runs a real store, and asserts the sentinel does
// not surface in any field. A binding assembled at runtime from two harmless
// strings is caught here and cannot be caught above.
//
// Credit: @c8d9a40e, whose prefix-cache panel proves zero bindings by mounting
// against a throwing Proxy rather than grepping for field names. Grep proves
// the bindings you thought of are absent; execution proves all of them are.

const BASE_URL = 'http://127.0.0.1:8123';

function fetchReturning(routes) {
  return async (url) => {
    const route = routes[url.replace(BASE_URL, '')];
    if (route === undefined) {
      return {
        ok: false,
        status: 404,
        async json() { return {}; },
        async text() { return ''; },
      };
    }
    return {
      ok: true,
      status: 200,
      async json() { return route; },
      async text() { return JSON.stringify(route); },
    };
  };
}

/** A models list whose entries carry a unique sentinel in every banned field. */
function modelsBodyPoisonedWith(sentinel) {
  const entry = { id: 'qwen-scatter', object: 'model', owned_by: 'demo', is_default: true };
  for (const { endpoint, field } of NEVER_BIND) {
    if (endpoint === ENDPOINTS.MODELS) entry[field] = `${sentinel}-${field}`;
  }
  return { object: 'list', data: [entry] };
}

async function pollWith(routes) {
  const store = createTelemetryStore({ baseUrl: BASE_URL, fetchImpl: fetchReturning(routes) });
  await store.pollOnce();
  return store.getSnapshot().fields;
}

const carriers = (fields, needle) =>
  Object.entries(fields)
    .filter(([, f]) => typeof f.value === 'string' && f.value.includes(needle))
    .map(([key]) => key);

// THE POSITIVE CONTROL, AND IT IS NOT OPTIONAL HERE. After the model-directory
// ban, NO provenance row reads /v1/models at all -- so "the sentinel did not
// surface" is trivially true and would stay true if the sentinel never reached
// the store, if the poll silently failed, or if the scan read nothing. This
// proves the whole apparatus can find a value that IS bound, using the same
// injection, the same poll and the same scan.
test('the sentinel apparatus finds a value that IS bound', async () => {
  const sentinel = 'CANARY-9f3a2b';
  const fields = await pollWith({
    [ENDPOINTS.HEALTH]: { status: 'ok', model: sentinel },
  });

  assert.deepEqual(
    carriers(fields, sentinel),
    ['server.model_id'],
    'a sentinel placed on a BOUND wire field must be found, or this file proves nothing',
  );
});

test('no never-bind field surfaces in any rendered field when the store runs', async () => {
  const sentinel = 'POISON-4c81de';
  const fields = await pollWith({
    [ENDPOINTS.HEALTH]: { status: 'ok', model: 'qwen-scatter' },
    [ENDPOINTS.MODELS]: modelsBodyPoisonedWith(sentinel),
  });

  // Anti-vacuity: a snapshot of nothing cannot leak anything.
  assert.ok(
    Object.keys(fields).length > 20,
    `expected a populated snapshot, got ${Object.keys(fields).length} fields`,
  );
  assert.equal(fields['server.model_id'].state, FIELD_STATES.MEASURED, 'the poll must have worked');

  assert.deepEqual(
    carriers(fields, sentinel),
    [],
    'a banned wire field reached a rendered field. Source scanning cannot see this ' +
      'if the key was computed, aliased or spread -- which is the direction that ships.',
  );
});
