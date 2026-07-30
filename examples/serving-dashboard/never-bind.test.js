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

import { NEVER_BIND, PROVENANCE } from './telemetry-provenance.js';

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
// bearing. When the last `entry.path` disappears, this goes red and the
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
