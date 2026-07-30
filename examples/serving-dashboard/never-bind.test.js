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

  for (const { field, why } of NEVER_BIND) {
    const patterns = [
      new RegExp(`\\.${field}\\b`),
      new RegExp(`\\[['"]${field}['"]\\]`),
      new RegExp(`['"]${field}['"]\\s*:`),
    ];
    for (const { rel, text } of sources) {
      // The declaration itself names the field, which is the point of it.
      if (rel === 'telemetry-provenance.js') continue;
      for (const pattern of patterns) {
        assert.ok(
          !pattern.test(text),
          `${rel} appears to read "${field}", which must never be bound. ${why}`,
        );
      }
    }
  }
});
