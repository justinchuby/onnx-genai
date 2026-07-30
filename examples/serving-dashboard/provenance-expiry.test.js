/**
 * D272 TRIPWIRE — AN HONESTY RULE THAT EXPIRES WHEN THE CODE SUCCEEDS.
 *
 * @c0de4c2e named the inverse of the defect this project was built to catch.
 * All session we have hunted FALSE NAMES OVER CORRECT VALUES. This is the
 * mirror: a CORRECT VALUE UNDER AN HONESTY RULE THAT CALLS IT A FABRICATION.
 * A `NOT_PLUMBED` entry is a claim about the server -- "no endpoint carries
 * this" -- and the moment someone wires the endpoint, that claim becomes a lie
 * told by the mechanism we built to prevent lies. The dashboard would then
 * suppress its own working telemetry, in the honesty footer, which is the one
 * component whose entire job is to be trusted.
 *
 * Nothing re-checks an honesty rule after the code it indicts gets fixed.
 * This does. It fails the moment a NOT_PLUMBED field starts being served --
 * which is a GOOD DAY, and the test says so in its message rather than
 * reading as a regression.
 *
 * The control case is the point (D264): a check that cannot tell a served
 * field from an unserved one would pass this suite while asserting nothing,
 * so `batch_utilization` -- classified MEASURED and genuinely computed at
 * admin.rs:169 -- is asserted to be seen as PLUMBED by the same parser. If
 * that control ever flips, the instrument is broken, not the server.
 */

import { describe, it, before } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ADMIN_RS = resolve(HERE, '../../crates/onnx-genai-server/src/routes/admin.rs');

/** The status handler is the only endpoint these entries claim is silent. */
const STATUS_ENDPOINT = '/v1/status';

let adminSource = '';
let register = null;

before(async () => {
  adminSource = readFileSync(ADMIN_RS, 'utf8');
  register = (await import('./telemetry-provenance.js')).PROVENANCE;
});

/**
 * Is `field` handed a real expression in the NodeStatus literal, or `None`?
 * Returns 'plumbed' | 'none' | 'absent'. Deliberately three-valued: a field
 * the handler never mentions is a different fact from one it explicitly
 * declines to fill, and collapsing them would hide a rename.
 */
function wireStatus(field) {
  const match = adminSource.match(new RegExp(`^\\s*${field}:\\s*([^\\n]*)$`, 'm'));
  if (!match) return 'absent';
  return /^None\s*,?\s*$/.test(match[1]) ? 'none' : 'plumbed';
}

describe('the provenance register expires when the server catches up', () => {
  it('found the status handler and the register', () => {
    // ANTI-VACUITY / AC106. Every assertion below is about a file resolved by
    // relative path across a crate boundary. A search of a directory that does
    // not exist returns zero hits and reports success, so the root is proven
    // to resolve BEFORE anything is concluded from an absence.
    assert.ok(
      adminSource.length > 1000,
      `expected the status handler at ${ADMIN_RS}; read ${adminSource.length} bytes`,
    );
    assert.ok(register && Object.keys(register).length > 10, 'register did not load');
  });

  it('can tell a served field from an unserved one', () => {
    // THE CONTROL. Without this, a parser that returned 'none' for everything
    // would make the real assertion below pass forever while checking nothing.
    assert.equal(
      wireStatus('batch_utilization'),
      'plumbed',
      'the parser cannot see a genuinely computed field (admin.rs assigns ' +
        'batch_utilization from fn batch_utilization). Every other result in ' +
        'this file is meaningless until this passes.',
    );
    assert.equal(
      wireStatus('kv_usage'),
      'none',
      'the parser cannot see an explicit None. Same consequence.',
    );
  });

  it('claims no field is unplumbed that the server has started serving', () => {
    const expired = [];
    let examined = 0;

    for (const [name, entry] of Object.entries(register)) {
      if (entry.classification !== 'NOT_PLUMBED') continue;
      if (entry.source !== STATUS_ENDPOINT || !entry.path) continue;

      examined += 1;
      // 'absent' is NOT a failure and treating it as one was this suite's own
      // first bug: a field the handler never mentions is MORE unplumbed than
      // one explicitly set to None, not less. Only 'plumbed' contradicts the
      // register. I nearly asserted that absence meant a rename -- which
      // cannot be distinguished from a field that was simply never added, so
      // it would have reported the register's most correct entries as defects.
      if (wireStatus(entry.path) === 'plumbed') {
        expired.push(`${name} -> admin.rs assigns '${entry.path}'`);
      }
    }

    assert.ok(
      examined > 0,
      'no NOT_PLUMBED entry was examined, so the assertion below would pass ' +
        'over an empty set. Either the register changed shape or the endpoint ' +
        `constant '${STATUS_ENDPOINT}' no longer matches its source values.`,
    );

    assert.deepEqual(
      expired,
      [],
      'GOOD NEWS, AND IT MUST NOT SHIP AS-IS: a field classified NOT_PLUMBED ' +
        'is now genuinely served. The register still tells the visitor we ' +
        'cannot measure it, so the honesty footer would deny our own working ' +
        'telemetry -- an honesty mechanism that expires the moment it ' +
        'succeeds. Reclassify to MEASURED with its evidence line; do not ' +
        `silence this test.\nFOUND:\n  ${expired.join('\n  ')}`,
    );
  });
});
