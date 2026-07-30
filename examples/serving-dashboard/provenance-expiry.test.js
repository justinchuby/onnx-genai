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

/**
 * D275 — A DUPLICATE KEY IN THE PROVENANCE TABLE IS THE DEFECT THIS PRODUCT REFUSES.
 *
 * @c0de4c2e found 'batch.capacity' defined TWICE in the catalogue. JS object
 * literals permit that silently: no syntax error, no warning, no lint. The
 * LAST definition wins and the first becomes dead code that still reads
 * perfectly in the file.
 *
 * Which half survives is the part that makes this worth a guard rather than a
 * fix. The dead entry at :497 is anchored to the SYMBOL `batch_capacity`. The
 * live entry at :637 is anchored to `admin.rs:178` -- A LINE NUMBER, the exact
 * citation form this crew ruled against tonight because it drifts. So the
 * catalogue silently prefers the fragile citation, and a reader who scrolls to
 * the first entry, finds an exemplary symbol-anchored one, and stops reading
 * will believe the good one is in force.
 *
 * AND THE TWO ENTRIES ARE NOT COPIES. They differ in `label` -- the string the
 * panel PAINTS. The discarded entry reads 'Effective batch capacity'; the
 * surviving one reads 'Batch limit'. The served value is
 * min(max_batch, max_queue_depth), so 'Batch limit' names max_batch: the RAW
 * ceiling, which is exactly the overstatement the discarded entry's own
 * comment was written to prevent. The whole --max-batch saga's lesson was
 * recorded in this file and silently discarded in this file, and the survivor
 * carries the one label the lesson forbids.
 *
 * The file states one field's provenance twice, the program believes one of
 * them, and nothing anywhere tells you which. That is absent-vs-zero -- the
 * defect class this dashboard exists to refuse -- sitting inside the
 * provenance table itself. The runtime object CANNOT show it: by the time the
 * module is imported, the loser is already gone. Only the source text can.
 */
describe('the provenance catalogue defines each field exactly once', () => {
  const REGISTER_PATH = new URL('./telemetry-provenance.js', import.meta.url);

  /** Keys at the register's own indent level; nested option keys are deeper. */
  function declaredKeys(source) {
    return [...source.matchAll(/^ {2}'([\w.]+)':\s*\{/gm)].map((m) => m[1]);
  }

  it('parses keys out of the register source at all', () => {
    // ANTI-VACUITY. A regex that matched nothing would report zero duplicates
    // forever -- an absence of search reported as an absence of evidence, which
    // is the failure @376a0297 named as AC106.
    const keys = declaredKeys(readFileSync(REGISTER_PATH, 'utf8'));
    assert.ok(keys.length > 20, `expected many field keys; parsed ${keys.length}`);
    assert.ok(keys.includes('batch.utilization'), 'a known key was not parsed');
  });

  it('declares no field twice', () => {
    const keys = declaredKeys(readFileSync(REGISTER_PATH, 'utf8'));
    const seen = new Set();
    const duplicates = [...new Set(keys.filter((k) => seen.has(k) || (seen.add(k), false)))];

    assert.deepEqual(
      duplicates,
      [],
      'A field is declared more than once in the provenance catalogue. JS ' +
        'keeps the LAST definition and discards the first silently -- no ' +
        'error, no warning, no lint -- so the file documents a provenance the ' +
        'program does not use, and a reader cannot tell which one is live. ' +
        'The honesty layer must not itself be ambiguous about where a number ' +
        `came from.\nDUPLICATED: ${duplicates.join(', ')}`,
    );
  });
});
