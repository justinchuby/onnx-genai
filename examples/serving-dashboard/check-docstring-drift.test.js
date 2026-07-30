// A doc comment is prose that lives in a code file. It inherits the AUTHORITY
// of code while carrying NONE of the guarantees -- nothing executes it, no test
// covers it, and it is the most convincing place a stale claim can hide.
//
// This project has already been bitten twice by exactly that. The `FieldState`
// @typedef listed four states for hours while `FIELD_STATES` exported five, and
// three people argued the enum's shape from the comment rather than the
// constant below it. It ALSO declared the measured state as 'measured' while
// the constant emitted 'ok' -- so a reader following our own documentation
// would write `field.state === 'measured'` and get a comparison that is never
// true, with no error to explain why.
//
// Both were fixed by hand. Neither could stay fixed by hand: the wire value is
// under an open ruling ('ok' -> 'measured'), so the docstring is one commit
// away from being wrong again, in the same way, for the same reason.
//
// So the union in the comment is now checked against the object that ships.
// Whichever way the rename lands, the two cannot silently disagree.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { FIELD_STATES } from './telemetry-field.js';

const demoDir = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(demoDir, 'telemetry-field.js'), 'utf8');

/** The `'a' | 'b'` union declared by a named @typedef, as a set of strings. */
function declaredUnion(typedefName) {
  const pattern = new RegExp(`@typedef\\s*\\{([^}]*)\\}\\s*${typedefName}\\b`);
  const match = source.match(pattern);
  assert.ok(match, `no @typedef found for ${typedefName} in telemetry-field.js`);

  const members = [...match[1].matchAll(/'([^']+)'/g)].map((m) => m[1]);
  assert.ok(members.length > 0, `the ${typedefName} typedef declares no string members`);
  return new Set(members);
}

test('the FieldState typedef declares exactly the values FIELD_STATES ships', () => {
  const documented = declaredUnion('FieldState');
  const shipped = new Set(Object.values(FIELD_STATES));

  const missing = [...shipped].filter((v) => !documented.has(v));
  const invented = [...documented].filter((v) => !shipped.has(v));

  assert.deepEqual(
    missing,
    [],
    `FIELD_STATES ships ${JSON.stringify(missing)}, which the FieldState typedef ` +
      `does not document. A reader trusting the comment will not know the state ` +
      `exists -- which is how the enum was argued as four-valued while the code ` +
      `exported five.`,
  );

  assert.deepEqual(
    invented,
    [],
    `the FieldState typedef documents ${JSON.stringify(invented)}, which ` +
      `FIELD_STATES does not emit. Following the documentation produces a ` +
      `comparison that is NEVER true, with no error to explain why.`,
  );
});

test('every FIELD_STATES value is explained in the typedef body, not just listed', () => {
  // Listing a state in the union without describing it is the shape the
  // four-state comment had: technically present, useless to the reader who
  // needs to know WHEN it applies. The bullet list is the part people read.
  const typedefBlock = source.slice(source.indexOf('@typedef'), source.indexOf('*/', source.indexOf('@typedef')));

  for (const value of Object.values(FIELD_STATES)) {
    assert.ok(
      typedefBlock.includes(`\`${value}\``),
      `FIELD_STATES ships '${value}' but the FieldState doc block never explains ` +
        `it. Add a bullet saying when it applies -- a state a reader cannot ` +
        `distinguish from its neighbours will be used interchangeably with them.`,
    );
  }
});
