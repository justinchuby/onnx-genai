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

// The README now DOCUMENTS the staleness ceiling as shipped behaviour: past the
// ceiling the value is removed and the age remains. That claim is the whole
// reason a stale number is safe to show at all -- "20.7 tok/s . 4m old" reads
// to every human as "20.7 tok/s", so the ceiling is what stops an age suffix
// being decoration on a fabricated-feeling number.
//
// If the ceiling is ever removed or renamed, nothing breaks and nothing fails.
// The panels keep rendering, the ages keep appearing, and the README keeps
// promising a safeguard that no longer runs -- documentation describing a
// mechanism that was deleted, which is the exact failure this file exists for.
test('README staleness-ceiling claim is backed by shipping code', () => {
  const readme = readFileSync(join(demoDir, 'README.md'), 'utf8');
  if (!readme.includes('staleness ceiling')) return;

  const fieldState = readFileSync(join(demoDir, 'dashboard/field-state.js'), 'utf8');
  const panelKit = readFileSync(join(demoDir, 'dashboard/panel-kit.js'), 'utf8');

  assert.ok(
    fieldState.includes('DEFAULT_STALE_CEILING_MS') && fieldState.includes('isPastStaleCeiling'),
    'README documents a staleness ceiling, but dashboard/field-state.js no ' +
      'longer exports DEFAULT_STALE_CEILING_MS / isPastStaleCeiling.',
  );

  // Exporting the check is not the same as USING it. A ceiling nothing calls is
  // an existence check standing in for behaviour -- the defect class that has
  // bitten this project repeatedly.
  assert.ok(
    panelKit.includes('isPastStaleCeiling('),
    'dashboard/panel-kit.js no longer CALLS isPastStaleCeiling. The ceiling ' +
      'would still exist and still be exported, and every stale value would ' +
      'render as an unbounded number with an age suffix -- exactly what the ' +
      'README promises cannot happen.',
  );
});

// ---------------------------------------------------------------------------
// CONTRACT.md is a doc comment with a bigger blast radius: panel authors follow
// it INSTEAD of reading the shell. So the same rule applies -- the normative
// method name it mandates must be the one the shell actually calls.
//
// It was wrong, and it cost four separate agents a wrong implementation.
// CONTRACT.md mandated `destroy()` in bolded "must" text citing AC22 by number,
// while dashboard/index.js:209 calls `handle.unmount()`. A conforming panel
// returned `{destroy}`, the shell invoked `undefined()`, and the shell's
// per-panel try/catch -- which exists for the good reason that one bad panel
// must not strand everyone else's subscriptions -- swallowed the TypeError.
// Every subscription leaked, silently, which is an AC22 failure that only
// appears as memory growth over a 60s run.
//
// What made it survive four reports: `destroy()` is CORRECT on two neighbouring
// objects in the same file (`roving.destroy()`, `adapter.destroy()`), so anyone
// grepping `destroy` in the shell got confirmation the contract was right.
// ---------------------------------------------------------------------------

test('the panel-handle teardown method in CONTRACT.md is the one the shell calls', async () => {
  const { readFileSync } = await import('node:fs');
  const contract = readFileSync(new URL('./CONTRACT.md', import.meta.url), 'utf8');
  const shell = readFileSync(new URL('./dashboard/index.js', import.meta.url), 'utf8');

  // What the shell invokes on a PANEL handle. Deliberately anchored to
  // `handle.` so the sibling roving/adapter `.destroy()` calls cannot answer
  // for it -- those are different objects and they are why grepping lies here.
  const called = [...shell.matchAll(/\bhandle\.(\w+)\(/g)].map((m) => m[1]);
  assert.ok(called.length > 0, 'found no `handle.<method>()` call in the shell; this scan is broken');
  const teardown = called[0];
  assert.equal(teardown, 'unmount', 'the shell changed its teardown call; CONTRACT.md must follow');

  // The contract's normative lifecycle clause must name that method.
  const lifecycle = contract.slice(contract.indexOf('### Lifecycle'));
  assert.ok(
    lifecycle.includes(`\`${teardown}()\`** — you **must** return this`),
    `CONTRACT.md's lifecycle section does not mandate \`${teardown}()\`, which is what ` +
      `the shell calls. A panel following the contract literally would leak every ` +
      `subscription, silently, and fail AC22.`,
  );

  // And the @returns signature must agree with the prose beside it.
  assert.ok(
    contract.includes(`@returns {{ ${teardown}: () => void, describe: () => string }}`),
    `CONTRACT.md's mount() @returns signature disagrees with its own lifecycle prose ` +
      `about the teardown method name.`,
  );
});

test('every shipped panel returns the teardown method the contract mandates', async () => {
  const { readdirSync, readFileSync } = await import('node:fs');
  const dir = new URL('./dashboard/', import.meta.url);
  const panelFiles = readdirSync(dir).filter(
    (f) => f.endsWith('.js') && !f.includes('.test.') && !['index.js', 'panel-kit.js', 'field-state.js', 'store-adapter.js', 'sparkline.js'].includes(f),
  );
  assert.ok(panelFiles.length >= 4, `expected several panel modules, found ${panelFiles.length}; scan is broken`);

  for (const file of panelFiles) {
    const src = readFileSync(new URL(file, dir), 'utf8');
    if (!/export function mount/.test(src)) continue;
    assert.ok(
      /\bunmount\s*\(\)\s*\{/.test(src),
      `${file} exports mount() but never defines unmount(); the shell would leak its subscription`,
    );
  }
});
