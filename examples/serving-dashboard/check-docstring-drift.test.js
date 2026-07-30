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
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { FIELD_STATES } from './telemetry-field.js';
import { assertShippingTree } from './shipping-tree.mjs';

// Provenance before content. Every path below is resolved from import.meta.url,
// so this file would read a parked worktree self-consistently and pass. Assert
// which tree we are in BEFORE asserting anything about what is in it.
assertShippingTree();

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

// ---------------------------------------------------------------------------
// The checks above name the doc comments they guard, one at a time. That is
// exactly as much coverage as somebody remembered to ask for, which makes this
// file an instance of the defect it exists to catch: it went stale in its own
// blind spots while looking directly at two known ones.
//
// The two checks below are structural instead. They do not know which comment
// they are guarding, so a doc block added tomorrow is covered the moment it is
// written, by nobody's decision.
// ---------------------------------------------------------------------------

/** Every shipping module (no tests, no fixtures), as [path, source] pairs. */
function shippingModules() {

  const out = [];
  const walk = (dir, prefix) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.name === 'node_modules' || entry.name === 'testing') continue;
      const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory()) walk(join(dir, entry.name), rel);
      else if (entry.name.endsWith('.js') && !entry.name.includes('.test.')) {
        out.push([rel, readFileSync(join(dir, entry.name), 'utf8')]);
      }
    }
  };
  walk(demoDir, '');
  return out;
}

// A doc block is bound to its subject by ADJACENCY and nothing else. There is
// no compiler to complain when that binding breaks, so when a function moves
// and its comment does not, the comment does not become an error — it becomes
// the documentation of whatever function happens to be underneath it now.
//
// This was live in four places, and one of them mattered a great deal: the
// block describing `renderField` as "THE FUNCTION THAT MAKES THE HONESTY RULE
// MECHANICAL", with its full @param list, had come to rest above
// `headlineSentence`. Every editor tooltip, every reader scrolling past, and
// every agent grepping for the honesty rule was reading that description
// attached to a one-line string helper, while `renderField` itself — the choke
// point the entire design rests on — carried no documentation at all.
//
// The signal is a doc block whose next non-blank line opens ANOTHER doc block.
// Nothing legitimately sits between a comment and the thing it describes, so
// this has no false positives by construction: it found four, all real.
test('no doc block has drifted away from the thing it documents', () => {
  const orphans = [];
  let blocksScanned = 0;

  for (const [path, src] of shippingModules()) {
    const lines = src.split('\n');
    for (let i = 0; i < lines.length; i++) {
      if (!/^\s*\/\*\*\s*$/.test(lines[i])) continue;
      let end = i;
      while (end < lines.length && !/\*\//.test(lines[end])) end++;
      const block = lines.slice(i, end + 1).join('\n');
      // Only blocks that document an INTERFACE can be orphaned in a way that
      // misleads. A free-standing prose block explaining a section is fine.
      if (!/@(param|returns)/.test(block)) continue;
      blocksScanned += 1;
      let next = end + 1;
      while (next < lines.length && lines[next].trim() === '') next++;
      if (/^\s*\/\*\*/.test(lines[next] ?? '')) {
        const subject = (block.split('\n')[1] ?? '').replace(/^\s*\*\s?/, '').trim();
        orphans.push(`${path}:${i + 1} — "${subject.slice(0, 60)}"`);
      }
    }
  }

  assert.ok(
    blocksScanned > 20,
    `only ${blocksScanned} @param/@returns blocks found across the tree; this scan is broken, ` +
      `not the tree clean`,
  );
  assert.deepEqual(
    orphans,
    [],
    `these doc blocks are followed by another doc block, so they describe nothing and the ` +
      `function below them is described by someone else's comment:\n  ${orphans.join('\n  ')}`,
  );
});

/** Top-level keys of an object literal or a `{{...}}` type, braces balanced. */
function topLevelKeys(inner) {
  const cleaned = inner
    .split('\n')
    .map((l) => l.replace(/^\s*\*\s?/, ''))
    .join('\n')
    // Comments inside a return literal are prose, and prose contains commas.
    // Without this the splitter reads an explanatory sentence as four more
    // keys and reports the real key beneath it as missing -- a check that
    // reddens correct code, which is worse than no check at all.
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/^[ \t]*\/\/.*$/gm, '');
  const parts = [];
  let depth = 0;
  let current = '';
  for (const ch of cleaned) {
    if (ch === '{' || ch === '(' || ch === '[') depth += 1;
    else if (ch === '}' || ch === ')' || ch === ']') depth -= 1;
    if (depth === 0 && ch === ',') {
      parts.push(current);
      current = '';
      continue;
    }
    current += ch;
  }
  parts.push(current);
  return parts
    .map((p) => p.match(/^\s*(?:\.\.\.)?\s*(?:async\s+)?([A-Za-z_$][\w$]*)/))
    .filter(Boolean)
    .map((m) => m[1]);
}

/** Index of the `}` closing the `{` at `open`. */
function matchBrace(src, open) {
  let depth = 0;
  for (let i = open; i < src.length; i++) {
    if (src[i] === '{') depth += 1;
    else if (src[i] === '}' && --depth === 0) return i;
  }
  return -1;
}

// A `@returns {{a, b}}` is the most load-bearing kind of doc comment we write:
// it is the only description of a function's shape that a caller reads, and
// destructuring a key that is not there yields `undefined` rather than an
// error. A caller who follows a stale @returns gets a variable that is quietly
// missing, at the call site, with no stack trace pointing anywhere near the
// function whose comment lied.
//
// So every documented key must appear in the object the function actually
// returns. Functions that return a variable rather than a literal are skipped
// — the shape is not statically visible and a guess would be worse than
// nothing — and the count of what was actually compared is asserted, because a
// skip-everything scan and a clean tree look identical from here.
test('a documented @returns shape names keys the function really returns', () => {
  const drifted = [];
  let compared = 0;

  for (const [path, src] of shippingModules()) {
    const pattern = /@returns \{\{([\s\S]*?)\}\}\s*\n\s*\*\/\s*\n(?:export )?(?:async )?function (\w+)/g;
    let match;
    while ((match = pattern.exec(src)) !== null) {
      const [, shape, name] = match;
      const documented = topLevelKeys(shape);
      assert.ok(
        documented.length > 0,
        `${path}: parsed no keys out of ${name}'s documented return shape. The parser is ` +
          `broken, which would silently pass this function forever.`,
      );

      const declaration = src.indexOf(`function ${name}`, match.index);
      const bodyEnd = matchBrace(src, src.indexOf('{', declaration));
      const body = src.slice(src.indexOf('{', declaration), bodyEnd);

      // The last object literal returned: panels build their handle at the end.
      let returned = null;
      const returns = /return \{/g;
      let hit;
      while ((hit = returns.exec(body)) !== null) {
        const open = body.indexOf('{', hit.index);
        const close = matchBrace(body, open);
        if (close > 0) returned = body.slice(open + 1, close);
      }
      if (returned === null) continue; // returns a variable; shape not visible

      compared += 1;
      const actual = topLevelKeys(returned);
      const missing = documented.filter((k) => !actual.includes(k));
      if (missing.length > 0) {
        drifted.push(
          `${path} ${name}(): documents ${JSON.stringify(missing)}, returns ${JSON.stringify(actual)}`,
        );
      }
    }
  }

  assert.ok(
    compared >= 5,
    `only ${compared} return shapes were actually compared; the scan is finding nothing to ` +
      `check rather than finding everything correct`,
  );
  assert.deepEqual(
    drifted,
    [],
    `a caller destructuring these documented keys gets undefined, with no error:\n  ` +
      `${drifted.join('\n  ')}`,
  );
});

// CONTRACT.md tells panel authors where in the shell to look. A citation that
// points past the end of the file it names is unambiguously wrong — there is
// no line there to be right about — and it sends the one reader who tried to
// verify the claim to nothing at all.
//
// Only CONTRACT.md is checked. Review records and verification logs cite code
// AS IT WAS on the day they were written; holding those to today's line
// numbers would demand we edit the evidence to match the conclusion.
test('CONTRACT.md cites lines that exist in the files it names', () => {
  const contract = readFileSync(join(demoDir, 'CONTRACT.md'), 'utf8');
  const stale = [];
  let checked = 0;

  for (const [, path, line] of contract.matchAll(/`([\w./-]+\.(?:js|css|mjs)):(\d+)`/g)) {
    let source;
    try {
      source = readFileSync(join(demoDir, path), 'utf8');
    } catch {
      stale.push(`${path}:${line} — no such file`);
      continue;
    }
    checked += 1;
    const total = source.split('\n').length;
    if (Number(line) > total) stale.push(`${path}:${line} — that file has ${total} lines`);
  }

  assert.ok(checked > 0, 'found no file:line citations in CONTRACT.md; this scan is broken');
  assert.deepEqual(
    stale,
    [],
    `CONTRACT.md sends a panel author to a line that does not exist:\n  ${stale.join('\n  ')}`,
  );

  // Past-EOF is only the half of this that is decidable in general. The
  // citation that matters most here is decidable exactly, because we know what
  // it is supposed to be pointing AT: pin it to its content, not its number.
  const cited = contract.match(/calls `handle\.unmount\(\)`\s*\n?\s*\(`dashboard\/index\.js:(\d+)`\)/);
  assert.ok(cited, 'CONTRACT.md no longer cites where the shell calls handle.unmount()');
  const shellLines = readFileSync(join(demoDir, 'dashboard/index.js'), 'utf8').split('\n');
  assert.match(
    shellLines[Number(cited[1]) - 1] ?? '',
    /handle\.unmount\(\)/,
    `CONTRACT.md points panel authors at dashboard/index.js:${cited[1]} for the teardown call, ` +
      `but that line is something else now. The line the shell actually calls it on is ` +
      `${shellLines.findIndex((l) => l.includes('handle.unmount()')) + 1}.`,
  );
});
