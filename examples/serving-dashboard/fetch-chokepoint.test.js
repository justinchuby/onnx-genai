// Every network read in the shell must go through `fetchWithDeadline`.
//
// WHY THIS FILE EXISTS, stated as a defect rather than as a principle:
//
// `app.js` used to call the global `fetch` directly for the /health probe that
// decides which server we are. That call is awaited at :95, and the failure-
// state UI does not mount until :122. A server that ACCEPTS the socket and then
// says nothing -- a paused process, or a firewall DROP, which is not the same
// thing as a refused connection -- makes that promise pend forever. `main()`
// never returns, the failure states never mount, and the visitor gets a blank
// page with no error and no launch command: precisely the outcome the failure-
// state machinery was built to prevent. A bare `catch` cannot help, because
// nothing is ever thrown.
//
// That call now uses `fetchWithDeadline` and the defect is gone. This file
// exists because of HOW it was gone: by hand, and nothing was left behind that
// would notice it coming back.
//
// The reason it survived as long as it did is the part worth pinning. The same
// bug existed at `telemetry-store.js`, was found, and was fixed first -- not
// because it was worse, but because the store takes an injectable `fetchImpl`
// and a test could therefore reach it. `app.js` exports NOTHING. No test can
// call `determineSelfClasses()`, so no behavioural test can ever cover that
// call site. Testability, not severity, decided which half got fixed first.
//
// So a behavioural test is not available here, and pretending otherwise would
// mean restructuring the shell's entry point under a commit freeze to make one
// possible. A source-level invariant is available, is strictly broader, and
// costs zero production bytes: NO module may call the global `fetch` at all.
// `request-deadline.js` is the single exemption, because being the one place
// that touches the global is its entire job.
//
// This catches the regression in `app.js`, and it also catches it in every
// panel nobody has written yet -- which is the actual risk, because the next
// author will copy a working line from a neighbouring file and will have no
// way to know that the neighbouring line is load-bearing.
//
// SCOPE, HONESTLY. This scanner matches text. It does not parse JavaScript.
//   - It reads HEAD, not the working tree, so it is reproducible and cannot be
//     reddened by another agent's uncommitted edit. The gap that leaves is
//     real and worth naming: `run-demo.sh` serves the WORKING TREE, so bytes a
//     visitor loads tonight are not necessarily the bytes scanned here. A
//     violation that is edited but never committed will not be caught.
//   - It scans comments too, and does not strip them. A comment containing the
//     literal text of a fetch call is therefore a false positive, and the
//     repair is to reword the comment. That is deliberate. A scanner that
//     stripped comments first would be fooled by a comment marker inside a
//     string literal -- a URL is the obvious one -- and would silently stop
//     scanning the rest of a line that might contain a real call. That fails
//     in the direction of ACCEPTING. This fails in the direction of REFUSING,
//     and for a guard standing over a hang that blanks the page, refusing is
//     the correct bias.
//   - It catches a direct call and a direct reference to the global. It does
//     not catch a determined evasion (`const f = fetch; f(url)`), and is not
//     trying to: the regression this guards against is someone writing the
//     obvious line, not someone working around a check.

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

const repoRoot = execFileSync('git', ['rev-parse', '--show-toplevel'], {
  encoding: 'utf8',
}).trim();

// Resolved ONCE, and every read below uses this SHA rather than the name HEAD.
//
// This tree is shared and is being committed to while the suite runs. `HEAD` is
// a moving target: enumerate with it, then read with it, and the two calls can
// straddle somebody else's commit. A file listed by the first call can be gone
// by the second, and `git show HEAD:<path>` would throw a subprocess error
// instead of reporting a clean verdict -- a failure that names neither the
// defect nor the race, and that nobody can reproduce.
//
// Pinning makes this file internally consistent at one instant. Two tests in
// this suite already flake for exactly this reason; this one will not join them.
const HEAD_SHA = execFileSync('git', ['rev-parse', 'HEAD'], {
  cwd: repoRoot,
  encoding: 'utf8',
}).trim();

// A DIRECTORY pathspec, never a glob. `git ls-tree` with a glob pathspec
// returns zero matches SILENTLY and exits 0, which would make every assertion
// below vacuously true. Enumerate the subtree and filter in JavaScript, where
// a mistake is visible.
const DASHBOARD = 'examples/serving-dashboard';

const trackedPaths = execFileSync(
  'git',
  ['ls-tree', '-r', '--name-only', HEAD_SHA, '--', DASHBOARD],
  { cwd: repoRoot, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 },
)
  .split('\n')
  .filter(Boolean)
  .map((path) => path.slice(`${DASHBOARD}/`.length));

/**
 * The one module allowed to name the global. Being the sole caller of
 * `globalThis.fetch` is what `request-deadline.js` is FOR.
 */
const CHOKEPOINT = 'request-deadline.js';

/**
 * Modules under scan: tracked, JavaScript, not a test.
 *
 * Tests are excluded because stubbing `fetch` is a legitimate and common thing
 * for a test to do, and flagging it would train everyone to ignore this file.
 */
const modulePaths = trackedPaths.filter(
  (path) => path.endsWith('.js') && !path.endsWith('.test.js'),
);

/** A direct call: `fetch(`, `await fetch (`, but NOT `fetchWithDeadline(`, */
/*  NOT `fetchImpl(`, and NOT a method call like `client.fetch(`.            */
const DIRECT_CALL = /(?:^|[^\w.$])fetch\s*\(/;

/** A direct reference to the global, which is the same defect one step out. */
const GLOBAL_REFERENCE = /(?:globalThis|window|self)\s*\.\s*fetch\b/;

/**
 * Find unguarded uses of the global fetch in one module's source.
 *
 * Pure and total: takes text, returns findings. It touches no file, which
 * means the positive controls below can prove this scanner fires WITHOUT
 * mutating anything on disk. That matters in a shared tree -- copying a file
 * aside and restoring it is a read-modify-write with a multi-second window,
 * and another agent's write inside that window is silently reverted.
 *
 * @param {string} source
 * @returns {{ line: number, text: string, kind: string }[]}
 */
function findUnguardedFetch(source) {
  const findings = [];
  source.split('\n').forEach((text, index) => {
    const kind = DIRECT_CALL.test(text)
      ? 'calls the global fetch directly'
      : GLOBAL_REFERENCE.test(text)
        ? 'names the global fetch'
        : null;
    if (kind) findings.push({ line: index + 1, text: text.trim(), kind });
  });
  return findings;
}

/** Read a path's committed bytes at the pinned SHA. See SCOPE above. */
function readFromHead(path) {
  return execFileSync('git', ['show', `${HEAD_SHA}:${DASHBOARD}/${path}`], {
    cwd: repoRoot,
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
  });
}

describe('every network read goes through the deadline chokepoint', () => {
  test('CAN RUN: the enumerator actually reached the shell modules', () => {
    // Without this, a broken pathspec yields an empty corpus and every
    // assertion below passes while measuring nothing. A green from an empty
    // set is indistinguishable from a green from a clean one.
    assert.ok(
      modulePaths.length >= 20,
      `CANNOT RUN: found ${modulePaths.length} non-test modules under ${DASHBOARD}; ` +
        `expected at least 20. The pathspec or the subtree moved -- fix the ` +
        `enumeration, do not lower this floor.`,
    );
    for (const required of ['app.js', 'telemetry-store.js', CHOKEPOINT]) {
      assert.ok(
        modulePaths.includes(required),
        `CANNOT RUN: ${required} is not in the scanned set, so a regression in ` +
          `it would go unnoticed. Scanned: ${modulePaths.length} modules.`,
      );
    }
  });

  test('the scanner fires on the lines it exists to catch', () => {
    // Positive controls. If these ever stop failing, the real assertion below
    // is green because the instrument is broken, not because the code is
    // clean -- and it would look exactly the same either way.
    const mustBeCaught = [
      ['bare call', `const r = await fetch(url);`],
      ['bare call, no await', `fetch('/health').then(done);`],
      ['space before paren', `await fetch (input, init);`],
      ['start of line', `fetch(u);`],
      ['global reference', `const f = globalThis.fetch;`],
      ['window reference', `window.fetch(input);`],
      ['self reference', `const g = self . fetch;`],
    ];
    for (const [label, line] of mustBeCaught) {
      assert.equal(
        findUnguardedFetch(line).length,
        1,
        `the scanner MISSED a ${label}: ${line}\n` +
          `Every clean result this file reports is worthless while this is true.`,
      );
    }
  });

  test('the scanner is not simply flagging every line it reads', () => {
    // The starved-class arm. A scanner that answered "violation" to everything
    // would pass the positive controls above and be useless. These are the
    // idioms the codebase actually uses, and every one of them must come back
    // clean or authors will route around the check.
    const mustBeClean = [
      `const r = await fetchWithDeadline(input, { timeoutMs: 2000 });`,
      `return await fetchImpl(input, { ...init, signal });`,
      `const { fetchImpl = globalThis_notReally } = options;`,
      `await client.fetch(url);`,
      `const again = refetch(url);`,
      `import { fetchWithDeadline } from './request-deadline.js';`,
      `let prefetch = 0;`,
    ];
    for (const line of mustBeClean) {
      assert.deepEqual(
        findUnguardedFetch(line),
        [],
        `the scanner FALSE-POSITIVED on a legitimate idiom: ${line}`,
      );
    }
  });

  test('no module outside the chokepoint touches the global fetch', () => {
    const violations = [];
    for (const path of modulePaths) {
      if (path === CHOKEPOINT) continue;
      for (const finding of findUnguardedFetch(readFromHead(path))) {
        violations.push(`  ${path}:${finding.line} ${finding.kind}\n      ${finding.text}`);
      }
    }

    assert.deepEqual(
      violations,
      [],
      `A module is reaching past the deadline chokepoint:\n${violations.join('\n')}\n\n` +
        `A fetch with no deadline against a server that accepts the socket and\n` +
        `never answers does not reject -- it pends forever. If it is awaited\n` +
        `before the failure states mount, the visitor gets a blank page with no\n` +
        `error and no launch command.\n\n` +
        `THE FIX IS NOT TO ADD AN EXEMPTION HERE. Import fetchWithDeadline from\n` +
        `./request-deadline.js and call that instead; it composes your signal\n` +
        `rather than replacing it, and clears its timer on every path.`,
    );
  });

  test('the chokepoint still earns its exemption', () => {
    // An exemption for a file that no longer needs one is not harmless: it is
    // a hole held open by nothing, and the next reader takes it as evidence
    // that naming the global is acceptable somewhere. If request-deadline.js
    // ever stops touching the global -- because it was refactored, renamed, or
    // gutted -- this exemption must be re-argued rather than inherited.
    const source = readFromHead(CHOKEPOINT);
    const findings = findUnguardedFetch(source);
    assert.ok(
      findings.length > 0,
      `${CHOKEPOINT} is exempted from this check because it is the one module ` +
        `that wraps the global fetch -- but it no longer references it at all. ` +
        `Either the wrapper moved (point CHOKEPOINT at its new home) or it is ` +
        `gone (delete the exemption). Do not leave the exemption standing.`,
    );
    assert.ok(
      source.includes('fetchImpl'),
      `${CHOKEPOINT} no longer takes an injectable fetchImpl. That parameter is ` +
        `the only reason any of this is testable at the call sites -- losing it ` +
        `is how app.js became untestable in the first place.`,
    );
  });
});
