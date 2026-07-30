// Served surface — what a visitor can actually FETCH from /demo/.
//
// Every other suite on this branch asks what the page RENDERS. This one asks
// what the origin will HAND OVER, which is a strictly larger set and, until
// this file existed, an unexamined one.
//
// WHY THIS SUITE EXISTS, and it is not a hypothetical:
//
//   run-demo.sh launches both servers with `--demo-assets-dir "${SCRIPT_DIR}"`,
//   and SCRIPT_DIR is this directory. There is no allowlist, no extension
//   filter and no exclusion of any kind. So the served surface is not "the
//   demo" -- it is THE ENTIRE SOURCE DIRECTORY, measured at 111 of 111 tracked
//   files returning 200, including every internal review document we have
//   written tonight.
//
//   Verified over HTTP against a live server rather than inferred from the
//   router, because inferring is how the previous number was wrong. Re-run it:
//
//     git ls-tree -r --name-only HEAD -- examples/serving-dashboard \
//       | sed 's|examples/serving-dashboard/||' \
//       | while read -r f; do
//           printf '%s %s\n' \
//             "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8133/demo/$f")" "$f"
//         done
//
//   Result at the time of writing: 111 of 111 served, 0 refused (112 with this file). `/demo/READABILITY-REVIEW.md`
//   returns 61588 bytes of `text/markdown`. `/demo/design/skeleton.html` returns
//   `text/html`, so a browser RENDERS it as a page -- and the paragraph it calls
//   "the entire thesis of this design" states that the prefix-cache 0.0% is "a
//   real measurement", a claim this branch has since established is a hardcoded
//   literal. The retraction is three directories away in demo-spec.md and a
//   visitor fetching the mock has no way to reach it.
//
//   This is NOT a traversal bug and must not be reported as one. `../` escapes
//   were tried five ways and all returned 404: the server confines to the
//   assets dir correctly. The defect is that the assets dir is the source dir.
//
// WHAT THIS SUITE DOES NOT DO: narrow the surface. The fix is a server-side
// allowlist or moving shipped assets into their own subtree, and both live in
// other people's lanes. What it does is make the surface a CLOSED SET that
// cannot silently grow, and put a one-way ratchet under the exposure count.
//
// ── THE GUARD THIS FILE IS REALLY ABOUT ──────────────────────────────────────
//
// @732c7548's observation, which is the sharpest thing said about our test
// apparatus tonight: A COVERAGE LIST IS THE ONE PART OF A TEST THAT NOTHING
// TESTS, AND MUTATION TESTING CANNOT REACH IT, BECAUSE EVERY MUTATION YOU RUN
// IS INSIDE THE CORPUS.
//
// They are right, and it had already bitten this directory: page-claims.test.js
// scans four hardcoded paths for withdrawn-feature claims. Mutating any of
// those four reddens it, so it looks robustly guarded. But `design/skeleton.html`
// -- which is SERVED, as text/html -- is not one of the four, and no mutation
// inside the four could ever say so.
//
// The way out is that a coverage list IS reachable by mutation; the mutation is
// just not the one people run. YOU DO NOT EDIT A FILE IN THE CORPUS. YOU ADD A
// FILE TO THE TREE. The test below fails on any tracked file it cannot
// classify, so the mutation that proves it is `touch a-new-file.xyz`.
//
// That is why the classifier has no catch-all bucket and never will. A default
// class would make this suite pass for every possible tree, which is the same
// green a deleted file gives.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

const HERE = dirname(fileURLToPath(new URL('./served-surface.test.js', import.meta.url)));

const repoRoot = execFileSync('git', ['rev-parse', '--show-toplevel'], {
  cwd: HERE,
  encoding: 'utf8',
}).trim();

// `ls-tree HEAD`, not `ls-files`, and not a disk walk: the served surface we
// can reason about is the one a clean clone reproduces. An untracked file on
// somebody's desk is served too, but it is not a property of the branch -- and
// run-tests.sh already reconciles untracked files separately.
const DASHBOARD = 'examples/serving-dashboard';
const tracked = execFileSync(
  'git',
  ['ls-tree', '-r', '--name-only', 'HEAD', '--', DASHBOARD],
  { cwd: repoRoot, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 },
)
  .split('\n')
  .filter(Boolean)
  .map((path) => path.slice(`${DASHBOARD}/`.length));

const launcher = readFileSync(new URL('./run-demo.sh', import.meta.url), 'utf8');

/**
 * The classes a tracked file can belong to, in first-match order.
 *
 * Ordered, because the categories genuinely overlap: `design/demo-ux.md` is
 * both markdown and design. First-match makes the overlap a decision rather
 * than an accident.
 *
 * There is deliberately NO trailing catch-all. A file matching nothing is the
 * signal this suite exists to raise.
 */
const CLASSES = Object.freeze([
  { name: 'TEST', needsToBeServed: false, matches: (f) => /\.test\.js$/.test(f) },
  { name: 'DESIGN', needsToBeServed: false, matches: (f) => f.startsWith('design/') },
  { name: 'INTERNAL_DOC', needsToBeServed: false, matches: (f) => /\.md$/.test(f) },
  {
    name: 'TOOLING',
    needsToBeServed: false,
    matches: (f) => /\.(sh|mjs|py)$/.test(f) || /(^|\/)package\.json$/.test(f),
  },
  { name: 'FIXTURE', needsToBeServed: false, matches: (f) => f.startsWith('fixtures/') },
  { name: 'PAGE_ASSET', needsToBeServed: true, matches: (f) => /\.(js|css|html)$/.test(f) },
]);

/** @returns {string|null} the class name, or null if nothing claims the file. */
function classify(file) {
  return CLASSES.find((c) => c.matches(file))?.name ?? null;
}

const classified = tracked.map((file) => ({ file, className: classify(file) }));
const unclassified = classified.filter((entry) => entry.className === null);
const servedButNotNeeded = classified.filter(
  (entry) => entry.className !== null && entry.className !== 'PAGE_ASSET',
);

// The ratchet. MEASURED, not chosen: 113 tracked files, 30 of them assets the
// page loads, 83 served for no reason a visitor benefits from. It may fall. It
// may never rise. A rise means somebody put another file inside the origin's
// reach, which is exactly the event nothing else here would report.
//
// THIS NUMBER COUNTS THE FILES THAT MEASURE IT. Test files are served -- I
// checked, `/demo/asset-graph.test.js` returns 44356 bytes -- so adding a suite
// that measures the exposure also enlarges it by one. It was 82 when this file
// was the only such suite; `fetch-chokepoint.test.js` made it 83, and
// `frame-metadata.test.js` -- the census asserting every store frame carries
// its catalogue caption -- makes it 84. A visitor may have that one: it
// contains no credential, no path, and no fixture, only assertions about
// captions the page already paints in front of them. Recording
// that rather than quietly pinning the pre-existing number: a ratchet whose
// author exempts their own contribution is not a ratchet, and the first
// exemption is the one that teaches everyone else the number is negotiable.
//
// Raising it is therefore a NORMAL and expected part of adding a guard, and it
// is not the same act as raising it to accommodate a new document or fixture.
// The diff shows which one you did.
const MAX_SERVED_BUT_NOT_NEEDED = 84;

describe('the served surface is a closed set', () => {
  it('CAN RUN: the corpus and the launcher both loaded', () => {
    // Floors before any verdict. A zero-length corpus and a clean bill of
    // health are the same green, and this branch has shipped that green twice.
    assert.ok(
      tracked.length > 100,
      `CANNOT RUN: git ls-tree returned ${tracked.length} files under ${DASHBOARD}. ` +
        'Expected >100. The path is wrong or HEAD is not what this test assumes, ' +
        'and every assertion below would pass against an empty corpus.',
    );
    assert.ok(
      launcher.includes('--demo-assets-dir'),
      'CANNOT RUN: run-demo.sh no longer passes --demo-assets-dir. The premise of ' +
        'this suite is that the launcher hands the server a directory; re-measure ' +
        'the served surface before trusting anything below.',
    );
  });

  it('the launcher still serves this entire directory', () => {
    // The premise, asserted rather than assumed. If someone narrows the assets
    // dir -- which is the real fix -- this test fails, and that failure is the
    // instruction to re-measure and drop the ratchet, not to delete the check.
    // Non-comment lines only. run-demo.sh DISCUSSES the flag in prose at :218
    // ("--demo-assets-dir is passed explicitly"), and a whole-file regex
    // happily matched that sentence as if it were an invocation -- caught by
    // this suite's own first run. A scanner that cannot tell a flag from a
    // sentence about a flag will eventually accept the sentence as the fix.
    const served = launcher
      .split('\n')
      .filter((line) => !/^\s*#/.test(line))
      .join('\n')
      .match(/--demo-assets-dir\s+"?([^"\s\\]+)"?/g) ?? [];

    assert.ok(
      served.length >= 2,
      `Expected both servers to be launched with --demo-assets-dir; found ` +
        `${served.length} occurrence(s). If the launcher changed shape, the ` +
        'served surface changed with it.',
    );
    // EXACT value, not `includes('SCRIPT_DIR')`. @732c7548 caught themselves
    // asserting `line.includes('dashboard/')` on a branch whose demo lives in
    // `examples/serving-dashboard/` -- so the broken path SATISFIED the guard
    // by containing its own name. I wrote the identical defect here and it
    // survived until mutation M3: narrowing the flag to `${SCRIPT_DIR}/public`
    // -- the actual fix this suite is asking for -- still contains the
    // substring `SCRIPT_DIR`, so a containment check calls the repair a
    // regression-free no-op and never tells anyone the surface shrank.
    const SERVES_WHOLE_DIRECTORY = /^--demo-assets-dir\s+"\$\{SCRIPT_DIR\}"$/;

    assert.ok(
      served.every((flag) => SERVES_WHOLE_DIRECTORY.test(flag)),
      'run-demo.sh no longer serves ${SCRIPT_DIR}.\n' +
        'If the assets directory was NARROWED, that is the fix this suite has ' +
        'been asking for -- re-run the HTTP sweep in the header comment, then ' +
        `lower MAX_SERVED_BUT_NOT_NEEDED from ${MAX_SERVED_BUT_NOT_NEEDED} to the new count.\n` +
        `FOUND: ${served.join(', ')}`,
    );
  });

  it('every tracked file is claimed by exactly one declared class', () => {
    // THE ESCAPE-FILE GUARD, and the reason this file exists.
    //
    // A coverage list cannot be tested from inside its own corpus. It CAN be
    // tested from the tree: add a file, and if the list does not notice, the
    // list is decorative. Reproduce with
    //     touch examples/serving-dashboard/escape.xyz && git add -N escape.xyz
    // -- except that ls-tree reads HEAD, so it must actually be committed to
    // bite, which is the correct threshold: an uncommitted file is not yet a
    // property of the branch.
    assert.deepEqual(
      unclassified.map((entry) => entry.file),
      [],
      'These tracked files are served at /demo/ and belong to no declared class.\n' +
        'That is not a formatting complaint: an unclassified file is one nobody ' +
        'decided to publish, sitting inside the origin a visitor points a browser at.\n' +
        'Give each one a class in CLASSES -- and if the honest class is ' +
        '"should not be reachable at all", say so in the review rather than ' +
        'inventing a bucket to silence this.\n' +
        `FOUND:\n  ${unclassified.map((e) => e.file).join('\n  ')}`,
    );
  });

  it('the classifier is not just answering PAGE_ASSET to everything', () => {
    // Anti-vacuity with teeth. The previous test passes trivially if some rule
    // grows broad enough to swallow the tree, and a single over-wide regex
    // would do it silently. So: every declared class must actually be
    // populated, which makes an over-broad rule starve its neighbours and fail
    // HERE rather than quietly disabling the guard above.
    const empty = CLASSES.map((c) => c.name).filter(
      (name) => !classified.some((entry) => entry.className === name),
    );

    assert.deepEqual(
      empty,
      [],
      'These declared classes matched nothing, which means a rule above them ' +
        'has grown wide enough to swallow their files. An over-broad rule makes ' +
        'the escape-file guard vacuous without ever turning it red.\n' +
        `STARVED: ${empty.join(', ')}`,
    );
  });

  it('the exposure ratchet has not been loosened', () => {
    const count = servedButNotNeeded.length;
    const byClass = CLASSES.map((c) => c.name)
      .map((name) => [name, servedButNotNeeded.filter((e) => e.className === name).length])
      .filter(([, n]) => n > 0)
      .map(([name, n]) => `${name} ${n}`)
      .join(' · ');

    assert.ok(
      count <= MAX_SERVED_BUT_NOT_NEEDED,
      `${count} tracked files are fetchable at /demo/ that the page never loads ` +
        `(was ${MAX_SERVED_BUT_NOT_NEEDED}).\n` +
        `BY CLASS: ${byClass}\n` +
        'Something new was added inside the served directory. That is allowed -- ' +
        'but it is a publishing decision, not a file-layout one, so make it ' +
        'deliberately: either move it outside the assets dir, or raise this ' +
        'number in the same commit with a sentence saying why a visitor may have it.',
    );
  });
});
