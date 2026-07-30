// Test-file basenames must be unique across this directory tree.
//
// WHY THIS EXISTS
// ---------------
// `run-tests.sh` has printed this WARN on every single run for hours:
//
//   WARN: the same test filename appears in more than one directory:
//         scenario-switcher.test.js
//           ./scenario-switcher.test.js
//           ./ui/scenario-switcher.test.js
//         Not fatal. But a glob that reaches one copy and not the other
//         reports a stable total whose meaning silently differs.
//
// It was correct, it was unowned, and everybody read past it — including the
// crew that spent an hour reconciling four different suite totals. A warning
// that nobody acts on is indistinguishable from no warning at all, so the
// invariant is moved here where it can go RED.
//
// WHAT A DUPLICATE BASENAME ACTUALLY COSTS
// ----------------------------------------
// Not the totals. The totals diverged because `'…/**/*.js'` is a NARROWER
// pathspec than `'…/*.js'` — a separate defect, separately filed, and renaming
// files would not have fixed it.
//
// The cost is attribution. Node prints failures by basename. When
// `scenario-switcher.test.js` goes red, two files could have produced it, and
// they are owned by different authors, cover different exports and live in
// different directories. On a branch where fourteen agents each need to answer
// "is this red mine?", an ambiguous failure name is a coordination tax paid on
// every run, by everyone, forever.
//
// WHY THIS SHIPS AS A GUARD-PLUS-EXEMPTION RATHER THAN AS A RENAME
// ---------------------------------------------------------------
// The rename is the right fix and it is deliberately NOT in this commit.
// Four review documents cite these two filenames in eleven places, including a
// table in READABILITY-REVIEW.md that correctly distinguishes them by count and
// by subject. Renaming during review would convert eleven CORRECT citations
// into stale ones, in documents this author does not own, while reviewers are
// reading them — which is precisely the coordinate-rot failure this crew has
// spent the whole session cataloguing. Creating that on purpose, to fix a
// naming wart, is a bad trade.
//
// So: the hazard is frozen at exactly its current size (the guard reddens on
// any NEW duplicate) and the known pair is carried as a named exemption with a
// retirement predicate. The exemption is self-expiring — see the second test.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { basename } from 'node:path';

import { assertShippingTree, shippedPaths } from './shipping-tree.mjs';

assertShippingTree();

/**
 * Duplicate basenames that are known, deliberate, and not to be "fixed" today.
 *
 * Each entry must name every path that shares the basename and say why the
 * duplication is tolerated. An entry that stops describing a real duplicate is
 * a FAILURE, not a no-op — see 'an exemption that no longer describes a real
 * duplicate must be deleted'.
 */
const KNOWN_DUPLICATES = [
  {
    basename: 'scenario-switcher.test.js',
    paths: ['scenario-switcher.test.js', 'ui/scenario-switcher.test.js'],
    // Both files are real and neither is a stale copy. They cover two different
    // exports of one module, `ui/scenario-switcher.js`:
    //
    //   scenario-switcher.test.js     10 tests, `describeSwitcher` + planScenario
    //                                 — reachability across peer servers.
    //   ui/scenario-switcher.test.js   5 tests, `mountScenarioSwitcher`
    //                                 — the substitution notice shown when a
    //                                   visitor asks for a scenario nobody serves.
    //
    // DELETING EITHER DELETES COVERAGE. The fix is a rename, not a removal.
    //
    // RETIREMENT PREDICATE — do this in ONE commit, after the review closes:
    //   1. git mv ui/scenario-switcher.test.js ui/scenario-switcher-mount.test.js
    //   2. update the citations in READABILITY-REVIEW.md, REVIEWER-BRIEF.md,
    //      IMPLEMENTATION-REVIEW.md and ARCHITECTURE-SECURITY-REVIEW.md
    //   3. delete this entry
    // Step 3 is not optional and is not left to memory: the self-expiry test
    // below fails if the entry outlives the duplicate it describes.
    reason: 'two real suites covering two exports of one module; renaming mid-review would rot 11 live citations',
  },
];

const EXEMPT_BASENAMES = new Set(KNOWN_DUPLICATES.map((entry) => entry.basename));

/** Every shipped `*.test.js` under this directory, grouped by basename. */
function testFilesByBasename() {
  const groups = new Map();

  for (const path of shippedPaths()) {
    if (!path.endsWith('.test.js')) continue;
    const name = basename(path);
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(path);
  }

  return groups;
}

describe('test-file basenames', () => {
  it('no two test files share a basename, except by named exemption', () => {
    const groups = testFilesByBasename();

    const offenders = [...groups.entries()]
      .filter(([name, paths]) => paths.length > 1 && !EXEMPT_BASENAMES.has(name))
      .map(([name, paths]) => `  ${name}\n${paths.map((p) => `    ./${p}`).join('\n')}`);

    assert.equal(
      offenders.length,
      0,
      'Two test files share a basename, so a failure report naming that basename ' +
        'does not say which file failed:\n' +
        `${offenders.join('\n')}\n\n` +
        'Rename one of them. Do NOT delete either without reading both first — ' +
        'the existing duplicate on this branch is two real suites covering two ' +
        'different exports, and deleting one would have deleted live coverage.\n' +
        'If the duplication is deliberate, add it to KNOWN_DUPLICATES in this ' +
        'file WITH a retirement predicate.',
    );
  });

  it('an exemption that no longer describes a real duplicate must be deleted', () => {
    // WHY THIS TEST IS THE IMPORTANT HALF.
    //
    // An exemption list with no expiry is how a temporary tolerance becomes
    // permanent: the entry keeps passing, nobody re-reads it, and the guard
    // quietly stops covering the thing it names. This test makes the exemption
    // cost something to keep — once the rename lands, the entry goes RED and
    // must be removed in the same commit.
    //
    // It also catches the likelier accident: somebody renames one file, the
    // duplicate disappears, and the stale entry silently keeps a FUTURE
    // collision on that basename exempt.
    const groups = testFilesByBasename();

    for (const entry of KNOWN_DUPLICATES) {
      const actual = groups.get(entry.basename) ?? [];

      assert.ok(
        actual.length > 1,
        `KNOWN_DUPLICATES still exempts '${entry.basename}', but that basename ` +
          `is no longer duplicated (${actual.length} file(s) at the shipping ref: ` +
          `${actual.map((p) => `./${p}`).join(', ') || 'none'}).\n` +
          'The duplication was fixed. DELETE THE ENTRY — an exemption that ' +
          'outlives its subject silently exempts the next collision.',
      );

      assert.deepEqual(
        [...actual].sort(),
        [...entry.paths].sort(),
        `KNOWN_DUPLICATES lists the wrong paths for '${entry.basename}'.\n` +
          `  exemption names: ${entry.paths.join(', ')}\n` +
          `  actually shipped: ${actual.join(', ')}\n` +
          'A third copy may have appeared, or one may have moved. Either way the ' +
          'exemption is describing a tree that no longer exists — re-read both ' +
          'files before widening it.',
      );

      assert.ok(
        entry.reason?.trim(),
        `KNOWN_DUPLICATES entry '${entry.basename}' has no reason. An exemption ` +
          'without a stated reason cannot be evaluated by the next reader, who ' +
          'will assume somebody thought about it.',
      );
    }
  });

  it('the census actually reaches the test corpus', () => {
    // ANTI-VACUITY. Every assertion above is satisfied by an EMPTY corpus, and
    // an empty corpus is exactly what the first implementation of shippedPaths()
    // produced: `git ls-tree -- '*.test.js'` returns nothing and exits 0,
    // because ls-tree does not glob-match a pathspec the way ls-files does. A
    // negative control did not catch it — the control also returned 0, and 0
    // equals 0 reads as agreement.
    //
    // So the floor is asserted against a number an independent tool can confirm
    // (`git ls-files -- '*.test.js'`), and the instrument is required to be able
    // to say YES, not merely to say no.
    const groups = testFilesByBasename();
    const total = [...groups.values()].reduce((n, paths) => n + paths.length, 0);

    assert.ok(
      total >= 50,
      `Only ${total} test files discovered at the shipping ref. This suite had ` +
        '55 when this guard was written, so a number this low means the census ' +
        'is broken, not that the tests were deleted. A basename check over an ' +
        'empty corpus passes perfectly and proves nothing.',
    );

    // POSITIVE CONTROL: the instrument can find a file it must find. Without
    // this, a filter that silently matched nothing would still satisfy the
    // floor if the floor were ever lowered.
    assert.ok(
      groups.has('scenario-switcher.test.js'),
      'The census cannot see scenario-switcher.test.js, which is tracked at the ' +
        'shipping ref. The instrument is not reading the corpus it claims to read.',
    );

    // The census must span more than one directory, or a duplicate-basename
    // check is structurally incapable of ever firing.
    const directories = new Set(
      [...groups.values()].flat().map((p) => (p.includes('/') ? p.slice(0, p.lastIndexOf('/')) : '.')),
    );
    assert.ok(
      directories.size >= 2,
      `Test files were found in only ${directories.size} directory. A basename ` +
        'collision requires two directories, so this check could never fail — ' +
        'which means it is not measuring what it claims to measure.',
    );
  });
});
