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

import { assertShippingTree, shipped, shippedPaths } from './shipping-tree.mjs';

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

// ---------------------------------------------------------------------------
// The `check-` prefix: what it means, and why it is not being applied by rename
// ---------------------------------------------------------------------------
//
// THE COMPLAINT THIS ANSWERS. Some suites are `check-foo.test.js` and some are
// `foo.test.js`, and a newcomer cannot infer which they should write. An
// unstated convention is worse than either consistent alternative: it looks
// meaningful, so readers infer a meaning, and the meaning they infer is wrong.
//
// WHAT THE PREFIX ACTUALLY MEANS, MEASURED RATHER THAN ASSUMED.
// Every `*.test.js` at the shipping ref was classified by what it DOES:
//
//   check- prefixed, reads repo artefacts .......... 14   (all of them)
//   check- prefixed, pure unit test ................  0
//   unprefixed, pure unit test .................... 18   (all unprefixed)
//   unprefixed, reads repo artefacts .............. 33   <- the drift
//
// So the prefix is already 100% reliable in ONE direction — `check-` always
// means "this reads the repository as data: sources, docs, the launcher, the
// committed tree" — and it is silent in the other. That one-directional rule is
// real, is load-bearing, and is what this file now states and enforces.
//
// The prefix carries NO functional meaning: `run-tests.sh` and `package.json`
// were both searched and neither selects on it. Nothing dispatches differently.
//
// WHY THIS IS NOT A RENAME COMMIT.
// Making the corpus consistent means renaming 33 files. This repo has already
// weighed that trade once, four exemption-comments above, and declined it: the
// review documents cite these filenames, and renaming mid-review converts
// correct citations into stale ones inside documents their authors are actively
// reading. Thirty-three renames is that hazard at eleven times the size, landed
// while three reviewers hold open findings.
//
// So the hazard is frozen instead. The 33 are grandfathered by exact set, which
// means a NEW unprefixed scanner cannot be added without failing this test —
// the corpus converges on the convention as files are added, and no citation
// anywhere rots today. The list shrinks by rename, one file at a time, whenever
// somebody is already touching a file for another reason.

/**
 * Evidence that a suite reads the repository as data rather than importing a
 * module and exercising it.
 *
 * Deliberately broad. A false POSITIVE here costs a rename argument; a false
 * negative silently lets a scanner ship unprefixed, which is the drift being
 * frozen. Broad-and-noisy is the safe direction for a ratchet.
 */
const SCANNER_EVIDENCE = /node:fs|node:child_process|shipping-tree|readFileSync|execFileSync/;

/**
 * Unprefixed suites that read repo artefacts, as committed.
 *
 * EXACT SET, not a count. A count would let one file be renamed and another
 * added without notice, which is precisely the quiet drift this pins.
 *
 * RETIREMENT: rename `foo.test.js` -> `check-foo.test.js`, update every
 * consumer named below, and delete the line here in the SAME commit. The
 * self-expiry test below fails if an entry outlives the file it names.
 *
 * ⚠️ READ THE CONSUMER INVENTORY BEFORE RENAMING ANY OF THESE. It was measured
 * across the whole tree, and the first version of this comment understated it
 * badly by mentioning only the review documents:
 *
 *   all 33 are referenced BY NAME somewhere      33 / 33
 *   code references (js/mjs/sh)                  ~79, between 1 and 8 per file
 *   documentation references (md)                ~101, up to 7 per file
 *
 * Most are prose cross-references in comments, which cost a reader nothing at
 * runtime and everything in trust. But at least one is a FUNCTIONAL dependency
 * on the filename:
 *
 *   check-binding-liveness.test.js:531 does
 *     readFileSync(join(HERE, 'dashboard', 'field-keys.test.js'))
 *   to LIFT NOT_YET_PUBLISHED rather than keep a second copy of it.
 *
 * That call is deliberately unguarded, so a rename throws ENOENT and the suite
 * fails loudly. That is the good case and it must STAY the good case: if you
 * ever wrap a by-name read like that in a try/catch with a fallback, a rename
 * would silently hand the guard an empty inventory and every suite would stay
 * green while the check did nothing.
 *
 * Discovery itself is rename-safe and was verified: run-tests.sh:136 finds
 * tests with `find . -name '*.test.js'`, which is recursive and matches on the
 * suffix, and run-tests.sh:378 ratchets on ANY decrease in the discovered
 * count. Nothing anywhere selects tests on the `check-` prefix. So a rename
 * cannot drop a file out of the run — it can only break a reference, loudly.
 */
const GRANDFATHERED_UNPREFIXED_SCANNERS = Object.freeze([
  'asset-graph.test.js',
  'backstop-reach.test.js',
  'caption-catalogue.test.js',
  'dashboard/field-keys.test.js',
  'dashboard/honesty.test.js',
  'dashboard/model-path-disclosure.test.js',
  'dashboard/panel-kit.test.js',
  'dashboard/panels.test.js',
  'dashboard/registry-prefix-tripwire.test.js',
  'dashboard/registry.test.js',
  'dashboard/scheduling.test.js',
  'dashboard/staleness.test.js',
  'dashboard/stylesheet.test.js',
  'denominator-binding.test.js',
  'fetch-chokepoint.test.js',
  'never-bind.test.js',
  'page-claims.test.js',
  'prefix-counters-forbidden.test.js',
  'provenance-expiry.test.js',
  'register-completeness.test.js',
  'repair-citations.test.js',
  'request-deadline.test.js',
  'run-tests-guards.test.js',
  'scenario-origins.test.js',
  'scenario-routes.test.js',
  'scenario-switcher.test.js',
  'served-surface-rendered.test.js',
  'served-surface.test.js',
  'shipping-tree.test.js',
  'state-channel.test.js',
  'state-treatments.test.js',
  'telemetry-field.test.js',
  'telemetry-key-namespace.test.js',
]);

/** Every shipped test path, tagged with whether it scans and whether it is prefixed. */
function classifiedTests() {
  return shippedPaths()
    .filter((path) => path.endsWith('.test.js'))
    .map((path) => ({
      path,
      prefixed: basename(path).startsWith('check-'),
      scans: SCANNER_EVIDENCE.test(shipped(path)),
    }));
}

describe('the check- test-file prefix', () => {
  it('THE RULE: a check- prefixed suite reads the repository as data', () => {
    // State it once, here, and enforce it. This is the direction of the rule
    // that is true today with zero renames, so it can be enforced immediately
    // rather than aspirationally.
    const offenders = classifiedTests()
      .filter((t) => t.prefixed && !t.scans)
      .map((t) => `  ${t.path}`);

    assert.deepEqual(
      offenders,
      [],
      'These suites carry the `check-` prefix but do not read any repository ' +
        `artefact:\n${offenders.join('\n')}\n\n` +
        'In this repo `check-` means "audits the repository as data" — sources, ' +
        'docs, the launcher, the committed tree. A unit test that imports a ' +
        'module and exercises it is named after the module, with no prefix.\n' +
        'Either drop the prefix, or if it really is an audit, make it read the ' +
        'artefact it audits.',
    );
  });

  it('THE RULE, other half: a pure unit test is not named check-', () => {
    // The contrapositive, asserted separately so the failure message can say
    // which mistake was made rather than making the reader work it out.
    const misnamed = classifiedTests()
      .filter((t) => t.prefixed && !t.scans)
      .map((t) => t.path);

    assert.equal(
      misnamed.length,
      0,
      `A suite named check-* must audit repo artefacts: ${misnamed.join(', ')}`,
    );

    // And the population that gives the rule its meaning must be non-empty.
    const prefixed = classifiedTests().filter((t) => t.prefixed);
    assert.ok(
      prefixed.length >= 10,
      `Only ${prefixed.length} check- prefixed suites found; there were 14 when ` +
        'this rule was written. A rule about a population of zero is not a rule.',
    );
  });

  it('no NEW unprefixed scanner may be added — the drift is frozen, not blessed', () => {
    const actual = classifiedTests()
      .filter((t) => !t.prefixed && t.scans)
      .map((t) => t.path)
      .sort();

    const expected = [...GRANDFATHERED_UNPREFIXED_SCANNERS].sort();
    const added = actual.filter((p) => !expected.includes(p));
    const removed = expected.filter((p) => !actual.includes(p));

    assert.deepEqual(
      added,
      [],
      'New suites read repository artefacts but are not named `check-`:\n' +
        `${added.map((p) => `  ${p}`).join('\n')}\n\n` +
        'Rename them to `check-<name>.test.js`. The 33 unprefixed scanners ' +
        'already here are grandfathered ONLY because renaming them mid-review ' +
        'would rot live citations in the review documents — that reprieve does ' +
        'not extend to files being written now.',
    );

    assert.deepEqual(
      removed,
      [],
      'GRANDFATHERED_UNPREFIXED_SCANNERS names files that are no longer ' +
        `unprefixed scanners:\n${removed.map((p) => `  ${p}`).join('\n')}\n\n` +
        'If you renamed one, delete its line here in the same commit. An ' +
        'exemption that outlives its subject silently exempts the next file ' +
        'that takes the name.',
    );
  });

  it('every by-name reference to a test file resolves to a real file', () => {
    // THE LEAD'S PRECONDITION, ENFORCED RATHER THAN DOCUMENTED.
    //
    // Some guards depend on another suite's FILENAME, not just its behaviour.
    // The load-bearing example is check-binding-liveness.test.js, which lifts
    // NOT_YET_PUBLISHED straight out of dashboard/field-keys.test.js instead of
    // keeping a second copy that could drift.
    //
    // A rename would break that. Today it breaks LOUDLY, because the read is an
    // unguarded readFileSync that throws ENOENT. But that is a property of the
    // consumer's error handling, and the next author may wrap it in a try/catch
    // with a fallback — at which point a rename silently hands the guard an
    // empty inventory and every suite stays green while the check does nothing.
    //
    // So this does not inspect error handling, which would be fragile and would
    // have to be re-tuned for every consumer. It asserts the target EXISTS.
    // That holds however the consumer behaves when the file is missing, which
    // makes it the stronger invariant of the two.
    const dangling = [];
    const seen = [];

    for (const path of shippedPaths()) {
      if (!/\.(?:js|mjs)$/.test(path)) continue;
      const source = shipped(path);

      for (const line of source.split('\n')) {
        // Only lines that actually READ a file by name. A prose cross-reference
        // in a comment is not a functional dependency and must not be policed
        // here — comments are how this codebase explains itself.
        if (!/readFileSync|readFile\(|createReadStream/.test(line)) continue;
        if (/^\s*(?:\/\/|\*)/.test(line)) continue;

        for (const [, name] of line.matchAll(/'([^']*\.test\.js)'/g)) {
          seen.push(`${path} -> ${name}`);
          const base = name.includes('/') ? name.slice(name.lastIndexOf('/') + 1) : name;
          const exists = shippedPaths().some(
            (p) => p === name || p.endsWith(`/${base}`) || p === base,
          );
          if (!exists) dangling.push(`${path} reads '${name}', which is not shipped`);
        }
      }
    }

    assert.deepEqual(
      dangling,
      [],
      'A guard reads another test file BY NAME, and that file is not at the ' +
        `shipping ref:\n${dangling.map((d) => `  ${d}`).join('\n')}\n\n` +
        'Renaming a test that another guard reads disarms that guard. Update ' +
        'the reader in the same commit as the rename. Do NOT make the read ' +
        'tolerant of a missing file — a guard that falls back to an empty ' +
        'inventory reports a confident green having checked nothing.',
    );

    // ANTI-VACUITY. If the line filter stopped matching, `dangling` is empty and
    // this test passes while policing nothing. The known dependency must be seen.
    assert.ok(
      seen.some((s) => s.includes('field-keys.test.js')),
      'The scan did not observe check-binding-liveness.test.js reading ' +
        "dashboard/field-keys.test.js, which it does at line 531. The extractor " +
        'is not reading the corpus it claims to read, so an empty result means ' +
        'nothing.\n' +
        `OBSERVED: ${seen.join(', ') || 'nothing at all'}`,
    );
  });

  it('the classifier can say both YES and NO', () => {
    // ANTI-VACUITY. Every assertion above is satisfied if `scans` is constant.
    // If SCANNER_EVIDENCE never matched, test one passes trivially and the
    // ratchet reports an empty set as agreement with a 33-entry list — which
    // would fail loudly, but only by luck rather than by design. Assert the
    // instrument's discrimination directly.
    const all = classifiedTests();

    assert.ok(
      all.length >= 50,
      `Only ${all.length} test files at the shipping ref; the census is broken.`,
    );

    const scanners = all.filter((t) => t.scans);
    const units = all.filter((t) => !t.scans);

    assert.ok(
      scanners.length >= 20 && units.length >= 10,
      `Classifier produced ${scanners.length} scanners and ${units.length} unit ` +
        'tests. It was 47 and 18 when written. A lopsided split means the ' +
        'evidence pattern stopped discriminating, not that the corpus changed.',
    );

    // POSITIVE CONTROL: a known scanner is recognised.
    assert.ok(
      all.find((t) => t.path === 'check-source-citations.test.js')?.scans,
      'check-source-citations.test.js reads the Rust sources and must classify ' +
        'as a scanner. The instrument cannot see what it is pointed at.',
    );

    // NEGATIVE CONTROL: a known pure unit test is NOT recognised as a scanner.
    assert.equal(
      all.find((t) => t.path === 'dashboard/sparkline.test.js')?.scans,
      false,
      'dashboard/sparkline.test.js imports a module and exercises it; if it ' +
        'classifies as a scanner then SCANNER_EVIDENCE matches everything and ' +
        'the prefix rule is enforcing nothing.',
    );
  });
});
