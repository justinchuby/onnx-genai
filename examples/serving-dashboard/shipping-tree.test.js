/**
 * shipping-tree.test.js — the guard on the module every other guard imports.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * Ten check files read their inputs through `shipped()`. If that one function
 * reads the wrong tree, ten guards report confident, detailed, internally
 * consistent results about an artefact nobody ships — and every one of them
 * stays green while doing it. This module is the single point where a whole
 * category of false green can be introduced, and it had no tests.
 *
 * WHAT IS ACTUALLY BEING ASSERTED HERE, AND WHY IT IS NOT "DOES IT READ FILES"
 * ---------------------------------------------------------------------------
 * The interesting property is not that `shipped()` returns bytes. It is that
 * every call in one process returns bytes FROM THE SAME COMMIT. On this branch
 * HEAD moves every twenty to forty seconds, so a module that spells the literal
 * string 'HEAD' in its `git show` reads a different tree on each call and
 * presents the mixture as one measurement. Each individual read is correct;
 * what is destroyed is comparability, which is the only thing a cross-file
 * check has.
 *
 * That defect is invisible to any test that calls `shipped()` twice and checks
 * the bytes, because in a quiet tree the answers agree. It is only visible in
 * the RESOLVED REF, which is why these tests assert on `SHIPPING_REF` directly
 * rather than on file contents.
 */

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';

import {
  DIVERGENCE_PHRASES,
  SHIPPING_BRANCH,
  SHIPPING_REF,
  STASH_PHRASE,
  countStashEntries,
  describeTree,
  divergenceReport,
  divergenceSummary,
  divergentPaths,
  matchStashedNames,
  parsePorcelain,
  stashLines,
  stashSuffix,
  shipped,
  stashedPaths,
} from './shipping-tree.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));

/**
 * Load `shipping-tree.mjs` in a FRESH process with a given environment.
 *
 * `SHIPPING_REF` is resolved once at module load, deliberately — so it cannot
 * be re-read in-process, and a test that tried would be testing a mock rather
 * than the mechanism. A subprocess is the only honest way to observe load-time
 * behaviour, and it is also exactly how a reviewer invokes it.
 *
 * @param {Record<string, string>} env extra variables for the child.
 * @returns {{ status: number, stdout: string, stderr: string }}
 */
function probe(env, script) {
  try {
    const stdout = execFileSync(process.execPath, ['-e', script], {
      cwd: HERE,
      encoding: 'utf8',
      env: { ...process.env, ...env },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    return { status: 0, stdout, stderr: '' };
  } catch (error) {
    return {
      status: error.status ?? 1,
      stdout: error.stdout?.toString() ?? '',
      stderr: error.stderr?.toString() ?? '',
    };
  }
}

const PRINT_REF = `import('./shipping-tree.mjs').then(m => console.log(m.SHIPPING_REF));`;

function git(...args) {
  return execFileSync('git', args, { cwd: HERE, encoding: 'utf8' }).trim();
}

describe('the commit every check reads is fixed for the whole process', () => {
  it('resolves to an immutable sha, never the literal pointer HEAD', () => {
    // THE test. A pointer cannot be compared across calls; a sha can.
    assert.notEqual(SHIPPING_REF, 'HEAD');
    assert.match(
      SHIPPING_REF,
      /^[0-9a-f]{40}$/,
      'SHIPPING_REF must be a full sha. A short sha is ambiguous across ' +
        'repositories and a symbolic ref can move mid-run, which is the ' +
        'defect this constant exists to remove.',
    );
  });

  it('two loads at the same HEAD agree, and the answer is that HEAD', () => {
    const head = git('rev-parse', 'HEAD');
    assert.equal(SHIPPING_REF, head, 'with no override, the pinned ref IS HEAD');

    const first = probe({}, PRINT_REF);
    assert.equal(first.status, 0, first.stderr);
    assert.equal(first.stdout.trim(), head);
  });

  it('reads bytes from the pinned commit, not from the working tree', () => {
    // Positive control FIRST: prove the reader reaches a real corpus at all,
    // so the assertions below cannot pass by reading nothing.
    const own = shipped('shipping-tree.mjs');
    assert.ok(own.length > 2000, `expected a substantial file, got ${own.length} bytes`);
    assert.ok(own.includes('export function shipped'));

    // The committed bytes must equal `git show <ref>:` for the same ref --
    // which is a different question from "equals the file on disk", and it is
    // the question that stays correct while somebody else edits the desk.
    const viaGit = execFileSync('git', ['show', `${SHIPPING_REF}:./shipping-tree.mjs`], {
      cwd: HERE,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
    });
    assert.equal(own, viaGit);
  });

  it('an absent path throws instead of returning an empty string', () => {
    // Every content check in this directory scores '' as clean, so a reader
    // that swallowed a missing file would make deletion the easiest way to go
    // green. Negative control for the positive one above.
    assert.throws(() => shipped('no-such-file-zz.js'));
  });
});

describe('a reviewer can pin the checks to a named tag', () => {
  it('honours SHIPPING_TREE_REF and reports that it was overridden', () => {
    // Use this repo's own HEAD~1: guaranteed to exist, guaranteed to be on the
    // shipping branch, and it does not depend on any tag existing locally.
    const previous = git('rev-parse', 'HEAD~1');
    const result = probe({ SHIPPING_TREE_REF: 'HEAD~1' }, PRINT_REF);

    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout.trim(), previous);
    assert.notEqual(
      result.stdout.trim(),
      SHIPPING_REF,
      'the override must actually change the ref, or this test proves nothing',
    );
  });

  // REVIEW_SHA is the spelling that was broadcast to the reviewers. Before
  // these tests it was read by nothing: setting it produced no ref change, no
  // warning and no error, so a reviewer would have scored a moving branch while
  // believing they were pinned to a tag. A silent no-op is the failure mode
  // worth a test, because unlike an unsupported variable it is TRUSTED.
  it('honours REVIEW_SHA, the name reviewers were actually given', () => {
    const previous = git('rev-parse', 'HEAD~1');
    const result = probe({ REVIEW_SHA: 'HEAD~1' }, PRINT_REF);

    assert.equal(result.status, 0, result.stderr);
    assert.equal(
      result.stdout.trim(),
      previous,
      'REVIEW_SHA was ignored — a reviewer setting it gets a silent no-op',
    );
    assert.notEqual(result.stdout.trim(), SHIPPING_REF);
  });

  it('accepts both spellings when they agree', () => {
    const previous = git('rev-parse', 'HEAD~1');
    const result = probe(
      { SHIPPING_TREE_REF: 'HEAD~1', REVIEW_SHA: previous },
      PRINT_REF,
    );

    // Deliberately spelled differently -- a symbolic ref and a raw SHA naming
    // one commit. Agreement is about the COMMIT, not about the string.
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout.trim(), previous);
  });

  it('refuses two spellings that name different commits', () => {
    const result = probe(
      { SHIPPING_TREE_REF: 'HEAD', REVIEW_SHA: 'HEAD~1' },
      PRINT_REF,
    );

    assert.notEqual(
      result.status,
      0,
      'two conflicting explicit instructions must not be silently reconciled',
    );
    assert.match(result.stderr, /SHIPPING_TREE_REF/);
    assert.match(result.stderr, /REVIEW_SHA/, 'the error must name BOTH variables');
  });

  it('a ref that cannot be resolved fails at load, naming itself', () => {    const result = probe({ SHIPPING_TREE_REF: 'zz-no-such-ref' }, PRINT_REF);

    assert.notEqual(result.status, 0, 'an unresolvable ref must not be ignored');
    assert.match(result.stderr, /zz-no-such-ref/);
    assert.match(
      result.stderr,
      /SHIPPING_TREE_REF/,
      'the error must name the variable, or the reader cannot act on it',
    );
  });

  it('rejects a pinned ref that is not on the shipping branch', () => {
    // The hole the override opens: HEAD is fine, so the standing provenance
    // check passes, while the bytes come from somewhere else entirely.
    // Build a commit that is genuinely off-branch to prove the check reaches.
    const orphan = execFileSync(
      'git',
      ['commit-tree', `${SHIPPING_REF}^{tree}`, '-m', 'off-branch probe'],
      { cwd: HERE, encoding: 'utf8', env: { ...process.env, GIT_AUTHOR_NAME: 'probe', GIT_AUTHOR_EMAIL: 'probe@example.invalid', GIT_COMMITTER_NAME: 'probe', GIT_COMMITTER_EMAIL: 'probe@example.invalid' } },
    ).trim();

    const result = probe(
      { SHIPPING_TREE_REF: orphan },
      `import('./shipping-tree.mjs').then(m => { m.assertShippingTree(); console.log('PASSED'); });`,
    );

    assert.notEqual(result.status, 0, 'an off-branch ref must be refused');
    assert.ok(!result.stdout.includes('PASSED'));
    assert.match(result.stderr, new RegExp(SHIPPING_BRANCH.replace(/[/\\]/g, '\\$&')));
  });

  it('the same check PASSES on an on-branch ref — the positive control', () => {
    // Without this, the test above is satisfied by any failure whatsoever,
    // including a typo that breaks the module for every ref equally.
    const result = probe(
      { SHIPPING_TREE_REF: 'HEAD~1' },
      `import('./shipping-tree.mjs').then(m => { m.assertShippingTree(); console.log('PASSED'); });`,
    );

    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /PASSED/);
  });
});

describe('a failure report names the tree it actually scored', () => {
  it('describeTree exposes the read ref alongside HEAD', () => {
    const tree = describeTree();
    assert.equal(tree.ref, SHIPPING_REF, 'the reported ref is the pinned one');
    assert.equal(tree.refIsOverridden, false, 'no override is set in this process');

    // NOT asserted: that `ref` and `head` are the same commit.
    //
    // The first draft of this test asserted `SHIPPING_REF.startsWith(tree.head)`
    // and it FAILED — because `head` is resolved live, inside describeTree(),
    // while `ref` was pinned when the module loaded, and another agent
    // committed in between. That is not a bug in either value. It is this
    // module's entire premise arriving in its own test: a live pointer and a
    // pinned commit disagree precisely when the branch is moving, which is
    // always, and encoding "they match" would re-introduce the assumption the
    // pinning exists to remove.
    //
    // What must hold is that both are real commits and the report shows both,
    // so a reader who sees a surprising result can tell which tree produced it.
    assert.match(tree.ref, /^[0-9a-f]{40}$/);
    assert.match(tree.head, /^[0-9a-f]{7,40}$/);
  });

  it('describeTree reports porcelain, the fourth fact it used to omit', () => {
    const tree = describeTree();

    // THE DEFECT THIS PINS: this module described toplevel, branch and SHA and
    // stopped. It was the shared instrument every other guard quoted its
    // provenance from, so the one missing fact was missing everywhere at once
    // -- and it is the fact that decides whether reading the working tree is
    // equivalent to reading HEAD or silently different from it.
    assert.ok(Array.isArray(tree.dirty), 'dirty must be a list, not a boolean');

    // CROSS-INSTRUMENT AGREEMENT. Not a re-implementation: this shells out
    // independently and compares, so the assertion fails if `describeTree`
    // starts reporting a stale, cached, or cwd-scoped answer. `cwd` is the
    // test file's own directory, which is the one thing describeTree()
    // deliberately does NOT use -- it resolves against the module's location.
    // Agreement across that difference is the point.
    const independent = execFileSync('git', ['status', '--porcelain'], {
      cwd: HERE,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
    })
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean);
    assert.deepEqual(tree.dirty, independent);

    // ANTI-VACUITY, AND IT IS THE HONEST KIND: this asserts the PARSE is real,
    // not that the tree is dirty. Every entry must carry a porcelain status
    // code and a path, so a field hardcoded to [] or to raw untrimmed lines
    // fails here the moment the tree is dirty at all.
    for (const entry of tree.dirty) {
      assert.match(entry, /^[A-Z?! ]{1,2}\s+\S/, `unparsed porcelain entry: ${entry}`);
    }

    // NOT DONE, DELIBERATELY: creating a scratch file to force `dirty`
    // non-empty. That is the textbook control and it is the wrong call HERE.
    // This suite runs in a worktree shared with a dozen other agents under a
    // commit freeze; a test that writes into it to prove a point can trip
    // somebody else's porcelain assertion, and the resulting red would be
    // attributed to their diff rather than to this test. A control that
    // corrupts the measurement it shares is not a control.
  });
});

describe('a check discloses which tree it actually read', () => {
  // The asymmetry this closes: `WORKTREE_DIVERGENCE` existed in three Python
  // scripts and in ZERO JavaScript. Every JS guard here reads the working tree
  // and none of them said so, which is not wrong but is UNFALSIFIABLE — nobody
  // reading the result later can tell which tree it described.
  //
  // Note what is NOT done: the guards are not repointed at `git show HEAD:`.
  // That would false-red the ordinary commit which adds a symbol and cites it
  // in the same change, and a guard that reddens on correct work gets deleted.
  // Disclose the tree you read; do not silently switch it.

  it('classifies the three ways a desk can disagree with a commit', () => {
    const d = parsePorcelain([' M edited.js', '?? brand-new.js', ' D removed.js'].join('\n'));

    assert.deepEqual(d.modified, ['edited.js']);
    assert.deepEqual(d.untracked, ['brand-new.js']);
    assert.deepEqual(d.deleted, ['removed.js']);
  });

  it('does not trim the status columns — a LEADING SPACE is data, not noise', () => {
    // The bug this pins. The shared `git()` helper in shipping-tree.mjs trims,
    // and reusing it here would have turned ` M path` into `M path`, shifting
    // every later offset by one. The status would still classify as modified,
    // so the buckets look right, and only the NAME is quietly wrong — it comes
    // back as `ath` and matches no file on earth. A green-looking corruption.
    const d = parsePorcelain(' M examples/kept.js');

    assert.deepEqual(d.modified, ['examples/kept.js'], 'the name must survive intact');
    assert.equal(d.modified[0].startsWith('e'), true, 'no leading character was eaten');
  });

  it('follows a rename to the name that now exists', () => {
    // `R  old -> new` in Python lands in `modified` under the literal string
    // "old -> new", which is not a path and matches nothing. Deliberate,
    // disclosed divergence from the original: take the destination.
    const d = parsePorcelain('R  old-name.js -> new-name.js');

    assert.deepEqual(d.modified, ['new-name.js']);
  });

  it("reports paths in THIS directory's coordinate system, never the repo root's", () => {
    // `git status --porcelain` prints repo-root-relative names while every
    // other function in shipping-tree.mjs speaks in paths relative to THIS
    // directory. shippedPaths() calls mixing the two "the whole hazard", so a
    // caller that passed `app.js` must not be handed back
    // `examples/serving-dashboard/app.js` — it would match nothing they hold.
    const d = parsePorcelain(' M examples/serving-dashboard/app.js', 'examples/serving-dashboard/');

    assert.deepEqual(d.modified, ['app.js'], 'the prefix must be stripped');
  });

  it('leaves a path outside this directory alone rather than mangling it', () => {
    // The negative half of the rule above: only strip a prefix that is there.
    const d = parsePorcelain(' M crates/server/src/main.rs', 'examples/serving-dashboard/');

    assert.deepEqual(d.modified, ['crates/server/src/main.rs']);
  });

  it('an empty porcelain is AGREEMENT, and produces no disclosure lines', () => {
    const d = parsePorcelain('');

    assert.deepEqual(d, { modified: [], untracked: [], deleted: [] });
    assert.deepEqual(divergenceReport([]), [], 'empty means agreement, not "unchecked"');
  });

  it('speaks on the GREEN run too — "0 of 0", never silence', () => {
    // A banner that appears only beside failures teaches readers that agreement
    // is the silent case, which is the same reflex that lets a vacuous OK pass
    // for a real one. And a missing banner is indistinguishable from one that
    // could not be computed.
    const summary = divergenceSummary([]);

    assert.match(summary, /WORKTREE_DIVERGENCE/);
    assert.match(summary, /differ on 0 of 0 file\(s\) read by this run/);
  });

  it('always states BOTH numerator and denominator for a real query', () => {
    // Shape, not value: whether this file is dirty right now depends on who is
    // mid-edit, so asserting "0" here would be a test of the crew's timing.
    // What must always hold is that the count names what it counted.
    const summary = divergenceSummary(['shipping-tree.mjs']);

    assert.match(summary, /differ on \d+ of 1 file\(s\) read by this run/);
    assert.match(summary, new RegExp(SHIPPING_REF), 'the caption must name the ref compared against');
  });

  it('says it COULD NOT COMPUTE rather than reporting zero divergence', () => {
    // The one place this mirror deliberately refuses to copy the Python
    // original. There, a failed `git status` returns an empty result, which
    // renders as "differ on 0 of N" — a false all-clear, and the exact defect
    // its own docstring warns about. An instrument that cannot run must say so.
    const d = divergentPaths(['/definitely/not/inside/this/repository']);

    assert.equal(d.computed, false, 'a git failure must be visible as a failure');
    const summary = divergenceSummary(['/definitely/not/inside/this/repository']);
    assert.match(summary, /could not be computed/);
    assert.ok(!/differ on 0 of/.test(summary), 'a failure must never render as agreement');
    assert.match(divergenceReport(['/definitely/not/inside/this/repository'])[0], /could not be computed/);
  });

  it('emits the same vocabulary as scripts/tree_context.py, which is the definition of record', () => {
    // Two agents defined this concept in two languages inside ten minutes.
    // This is what stops them drifting: one `git grep WORKTREE_DIVERGENCE`
    // must keep finding both, saying the same thing.
    const pythonSource = git('show', `${SHIPPING_REF}:scripts/tree_context.py`);
    // Join Python's implicitly-concatenated string literals before searching.
    // Without this the check fails on a phrase that IS present: tree_context.py
    // wraps the "modified" message across two adjacent f-strings, so the
    // contiguous sentence exists only at runtime and never in the source.
    // That is the line-break blind spot this crew has hit four times tonight,
    // arriving inside the very test written to stop two languages drifting.
    const python = pythonSource.replace(/"\s*f?"/g, '').replace(/\s+/g, ' ');

    for (const [klass, phrase] of Object.entries(DIVERGENCE_PHRASES)) {
      assert.ok(
        python.includes(phrase),
        `the JS "${klass}" wording has drifted from tree_context.py: ${phrase}`,
      );
    }
    assert.ok(python.includes('WORKTREE_DIVERGENCE'), 'the shared token must be in both');
    // Negative control: prove this instrument can say NO. Without it, a
    // `includes()` against a file that failed to load passes for everything.
    assert.ok(
      !python.includes('zzz-phrase-that-was-never-written-1cb'),
      'the cross-language check cannot distinguish present from absent',
    );
  });
});

describe('work parked in a stash is disclosed, not counted as agreement', () => {
  /** What git itself reports, as an independent second opinion on `entries`. */
  const gitStashCount = () => git('stash', 'list', '--format=%gd')
    .split('\n')
    .filter((line) => line.trim() !== '')
    .length;

  // The gap this closes, found by reading `git stash list` in this repository
  // rather than by imagining a case: a stash on this branch holds five files of
  // unlanded work, `git status` does not mention stashes at all, and
  // `divergenceSummary` therefore reported "differ on 0 of 2" about two of
  // those very files. Every check in this directory called them clean.
  //
  // The incentive is the sharp end. Several guards here assert a clean
  // porcelain, and the cheapest way to turn that green is to stash -- to hide
  // the evidence rather than land the work. A cleanliness check that can be
  // satisfied by concealment needs a companion that looks where things get
  // concealed.

  it('finds a path that is parked in a stash entry', () => {
    const hits = matchStashedNames([['format.js', 'other.js']], ['format.js']);

    assert.deepEqual(hits.paths, ['format.js']);
    assert.equal(hits.unreadable, 0, 'a readable entry must not be counted as unreadable');
  });

  it('translates root-relative stash names into the coordinates the caller used', () => {
    // `git stash show --name-only` prints REPO-ROOT-relative names while
    // callers hold paths relative to THIS directory -- the same mismatch
    // `parsePorcelain` handles, and the one `shippedPaths()` calls the whole
    // hazard. What comes back must be the caller's own spelling, or they
    // cannot match it against anything they hold.
    const hits = matchStashedNames(
      [['examples/serving-dashboard/ui/model-card.js']],
      ['ui/model-card.js'],
      'examples/serving-dashboard/',
    );

    assert.deepEqual(hits.paths, ['ui/model-card.js'], 'the prefix must be stripped back off');
  });

  it('does not match a path outside this directory that merely looks similar', () => {
    // Negative control for the translation above: without it, a prefix-strip
    // implemented as a substring test would match half the repository.
    const hits = matchStashedNames(
      [['crates/server/src/format.js']],
      ['format.js'],
      'examples/serving-dashboard/',
    );

    assert.deepEqual(hits.paths, [], 'a different file with the same basename is a different file');
  });

  it('reports a path once even when several stashes touch it', () => {
    // 28 stash entries exist in this repository right now. A file parked in
    // three of them is one disclosure, not three.
    const hits = matchStashedNames(
      [['format.js'], ['format.js'], ['format.js', 'other.js']],
      ['format.js', 'other.js'],
    );

    assert.deepEqual(hits.paths, ['format.js', 'other.js']);
  });

  it('an empty stash list is a real "nothing parked", and reads as such', () => {
    assert.deepEqual(matchStashedNames([], ['format.js']).paths, []);
    assert.deepEqual(stashedPaths([]).paths, [], 'no inputs cannot produce a finding');
  });

  it('counts the stash entries it examined, so a zero can be told from a no-op', () => {
    // A zero with no denominator is the failure mode this crew has hit three
    // times tonight: an instrument that never reached its subject returns
    // exactly what a clean subject returns. `entries` is that denominator.
    const s = stashedPaths(['shipping-tree.mjs']);

    assert.equal(s.unreadable, 0, 'a healthy repository must read every entry');
    assert.equal(s.entries, gitStashCount(), 'the count must match what git itself reports');
    assert.equal(typeof s.entries, 'number');
    assert.ok(s.entries >= 0, 'the number of stashes examined must be stated');
  });

  it('an unreadable stash entry is NOT reported as "nothing is stashed"', () => {
    // Same refusal `divergentPaths` makes for a failed `git status`: an
    // instrument that could not run must say so rather than say "clean".
    //
    // This assertion had to be earned. The first version of it read
    // `assert.equal(s.computed, s.unreadable === 0)` against a live call, which
    // is VACUOUS -- `unreadable` is 0 in any healthy repository, so it compared
    // true to true and a mutation deleting the whole branch stayed green. The
    // fix was structural, not a stronger assertion: unreadability is now
    // representable in the PURE matcher as a `null` entry, so the branch can be
    // reached without stashing anything in a tree the crew shares.
    const mixed = matchStashedNames([['format.js'], null, ['other.js']], ['format.js']);

    assert.deepEqual(mixed.paths, ['format.js'], 'readable entries are still examined');
    assert.equal(mixed.unreadable, 1, 'the entry that could not be read must be counted');
  });

  it('one unreadable entry poisons the verdict even when a hit was found', () => {
    // The dangerous shape: a partial read that found something looks like a
    // successful read. If four entries are unreadable and the fifth yields a
    // hit, the answer is still "I do not know what is parked".
    const partial = matchStashedNames([null, null, ['format.js']], ['format.js', 'other.js']);

    assert.deepEqual(partial.paths, ['format.js']);
    assert.equal(partial.unreadable, 2);
    assert.ok(partial.unreadable > 0, 'a partial answer must be distinguishable from a whole one');
  });

  it('the summary carries the stash count, not just the per-file lines', () => {
    // The regression that made this necessary: the disclosure existed only in
    // divergenceReport, so the ONE LINE a reader absorbs still said
    // "differ on 0 of N" about files with unlanded work. A disclosure nobody
    // reads is the same as no disclosure.
    //
    // Asserted as an IF-AND-ONLY-IF rather than "contains the phrase", because
    // whether any given file is stashed depends on what the crew is doing this
    // minute. The biconditional is true whichever way that falls, and it is
    // non-vacuous in BOTH directions: it fails if the summary forgets parked
    // work, and equally if it announces parked work that does not exist.
    const probe = ['shipping-tree.mjs', 'format.js', 'ui/model-card.js'];
    const summary = divergenceSummary(probe);
    const s = stashedPaths(probe);

    assert.match(summary, /differ on \d+ of 3 file\(s\) read by this run/);
    assert.equal(
      summary.includes(STASH_PHRASE),
      s.paths.length > 0,
      'the summary must mention parked work exactly when there is parked work',
    );
  });

  it('a per-file line exists for every parked path the summary counts', () => {
    // Report and summary must not be able to disagree. Two renderings of one
    // fact drifting apart is how a green banner ends up contradicting the
    // detail directly beneath it.
    const probe = ['shipping-tree.mjs', 'format.js', 'ui/model-card.js'];
    const lines = divergenceReport(probe);
    const s = stashedPaths(probe);

    const stashLines = lines.filter((line) => line.includes(STASH_PHRASE));
    assert.equal(stashLines.length, s.paths.length, 'one line per parked path, no more, no fewer');
    for (const name of s.paths) {
      assert.ok(
        stashLines.some((line) => line.includes(name)),
        `${name} is counted as parked but never named in the report`,
      );
    }
  });

  it('keeps the stash wording OUT of the cross-language contract', () => {
    // DIVERGENCE_PHRASES is the shared vocabulary with scripts/tree_context.py
    // and a test asserts every key of it appears in that Python source. Python
    // has no stash concept, so putting STASH_PHRASE in there would redden a
    // true statement about a real hazard -- and a guard that reddens on correct
    // work gets deleted rather than fixed.
    assert.ok(!Object.values(DIVERGENCE_PHRASES).includes(STASH_PHRASE));
    assert.equal(typeof STASH_PHRASE, 'string');
    assert.ok(STASH_PHRASE.length > 0, 'the wording must exist to be shared');
  });
});

describe('a stash check that cannot look says so, instead of saying "clean"', () => {
  // Every test in this block was IMPOSSIBLE to write an hour ago. The logic
  // lived inside a `catch` in a function that shells out to git, so the only
  // way to reach it was to break git -- and the branches went untested. Three
  // mutations survived because of it, including one that made a failed stash
  // read render as an all-clear. The cure was to move the DECISIONS out of the
  // IO, not to write cleverer assertions against the IO.

  it('tells "there are no stashes" apart from "I could not look"', () => {
    // These two states are identical in every downstream rendering unless this
    // distinction is preserved here, and confusing them is the single failure
    // this module exists to prevent.
    assert.deepEqual(countStashEntries(''), { entries: 0, unreadable: 0 });
    assert.deepEqual(countStashEntries(null), { entries: 0, unreadable: 1 });
  });

  it('counts entries without being fooled by trailing blank lines', () => {
    assert.equal(countStashEntries('stash@{0}\nstash@{1}\n').entries, 2);
    assert.equal(countStashEntries('stash@{0}\n\n\n').entries, 1);
  });

  it('a failed stash read renders as UNKNOWN, never as an absence of parked work', () => {
    const suffix = stashSuffix({ paths: [], unreadable: 1 });

    assert.match(suffix, /UNKNOWN/);
    assert.ok(!suffix.includes(STASH_PHRASE), 'it must not imply a finding it does not have');
    assert.notEqual(suffix, '', 'silence is what a clean result looks like -- this is not one');
  });

  it('an unreadable stash outranks a hit: a partial answer is not an answer', () => {
    // If four entries are unreadable and the fifth yields a hit, reporting
    // "1 file is parked" states a floor as though it were a total.
    const suffix = stashSuffix({ paths: ['format.js'], unreadable: 4 });

    assert.match(suffix, /UNKNOWN/, 'the doubt must win over the partial finding');
  });

  it('says nothing at all when there is genuinely nothing parked', () => {
    // The other half of the contract. A banner that fires unconditionally is a
    // constant, and a constant carries no information.
    assert.equal(stashSuffix({ paths: [], unreadable: 0 }), '');
    assert.deepEqual(stashLines({ paths: [], unreadable: 0 }, 'abc123'), []);
  });

  it('names the ref it compared against in every per-file line', () => {
    const lines = stashLines({ paths: ['format.js'], unreadable: 0 }, 'deadbeef');

    assert.equal(lines.length, 1);
    assert.match(lines[0], /deadbeef/, 'a disclosure that names no tree discloses nothing');
    assert.match(lines[0], /format\.js/);
    assert.ok(lines[0].includes(STASH_PHRASE));
  });

  it('emits a line for the unreadable case even with zero known paths', () => {
    const lines = stashLines({ paths: [], unreadable: 2 }, 'deadbeef');

    assert.equal(lines.length, 1, 'the doubt itself is the disclosure');
    assert.match(lines[0], /not the same as "none do"/);
  });

  it('orders parked paths by name, not by which stash happened to be newest', () => {
    // Pins the sort. Without it the order is Set-insertion order, which is
    // stash recency -- so the same tree prints a different report depending on
    // who stashed last, and a diff of two runs shows spurious churn.
    const out = matchStashedNames([['zebra.js'], ['alpha.js'], ['middle.js']],
      ['middle.js', 'zebra.js', 'alpha.js']);

    assert.deepEqual(out.paths, ['alpha.js', 'middle.js', 'zebra.js']);
  });
});
