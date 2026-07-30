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

import { SHIPPING_BRANCH, SHIPPING_REF, describeTree, shipped } from './shipping-tree.mjs';

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
});
