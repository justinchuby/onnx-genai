/**
 * shipping-tree.mjs — assert that a check is reporting on the tree we ship.
 *
 * WHY THIS EXISTS
 * ---------------
 * Five false negatives in one session came from one parked checkout. Every one
 * of them looked like a clean, correct, well-formed command. There was nothing
 * wrong with any of the commands: a command cannot tell you it is pointed at
 * the wrong universe, and nothing in its output ever will. An absence found in
 * the wrong tree and a verified absence are byte-identical.
 *
 * The checks in this directory are NOT vulnerable in the way a bare `grep` is.
 * They resolve every path from `import.meta.url`, so the working directory a
 * developer happens to be standing in cannot steer them. That immunity is real
 * and it is also the trap: a copy of this suite sitting in a parked worktree
 * reads THAT worktree, self-consistently, and goes green. It is not confused,
 * it is not misconfigured, and its output is entirely accurate — about a tree
 * nobody ships. CWD-independence removes the noisy failure and leaves the
 * silent one.
 *
 * So the thing to assert is not "which directory am I standing in" but "is the
 * artefact I am about to make claims about the artefact that ships".
 *
 * WHY NOT A BARE STRING EQUALITY ON THE BRANCH NAME
 * -------------------------------------------------
 * Reviewers read this repo from clean DETACHED worktrees at a named SHA — that
 * is the practice we asked them to adopt so their findings are reproducible.
 * `git rev-parse --abbrev-ref HEAD` prints `HEAD` there, so a naive equality
 * check fails every reviewer while passing anyone whose parked copy happens to
 * carry the right branch name. That gets the discipline exactly backwards: it
 * punishes the rigorous workflow and waves through the sloppy one.
 *
 * A detached HEAD is therefore accepted IF AND ONLY IF the commit is contained
 * in the shipping branch. That is a content-addressed question about history
 * rather than a string comparison on a label, and it is the same rule that let
 * us settle the cherry-pick argument: match the commits, not the names.
 */

import { execFileSync } from 'node:child_process';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

export const SHIPPING_BRANCH = 'feat/genai-demo-dashboard';

const HERE = dirname(fileURLToPath(import.meta.url));

function git(...args) {
  return execFileSync('git', args, { cwd: HERE, encoding: 'utf8' }).trim();
}

/**
 * The commit every `shipped()` read resolves against, fixed for this process.
 *
 * WHY THIS IS RESOLVED ONCE INSTEAD OF PER CALL
 * ---------------------------------------------
 * `HEAD` is not a commit, it is a POINTER, and on a branch fourteen agents are
 * committing to it moves every twenty to forty seconds. A guard that spells
 * `HEAD` in ten separate `git show` invocations therefore reads up to ten
 * DIFFERENT TREES within a single run, and reports the result as one
 * measurement. That is not a hypothetical: `check-perf-claims` failed in a full
 * suite run and passed in isolation, and the only difference was that HEAD
 * moved three times while the suite was executing.
 *
 * The failure is invisible in exactly the way that matters. Each individual
 * read is correct. The bytes are real, the paths resolve, no command errors.
 * What is destroyed is the ONE property a cross-file check depends on: that
 * file A and file B were read from the same tree. A check that asserts "the
 * README quotes the figure the baseline publishes" can read a repaired README
 * and a stale baseline and report a contradiction that never existed in any
 * commit.
 *
 * Resolving to an immutable SHA at module load costs one `rev-parse` and makes
 * every read in the process self-consistent. It does not make the answer more
 * correct — it makes the answer be ABOUT SOMETHING, which is a different and
 * more basic property, and the one that was missing.
 *
 * WHY THE OVERRIDE IS AN ENV VAR AND NOT AN ARGUMENT
 * -------------------------------------------------
 * Reviewers score a named tag (`review-1`) rather than a moving branch. The
 * override has to reach ten check files that a reviewer runs through
 * `node --test`, which passes no arguments through to them. An environment
 * variable is the only channel that exists, and it costs a caller nothing:
 *
 *   SHIPPING_TREE_REF=review-1 node --test './*.test.js'
 *
 * An unresolvable ref throws HERE, at load, naming itself — rather than
 * surfacing later as a confusing per-file "path does not exist in HEAD".
 */
export const SHIPPING_REF = (() => {
  const requested = process.env.SHIPPING_TREE_REF?.trim() || 'HEAD';
  try {
    return git('rev-parse', requested);
  } catch {
    throw new Error(
      `SHIPPING_TREE_REF is set to '${requested}', which this repository cannot ` +
        `resolve to a commit.\n` +
        `  Checks read their inputs from that commit, so there is nothing to read ` +
        `and no honest result to report.\n` +
        `  Unset it to score the current HEAD, or name a ref that exists ` +
        `(e.g. a review tag).`,
    );
  }
})();

/**
 * Describe the tree this file physically lives in.
 *
 * Every field is resolved with `cwd: HERE` — the directory of THIS MODULE, not
 * the caller's working directory. A check must report on the tree it read its
 * inputs from, and its inputs are resolved relative to itself.
 */
export function describeTree() {
  return {
    toplevel: git('rev-parse', '--show-toplevel'),
    branch: git('rev-parse', '--abbrev-ref', 'HEAD'),
    head: git('rev-parse', '--short', 'HEAD'),
    detached: git('rev-parse', '--abbrev-ref', 'HEAD') === 'HEAD',
    // The commit checks actually READ, which is not necessarily `head`: it is
    // pinned at load and can be overridden. A failure report that names only
    // HEAD would send the reader to the wrong tree whenever they differ.
    ref: SHIPPING_REF,
    refIsOverridden: Boolean(process.env.SHIPPING_TREE_REF?.trim()),
  };
}

/**
 * Read a file AS IT SHIPS — from HEAD, never from the desk.
 *
 * WHY THIS LIVES HERE AND NOT IN EACH CHECK
 * -----------------------------------------
 * Two check files independently grew a byte-identical private copy of this
 * function, and both copies are correct. That is not two authors being
 * careless; it is an API gap manufacturing duplicates. This module already
 * owned the question "is the artefact I am about to describe the artefact that
 * ships" — it simply had no way to HAND YOU that artefact, so everyone who
 * needed the bytes wrote their own reader. The gap, not the authors, produced
 * the duplication, and closing it is the only fix that stops the next one.
 *
 * WHY HEAD AND NOT `readFileSync`
 * -------------------------------
 * Reading the working tree is correct and means the wrong thing, and the two
 * are indistinguishable whenever the tree is clean — which is exactly when you
 * are most likely to trust the result. The failure it permits is
 * one-directional and it is the bad direction: a defect still present in HEAD
 * but repaired only on disk scores GREEN, and the repair evaporates on the next
 * checkout. A reviewer clones HEAD. So does CI. So does the demo. Nobody clones
 * your working tree.
 *
 * The inverse failure — a fix on disk that is not yet committed reads RED — is
 * the safe one, and its remedy is the thing you were going to do anyway.
 *
 * THIS FUNCTION THROWS, DELIBERATELY, AND CALLERS MUST NOT "FIX" THAT
 * ------------------------------------------------------------------
 * If `rel` is not in HEAD, `git show` exits non-zero and this throws. That is
 * the existing contract of both private copies and it is preserved here byte
 * for byte, because the alternative — returning '' — makes an ABSENT file
 * indistinguishable from an EMPTY one, and every content check in this suite
 * scores an empty string as CLEAN. A file that vanished would then score green
 * for having vanished, which is the vacuity this whole directory exists to
 * prevent.
 *
 * @param {string} rel path relative to THIS directory, not the repo root.
 * @returns {string} the file's bytes as committed at HEAD.
 */
export function shipped(rel) {
  // The `./` is load-bearing: `git show <ref>:<path>` resolves from the REPO
  // ROOT, not the cwd, so a bare relative path silently resolves to nothing.
  // `<ref>:./<path>` is the form that honours `cwd`.
  //
  // SHIPPING_REF is a resolved SHA, never the literal 'HEAD'. See its docstring:
  // spelling 'HEAD' here would let a single run read several different trees.
  return execFileSync('git', ['show', `${SHIPPING_REF}:./${rel}`], {
    cwd: HERE,
    maxBuffer: 64 * 1024 * 1024,
  }).toString();
}

/**
 * Fail loudly unless this file lives in the tree we ship.
 *
 * Call this FIRST, before any other assertion in a check file. A check that
 * validates content before validating its own provenance can report a
 * confident, detailed, entirely accurate result about the wrong artefact.
 */
export function assertShippingTree() {
  const tree = describeTree();
  const where =
    `worktree ${tree.toplevel}\n  branch   ${tree.branch}\n  HEAD     ${tree.head}` +
    (tree.refIsOverridden ? `\n  reading  ${tree.ref} (SHIPPING_TREE_REF)` : '');

  // MUTATION-TESTING ESCAPE HATCH, OPT-IN AND LOUD.
  //
  // This crew's ratified way to prove a guard works is: mutate, commit, observe
  // RED. For any guard that reads HEAD, that must happen in a throwaway
  // detached worktree -- and the commit is then, BY CONSTRUCTION, not contained
  // in the shipping branch. So this containment check threw FIRST, before any
  // assertion ran, and produced a red that had nothing to do with the guard
  // under test.
  //
  // That is the worst failure available to a verification method: IT RETURNS
  // THE EXPECTED ANSWER FOR THE WRONG REASON. Two of my own mutations "passed"
  // this way before I read the failure text instead of the exit status. A red
  // you cannot attribute is not evidence, and this one was byte-indistinguish-
  // able from the red I was hoping for.
  //
  // Opt-in, env-gated, and it announces itself on stderr on EVERY run, so a
  // result produced under it cannot be quoted as a normal one.
  if (process.env.SHIPPING_TREE_ALLOW_DETACHED === '1') {
    process.stderr.write(
      `⚠️  SHIPPING_TREE_ALLOW_DETACHED=1 — provenance check BYPASSED.\n` +
        `  ${where}\n` +
        `  This result describes an artefact that is NOT on the shipping branch.\n` +
        `  Valid for mutation testing only. Never quote it as a property of the branch.\n`,
    );
    return tree;
  }

  // An overridden ref is checked on ITS OWN merits, not HEAD's.
  //
  // Everything below this line validates the tree the process is STANDING IN.
  // Once SHIPPING_TREE_REF is set, that is no longer the tree the checks READ,
  // and the two can disagree: a reviewer standing on a perfectly good HEAD can
  // point the checks at a ref from another branch entirely. HEAD would pass,
  // every content assertion would describe an artefact nobody is merging, and
  // nothing would say so. Provenance has to follow the bytes.
  if (tree.refIsOverridden) {
    let containing = '';
    try {
      containing = git('branch', '--contains', tree.ref, '--format=%(refname:short)');
    } catch {
      containing = '';
    }
    const branches = containing.split('\n').map((s) => s.trim()).filter(Boolean);
    if (!branches.includes(SHIPPING_BRANCH)) {
      throw new Error(
        `SHIPPING_TREE_REF resolves to ${tree.ref}, which is not contained in ` +
          `'${SHIPPING_BRANCH}'.\n  ${where}\n  contained in: ${branches.join(', ') || '(no local branch)'}\n\n` +
          `Pinning to a review tag is the right way to score this repo, but the ` +
          `tag has to be on the branch we ship. Every result below would ` +
          `describe an artefact nobody is merging.`,
      );
    }
  }

  if (tree.detached) {
    // Content-addressed: is this commit actually ON the shipping branch?
    let containing = '';
    try {
      containing = git('branch', '--contains', 'HEAD', '--format=%(refname:short)');
    } catch {
      containing = '';
    }
    const branches = containing.split('\n').map((s) => s.trim()).filter(Boolean);
    if (!branches.includes(SHIPPING_BRANCH)) {
      throw new Error(
        `This check is running from a DETACHED HEAD that is not contained in ` +
          `'${SHIPPING_BRANCH}'.\n  ${where}\n  contained in: ${branches.join(', ') || '(no local branch)'}\n\n` +
          `A detached worktree at a named SHA is a legitimate way to review this ` +
          `repo, but only if the SHA is on the branch we ship. This one is not, ` +
          `so every result below would describe an artefact nobody is merging.`,
      );
    }
    return tree;
  }

  if (tree.branch !== SHIPPING_BRANCH) {
    throw new Error(
      `This check is running from the wrong tree.\n  ${where}\n  expected branch '${SHIPPING_BRANCH}'\n\n` +
        `Every path in this suite is resolved from import.meta.url, so this file ` +
        `read its inputs from the worktree named above and would have reported on ` +
        `it accurately and silently. Five false negatives in one session came from ` +
        `exactly this, and not one of them looked wrong. Re-run from the shipping ` +
        `worktree; a green result from here proves nothing about what merges.`,
    );
  }

  return tree;
}
