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
  };
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
  const where = `worktree ${tree.toplevel}\n  branch   ${tree.branch}\n  HEAD     ${tree.head}`;

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
