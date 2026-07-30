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
 * Build the Error for a condition that stops this module MEASURING ANYTHING,
 * and label it on the way out. Call as `throw cannotRun(...)`.
 *
 * WHY A LABEL, AND WHY IT IS NOT AN EXIT CODE
 * -------------------------------------------
 * Every refusal in this file is the same class: NOT "a check found a defect"
 * but "a check could not run". Those two outcomes were rendered identically --
 * a stack trace and a non-zero exit -- and a reader who sees red next to a
 * guard's name reads "the guard found something". An absence of CAPABILITY was
 * displayed as an absence of COMPLIANCE, which inverts the meaning entirely.
 *
 * The house convention for that distinction is `exit 2`, and it CANNOT be used
 * here. Measured on node v25.6.1, three arms plus a control:
 *
 *   node --test <file that exits 2>  ->  1   the runner flattens it
 *   node        <same file>          ->  2   the file is correct
 *   node --test <file that exits 0>  ->  0   control: success propagates exactly
 *
 * `node --test` collapses every non-zero child exit to 1, and `run-tests.sh`
 * runs all suites in ONE invocation, so a per-file exit code does not survive
 * to the caller even in principle. The exit code is not an available channel.
 * The TEXT is, so the text carries the distinction.
 *
 * It goes to STDERR, beside the message it qualifies, rather than into a
 * summary. A qualifier that is not adjacent to the claim is not attached to
 * it: readers paste tails, and a banner printed at the top is a banner nobody
 * quotes. Under `node --test` the child's stderr is relayed onto the parent's
 * stdout, so it reaches the reader on either path.
 *
 * The Error is still THROWN, not swallowed into `process.exit`. A refusal that
 * exits is untestable in-process, and `shipping-tree.test.js` proves these
 * refusals reach stderr by reading exactly that.
 */
function cannotRun(message) {
  process.stderr.write(
    `\nCANNOT_RUN: ${message}\n\n` +
      `THIS IS NOT A FINDING ABOUT ANY DOCUMENT OR ANY SOURCE FILE.\n` +
      `No check below this point ran, so nothing below is evidence either way.\n` +
      `Fix the condition above and re-run; do not read this as a defect report.\n\n`,
  );
  const err = new Error(message);
  err.cannotRun = true;
  return err;
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
 * REVIEW_SHA is accepted as an equal spelling of the same thing, because that
 * is the name that was broadcast to reviewers. Two NAMES resolved at ONE point
 * cannot disagree; two independent implementations reading two names would,
 * and that is the distinction worth holding. Honouring only the name this
 * module happened to pick first is the worse failure: a reviewer sets
 * REVIEW_SHA, nothing reads it, nothing errors, and they score a moving branch
 * believing they are pinned to a tag. A silent no-op is worse than an
 * unsupported variable, because it is trusted.
 *
 * Both set to different commits is REFUSED rather than resolved by precedence.
 * A precedence rule silently discards one of two explicit instructions and the
 * caller never learns which one lost.
 *
 * An unresolvable ref throws HERE, at load, naming itself — rather than
 * surfacing later as a confusing per-file "path does not exist in HEAD".
 */
export const SHIPPING_REF = (() => {
  const fromTreeRef = process.env.SHIPPING_TREE_REF?.trim();
  const fromReviewSha = process.env.REVIEW_SHA?.trim();

  const resolve = (ref, name) => {
    try {
      return git('rev-parse', `${ref}^{commit}`);
    } catch {
      throw cannotRun(
        `${name} is set to '${ref}', which this repository cannot resolve to a ` +
          `commit.\n` +
          `  Checks read their inputs from that commit, so there is nothing to ` +
          `read and no honest result to report.\n` +
          `  Unset it to score the current HEAD, or name a ref that exists ` +
          `(e.g. a review tag).`,
      );
    }
  };

  if (fromTreeRef && fromReviewSha) {
    const a = resolve(fromTreeRef, 'SHIPPING_TREE_REF');
    const b = resolve(fromReviewSha, 'REVIEW_SHA');
    if (a !== b) {
      throw cannotRun(
        `SHIPPING_TREE_REF='${fromTreeRef}' and REVIEW_SHA='${fromReviewSha}' name ` +
          `different commits (${a.slice(0, 8)} vs ${b.slice(0, 8)}).\n` +
          `  They are two spellings of one setting, so there is no correct way to ` +
          `choose between two explicit instructions.\n` +
          `  Unset one.`,
      );
    }
    return a;
  }

  if (fromTreeRef) return resolve(fromTreeRef, 'SHIPPING_TREE_REF');
  if (fromReviewSha) return resolve(fromReviewSha, 'REVIEW_SHA');
  return resolve('HEAD', 'HEAD');
})();

let announced = false;

/**
 * Print the resolved shipping ref once per process, so a failure is self-dating.
 *
 * A red from one of these checks is read by someone who was not present when it
 * ran, and their first question is always "against what?". Without an answer in
 * the output the only available reading is "against now", which is the one thing
 * a recorded failure can never mean. This branch moves every few tens of seconds;
 * by the time a reviewer opens the log, `HEAD` denotes a different tree than the
 * one that produced the message they are reading.
 *
 * So the ref is printed, not merely resolved. Pinning makes a run internally
 * consistent; printing makes the run's SUBJECT recoverable afterwards. They are
 * different properties and a review needs both — a deterministic result that
 * cannot be attributed still costs an argument to re-derive.
 *
 * Print-once, because ten guards import this and ten identical banners would
 * train readers to skip the line that carries the whole provenance. Emitted on
 * stderr: TAP consumers parse stdout, and provenance is not a test result.
 */
/**
 * The repository's top level, resolved ONCE at module load.
 *
 * Eight test and tooling files each ran their own
 * `execFileSync('git', ['rev-parse', '--show-toplevel'])` under four different
 * local names (`repoRoot`, `REPO_ROOT`, `TOPLEVEL`, `REPO`), with three
 * different maxBuffer ceilings and one at the 1 MiB default. Four spellings of
 * one fact is four places for it to drift, and a reader grepping for any single
 * spelling finds a quarter of them -- which is how the population was
 * repeatedly counted as two.
 *
 * This is a constant for the life of a process, so resolving it per caller
 * bought nothing. `describeTree()` below already computed it privately; this
 * export is that same call, named and shared rather than duplicated.
 */
export const REPO_ROOT = git('rev-parse', '--show-toplevel');

export function announceShippingRef() {
  if (announced) return SHIPPING_REF;
  announced = true;
  // Names the variable actually consulted, not just that an override happened:
  // a reviewer debugging an unexpected ref needs to know WHICH of the two
  // spellings the process read, and REVIEW_SHA exists precisely because people
  // reach for the other name.
  const treeRef = process.env.SHIPPING_TREE_REF?.trim();
  const reviewSha = process.env.REVIEW_SHA?.trim();
  let via = 'default (HEAD at load)';
  if (treeRef && reviewSha) via = `SHIPPING_TREE_REF=${treeRef} (REVIEW_SHA agrees)`;
  else if (treeRef) via = `SHIPPING_TREE_REF=${treeRef}`;
  else if (reviewSha) via = `REVIEW_SHA=${reviewSha}`;
  process.stderr.write(`# shipping ref: ${SHIPPING_REF} [${via}]\n`);

  // STATE THE CORPUS BESIDE THE RESULT, ON PASS AS WELL AS ON FAIL.
  //
  // `assertShippingTree` folds porcelain into its `where` banner, but that
  // banner is only ever rendered into a THROWN error -- so the disclosure
  // reaches exactly the readers who already know something is wrong, and
  // never the ones quoting a green run. That is the same defect this suite
  // has now found in three separate guards: a limitation documented where
  // the passing verdict cannot print it.
  //
  // It does NOT throw. Eleven checks read the desk after asserting, and a
  // dirty tree makes their result describe the desk rather than the branch --
  // but a hard failure here would red the whole suite for an untracked file
  // in a directory nobody's check reads, and a guard that cries wolf on
  // unrelated dirt is one people learn to run with a bypass flag set.
  const dirty = describeTree().dirty;
  if (dirty.length > 0) {
    process.stderr.write(
      `⚠️  working tree is NOT clean: ${dirty.length} path(s).\n` +
        dirty.map((d) => `#     ${d}\n`).join('') +
        `#   Checks that read HEAD via shipped()/shippedPaths() scored the\n` +
        `#   COMMITTED bytes. A PASS therefore says NOTHING about uncommitted\n` +
        `#   edits to the paths above: mutate one and the green will not move,\n` +
        `#   so a control run against the desk cannot fire. Commit, then re-run.\n` +
        `#   Checks that readFileSync the desk are scoring THESE bytes, not the branch.\n`,
    );
  }
  return SHIPPING_REF;
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
    // The commit checks actually READ, which is not necessarily `head`: it is
    // pinned at load and can be overridden. A failure report that names only
    // HEAD would send the reader to the wrong tree whenever they differ.
    ref: SHIPPING_REF,
    refIsOverridden: Boolean(process.env.SHIPPING_TREE_REF?.trim()),
    // THE FOURTH FACT, AND THIS MODULE SHIPPED WITHOUT IT.
    //
    // This crew requires four facts beside every result: toplevel, branch,
    // SHA, porcelain. `describeTree` implemented the first three and was the
    // shared instrument everyone else quoted, so the missing one was missing
    // everywhere at once.
    //
    // It is the fact that decides whether the paragraph 100 lines above this
    // is describing a hazard or a non-event. That paragraph says reading the
    // working tree "is correct and means the wrong thing, and the two are
    // indistinguishable whenever the tree is clean". Eleven checks in this
    // directory call `assertShippingTree()` and then `readFileSync` the desk.
    // Their safety rests ENTIRELY on cleanliness, and nothing measured it.
    //
    // Deliberately a LIST, not a boolean. "Dirty" is not the question; the
    // question is whether the dirt intersects what a given check reads, and
    // only the caller knows that. A boolean would force every caller to
    // re-shell out to recover the paths, which is the API gap that already
    // manufactured two duplicate readers in this repo.
    dirty: git('status', '--porcelain')
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean),
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
 * List the files that SHIP under this directory, as committed.
 *
 * The companion to `shipped()`. That function answers "what are this file's
 * bytes"; this one answers "which files are there at all". A check that
 * enumerates its own corpus from the working tree can be handed files nobody
 * ships (another agent's untracked draft) or be silently denied files that do
 * ship, and in both cases it reports a confident total for a corpus that no
 * reviewer will ever see.
 *
 * WHY THIS TAKES NO PATHSPEC — FILTER THE RESULT INSTEAD
 * -----------------------------------------------------
 * The obvious signature is `shippedPaths('*.test.js')`. It is a trap, and it
 * was this function's first implementation:
 *
 *   git ls-tree -r --name-only <ref> -- '*.test.js'   ->  0        SILENTLY
 *   git ls-tree -r --name-only <ref> .                ->  114      correct
 *
 * `git ls-tree` does not glob-match a pathspec the way `git ls-files` does, so
 * the filtered form returns an empty list and exits 0. A negative control does
 * not catch it: the control returns 0 too, and 0 == 0 reads as agreement. This
 * is the same class as `**\/*.js` reaching half the tree and `| tail` truncating
 * a census — a decorative-looking token silently changes what was measured and
 * nothing errors. So the pathspec is not offered at all: callers get everything
 * and filter in JS, where a typo produces an empty array they can see rather
 * than a query that quietly matches nothing.
 *
 * WHY `cwd: HERE` AND NO `--full-tree`
 * ------------------------------------
 * `git ls-tree` is cwd-relative: run from a subdirectory it restricts itself to
 * that subdirectory AND prints paths relative to it. That behaviour cost an
 * architect a corpus declaration that under-reported coverage 36x — it printed
 * "not examined: 15" when the truth was 546 of 561 — and it under-reported in
 * the FLATTERING direction, which is the direction nobody double-checks.
 *
 * Here that same behaviour is exactly what is wanted, so it is PINNED rather
 * than inherited: `cwd` is hardcoded to this directory, so the result does not
 * depend on where the caller stood. Paths come back relative to this directory,
 * which is the same coordinate system `shipped()` takes. Mixing the two
 * coordinate systems is the whole hazard, so the pair uses one.
 *
 * @returns {string[]} every shipped path under this directory, relative to it,
 *   sorted. Never empty for a healthy tree — callers should assert a floor.
 */
export function shippedPaths() {
  const out = execFileSync(
    'git',
    ['ls-tree', '-r', '--name-only', SHIPPING_REF, '.'],
    { cwd: HERE, maxBuffer: 64 * 1024 * 1024 },
  ).toString();

  return out
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .sort();
}

/**
 * Path from the repo root down to THIS directory, e.g. `examples/serving-dashboard/`.
 *
 * Needed because `git status --porcelain` prints names relative to the REPO
 * ROOT while every other function here speaks in paths relative to THIS
 * directory. `shippedPaths()` documents mixing those two coordinate systems as
 * "the whole hazard", so the divergence helpers below translate rather than
 * leak a second system into callers that already use the first.
 */
const HERE_PREFIX = git('rev-parse', '--show-prefix');

/**
 * Classify `git status --porcelain` output. Pure — no git, no disk, no clock.
 *
 * Split out from `divergentPaths()` so the risky part can be tested without
 * writing into the shared tree. This directory's own norm forbids that: a test
 * that creates a file here to prove a point trips somebody else's `porcelain 0`
 * assertion, and the red is attributed to their diff rather than to this test.
 * So the I/O stays in the caller and the CLASSIFICATION — which is where every
 * plausible bug lives — is exercised on synthetic input.
 *
 * @param {string} raw stdout of `git status --porcelain`, UNTRIMMED.
 * @param {string} prefix repo-root-to-here path, e.g. `examples/serving-dashboard/`.
 * @returns {{modified: string[], untracked: string[], deleted: string[]}}
 */
export function parsePorcelain(raw, prefix = '') {
  const result = { modified: [], untracked: [], deleted: [] };

  for (const line of String(raw).split('\n')) {
    if (!line.trim()) continue;
    // Columns 0-1 ARE the status and a leading space is DATA, so this slices
    // from the untrimmed line. Trimming ` M path` to `M path` shifts every
    // subsequent offset by one and yields a name that matches no file.
    const code = line.slice(0, 2);
    let name = line.slice(3).trim().replace(/^"|"$/g, '');
    // `R  old -> new` — the destination is the path that now exists.
    if (name.includes(' -> ')) name = name.slice(name.lastIndexOf(' -> ') + 4);
    if (prefix && name.startsWith(prefix)) name = name.slice(prefix.length);

    if (code === '??') result.untracked.push(name);
    else if (code.includes('D')) result.deleted.push(name);
    else result.modified.push(name);
  }
  return result;
}

/**
 * Which of `rels` disagree between the desk and the commit this run reads.
 *
 * @param {string[]} rels paths relative to THIS directory — the same coordinate
 *   system `shipped()` and `shippedPaths()` use.
 * @returns {{modified: string[], untracked: string[], deleted: string[],
 *   computed: boolean}} `computed: false` means git could not answer.
 *
 * SCOPED TO WHAT THE CALLER ACTUALLY READ, NOT THE TREE AT LARGE
 * -------------------------------------------------------------
 * A checker that reads six files does not care that a seventh is dirty. This
 * branch has carried a non-zero global porcelain for most of its life, so a
 * whole-tree banner is a constant, and a constant carries no information.
 * Scoping the question to the inputs is what turns it back into a signal.
 *
 * TWO DELIBERATE DIVERGENCES FROM scripts/tree_context.py, BOTH DISCLOSED
 * ----------------------------------------------------------------------
 * The Python original is the definition of record and this is its mirror; the
 * message strings and bucket names are copied verbatim so one
 * `git grep WORKTREE_DIVERGENCE` finds both languages. Two things are not
 * copied, on purpose:
 *
 *  1. ON GIT FAILURE PYTHON RETURNS AN EMPTY RESULT, which renders as
 *     "differ on 0 of N" — a false all-clear that is indistinguishable from
 *     real agreement. That contradicts its own docstring ("a missing banner is
 *     indistinguishable from a banner that could not be computed"), so here a
 *     failure is reported AS a failure. An instrument that cannot run must say
 *     so; it must never say "clean".
 *  2. A RENAME (`R  old -> new`) would land in Python's `modified` bucket under
 *     the literal name `old -> new`, which is not a path and matches nothing.
 *     Here the destination name is taken.
 */
export function divergentPaths(rels) {
  const result = { modified: [], untracked: [], deleted: [], computed: true };
  if (!Array.isArray(rels) || rels.length === 0) return result;

  let raw;
  try {
    // NOT the shared `git()` helper: it trims, and porcelain's first two
    // columns ARE the status — a leading space is data. Trimming ` M path`
    // to `M path` shifts every subsequent slice by one and mangles the name.
    raw = execFileSync('git', ['status', '--porcelain', '--', ...rels], {
      cwd: HERE,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
    });
  } catch {
    return { ...result, computed: false };
  }

  return { ...parsePorcelain(raw, HERE_PREFIX), computed: true };
}
/**
 * Human-readable disclosure lines. EMPTY MEANS AGREEMENT, not "unchecked".
 *
 * Callers print these alongside their verdict. A result that quietly measured
 * uncommitted bytes is not wrong — it is UNFALSIFIABLE, which is worse, because
 * nobody can tell later which tree it described.
 *
 * @param {string[]} rels paths relative to THIS directory.
 * @returns {string[]}
 */
/**
 * The wording each divergence class uses, shared with `scripts/tree_context.py`.
 *
 * Exported so a test can assert the two languages still say the same thing. The
 * Python module is the definition of record; this is its mirror. A shared
 * concept with two spellings is two concepts, and the whole point of the token
 * `WORKTREE_DIVERGENCE` is that ONE `git grep` finds every disclosure in the
 * repo no matter which language emitted it.
 */
export const DIVERGENCE_PHRASES = Object.freeze({
  modified: 'read from the working tree, which differs from',
  untracked: 'untracked. Absent from',
  deleted: 'deleted on disk but present in',
});

export function divergenceReport(rels) {
  const d = divergentPaths(rels);
  if (!d.computed) {
    return [
      'WORKTREE_DIVERGENCE could not be computed: git status failed. This run '
        + 'does not know whether it read committed bytes or not.',
    ];
  }

  const lines = [];
  for (const name of [...d.modified].sort()) {
    lines.push(
      `WORKTREE_DIVERGENCE ${name}: ${DIVERGENCE_PHRASES.modified} `
        + `${SHIPPING_REF}. This result describes uncommitted bytes.`,
    );
  }
  for (const name of [...d.untracked].sort()) {
    lines.push(
      `WORKTREE_DIVERGENCE ${name}: ${DIVERGENCE_PHRASES.untracked} ${SHIPPING_REF} `
        + 'entirely -- it is on one desk and is not part of what we ship.',
    );
  }
  for (const name of [...d.deleted].sort()) {
    lines.push(
      `WORKTREE_DIVERGENCE ${name}: ${DIVERGENCE_PHRASES.deleted} `
        + `${SHIPPING_REF}. Any result about it describes a file that is going away.`,
    );
  }
  return lines;
}

/**
 * One line: how many of the files this run READ disagree with the shipping ref.
 *
 * The per-file lines above are correct and nobody counts them. An aggregate is
 * what a reader absorbs, and it must be printed on the GREEN run too — a
 * banner that appears only next to failures teaches people that agreement is
 * the silent case, which is the reflex that lets a vacuous OK pass for a real
 * one. Says "0 of N" rather than going quiet, because a missing banner is
 * indistinguishable from one that could not be computed.
 *
 * WHY THIS DISCLOSES INSTEAD OF SWITCHING TO `git show HEAD:`
 * ----------------------------------------------------------
 * Repointing a desk-reading guard at committed bytes false-reds the ordinary
 * commit that adds a symbol and cites it in the same change — the citation is
 * correct on disk and absent from HEAD until the moment it lands. A guard that
 * reddens on correct work gets deleted, and then we have neither the guard nor
 * the disclosure. Say which tree you read; do not silently change it.
 *
 * @param {string[]} rels paths relative to THIS directory.
 * @returns {string}
 */
export function divergenceSummary(rels) {
  const total = Array.isArray(rels) ? rels.length : 0;
  const d = divergentPaths(rels);
  if (!d.computed) {
    return `WORKTREE_DIVERGENCE could not be computed for ${total} file(s) read by this run`;
  }
  const n = d.modified.length + d.untracked.length + d.deleted.length;
  return (
    `WORKTREE_DIVERGENCE tree and ${SHIPPING_REF} differ on ${n} of ${total} `
    + 'file(s) read by this run'
  );
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
    `\n  porcelain ${tree.dirty.length}` +
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
      throw cannotRun(
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
      throw cannotRun(
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
    throw cannotRun(
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
