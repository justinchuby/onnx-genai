// A review is a MEASUREMENT, and a measurement decays at the rate the tree moves.
//
// This session produced the same defect in three reviewers' documents independently:
// a row asserting a live defect that had been fixed an hour earlier. Nobody was
// careless. The documents were true when written and nothing re-scored them, so the
// staleness was invisible to every instrument we owned -- including the reviewers who
// wrote them, who each re-read their own file and saw prose they still agreed with.
//
// The cost is not symmetric with an ordinary wrong finding, which is why this is a
// test and not a convention. A stale RED gets work REDONE: three agents were
// dispatched onto one already-closed item tonight, and one was dispatched to fix a
// defect that had been publicly retracted twice. A false GREEN gets work SKIPPED.
// Both are expensive, and neither announces itself.
//
// So: every review document declares the commit it was measured at, and this test
// fails when that commit is no longer part of the history the branch is on.
//
// WHY THE VALUE MUST BE A RAW HEX SHA AND NEVER A REF NAME
// -------------------------------------------------------
// The obvious spelling is `MEASURED-AT: review-0`, and it is unfalsifiable. Tonight
// `review-0` named 6ecd9183 at 03:57 and 0aac6bb1 at 04:21 -- SIXTY COMMITS apart.
// It was re-pointed deliberately and announced in a commit message, and three
// reviewers went on quoting the old mapping because nothing errored: a tag is a
// MUTABLE POINTER to an IMMUTABLE OBJECT, and the object's immutability is exactly
// what hides the move. Every old SHA still resolves. Nothing breaks.
//
// A document anchored to a name therefore re-anchors itself, silently, whenever
// somebody moves the name -- and a freshness guard reading that name would go green
// forever while the document rotted. The hex requirement is the whole point of the
// marker, not a formatting preference.
//
// WHY RESOLUTION IS CHECKED AND NOT JUST SHAPE
// -------------------------------------------
// Our agent IDs are eight hex characters. `73e77d95` is an agent, not a commit, and
// it appears in a review document's header three lines above a real SHA. No regex
// can separate them -- they are the same bytes in the same alphabet at the same
// length. Only asking git whether the object exists, and is a COMMIT, distinguishes
// them, and that check costs one command.
//
// ANTI-VACUITY, NOT COMPLETENESS
// ------------------------------
// The floor below asserts that at least one document still carries the marker. It
// does NOT assert that every review document has adopted it -- other reviewers own
// their files and adoption is theirs to opt into. Non-adopters are printed by name
// rather than passed over in silence, because the failure mode this test exists to
// prevent is a guard that protects nothing and reports success. If the marker is
// deleted everywhere, this goes red rather than vacuously green.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));

// Discovered, not enumerated: a hardcoded list stops covering whatever was added
// last, which is how four independent reviewers all missed the `ui/` test directory.
const REVIEW_DOCS = readdirSync(HERE)
  .filter((name) => /(REVIEW|BRIEF)/.test(name) && name.endsWith('.md'))
  .filter((name) => name !== 'REVIEW-POINT.md')
  .sort();

const MARKER = /^MEASURED-AT:\s*(\S+)\s*$/m;

function git(...args) {
  return execFileSync('git', args, { cwd: HERE, encoding: 'utf8' }).trim();
}

// THE THIRD EXIT STATE: 0 = clean, 1 = a defect was found, 2 = THIS GUARD COULD NOT RUN.
//
// Measured before this existed: run this file from a directory that is not a work
// tree -- a tarball extract, which is exactly how reviewers receive the branch -- and
// it exited 1 with THREE RED TESTS, the first reading "expected to discover at least
// 3 review documents, found 1". That sentence accuses the DOCUMENTS. The actual cause
// was that `git` had nothing to answer with. A reader would go looking for missing
// review files that were never missing.
//
// Exit 1 is byte-identical to "a checker ran and found a genuine defect", so a crash
// in our tooling reads as a finding against the branch. The python instruments in
// scripts/ already separate these; the JS guards did not, and that asymmetry was
// aimed at whoever reads the extract. This closes it for this file only -- the other
// JS guards that call `git` still report a missing work tree as a test failure.
//
// The refusal happens at import time, BEFORE any test is registered, because a
// refusal printed after the results is read as a footnote to numbers the reader has
// already believed.
function refuseToRun(reason) {
  process.stdout.write(
    `\nCANNOT_RUN (exit 2): ${reason}.\n` +
      `THIS IS NOT A FINDING ABOUT ANY REVIEW DOCUMENT.\n` +
      `This guard compares each document's MEASURED-AT commit against the branch's\n` +
      `history, so with no work tree it can measure nothing -- and every number it\n` +
      `would print below would describe the extraction, not the branch.\n` +
      `Re-run from a checkout of the repository.\n\n`,
  );
  process.exit(2);
}

try {
  const inside = execFileSync('git', ['rev-parse', '--is-inside-work-tree'], {
    cwd: HERE,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
  }).trim();
  if (inside !== 'true') refuseToRun(`git reports "${inside}" for --is-inside-work-tree`);
} catch {
  refuseToRun(`no git work tree contains ${HERE}`);
}

test('the review corpus is discoverable at all', () => {
  // Positive control. Without this the whole file passes when the glob is wrong,
  // which is the exact shape of every false green measured this session.
  assert.ok(
    REVIEW_DOCS.length >= 3,
    `expected to discover at least 3 review documents, found ${REVIEW_DOCS.length}: ${REVIEW_DOCS.join(', ')}`,
  );
});

test('every review document that declares a measurement SHA declares a real one', () => {
  const adopters = [];
  const abstainers = [];

  for (const doc of REVIEW_DOCS) {
    const match = MARKER.exec(readFileSync(join(HERE, doc), 'utf8'));
    if (match) adopters.push([doc, match[1]]);
    else abstainers.push(doc);
  }

  // Anti-vacuity floor. Not a completeness claim -- see the header.
  assert.ok(
    adopters.length >= 1,
    `no review document carries a MEASURED-AT marker. This guard would pass over an ` +
      `entirely stale corpus. Documents seen: ${REVIEW_DOCS.join(', ')}`,
  );

  // DRAINED-CORPUS GUARD. @12e42da8's rule: an exemption is a statement about RAW
  // EVIDENCE, and the moment an exempt file states a CONCLUSION the exemption is a
  // suppression. This guard's `adopters` loop silently skips every document that
  // carries no marker -- and both current abstainers publish verdicts. So the
  // abstention is recorded here BY NAME with its cost, and it self-expires:
  // adopting the marker makes this red until the name is removed, and a NEW
  // abstainer makes it red immediately. Green means the drain has not widened,
  // never that the corpus is complete.
  const KNOWN_ABSTAINERS = ['REVIEWER-BRIEF.md'];
  const unexpected = abstainers.filter((d) => !KNOWN_ABSTAINERS.includes(d));
  assert.deepEqual(
    unexpected,
    [],
    `${unexpected.join(', ')} publishes review conclusions but declares no ` +
      `MEASURED-AT, so this guard skips it silently. Either adopt the marker or ` +
      `add the file to KNOWN_ABSTAINERS with the reason -- an unrecorded skip is ` +
      `printed in the same column as a pass.`,
  );
  const retired = KNOWN_ABSTAINERS.filter((d) => !abstainers.includes(d));
  assert.deepEqual(
    retired,
    [],
    `${retired.join(', ')} now declares MEASURED-AT, so its entry in ` +
      `KNOWN_ABSTAINERS is stale. Remove it: an exemption that outlives its ` +
      `subject is how a corpus drains without any single commit being wrong.`,
  );
  console.log(
    `  corpus: ${adopters.length} checked, ${abstainers.length} abstaining (${abstainers.join(', ') || 'none'})`,
  );

  for (const [doc, declared] of adopters) {
    assert.match(
      declared,
      /^[0-9a-f]{7,40}$/,
      `${doc} anchors itself to "${declared}", which is not a raw hex SHA. A ref name ` +
        `re-points without warning -- review-0 moved 60 commits during one session.`,
    );

    let type = null;
    try {
      type = git('cat-file', '-t', declared);
    } catch {
      assert.fail(
        `${doc} declares MEASURED-AT ${declared}, which is not an object in this ` +
          `repository. Eight hex characters is also the shape of an agent id.`,
      );
    }
    assert.equal(
      type,
      'commit',
      `${doc} declares MEASURED-AT ${declared}, which is a ${type}, not a commit.`,
    );

    // The load-bearing assertion. Not "is it recent" -- recency is a judgement and
    // would go red on every fast branch. "Is it still on this history" is a fact:
    // if the measured commit is not an ancestor of HEAD, the document describes a
    // tree that this branch never passed through, and every row in it is unmoored.
    let isAncestor = true;
    try {
      git('merge-base', '--is-ancestor', declared, 'HEAD');
    } catch {
      isAncestor = false;
    }
    assert.ok(
      isAncestor,
      `${doc} was measured at ${declared}, which is NOT an ancestor of HEAD. The ` +
        `document describes a tree this branch is not on. Re-measure and update the ` +
        `marker, or the rows below it are claims about a history that was abandoned.`,
    );
  }

  if (abstainers.length > 0) {
    // Printed, never silent: a guard whose scope shrinks quietly is the defect.
    console.log(
      `  note: ${abstainers.length} review document(s) have not adopted MEASURED-AT: ` +
        `${abstainers.join(', ')}`,
    );
  }
});

// BACKWARD DRIFT -- the half the ancestor check above cannot see.
//
// Asserting that a document's SHA is an ancestor of HEAD proves it is on this
// history. It does NOT prove the document was measured against the tree reviewers
// are actually extracting. Those are different failures and only one of them has
// been instrumented all session.
//
// A review tag freezes the artifact and leaves the VERDICTS floating. Two reviewers
// tonight formed blocking verdicts four minutes and two minutes before a tag was cut,
// on SHAs that are ancestors of it. Both verdicts were true when written, both were
// false of the tagged tree, and neither reviewer had any way to notice: nothing
// re-scores a finding when the branch moves past it, and the finding does not know a
// tag was cut afterwards.
//
// THE REMEDY'S OWN TRAP, WHICH IS WHY THIS RESOLVES THE TAG ONCE AND PRINTS IT:
// the obvious check is `git merge-base --is-ancestor <mine> review-0`, and that
// re-introduces exactly the defect the hex requirement above exists to block --
// it anchors a freshness decision to a MUTABLE NAME. `review-0` moved 60 commits
// during this session. A check against a moving boundary silently changes which
// documents it condemns, and the run that condemned you is not reproducible from
// its own output. So the tag is resolved to a SHA once, that SHA is printed, and
// the printed SHA is what the assertion talks about.
test('no review document was measured before the tree reviewers extract', () => {
  // The review point is DECLARED, never inferred. An earlier version of this test
  // picked the newest review-* tag by commit date, which is a reasonable heuristic
  // and was WRONG: the lead designated review-1 (04:02), while the newest by date is
  // review-2 (04:19). The heuristic would have enforced a boundary nobody chose,
  // silently, on every review document, with a plausible justification attached.
  //
  // That is the more dangerous of the two failure modes. A wrong answer a human
  // states can be argued with; a wrong answer a test computes gets obeyed. So when
  // the declaration is missing this fails and names the candidates rather than
  // picking one -- refusing to answer is a valid measurement, inventing a
  // denominator is not.
  const declarationPath = join(HERE, 'REVIEW-POINT.md');
  let declaration = null;
  try {
    declaration = readFileSync(declarationPath, 'utf8');
  } catch {
    const candidates = git('tag', '--list', 'review-*').split('\n').filter(Boolean);
    assert.fail(
      `REVIEW-POINT.md is missing, so no tree is declared as the review point. ` +
        `Candidates, none authoritative: ${candidates.join(', ') || '(none)'}. ` +
        `Declare one rather than letting this test guess.`,
    );
  }

  const declared = /^REVIEW-POINT-SHA:\s*([0-9a-f]{7,40})\s*$/m.exec(declaration);
  assert.ok(
    declared,
    'REVIEW-POINT.md declares no REVIEW-POINT-SHA, or declares it as a ref name. ' +
      'It must be a raw hex SHA: a tag name re-points without warning.',
  );

  const boundary = git('rev-parse', `${declared[1]}^{commit}`);
  console.log(`  review point: ${declared[1]} -> ${boundary.slice(0, 8)}`);

  for (const doc of REVIEW_DOCS) {
    const match = MARKER.exec(readFileSync(join(HERE, doc), 'utf8'));
    if (!match) continue;
    const measuredAt = match[1];

    let predatesBoundary = false;
    try {
      git('merge-base', '--is-ancestor', measuredAt, boundary);
      predatesBoundary = git('rev-parse', measuredAt) !== boundary;
    } catch {
      predatesBoundary = false;
    }

    assert.ok(
      !predatesBoundary,
      `${doc} was measured at ${measuredAt}, which is an ancestor of the declared ` +
        `review point ${boundary.slice(0, 8)}. Every row in it describes a tree that ` +
        `the review has already moved past. Re-measure and update MEASURED-AT.`,
    );
  }
});
