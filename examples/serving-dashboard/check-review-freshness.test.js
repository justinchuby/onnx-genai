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

// WHY THE ANCHOR IS GONE. This was /^MEASURED-AT:\s*(\S+)\s*$/ -- end-of-line anchored.
// Two of the four declarations on this branch write prose after the SHA on the same
// line ("MEASURED-AT: `9b06d922`. Not at review-2, and the refusal is..."), so they did
// not match AT ALL -- not mis-parsed, INVISIBLE. The newest declaration in the densest
// review document was one of them. Capturing the first token and ignoring the rest of
// the line costs nothing and is what every author already assumed the rule was.
const MARKER = /^MEASURED-AT:\s*(\S+)/gm;

// WHY EVERY DECLARATION AND NOT THE FIRST. This read `MARKER.exec(text)` until 06:40,
// which returns match ONE. Re-measuring is an APPEND -- the project lead's own
// amendment mandates appending in shared regions -- so the first marker in a file is
// its OLDEST. The guard was therefore judging every document by the first measurement
// it ever took and could never see a re-measurement. It read 4 of 8 live declarations,
// and the 4 it skipped were the 4 that were current.
//
// WHY THE STRIPPING. Three declarations on this branch are written  MEASURED-AT: `sha`.
// with backticks and a full stop, because markdown authors write markdown. `(\S+)`
// captures those delimiters, `git rev-parse` rejects the result, and the catch below
// scored it "stale". The delimiters are formatting, not identity.
//
// These two defects concealed each other: widening to matchAll() without stripping
// turns this RED on three valid commits, and whoever did that would conclude they had
// broken the guard and revert BOTH fixes. Order matters; they ship together.
function declarationsIn(text) {
  return [...text.matchAll(MARKER)]
    .map(([, raw]) => raw.replace(/^[`'"(<[]+|[`'"),.\]>]+$/g, ''))
    .filter((sha) => /^[0-9a-f]{7,40}$/.test(sha));
}

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
    const declared = declarationsIn(readFileSync(join(HERE, doc), 'utf8'));
    if (declared.length > 0) adopters.push([doc, declared[declared.length - 1]]);
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
  // EMPTY, AND THAT IS THE SUCCESS CONDITION, NOT A DISABLED CHECK.
  //
  // This list held two entries at 05:23. IMPLEMENTATION-REVIEW.md adopted the marker
  // and the assertion below retired it at 05:34; REVIEWER-BRIEF.md adopted it and was
  // retired at 05:43. Both times the guard went red naming the file and the remedy,
  // and both times someone cleared it without discussion. An exemption list that
  // empties itself is the only kind that cannot quietly become a suppression list.
  //
  // An empty list is NOT vacuous here: the `unexpected` assertion above still reddens
  // the moment any review document appears without a MEASURED-AT marker, and the
  // discovery floor still reddens if the corpus itself disappears.
  const KNOWN_ABSTAINERS = [];
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

  // Resolve HEAD ONCE, before the loop. This call site names HEAD a single time
  // in the source, but it used to sit inside the loop below -- so it re-resolved
  // per document, and on a branch taking ~79 commits an hour the first document
  // and the last were scored against different trees. One verdict, two subjects.
  // The syntactic rule ("never name HEAD twice") does not catch this shape: one
  // occurrence inside a loop IS N occurrences at runtime.
  const headSha = git('rev-parse', 'HEAD');
  console.log(`  scored against: ${headSha}`);

  // Collect every document's verdict, then assert once. This loop used to assert
  // inside itself, so the FIRST bad document threw and the rest were never
  // examined -- while the corpus line above still announced all of them as
  // "checked". A printed denominator that counts what was ENUMERATED rather than
  // what was VERIFIED is the same false-green we have been hunting all night, and
  // it was in my own guard. One red document must not conceal three others.
  const problems = [];
  let verified = 0;

  for (const [doc, declared] of adopters) {
    if (!/^[0-9a-f]{7,40}$/.test(declared)) {
      problems.push(
        `${doc} anchors itself to "${declared}", which is not a raw hex SHA. A ref ` +
          `name re-points without warning -- review-0 moved 60 commits in one session.`,
      );
      continue;
    }

    let type = null;
    try {
      type = git('cat-file', '-t', declared);
    } catch {
      problems.push(
        `${doc} declares MEASURED-AT ${declared}, which is not an object in this ` +
          `repository. Eight hex characters is also the shape of an agent id.`,
      );
      continue;
    }
    if (type !== 'commit') {
      problems.push(
        `${doc} declares MEASURED-AT ${declared}, which is a ${type}, not a commit.`,
      );
      continue;
    }

    // The load-bearing check. Not "is it recent" -- recency is a judgement and
    // would go red on every fast branch. "Is it still on this history" is a fact:
    // if the measured commit is not an ancestor of the pinned head, the document
    // describes a tree this branch never passed through, and every row is unmoored.
    let isAncestor = true;
    try {
      git('merge-base', '--is-ancestor', declared, headSha);
    } catch {
      isAncestor = false;
    }
    if (!isAncestor) {
      problems.push(
        `${doc} was measured at ${declared}, which is NOT an ancestor of ${headSha}. ` +
          `The document describes a tree this branch is not on. Re-measure and update ` +
          `the marker, or the rows below it are claims about a history that was ` +
          `abandoned. (The head is named as a SHA, not as "HEAD", so this verdict is ` +
          `reproducible after the branch moves on.)`,
      );
      continue;
    }
    verified += 1;
  }

  // Anti-vacuity: every document must reach a verdict. If this ever fails, the
  // loop above grew a path that neither verifies nor complains -- a silent skip
  // printed in the same column as a pass.
  assert.equal(
    verified + problems.length,
    adopters.length,
    `${adopters.length} documents were enumerated but only ${verified + problems.length} ` +
      `reached a verdict. A document that is neither verified nor reported is a hole ` +
      `in the corpus, not a pass.`,
  );
  // Say WHICH predicate this verdict is about. "fresh"/"stale" was used here AND by the
  // review-point test below for two different questions -- "does every declared SHA
  // resolve and sit on this branch" versus "is any declared SHA at or after the review
  // point" -- so one run printed "0 stale" in the same breath as it failed for
  // staleness. One word, two predicates, and a reader has no way to tell which one just
  // spoke. This is the co-location defect in its smallest form: a summary line that does
  // not name its own subject.
  console.log(`  on-branch: ${verified} resolved, ${problems.length} unresolved-or-off-branch`);

  assert.deepEqual(
    problems,
    [],
    `${problems.length} of ${adopters.length} review document(s) declare a SHA that does ` +
      `not resolve or is not on this branch at ${headSha}:\n  - ${problems.join('\n  - ')}`,
  );

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
  // Print HOW FAR BACK the boundary sits, because this check's strength is entirely a
  // function of a value chosen for other reasons. REVIEW-POINT-SHA is picked to name the
  // tree reviewers should read CODE at -- stability is a virtue there, so it is chosen
  // OLD on purpose. But this test uses it as a FRESHNESS BOUNDARY, where old is weak:
  // measured here, moving the pin from d5da0061 back to 37d0d72e (205 commits) takes this
  // test from 2 stale documents to 0 without a single document being re-measured.
  // One field, two consumers, opposite requirements -- and the pin's own reason to move is
  // invisible from inside this test. So the distance ships beside the verdict: a green
  // against a boundary hundreds of commits behind HEAD is not the same green as one
  // against a recent boundary, and nobody can tell them apart from a pass/fail alone.
  const behind = git('rev-list', '--count', `${boundary}..HEAD`);
  console.log(`  review point: ${declared[1]} -> ${boundary.slice(0, 8)} (${behind} commits behind HEAD)`);
  if (Number(behind) > 50) {
    console.log(
      `  WEAK BOUNDARY: ${behind} commits behind HEAD. A document measured anywhere in ` +
        `that window scores fresh. This test is only as strong as the pin is recent.`,
    );
  }

  const stale = [];
  for (const doc of REVIEW_DOCS) {
    const declared = declarationsIn(readFileSync(join(HERE, doc), 'utf8'));
    if (declared.length === 0) continue;

    // A document is fresh if ANY of its declarations is at-or-after the review point.
    // Not the last one positionally: nothing forces append order, and a document that
    // was re-measured is fresh regardless of where on the page it said so.
    // PEEL BOTH SIDES. `merge-base --is-ancestor` peels an annotated tag automatically;
    // `rev-parse` does NOT. That asymmetry lived inside this one comparison and made it
    // accuse the most compliant possible author: the lead's order is "write the raw hex",
    // the obvious way to obtain it is `git rev-parse review-3`, and for an annotated tag
    // that returns the TAG OBJECT (02249627), not the commit (37d0d72e). is-ancestor said
    // TRUE, rev-parse said "different object", and the doc was scored stale for naming the
    // review point exactly. Only review-3 is annotated, so this was one tag away from
    // firing at three reviewers at once.
    const fresh = declared.filter((measuredAt) => {
      // RESOLVE FIRST, AND SEPARATELY. `merge-base --is-ancestor` exits non-zero for two
      // unrelated reasons: "resolved, and is not an ancestor" -- which means the
      // measurement sits at or after the boundary, i.e. genuinely fresh -- and "did not
      // resolve at all", which means we can tell nothing. A single catch collapsed both
      // into `return true`, so an UNREADABLE marker was scored FRESH. That is the
      // fail-open shape, and it is the same line as `?? SOURCE_BADGES.derived` and
      // `else { return true }` in the asset guard: unrecognised input granted the
      // permissive answer. A marker that does not resolve is now stale, loudly -- which
      // is the entire reason the marker was specified as a bare hex in the first place.
      let commit;
      try {
        commit = git('rev-parse', `${measuredAt}^{commit}`);
      } catch {
        return false;
      }
      try {
        git('merge-base', '--is-ancestor', commit, boundary);
        return commit === boundary;
      } catch {
        return true; // resolved, and not an ancestor => at or after the boundary
      }
    });

    if (fresh.length === 0) stale.push(`${doc} -- ${declared.join(', ')}`);
  }

  // Report EVERY stale document, not merely the first. This assertion used to sit INSIDE
  // the loop above, so it threw on the alphabetically-first offender and never evaluated
  // the rest: an N-document audit silently degraded into a one-document report, and the
  // number it never printed was the only number worth having. It accused
  // ARCHITECTURE-SECURITY-REVIEW.md for hours; the moment that document was repaired it
  // accused IMPLEMENTATION-REVIEW.md instead -- and READABILITY-REVIEW.md, this guard's
  // own author's document, carrying 22 declarations with not one of them fresh, had been
  // stale behind both of them the whole time and was never once named.
  // A guard that stops at its first failure hides its own denominator, and sorts its
  // author's own file to the back of the queue.
  assert.ok(
    stale.length === 0,
    `${stale.length} of ${REVIEW_DOCS.length} review document(s) declare ONLY SHAs that ` +
      `are strict ancestors of the review point ${boundary.slice(0, 8)}, so every row in ` +
      `them describes a tree the review has already moved past. Re-measure and add a ` +
      `current MEASURED-AT to each of:\n    ${stale.join('\n    ')}`,
  );
});
