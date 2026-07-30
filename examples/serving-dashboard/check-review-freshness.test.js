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
  .sort();

const MARKER = /^MEASURED-AT:\s*(\S+)\s*$/m;

function git(...args) {
  return execFileSync('git', args, { cwd: HERE, encoding: 'utf8' }).trim();
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
