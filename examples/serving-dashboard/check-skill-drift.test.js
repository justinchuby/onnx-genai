// A skill document may REFER to a guard. It may not RESTATE what the guard enforces.
//
// WHY THIS EXISTS
// ---------------
// `.github/skills/claims-of-absence/SKILL.md` and the guards it describes were
// written from the same understanding at the same hour, in two independent sets
// of words. That is two statements of one rule, and only one of them can go red.
// The guard is executable; the skill is prose an agent loads INSTEAD of reading
// the guard. So when the code moves, the skill keeps confidently describing a
// mechanism that no longer exists, to precisely the audience that cannot check.
//
// This is not hypothetical. It had ALREADY happened when this guard was written:
//
//   SKILL.md §6 listed `served-surface.test.js` among the guards that read
//   `git show HEAD:` rather than the working tree, and told the reader those
//   guards "cannot go green before you commit". served-surface.test.js does no
//   such thing — it reads the working tree with `readFileSync` (line 93). An
//   agent following that advice would commit to chase a red that was never
//   about committed bytes.
//
// The doc was wrong in the one direction documentation is never audited in, and
// nothing anywhere went red. That is the same defect class this branch exists to
// kill: a claim that cannot be falsified is a claim that will rot.
//
// WHAT THIS GUARD DOES, AND WHAT IT DELIBERATELY DOES NOT DO
// ---------------------------------------------------------
// The first design here was a text-similarity detector: shingle SKILL.md, shingle
// every guard, red on any long shared passage. It was BUILT AND MEASURED before
// being thrown away, and the measurement is why it is not here.
//
// Across the whole corpus, SKILL.md shares exactly ONE eight-word run with any
// guard ("the names the server would serve it under") plus a quoted server
// string. Verbatim duplication was already near zero. The duplication is
// SEMANTIC — the same rule explained twice in different words — so a similarity
// detector would have shipped permanently green while the real hazard sat
// untouched. A green that was never capable of red is the thing this repo keeps
// getting caught by, and it is not worth adding another one.
//
// So this guard checks the property that survives a rewording: every mechanism
// SKILL.md names must still EXIST, in the file SKILL.md says it lives in, and
// must still have the property SKILL.md ascribes to it. Reword the doc freely.
// Delete the mechanism and the doc goes red.
//
// THE DIVISION OF AUTHORITY THIS ENCODES
// --------------------------------------
//   The GUARD is authoritative for how a rule is enforced here. It sits beside
//   the code, it runs, and it fails.
//   The SKILL is authoritative for the transferable lesson — the shape of the
//   mistake, and what to do in code this repo has not written yet.
//   Where they overlap, the skill CITES and the guard STATES.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

import { assertShippingTree, REPO_ROOT, SHIPPING_REF } from './shipping-tree.mjs';

assertShippingTree();

const SKILL_PATH = '.github/skills/claims-of-absence/SKILL.md';

/**
 * Read a path as COMMITTED, not as it sits on disk.
 *
 * SKILL.md ships to agents from the repository, and an agent that loads it is
 * reading committed bytes. Auditing the working copy would let an uncommitted
 * edit turn this guard green for a document nobody else can see — the exact
 * one-tree/other-tree confusion `shipping-tree.mjs` exists to prevent.
 *
 * Resolved from REPO_ROOT because SKILL.md lives above this directory.
 */
function shippedFromRoot(rel) {
  return execFileSync('git', ['show', `${SHIPPING_REF}:./${rel}`], {
    cwd: REPO_ROOT,
    maxBuffer: 16 * 1024 * 1024,
  }).toString();
}

/** Every path tracked at the shipping ref, for citation-existence checks. */
function trackedPaths() {
  return new Set(
    execFileSync('git', ['ls-tree', '-r', '--name-only', SHIPPING_REF], {
      cwd: REPO_ROOT,
      maxBuffer: 16 * 1024 * 1024,
    })
      .toString()
      .split('\n')
      .filter(Boolean),
  );
}

const SKILL_TEXT = shippedFromRoot(SKILL_PATH);

/**
 * File paths SKILL.md cites, extracted from backticked spans.
 *
 * Only backticked spans count. Prose mentions a filename in passing; a backtick
 * is the doc asserting "this artefact exists and you can go read it", which is
 * the claim worth policing.
 */
function citedPaths(text) {
  const cited = new Set();
  for (const [, span] of text.matchAll(/`([^`\n]+)`/g)) {
    const token = span.trim();
    if (!/\.(?:js|mjs|md|rs|sh|json)$/.test(token)) continue;
    if (token.includes(' ')) continue;
    cited.add(token);
  }
  return [...cited];
}

/**
 * Resolve a cited path against the tracked corpus.
 *
 * SKILL.md cites some files by bare basename (`field-keys.test.js`) and some by
 * repo-relative path. A bare basename resolves if exactly one tracked file ends
 * with it; an AMBIGUOUS basename is reported, because a citation that could mean
 * two files sends the reader to the wrong one — the same attribution cost
 * `check-test-basenames.test.js` polices for failure reports.
 */
function resolveCitation(citation, tracked) {
  if (tracked.has(citation)) return { matches: [citation] };
  const matches = [...tracked].filter(
    (p) => p === citation || p.endsWith(`/${citation}`),
  );
  return { matches };
}

/**
 * Mechanisms SKILL.md describes, and the falsifier for each.
 *
 * Each entry is a claim the DOCUMENT makes about the CODE. `symbol` must appear
 * in `file`; if the doc says a guard reads committed bytes, `readsCommittedBytes`
 * requires it to actually do so.
 *
 * Adding a mechanism to SKILL.md without adding it here is not punished by this
 * guard and is not meant to be — the coverage floor below is what stops the
 * table from being quietly emptied.
 */
const CITED_MECHANISMS = [
  {
    claim: 'a deferral entry must declare the wire names that would falsify it',
    file: 'examples/serving-dashboard/check-unplumbed-claims.test.js',
    symbol: 'absentWireNames',
  },
  {
    claim: 'a permanent control proves the scanner strips comments before matching',
    file: 'examples/serving-dashboard/check-unplumbed-claims.test.js',
    symbol: 'COMMENT_ONLY_CONTROL',
  },
  {
    claim: 'call sites are reconciled against the provenance catalogue',
    file: 'examples/serving-dashboard/dashboard/field-keys.test.js',
    symbol: 'KEY_LITERAL',
  },
  {
    claim: 'deferrals are carried in a named, evidenced list rather than silence',
    file: 'examples/serving-dashboard/dashboard/field-keys.test.js',
    symbol: 'NOT_YET_PUBLISHED',
  },
  {
    claim: 'an exact-set ratchet fails until it is shrunk in the repairing commit',
    file: 'examples/serving-dashboard/telemetry-key-namespace.test.js',
    symbol: 'KNOWN_UNSERVABLE_KEYS',
  },
  {
    claim: 'a counted ratchet pins how much served-but-unused surface is tolerated',
    file: 'examples/serving-dashboard/served-surface.test.js',
    symbol: 'MAX_SERVED_BUT_NOT_NEEDED',
  },
  {
    claim: 'some guards audit committed bytes, so they cannot go green pre-commit',
    file: 'examples/serving-dashboard/caption-catalogue.test.js',
    symbol: 'shippedSources',
    readsCommittedBytes: true,
  },
  {
    claim: 'some guards audit committed bytes, so they cannot go green pre-commit',
    file: 'examples/serving-dashboard/check-perf-claims.test.js',
    symbol: 'git',
    readsCommittedBytes: true,
  },
];

/**
 * Evidence that a file reads COMMITTED bytes rather than the working tree.
 *
 * `SHIPPING_REF` is included because the house form is a resolved SHA, never the
 * literal string 'HEAD' — spelling 'HEAD' lets one run read several trees. A
 * guard that reaches for the shipping helpers is reading what a reviewer clones.
 */
const COMMITTED_READ_EVIDENCE = /shipped\s*\(|shippedSources|shippedPaths|SHIPPING_REF|git\s*\(\s*'show'|\['show'/;

describe('SKILL.md refers to guards without restating them', () => {
  it('every file SKILL.md cites exists at the shipping ref, unambiguously', () => {
    const tracked = trackedPaths();
    const cited = citedPaths(SKILL_TEXT);

    const missing = [];
    const ambiguous = [];
    for (const citation of cited) {
      const { matches } = resolveCitation(citation, tracked);
      if (matches.length === 0) missing.push(citation);
      else if (matches.length > 1) ambiguous.push(`${citation} -> ${matches.join(', ')}`);
    }

    assert.deepEqual(
      missing,
      [],
      `${SKILL_PATH} cites files that do not exist at the shipping ref:\n` +
        `${missing.map((m) => `  ${m}`).join('\n')}\n\n` +
        'A skill is loaded INSTEAD of reading the code, so a citation that points ' +
        'nowhere is not a broken link — it is advice about a mechanism the reader ' +
        'cannot check and will assume is there.',
    );

    assert.deepEqual(
      ambiguous,
      [],
      `${SKILL_PATH} cites a basename that matches more than one tracked file:\n` +
        `${ambiguous.map((m) => `  ${m}`).join('\n')}\n\n` +
        'Cite the repo-relative path. An ambiguous citation sends the reader to ' +
        'the wrong file with full confidence.',
    );
  });

  it('every mechanism SKILL.md describes still exists in the file it names', () => {
    const tracked = trackedPaths();
    const broken = [];

    for (const entry of CITED_MECHANISMS) {
      if (!tracked.has(entry.file)) {
        broken.push(`${entry.file} is not tracked at the shipping ref (claim: ${entry.claim})`);
        continue;
      }
      const text = shippedFromRoot(entry.file);
      if (!text.includes(entry.symbol)) {
        broken.push(
          `${entry.file} no longer contains '${entry.symbol}'\n` +
            `      SKILL.md still tells the reader: ${entry.claim}`,
        );
      }
    }

    assert.deepEqual(
      broken,
      [],
      'SKILL.md describes mechanisms that have moved or been deleted:\n' +
        `${broken.map((b) => `  ${b}`).join('\n')}\n\n` +
        'Either restore the mechanism or update the skill. Prose describing a ' +
        'mechanism that no longer exists is worse than no prose: it is confident ' +
        'and wrong, and it is read by agents who will not verify it.',
    );
  });

  it('guards SKILL.md calls committed-bytes readers actually read committed bytes', () => {
    // THE REGRESSION THIS GUARD WAS BUILT FOR.
    //
    // SKILL.md told readers that a named set of guards audit committed bytes and
    // therefore "cannot go green before you commit". For served-surface.test.js
    // that was FALSE — it reads the working tree via readFileSync. The advice
    // inverted the correct diagnosis: an agent would commit to chase a red that
    // had nothing to do with committed bytes.
    const wrong = [];

    for (const entry of CITED_MECHANISMS) {
      if (!entry.readsCommittedBytes) continue;
      const text = shippedFromRoot(entry.file);
      if (!COMMITTED_READ_EVIDENCE.test(text)) {
        wrong.push(
          `${entry.file} is described as reading committed bytes, but contains no ` +
            'shipping-ref read (no shipped()/shippedSources/SHIPPING_REF/git show)',
        );
      }
    }

    assert.deepEqual(
      wrong,
      [],
      'SKILL.md misdescribes how these guards read their corpus:\n' +
        `${wrong.map((w) => `  ${w}`).join('\n')}\n\n` +
        'This claim changes what a reader DOES — it tells them to commit before ' +
        're-running. Naming a working-tree guard here sends them to commit for a ' +
        'red that committing cannot fix.',
    );
  });

  it('SKILL.md classifies each named guard by how it actually reads its corpus', () => {
    // THE REGRESSION THIS GUARD WAS BUILT FOR, checked in BOTH directions.
    //
    // SKILL.md §6 carries two labelled lines: guards that read committed bytes,
    // and guards that read the working tree. Both are asserted. Checking only
    // the positive line would let the counter-example — the file that caused the
    // original defect — sit unverified, and an unverified counter-example is
    // just an exception with a story attached.
    //
    // ANCHORED ON HEADING NUMBERS, NOT ON PROSE. The first version anchored on
    // the phrase 'git show', which the very edit that fixed the defect deleted
    // from the heading — silently collapsing the section to zero length and
    // making this test vacuous. Anchors must not be phrases the fix is likely
    // to touch; the emptiness assertion below is what caught that.
    const start = SKILL_TEXT.indexOf('## 6.');
    const end = SKILL_TEXT.indexOf('## 7.');
    const section = start >= 0 && end > start ? SKILL_TEXT.slice(start, end) : '';

    assert.ok(
      section.length > 0,
      'Could not locate section 6 of SKILL.md (the committed-bytes section). ' +
        'This test is anchored on its heading number; if the sections were ' +
        're-numbered, re-anchor it rather than deleting the check — a section ' +
        'that resolves to the empty string passes every assertion below it.',
    );

    /** Pull one labelled bullet's citations, including its wrapped continuation lines. */
    function labelled(label) {
      const at = section.indexOf(`- ${label}:`);
      if (at < 0) return null;
      const rest = section.slice(at + label.length + 3);
      const stop = rest.search(/\n\s*\n|\n- /);
      return citedPaths(stop < 0 ? rest : rest.slice(0, stop)).filter((c) =>
        c.endsWith('.test.js'),
      );
    }

    const committed = labelled('Reads committed bytes');
    const workingTree = labelled('Reads the working tree');

    assert.ok(
      committed && workingTree,
      "SKILL.md §6 no longer carries both labelled lists ('Reads committed " +
        "bytes:' and 'Reads the working tree:'). This test reads those labels; " +
        'restore them or re-anchor, but do not leave the classification unchecked.',
    );

    const tracked = trackedPaths();
    const misclassified = [];

    for (const [citation, shouldReadCommitted] of [
      ...committed.map((c) => [c, true]),
      ...workingTree.map((c) => [c, false]),
    ]) {
      const { matches } = resolveCitation(citation, tracked);
      if (matches.length !== 1) {
        misclassified.push(`${citation} does not resolve to exactly one tracked file`);
        continue;
      }
      const reads = COMMITTED_READ_EVIDENCE.test(shippedFromRoot(matches[0]));
      if (reads !== shouldReadCommitted) {
        misclassified.push(
          `${matches[0]} is listed as reading ${shouldReadCommitted ? 'committed bytes' : 'the working tree'}, ` +
            `but it reads ${reads ? 'committed bytes' : 'the working tree'}`,
        );
      }
    }

    assert.deepEqual(
      misclassified,
      [],
      'SKILL.md §6 misclassifies how these guards read their corpus:\n' +
        `${misclassified.map((m) => `  ${m}`).join('\n')}\n\n` +
        'This claim changes what a reader DOES — it tells them whether to commit ' +
        'before re-running. Listing a working-tree guard as committed-bytes sends ' +
        'them to commit for a red that committing cannot fix.',
    );

    assert.ok(
      committed.length >= 2 && workingTree.length >= 1,
      `Section 6 names ${committed.length} committed-bytes and ${workingTree.length} ` +
        'working-tree guard(s). Both lists are satisfied when empty, so an empty ' +
        'one is a green that classified nothing.',
    );
  });

  it('the audit actually reaches SKILL.md and can report a fault', () => {
    // ANTI-VACUITY. Every assertion above passes over an empty citation set, and
    // an empty set is the likeliest failure of the extractor: tighten the
    // backtick regex slightly and it silently matches nothing, forever green.
    const cited = citedPaths(SKILL_TEXT);

    assert.ok(
      SKILL_TEXT.length > 2000,
      `${SKILL_PATH} read back as ${SKILL_TEXT.length} bytes. That is too short ` +
        'to be the skill; the read is broken, not the document.',
    );

    assert.ok(
      cited.length >= 6,
      `Only ${cited.length} file citations extracted from ${SKILL_PATH}. The ` +
        'extractor is not reading the document it claims to audit.',
    );

    assert.ok(
      CITED_MECHANISMS.length >= 6,
      `CITED_MECHANISMS holds ${CITED_MECHANISMS.length} entries. The table has ` +
        'been drained; a mechanism audit over an empty table is a green that ' +
        'checked nothing.',
    );

    // POSITIVE CONTROL: a citation known to be in the document IS extracted.
    assert.ok(
      cited.some((c) => c.endsWith('field-keys.test.js')),
      'The extractor cannot see field-keys.test.js, which SKILL.md cites. The ' +
        'instrument is not reading the corpus it claims to read.',
    );

    // NEGATIVE CONTROL: a path that is not tracked IS reported as missing.
    const tracked = trackedPaths();
    const bogus = resolveCitation('definitely-not-a-real-guard.test.js', tracked);
    assert.equal(
      bogus.matches.length,
      0,
      'The resolver matched a deliberately bogus citation, so it cannot ' +
        'distinguish a real file from an invented one.',
    );

    // POSITIVE CONTROL on the committed-read detector: it must be able to say YES.
    assert.ok(
      COMMITTED_READ_EVIDENCE.test(shippedFromRoot('examples/serving-dashboard/caption-catalogue.test.js')),
      'The committed-read detector cannot recognise caption-catalogue.test.js, ' +
        'which demonstrably reads committed bytes. A detector that only ever ' +
        'says NO would fail every file for the wrong reason.',
    );
  });
});
