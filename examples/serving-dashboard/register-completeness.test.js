// Copyright (c) Microsoft Corporation.
//
// The provenance register is the page's court of last resort: a sceptical
// visitor consults it BECAUSE they distrust the panels. These are the checks
// that the register agrees with the table it claims to be generated from.
//
// WHY THIS FILE EXISTS. Two defects shipped to a live page at once, and both
// were invisible to every other guard in the tree.
//
//  1. The register rendered the BASE classification for every row. It was the
//     one consumer of the provenance table that never called resolveForOrigin,
//     so on the batching server it certified the three prefix-cache counters as
//     "Measured by the server" while the panels beside it — reading the SAME
//     table, correctly resolved — showed them structurally bypassed. The two
//     halves of the page disagreed about the same fact, and the half that was
//     wrong was the half that exists to be trusted.
//
//  2. The classification-to-English map had no entry for
//     STRUCTURALLY_BYPASSED, and the lookup fell back to `?? entry.classification`.
//     So the register printed the raw constant — an internal enum token, in
//     screaming snake case, in a table cell, beside nine carefully written
//     English sentences. A `??` fallback on a display string cannot fail; it
//     renders something, and something is indistinguishable from working.
//
// Neither is visible from either artefact alone. app.js is internally
// consistent, telemetry-provenance.js is internally consistent, and the defect
// is in the relationship. app.js self-starts on import (it calls main() at the
// bottom), so the map is read off disk rather than imported, the same way
// state-treatments.test.js reads shell.css.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { PROVENANCE, allFieldKeys, resolveForOrigin } from './telemetry-provenance.js';

const appSource = readFileSync(new URL('./app.js', import.meta.url), 'utf8');
const fieldSource = readFileSync(new URL('./telemetry-field.js', import.meta.url), 'utf8');
const provenanceSource = readFileSync(
  new URL('./telemetry-provenance.js', import.meta.url),
  'utf8',
);

/** Every origin the register can be rendered for, plus the unresolved case. */
const ORIGINS = Object.freeze([null, 'scatter', 'dynamic']);

/** The keys of CLASSIFICATION_TEXT, read from app.js rather than restated. */
function labelledClassifications(source) {
  const block = source.match(/const CLASSIFICATION_TEXT = Object\.freeze\(\{([\s\S]*?)\n\}\);/);
  assert.ok(
    block,
    'CLASSIFICATION_TEXT was not found in app.js. If it was renamed or reshaped, ' +
      'update this matcher — do NOT delete the test. A parser that silently ' +
      'matches nothing reports a clean bill of health, which is the one failure ' +
      'mode indistinguishable from success.',
  );
  const keys = [...block[1].matchAll(/^\s{2}([A-Z_]+):/gm)].map((m) => m[1]);
  assert.ok(
    keys.length > 0,
    'Parsed CLASSIFICATION_TEXT but found zero keys. See above: an under-matching ' +
      'checker passes for the wrong reason.',
  );
  return new Set(keys);
}

/** The `Classification` union, read from its typedef rather than restated. */
function declaredClassifications() {
  const union = provenanceSource.match(/@typedef \{([^}]*)\} Classification/);
  assert.ok(
    union,
    'The Classification typedef was not found in telemetry-provenance.js. If it ' +
      'was renamed, update this matcher — do NOT delete the test. A parser that ' +
      'silently matches nothing reports a clean bill of health, which is the one ' +
      'failure mode indistinguishable from success.',
  );
  const names = [...union[1].matchAll(/'([A-Z_]+)'/g)].map((m) => m[1]);
  assert.ok(names.length > 0, 'Parsed the Classification typedef and found zero members.');
  return new Set(names);
}

/** Every classification actually reachable, at any origin, after resolution. */
function reachableClassifications() {
  const reached = new Set();
  for (const key of allFieldKeys()) {
    for (const origin of ORIGINS) {
      reached.add(resolveForOrigin(PROVENANCE[key], origin).classification);
    }
  }
  return reached;
}

test('every declared classification has English text for the visitor', () => {
  // Checked against the DECLARED vocabulary, not against what entries happen to
  // use today. DOCUMENTED_ZERO is currently declared and documented but unused;
  // requiring its text now means the leak cannot reappear the day someone
  // classifies a field that way. Waiting for a member to be reachable before
  // requiring its label is waiting for the defect to ship.
  const labelled = labelledClassifications(appSource);
  const unlabelled = [...declaredClassifications()].filter((c) => !labelled.has(c));

  assert.deepEqual(
    unlabelled,
    [],
    `The register can render ${unlabelled.join(', ')} and has no sentence for it. ` +
      'The visitor sees the raw constant — an internal identifier presented as a ' +
      'status, in the table that exists to prove we are not hiding anything.',
  );
});

test('the register invents no status of its own', () => {
  const declared = declaredClassifications();
  const invented = [...labelledClassifications(appSource)].filter((c) => !declared.has(c));

  assert.deepEqual(
    invented,
    [],
    `CLASSIFICATION_TEXT describes ${invented.join(', ')}, which is not in the ` +
      'Classification union. Either the vocabulary gained a member without its ' +
      'documentation, or the register is describing a status that does not exist.',
  );
});

test('no entry carries a classification outside the vocabulary', () => {
  // Classifications are bare strings in a large hand-edited object literal, so a
  // typo produces a value that is in no list — and being in no list means being
  // in neither NEVER_MEASURED_CLASSIFICATIONS nor any suppression check, so a
  // misspelled classification FAILS OPEN and the field renders as measured.
  const declared = declaredClassifications();
  const undeclared = [...reachableClassifications()].filter((c) => !declared.has(c));

  assert.deepEqual(
    undeclared,
    [],
    `${undeclared.join(', ')} is used as a classification but is not declared. A ` +
      'classification in no list is suppressed by nothing: the field renders as a ' +
      'measured value, which is the exact failure this table exists to prevent.',
  );
});

test('the register resolves per origin, so it cannot contradict the panels', () => {
  // The regression that motivated the file. The bug was a MISSING ARGUMENT, so
  // asserting the call is not pedantry: the previous code read `PROVENANCE[key]`
  // directly and every count, key and label in the table was still correct.
  assert.match(
    appSource,
    /resolveForOrigin\(PROVENANCE\[key\], origin\)/,
    'The provenance register must resolve each entry for the origin that served ' +
      'the page. Reading the base entry makes it disagree with the panels on any ' +
      'field whose classification is per-origin — silently, and only on one server.',
  );
  assert.ok(
    !/CLASSIFICATION_TEXT\[[^\]]+\]\s*\?\?/.test(appSource),
    'A `??` fallback on the classification text renders the raw enum token to a ' +
      'visitor. classificationText() throws instead: unlabelled is a bug, not a ' +
      'display state.',
  );
});

test('the prefix-cache counters are not certified as measured on the batching server', () => {
  // The specific live defect, pinned so it cannot come back by another route.
  // This server never consults the prefix cache, so a register row claiming the
  // counters are "measured by the server" certifies a feature this project
  // cut — on the one surface a sceptic reads precisely because they doubt us.
  const counters = allFieldKeys().filter((key) => /prefix_cache/.test(key));
  assert.ok(counters.length >= 3, 'expected the prefix-cache counter family in the register');

  for (const key of counters) {
    const entry = resolveForOrigin(PROVENANCE[key], 'scatter');
    assert.notEqual(
      entry.classification,
      'MEASURED',
      `${key} resolves to MEASURED on the batching origin, which never runs that ` +
        'code path. The register would print "Measured by the server" beside ' +
        'evidence explaining that nothing measured it.',
    );
  }
});

// ---------------------------------------------------------------------------
// Visitor-facing citations. @fc8b5d97 counted 24 table cells rendering
// file:LINE to a visitor. A line number is the one part of a citation that
// rots, and a rendered table is where it rots unseen: a visitor cannot
// re-resolve it, and unlike a code comment nobody re-reads it against the tree.

// app.js is a browser entry with load-time side effects, so it cannot be
// imported here. The previous version of this file coped by COPYING the
// transform's regex, which made this suite a second implementation of the
// thing it audits: deleting `rs` from the shipped regex left every `.rs:511`
// citation rendering its line number to a visitor and all 8 tests still
// passed, because they were exercising the copy. So the pattern is LIFTED
// from app.js's source text instead. It cannot drift, because there is only
// one of it.
const SHIPPED_STRIPPER = (() => {
  // Anchored on the trailing `, '$1')` rather than on the closing delimiter:
  // the pattern contains an unescaped `/` inside its own character class, so a
  // non-greedy scan to the first `/` truncates it into an invalid expression.
  // REPOINTED. The stripper used to live inline in app.js and sanitised the
  // Evidence column only -- so 8 byOrigin `reason` strings carried a raw
  // `metrics.rs:232-237` to the page through format.js and model-card.js,
  // which this suite could not see because it was auditing the wrong channel.
  // The transform now lives in telemetry-field.js and is applied at the field
  // constructors, so BOTH channels are covered by one implementation. This
  // extractor follows it there rather than keeping a copy.
  const match = fieldSource.match(
    /return text\.replace\(\s*\/(.+)\/([gimsuy]*)\s*,\s*'\$1'\);/,
  );
  assert.ok(
    match,
    'Could not locate the citation stripper in telemetry-field.js. This assertion ' +
      'is the non-vacuity guard for every citation test below: if the transform is ' +
      'renamed or restructured, they must fail loudly rather than silently ' +
      'auditing a regex that no longer ships.',
  );
  // And app.js must still CONSUME it rather than re-growing its own. Two
  // sanitisers with one job drift the moment either is fixed, silently, because
  // both continue to look like they work.
  assert.match(
    appSource,
    /citationForVisitor\s*=\s*withoutSourceCitations/,
    'app.js no longer imports the shared citation stripper',
  );
  try {
    return new RegExp(match[1], match[2] || '');
  } catch (cause) {
    throw new Error(
      `Extracted an unusable pattern from telemetry-field.js: /${match[1]}/${match[2]}. ` +
        'The extractor, not the shipped transform, is what needs fixing here.',
      { cause },
    );
  }
})();

/** The render-time citation transform, as app.js actually ships it. */
function citationForVisitor(evidence) {
  return evidence.replace(SHIPPED_STRIPPER, '$1');
}

// Deliberately NOT the stripper's extension list. A detector built from the
// same assumption as the thing it checks is blind in exactly the same place:
// the stripper knows rs|js|toml|md, so a `build_qwen.sh:99` citation — a form
// this repository actually uses — would sail past both and reach the visitor
// with the test still green. This matches any file-looking token followed by a
// line number, which makes it strictly stronger than the stripper: a citation
// in a NEW extension now goes red instead of shipping.
const ANY_FILE_AND_LINE = /[A-Za-z0-9_\-/.]+\.[A-Za-z]{1,6}:\d+/;

test('no source citation reaches a visitor with a line number', () => {
  const offenders = [];
  for (const key of allFieldKeys()) {
    for (const origin of ORIGINS) {
      const { evidence } = resolveForOrigin(PROVENANCE[key], origin);
      if (!evidence) continue;
      const rendered = citationForVisitor(evidence);
      if (ANY_FILE_AND_LINE.test(rendered)) offenders.push(`${key} @ ${origin ?? 'base'}`);
    }
  }
  assert.deepEqual(
    offenders,
    [],
    `${offenders.join(', ')} renders a line number to a visitor. Line numbers ` +
      'drifted repeatedly in this tree, once onto a blank line. The file path and ' +
      'the symbol survive a refactor; the number does not.',
  );
});

test('stripping a line number keeps the file and the prose', () => {
  // The failure mode that would make the previous test pass for the wrong
  // reason: a transform that deleted the whole citation would satisfy it
  // perfectly while destroying the checkability the register exists to provide.
  const sample =
    'crates/onnx-genai-server/src/routes/admin.rs:126-130 — hits/lookups, but emits 0.0.';
  const rendered = citationForVisitor(sample);

  assert.match(rendered, /crates\/onnx-genai-server\/src\/routes\/admin\.rs/);
  assert.match(rendered, /hits\/lookups, but emits 0\.0\./);
  assert.ok(!rendered.includes(':126-130'), 'the volatile range should be gone');
  assert.equal(citationForVisitor('a rate of 0.9375 and 12 hits'), 'a rate of 0.9375 and 12 hits');
});

test('the citation detector is stronger than the stripper, not equal to it', () => {
  // The control that makes the sweep above worth running. The stripper knows
  // four extensions; the detector must not, or the two share a blind spot and
  // the sweep reports green on exactly the citations nobody thought to handle.
  // `.sh` is not hypothetical here: build_qwen.sh:99 is cited by name in this
  // repository's own gate discussion.
  for (const unknown of ['build_qwen.sh:99', 'gen.py:12', 'capture.mjs:7', 'cfg.yaml:3']) {
    const rendered = citationForVisitor(`see ${unknown}`);
    assert.equal(rendered, `see ${unknown}`, `${unknown} is outside the stripper's extension set`);
    assert.ok(
      ANY_FILE_AND_LINE.test(rendered),
      `A ${unknown} citation would reach a visitor unstripped and this sweep ` +
        'would call it green. Widen the stripper in app.js, not this detector.',
    );
  }

  // And it must still not fire on the prose the register is made of, or the
  // sweep becomes unpassable and someone will weaken it back.
  assert.ok(!ANY_FILE_AND_LINE.test('a rate of 0.9375 and 12 hits'));
  assert.ok(!ANY_FILE_AND_LINE.test('hits/lookups, but emits 0.0.'));
});

test('app.js applies the citation transform rather than the raw evidence', () => {
  assert.match(
    appSource,
    /citationForVisitor\(entry\.evidence\)/,
    'The register must render citationForVisitor(entry.evidence). Passing ' +
      'entry.evidence directly puts line numbers back in front of a visitor, and ' +
      'every other cell in the row would still be correct.',
  );
});

// ─── A TOP-LEVEL CLASSIFICATION IS WHAT THE LEAST-INFORMED READER SEES ───────
//
// `resolveForOrigin(entry, origin)` returns an override ONLY when `origin` is
// truthy and names a declared arm. Every other case -- a null origin before
// /health resolves, or a server that is neither `scatter` nor `dynamic` --
// renders the TOP-LEVEL classification.
//
// So an entry whose every declared arm disqualifies it, while its top level
// says MEASURED, shows its most confident claim to the reader who has the
// LEAST information about which server answered. Two entries shipped exactly
// that: `prefix_cache.hits` and `metrics.prefix_cache_hits`, the second found
// only by censusing the register rather than by searching for the first.
//
// WHY THIS IS A RULE AND NOT TWO FIXES: the defect is invisible at the site.
// Each arm is individually correct and carefully reasoned, and the top-level
// line reads as a harmless default. It is only wrong in relation to its arms,
// which is precisely the shape no code review catches by reading a diff.

function entriesWithClassifiedArms() {
  return Object.entries(PROVENANCE).filter(([, entry]) => {
    const arms = Object.values(entry?.byOrigin ?? {});
    return arms.length > 0 && arms.every((arm) => arm && arm.classification);
  });
}

test('no top-level classification is contradicted by every one of its origins', () => {
  const offenders = entriesWithClassifiedArms()
    .filter(([, entry]) =>
      Object.values(entry.byOrigin).every((arm) => arm.classification !== entry.classification),
    )
    .map(([key, entry]) => {
      const arms = Object.entries(entry.byOrigin)
        .map(([origin, arm]) => `${origin}=${arm.classification}`)
        .join(' ');
      return `  ${key}: top-level ${entry.classification}, but ${arms}`;
    });

  assert.deepEqual(
    offenders,
    [],
    'A top-level classification that every origin overrides is a claim true on no ' +
      'server we point at, and it is what renders when the origin is unknown — to the ' +
      'reader with the least information. State what is true EVERYWHERE at the top ' +
      'level and let each arm make it more specific:\n' +
      offenders.join('\n'),
  );
});

test('CAN RUN: entries with per-origin classifications exist, so the check above is not vacuous', () => {
  // Without this, deleting `byOrigin` from the whole register would make the
  // assertion above pass over an empty list — a green earned by having nothing
  // left to check, which is the failure this suite exists to refuse.
  const denominator = entriesWithClassifiedArms();
  assert.ok(
    denominator.length >= 3,
    `expected at least 3 entries with fully-classified origin arms, found ${denominator.length}`,
  );

  // And the predicate must be able to say NO. A checker that cannot fail is not
  // a checker, and every arm above would pass against a broken one.
  const contrived = {
    classification: 'MEASURED',
    byOrigin: { scatter: { classification: 'MISATTRIBUTED' }, dynamic: { classification: 'NOT_PLUMBED' } },
  };
  assert.ok(
    Object.values(contrived.byOrigin).every((a) => a.classification !== contrived.classification),
    'the offender predicate failed to flag a hand-built offender',
  );
});
