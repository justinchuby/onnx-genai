// Copyright (c) Microsoft Corporation.
//
// WHY THIS FILE EXISTS
//
// `renderField` resolves its caption as:
//
//     options.label ?? field?.label ?? 'value'          panel-kit.js
//
// The CALLER WINS. The catalogue in telemetry-provenance.js is only a fallback.
// That precedence is the generator behind a family of caption defects found
// independently by several people: a caption is corrected in the catalogue, the
// correction is verified in the catalogue, and the page does not change, because
// a hardcoded string at the call site was silently outranking it.
//
// Two specimens, both real, both previously shipped:
//
//   'Batch limit'          masked the catalogue's 'Effective batch capacity'.
//                          The value is `max_batch.min(max_queue_depth)`;
//                          "batch limit" names `max_batch` alone. See the
//                          comment in scheduling.js.
//
//   'KV blocks in use'     masked the catalogue's 'KV pages in use'. The server
//   'KV blocks total'      has no blocks: the wire fields are `kv_pages_used`
//                          and `kv_pages_total` in routes/mod.rs, and the
//                          provenance entry cites admin.rs by name. The noun
//                          was invented in the dashboard and the visible unit
//                          beside it read 'blocks' too.
//
// WHAT THIS FILE DOES NOT DO
//
// It does not invert the precedence. That was considered and measured, and it
// is the wrong fix: five call sites pass TEMPLATE literals that the catalogue
// cannot supply, and for at least three of them the catalogue label would be
// actively false if it won —
//
//   throughput.js  a rate derived from `metrics.tokens_generated_total` would
//                  be captioned 'Tokens generated (cumulative)'. It is tok/s.
//   throughput.js  the p50/p95/max cells of one latency row are three distinct
//                  keys rendered into three <td>s; the override is the only
//                  thing that distinguishes them to a screen reader.
//   system.js      a ceiling is captioned '<name> (a configured ceiling, not a
//                  usage reading)'. Inverting deletes that qualifier, which is
//                  the entire point of the row.
//
// So instead of changing who wins, this makes the overrides a CLOSED, DECLARED
// SET. Any NEW hardcoded caption is red until somebody writes down why it is
// not a `'Batch limit'`. That is the machine-level fix: it does not depend on
// the next author knowing this history.
//
// SCOPE LIMIT, STATED PLAINLY: this audits STRING-LITERAL overrides only.
// Template-literal overrides are dynamic by construction and cannot be a
// stale copy of a catalogue entry, so they are out of scope and are counted
// but not policed. It reads `git show HEAD:` rather than the working tree,
// so a fix must be committed before this file will agree that it landed.

import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { PROVENANCE } from './telemetry-provenance.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));

// Anchored to the repository root, NOT to this file's directory. `ls-tree`
// resolves its pathspec relative to the working directory, so running it from
// here silently returned a corpus of zero and the audit below passed green over
// nothing. The CAN RUN floor caught it. Both the anchor and the floor stay.
const TOPLEVEL = execFileSync('git', ['rev-parse', '--show-toplevel'], {
  cwd: HERE,
  encoding: 'utf8',
}).trim();
const DASHBOARD = 'examples/serving-dashboard/dashboard';

function git(...args) {
  return execFileSync('git', args, {
    cwd: TOPLEVEL,
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
  });
}

/** Source files as HEAD has them, never as this desk has them. */
function shippedSources() {
  const names = git('ls-tree', '-r', '--name-only', 'HEAD', '--', DASHBOARD)
    .split('\n')
    .filter((name) => name.endsWith('.js') && !name.endsWith('.test.js'))
    .filter((name) => !name.includes('/testing/'));
  return names.map((name) => ({ name, text: git('show', `HEAD:${name}`) }));
}

/**
 * Every `renderField(...)` call carrying a `label:` option, found by matching
 * parentheses rather than by line, because several of these calls wrap.
 *
 * @param {string} text
 * @returns {Array<{line: number, literal: string|null}>}
 */
export function findCaptionOverrides(text) {
  const found = [];
  let index = 0;
  while ((index = text.indexOf('renderField(', index)) !== -1) {
    let depth = 0;
    let cursor = index + 'renderField'.length;
    for (; cursor < text.length; cursor += 1) {
      const character = text[cursor];
      if (character === '(') depth += 1;
      else if (character === ')') {
        depth -= 1;
        if (depth === 0) break;
      }
    }
    const call = text.slice(index, cursor + 1);
    const match = call.match(/label:\s*('[^']*'|`[^`]*`)/);
    if (match) {
      found.push({
        line: text.slice(0, index).split('\n').length,
        literal: match[1][0] === "'" ? match[1].slice(1, -1) : null,
      });
    }
    index = cursor + 1;
  }
  return found;
}

// Every hardcoded caption that is allowed to outrank the catalogue, and the
// reason it is allowed. A caption NOT on this list is a defect until somebody
// puts it here; a caption on this list that matches a catalogue entry is ALSO a
// defect, because then it is pure duplication and the catalogue can no longer
// move it. Both directions are asserted below.
const DECLARED_CAPTION_OVERRIDES = new Map([
  // 'Refcount distribution' and 'KV tiers' were declared here on the grounds
  // that kv.refcount_histogram and kv.tiers "have no catalogue entry, so there
  // is nothing to mask". Both now HAVE catalogue entries, so the stated reason
  // became false and the overrides became exactly the duplication this list
  // exists to forbid. Removed from the call sites in dashboard/kv-memory.js in
  // the same change, rather than left here with a rewritten excuse.
  [
    'Aggregate output tokens per second',
    'A RATE derived from metrics.tokens_generated_total. The catalogue entry for ' +
      'that key reads "Tokens generated (cumulative)", which is the counter, not ' +
      'the rate. Here the catalogue label would be the wrong caption.',
  ],
]);

const CATALOGUE_LABELS = new Set(
  Object.values(PROVENANCE)
    .map((entry) => entry?.label)
    .filter((label) => typeof label === 'string'),
);

test('CAN RUN: the corpus and the catalogue both loaded', () => {
  const sources = shippedSources();
  assert.ok(
    sources.length >= 8,
    `CANNOT RUN: only ${sources.length} dashboard sources found at HEAD. An empty ` +
      'corpus and a clean audit are the same green, and this branch has shipped ' +
      'that confusion before.',
  );
  assert.ok(
    CATALOGUE_LABELS.size >= 20,
    `CANNOT RUN: the catalogue yielded ${CATALOGUE_LABELS.size} labels. Without it ` +
      'the duplication check below cannot fail and would pass by vacuity.',
  );
  assert.ok(
    sources.some(({ text }) => findCaptionOverrides(text).length > 0),
    'CANNOT RUN: no renderField call anywhere carries a label option. Either the ' +
      'API was renamed or the parser is broken; in both cases a green result here ' +
      'means nothing.',
  );
});

test('the detector finds a hardcoded caption in a synthetic source', () => {
  // A POSITIVE CONTROL on the parser itself. Without it, every assertion in
  // this file could be passing because `findCaptionOverrides` returns nothing.
  // Deliberately wrapped across lines, which is how the real ones are written
  // and is precisely what a line-oriented grep gets wrong.
  const synthetic = `
    row.append(
      renderField(pagesUsed, {
        label: 'KV blocks in use',
        format: formatNumber,
      }),
    );
  `;
  const found = findCaptionOverrides(synthetic);
  assert.equal(found.length, 1, 'the parser missed a wrapped renderField call');
  assert.equal(found[0].literal, 'KV blocks in use');
});

test('the detector does not claim a call that has no label option', () => {
  // The matching negative control: a parser that returns a hit for everything
  // would pass the test above and would make the audit below meaningless.
  assert.deepEqual(findCaptionOverrides('row.append(renderField(percentage));'), []);
});

test('a caption named in a comment is not treated as a caption in use', () => {
  // A REAL specimen, not a synthetic one: kv-memory.js documents the two
  // captions it deleted, so the retired strings 'KV blocks in use' and 'KV
  // blocks total' still appear in the file. A grep cannot tell an epitaph from
  // a use, and rewriting this detector as a grep is the obvious "simplification"
  // for the next person to reach for. This is what makes that regression loud.
  const commented = `
    function renderBlockFraction(pagesUsed, pagesTotal) {
      // NO label OVERRIDE. These two carried 'KV blocks in use' and
      // 'KV blocks total', and the visible unit read 'blocks'.
      return element('span', { children: [renderField(pagesUsed)] });
    }
  `;
  assert.deepEqual(
    findCaptionOverrides(commented),
    [],
    'the detector claimed a caption that only appears inside a comment explaining ' +
      'why that caption was removed',
  );

  const shipped = shippedSources().find(({ name }) => name.endsWith('kv-memory.js'));
  assert.ok(shipped, 'kv-memory.js vanished from HEAD');
  assert.match(
    shipped.text,
    /KV blocks in use/,
    'the epitaph this test relies on is gone from kv-memory.js, so the check above ' +
      'is no longer anchored to a real specimen',
  );
});

test('every hardcoded caption in the shipped dashboard is declared', () => {
  const undeclared = [];
  for (const { name, text } of shippedSources()) {
    for (const { line, literal } of findCaptionOverrides(text)) {
      if (literal !== null && !DECLARED_CAPTION_OVERRIDES.has(literal)) {
        undeclared.push(`${name}:${line} -> '${literal}'`);
      }
    }
  }
  assert.deepEqual(
    undeclared,
    [],
    'A hardcoded caption is outranking the catalogue and nobody has said why.\n' +
      undeclared.map((entry) => `  ${entry}`).join('\n') +
      '\n\nThis is how a caption gets corrected in telemetry-provenance.js and does ' +
      'not change on the page. Either delete the override and let the catalogue ' +
      'supply the caption, or add it to DECLARED_CAPTION_OVERRIDES with the reason ' +
      'the catalogue entry is the wrong words for this particular call site.',
  );
});

test('no declared caption merely restates a catalogue entry', () => {
  const duplicated = [...DECLARED_CAPTION_OVERRIDES.keys()].filter((caption) =>
    CATALOGUE_LABELS.has(caption),
  );
  assert.deepEqual(
    duplicated,
    [],
    'A declared override is character-for-character identical to a catalogue ' +
      `label: ${duplicated.join(', ')}. That is not harmless. It pins the caption ` +
      'at the call site, so the next person to correct the catalogue will correct ' +
      'it, verify it, and watch the page not change. Delete the override.',
  );
});

test('the catalogue keeps the server vocabulary for paged KV', () => {
  // The specific regression that motivated this file, pinned by name. The
  // server has pages; the dashboard invented blocks. This asserts the
  // CATALOGUE is right, so that if someone "fixes" the catalogue to match a
  // stale caption the failure lands here rather than on a visitor.
  for (const key of ['kv.pages_used', 'kv.pages_total']) {
    const label = PROVENANCE[key]?.label;
    assert.ok(label, `${key} vanished from the catalogue`);
    assert.match(
      label,
      /pages/i,
      `The catalogue calls ${key} '${label}'. The wire field is ` +
        `\`${PROVENANCE[key]?.path}\` and the server has no blocks.`,
    );
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// AC196 — THE ACCESSIBLE DESCRIPTION IS A CAPTION SURFACE, AND IT HAD NO GUARD.
//
// Everything above audits `renderField` captions: the words a SIGHTED visitor
// reads. `buildDescription` builds a second, parallel set of captions for the
// screen-reader text, through
//
//     describeFieldText('<prose>', telemetryStore.field('<key>'))
//
// and NOTHING reconciled that prose against the catalogue. This is the same
// generator as 'Batch limit' masking 'Effective batch capacity' — a hardcoded
// string outranking the catalogue — except it is invisible to every reviewer
// who looks at the page, because the divergence only exists in the audio.
//
// WHY THE OBVIOUS VERSION OF THIS RULE IS WORTHLESS, MEASURED BEFORE WRITING IT:
// the first form I built joined "labels of absent catalogue keys" against
// "string literals in the shipped corpus". It returned three hits and ZERO
// defects — two were the SAME COMMENT LINE in kv-memory.js (the epitaph that
// the test 'a caption named in a comment is not treated as a caption in use'
// above exists to protect), and one was an acronym glossary DEFINING the term.
// Shipping it would have filed a false positive against the exact file this
// file already guards, one function below the guard that forbids it.
//
//   A CAPTION THAT NAMES A FIELD IS NOT A PROMISE. A CAPTION BOUND TO A FIELD
//   IS. The binding is what makes it a claim, so the binding is what to parse.
const ACCESSIBLE_CAPTION = /describeFieldText\(\s*'([^']*)'\s*,\s*[A-Za-z_$][\w$]*\.field\(\s*'([^']*)'/g;

/**
 * Every accessible-description caption bound to a telemetry key, at HEAD.
 *
 * @param {string} text
 * @returns {Array<{prose: string, key: string}>}
 */
export function findAccessibleCaptions(text) {
  const found = [];
  for (const match of text.matchAll(ACCESSIBLE_CAPTION)) {
    found.push({ prose: match[1], key: match[2] });
  }
  return found;
}

// Accessible captions whose prose deliberately differs from the catalogue
// label, and why. An entry here is a DECLARATION, not an absolution: two of
// these are open defects, recorded so they cannot grow silently and so the
// next reader meets them beside the green rather than instead of it.
const DECLARED_PROSE_DIVERGENCE = new Map([
  [
    'server.model_id',
    "Prose says 'model', catalogue says 'Model id'. The sentence reads 'System: " +
      "model is X' — 'Model id is X' is worse English in a spoken sentence. A " +
      'register difference, not a meaning difference.',
  ],
  [
    'resources.vram_limit_bytes',
    "OPEN DEFECT, NOT A JUSTIFICATION. Prose says 'KV bytes reserved'; the " +
      "catalogue says 'VRAM limit'. Those are DIFFERENT CONCEPTS — the VRAM " +
      'ceiling is not the KV reservation — so a screen-reader user is told a ' +
      'different fact than the page shows. This is exactly the `Batch limit` ' +
      'defect in the audio channel. Pinned here so it cannot be joined by a ' +
      'second one; it needs a product decision about which noun is true.',
  ],
]);

// Keys named in accessible prose that the catalogue does not cover, and why.
const DECLARED_UNCATALOGUED_KEYS = new Map([
  [
    'client.poll_rtt_ms',
    'OPEN GAP, NOT A JUSTIFICATION. This is a CLIENT-side measurement — the ' +
      'browser times its own poll — so it has no server evidence line and no ' +
      'wire path, which is why it was never given a catalogue entry. But it is ' +
      'spoken to a screen-reader user as a fact, and nothing classifies it. ' +
      'A client measurement is still a measurement and should carry provenance.',
  ],
]);

test('CAN RUN: the accessible-caption detector reaches real bound captions', () => {
  const total = shippedSources().reduce(
    (count, { text }) => count + findAccessibleCaptions(text).length,
    0,
  );
  assert.ok(
    total >= 3,
    `CANNOT RUN: only ${total} bound accessible captions found at HEAD. Either ` +
      'describeFieldText was renamed or the pattern is broken — and a green ' +
      'result over an empty set is the failure this whole file exists to stop.',
  );
});

test('the accessible-caption detector fires, and does not claim an unbound one', () => {
  // Positive and negative in one test, because a detector that matches
  // everything and one that matches nothing both make the audit below green.
  const bound = `parts.push(\`\${describeFieldText('Execution provider', store.field('server.execution_provider'))}.\`);`;
  assert.deepEqual(findAccessibleCaptions(bound), [
    { prose: 'Execution provider', key: 'server.execution_provider' },
  ]);

  // A caption naming a field WITHOUT binding it is prose, not a promise. The
  // acronym glossary in system.js is exactly this shape and must stay legal.
  const unbound = `acronyms: { EP: 'Execution provider — the backend ONNX Runtime dispatches to' },`;
  assert.deepEqual(findAccessibleCaptions(unbound), []);
});

test('an accessible caption inside a comment is not treated as one in use', () => {
  // The specimen that nearly cost me a false positive: kv-memory.js:186 names
  // two catalogue labels inside a comment explaining the captions it removed.
  const commented = `
    // The catalogue says 'KV pages in use' and 'KV pages total', and the
    // dashboard used to say blocks. Do not reintroduce:
    //   describeFieldText('KV blocks in use', telemetryStore.field('kv.pages_used'))
    return null;
  `;
  const found = findAccessibleCaptions(commented);
  assert.deepEqual(
    found.map(({ key }) => key),
    ['kv.pages_used'],
    'the comment specimen changed shape; this test is no longer anchored',
  );
  // Documented limitation, asserted rather than claimed: this detector DOES
  // see into comments. It is acceptable here only because a commented-out
  // `describeFieldText` call is itself a thing worth deleting, whereas a
  // commented LABEL (the kv-memory.js epitaph) is not — and the label-only
  // form, which is the one that actually appears in our tree, is not matched.
  assert.deepEqual(findAccessibleCaptions("// catalogue says 'KV pages in use'"), []);
});

test('every key spoken to a screen reader is in the provenance catalogue', () => {
  const uncatalogued = [];
  for (const { name, text } of shippedSources()) {
    for (const { prose, key } of findAccessibleCaptions(text)) {
      if (!PROVENANCE[key] && !DECLARED_UNCATALOGUED_KEYS.has(key)) {
        uncatalogued.push(`${name} -> '${prose}' is bound to ${key}`);
      }
    }
  }
  assert.deepEqual(
    uncatalogued,
    [],
    'The accessible description states a field as fact, and the catalogue has no ' +
      'entry for it — so nothing records where it came from or whether it is ' +
      'measured:\n' +
      uncatalogued.map((entry) => `  ${entry}`).join('\n') +
      '\n\nThe screen-reader text is the one surface no sighted reviewer checks. ' +
      'Give the key a catalogue entry, or declare it in ' +
      'DECLARED_UNCATALOGUED_KEYS with the reason it cannot have one.',
  );
});

test('accessible prose does not silently rename a catalogue entry', () => {
  const drifted = [];
  for (const { name, text } of shippedSources()) {
    for (const { prose, key } of findAccessibleCaptions(text)) {
      const label = PROVENANCE[key]?.label;
      if (label && label !== prose && !DECLARED_PROSE_DIVERGENCE.has(key)) {
        drifted.push(`${name} -> ${key}: spoken '${prose}', catalogue '${label}'`);
      }
    }
  }
  assert.deepEqual(
    drifted,
    [],
    'The screen-reader text calls a field something the catalogue does not:\n' +
      drifted.map((entry) => `  ${entry}`).join('\n') +
      '\n\nThis is `Batch limit` in the audio channel: a hardcoded string ' +
      'outranking the catalogue, on the one surface a reviewer cannot see. ' +
      'Correcting the catalogue will NOT change what is spoken. Either drop the ' +
      'hardcoded prose or declare it in DECLARED_PROSE_DIVERGENCE with a reason.',
  );
});
