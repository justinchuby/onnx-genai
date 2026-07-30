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
  [
    'Refcount distribution',
    'kv.refcount_histogram has no catalogue entry, so there is nothing to mask.',
  ],
  ['KV tiers', 'kv.tiers has no catalogue entry, so there is nothing to mask.'],
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
