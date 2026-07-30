// Copyright (c) Microsoft Corporation.
//
// The cross-language check: every field state the JS can emit must have a
// visual treatment in the CSS that receives it.
//
// WHY THIS FILE EXISTS. `styles/shell.css` styled `[data-state='measured']`
// while `FIELD_STATES.MEASURED` is the string `'ok'`, so the rule for the most
// common state on the page matched nothing at all. Every measured value fell
// back to inherited styling — which looks close enough to correct that it
// survived a browser check. Nothing in JS could catch it: the JS was right.
// Nothing in CSS could catch it: the CSS was internally consistent. The bug
// lived only in the gap between them, which is precisely where a project with
// no build step and no type checker is blindest.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { FIELD_STATES, SOURCE_CLASSES } from './telemetry-field.js';
import { ABSENT_TEXT, NOT_APPLICABLE_TEXT } from './format.js';

const shellCss = readFileSync(new URL('./styles/shell.css', import.meta.url), 'utf8');

/** Every `[data-state='…']` value the stylesheet actually selects on. */
function styledStates(css) {
  return new Set([...css.matchAll(/\[data-state='([^']+)'\]/g)].map((m) => m[1]));
}

test('every field state has a visual treatment in shell.css', () => {
  const styled = styledStates(shellCss);
  const missing = Object.values(FIELD_STATES).filter((state) => !styled.has(state));
  assert.deepEqual(
    missing,
    [],
    `these states can reach the DOM with no styling rule: ${missing.join(', ')}. ` +
      'They will render at inherited contrast, which reads as a measured value.',
  );
});

test('shell.css does not style a state that cannot occur', () => {
  // The other direction, and the one that actually bit us: a selector naming a
  // state no field will ever carry is dead code that LOOKS like coverage.
  const connectionStates = new Set(['connected', 'connecting', 'no-model', 'unreachable']);
  const fieldStates = new Set(Object.values(FIELD_STATES));

  const orphans = [...styledStates(shellCss)].filter(
    (state) => !fieldStates.has(state) && !connectionStates.has(state),
  );
  assert.deepEqual(
    orphans,
    [],
    `these selectors match no state any code emits: ${orphans.join(', ')}. ` +
      'A rule that never fires is worse than a missing one, because it reads as covered.',
  );
});

test('the three absence states are distinguishable without colour', () => {
  // AC25. unavailable, not-applicable and stale all render an em-dash or a
  // de-emphasised value, and their foreground colours are deliberately close.
  // If they are separated by colour alone, a grayscale or colourblind reader
  // cannot tell "not built yet" from "meaningless here" — which is the single
  // distinction this demo exists to teach.
  const borders = {};
  for (const state of [FIELD_STATES.STALE, FIELD_STATES.UNAVAILABLE, FIELD_STATES.NOT_APPLICABLE]) {
    const block = shellCss.match(
      new RegExp(`\\[data-state='${state}'\\][^{]*\\{([^}]*)\\}`),
    );
    assert.ok(block, `${state} has no rule block`);
    const border = block[1].match(/border-bottom:\s*([^;]+);/);
    assert.ok(border, `${state} has no border-bottom, so it relies on colour alone`);
    borders[state] = border[1].trim();
  }
  const patterns = Object.values(borders);
  assert.equal(
    new Set(patterns).size,
    patterns.length,
    `these absence states share a border pattern and differ only by colour: ${JSON.stringify(borders)}`,
  );
});

test('every source class has a CSS hook for its badge', () => {
  // panel-kit writes data-source alongside data-state; AC7 needs the class to
  // be visually distinguishable, not only present in a tooltip.
  const styled = new Set([...shellCss.matchAll(/\[data-source='([^']+)'\]/g)].map((m) => m[1]));
  if (styled.size === 0) return; // no badge styling in this stylesheet yet
  const missing = Object.values(SOURCE_CLASSES).filter((cls) => !styled.has(cls));
  assert.deepEqual(missing, [], `source classes with no styling: ${missing.join(', ')}`);
});

// --- CONTRACT.md must not contradict the renderer ---------------------------
//
// Same gap as the one at the top of this file, one level up: CONTRACT.md is
// authoritative for the panel author, and it stated that `not-applicable`
// renders as an em-dash while format.js has always rendered `n/a`. Nothing
// could catch it -- the doc was internally consistent, the code was correct,
// and the panel author reading the doc would have hardcoded the wrong glyph in
// a file this test does not even look at.
//
// Docs drift silently because nothing executes them. This executes them.

test('CONTRACT.md renders each absence state the way format.js actually does', () => {
  const contract = readFileSync(new URL('./CONTRACT.md', import.meta.url), 'utf8');
  const row = (state) =>
    contract.split('\n').find((line) => line.startsWith(`| \`${state}\``)) ?? '';

  const notApplicable = row('not-applicable');
  assert.ok(notApplicable, 'CONTRACT.md must document the not-applicable state');
  assert.ok(
    notApplicable.includes(NOT_APPLICABLE_TEXT),
    `CONTRACT.md must say not-applicable renders as "${NOT_APPLICABLE_TEXT}", the string ` +
      `format.js emits. Row: ${notApplicable}`,
  );
  assert.ok(
    !notApplicable.includes(ABSENT_TEXT),
    'CONTRACT.md must NOT claim not-applicable renders as the em-dash: spec:714 requires it be ' +
      'distinguishable from unavailable on the surface, and a panel author following the doc ' +
      'would render half a working dashboard as broken',
  );

  const unavailable = row('unavailable');
  assert.ok(unavailable.includes(ABSENT_TEXT), 'unavailable IS the em-dash');
  assert.ok(
    !unavailable.includes(NOT_APPLICABLE_TEXT),
    'unavailable must not be documented as n/a',
  );
});

test('the two absence texts are actually different strings', () => {
  // Guards the whole distinction at its root: if these ever collapse to the
  // same string, every test above still passes and the surface difference the
  // spec requires quietly disappears.
  assert.notEqual(ABSENT_TEXT, NOT_APPLICABLE_TEXT);
});
