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

/**
 * `:not(...)` groups removed.
 *
 * LOAD-BEARING, AND IT IS A DEFECT THIS COMMIT WOULD OTHERWISE HAVE INTRODUCED.
 * The backstop rule excludes the ruled states by name inside a `:not()` chain,
 * so the raw text of shell.css now contains `[data-state='measured']` even in a
 * world where the real `[data-state='measured'] { … }` rule has been deleted.
 * Scraped naively, every state below would look styled forever and the test
 * above would be permanently, silently vacuous — a check that cannot fail
 * because the fix for a different problem fed it its own answer.
 */
function withoutNotGroups(css) {
  return css.replace(/:not\([^)]*\)/g, '');
}

/** Every `[data-state='…']` value the stylesheet actually selects on. */
function styledStates(css) {
  return new Set([...withoutNotGroups(css).matchAll(/\[data-state='([^']+)'\]/g)].map((m) => m[1]));
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

test('every non-measured state is distinguishable without colour', () => {
  // AC25. unavailable, not-applicable and stale all render an em-dash or a
  // de-emphasised value, and their foreground colours are deliberately close.
  // If they are separated by colour alone, a grayscale or colourblind reader
  // cannot tell "not built yet" from "meaningless here" — which is the single
  // distinction this demo exists to teach.
  //
  // PENDING IS IN THIS LIST AND USED NOT TO BE, AND ITS ABSENCE IS WHY A REAL
  // DEFECT SURVIVED. This test read "the three absence states" and iterated a
  // hardcoded three, so the one state whose second channel was a no-op —
  // `font-style: italic` on '···', which has no stroke to slant — sat outside
  // the loop. The guard could not reach the failing branch, so it was green for
  // a reason unrelated to the code being right. Pending is not an absence
  // state, but it is a NON-MEASURED one, and the reader's need is identical:
  // do not mistake a value that has not arrived for one that has.
  const borders = {};
  const covered = [
    FIELD_STATES.PENDING,
    FIELD_STATES.STALE,
    FIELD_STATES.UNAVAILABLE,
    FIELD_STATES.NOT_APPLICABLE,
  ];
  // Derived, not hardcoded: every state except MEASURED must appear above. If a
  // sixth state is ever added, this fails until someone decides its treatment.
  const expected = Object.values(FIELD_STATES).filter((s) => s !== FIELD_STATES.MEASURED);
  assert.deepEqual(
    [...covered].sort(),
    [...expected].sort(),
    'a field state is missing from this check, so nothing verifies it is legible ' +
      'without colour. Add it to `covered` and give it a border style no other ' +
      'state uses.',
  );

  for (const state of covered) {
    // Stripped: the backstop's `:not()` chain names every ruled state, so an
    // unstripped scan can match the exclusion list instead of the real rule.
    const block = withoutNotGroups(shellCss).match(
      new RegExp(`\\[data-state='${state}'\\][^{]*\\{([^}]*)\\}`),
    );
    assert.ok(block, `${state} has no rule block`);
    const border = block[1].match(/border-bottom:\s*([^;]+);/);
    assert.ok(border, `${state} has no border-bottom, so it relies on colour alone`);
    borders[state] = border[1].trim();
  }

  // Compare the STYLE keyword, not the whole declaration: two states with
  // different colours but both `1px dotted` are identical in grayscale, and
  // comparing full strings would call that distinct.
  const styleOf = (decl) => decl.split(/\s+/).find((t) => /^(solid|dashed|dotted|double)$/.test(t));
  const patterns = Object.fromEntries(
    Object.entries(borders).map(([state, decl]) => [state, styleOf(decl)]),
  );
  assert.ok(
    Object.values(patterns).every(Boolean),
    `a state's border has no recognised line style: ${JSON.stringify(borders)}`,
  );
  assert.equal(
    new Set(Object.values(patterns)).size,
    Object.values(patterns).length,
    `these states share a border pattern and differ only by colour: ${JSON.stringify(patterns)}`,
  );
});

/**
 * The backstop rule: `{selector, body}`, or null if nobody wrote one.
 *
 * Anchored on `.value:not([data-state])` because that is the case with no
 * possible alternative reading — an element that renders a field and carries no
 * state at all.
 */
function backstopRule(css) {
  const match = /(\.value:not\(\[data-state\]\)[^{]*)\{([^}]*)\}/.exec(css);
  return match ? { selector: match[1], body: match[2] } : null;
}

test('an absent data-state renders obviously wrong, not as a measurement', () => {
  // THE DEFAULT WAS THE MAXIMALLY DISHONEST ONE. Every other rule in this
  // section is POSITIVE — it names a state and styles it — so an element with
  // no `data-state`, or one outside the ruled vocabulary, inherited `--og-fg`
  // and rendered EXACTLY as a trusted measurement. Screenshots of the unknown
  // state, the absent-attribute case and a real measurement were byte-identical.
  //
  // `renderStateOf` does degrade correctly and always has. That is not enough
  // on its own: it is one line in another language, and nothing in this file
  // noticed if it changed. The honesty rule has to be structural on both sides.
  const rule = backstopRule(shellCss);
  assert.ok(
    rule,
    'shell.css has no backstop for `.value:not([data-state])`. An element that ' +
      'renders a field but sets no state inherits measured contrast, so the ' +
      "honesty layer's default is its most confident rendering.",
  );
  assert.ok(
    /color:/.test(rule.body),
    'the backstop sets no colour, so it does not visibly differ from a measurement',
  );
  assert.ok(
    !/var\(--og-fg\)/.test(rule.body),
    'the backstop paints itself with --og-fg, which IS the measured colour',
  );
});

test('the backstop excludes exactly the ruled vocabulary', () => {
  // THE RATCHET, AND THE REASON THIS TEST IS NOT DECORATIVE. The backstop works
  // by excluding the five ruled states by name. That list is a hardcoded copy of
  // the enum living in a stylesheet, which is the same cross-language gap that
  // produced the 'measured'/'ok' defect this whole file exists to catch — with
  // the failure inverted and worse: a sixth ruled state added to FIELD_STATES
  // and given a proper rule above would ALSO match the backstop and render as
  // broken, so a correct new state would ship looking like an error.
  //
  // Derived from the enum, never enumerated here.
  const rule = backstopRule(shellCss);
  assert.ok(rule, 'no backstop rule to check');
  const excluded = [...rule.selector.matchAll(/:not\(\[data-state='([^']+)'\]\)/g)].map(
    (m) => m[1],
  );
  assert.deepEqual(
    [...new Set(excluded)].sort(),
    [...new Set(Object.values(FIELD_STATES))].sort(),
    'the backstop\'s exclusion list has drifted from FIELD_STATES. A ruled state ' +
      'missing from it renders as an error; a stale name left in it lets a ' +
      'retired state keep rendering as a measurement.',
  );
});

test('the backstop is legible without colour, and its channel is unused', () => {
  // Same requirement AC25 puts on every non-measured state: a projector and a
  // greyscale screenshot must still show something is wrong. The four border
  // styles that read distinctly at 1px are already spoken for, so the backstop
  // takes a text-decoration line style instead of a fifth border.
  const rule = backstopRule(shellCss);
  assert.ok(rule, 'no backstop rule to check');
  const decoration = /text-decoration:\s*([^;]+);/.exec(rule.body);
  assert.ok(
    decoration,
    'the backstop is carried by colour alone, so it vanishes in greyscale and ' +
      'for a colourblind reader — the readers this distinction matters most to',
  );
  const style = decoration[1]
    .split(/\s+/)
    .find((token) => /^(solid|dashed|dotted|double|wavy)$/.test(token));
  assert.ok(style, `the backstop's text-decoration has no line style: ${decoration[1]}`);
  // The four border styles belong to pending/stale/unavailable/not-applicable.
  // Reusing one would make "we do not know what this is" look like a ruled
  // absence, which is a different and much calmer claim.
  assert.equal(
    style,
    'wavy',
    `the backstop reuses '${style}', which a ruled absence state already owns`,
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
