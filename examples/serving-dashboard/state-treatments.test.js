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
const tokensCss = readFileSync(new URL('./styles/tokens.css', import.meta.url), 'utf8');

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
  const match = /(\.value:not\(\[data-state\]\),[^{]*)\{([^}]*)\}/.exec(css);
  return match ? { selector: match[1], body: match[2] } : null;
}

/** The `NO STATE` chip rule: `{selector, body}`, or null. */
function chipRule(css) {
  const match = /(\.value:not\(\[data-state\]\)::after,[^{]*)\{([^}]*)\}/.exec(css);
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

/** WCAG relative luminance of a #rrggbb string. */
function luminance(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const lin = (c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

/** WCAG contrast ratio between two #rrggbb strings. */
function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

/** Resolve a `var(--og-x)` reference against tokens.css. */
function token(name) {
  const match = new RegExp(`${name}:\\s*(#[0-9a-fA-F]{6})`).exec(tokensCss);
  return match ? match[1] : null;
}

test('the NO STATE chip is CLOSED — it inherits no colour from the page', () => {
  // @0837fdf9 retracted their own inversion proposal after measuring it: it
  // would have put #e6edf3 text on an #e6edf3 background at 1.00:1, DELETING
  // the value it was meant to qualify, because `.value__num` sets its own
  // colour directly and a direct rule beats inheritance from the parent.
  //
  // The chip is the answer to that class, not a patch for that instance. It
  // declares BOTH background and foreground, so none of the unconditional
  // `color` rules on `.value` descendants can reach inside it — including ones
  // nobody has written yet. THAT is the property under test here: not the
  // colours it picked, but that it picks both of them.
  const rule = chipRule(shellCss);
  assert.ok(rule, 'shell.css has no `NO STATE` chip rule');
  assert.match(rule.body, /background:\s*var\(--og-[a-z-]+\)/, 'the chip sets no background');
  assert.match(rule.body, /(^|[^-])color:\s*var\(--og-[a-z-]+\)/, 'the chip sets no colour');
});

test('the chip carries a WORD, and it is legible against its own background', () => {
  const rule = chipRule(shellCss);
  assert.ok(rule, 'no chip rule');
  const content = /content:\s*'([^']*)'/.exec(rule.body);
  assert.ok(content && content[1].trim(), 'the chip renders no text, so it encodes by hue alone');

  const fg = token(/color:\s*var\((--og-[a-z-]+)\)/.exec(rule.body.replace(/background:[^;]*;/, ''))[1]);
  const bg = token(/background:\s*var\((--og-[a-z-]+)\)/.exec(rule.body)[1]);
  assert.ok(fg && bg, `chip colours did not resolve in tokens.css: fg=${fg} bg=${bg}`);
  const ratio = contrast(fg, bg);
  assert.ok(ratio >= 4.5, `the chip reads at ${ratio.toFixed(2)}:1 against its own background`);

  // KNOWN-FAILING CONTROL, @0837fdf9's requirement: the ratio function must be
  // able to return a SMALL number, or "every ratio is large" proves nothing.
  // Their measured case: --og-simulated-fg on the inverted background they
  // withdrew. If this stops being ~1.91 the instrument has drifted.
  const control = contrast(token('--og-simulated-fg'), token('--og-fg'));
  assert.ok(
    control < 2.5,
    `the contrast function cannot produce a failing value (control ${control.toFixed(2)}:1), ` +
      'so the assertion above is unfalsifiable',
  );
});

test('the chip and the backstop exclude the SAME ruled vocabulary', () => {
  // Two hand-written `:not()` chains naming the same five states is exactly the
  // duplication that produced tonight's duplicate-provenance-key defect. They
  // cannot be aliased in CSS, so they are pinned to the enum instead — and to
  // each other, because a chain that drifts gives a state a chip while styling
  // it correctly, or the reverse.
  const backstop = backstopRule(shellCss);
  const chip = chipRule(shellCss);
  assert.ok(backstop && chip, 'both rules must exist to compare them');
  const excluded = (sel) =>
    [...new Set([...sel.matchAll(/:not\(\[data-state='([^']+)'\]\)/g)].map((m) => m[1]))].sort();
  const ruled = [...new Set(Object.values(FIELD_STATES))].sort();
  assert.deepEqual(excluded(chip.selector), ruled, "the chip's exclusions have drifted from FIELD_STATES");
  assert.deepEqual(excluded(chip.selector), excluded(backstop.selector), 'the two chains disagree');
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

// ---------------------------------------------------------------------------
// WIRING, NOT VALUES.
//
// Every assertion above this line reads shell.css for STRUCTURE -- does the
// state have a rule, does it have a border, is its line style unique. Every
// assertion in @0837fdf9's eight token suites reads tokens.css for VALUES --
// is the contrast ratio legal, is the ramp monotonic. Both were green all
// session while `[data-state='not-applicable']` painted itself in
// `--og-unavail-fg`.
//
// Neither could see it, and not by oversight: the defect is not in the palette
// and not in the structure, it is in the WIRE BETWEEN THEM. The palette can be
// flawless and the selector can still spend the wrong entry. A guard has to
// read the token NAME inside the state's own rule to see it, which is a third
// question neither file was asking.
//
// Guard by @0837fdf9, who measured it, built it, and proved the fix sufficient
// before asking for it. Ported here because it lived in /tmp: uncommitted
// coverage is not coverage, and it belongs next to the other reader of these
// same blocks rather than in a ninth parser.

/**
 * The token family a state is allowed to spend, keyed by state.
 *
 * Trailing `-` means prefix; otherwise exact. Derived-not-hardcoded is enforced
 * below: this map must cover FIELD_STATES exactly, so a sixth state fails here
 * until somebody decides which family it draws from -- rather than silently
 * sitting outside the loop, which is the failure the pending case above records.
 */
const STATE_TOKEN_FAMILY = {
  [FIELD_STATES.MEASURED]: ['--og-fg'],
  [FIELD_STATES.PENDING]: ['--og-pending-'],
  [FIELD_STATES.STALE]: ['--og-stale-'],
  [FIELD_STATES.UNAVAILABLE]: ['--og-unavail-'],
  [FIELD_STATES.NOT_APPLICABLE]: ['--og-na-'],
};

const allows = (family, token) =>
  family.some((p) => (p.endsWith('-') ? token.startsWith(p) : token === p));

/**
 * Colour/border token usages inside BARE `[data-state='…']` rules.
 *
 * Bare only. `.connection-indicator[data-state='connected']` is a different
 * state vocabulary -- connection health -- and correctly spends semantic
 * --og-ok/--og-warn/--og-bad. Flagging correct code is how a guard earns its
 * deletion, so the leading-anchor is load-bearing rather than incidental.
 */
function auditStateWiring(css) {
  const lines = withoutNotGroups(css).split('\n');
  const blocks = [];
  let cur = null;
  lines.forEach((line, i) => {
    const opened = line.match(/^\s*\[data-state='([^']+)'\]/);
    if (opened) cur = { state: opened[1], line: i + 1, uses: [] };
    if (cur && /(^|\s)(color|border[a-z-]*)\s*:/.test(line)) {
      for (const t of line.matchAll(/var\((--og-[a-z0-9-]+)\)/g)) {
        cur.uses.push({ token: t[1], line: i + 1 });
      }
    }
    if (cur && !opened && /^\s*}/.test(line)) {
      blocks.push(cur);
      cur = null;
    }
  });
  if (cur) blocks.push(cur);
  return blocks;
}

/**
 * @e00032a4's rule, applied to a runner with no exit codes: a check that CANNOT
 * RUN must not be confusable with one that ran and found nothing. There is no
 * exit 2 here, so the distinction is carried by the message -- `CANNOT RUN` vs
 * `FAIL` -- and by the floors below firing before any verdict is reached.
 */
function assertCanRun(blocks) {
  const states = Object.keys(STATE_TOKEN_FAMILY);
  assert.ok(
    blocks.length >= states.length,
    `CANNOT RUN: parsed ${blocks.length} bare [data-state] blocks, expected >= ${states.length}. ` +
      'The parser broke or the file moved; a pass here would be vacuous.',
  );
  const uses = blocks.reduce((n, b) => n + b.uses.length, 0);
  assert.ok(
    uses >= 8,
    `CANNOT RUN: inspected ${uses} token usages, expected >= 8. ` +
      'The declaration matcher stopped matching, so every block looks clean.',
  );
}

test('the family map covers exactly the ruled state vocabulary', () => {
  assert.deepEqual(
    Object.keys(STATE_TOKEN_FAMILY).sort(),
    Object.values(FIELD_STATES).sort(),
    'a state has no declared token family, so nothing checks which colours it spends',
  );
});

test('every state selector spends the token family named for that state', () => {
  const blocks = auditStateWiring(shellCss);
  assertCanRun(blocks);

  const findings = [];
  for (const b of blocks) {
    const family = STATE_TOKEN_FAMILY[b.state];
    assert.ok(family, `FAIL: [data-state='${b.state}'] (shell.css:${b.line}) has no family`);
    for (const u of b.uses) {
      if (!allows(family, u.token)) {
        findings.push(
          `shell.css:${u.line}  [data-state='${b.state}'] consumes ${u.token} ` +
            `(expected ${family.join(', ')})`,
        );
      }
    }
  }
  assert.deepEqual(findings, [], `FAIL: a state renders in another state's colours:\n${findings.join('\n')}`);
});

// The detector's own positive controls. @0837fdf9 ran these as manual edits to
// a real file; as synthetic input they ship, so the guard keeps proving it can
// still fail long after the defect that motivated it is gone.
test('the wiring audit detects a DIFFERENT state borrowing tokens', () => {
  const blocks = auditStateWiring(
    `[data-state='stale'] {\n  color: var(--og-pending-rule);\n}\n`,
  );
  assert.equal(blocks.length, 1);
  assert.ok(
    !allows(STATE_TOKEN_FAMILY[FIELD_STATES.STALE], blocks[0].uses[0].token),
    'the audit is hardcoded at one state and cannot see a borrow elsewhere',
  );
});

test('the wiring audit refuses to vouch for an unmapped state', () => {
  const blocks = auditStateWiring(`[data-state='estimated'] {\n  color: var(--og-fg);\n}\n`);
  assert.equal(STATE_TOKEN_FAMILY[blocks[0].state], undefined,
    'an unknown state must be a finding, not a silent pass through a stale coverage list');
});

test('the wiring audit cannot pass on input it never read', () => {
  assert.throws(
    () => assertCanRun(auditStateWiring('')),
    /CANNOT RUN: parsed 0 bare/,
    'an empty stylesheet produced a clean bill of health',
  );
});

test('a nested state selector is not audited as a bare one', () => {
  // The false-positive guard, pinned. --og-ok is correct for connection health
  // and must never be reported against the field-state families.
  const blocks = auditStateWiring(
    `.connection-indicator[data-state='connected'] {\n  color: var(--og-ok);\n}\n`,
  );
  assert.deepEqual(blocks, [], 'a different state vocabulary was audited against field-state families');
});
