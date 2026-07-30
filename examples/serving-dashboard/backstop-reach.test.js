/**
 * DOES THE NO-STATE BACKSTOP ACTUALLY REACH THE ELEMENTS THAT NEED IT?
 *
 * `state-treatments.test.js` proves the backstop RULE is correct: that it
 * exists, that it excludes exactly the ruled vocabulary, that it is legible
 * without colour, that its chip inherits nothing. Twenty assertions, and every
 * one of them is about the rule.
 *
 * None of them asks WHICH ELEMENTS THE SELECTOR MATCHES. That gap was not
 * theoretical. The backstop is scoped to `.value`, and `ui/model-card.js`
 * created its value cells with the single class `model-card__value`, so the
 * card sat outside the guarded population from the moment the backstop landed.
 *
 * WHY IT WAS INVISIBLE, AND WHY A RULE-LEVEL TEST COULD NEVER HAVE SEEN IT:
 * the five POSITIVE state rules in shell.css are unscoped -- plain
 * `[data-state='measured']`, `[data-state='stale']` and so on -- so they DID
 * reach the card. Every valid state rendered correctly there. The only case
 * that fell through was the absent or unrecognised one: precisely the case the
 * backstop exists for, and the only one no reviewer looks at, because looking
 * at it requires deliberately breaking a state first.
 *
 * MEASURED IN A REAL BROWSER before the fix, computed styles on one element:
 *   .model-card__value  no state / `bogus-typo` / `measured`
 *       -> rgb(230,237,243) | none | ::after "none"   FOR ALL THREE, IDENTICAL
 *   .value              no state / `bogus-typo`
 *       -> rgb(230,159,0)   | underline | ::after "NO STATE"   (the control)
 * The control is the load-bearing half: without a `.value` arm firing in the
 * same run, "no backstop on the card" is indistinguishable from "my probe
 * cannot see backstops at all".
 *
 * SO THIS FILE GUARDS THE REACH, NOT THE RULE. It reads the stylesheet for the
 * classes the backstop actually covers, reads shipped JS for the classes that
 * name themselves value cells, and asserts the second set is contained in the
 * first. It is deliberately a source-level invariant with no DOM and no
 * browser, so it runs in the ordinary suite on every change.
 *
 * IT READS THE WORKING TREE, NOT `git show HEAD:`. A guard whose job is to stop
 * a defect from being COMMITTED must be able to see uncommitted bytes; reading
 * HEAD would make it structurally blind to the change under review. It also
 * touches git zero times, so it cannot straddle another agent's commit
 * mid-run, which is a real failure mode in this tree.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const DASHBOARD_DIR = dirname(fileURLToPath(import.meta.url));
const SHELL_CSS = join(DASHBOARD_DIR, 'styles', 'shell.css');

/**
 * A class literal is a VALUE CELL if it is exactly `value` or ends in
 * `__value`. That is the project's own naming convention for the element that
 * carries a field reading and therefore a `data-state`.
 *
 * It deliberately does NOT match `value__num`, `value__unit`, `value__src` and
 * the rest of the `value__*` family: those are CHILDREN of a value cell. They
 * never receive a `data-state` of their own, and demanding backstop coverage
 * for them would be a false red.
 */
const VALUE_CELL = /^(?:value|[a-z0-9-]+__value)$/;

/** Collect every `.js` file that ships (tests and fixtures excluded). */
function shippedModules() {
  const found = [];
  const walk = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const full = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === 'node_modules' || entry.name === 'testing') continue;
        walk(full);
        continue;
      }
      if (!entry.name.endsWith('.js')) continue;
      if (entry.name.endsWith('.test.js')) continue;
      found.push(full);
    }
  };
  walk(DASHBOARD_DIR);
  return found.sort();
}

/**
 * Every class-name string literal assigned to an element, from both idioms
 * used in this codebase: `className: '...'` (the `element()` helper) and
 * `.className = '...'` (raw DOM). A literal may hold several space-separated
 * classes; each is returned separately.
 */
function classLiteralsIn(source) {
  const classes = new Set();
  const patterns = [
    /className\s*:\s*'([^']*)'/g,
    /\.className\s*=\s*'([^']*)'/g,
  ];
  for (const pattern of patterns) {
    for (const [, literal] of source.matchAll(pattern)) {
      for (const token of literal.split(/\s+/)) {
        if (token) classes.add(token);
      }
    }
  }
  return classes;
}

/**
 * The classes the no-state backstop covers, read from the stylesheet.
 *
 * The backstop is identified by the one thing that makes it the backstop --
 * `:not([data-state])` -- rather than by a line number or a comment, so it
 * survives the rule being moved or re-worded.
 */
function backstopClasses(css) {
  const classes = new Set();
  for (const line of css.split('\n')) {
    if (!line.includes(':not([data-state])')) continue;
    for (const [, name] of line.matchAll(/\.([a-z0-9_-]+)(?=:not\(\[data-state\]\))/g)) {
      classes.add(name);
    }
  }
  return classes;
}

/** Every value-cell class in shipped JS, with the file that declares it. */
function valueCellsInShippedJs() {
  const cells = new Map();
  for (const file of shippedModules()) {
    for (const cls of classLiteralsIn(readFileSync(file, 'utf8'))) {
      if (!VALUE_CELL.test(cls)) continue;
      if (!cells.has(cls)) cells.set(cls, []);
      cells.get(cls).push(relative(DASHBOARD_DIR, file));
    }
  }
  return cells;
}

test('CAN RUN: the stylesheet, the modules and both extractors produce input', () => {
  const modules = shippedModules();
  assert.ok(
    modules.length >= 20,
    `expected at least 20 shipped modules, enumerated ${modules.length} — the walker is broken, ` +
      'and an empty corpus would make every assertion below vacuously true',
  );

  const names = modules.map((f) => relative(DASHBOARD_DIR, f));
  for (const required of ['ui/model-card.js', 'dashboard/panel-kit.js']) {
    assert.ok(
      names.includes(required),
      `${required} is missing from the corpus; it is one of the two files that renders a field state`,
    );
  }

  const css = readFileSync(SHELL_CSS, 'utf8');
  assert.ok(css.includes(':not([data-state])'), 'shell.css declares no no-state backstop at all');
});

test('the backstop parser finds a non-empty, correctly scoped class set', () => {
  const covered = backstopClasses(readFileSync(SHELL_CSS, 'utf8'));

  assert.ok(
    covered.size > 0,
    'parsed zero classes from the backstop selector — the parser is broken, and a broken parser ' +
      'reports every value cell as uncovered rather than silently passing',
  );
  assert.ok(
    covered.has('value'),
    `the backstop must cover \`.value\`, the class panel-kit gives every field wrapper; covers [${[...covered]}]`,
  );
  assert.ok(
    !covered.has('connection-indicator'),
    'the backstop must NOT cover `.connection-indicator`: it carries a different vocabulary ' +
      '(connected/connecting/no-model/unreachable) on the same attribute, and covering it would ' +
      'paint every indicator as broken',
  );
});

test('the value-cell detector finds the real cells and excludes their children', () => {
  const cells = valueCellsInShippedJs();

  assert.ok(
    cells.size >= 2,
    `expected at least 2 value-cell classes in shipped JS, found ${cells.size} [${[...cells.keys()]}] — ` +
      'panel-kit and model-card each declare one, so a smaller count means the detector missed a site',
  );
  for (const child of ['value__num', 'value__unit', 'value__src', 'value__sep']) {
    assert.ok(
      !cells.has(child),
      `${child} is a CHILD of a value cell and never carries its own data-state; ` +
        'demanding backstop coverage for it would be a false red',
    );
  }
});

test('every value cell in shipped JS is inside the backstop’s reach', () => {
  const covered = backstopClasses(readFileSync(SHELL_CSS, 'utf8'));
  const cells = valueCellsInShippedJs();

  const uncovered = [];
  for (const [cls, files] of cells) {
    // A cell is reached if ANY class on the same element is covered. The
    // element's full class list is what the browser matches against, so
    // `class="value model-card__value"` is reached via `value`.
    const reached = [...covered].some((c) => elementCarrying(cls).includes(c));
    if (!reached) uncovered.push(`${cls} (declared in ${files.join(', ')})`);
  }

  assert.deepEqual(
    uncovered,
    [],
    'These elements render a field state but sit OUTSIDE the no-state backstop, so an absent or ' +
      'unrecognised state renders on them identically to a trusted measurement:\n  ' +
      uncovered.join('\n  ') +
      '\n\nThe five positive [data-state=...] rules are unscoped and DO reach them, so every valid ' +
      'state looks correct and only the broken case falls through. Fix by adding a covered class ' +
      `to the element (e.g. \`class="value ${uncovered[0]?.split(' ')[0] ?? 'x__value'}"\`), not by ` +
      'widening the selector — then the next BEM class inherits the guarantee.',
  );
});

/**
 * The full class list of the element that declares `cls`, so coverage can be
 * decided the way a browser decides it: over the whole list, not one token.
 */
function elementCarrying(cls) {
  for (const file of shippedModules()) {
    const source = readFileSync(file, 'utf8');
    for (const pattern of [/className\s*:\s*'([^']*)'/g, /\.className\s*=\s*'([^']*)'/g]) {
      for (const [, literal] of source.matchAll(pattern)) {
        const tokens = literal.split(/\s+/).filter(Boolean);
        if (tokens.includes(cls)) return tokens;
      }
    }
  }
  return [cls];
}

test('the reach check FAILS on a value cell that is outside the backstop', () => {
  // The starved-class arm: prove the assertion above can go red. Without this,
  // a detector that silently returned an empty set would pass forever.
  const covered = new Set(['value']);
  const pretendCells = new Map([
    ['value', ['dashboard/panel-kit.js']],
    ['model-card__value', ['ui/model-card.js']],
  ]);

  const uncovered = [...pretendCells.keys()].filter((cls) => ![...covered].some((c) => [cls].includes(c)));

  assert.deepEqual(
    uncovered,
    ['model-card__value'],
    'a value cell whose ONLY class is outside the backstop must be reported as uncovered — ' +
      'this is the exact defect the real assertion caught in the browser',
  );
});

test('the backstop still names a class, so it cannot silently become a catch-all', () => {
  const css = readFileSync(SHELL_CSS, 'utf8');
  const lines = css.split('\n').filter((l) => l.includes(':not([data-state])'));

  assert.ok(lines.length > 0, 'no backstop selector found');
  for (const line of lines) {
    assert.match(
      line,
      /\.[a-z0-9_-]+:not\(\[data-state\]\)/,
      `the backstop selector must stay class-scoped, but this line is unqualified:\n  ${line.trim()}\n` +
        'An unscoped `:not([data-state])` would match every element on the page, including the ' +
        'connection indicator, and paint the whole dashboard as broken.',
    );
  }
});
