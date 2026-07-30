// A FILE THAT EXISTS AND A FILE THAT LOADS ARE DIFFERENT CLAIMS.
// A TOKEN THAT IS DEFINED AND A TOKEN THAT RENDERS ARE DIFFERENT CLAIMS.
//
// @e00032a4's formulation, which this file exists to mechanise:
//   "When you finish a file, verify something actually loads it."
//
// Every honesty mechanism in this project inspects a THING -- a value, a
// field, an identifier, a claim in prose. This file inspects RELATIONSHIPS,
// because that is where the project's real defects have been: three agents
// each correct about what they owned, and the seam between them broken. No
// tool any of them would naturally run can see a missing relationship, and a
// missing relationship produces NO ERROR, NO 404, AND NOTHING IN DEVTOOLS.
//
// WHY THIS MATTERS MORE HERE THAN IN AN ORDINARY APP:
//   The absence states -- em-dash, hatch, `n/a` -- are the entire mechanism by
//   which this page admits it does not know something. Per D142 the non-colour
//   channel is not reinforcement, it IS the signal: the absence tokens sit
//   within 1.001:1 of each other in grayscale, so a border-bottom or a hatch
//   is the ONLY thing distinguishing four states. An unloaded stylesheet
//   therefore does not degrade the honesty layer, it DELETES it -- and leaves
//   a page that still looks fine, because an unstyled em-dash reads as
//   ordinary content rather than as an admission.
//
//   That is the exact failure the demo exists to argue against, shipped inside
//   the demo: a page whose central claim is "we never fabricate" with the
//   machinery that communicates non-fabrication silently absent.
//
// THIS ALSO SETTLES A QUESTION THAT HAS BEEN ASKED FIVE TIMES BY HAND.
// `styles/panels.css` has been reported orphaned five times. It was linked at
// 00:09 in commit 3af5c8d7; `css/shell.css` (the path those reports cite) was
// deleted at 00:12 in f8c7d003; the `--og-na-*` consumers landed at 00:51 in
// 1089d39f. Every report was accurate when its author last read the disk and
// wrong when they sent it. Re-running a check against a cached read produces a
// FRESH TIMESTAMP ON A STALE FACT, and confidence rises while accuracy does
// not. A test settles it permanently, in the one place nobody has to trust a
// quotation: it reads the bytes at the moment it runs.
//
// MUTATIONS THIS TEST IS KNOWN TO FAIL ON (run them, don't trust this comment):
//   1. delete the panels.css <link> from index.html   -> orphaned stylesheet
//   2. add styles/unused.css                          -> orphaned stylesheet
//   3. add `--og-foo: red` to tokens.css              -> token with no consumer
//   4. rename --og-na-fg in panels.css + shell.css    -> token with no consumer

import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

const dir = (p) => fileURLToPath(new URL(p, import.meta.url));
const read = (p) => readFileSync(dir(p), 'utf8');

const html = read('./index.html');
const styleFiles = readdirSync(dir('./styles')).filter((f) => f.endsWith('.css'));
const css = Object.fromEntries(styleFiles.map((f) => [f, read(`./styles/${f}`)]));

describe('the asset graph has no orphans', () => {
  it('found stylesheets to check', () => {
    // Guards the guard: a zero-file scan would report PASS while inspecting
    // nothing, which is the "green because it looked at nothing" failure this
    // suite exists to stop.
    assert.ok(
      styleFiles.length >= 3,
      `Expected at least 3 stylesheets in styles/; found ${styleFiles.length}. ` +
        'If the directory moved, every assertion below is vacuously true.',
    );
  });

  it('links every stylesheet from index.html', () => {
    const linked = [...html.matchAll(/<link[^>]+href="([^"]+\.css)"/g)].map((m) =>
      m[1].split('/').pop(),
    );
    const orphans = styleFiles.filter((f) => !linked.includes(f));
    assert.deepEqual(
      orphans,
      [],
      'Stylesheet(s) exist but are never fetched by the browser. This produces ' +
        'no error, no 404 and nothing in DevTools -- the page simply renders ' +
        'without them. Per D142 the absence states are separated ONLY by their ' +
        'non-colour channel (they are within 1.001:1 in grayscale), so an ' +
        'unloaded panel stylesheet does not degrade the honesty layer, it ' +
        `deletes it while the page still looks fine.\nORPHANED: ${orphans.join(', ')}`,
    );
  });

  it('links no stylesheet that does not exist', () => {
    // The complement, and it is not symmetric: a dead <link> 404s silently and
    // is the reason five separate reports cited `css/shell.css`, a path
    // deleted at 00:12 in f8c7d003.
    const linked = [...html.matchAll(/<link[^>]+href="([^"]+\.css)"/g)].map((m) => m[1]);
    const dead = linked.filter((href) => {
      const name = href.split('/').pop();
      return !href.startsWith('http') && !styleFiles.includes(name);
    });
    assert.deepEqual(dead, [], `index.html links stylesheet(s) that do not exist: ${dead.join(', ')}`);
  });
});

describe('every design token reaches the screen', () => {
  // THE DESIGNER-OWNED HALF, and the one I care about most: tokens.css is
  // mine, and a token I define but nobody applies is a design decision I
  // believe I shipped and did not. It is the CSS form of the stale doc comment
  // (D163) -- it converts "I should check whether this renders" into "I
  // already know it does", and it is invisible to the person who wrote it.
  const definitions = [...css['tokens.css'].matchAll(/^\s*(--og-[\w-]+)\s*:/gm)].map(
    (m) => m[1],
  );
  // A TOKEN CAN BE CONSUMED FROM THREE PLACES, AND MY FIRST VERSION OF THIS
  // TEST ONLY READ ONE OF THEM. It scanned styles/*.css, reported
  // `--og-unavail-label` unconsumed, and was WRONG: sparkline.js:241 reads it
  // at runtime via readToken(), because a <canvas> cannot inherit a CSS custom
  // property and must fetch it from the computed style.
  //
  // I wrote an orphan-detector that was itself missing a relationship. That is
  // the same defect it exists to catch, in the instrument, on its first run --
  // and had I trusted it I would have deleted a live token and silently
  // unstyled the sparkline captions. An instrument that inspects PART of a
  // graph reports absence with exactly the confidence of one that inspected
  // all of it.
  const consumers = [
    ...styleFiles.filter((f) => f !== 'tokens.css').map((f) => css[f]),
    ...readdirSync(dir('.'))
      .filter((f) => f.endsWith('.js') || f.endsWith('.html'))
      .map((f) => read(`./${f}`)),
    ...readdirSync(dir('./dashboard'))
      .filter((f) => f.endsWith('.js'))
      .map((f) => read(`./dashboard/${f}`)),
  ].join('\n');

  it('defines tokens at all', () => {
    assert.ok(definitions.length > 20, `Expected many --og-* tokens; found ${definitions.length}.`);
  });

  it('has a consumer for every absence-state token', () => {
    // Scoped to the ABSENCE tokens rather than all of them, deliberately.
    // A palette may legitimately carry a spare shade; the absence set may not,
    // because each of those tokens IS a render state that the honesty layer
    // has promised a visitor it will show. `--og-na-*` in particular styles
    // `not-applicable`, which the Lead ruled a teaching surface and which must
    // be on screen rather than behind a hover (D169: no hover on touch, none
    // on keyboard, none in a screenshot).
    const absence = definitions.filter((t) =>
      /^--og-(unavail|na|pending|stale)-/.test(t),
    );
    assert.ok(absence.length >= 8, `Expected the absence token set; found ${absence.length}.`);

    const unused = absence.filter((t) => !consumers.includes(t));
    assert.deepEqual(
      unused,
      [],
      'Absence-state token(s) are defined but applied by no rule, so the state ' +
        'they style CANNOT RENDER. This is worse than an ugly treatment: the ' +
        'field still shows its em-dash, but with no hatch and no rule it reads ' +
        'as ordinary content rather than as an admission that we do not know ' +
        'something. The mechanism by which this page refuses to fabricate ' +
        `would be silently absent.\nUNCONSUMED: ${unused.join(', ')}`,
    );
  });
});
