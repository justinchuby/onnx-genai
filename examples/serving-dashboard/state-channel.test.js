// THE FIELD-STATE TREATMENTS MUST NOT BE DISTINGUISHED BY COLOUR ALONE.
//
// demo-ux.md §21 (five treatments) and D21: a visitor must be able to tell a
// measured value from an absent one in greyscale, in a compressed screenshot,
// and on a projector with the contrast wound down. Colour is the FIRST channel;
// every non-default state must carry a SECOND, non-colour channel.
//
// This test exists because it did not, and its absence was invisible. Deleting
// every `border-bottom` from the four non-default states in styles/shell.css
// left the full suite at 357/357 green while making `stale`, `unavailable` and
// `not-applicable` render identically -- their foreground tokens sit within a
// few RGB values of each other by design, because pattern was supposed to be
// the real carrier. An em-dash that cannot be told from a stale number is AC8
// dying silently, and nothing in the tree noticed.
//
// MUTATIONS THIS TEST IS KNOWN TO FAIL ON (run them, don't trust this comment):
//   1. remove `border-bottom` from [data-state='stale']        -> stale has no second channel
//   2. change not-applicable's `3px double` to `1px dotted`    -> collides with unavailable
//   3. delete the [data-state='not-applicable'] rule entirely  -> state has no rule at all
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import { FIELD_STATES } from './telemetry-field.js';

const SHELL_CSS = fileURLToPath(new URL('./styles/shell.css', import.meta.url));
const css = readFileSync(SHELL_CSS, 'utf8');

// DERIVED, NEVER HARDCODED. The first version of this file spelled these five
// strings literally, which made it blind to the one defect most likely to occur:
// the stylesheet and the JS constant drifting apart. They HAVE drifted twice in
// one session -- shell.css selected [data-state='measured'] while the DOM
// carried 'ok', so the rule matched nothing and looked perfect, because
// `measured` is styled as the inherited default. Reading the enum from its
// source of truth means a rename lands here as a red test instead of as a
// silent loss of every absence treatment on the page.
const DEFAULT_STATE = FIELD_STATES.MEASURED;
const ABSENCE_STATES = Object.freeze(
  Object.values(FIELD_STATES).filter((state) => state !== DEFAULT_STATE),
);

/**
 * Declarations that convey information through colour only. A state that
 * differs from another in these properties ALONE is invisible in greyscale.
 */
const COLOUR_ONLY = /^(color|background|background-color|fill|stroke|border-.*-color)$/;

/**
 * The non-colour declarations of a `[data-state='<state>']` rule, normalised so
 * two rules that render identically compare equal.
 * @param {string} state
 * @returns {string[]} sorted `prop:value` pairs, colour declarations removed
 */
function channelSignature(state) {
  const rule = new RegExp(`\\[data-state=['"]${state}['"]\\]\\s*\\{([^}]*)\\}`).exec(css);
  assert.ok(rule, `styles/shell.css has no [data-state='${state}'] rule at all`);

  return rule[1]
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .split(';')
    .map((decl) => decl.trim())
    .filter(Boolean)
    .map((decl) => {
      const at = decl.indexOf(':');
      return [decl.slice(0, at).trim().toLowerCase(), decl.slice(at + 1).replace(/\s+/g, ' ').trim()];
    })
    .filter(([prop]) => !COLOUR_ONLY.test(prop))
    .map(([prop, value]) => `${prop}:${value}`)
    .sort();
}

describe('field states are legible without colour', () => {
  it('gives every non-default state a second, non-colour channel', () => {
    for (const state of ABSENCE_STATES) {
      const signature = channelSignature(state);
      assert.ok(
        signature.length > 0,
        `[data-state='${state}'] is distinguished by colour alone. A visitor reading ` +
          'a greyscale screenshot or a projected slide cannot tell it from a measured ' +
          'value. Give it an underline, a font-style, or another non-colour channel.',
      );
    }
  });

  it('never lets two states share a channel signature', () => {
    // The absence states' foreground colours sit within a few RGB values of one
    // another ON PURPOSE -- pattern is meant to be the real carrier. So two
    // states with the SAME non-colour channel are, in greyscale, the same state.
    const seen = new Map();
    for (const state of ABSENCE_STATES) {
      const signature = channelSignature(state).join(' | ');
      const collision = seen.get(signature);
      assert.equal(
        collision,
        undefined,
        `'${state}' and '${collision}' are identical once colour is removed (${signature}). ` +
          'These are different facts -- stale is a value that WAS measured, unavailable is a ' +
          'gap plumbing could close, not-applicable is permanent -- and a greyscale reader ' +
          'would see one state where there are two.',
      );
      seen.set(signature, state);
    }
  });

  it('has a rule for every state the enum can produce', () => {
    // THE SEAM. A state with no rule falls through to default contrast and
    // renders like a measurement -- the exact failure this stylesheet exists to
    // prevent, arriving through the stylesheet itself. This assertion is the
    // only thing in the tree that observes the JS enum and the CSS together;
    // every other test reads one side and is satisfied.
    for (const state of Object.values(FIELD_STATES)) {
      assert.match(
        css,
        new RegExp(`\\[data-state=['"]${state}['"]\\]`),
        `FIELD_STATES contains '${state}' but styles/shell.css has no rule for it. ` +
          'Either the enum was renamed without the stylesheet, or the stylesheet was ' +
          'renamed without the enum. A state with no rule renders at full contrast, ' +
          'which is indistinguishable from a measured value.',
      );
    }
  });

  it('leaves the default state undecorated, so decoration always means doubt', () => {
    // demo-ux.md D65/§21: `measured` is the ABSENCE of a treatment. If it grew
    // an underline, decoration would stop meaning "do not trust this number".
    assert.deepEqual(
      channelSignature(DEFAULT_STATE),
      [],
      `[data-state='${DEFAULT_STATE}'] carries a non-colour treatment. The default state ` +
        'must stay undecorated so that any decoration reliably signals doubt.',
    );
  });

  it('reserves the value slot on states that render an em-dash', () => {
    // Layout shift on data arrival makes a page feel unreliable at exactly the
    // moment it becomes more reliable.
    for (const state of ['unavailable', 'not-applicable']) {
      assert.ok(
        channelSignature(state).some((decl) => decl.startsWith('min-width:')),
        `[data-state='${state}'] renders an em-dash but reserves no width, so the panel ` +
          'reflows the moment a real value arrives.',
      );
    }
  });
});
