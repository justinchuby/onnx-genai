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

  /**
   * D273 — CONTRAST IS A MEASUREMENT, AND NOTHING WAS MEASURING IT.
   *
   * `--og-unavail-rule` sat at 1.86:1 against a 3:1 requirement for the entire
   * session, underneath a comment in shell.css calling it "legible in greyscale,
   * in a compressed screenshot, and on a projector". `--og-stale-rule` sat at
   * 2.28:1. Both are the SECOND CHANNEL -- the dashed/dotted/double underline
   * that lets the four absence states stay distinguishable when colour is taken
   * away. A second channel nobody can see is not a second channel.
   *
   * Neither token carried a contrast annotation; their siblings all did. An
   * unannotated token is an unchecked one, so this stops relying on annotation
   * and computes the ratio from the shipped hex values.
   */
  describe('every state colour meets its contrast floor', () => {
    /** WCAG 2.2: 4.5:1 for body text, 3:1 for non-text (1.4.11) such as a rule. */
    const TEXT_MIN = 4.5;
    const NON_TEXT_MIN = 3.0;

    const relativeLuminance = (hex) =>
      [1, 3, 5]
        .map((i) => parseInt(hex.substr(i, 2), 16) / 255)
        .map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4))
        .reduce((sum, v, i) => sum + [0.2126, 0.7152, 0.0722][i] * v, 0);

    const ratio = (a, b) => {
      const [hi, lo] = [relativeLuminance(a), relativeLuminance(b)].sort((x, y) => y - x);
      return (hi + 0.05) / (lo + 0.05);
    };

    const hexOf = (token) => {
      const match = css['tokens.css'].match(new RegExp(`${token}:\\s*(#[0-9a-fA-F]{6})`));
      return match && match[1];
    };

    it('computes a known ratio correctly', () => {
      // ANTI-VACUITY, and the control D264 demands: an implementation that
      // returned a large number for everything would pass every assertion below
      // while measuring nothing. White on black is 21:1 by definition.
      assert.ok(Math.abs(ratio('#ffffff', '#000000') - 21) < 0.01, 'ratio() is wrong');
      assert.ok(Math.abs(ratio('#151b23', '#151b23') - 1) < 0.01, 'ratio() is wrong');
    });

    it('gives every absence state a legible foreground and rule', () => {
      const background = hexOf('--og-bg-raised');
      assert.ok(background, 'no --og-bg-raised to measure against');

      const failures = [];
      for (const token of definitions) {
        if (!/^--og-(unavail|na|pending|stale)-(fg|note|rule)$/.test(token)) continue;
        const hex = hexOf(token);
        if (!hex) continue;

        const isRule = token.endsWith('-rule');
        const floor = isRule ? NON_TEXT_MIN : TEXT_MIN;
        const measured = ratio(hex, background);
        if (measured < floor) {
          failures.push(`${token} ${hex} = ${measured.toFixed(2)}:1, needs ${floor}:1`);
        }
      }

      assert.deepEqual(
        failures,
        [],
        'A state colour is below its WCAG floor on the shipped background. The ' +
          'rule tokens are the second channel that carries absence when colour ' +
          'is removed -- on a projector, in greyscale, for a colourblind ' +
          'viewer -- so an invisible rule collapses four distinct admissions ' +
          `into one indistinguishable grey.\nFAILING:\n  ${failures.join('\n  ')}`,
      );
    });

    /**
     * D291. THE HEADER BLOCK CLAIMED "17.4 / 8.1 / 4.6 -- All clear WCAG AA"
     * against a background it NAMED, and all three figures were wrong on that
     * background. One of them, --og-fg-subtle, was really 4.10:1 and FAILED
     * AA while its own annotation certified it passed -- across 25 usages
     * including the provenance badges. Two earlier fixes (D270, D273) each
     * corrected a token whose comment asserted the property its value failed,
     * and BOTH LEFT THIS BLOCK UNTOUCHED, because a comment is not reachable
     * from the thing it describes.
     *
     * So the annotations are now parsed and recomputed. This deliberately
     * guards the PROPERTY, not the spelling: a test that compared hex
     * literals would pass when someone swapped two indistinguishable greys
     * and fail when someone made an improvement.
     */
    it('proves every written contrast annotation is arithmetically true', () => {
      const background = hexOf('--og-bg-raised');
      // `--token: #hex;  /* N:1 on --og-bg-raised ... */` on one line.
      const annotated =
        /(--og-[a-z0-9-]+):\s*(#[0-9a-fA-F]{6});\s*\/\*[^*]*?([0-9]+(?:\.[0-9]+)?):1 on --og-bg-raised/g;

      const checked = [];
      const failures = [];
      for (const [, token, hex, claimed] of css['tokens.css'].matchAll(annotated)) {
        const measured = ratio(hex, background);
        checked.push(token);
        // 0.05 absorbs the rounding of a 2-decimal annotation, nothing more.
        if (Math.abs(measured - Number(claimed)) > 0.05) {
          failures.push(
            `${token} ${hex} is annotated ${claimed}:1 but measures ${measured.toFixed(2)}:1`,
          );
        }
      }

      // ANTI-VACUITY, and it is the assertion that matters most here: if the
      // regex ever stops matching -- a reformat, a moved comment, a second
      // line -- the loop above runs zero times and reports a confident green
      // while checking nothing. That is the exact failure this whole file
      // exists to catch, so the checker must prove it still has subjects.
      // Pinned to the count observed when this landed. It caught its first
      // real gap immediately: --og-na-rule was annotated "3.01:1" while
      // naming NO background, so the claim was unfalsifiable as written and
      // the parser could not see it. An annotation without a reference
      // surface is not a weak check, it is not a check.
      assert.ok(
        checked.length >= 9,
        `only ${checked.length} annotated tokens found; the parser has lost ` +
          'its subjects and is now certifying an empty set.',
      );

      assert.deepEqual(
        failures,
        [],
        'A contrast annotation in tokens.css states a ratio its own colour ' +
          'does not have. These annotations are load-bearing: they are what ' +
          'a reader trusts instead of re-measuring, and a wrong one hid a ' +
          'real WCAG AA failure on the provenance badge for the whole ' +
          `session.\nFAILING:\n  ${failures.join('\n  ')}`,
      );
    });

    /**
     * D291, SECOND HALF -- AND I CREATED THIS GAP MYSELF, IN THE COMMIT
     * DIRECTLY ABOVE THIS ONE.
     *
     * The checker above only sees a ratio written in DECLARATION form:
     *   `--token: #hex;` followed by an inline comment reading
     *   `N:1 on --og-bg-raised`.
     * Its regex requires the `--token: #hex;` prefix. I then committed a
     * PROSE margin table into tokens.css -- four tokens, their ratios,
     * their floors and their margins to three decimals -- sitting in a
     * comment block with no declaration in front of it. Every number in it
     * was unguarded the moment I wrote it, in the file whose own header
     * says "a hand-written ratio is a CLAIM, and this file was full of
     * claims and had no way to check one".
     *
     * THE RULE: the guard was scoped to a SYNTAX when the property is about
     * a KIND OF SENTENCE. Anywhere a ratio is written down, a reader trusts
     * it instead of re-measuring -- that is what makes it load-bearing, and
     * the surrounding punctuation has nothing to do with it. A checker
     * pinned to one spelling leaves every other spelling free, and the next
     * author writes the other spelling because it reads better in prose.
     */
    it('proves every prose margin claim is arithmetically true', () => {
      const background = hexOf('--og-bg-raised');
      // `--token   N:1  vs FLOOR <label>   +MARGIN` inside any comment.
      const prose =
        /(--og-[a-z0-9-]+)\s+([0-9]+\.[0-9]+):1\s+vs\s+([0-9]+\.[0-9]+)[^\n]*?([+-][0-9]+\.[0-9]+)/g;

      const checked = [];
      const failures = [];
      for (const [, token, claimed, floor, margin] of css['tokens.css'].matchAll(prose)) {
        const hex = hexOf(token);
        if (!hex) {
          failures.push(`${token} is quoted in a margin table but is not declared in this file`);
          continue;
        }
        const measured = ratio(hex, background);
        // A rule or a stroke is non-text (WCAG 1.4.11); everything else here
        // is read as text (1.4.3). Getting this backwards is the whole point
        // of checking it: it is how a 3.0 floor gets applied to body copy.
        const expectedFloor = /-(rule|stroke)$/.test(token) ? 3.0 : 4.5;
        checked.push(token);

        if (Math.abs(measured - Number(claimed)) > 0.005) {
          failures.push(
            `${token} is quoted at ${claimed}:1 but measures ${measured.toFixed(3)}:1`,
          );
        }
        if (Number(floor) !== expectedFloor) {
          failures.push(
            `${token} is quoted against a ${floor} floor; a ${
              /-(rule|stroke)$/.test(token) ? 'rule/stroke is non-text (3.0)' : 'text token is 4.5'
            }`,
          );
        }
        if (Math.abs(measured - Number(floor) - Number(margin)) > 0.005) {
          failures.push(
            `${token} is quoted with margin ${margin} but ${measured.toFixed(3)} - ${floor} = ` +
              `${(measured - Number(floor)).toFixed(3)}`,
          );
        }
      }

      // ANTI-VACUITY. The prose table is the only subject this checker has,
      // and prose is far easier to reformat than a declaration -- rewrap the
      // paragraph and the regex silently matches nothing. A zero here must
      // be a failure, never a pass.
      assert.ok(
        checked.length >= 4,
        `only ${checked.length} prose margin claims found; the table was ` +
          'reformatted or removed and this checker is now certifying an ' +
          'empty set. Re-anchor it or delete it -- do not leave it green.',
      );

      assert.deepEqual(
        failures,
        [],
        'A margin table in tokens.css states a ratio, a floor or a margin ' +
          'that its own declared colour does not have. This table exists to ' +
          'tell the next author which token fails FIRST when the reference ' +
          'surface moves; a wrong row sends them to the wrong token.\n' +
          `FAILING:\n  ${failures.join('\n  ')}`,
      );
    });

    /**
     * D292. unavailable and pending were #758493 and #748494 -- a two-digit
     * transposition, 1.0014:1 apart. Colour is NOT the primary signal for
     * state (the glyph and the words are, and the border grammar is the
     * second channel), so this floor is deliberately modest: it is defence
     * in depth for a projector with crushed blacks, which is this demo's
     * actual viewing condition. It sits below the shipped separation so that
     * tuning stays possible, and far above the transposition that prompted
     * it so the collapse cannot silently return.
     */
    it('keeps the four absence states distinguishable from one another', () => {
      const family = ['--og-unavail-fg', '--og-pending-fg', '--og-stale-fg', '--og-na-fg'];
      const MIN_SEPARATION = 1.05;

      const hexes = family.map((t) => [t, hexOf(t)]);
      for (const [token, hex] of hexes) {
        assert.ok(hex, `${token} is missing; the ramp cannot be measured`);
      }

      const failures = [];
      let pairs = 0;
      for (let i = 0; i < hexes.length; i += 1) {
        for (let j = i + 1; j < hexes.length; j += 1) {
          pairs += 1;
          const separation = ratio(hexes[i][1], hexes[j][1]);
          if (separation < MIN_SEPARATION) {
            failures.push(`${hexes[i][0]} vs ${hexes[j][0]} = ${separation.toFixed(4)}:1`);
          }
        }
      }

      assert.equal(pairs, 6, 'expected all six pairs of a four-state family');
      assert.deepEqual(
        failures,
        [],
        'Two absence states render as the same colour. They mean different ' +
          'things -- we have no number / we are waiting / we had one and it ' +
          'aged out / this cannot apply here -- and a visitor who cannot ' +
          'separate them reads four different admissions as one.\nFAILING:\n  ' +
          failures.join('\n  '),
      );
    });

    /**
     * D292, and this is the assertion the previous one CANNOT make. A first
     * attempt at a wider 1.20:1 absence ramp cleared all six in-family pairs
     * and put --og-na-fg at 1.0359:1 against --og-fg-muted -- which is a
     * sibling in the same widget (.request-state--sent is --og-fg-muted,
     * .request-state--unknown is --og-unavail-fg, five lines apart in
     * panels.css). It would have MOVED the collision, not fixed it, and an
     * in-family-only test would have certified it green.
     *
     * A separation guard that only looks inside the family it is tuning has
     * the same blind spot as a coverage list that only names the files it
     * already knows about.
     */
    it('keeps absence states clear of the general-purpose text tokens', () => {
      const family = ['--og-unavail-fg', '--og-pending-fg', '--og-stale-fg', '--og-na-fg'];
      const neighbours = ['--og-fg', '--og-fg-muted'];
      const MIN_CLEARANCE = 1.15;

      const failures = [];
      let compared = 0;
      for (const token of family) {
        for (const neighbour of neighbours) {
          const a = hexOf(token);
          const b = hexOf(neighbour);
          assert.ok(a && b, `${token} or ${neighbour} is missing`);
          compared += 1;
          const clearance = ratio(a, b);
          if (clearance < MIN_CLEARANCE) {
            failures.push(`${token} vs ${neighbour} = ${clearance.toFixed(4)}:1`);
          }
        }
      }

      assert.equal(compared, 8, 'expected every absence state against every neighbour');
      assert.deepEqual(
        failures,
        [],
        'An absence colour has drifted into a general-purpose text token. ' +
          'That makes "we could not measure this" look identical to ordinary ' +
          'secondary text sitting beside it, which is worse than the ' +
          'in-family collapse: it does not blur two admissions together, it ' +
          `disguises an admission as a fact.\nFAILING:\n  ${failures.join('\n  ')}`,
      );
    });

    // The block map is the first surface with no text to underline, so the
    // dashed/dotted/double grammar cannot reach it. Its channel is hollow-vs-
    // filled, and these two assertions are what stop that channel decaying
    // into "two slightly different greys" the next time someone tunes them.
    it('keeps every block-map fill above the non-text floor', () => {
      const background = hexOf('--og-bg-raised');
      const ramp = definitions.filter((t) => /^--og-cell-[0-4]$/.test(t));

      // Anti-vacuity: a ramp that silently lost its tokens would pass the
      // loop below by iterating zero times, which is the failure mode that
      // cost this crew four false greens tonight.
      assert.equal(ramp.length, 5, `expected 5 fill levels, found ${ramp.length}`);

      const failures = [];
      for (const token of [...ramp, '--og-cell-stroke']) {
        const measured = ratio(hexOf(token), background);
        if (measured < NON_TEXT_MIN) {
          failures.push(`${token} = ${measured.toFixed(2)}:1, needs ${NON_TEXT_MIN}:1`);
        }
      }

      assert.deepEqual(
        failures,
        [],
        'A block-map cell is below the WCAG 1.4.11 floor. EVERY fill level is ' +
          'a MEASUREMENT -- including level 0, which means "we looked at this ' +
          'page and it was empty", not "we have nothing". Dimming level 0 ' +
          'towards the background to make the map look calmer is the exact ' +
          'move that turns a measured zero back into an absence.\n' +
          `FAILING:\n  ${failures.join('\n  ')}`,
      );
    });

    it('separates never-observed from measured-zero on two channels', () => {
      const separation = ratio(hexOf('--og-cell-stroke'), hexOf('--og-cell-0'));
      const adjacent = ratio(hexOf('--og-cell-0'), hexOf('--og-cell-1'));

      assert.ok(
        separation >= 1.3,
        'The CATEGORICAL boundary -- never-observed vs measured-zero -- has ' +
          `collapsed to ${separation.toFixed(3)}:1. This pair must stay ` +
          'redundantly coded: hollow-vs-filled AND a luminance step. The four ' +
          'text absence states share a 1.0014:1 grey and survive only on ' +
          'their underline grammar; a grid cell has no underline, so if this ' +
          'number goes flat the map has ONE channel and it is the one already ' +
          'measured worthless.',
      );

      assert.ok(
        separation > adjacent,
        `Categorical separation ${separation.toFixed(3)}:1 is no longer wider ` +
          `than the ordinal step ${adjacent.toFixed(3)}:1. Confusing 60% full ` +
          'with 80% full is a small error in a quantity; confusing "never ' +
          'looked" with "looked and found nothing" is a lie. The contrast ' +
          'budget belongs on the categorical boundary.',
      );
    });
  });
});

/**
 * D297 — A RULE IS A NON-TEXT CHANNEL AND MUST BE DRAWN FROM A NON-TEXT TOKEN.
 *
 * `pending` was the only state whose underline was painted with its own TEXT
 * colour (`var(--og-pending-fg)`). Every sibling already had a dedicated
 * `-rule`. That is not untidiness: a `-fg` is tuned to 4.5:1 for text
 * (WCAG 1.4.3) and a rule to 3:1 for non-text (1.4.11), so ONE VALUE WAS
 * ANSWERING TWO FLOORS and a retune for either silently moved the other.
 *
 * I did exactly that at 03:03:28 -- raised --og-pending-fg for TEXT contrast
 * and moved this underline with it, without knowing the underline existed.
 * The coupling was invisible because the rule had no name of its own; the
 * sparkline fossils survived the same way. A VALUE WITH NO NAME IS A VALUE
 * NOBODY SWEEPS.
 *
 * So this is the class, not the instance: one offender is a defect, a second
 * would be a pattern, and the check costs nothing.
 */
describe('every state rule is drawn from a -rule token', () => {
  // Comments are stripped FIRST. This file's own fix writes the string
  // `--og-pending-fg` into a shell.css comment explaining what not to do, and
  // a matcher that cannot tell a declaration from prose about a declaration
  // would read that warning as the very defect it warns about.
  const shell = css['shell.css'].replace(/\/\*[\s\S]*?\*\//g, '');
  const blocks = [...shell.matchAll(/\[data-state='([a-z-]+)'\][^{]*\{([^}]*)\}/g)].map(
    (m) => ({ state: m[1], body: m[2] }),
  );
  const ruled = blocks
    .map(({ state, body }) => {
      const m = body.match(/border-bottom\s*:[^;]*var\(\s*(--og-[\w-]+)\s*\)/);
      return m ? { state, token: m[1] } : null;
    })
    .filter(Boolean);

  it('actually finds the state rules (anti-vacuity)', () => {
    // Without this, a selector rename or a comment-stripping bug yields zero
    // blocks, zero offenders, and a permanently green test that inspects
    // nothing -- the vacuous pass that has caught six instruments this session.
    assert.ok(
      ruled.length >= 4,
      `Expected the state underlines; found ${ruled.length} ` +
        `(${ruled.map((r) => r.state).join(', ') || 'none'}). The matcher has ` +
        'gone blind -- fix the matcher, do not relax the threshold.',
    );
  });

  it('never paints a rule with a text token', () => {
    const offenders = ruled
      .filter(({ token }) => !token.endsWith('-rule'))
      .map(({ state, token }) => `[data-state='${state}'] -> ${token}`);
    assert.deepEqual(
      offenders,
      [],
      'A state underline is painted with a non-`-rule` token. The underline is ' +
        'the NON-COLOUR CHANNEL: with colour switched off it is the only thing ' +
        'separating four absence states, which sit within 1.05:1 of each other ' +
        'in greyscale. Binding it to a text token makes it collateral damage of ' +
        'the next text-contrast retune, and that retune will look correct and ' +
        `test green.\nOFFENDERS: ${offenders.join(', ')}`,
    );
  });
});

// D298. THE EXPIRY GATE FOR A SPECIFIED-NOT-BUILT CAVEAT.
//
// The provenance badge has no `unknown` variant: `panel-kit.js` resolves an
// absent or unrecognised source to `derived` via `?? SOURCE_BADGES.derived`,
// so "we have no provenance for this" renders as a confident "derived by
// arithmetic on measured inputs". It is SPECIFIED in demo-ux.md D298 and
// deliberately NOT BUILT tonight -- the two developers who could build it hold
// the security and poll-loop blockers.
//
// This test does NOT assert the defect is correct. Pinning a defect in place
// is exactly the anti-pattern @73e77d95 found in telemetry-store.test.js:1348,
// where an assertion required the very disclosure its comment diagnosed.
//
// It asserts something narrower and safe in both directions: THE DOCUMENT AND
// THE STYLESHEET MUST NOT DISAGREE ABOUT WHETHER THE FIX EXISTS. @12e42da8's
// ruling on caveats -- "the caveat must be present while the fields are
// unpublished, and must be DELETED the day they land" -- has no mechanism
// anywhere in this repository, and a caveat that outlives its defect trains
// readers to skip caveats. Nobody ever files that bug.
//
// GREEN TODAY (not built, documented as not built). GREEN AFTER THE FIX, IF
// AND ONLY IF the section is retired in the same change. RED for whoever
// builds it and leaves the obituary armed.
describe('a specified-not-built caveat expires with its defect', () => {
  const panels = css['panels.css'] ?? '';
  const spec = read('./design/demo-ux.md');

  const variants = [...panels.matchAll(/\.value__src--([a-z-]+)/g)].map((m) => m[1]);
  const built = variants.includes('unknown');
  const caveatArmed = spec.includes('SPECIFIED, NOT BUILT');

  it('actually finds the provenance vocabulary (anti-vacuity)', () => {
    assert.ok(
      variants.length >= 4,
      `Expected the .value__src-- badge variants; found ${variants.length} ` +
        `(${variants.join(', ') || 'none'}). Either the block was renamed or the ` +
        'matcher has gone blind. A guard that finds nothing has not passed, it ' +
        'has failed to run -- fix the matcher, do not relax the threshold.',
    );
  });

  it('retires the caveat in the same change that builds the fix', () => {
    assert.ok(
      !(built && caveatArmed),
      'RESOLVED: `.value__src--unknown` now exists in panels.css, so the D298 ' +
        'provenance-unknown badge HAS been built -- but demo-ux.md still carries ' +
        'the "SPECIFIED, NOT BUILT" section describing it as outstanding. Delete ' +
        'that section (§87) and record the commit that built it. A caveat that ' +
        'outlives its defect is worse than no caveat: it is committed, confident ' +
        'and false, and it teaches the next reader to skip the real ones.',
    );
  });
});

// D301. THE TEMPLATE-BLINDNESS FIX, AND THE CLOSED-SET RECONCILIATION.
//
// Every other check in this file matches LITERAL class names in CSS. The
// provenance badge defeats that by construction:
//
//     className: ['value__src', `value__src--${sourceClass}`]   sourceBadge()
//
// The name is BUILT AT RUNTIME, so a literal scan sees zero emitters and
// concludes the entire vocabulary is dead CSS. That is not a hypothetical:
// scanning literally would condemn the provenance badge and the connection
// indicator -- the honesty layer itself -- as unreachable.
//
// THE FIX IS TO STOP SCANNING FOR THE NAME AND RECONCILE THE CLOSED SET.
// `SOURCE_BADGES` is a frozen object literal, so the complete set of values
// `sourceClass` can take is DECLARED, in one place, statically. Resolving the
// template against that enum turns an unanswerable question ("what strings
// reach this interpolation?") into an answerable one ("do the two declared
// sets agree?").
//
// ⛔ AND THE HONEST LIMIT, WHICH IS WHY D301 IS SPECIFIED AND NOT CLAIMED AS
// SOLVED: this proves a name CAN BE BUILT and that both sides declare it. It
// CANNOT prove a branch is ever taken. The levels are styled -> constructible
// -> reachable; this measures the second. Only a browser load measures the
// third, and `?? SOURCE_BADGES.derived` fires on a branch no static instrument
// can prove is entered.
describe('the provenance vocabulary reconciles as a closed set', () => {
  const panelKit = read('./dashboard/panel-kit.js');
  const panels = css['panels.css'] ?? '';

  const block = panelKit.match(/SOURCE_BADGES = Object\.freeze\(\{([\s\S]*?)\n\}\)/);
  const declared = block ? [...block[1].matchAll(/^\s{2}([a-z][a-z-]*):/gm)].map((m) => m[1]) : [];
  const styled = [...new Set([...panels.matchAll(/\.value__src--([a-z-]+)/g)].map((m) => m[1]))];

  it('actually finds both declared sets (anti-vacuity)', () => {
    assert.ok(
      declared.length >= 4 && styled.length >= 4,
      `Expected both vocabularies; found ${declared.length} declared in ` +
        `SOURCE_BADGES (${declared.join(', ') || 'none'}) and ${styled.length} ` +
        `styled in panels.css (${styled.join(', ') || 'none'}). An empty side ` +
        'makes the reconciliation below vacuously true -- it would agree that ' +
        'nothing matches nothing. Fix the matcher, do not relax the threshold.',
    );
  });

  it('styles every badge it can emit, and emits every badge it styles', () => {
    const unstyled = declared.filter((k) => !styled.includes(k));
    const unemittable = styled.filter((k) => !declared.includes(k));
    assert.deepEqual(
      { unstyled, unemittable },
      { unstyled: [], unemittable: [] },
      'The provenance vocabulary has drifted between JS and CSS.\n' +
        `UNSTYLED (SOURCE_BADGES can emit it, panels.css has no rule): ${unstyled.join(', ') || 'none'}\n` +
        `UNEMITTABLE (panels.css styles it, nothing can produce it): ${unemittable.join(', ') || 'none'}\n` +
        'An UNSTYLED badge renders with no provenance colour and looks like a ' +
        'rendering glitch rather than a claim. An UNEMITTABLE rule is dead CSS ' +
        'that reads as coverage -- it makes the vocabulary look complete to the ' +
        'next person who greps it, which is exactly how D298 stayed invisible.',
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// D302-D304  ONE CONCEPT, THREE DECLARATIONS, TWO ANSWERS.
//
// The block above reconciles `SOURCE_BADGES` against `panels.css` and proves
// they agree. It passes. It has always passed. AND IT WAS BLIND TO THE FACT
// THAT `SOURCE_BADGES` IS NOT THE ONLY PROVENANCE VOCABULARY IN THE TREE.
//
// There are THREE declarations of the same concept:
//
//   SOURCE_BADGES        dashboard/panel-kit.js   5 keys, incl. `simulated`
//   SOURCE_CLASS_BADGES  format.js                4 keys, no `simulated`
//   SOURCE_CLASSES       telemetry-field.js       4 values -- THE CANONICAL ENUM
//
// ⛔ THIS GUARD IS THE SEVENTH INSTRUMENT TONIGHT CARRYING THE DEFECT IT HUNTS,
// AND IT IS MINE FOR THE SECOND TIME. I built a closed-set reconciliation to
// kill the "does this name reach the screen?" question, published its honest
// limit (styled -> constructible -> reachable), and never asked whether the
// SET I closed over was the only one. A reconciliation between two declarations
// proves those two agree. IT SAYS NOTHING ABOUT A THIRD, and its green is read
// as a statement about the vocabulary rather than about the pair it compared.
// That is @e00032a4's `0 positional` exactly: a guard's green is a claim about
// the subset it can see, and mine never said which subset that was.
//
// ☠️ AND THE DIVERGENCE IS NOT COSMETIC -- THE FALLBACKS DISAGREE ON WHETHER
// AN UNKNOWN PROVENANCE IS A CLAIM OR AN ABSENCE:
//
//   panel-kit.js  sourceBadge()           SOURCE_BADGES[sourceClass] ?? SOURCE_BADGES.derived
//   panel-kit.js  normaliseSourceClass()  trailing `return 'derived';`
//   format.js     badge lookup            SOURCE_CLASS_BADGES[field.sourceClass] ?? null   ✅
//
// Two sites answer "we don't know" with a confident DERIVED badge -- a claim
// that we computed the number ourselves. One answers with `null`, an honest
// absence. THE HONEST ANSWER ALREADY EXISTS IN THIS REPOSITORY. D302 is
// therefore not a design demand; it is a request that the fix TRAVEL to the
// two sites that never heard it.
//
// ✅ THE `simulated` ASYMMETRY IS REAL AND IS DECLARED HERE RATHER THAN
// SILENTLY TOLERATED. It is styled, and it is constructible only via the
// `source in SOURCE_BADGES` branch -- it is absent from the canonical enum, so
// no writer of `sourceClass` can ever produce it. Recording it as a dated
// exemption means the guard passes today, goes RED if a SECOND asymmetry
// appears, AND goes RED when this one is resolved -- so the note cannot outlive
// the condition it describes.
const PROVENANCE_ASYMMETRIES = Object.freeze({
  simulated: Object.freeze({
    recordedAt: '2026-07-30',
    absentFrom: Object.freeze(['SOURCE_CLASS_BADGES', 'SOURCE_CLASSES']),
    reason:
      'Declared in SOURCE_BADGES and styled in panels.css, but absent from the ' +
      'canonical SOURCE_CLASSES enum, so no writer of field.sourceClass can ' +
      'produce it. Constructible only if a field.source is literally the ' +
      'string "simulated". Tracked by D303 -- resolve by adding it to the enum ' +
      'or removing it from the badge map, then delete this entry.',
  }),
});

describe('one provenance concept has one vocabulary', () => {
  const panelKit = read('./dashboard/panel-kit.js');
  const format = read('./format.js');
  const telemetryField = read('./telemetry-field.js');

  // Keys of a frozen badge map. Handles both the literal form (`derived:`) and
  // the computed form (`[SOURCE_CLASSES.DERIVED]:`) so the two maps are read by
  // ONE instrument -- a per-map matcher would let a spelling difference read as
  // a vocabulary difference.
  const badgeKeys = (src, name) => {
    const block = src.match(
      new RegExp(`${name} = Object\\.freeze\\(\\{([\\s\\S]*?)\\n\\}\\)`),
    );
    if (!block) return [];
    return [
      ...block[1].matchAll(/^\s{2}(?:\[SOURCE_CLASSES\.([A-Z_]+)\]|([a-z][a-z-]*)):/gm),
    ].map((m) => (m[1] ? m[1].toLowerCase() : m[2]));
  };

  // The canonical enum declares VALUES, not keys: `SERVER: 'server'`.
  const enumValues = (src, name) => {
    const block = src.match(
      new RegExp(`${name} = Object\\.freeze\\(\\{([\\s\\S]*?)\\n\\}\\)`),
    );
    if (!block) return [];
    return [...block[1].matchAll(/^\s{2}[A-Z_]+:\s*'([a-z-]+)'/gm)].map((m) => m[1]);
  };

  const sets = {
    SOURCE_BADGES: badgeKeys(panelKit, 'SOURCE_BADGES'),
    SOURCE_CLASS_BADGES: badgeKeys(format, 'SOURCE_CLASS_BADGES'),
    SOURCE_CLASSES: enumValues(telemetryField, 'SOURCE_CLASSES'),
  };

  it('actually finds all three declarations (anti-vacuity)', () => {
    const found = Object.entries(sets).map(([k, v]) => `${k}=${v.length}`);
    assert.ok(
      Object.values(sets).every((v) => v.length >= 4),
      `Expected three provenance declarations of >=4 entries; found ${found.join(', ')}.\n` +
        'A set that reads EMPTY makes every comparison below vacuously true -- ' +
        'it would agree that nothing differs from nothing, and report GREEN for ' +
        'a vocabulary it never located. This is the arm that distinguishes ' +
        '"the vocabularies agree" from "my matcher broke". FIX THE MATCHER, ' +
        'DO NOT RELAX THE THRESHOLD.',
    );
  });

  it('declares every divergence between the three, and no more', () => {
    const union = [...new Set(Object.values(sets).flat())].sort();
    const undeclared = [];
    for (const name of union) {
      const absentFrom = Object.keys(sets).filter((s) => !sets[s].includes(name));
      if (absentFrom.length === 0) continue;
      const known = PROVENANCE_ASYMMETRIES[name];
      if (!known) {
        undeclared.push(`${name} (absent from ${absentFrom.join(', ')})`);
        continue;
      }
      const expected = [...known.absentFrom].sort().join(', ');
      const actual = [...absentFrom].sort().join(', ');
      if (expected !== actual) {
        undeclared.push(`${name} (declared absent from ${expected}; ACTUALLY ${actual})`);
      }
    }
    assert.deepEqual(
      undeclared,
      [],
      'The provenance vocabularies have diverged in a way nobody wrote down.\n' +
        `${undeclared.join('\n')}\n\n` +
        'One concept is declared three times -- SOURCE_BADGES (panel-kit.js), ' +
        'SOURCE_CLASS_BADGES (format.js) and the canonical SOURCE_CLASSES enum ' +
        '(telemetry-field.js). A name present in one and missing from another ' +
        'is either a badge that can never be produced or a class that can never ' +
        'be painted, and BOTH render as a confident DERIVED via the `??` ' +
        'fallbacks in sourceBadge() and normaliseSourceClass(). Add it everywhere, remove it ' +
        'everywhere, or record it in PROVENANCE_ASYMMETRIES with a reason.',
    );
  });

  it('retires an asymmetry note once the asymmetry is gone', () => {
    const union = [...new Set(Object.values(sets).flat())];
    const stale = Object.keys(PROVENANCE_ASYMMETRIES).filter((name) => {
      const absentFrom = Object.keys(sets).filter((s) => !sets[s].includes(name));
      return absentFrom.length === 0 && union.includes(name);
    });
    const orphaned = Object.keys(PROVENANCE_ASYMMETRIES).filter(
      (name) => !union.includes(name),
    );
    assert.deepEqual(
      { stale, orphaned },
      { stale: [], orphaned: [] },
      'A declared asymmetry no longer describes the tree.\n' +
        `RESOLVED but still noted: ${stale.join(', ') || 'none'}\n` +
        `NOTED but absent from every vocabulary: ${orphaned.join(', ') || 'none'}\n` +
        'A caveat that outlives its defect is worse than no caveat: it tells ' +
        'the next reader that a known problem is still open, and they will ' +
        'either re-investigate a corpse or treat the whole list as noise. ' +
        'Delete the entry in the SAME commit that closes the asymmetry.',
    );
  });
});

// ---------------------------------------------------------------------------
// D309 -- the field-state grammar must not leak onto non-field elements.
//
// FOUND IN A REAL BROWSER, NOT IN THIS SUITE. `[data-state='not-applicable']`
// was declared UNQUALIFIED in shell.css, so it matched every element carrying
// the attribute -- including `<aside class="scenario-switcher__note">`, a
// multi-paragraph explanatory panel. Measured with getComputedStyle against the
// SERVED stylesheets, with a paired control element that omitted the attribute:
//
//   property             with data-state     without (control)
//   display              inline-block        block        <- layout change
//   min-width            40.94px             0px          <- a VALUE SLOT
//   border-bottom        3px double          none
//   cursor               help                auto
//   color                rgb(133,151,171)    rgb(133,151,171)   <- IDENTICAL
//   border-left-width    3px                 3px               <- IDENTICAL
//
// The two identical rows are why this survived every review: `.scenario-
// switcher__note` and the not-applicable state BOTH resolve to --og-na-fg, so
// the one channel a human eye checks agreed exactly. THE DEFECT WAS HIDDEN
// BEHIND A COINCIDENTAL COLOUR MATCH, and only `display` and `min-width`
// disclosed it.
//
// The correct form already exists in the same file -- `.value[data-state=...]`
// -- so this is not a new convention, it is an unenforced one.
//
// MUTATION PROOF (run, not assumed):
//   remove the `.value` qualifier from any qualified rule  -> RED, names it
//   set ALLOWED_UNQUALIFIED to []                          -> RED, lists 5
//   delete every [data-state] rule from the corpus         -> RED via the
//                                                             anti-vacuity arm
describe('the field-state grammar stays on fields', () => {
  // DATED EXEMPTION, 04:29. These five are the CURRENT unqualified rules. They
  // are recorded rather than fixed here because the fix is NOT the obvious one
  // and belongs to whoever owns shell.css.
  //
  // ⛔ DO NOT "FIX" THESE BY QUALIFYING THEM TO `.value[data-state=...]`.
  // I nearly did. A census of the RENDERED DOM says that would be a regression
  // of exactly the kind this project exists to prevent:
  //
  //    53  <span class="value">            .value   ✅ covered
  //     3  <dd   class="model-card__value"> NOT .value  ⛔ WOULD LOSE STYLING
  //     1  <div  class="connection-indicator">  has its own qualified rules ✅
  //     1  <aside class="scenario-switcher__note">  THE DEFECT ⛔
  //
  // Two of those three <dd> are `unavailable` and one is `measured`. Qualifying
  // to `.value` alone would strip the absence grammar from the model card, so
  // an UNAVAILABLE value would render as confident plain text. THE NAIVE FIX
  // CONVERTS A COSMETIC LEAK INTO AN HONESTY DEFECT.
  //
  // ✅ THE SAFE FIX, VERIFIED AGAINST THE DOM -- name both value classes:
  //      .value[data-state='X'], .model-card__value[data-state='X'] { ... }
  //    which covers all 56 genuine field values and excludes the <aside>.
  //
  // This list must SHRINK. Adding to it requires the same DOM census.
  const ALLOWED_UNQUALIFIED = [
    "[data-state='measured']",
    "[data-state='pending']",
    "[data-state='stale']",
    "[data-state='unavailable']",
    "[data-state='not-applicable']",
  ];

  const stateRules = Object.entries(css).flatMap(([file, text]) =>
    [...text.matchAll(/^([^\n{}]*\[data-state=[^\]]+\][^\n{}]*)\{/gm)].map((m) => ({
      file,
      selector: m[1].trim(),
    })),
  );

  it('found [data-state] rules at all (anti-vacuity)', () => {
    assert.ok(
      stateRules.length >= 5,
      `Only ${stateRules.length} [data-state] rules found across ${
        Object.keys(css).length
      } stylesheets. The matcher has stopped seeing the state grammar, so every ` +
        'assertion below is vacuously green. Fix the pattern, not this number.',
    );
  });

  it('no [data-state] rule is unqualified', () => {
    const unqualified = stateRules
      .filter(({ selector }) => /^\[data-state=/.test(selector))
      .filter(({ selector }) => !ALLOWED_UNQUALIFIED.includes(selector))
      .map(({ file, selector }) => `${file}: ${selector}`);

    assert.deepEqual(
      unqualified,
      [],
      'These [data-state] rules are unqualified, so they style ANY element ' +
        'carrying the attribute -- not just field values:\n  ' +
        unqualified.join('\n  ') +
        '\n\nThis is not hypothetical: <aside class="scenario-switcher__note" ' +
        'data-state="not-applicable"> renders as display:inline-block with a ' +
        'numeric value-slot min-width because of exactly this. Qualify each ' +
        'rule with BOTH value classes -- `.value[data-state=X], ' +
        '.model-card__value[data-state=X]` -- and read the comment above ' +
        'before using `.value` alone, which strips the model card.',
    );
  });

  it('the exemption list has not become fiction', () => {
    const stillBare = stateRules
      .filter(({ selector }) => /^\[data-state=/.test(selector))
      .map(({ selector }) => selector);
    const retired = ALLOWED_UNQUALIFIED.filter((s) => !stillBare.includes(s));

    assert.deepEqual(
      retired,
      [],
      `These selectors are exempted as unqualified but are no longer bare in ` +
        `the stylesheets:\n  ${retired.join('\n  ')}\n\n` +
        'Somebody fixed them -- thank you. DELETE THEM FROM ALLOWED_UNQUALIFIED ' +
        'in the same commit. An exemption that has stopped describing the tree ' +
        'is a standing invitation to re-investigate a corpse, and it makes the ' +
        'remaining entries look equally stale.',
    );
  });
});

// A withdrawn figure is not removed from this document -- 14 sites are left
// visible, because a correction that silently patches the record is
// indistinguishable from never having been wrong. What must never happen is the
// sites outliving the notice that they are withdrawn.
//
// The banner is delimited by HTML comments so the guard can tell a WITHDRAWAL
// from a CLAIM. Without that, the banner (which must name the figure to be
// useful) is byte-identical to the thing it withdraws -- the defect class found
// five times tonight in four languages: documenting a defect trips the guard
// against the defect. The delimiter is the markdown analogue of `isCommand()`
// and of `grep -v '^\s*///'`.
describe('a withdrawn figure keeps its withdrawal notice', () => {
  const WITHDRAWN = '2.46';
  const START = 'WITHDRAWN-METRIC-BANNER:START';
  const END = 'WITHDRAWN-METRIC-BANNER:END';
  const doc = read('./design/demo-ux.md');

  // Strip the banner, then count. Counting the raw document would include the
  // banner's own mention and could never reach zero, so the retirement arm
  // below would be unreachable -- a guard that cannot retire is a guard that
  // becomes fiction.
  const outsideBanner = () => {
    const s = doc.indexOf(START);
    const e = doc.indexOf(END);
    if (s === -1 || e === -1) return doc;
    return doc.slice(0, s) + doc.slice(e + END.length);
  };
  const liveSites = () => outsideBanner().split(WITHDRAWN).length - 1;

  it('can still find the document and the figure (anti-vacuity)', () => {
    assert.ok(
      doc.length > 10000,
      `design/demo-ux.md read as ${doc.length} bytes. The path is wrong or the ` +
        `file moved, and every assertion below would pass vacuously.`,
    );
    assert.ok(
      doc.includes(WITHDRAWN),
      `The string ${WITHDRAWN} does not appear in demo-ux.md at all -- not even ` +
        `in the banner. Either the figure was fully removed (in which case delete ` +
        `this whole describe block and the banner together) or the matcher broke.`,
    );
  });

  it('names the withdrawal wherever the figure still stands', () => {
    if (liveSites() === 0) return; // the retirement arm below owns this case
    assert.ok(
      doc.includes(START) && doc.includes(END),
      `demo-ux.md states the withdrawn figure ${WITHDRAWN} in ${liveSites()} ` +
        `place(s), but carries no WITHDRAWN-METRIC-BANNER. The figure was ` +
        `withdrawn at its source in 2d6b36ac for a reason that RE-RUNNING CANNOT ` +
        `FIX (model provenance, not arithmetic). Restore the banner or remove ` +
        `every site -- shipping the sites without the notice is the only ` +
        `unacceptable state.`,
    );
    assert.ok(
      doc.includes('four sequences decoded in one step'),
      `The withdrawal banner exists but no longer names its replacement. A ` +
        `withdrawal that leaves a hole gets refilled with the withdrawn value ` +
        `by the next author. Name the count, not the ratio.`,
    );
  });

  it('retires the banner once the figure is gone', () => {
    if (liveSites() > 0) return;
    assert.ok(
      !doc.includes(START),
      `No live site states ${WITHDRAWN} any more, but the withdrawal banner is ` +
        `still here. It now warns about nothing and is the ONLY remaining source ` +
        `of the figure in this document -- delete it. An expired notice does not ` +
        `merely stop teaching, it re-teaches the thing it retracted.`,
    );
  });
});

// The palette/page boundary. Every guard above this line reads token
// DEFINITIONS; the state rules read token USES. Nothing owned the wire between
// them, which is how `[data-state='not-applicable']` spent the `--og-unavail-*`
// family -- the palette declared a brightness gap, the page never asked for it,
// and both halves were independently correct. Mutating a token value proves
// nothing about that: `I broke it and it went red` is only evidence if the test
// was looking at the thing you broke.
//
// COMMENTS ARE STRIPPED FIRST, and that is not defensive tidiness -- shell.css
// contains `the colour is --og-pending-rule, NOT --og-pending-fg` inside the
// `pending` block. Matching raw text reads a warning ABOUT a token as a USE of
// it: the same defect class as a fix quoting the bug it killed.
describe('every state selector consumes the token family named for it', () => {
  const shell = read('./styles/shell.css');
  // measured is a DOCUMENTED exemption and the documentation is NOT mine:
  // tokens.css says there is deliberately NO --og-measured-fg, because measured
  // "is not a treatment, it is the ABSENCE of one" -- full-contrast --og-fg, the
  // page default. The exemption below is not a concession, it is that ruling.
  //
  // I first recorded this exemption as "--og-measured-fg exists and is consumed
  // by nothing". FALSE: a name-only grep matched the token inside the very
  // comment declaring it must never exist. Third time tonight that prose ABOUT
  // a thing read as the thing. The control below binds this exemption to that
  // note, so if anyone ever mints the token the exemption is revisited rather
  // than silently protecting a state from the guard.
  const FAMILY = {
    measured: null,
    pending: 'pending',
    stale: 'stale',
    unavailable: 'unavail',
    'not-applicable': 'na',
  };
  const stripComments = (s) => s.replace(/\/\*[\s\S]*?\*\//g, '');

  const block = (state) => {
    const m = shell.match(
      new RegExp(`^\\[data-state='${state}'\\]\\s*\\{([\\s\\S]*?)^\\}`, 'm'),
    );
    return m ? stripComments(m[1]) : null;
  };

  it('finds a bare rule for every state (non-zero floor)', () => {
    const missing = Object.keys(FAMILY).filter((s) => block(s) === null);
    assert.deepEqual(
      missing,
      [],
      `No [data-state='X'] block found in shell.css for: ${missing.join(', ')}. ` +
        `Either the rule was deleted -- in which case that state falls through to ` +
        `default contrast and renders like a measurement -- or this matcher no ` +
        `longer matches, and every assertion below would pass by finding nothing.`,
    );
  });

  // HONEST SCOPE: no state block contains `var(--og-…)` inside a comment TODAY,
  // and the matcher requires `var(`, so stripping changes no current result. It
  // is defence against where this file is already trending: the not-applicable
  // block's comment names `--og-unavail-` in prose, and the more precisely a
  // future author quotes the old wiring -- `var(--og-unavail-fg)` -- the more
  // confidently an unstripped guard reports the defect that comment commemorates.
  it('removes commented-out token uses before matching (control)', () => {
    const synthetic = 'color: var(--og-na-fg);\n/* was var(--og-unavail-fg) */';
    assert.ok(
      /var\(\s*--og-unavail-fg\s*\)/.test(synthetic),
      'Control setup is broken: the synthetic block does not contain the ' +
        'commented token it exists to test.',
    );
    assert.ok(
      !/var\(\s*--og-unavail-fg\s*\)/.test(stripComments(synthetic)),
      'stripComments left a commented var() in place. Any state block whose ' +
        'comment quotes the wiring it replaced would be read as still having it.',
    );
    const na = shell.match(/^\[data-state='not-applicable'\]\s*\{([\s\S]*?)^\}/m)[1];
    assert.ok(
      na.includes('--og-unavail-'),
      'The not-applicable block no longer names --og-unavail- in its comment. ' +
        'This half of the control is now dead: it proved the stripper runs on ' +
        'REAL text, not just a synthetic string. Re-point it or drop it.',
    );
    assert.ok(
      !stripComments(na).includes('--og-unavail-'),
      'stripComments did not run on the real not-applicable block.',
    );
  });

  it('keeps the measured exemption tied to the ruling that grants it', () => {
    const tokens = read('./styles/tokens.css');
    assert.ok(
      /deliberately NO .*--og-measured-fg/.test(tokens),
      'tokens.css no longer states that --og-measured-fg deliberately does not ' +
        'exist. The measured exemption in this guard rests on that ruling, so ' +
        'it is now an unexplained hole: measured is the ONE state this guard ' +
        'does not check. Either restore the note or wire measured to a family ' +
        'and delete the exemption.',
    );
    assert.ok(
      !/var\(\s*--og-measured-fg\s*\)/.test(read('./styles/shell.css')),
      'shell.css now CONSUMES --og-measured-fg, which tokens.css says must not ' +
        'exist. One of the two is wrong and the page is the one that renders.',
    );
  });

  it('wires each state to its own family, for colour and for rule', () => {
    const wrong = [];
    for (const [state, fam] of Object.entries(FAMILY)) {
      const body = block(state);
      if (body === null || fam === null) continue;
      const used = [...body.matchAll(/var\(\s*(--og-[a-z-]+?)-(fg|rule)\s*\)/g)];
      for (const [, prefix, kind] of used) {
        const want = `--og-${fam}`;
        if (prefix !== want) {
          wrong.push(
            `[data-state='${state}'] consumes ${prefix}-${kind} but is named ` +
              `for the '${fam}' family (${want}-${kind}). A state that spends ` +
              `its neighbour's tokens renders as that neighbour, and the ` +
              `palette cannot tell you -- both halves stay individually correct.`,
          );
        }
      }
    }
    assert.deepEqual(wrong, [], `Cross-family token use:\n  ${wrong.join('\n  ')}`);
  });

  it('actually inspected some token uses (anti-vacuity)', () => {
    let n = 0;
    for (const [state, fam] of Object.entries(FAMILY)) {
      if (fam === null) continue;
      const body = block(state);
      if (body) n += [...body.matchAll(/var\(\s*--og-[a-z-]+?-(fg|rule)\s*\)/g)].length;
    }
    assert.ok(
      n >= 6,
      `Only ${n} state token use(s) inspected across 4 families. Every state ` +
        `declares at least a colour, so a number this low means the var() ` +
        `matcher or the block matcher narrowed and the wiring check above is ` +
        `passing on an empty set.`,
    );
  });
});

/*
 * ═══════════════════════════════════════════════════════════════════════════
 * AN UNKNOWN STATE CANNOT RENDER AS A MEASURED ONE
 *
 * `state-channel.test.js` pins EVERY ENUM VALUE HAS A SELECTOR. That is a
 * bijection over the enum, and a bijection guard is blind to the COMPLEMENT of
 * its own domain -- which is infinite. Nothing in this repository has ever
 * asked what renders for a value OUTSIDE the enum: a typo, a renamed constant,
 * a state from a newer server, an attribute that never got stamped.
 *
 * WHICH WAY DOES THE CATCH-ALL ROT? This was ordered on the premise that the
 * `:not()` chain at `shell.css` "rots the day a sixth state lands." It does,
 * but NOT in the direction the order assumed, and the difference decides the
 * fix. A sixth state falls THROUGH the chain and collects the warn colour, the
 * wavy underline and the `NO STATE` chip -- it fails CLOSED, loudly. The chain
 * rots toward FALSE ALARM, which is the safe direction and the opposite of
 * rendering as confidently measured.
 *
 * THE DIRECTION THAT FAILS OPEN IS THE EXEMPTION, NOT THE FALL-THROUGH. Adding
 * `:not([data-state='foo'])` to the chain WITHOUT adding a `[data-state='foo']`
 * treatment rule leaves `foo` matching no rule at all -- a bare `.value`,
 * inheriting `--og-fg`, PIXEL-IDENTICAL TO MEASURED. The chain is an exemption
 * list, and every exemption must be PAID FOR by a treatment. Nothing checked
 * that, and it is one line of CSS away at any moment.
 *
 * THIS IS ALSO WHY THE ORDERED CSS INVERSION IS NOT WHAT LANDED. Making
 * `.value` untrusted-by-default and having `[data-state='measured']` assert
 * trust is the right principle, and it is already how this section behaves --
 * but as a COLOUR rule it is inert where it counts: `panels.css:41` sets
 * `.value__num { color: var(--og-fg) }` directly on the child, in the LATER
 * stylesheet, so the wrapper's colour never reaches the number. Worse, `.value`
 * is defined in `panels.css` at (0,1,0) and `[data-state='measured']` lives in
 * `shell.css` at (0,1,0) -- EQUAL specificity, and panels.css loads second, so
 * an untrusted base written beside `.value` would have beaten the measured
 * override on source order and muted EVERY MEASURED VALUE ON THE PAGE.
 *
 * So the trust inversion is carried by the chip, which declares its own
 * background AND foreground, inherits nothing, and spells a WORD. What was
 * missing was never the CSS. It was this file.
 * ═══════════════════════════════════════════════════════════════════════════
 */
describe('an unknown state cannot render as a measured one', () => {
  const strip = (s) => s.replace(/\/\*[\s\S]*?\*\//g, '');
  const shell = strip(css['shell.css']);

  const rules = [...shell.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({
    sel: m[1].trim().replace(/\s+/g, ' '),
    body: m[2],
  }));

  // A catch-all is any rule that exempts named states from a blanket treatment.
  const catchAlls = rules.filter((r) => r.sel.includes(":not([data-state="));
  const exemptedBy = (r) =>
    [...r.sel.matchAll(/:not\(\[data-state='([a-z-]+)'\]\)/g)].map((m) => m[1]);

  // A treatment rule names a state OUTSIDE any :not(). Strip the :not() groups
  // first, or every exemption would read as its own treatment and this guard
  // would certify the exact defect it exists to catch.
  const treated = new Set(
    rules
      .flatMap((r) => [...r.sel.replace(/:not\([^)]*\)/g, '').matchAll(/\[data-state='([a-z-]+)'\]/g)])
      .map((m) => m[1]),
  );

  const tf = read('./telemetry-field.js');
  const frozen = tf.slice(tf.indexOf('FIELD_STATES = Object.freeze({'));
  const enumValues = [...frozen.slice(0, frozen.indexOf('\n});')).matchAll(/^\s{2}[A-Z_]+:\s*'([a-z-]+)'/gm)].map(
    (m) => m[1],
  );

  it('found the catch-all rules at all (non-zero floor)', () => {
    // Without this arm, DELETING the catch-all makes every assertion below
    // pass over an empty set. A guard that goes green when its subject is
    // removed is worse than no guard: it reports safety caused by absence.
    assert.ok(
      catchAlls.length >= 2,
      `Only ${catchAlls.length} catch-all rule(s) found in shell.css; expected ` +
        `at least 2 (the colour/underline rule and the chip). If the scan ` +
        `matched nothing, an unknown state now renders as a bare .value -- ` +
        `which is to say, as a measured one.`,
    );
  });

  it('every catch-all still positively binds [data-state]', () => {
    // The identification predicate above keys on the :not() chain -- the
    // NEGATIVE half of the selector. Rename the POSITIVE attribute and every
    // exemption survives verbatim: all the arms in this suite still find their
    // catch-all, parse the right exemption set, match it against FIELD_STATES,
    // locate the chip, and pass -- over a rule that selects NOTHING, because
    // no element in the DOM carries the renamed attribute.
    //
    // This is not hypothetical. Measured: rewriting `[data-state]:not(...)` to
    // `[data-XXXX]:not(...)` in shell.css left all 45 tests in this file green
    // and the whole 740-test suite green, with the unknown-state treatment
    // completely dead. A guard satisfied by the artefact it exists to reject.
    for (const r of catchAlls) {
      const positive = r.sel.replace(/:not\([^)]*\)/g, '').trim();
      assert.ok(
        positive.includes('[data-state]'),
        `A catch-all exempts named states but no longer SELECTS on ` +
          `[data-state]:\n  ${r.sel}\n` +
          `Its positive half is \`${positive}\`. The :not() chain is intact, ` +
          `so every other arm here passes over a rule that matches no element. ` +
          `An unknown state falls through to a bare .value -- which is to say, ` +
          `it renders as a measured one.`,
      );
    }
  });

  it('parsed a non-empty exemption set and a non-empty enum (anti-vacuity)', () => {
    for (const r of catchAlls) {
      assert.ok(
        exemptedBy(r).length > 0,
        `A catch-all was found but zero exemptions parsed out of it:\n  ${r.sel}\n` +
          `The :not() matcher has drifted from the selector syntax, and every ` +
          `set comparison below is comparing empty sets to empty sets.`,
      );
    }
    assert.ok(
      enumValues.length >= 5,
      `Parsed only ${enumValues.length} FIELD_STATES value(s) from ` +
        `telemetry-field.js. The enum matcher has drifted; the bijection arm ` +
        `below is vacuous.`,
    );
  });

  it('pays for every exemption with a treatment rule', () => {
    // THE ARM THAT MATTERS. An exempted state with no treatment matches no
    // rule at all and renders as an unstyled .value -- indistinguishable from
    // measured, with nothing red anywhere.
    //
    // `measured` is exempt from THIS check by ruling, not by oversight:
    // tokens.css states there is deliberately NO `--og-measured-fg`, because
    // measured is the one state whose treatment is ADD NOTHING.
    const unpaid = [];
    for (const r of catchAlls) {
      for (const state of exemptedBy(r)) {
        if (state === 'measured') continue;
        if (!treated.has(state)) unpaid.push(state);
      }
    }
    assert.deepEqual(
      [...new Set(unpaid)],
      [],
      `State(s) exempted from the catch-all with no treatment rule of their ` +
        `own: ${[...new Set(unpaid)].join(', ')}. Such a state matches NOTHING ` +
        `-- it renders as a bare .value inheriting --og-fg, which is pixel- ` +
        `identical to measured. This is the only direction in which the ` +
        `catch-all fails OPEN, and it is one line of CSS away at any time.`,
    );
  });

  it('exempts the identical set on every channel', () => {
    // The exemption list is duplicated across the colour rule and the chip. If
    // they drift, a state collects one channel and not the other -- e.g. the
    // chip without the underline, or the underline with no word to explain it.
    // Two lists that must agree and nothing checking that they do is the defect
    // class that has bitten this branch all night.
    const sets = catchAlls.map((r) => [...new Set(exemptedBy(r))].sort().join(','));
    assert.equal(
      new Set(sets).size,
      1,
      `The catch-all channels exempt DIFFERENT state sets:\n  ` +
        sets.map((s, i) => `${i}: ${s}`).join('\n  ') +
        `\nA state exempted on one channel and not the other renders half- ` +
        `qualified: the reader gets a signal with no word, or a word with no ` +
        `signal.`,
    );
  });

  it('exempts exactly the enum -- no more, no less', () => {
    // Catches the sixth state landing on EITHER side: a new FIELD_STATES value
    // whose CSS was never written (it will scream NO STATE at a legitimate
    // field), and a stale exemption for a state the enum has dropped.
    const exempt = [...new Set(catchAlls.flatMap(exemptedBy))].sort();
    assert.deepEqual(
      exempt,
      [...enumValues].sort(),
      `The catch-all's exemption list and FIELD_STATES have diverged.\n  ` +
        `CSS exempts : ${exempt.join(', ')}\n  ` +
        `enum defines: ${[...enumValues].sort().join(', ')}\n` +
        `A state in the enum but not the CSS renders as NO STATE though it is ` +
        `perfectly legitimate; a state in the CSS but not the enum is a dead ` +
        `exemption that will silently adopt any future state given that name.`,
    );
  });

  it('distinguishes a garbage state from measured on a NON-COLOUR channel', () => {
    // Simulate the cascade for one value inside the enum and one outside it,
    // then require the difference to survive colour being switched off --
    // greyscale, a projector, colour-blindness -- and to survive
    // `panels.css:41`, which sets .value__num's colour directly on the child
    // and therefore eats any colour the wrapper tries to assert.
    // The :not() groups MUST come off before looking for a positive match. A
    // catch-all selector literally contains the substring `[data-state='X']`
    // for every state it EXEMPTS, so a naive includes() reports that 'measured'
    // matches the rule that exists to skip it -- and the two declaration sets
    // come back identical for the most reassuring possible reason. This guard
    // failed exactly that way on its first run.
    const matches = (r, state) => {
      if (!r.sel.includes('.value')) return false;
      const positive = r.sel.replace(/:not\([^)]*\)/g, '');
      if (positive.includes(`[data-state='${state}']`)) return true;
      return r.sel.includes(':not([data-state=') && !exemptedBy(r).includes(state);
    };

    const declsFor = (state) =>
      rules
        .filter((r) => matches(r, state))
        .flatMap((r) => [...r.body.matchAll(/([a-z-]+)\s*:/g)].map((m) => m[1]));

    const garbage = declsFor('__not_a_real_state__');
    const measured = declsFor('measured');

    assert.notDeepEqual(
      [...new Set(garbage)].sort(),
      [...new Set(measured)].sort(),
      `A state outside the enum resolves to the SAME declarations as ` +
        `'measured'. An unknown value is rendering with the page's full ` +
        `confidence.`,
    );
    assert.ok(
      garbage.includes('content'),
      `A garbage state gains no 'content' declaration, so it gets no chip. ` +
        `Colour cannot carry this on its own: panels.css sets .value__num's ` +
        `colour directly on the child, which beats inheritance from the ` +
        `wrapper, so the number -- the one glyph a reader actually looks at -- ` +
        `stays full brightness. The word is the channel.`,
    );
    assert.ok(
      !measured.includes('content'),
      `'measured' gains a 'content' declaration, meaning the chip is being ` +
        `drawn on correctly-measured values. That inverts the signal: the ` +
        `warning becomes the background condition and stops meaning anything.`,
    );
  });

  it('keeps the chip closed and worded', () => {
    const chip = catchAlls.find((r) => /content\s*:/.test(r.body));
    assert.ok(chip, 'No catch-all declares `content`; the chip is gone.');
    const word = chip.body.match(/content\s*:\s*'([^']*)'/);
    assert.ok(
      word && word[1].trim().length > 0,
      `The chip's content is empty or non-literal. It must spell a WORD: ` +
        `a word survives a projector, greyscale and colour-blindness with no ` +
        `encoding at all, and tells the reader WHAT is wrong rather than only ` +
        `that something is.`,
    );
    // The chip must inherit nothing. There are eighteen unconditional colour
    // rules on .value descendants; a chip that inherits is one new child rule
    // away from being unreadable, and nothing would go red.
    for (const prop of ['background', 'color']) {
      assert.ok(
        new RegExp(`(^|[;{\\s])${prop}\\s*:`).test(chip.body),
        `The chip does not declare its own \`${prop}\`. A treatment every ` +
          `future sibling rule must know about is a debt, not a design.`,
      );
    }
  });

  it('scopes the catch-all to .value, so it cannot paint other vocabularies', () => {
    // `.connection-indicator` carries a DIFFERENT vocabulary on the same
    // attribute (connected/connecting/no-model/unreachable). An unscoped
    // catch-all would stamp NO STATE on every one of them -- four false alarms
    // on the one widget whose job is to report trustworthiness.
    const unscoped = catchAlls.flatMap((r) =>
      r.sel
        .split(',')
        .map((s) => s.trim())
        .filter((s) => s.includes(':not([data-state=') && !s.startsWith('.value')),
    );
    assert.deepEqual(
      unscoped,
      [],
      `Unscoped catch-all selector(s): ${unscoped.join(' | ')}. These match ` +
        `every element carrying data-state, including .connection-indicator, ` +
        `whose four states are all legitimate and none of which are in ` +
        `FIELD_STATES.`,
    );
  });
});

/*
 * ═══════════════════════════════════════════════════════════════════════════
 * THE -rule FAMILY: EXTENDING THE COVERAGE LIST, AND REFUSING THE FLOOR
 *
 * Two coverage lists in this file enumerate `--og-{unavail,pending,stale,na}-fg`
 * and check all six pairs against a 1.05 separation floor. Neither has ever
 * named a `-rule` token. The order was to extend both lists to `-rule`.
 *
 * I MEASURED THE `-rule` FAMILY BEFORE EXTENDING ANYTHING, AND THE FLOOR IS THE
 * WRONG ASSERTION FOR IT:
 *
 *     -fg   family: 6 pairs, 0 below 1.05
 *     -rule family: 6 pairs, 3 BELOW 1.05
 *       unavail/stale 1.0149 · unavail/na 1.0258 · stale/na 1.0411
 *
 * Extending the lists verbatim would have shipped a RED guard, and a guard that
 * reds on correct work gets deleted by lunch. But the three pairs are NOT a
 * defect, and the reason is written in `shell.css` in the author's own words:
 * "the border grammar remains the entire signal." The rule COLOUR was never the
 * channel for these states -- the border STYLE is:
 *
 *     pending 1px solid · stale 1px dashed · unavailable 1px dotted
 *     not-applicable 3px double
 *
 * ☠️ SO THE REAL DEFECT IS NEITHER THE VALUES NOR THE MISSING FLOOR. IT IS THAT
 * NOTHING ANYWHERE RECORDED **WHICH CHANNEL CARRIES WHICH PAIR.** Three pairs
 * have been resting on border-style alone all night, in a list nobody wrote
 * down, and the composite tests cannot report it: a pair carried by exactly ONE
 * channel is one CSS edit from identical WITH NOTHING GOING RED.
 *
 * That is what these arms pin. Not a floor the palette was never designed to
 * meet -- the ACTUAL load-bearing channel, named per pair, so that removing it
 * fails.
 * ═══════════════════════════════════════════════════════════════════════════
 */
describe('every absence pair is separated on a channel that is written down', () => {
  const SEP = 1.05;
  const FAMILY = { unavailable: 'unavail', pending: 'pending', stale: 'stale', 'not-applicable': 'na' };
  const STATES = Object.keys(FAMILY);

  const relativeLuminance = (hex) =>
    [1, 3, 5]
      .map((i) => parseInt(hex.substr(i, 2), 16) / 255)
      .map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4))
      .reduce((sum, v, i) => sum + [0.2126, 0.7152, 0.0722][i] * v, 0);

  const ratio = (a, b) => {
    const [hi, lo] = [relativeLuminance(a), relativeLuminance(b)].sort((x, y) => y - x);
    return (hi + 0.05) / (lo + 0.05);
  };

  const hexOf = (token) => {
    const m = css['tokens.css'].match(new RegExp(`${token}:\\s*(#[0-9a-fA-F]{6})`));
    return m && m[1];
  };

  // The border declaration for a state, read from its OWN block. A windowed
  // grep bleeds into the next rule -- it has produced two false attributions in
  // this file's history -- so the block is bounded by its closing brace.
  const borderOf = (state) => {
    const src = css['shell.css'];
    const start = src.indexOf(`[data-state='${state}']`);
    if (start === -1) return null;
    const open = src.indexOf('{', start);
    const close = src.indexOf('}', open);
    if (open === -1 || close === -1) return null;
    const body = src.slice(open, close).replace(/\/\*[\s\S]*?\*\//g, '');
    const m = body.match(/border-bottom:\s*([0-9]+px)\s+([a-z]+)/);
    return m ? { width: m[1], style: m[2], decl: `${m[1]} ${m[2]}` } : null;
  };

  const pairs = [];
  for (let i = 0; i < STATES.length; i += 1) {
    for (let j = i + 1; j < STATES.length; j += 1) pairs.push([STATES[i], STATES[j]]);
  }

  it('computes a known ratio correctly (anti-vacuity)', () => {
    // An implementation returning a large number for everything would pass every
    // separation assertion below while measuring nothing.
    assert.ok(Math.abs(ratio('#ffffff', '#000000') - 21) < 0.01, 'ratio() is wrong');
    assert.ok(Math.abs(ratio('#151b23', '#151b23') - 1) < 0.01, 'ratio() is wrong');
  });

  it('resolves every token and every border declaration (non-zero floor)', () => {
    // Without this, DELETING a state's rule makes the comparisons below skip it
    // and the suite reports separation caused by absence.
    const missing = [];
    for (const [state, fam] of Object.entries(FAMILY)) {
      for (const kind of ['fg', 'rule']) {
        if (!hexOf(`--og-${fam}-${kind}`)) missing.push(`--og-${fam}-${kind}`);
      }
      if (!borderOf(state)) missing.push(`[data-state='${state}'] border-bottom`);
    }
    assert.deepEqual(missing, [], `Unresolvable, so unmeasurable:\n  ${missing.join('\n  ')}`);
    assert.equal(pairs.length, 6, 'expected all six pairs of a four-state family');
  });

  it('gives every absence state a distinct border style', () => {
    // THIS IS THE CHANNEL THAT ACTUALLY CARRIES ALL SIX PAIRS, and until now
    // nothing asserted it. Set-based, so it cannot be satisfied by three states
    // agreeing and one differing.
    const decls = STATES.map((s) => [s, borderOf(s).decl]);
    const seen = new Map();
    const collisions = [];
    for (const [state, decl] of decls) {
      if (seen.has(decl)) collisions.push(`${seen.get(decl)} and ${state} both render '${decl}'`);
      else seen.set(decl, state);
    }
    assert.deepEqual(
      collisions,
      [],
      `Two absence states share a border treatment:\n  ${collisions.join('\n  ')}\n` +
        `Border style is the ONLY channel separating three of these six pairs ` +
        `(their rule colours sit within 1.05), so a collision here renders them ` +
        `identical with nothing else to fall back on.`,
    );
  });

  it('requires a distinct border style wherever the rule COLOUR collapses', () => {
    // The load-bearing arm. For every pair whose rule colours are closer than
    // the floor, the border style MUST differ -- otherwise the pair is carried
    // by nothing at all on the rule channel.
    const unguarded = [];
    let collapsed = 0;
    for (const [a, b] of pairs) {
      const sep = ratio(hexOf(`--og-${FAMILY[a]}-rule`), hexOf(`--og-${FAMILY[b]}-rule`));
      if (sep >= SEP) continue;
      collapsed += 1;
      if (borderOf(a).decl === borderOf(b).decl) {
        unguarded.push(`${a} vs ${b}: rule colour ${sep.toFixed(4)}:1 AND identical border`);
      }
    }
    assert.ok(
      collapsed >= 1,
      `No rule-colour pair measured below ${SEP}. Either the palette changed or ` +
        `the token matcher has drifted -- three pairs were below it when this ` +
        `arm was written, so a zero here means the instrument stopped reading.`,
    );
    assert.deepEqual(
      unguarded,
      [],
      `Pair(s) separated by NOTHING on the rule channel:\n  ${unguarded.join('\n  ')}`,
    );
  });

  it('separates every pair on at least one named channel', () => {
    const orphans = [];
    for (const [a, b] of pairs) {
      const carriers = [];
      if (ratio(hexOf(`--og-${FAMILY[a]}-fg`), hexOf(`--og-${FAMILY[b]}-fg`)) >= SEP) carriers.push('fg');
      if (ratio(hexOf(`--og-${FAMILY[a]}-rule`), hexOf(`--og-${FAMILY[b]}-rule`)) >= SEP) carriers.push('rule');
      if (borderOf(a).decl !== borderOf(b).decl) carriers.push('border');
      if (carriers.length === 0) orphans.push(`${a} vs ${b}`);
    }
    assert.deepEqual(
      orphans,
      [],
      `Pair(s) indistinguishable on EVERY channel: ${orphans.join(', ')}. These ` +
        `states mean different things -- we have no number / we are waiting / we ` +
        `had one and it aged out / this cannot apply here -- and a visitor who ` +
        `cannot separate them reads four different admissions as one.`,
    );
  });
});

describe('every token actually used as text clears WCAG AA', () => {
  // WHY THIS IS DERIVED FROM USAGE AND NOT FROM A LIST OF TOKENS.
  //
  // tokens.css already checks every "N:1 on --og-bg-raised" annotation it
  // carries. That guard validates the claims that are PRESENT and says nothing
  // about the ones that are ABSENT -- so a token nobody annotated is a token
  // nobody checked. --og-bad sat at 4.48:1 (AA is 4.5) as the colour of
  // `.request-state--error` and `.connection--offline`: the text that says
  // something is broken was the text that failed to be legible. Found by
  // @c8d9a40e; confirmed here by a third independent implementation.
  //
  // Enumerating tokens would reproduce the defect one level up, because the
  // enumeration gets written by the same person who forgot the annotation. So
  // this scans for `color: var(--og-*)` and scores whatever it finds. A new
  // token used as text is covered the moment it is written, with no edit here.
  //
  // THE PAIRING, NOT THE TOKEN, IS THE UNIT OF A CONTRAST CLAIM. --og-accent is
  // 3.34:1 on --og-bg-raised and entirely correct, because it is only ever a
  // `background`. My own first version of this guard scored every rule against
  // the page background and red-flagged the `NO STATE` chip -- which declares
  // its own background two lines above its own colour. A guard that assumes
  // one background is a guard that cannot read the thing it is scoring.

  const lin = (c) => (c / 255 <= 0.04045 ? c / 255 / 12.92 : Math.pow((c / 255 + 0.055) / 1.055, 2.4));
  const lum = (hex) => {
    const h = hex.replace('#', '');
    return [0, 2, 4]
      .map((i) => lin(parseInt(h.slice(i, i + 2), 16)))
      .reduce((s, v, i) => s + [0.2126, 0.7152, 0.0722][i] * v, 0);
  };
  const contrast = (a, b) => {
    const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
    return (hi + 0.05) / (lo + 0.05);
  };
  const hexOf = (token) => {
    const m = css['tokens.css'].match(new RegExp(`--${token}:\\s*(#[0-9a-fA-F]{6})`));
    return m && m[1];
  };

  // Decorative text, exempted BY NAMED SELECTOR and never by token. Exempting a
  // token would silently exempt every future use of it, including informative
  // ones; exempting a selector exempts exactly the element whose reason is
  // written down here.
  const DECORATIVE = {
    '.value__sep':
      'a separator glyph present so a value and its age never fuse in textContent; it carries no information of its own',
    '__marker':
      'sequence markers encode identity as glyph SHAPE plus fill PATTERN plus colour (tokens.css:208-213); the colour is the third of three encodings and carries nothing alone',
  };

  // Comments carry prose that looks like selectors and declarations. Strip them
  // before parsing or the "selector" of a rule becomes its own documentation.
  const stripped = Object.fromEntries(
    Object.entries(css).map(([f, t]) => [f, t.replace(/\/\*[\s\S]*?\*\//g, '')]),
  );

  const rules = Object.entries(stripped).flatMap(([file, text]) =>
    [...text.matchAll(/([^{}]+)\{([^}]*)\}/g)].map((m) => ({
      file,
      sel: m[1].trim().replace(/\s+/g, ' '),
      decl: m[2],
    })),
  );

  // If a rule paints its own background, THAT is what its text sits on.
  const bgOf = (decl) => {
    const m = decl.match(/background(?:-color)?:\s*var\(--(og-[a-z0-9-]+)\)/);
    return m ? hexOf(m[1]) : null;
  };

  const textRules = rules.flatMap((r) =>
    [...r.decl.matchAll(/(?:^|[;\s])color:\s*var\(--(og-[a-z0-9-]+)\)/g)].map((d) => ({
      ...r,
      token: d[1],
      against: bgOf(r.decl) || hexOf('og-bg-raised'),
    })),
  );

  it('found text rules to score (anti-vacuity)', () => {
    // A regex drift that matched nothing would make every assertion below pass
    // over an empty set -- green caused by absence, which this file exists to
    // refuse.
    assert.ok(
      textRules.length >= 15,
      `Only ${textRules.length} \`color: var(--og-*)\` rule(s) found across ` +
        `${Object.keys(css).length} stylesheet(s). The scanner has drifted from ` +
        `the CSS syntax and is scoring nothing.`,
    );
  });

  it('resolves its background and known ratios (control)', () => {
    assert.ok(hexOf('og-bg-raised'), 'background token did not resolve; every ratio is meaningless.');
    assert.equal(contrast('#ffffff', '#000000').toFixed(2), '21.00', 'white/black must be 21:1');
    assert.equal(contrast('#151b23', '#151b23').toFixed(2), '1.00', 'identical colours must be 1:1');
  });

  it('every token used as `color:` clears 4.5:1 against what it sits on', () => {
    const failures = [];
    for (const r of textRules) {
      if (Object.keys(DECORATIVE).some((k) => r.sel.includes(k))) continue;
      const hex = hexOf(r.token);
      if (!hex || !r.against) continue;
      const c = contrast(hex, r.against);
      if (c < 4.5) {
        failures.push(`${r.file}  ${r.sel.slice(0, 70)}  --${r.token} ${hex} on ${r.against} = ${c.toFixed(2)}:1`);
      }
    }
    assert.deepEqual(
      failures,
      [],
      `Text below WCAG AA 1.4.3 (4.5:1):\n  ${failures.join('\n  ')}\n` +
        `Either raise the token, or -- if the text is genuinely decorative -- add ` +
        `its SELECTOR to DECORATIVE above WITH THE REASON. Do not exempt the token.`,
    );
  });

  it('every decorative exemption still matches something, and carries a reason', () => {
    // An exemption that outlives its selector is a permanent hole nobody can
    // see. This is D337's complement arm, one file over: assert that the thing
    // you are exempting still EXISTS, or the exemption silently covers nothing.
    for (const [key, reason] of Object.entries(DECORATIVE)) {
      assert.ok(reason.length > 40, `Exemption \`${key}\` has no substantive reason.`);
      assert.ok(
        textRules.some((r) => r.sel.includes(key)),
        `Exemption for \`${key}\` matches no \`color: var(--og-*)\` rule any more. ` +
          `Either it was renamed -- so this exemption now covers nothing and must be ` +
          `re-pointed -- or the rule is gone and this entry should be deleted.`,
      );
    }
  });
});

// ---------------------------------------------------------------------------
// D345 -- A STYLESHEET MUST BE STRUCTURALLY WELL-FORMED, AND UNTIL NOW NOTHING
// HERE CHECKED THAT.
//
// This arm exists because of a defect I shipped into `shell.css` and caught by
// hand rather than by instrument. An edit left a duplicated paragraph sitting
// OUTSIDE the comment that used to contain it: the file had 41 `*/` against 40
// `/*`, and several lines of English prose were, as far as any CSS parser is
// concerned, garbage in the middle of the stylesheet.
//
// I then ran every guard I own -- 68 tests across two suites -- and got
// `pass 68 · fail 0 · exit 0`. I re-confirmed it deliberately afterwards by
// appending known-orphan text to a scratch copy and running them again: still
// 68/68, still exit 0.
//
// ⛔ THE REASON IS THE ONE THIS CREW FOUND IN SIX OTHER PLACES TONIGHT, AND IT
// IS WORTH STATING PLAINLY BECAUSE IT IS A PROPERTY OF EVERY CHECK IN THIS
// FILE: MY GUARDS ALL EXTRACT WHAT THEY WANT WITH A REGEX. A REGEX LOOKING FOR
// `color: var(--og-*)` FINDS EVERY SUCH DECLARATION WHETHER OR NOT THE FILE
// AROUND IT PARSES. Tolerating malformed input is exactly the property that
// makes a scanner robust, and it is therefore the property that makes it blind
// to the file being malformed. Robust to the thing is blind to the thing when
// the thing is the defect.
//
// So this does not check any declaration. It checks that the CONTAINER of every
// declaration is intact, which is the one question the other 68 cannot ask.
describe('every stylesheet is structurally well-formed', () => {
  // CSS comments do not nest. Strip them with an explicit scanner rather than a
  // regex, because the scanner can REPORT AN UNTERMINATED COMMENT, and an
  // unterminated comment is the failure that silently swallows the rest of a
  // file -- every rule after it stops applying, and the page keeps rendering.
  const stripComments = (text) => {
    let out = '';
    let i = 0;
    while (i < text.length) {
      const open = text.indexOf('/*', i);
      if (open === -1) {
        out += text.slice(i);
        break;
      }
      out += text.slice(i, open);
      const close = text.indexOf('*/', open + 2);
      if (close === -1) return { out, unterminated: true };
      // Preserve newlines so any line number we report still means something.
      out += text.slice(open, close + 2).replace(/[^\n]/g, '');
      i = close + 2;
    }
    return { out, unterminated: false };
  };

  const defects = (text) => {
    const { out, unterminated } = stripComments(text);
    if (unterminated) return ['an unterminated /* comment -- everything after it is dead'];
    const found = [];
    out.split('\n').forEach((line, n) => {
      // A stray `*/` outside any comment. This is what an edit produces when it
      // adds a closer the original text already had.
      if (line.includes('*/')) found.push(`${n + 1}: a \`*/\` that closes nothing`);
      // A comment-continuation line that escaped its comment. Prose in the
      // declaration stream. This is the exact shape of the defect above.
      //
      // ⛔ THE DISCRIMINATOR IS NARROWER THAN IT LOOKS, AND ITS FIRST VERSION
      // WAS WRONG. `*` CARRIES TWO UNRELATED MEANINGS IN CSS: the comment
      // continuation in ` * prose`, and the UNIVERSAL SELECTOR in `*,`,
      // `*::before,` and `*::after {` -- all three of which are real, correct,
      // and present at the top of this very file. A bare /^\s*\*/ flagged all
      // three on its first run. What separates them is that a continuation has
      // WHITESPACE between the asterisk and its text, while a selector binds
      // the asterisk tight to `,` `:` or `{`; and prose never carries `{`, a
      // trailing `,` or a trailing `;`. A name that carries two meanings makes
      // a naive guard impossible to write -- so the guard must key on the SHAPE
      // the two meanings do not share, not on the character they do.
      else if (/^\s*\*\s+\S/.test(line) && !line.includes('{') && !/[,;]\s*$/.test(line)) {
        found.push(`${n + 1}: \`${line.trim().slice(0, 48)}\` is prose outside a comment`);
      }
    });
    return found;
  };

  it('has no prose, stray comment closer, or unterminated comment in the declaration stream', () => {
    for (const file of styleFiles) {
      assert.deepEqual(
        defects(css[file]),
        [],
        `\`styles/${file}\` is not well-formed CSS. A parser stops applying rules at ` +
          `the first structural error and the page keeps rendering, so this fails ` +
          `SILENTLY on screen -- which is why it needs a test rather than a look.`,
      );
    }
  });

  // ANTI-VACUITY. Without this the suite above passes just as happily on an
  // empty file list or a detector that can only return []. Each arm below is a
  // DIFFERENT structural defect, because a detector that catches one shape and
  // reports it as all three would pass a single-case check.
  it('can actually detect each defect, so a clean run means something', () => {
    const orphan = 'a { color: red }\n * prose that escaped its comment\n';
    const stray = 'a { color: red }\n */\n';
    const open = '/* this comment never closes\na { color: red }\n';

    assert.equal(defects(orphan).length, 1, 'the detector cannot see prose outside a comment');
    assert.equal(defects(stray).length, 1, 'the detector cannot see a stray `*/`');
    assert.equal(defects(open).length, 1, 'the detector cannot see an unterminated comment');
    assert.match(defects(open)[0], /unterminated/);

    // And the negative control: a correct stylesheet, including a comment whose
    // body contains lines that START with `*`, must produce nothing. Without
    // this arm the detector could pass the three above by flagging everything.
    const good = '/*\n * a normal banner comment\n * second line\n */\na { color: red }\n';
    assert.deepEqual(defects(good), [], 'the detector fires on correct CSS');

    // ⛔ AND THE CONTROL THAT THIS GUARD ACTUALLY FAILED ON ITS FIRST RUN, kept
    // as a permanent arm because it is the case a future simplification of the
    // regex would silently reintroduce: the CSS UNIVERSAL SELECTOR. All three
    // spellings below are live at the top of `shell.css` and `tokens.css`, and
    // a bare /^\s*\*/ reports every one of them as prose.
    const universal = '*,\n*::before,\n*::after {\n  box-sizing: border-box;\n}\n';
    assert.deepEqual(
      defects(universal),
      [],
      'the detector mistakes the universal selector for comment prose -- the two ' +
        'share the `*` character and nothing else, so key on the shape, not the char',
    );
  });
});
