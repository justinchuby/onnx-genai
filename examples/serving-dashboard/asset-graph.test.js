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
