// THE PAGE'S CLAIMS ABOUT ITSELF MUST BE TRUE, AND NOTHING ELSE CHECKS THEM.
//
// Every honesty mechanism in this project operates on FIELDS. The provenance
// envelope, the five-state vocabulary, the classification registry, the
// generated "what's real" footer, the prefix-counter tripwire -- all of them
// inspect values flowing from a server into a panel. NONE of them can see
// `<title>` or `<meta name="description">`.
//
// That gap matters more than its size suggests, because those two strings are:
//   1. the FIRST claim anyone reads, above every affordance we built;
//   2. rendered where our machinery cannot reach -- search results, Slack and
//      GitHub link previews, browser tab tooltips. A visitor can read a false
//      claim here WITHOUT EVER LOADING the page that would have corrected it.
//      No envelope, no `n/a`, no honesty footer, no data-state;
//   3. invisible to every other check, including the prefix tripwire, because
//      they do not name a counter. They name the FEATURE, in plain English --
//      which is exactly what makes a claim about a cut feature a lie.
//
// This file was written after `index.html` was found advertising "prefix
// caching" in its meta description, ~50 minutes after prefix caching was cut
// from the demo as PROVEN ABSENT (n=20 control: shared-prefix requests no
// faster than requests sharing nothing, against a sensitivity floor where a
// working cache would have collapsed TTFT by ~90%).
//
// WHY THIS IS NARROW ON PURPOSE, and the rule it applies:
//   Cut features MUST still be discussed in prose. The re-scoped Scenario B
//   SHIPS the prefix-cache null result with its control arm, and demo-ux.md
//   D155 argues that panel is the most credible thing on the page. So banning
//   the words outright would forbid the honest treatment along with the
//   dishonest one.
//   The rule is positional, exactly like D149's "batch size": a cut feature is
//   BANNED IN A CLAIM POSITION and PERMITTED IN AN EXPLANATION. `<title>` and
//   `<meta name="description">` are pure claim positions -- they are single
//   sentences with no room for a control arm, a detection floor, or an
//   em-dash. There is no honest way to say "prefix caching" in nineteen words
//   of marketing copy.
//
// MUTATIONS THIS TEST IS KNOWN TO FAIL ON (run them, don't trust this comment):
//   1. restore "and prefix caching" to the meta description  -> claim position
//   2. add "prefix cache" to <title>                         -> claim position
//   3. write "demonstrates preemption" in the description    -> cut subsystem
//   4. empty/absent <meta name="description">                -> cannot silently
//      pass by deleting the string it checks
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

const INDEX = fileURLToPath(new URL('./index.html', import.meta.url));
const html = readFileSync(INDEX, 'utf8');

// Features verified ABSENT in this runtime and cut from the demo. Each entry
// names the evidence, because a ban with no citation becomes cargo cult the
// moment its author leaves -- and the next contributor deletes it.
const CUT_FEATURES = Object.freeze([
  {
    pattern: /prefix[\s-]*cach/i,
    name: 'prefix caching',
    evidence:
      'QA n=20 control: shared-prefix requests were not faster than requests ' +
      'sharing nothing from token 0, against a sensitivity floor where a ' +
      'working cache collapses TTFT ~1380ms -> ~140ms. Reuse is PROVEN ABSENT, ' +
      'not merely unobserved. Scenario B was re-scoped to paged-KV allocation ' +
      'plus the published null result.',
  },
  {
    pattern: /preempt/i,
    name: 'preemption',
    evidence:
      'ContinuousBatchManager has no Scheduler field at all -- preemption is ' +
      'not disabled, the component is absent. batched.rs:757 hardcodes ' +
      'PreemptionPolicy::Disabled on the other path.',
  },
  {
    pattern: /evict/i,
    name: 'KV eviction',
    evidence:
      'ByteBudget::reconfigure changes the ceiling and never touches `used`; ' +
      "the repo's own test is named reconfigure_lower_reports_overage_without" +
      '_evicting. The governor computes eviction_order and nothing consumes it.',
  },
]);

function extract(label, regex) {
  const match = html.match(regex);
  assert.ok(
    match && match[1] && match[1].trim().length > 0,
    `${label} is missing or empty in index.html. This test cannot be satisfied ` +
      'by deleting the string it checks: the page must make a claim about ' +
      'itself, and that claim must be true. A page with no description is not ' +
      'honest, it is unlabelled -- and link previews will fall back to scraping ' +
      'arbitrary body text, which is strictly less controlled than writing it.',
  );
  return match[1];
}

describe("the page's claims about itself", () => {
  const positions = [
    ['<title>', /<title>([\s\S]*?)<\/title>/],
    [
      '<meta name="description">',
      /<meta\s+name="description"\s+content="([\s\S]*?)"/,
    ],
  ];

  for (const [label, regex] of positions) {
    it(`${label} names no feature this runtime does not have`, () => {
      const text = extract(label, regex);

      for (const { pattern, name, evidence } of CUT_FEATURES) {
        assert.equal(
          pattern.test(text),
          false,
          `${label} claims "${name}", which this runtime does not do.\n\n` +
            `EVIDENCE: ${evidence}\n\n` +
            `FOUND: ${text.trim()}\n\n` +
            'This is a CLAIM POSITION -- a single sentence with no room for a ' +
            'control arm, a detection floor or an em-dash, rendered in search ' +
            'results and link previews where none of our honesty machinery ' +
            'reaches. A visitor can read it without ever loading the page that ' +
            'would correct it.\n' +
            'Discussing a cut feature in PROSE is required, not banned: the ' +
            'null-result panel ships the prefix-cache experiment with its ' +
            'control, and demo-ux.md D155 argues it is the most credible thing ' +
            'on the page. Remove the claim; keep the explanation.',
        );
      }
    });
  }

  it('describes the demo using only capabilities that were verified', () => {
    const text = extract(
      '<meta name="description">',
      /<meta\s+name="description"\s+content="([\s\S]*?)"/,
    );

    // Both survivors are independently verified: continuous batching is the
    // 2.46x baseline, and paged-KV page allocation was confirmed by direct
    // observation (allocated 3, freed 3, 14612 pages).
    const verified = [/continuous batch/i, /paged.?kv|kv block|block allocation/i];
    const claimed = verified.filter((p) => p.test(text));

    assert.ok(
      claimed.length > 0,
      'The description names none of the demo\'s verified capabilities. ' +
        'Stripping a false claim must not leave the page silent about what it ' +
        'genuinely does -- honesty that removes the true claims along with the ' +
        'false ones has overcorrected, and undersells work that was measured. ' +
        `FOUND: ${text.trim()}`,
    );
  });
});
