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
import { createHash } from 'node:crypto';
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

// ---------------------------------------------------------------------------
// THE DESIGN DOCUMENT IS A BUILD INSTRUCTION, AND NOTHING HAS EVER READ IT.
//
// Three cut-or-qualified field bindings were found in ONE hour tonight, all in
// PROSE: the <meta> description above, the Profile D hero-strip slot, and a
// benchmark sketch showing 2.46x with no per-stream figure. Every one of them
// was invisible to all five test files in this directory, because:
//   - state-channel / prefix-counters read MODULE IDENTIFIERS
//   - page-claims (above) reads SHIPPED HTML
//   - nothing at all reads demo-ux.md -- THE FILE DEVELOPERS BUILD FROM.
//
// A design document that names a cut field is worse than shipped code that
// does, because code gets deleted once and prose gets BUILT FROM REPEATEDLY.
// The Profile D slot proved it: `prefix hit rate` survived the cut by 40
// minutes inside a table headed "to be used verbatim".
//
// SCOPE: FENCED CODE BLOCKS ONLY, and that restriction is the whole design.
// demo-ux.md must keep discussing prefix caching at length -- the re-scoped
// Scenario B SHIPS the null result, and D155 argues it is the most credible
// artifact we have. Banning the words would forbid the honest treatment along
// with the dishonest one. But a FENCED BLOCK is not discussion: it is a
// layout a developer copies. Prose explains; a sketch instructs. Same
// positional rule as D149 and as the claim-position rule above.
const UX_DOC = fileURLToPath(new URL('./design/demo-ux.md', import.meta.url));
const uxDoc = readFileSync(UX_DOC, 'utf8');

const sketches = [...uxDoc.matchAll(/```[^\n]*\n([\s\S]*?)```/g)].map((m) => ({
  body: m[1],
  line: uxDoc.slice(0, m.index).split('\n').length,
  hash: createHash('sha1')
    .update(m[1].replace(/\s+/g, ' ').trim())
    .digest('hex')
    .slice(0, 12),
}));

describe('the design document does not instruct a build it has ruled against', () => {
  it('has sketches to check at all', () => {
    // Guards the guard. If the fence syntax or the filename ever changes, a
    // zero-sketch scan would report PASS while checking nothing -- the exact
    // "green because it looked at nothing" failure this suite exists to stop.
    assert.ok(
      sketches.length > 20,
      `Expected demo-ux.md to contain many fenced sketches; found ${sketches.length}. ` +
        'If the document moved or the fences changed, this scanner is silently ' +
        'inspecting an empty set and every assertion below is vacuously true.',
    );
  });

  it('binds no cut field in an unreviewed fenced sketch', () => {
    // EXEMPTIONS ARE DECLARED, NEVER INFERRED, AND THE LIST ONLY SHRINKS.
    //
    // I first tried to infer this: exempt a sketch if a supersession marker
    // appears within 15 lines above it. It "worked" -- and it was the wrong
    // mechanism, because a heuristic exemption means NOBODY DECIDED. A sketch
    // would go quiet because of its neighbours rather than because a person
    // judged it, which is the same defect as a stale doc comment: authority
    // with no author.
    //
    // Keyed on a hash of the sketch BODY, not on a line number, for two
    // reasons: line numbers churn on every edit to a 3500-line document, and
    // more importantly EDITING AN EXEMPTED SKETCH REVOKES ITS EXEMPTION. You
    // cannot quietly add a field to a grandfathered layout.
    const exempt = new Map([
      ['8d37e2cc3656', 'file tree: names scenarios/prefix.js as a path, binds no field'],
      ['970cf30e5349', "honesty footer sample: the generated WHAT'S REAL / DERIVED table"],
      ['eb830f5dee14', 'PLACEHOLDER treatment: defines the confession copy, renders no value'],
      ['652f178b3ebf', 'KV memory panel: ships; names eviction only in a not-plumbed row'],
      ['b9cc9dfbf776', 'paged KV block table: ships, verified (allocated 3, freed 3, 14612 pages)'],
      ['23e9cd2205fb', 'the not-applicable card itself -- the honest treatment, D30'],
      ['041f176dc271', 'prose explanation of WHY the scatter engine never consults the trie'],
      ['05fde69af3e7', 'lifecycle states; preemption named only to record that it is absent'],
      ['c1df66654c38', 'not-applicable card with file:line evidence -- the honest treatment'],
      ['d5ee792cbc61', 'source citation: pipeline/paged_decode.rs call sites'],
      ['24674e7e5804', 'source citation: prefix_cache.rs:151 evict_lru, evidence not layout'],
      ['07de135dbe16', 'source citation: page_table.rs:1068 evict_lru, evidence not layout'],
      ['2be978152568', 'code citation showing the WRONG pattern being diagnosed'],
      ['e1868d09d532', 'STRUCK S8.3: the fabricated 48x TTFT ladder, retained as evidence'],
      ['dade37765504', 'the null-result panel (S51) -- ships, and must name what it tested'],
      // S58: verbatim quote of scenario-origins.js:56-61, the DEFECT ITSELF.
      // This is the same legitimate use the sibling tripwire grants
      // telemetry-provenance.js -- 'the register that forbids them'. A block
      // that EXHIBITS a cut binding as evidence and a block that INSTRUCTS one
      // are textually identical, and no scanner can separate them, which is
      // precisely why the exemption must be WRITTEN rather than inferred (D166).
      // The hash is the safeguard: edit the quote and the exemption dies.
      ['30ffe828eee9', 'S58: verbatim quote of the defect being reported, not a layout'],
    ]);

    const offenders = [];
    const seen = new Set();
    for (const sketch of sketches) {
      const bound = CUT_FEATURES.filter((f) => f.pattern.test(sketch.body));
      if (bound.length === 0) continue;
      if (exempt.has(sketch.hash)) {
        seen.add(sketch.hash);
        continue;
      }
      offenders.push(
        `demo-ux.md:~${sketch.line} hash ${sketch.hash} binds ` +
          bound.map((f) => `"${f.name}"`).join(', '),
      );
    }

    assert.deepEqual(
      offenders,
      [],
      'A fenced block in the design document binds a CUT field and is not on ' +
        'the reviewed exemption list. Prose may discuss a cut feature at ' +
        'length -- the re-scoped Scenario B SHIPS the null result and D155 ' +
        'argues it is the most credible thing we have -- but a SKETCH is ' +
        'copied rather than read, so it is a build instruction. Two sketches ' +
        'in this document carried invented values under the `s` ' +
        'SERVER-MEASURED badge (a 87.2% hit rate, and a 48x TTFT collapse) ' +
        'drawn BEFORE the feature was measured, and the real measurement went ' +
        'the other way. If this is honest, add the hash with a reason. If it ' +
        `is a layout, strike it.\nFOUND:\n  ${offenders.join('\n  ')}`,
    );

    // The list only shrinks. A stale exemption is a permission that outlived
    // the thing it permitted -- the same argument that deleted the CORS layer.
    const stale = [...exempt.keys()].filter((h) => !seen.has(h));
    assert.deepEqual(
      stale,
      [],
      'Exemption(s) match no sketch that binds a cut field. Either the sketch ' +
        'was fixed (delete the entry -- this list is a ratchet) or it was ' +
        `edited, which REVOKES the exemption by design.\nSTALE: ${stale.join(', ')}`,
    );
  });

  it('never sketches the aggregate speedup without its per-stream cost', () => {
    // AC50 / D85. The hero is a TRADEOFF, not a number: aggregate decode is
    // 2.46x at 4 concurrent, but per-stream throughput falls to ~0.62x.
    // Batching makes no single request faster. 2.46x alone is technically
    // accurate and substantively misleading -- a tradeoff presented as a pure
    // win is a lie told with true numbers, which is the one failure class this
    // project cannot ship, since every value in it would pass a provenance
    // check individually.
    const offenders = sketches
      .filter((s) => /2\.46/.test(s.body) && !/0\.62/.test(s.body))
      .map((s) => `demo-ux.md:${s.line}`);
    assert.deepEqual(
      offenders,
      [],
      'A sketch shows the 2.46x aggregate speedup with no per-stream figure ' +
        'beside it. Per D85 the two render at IDENTICAL type size, because ' +
        'typographic hierarchy is itself a claim about which number matters -- ' +
        'a big number with a small caveat is still the lie, told more quietly. ' +
        `FOUND: ${offenders.join(', ')}`,
    );
  });
});

// ---------------------------------------------------------------------------
// A NAVIGABLE SCENARIO IS THE STRONGEST CLAIM THE PAGE MAKES.
//
// Per D139: panels display VALUES and may be grouped or collapsed; TABS
// ADVERTISE CAPABILITIES. A scenario in the switcher is a labelled, clickable
// promise that the product does this thing -- made before the visitor has seen
// a single number, in the one control whose entire purpose is to enumerate what
// is on offer.
//
// So a cut feature surviving in the scenario registry is worse than a cut field
// surviving in a panel. The panel says "here is a number about X"; the tab says
// "X IS ONE OF THE THINGS THIS RUNTIME DOES." The first is a wrong reading, the
// second is a wrong product.
//
// AND THE TRIPWIRE NEXT DOOR CANNOT SEE THIS, BY CONSTRUCTION.
// prefix-counters-forbidden.test.js bans the IDENTIFIERS -- prefix_cache_hits,
// prefix_cache_lookups, hit_rate, prefix_hashes. A scenario registers the
// string 'prefix-cache' with the human label 'Prefix caching'. Neither spelling
// is an identifier, so the ban misses both, and the allowlist that tracks the
// panel's removal debt never mentions the file. This is the same blind spot
// that left "prefix caching" in the <meta> description and in the Profile D
// hero strip: OUR ENFORCEMENT COVERS CODE AND FIELDS, NOT USER-FACING PROSE.
//
// MUTATIONS THIS TEST IS KNOWN TO FAIL ON:
//   1. re-add a 'prefix-cache' entry to SCENARIOS   -> cut scenario navigable
//   2. relabel a scenario 'KV eviction'             -> cut feature in a label
import { readFileSync as readScenarios } from 'node:fs';

describe('the scenario switcher advertises no cut capability', () => {
  const source = readScenarios(
    fileURLToPath(new URL('./scenario-origins.js', import.meta.url)),
    'utf8',
  );

  it('registers scenarios at all', () => {
    const ids = [...source.matchAll(/^\s{2}'([\w-]+)':\s*Object\.freeze/gm)].map((m) => m[1]);
    assert.ok(
      ids.length >= 2,
      `Expected the scenario registry to define several scenarios; found ${ids.length}. ` +
        'If the shape changed, the assertion below is vacuously true.',
    );
  });

  it('registers no scenario id or label naming a cut feature', () => {
    const ids = [...source.matchAll(/id:\s*'([\w-]+)'/g)].map((m) => m[1]);
    const labels = [...source.matchAll(/label:\s*'([^']+)'/g)].map((m) => m[1]);
    const offenders = [];
    for (const feature of CUT_FEATURES) {
      for (const id of ids) {
        if (feature.pattern.test(id.replace(/-/g, ' '))) offenders.push(`id '${id}'`);
      }
      for (const label of labels) {
        if (feature.pattern.test(label)) offenders.push(`label '${label}'`);
      }
    }
    assert.deepEqual(
      offenders,
      [],
      'The scenario registry still advertises a CUT capability. A tab is not a ' +
        'panel: per D139 panels display values and may be grouped, but TABS ' +
        'ADVERTISE CAPABILITIES -- a labelled, clickable promise that the ' +
        'product does this, made before the visitor sees a single number. ' +
        'Scenario B was cut in every form on every origin, so a reachable ' +
        '?scenario= URL for it is a navigable route to a feature we proved ' +
        'absent. The identifier tripwire cannot catch this: it bans ' +
        "prefix_cache_hits and friends, not the string 'prefix-cache' or the " +
        `label 'Prefix caching'.\nFOUND: ${offenders.join(', ')}`,
    );
  });
});
