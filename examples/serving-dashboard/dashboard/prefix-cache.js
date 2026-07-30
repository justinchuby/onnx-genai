// Copyright (c) Microsoft Corporation.
//
// Panel — Prefix cache. demo-ux.md §51, D150–D155, ratified at D279.
//
// 🔴 THIS PANEL DELIBERATELY BINDS NO TELEMETRY. THAT IS THE POINT OF IT.
//
// It is the only panel in the registry that renders a FINDING rather than a
// value, and it exists because deleting it is the one option that is definitely
// dishonest: a visitor who sees no prefix panel concludes prefix caching works,
// since nothing on the page says otherwise. Silence is a claim.
//
// ── WHY NO NUMBER APPEARS HERE ──────────────────────────────────────────────
//
// An earlier version of this panel argued from a timing A/B (a shared-prefix
// arm against a zero-sharing control). THAT TIMING RESULT WAS WITHDRAWN by the
// agent who produced it: a re-run with interleaved arms moved in the opposite
// direction, and both deltas sat inside the noise band of an n=6 sample. No
// prefix timing number ships anywhere in this tree, and reintroducing one here
// is a guarded offence — see `check-perf-claims.test.js`.
//
// THE CONCLUSION DID NOT MOVE, BECAUSE IT NEVER RESTED ON THE TIMING. It rests
// on two legs that carry no stopwatch at all, and a conclusion resting on two
// independent legs does not fall when one is withdrawn:
//
//   1. COUNTER BEHAVIOUR. The hit counter reported ~95% on every control
//      request — requests built to share nothing. A counter that reports reuse
//      where reuse is impossible is not measuring reuse. This is arithmetic
//      about the counter, and no re-run on a quieter machine can overturn it.
//
//   2. THE SOURCE ITSELF. Both execution paths are readable, and neither one
//      can reuse KV state, for two different reasons (below). This is the
//      strongest kind of absence evidence: not "we did not observe it" but
//      "the code that would perform it is not there".
//
// ── THE TWO PATHS FAIL DIFFERENTLY, AND THAT PAIRING IS THE LESSON ──────────
//
//   • CONTINUOUS-BATCH PATH — never consults the trie at all. The call sites
//     pass a hardcoded literal 0 where a matched-prefix length belongs, so no
//     lookup is even attempted.
//   • PAGED PATH — consults the cache and REPORTS a hit, but the reporting
//     branch loads no KV state and never sets `loaded_prompt_prefix`, so the
//     very next statement queues the full prompt and prefill recomputes every
//     token. The number it returns has no compute saving behind it.
//
// Thirty lines below that branch, the connector path states the correct rule
// for itself — "never claiming a hit we can't serve". The reporting branch
// violates the rule its own neighbour states.
//
// This panel therefore renders IDENTICALLY ON BOTH ORIGINS. An origin-dependent
// render would preserve a distinction that no longer exists: the gap is real on
// both paths, and hiding the panel on either would conceal half the finding.
// `prefix-cache.test.js` pins that byte-for-byte.
//
// ── WHY THE COUNTER MUST NOT BE SHOWN, EVEN AS A CURIOSITY ──────────────────
//
// It increments on ANY nonzero token match, and every chat-completions request
// shares the chat-template preamble — so it reads ~95% from the first request
// and never moves. It is not a stub, and not a misnamed-but-real number. It is
// a precisely-computed, beautifully-behaved, entirely FALSE value.
//
// Every other safeguard in this tree hunts fabricated ZEROS, because a zero
// looks broken and invites scrutiny. A confident 95% invites none. That is
// exactly why this needed a ruling rather than a guard.
//
// This is a FUNCTIONAL gap in the engine, not a reporting gap. No telemetry
// work will incidentally fix it, and there is no client-side substitute.

import {
  element,
  observeVisibility,
  replaceChildren,
  sectionLabel,
} from './panel-kit.js';

// NOTE: no `requires` key. Panel-level server-mode gating was deleted wholesale
// (every panel now renders on every profile), and `honesty.test.js` fails any
// panel that declares one. The both-paths reasoning that `requires: null` used
// to carry in a comment is preserved in the header above, where it is readable
// rather than encoded in a dead key.
export const meta = Object.freeze({
  id: 'prefix-cache',
  title: 'Prefix cache',
  group: 'cache',
  span: 1,
  // Static content. Nothing here polls, so cadence describes a panel that never
  // repaints from telemetry — it is present because the finding is.
  cadence: 0,
  staleCeilingMs: null,
  defaultOpen: true,
  acronyms: {
    prefix: 'A leading run of tokens shared by several prompts, whose KV state can be reused',
    prefill: 'The forward pass that computes KV state for the prompt before decoding begins',
    KV: 'Key/value attention state — the thing a prefix cache would reuse instead of recomputing',
  },
});

const HEADLINE =
  'Prefix reuse is not happening on either execution path. This panel reports that ' +
  'finding instead of a hit rate, because the hit counter is measurably false.';

const EXPLANATION =
  'Both execution paths are readable in the engine, and neither one can reuse attention ' +
  'state. The continuous batching path passes a hardcoded zero where a matched-prefix length ' +
  'belongs, so it never consults the cache at all. The paged path does consult it, but the ' +
  'branch that reports a hit loads no state and leaves the prompt marked unloaded, so the ' +
  'next statement queues the full prompt and prefill recomputes every token.';

const COUNTER_WARNING =
  'The engine\u2019s own hit counter reported about 95% throughout, including on every ' +
  'control request \u2014 requests built to share no prefix at all. It rises on any single ' +
  'matching token, and every chat request shares the template preamble, so it reads ~95% ' +
  'from the first request and never moves. A counter that reports reuse where reuse is ' +
  'impossible is not measuring reuse. Showing it would have been the most convincing false ' +
  'number on this dashboard.';

const NO_NUMBER_NOTE =
  'No timing figure appears here on purpose. An earlier draft of this panel argued from a ' +
  'latency comparison; that measurement was withdrawn by the engineer who ran it after a ' +
  're-run moved the other way inside the noise band. The finding stands without it, because ' +
  'it never rested on it \u2014 counter arithmetic and the source are both timing-free.';

/**
 * Source sites a reader can open to check the claim themselves. Symbol anchors
 * lead each entry: line numbers drift as the engine moves, and a citation that
 * has silently slid onto a different statement is worse than no citation.
 */
const CITATIONS = Object.freeze([
  'engine/runtime.rs \u2014 fn allocate_prompt, the token-cache branch: reports a hit, loads no KV, never sets `loaded_prompt_prefix` (~:1017-1024)',
  'engine/runtime.rs \u2014 the connector branch states the rule its neighbour breaks: "never claiming a hit we can\u2019t serve" (~:1097-1099)',
  'engine/batched.rs \u2014 the continuous batching call sites pass a literal 0 for matched prefix length, so the trie is never consulted (~:262, ~:486)',
  'engine/metrics.rs \u2014 the lookup counter increments per completed generation, not per cache consultation (~:130-135)',
]);

/**
 * Mounts the finding.
 *
 * The signature takes no store, and that is load-bearing rather than
 * incidental: the registry calls `mount(root, store)` for every panel, so this
 * panel is structurally incapable of reading telemetry — there is no binding to
 * audit and none to regress. `prefix-cache.test.js` proves it by mounting with
 * a store that throws on any access.
 *
 * @param {HTMLElement} rootElement
 * @returns {{unmount(): void, describe(): string}}
 */
export default function mount(rootElement) {
  const finding = element('div', { className: 'panel-prefix-cache__finding' });
  const citations = element('ul', { className: 'panel-prefix-cache__citations' });

  replaceChildren(finding, [
    element('p', { className: 'panel-prefix-cache__headline', text: HEADLINE }),
    element('p', { className: 'panel-prefix-cache__body', text: EXPLANATION }),
    element('p', { className: 'panel-prefix-cache__body', text: COUNTER_WARNING }),
    element('p', { className: 'panel-prefix-cache__body', text: NO_NUMBER_NOTE }),
  ]);

  replaceChildren(
    citations,
    CITATIONS.map((text) => element('li', { className: 'panel-prefix-cache__citation', text })),
  );

  rootElement.append(
    finding,
    sectionLabel('verify in source'),
    citations,
    element('p', {
      className: 'panel-prefix-cache__provenance',
      text:
        'This panel binds no telemetry and never repaints. It reports a null result held ' +
        'open for inspection \u2014 the claim is checkable in the engine source cited above, ' +
        'which is why it ships as a finding rather than as a number.',
    }),
  );

  const stopObserving = observeVisibility(rootElement, () => {});

  return {
    unmount() {
      stopObserving();
      rootElement.replaceChildren();
    },
    describe() {
      return (
        'Prefix cache: no hit rate is shown, because prefix reuse was measured and found ' +
        'absent on both execution paths. The continuous batching path passes a hardcoded zero ' +
        'where a matched prefix length belongs, so it never consults the cache. The paged ' +
        'path reports a hit but loads no attention state, so prefill recomputes every token. ' +
        'The engine\u2019s hit counter reports about 95 percent regardless, including on ' +
        'control requests that share nothing, so it is not shown. No timing figure is ' +
        'reported here, and this panel binds no live telemetry.'
      );
    },
  };
}

export { mount };
