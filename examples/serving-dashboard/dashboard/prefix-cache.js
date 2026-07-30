// Copyright (c) Microsoft Corporation.
//
// Panel 4 — Prefix cache. demo-ux.md §5.5.
//
// 🔴 THIS PANEL DELIBERATELY BINDS NO TELEMETRY. THAT IS THE POINT OF IT.
//
// Every counter this panel used to read has been ruled unshippable, and the
// reason is stronger than "unmeasured": @fc8b5d97 ran a controlled A/B and
// found prefix reuse is PROVEN ABSENT on both execution paths.
//
//   ARM A   one identical ~900-token prefix, fired 6x   warm TTFT 1341 ms
//   ARM B   six prefixes differing FROM TOKEN 0         warm TTFT 1254 ms
//
// The shared-prefix arm was 7.0% SLOWER than the arm that shared nothing. And
// the result is PROVEN ABSENT rather than merely unobserved because the same
// run carried a sensitivity control: prefill is ~90% of TTFT (140 ms for a
// ~10-token prompt vs 1380 ms for a ~900-token one), so a working cache would
// have collapsed TTFT from ~1380 ms to ~140 ms. An effect that large cannot
// hide. Observed: +7.0%.
//
// ☠️ WHY THE COUNTER MUST NOT BE SHOWN EVEN AS A CURIOSITY. It reported 19/20
// = 95%. It increments on ANY nonzero token match, and every chat-completions
// request shares the chat-template preamble, so it reads ~95% from the first
// request and never moves. It fired on all six ARM B controls, which share
// nothing. That is not a stub and not a misnamed-but-real number — it is a
// precisely-computed, beautifully-behaved, entirely FALSE value.
//
// Every other safeguard in this tree hunts fabricated ZEROS, because a zero
// looks broken and invites scrutiny. A confident 95% invites none. That is
// exactly why it needed a ruling rather than a guard.
//
// ROOT CAUSE, in the engine, so the next person does not "fix" this by wiring
// the counter back up: engine/runtime.rs:997 forks on uses_token_prefix_cache().
// The token-cache branch (runtime.rs:1017-1024) is REPORTING-ONLY — it loads no
// KV state and never sets `loaded_prompt_prefix`, so the very next statement
// queues the FULL prompt and prefill recomputes every token. The number it
// returns has no compute saving behind it. Thirty lines below, the connector
// path states the correct rule for itself: "never claiming a hit we can't
// serve" (runtime.rs:1097-1099). The reporting branch violates the rule its own
// neighbour states.
//
// So this is a FUNCTIONAL gap in the engine, not a reporting gap. No telemetry
// work will incidentally fix it, and there is no client-side substitute: a TTFT
// delta computed in the browser would measure the same absent effect and land
// at ~0% with a noise band straddling zero.
//
// WHAT SHIPS INSTEAD: the finding itself, with its citations and no live
// numbers. A panel that renders an honest, well-evidenced absence teaches more
// than a panel showing 95% — and vastly more than no panel at all, which would
// leave a visitor to assume the feature works.

import {
  element,
  observeVisibility,
  replaceChildren,
  sectionLabel,
} from './panel-kit.js';

export const meta = Object.freeze({
  // Null, and for a NEW reason since the counters were cut. This is no longer
  // "applicable on one profile, structural n/a on the other" — the gap is real
  // on BOTH paths, for two different architectural reasons, and that pairing is
  // the most teachable thing on the page:
  //   • continuous-batch path — the batch never consults the trie at all
  //     (batched.rs:262 and :486 pass a hardcoded literal 0).
  //   • paged path — the cache IS consulted, but the branch that reports a hit
  //     serves nothing (runtime.rs:1017-1024).
  // Hiding the panel on either profile would conceal half of that.
  requires: null,
  id: 'prefix-cache',
  title: 'Prefix cache',
  group: 'cache',
  span: 1,
  // Static content. Nothing here polls, so cadence describes a panel that never
  // repaints from telemetry — it is present because the contract requires it.
  cadence: 0,
  staleCeilingMs: null,
  defaultOpen: true,
  acronyms: {
    prefix: 'A leading run of tokens shared by several prompts, whose KV state can be reused',
    prefill: 'The forward pass that computes KV state for the prompt before decoding begins',
    TTFT: 'Time to first token — dominated by prefill, so it is where prefix reuse would show',
  },
});

/**
 * The recorded experiment. These are NOT live telemetry and are never presented
 * as such — they are a cited measurement shown with its conditions, per D72, so
 * a reader can judge it rather than take it on faith.
 */
const EVIDENCE = Object.freeze([
  Object.freeze({
    arm: 'Shared prefix',
    detail: 'one identical ~900-token prefix, 6 requests',
    ttft: '1341 ms',
  }),
  Object.freeze({
    arm: 'Control',
    detail: 'six prefixes differing from token 0 — no sharing possible',
    ttft: '1254 ms',
  }),
  Object.freeze({
    arm: 'Sensitivity',
    detail: '~10-token prompt vs ~900-token prompt — prefill is ~90% of TTFT',
    ttft: '140 ms vs 1380 ms',
  }),
]);

const HEADLINE =
  'Prefix reuse is not happening on either execution path. This panel reports that ' +
  'finding instead of a hit rate, because the hit counter is measurably false.';

const EXPLANATION =
  'A controlled A/B found requests sharing a long prefix ran 7.0% slower than requests ' +
  'sharing nothing at all. The same run proved the test could detect reuse if it existed: ' +
  'prefill is about 90% of time-to-first-token, so a working cache would have collapsed ' +
  'TTFT from roughly 1380 ms to 140 ms. Nothing of the sort occurred.';

const COUNTER_WARNING =
  'The engine\u2019s own hit counter reported 95% throughout, including on every control ' +
  'request that shared no prefix. It rises on any single matching token, and every chat ' +
  'request shares the template preamble, so it reads ~95% from the first request and never ' +
  'moves. Showing it would have been the most convincing false number on this dashboard.';

const CITATIONS = Object.freeze([
  'engine/runtime.rs:1017-1024 — the reporting-only branch: no KV loaded, `loaded_prompt_prefix` never set, so prefill recomputes the whole prompt',
  'engine/runtime.rs:1097-1099 — the correct rule, stated for the neighbouring path: "never claiming a hit we can\u2019t serve"',
  'batched.rs:262, :486 — the continuous-batch path passes a hardcoded literal 0, so it never consults the trie',
  'metrics.rs:130-135 — the lookup counter increments on every completed generation, not on every cache consultation',
]);

/**
 * @param {HTMLElement} rootElement
 * @returns {{unmount(): void, describe(): string}}
 */
export default function mount(rootElement) {
  const finding = element('div', { className: 'panel-prefix-cache__finding' });
  const evidence = element('div', { className: 'panel-prefix-cache__evidence' });
  const citations = element('ul', { className: 'panel-prefix-cache__citations' });

  replaceChildren(finding, [
    element('p', { className: 'panel-prefix-cache__headline', text: HEADLINE }),
    element('p', { className: 'panel-prefix-cache__body', text: EXPLANATION }),
    element('p', { className: 'panel-prefix-cache__body', text: COUNTER_WARNING }),
  ]);

  replaceChildren(
    evidence,
    EVIDENCE.map((row) => {
      const line = element('div', { className: 'panel-prefix-cache__arm' });
      line.append(
        element('span', { className: 'panel-prefix-cache__arm-name', text: row.arm }),
        element('span', { className: 'panel-prefix-cache__arm-detail', text: row.detail }),
        element('span', { className: 'panel-prefix-cache__arm-value', text: row.ttft }),
      );
      return line;
    }),
  );

  replaceChildren(
    citations,
    CITATIONS.map((text) => element('li', { className: 'panel-prefix-cache__citation', text })),
  );

  rootElement.append(
    finding,
    sectionLabel('recorded measurement — not live telemetry'),
    evidence,
    sectionLabel('verify in source'),
    citations,
    element('p', {
      className: 'panel-prefix-cache__provenance',
      text:
        'Measured by QA on the dynamic model. Recorded once, not polled \u2014 these numbers ' +
        'do not change while you watch, and are not claimed to.',
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
        'absent on both execution paths. Requests sharing a 900-token prefix took 1341 ' +
        'milliseconds to first token, while control requests sharing nothing took 1254 ' +
        'milliseconds \u2014 7 percent faster. A sensitivity check showed prefill is about 90 ' +
        'percent of time to first token, so working reuse would have been unmistakable. The ' +
        'engine\u2019s hit counter reports 95 percent regardless, including on the controls, so ' +
        'it is not shown. These are recorded measurements, not live telemetry.'
      );
    },
  };
}

export { mount };
