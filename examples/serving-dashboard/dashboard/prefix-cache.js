// Copyright (c) Microsoft Corporation.
//
// Panel 4 — Prefix cache. demo-ux.md §5.5.
//
// THIS PANEL SHIPS UNCONDITIONALLY, whatever the numbers say. If the cache
// genuinely reports few hits, this panel renders that at full contrast. A real
// measurement of a real system in an unflattering state is worth more to this
// project's reputation than a hidden panel. The SCENARIO is cuttable; the panel
// is not.
//
// ⚠️ BOUND AGAINST THE PROVENANCE AUDIT, NOT AGAINST WHAT LOOKS AVAILABLE.
// @d7cf9b84's provenance-audit.md §2 classifies the wire fields as follows, and
// this panel follows the audit rather than the field names:
//
//   • `prefix_cache_hits` — the counter mechanism is REAL (metrics.rs:135-137,
//     incremented only when prefix_cache_hit_len > 0), but it is PATH-DEPENDENT:
//     on the continuous-batch (static-cache) path it is structurally pinned at
//     zero, because batched.rs:262 and :486 construct DecodeLoopState with a
//     hardcoded literal 0. So a zero here can mean "no reuse" OR "this code path
//     cannot report reuse" — and those must not look the same.
//
//   • `prefix_cache_lookups` — MISNAMED. metrics.rs:130-132 does fetch_add(1) on
//     EVERY completed generation, whether or not any cache was consulted. It is
//     a genuine counter of the wrong noun. This panel therefore labels it
//     "completed generations", NOT "lookups". Labelling it correctly costs us a
//     nicer-looking headline and buys the only thing that matters here.
//
//   • `prefix_cache_hit_rate` — DERIVED FROM THE MISNAMED DENOMINATOR, and it
//     emits a literal 0.0 when the denominator is zero, so "no data" and "0%
//     hit rate" are the same six characters on the wire. This panel does NOT
//     render the server's hit_rate field. It shows the hit COUNT, which is real,
//     and em-dashes the rate with a reason naming the defect.
//
// If that reads as an unusually loud comment for a small panel: this is the one
// panel where the field names actively mislead, and the next person to touch it
// will otherwise "fix" it back to a fabricated number in good faith.

import { isRenderable, numericValueOf } from './field-state.js';
import {
  createRepaintScheduler,
  bindPanel,
  createSparklineSlot,
  renderSparkline,
  describeFieldText,
  element,
  formatDuration,
  observeVisibility,
  replaceChildren,
  sectionLabel,
} from './panel-kit.js';

const WINDOW_MS = 60_000;

/** Reasons specific to this panel's known defects, shown verbatim on hover. */
const PREFIX_REASONS = Object.freeze({
  RATE_FROM_MISNAMED_DENOMINATOR:
    'The server computes hit rate as hits ÷ lookups, but its lookup counter increments on ' +
    'every completed generation rather than on every cache consultation, so the ratio has no ' +
    'defensible denominator. The hit count beside it is real.',
  NO_GENERATIONS_YET:
    'No generations have completed yet, so there is nothing to compute a rate from. ' +
    'Run a scenario.',
  PATH_PINNED_AT_ZERO:
    'This model runs on the continuous-batch (static cache) path, where the decode loop is ' +
    'constructed with a prefix-cache hit length of zero, so the counter cannot rise here. ' +
    'Run a scenario on the dynamic-cache model to exercise the prefix cache.',
});

export const meta = Object.freeze({
  id: 'prefix-cache',
  title: 'Prefix cache',
  group: 'cache',
  span: 1,
  cadence: 250,
  // Cache counters are cumulative; a slightly old count still describes the same run.
  staleCeilingMs: 15000,
  defaultOpen: true,
  acronyms: {
    prefix: 'A leading run of tokens shared by several prompts, whose KV state can be reused',
    prefill: 'The forward pass that computes KV state for the prompt before decoding begins',
    eviction: 'Reclaiming cached prefix state so its memory can serve another sequence',
  },
});

/** Panel-scoped renderers carrying this panel's stale ceiling (AC45(c)). */
const { metricRow, renderField } = bindPanel(meta);

/**
 * @param {HTMLElement} rootElement
 * @param {any} telemetryStore
 * @returns {{destroy(): void, describe(): string}}
 */
export default function mount(rootElement, telemetryStore) {
  const hero = element('div', { className: 'panel-prefix-cache__hero' });
  const counts = element('div', { className: 'panel-prefix-cache__counts' });
  const savings = element('div', { className: 'panel-prefix-cache__savings' });
  const spark = createSparklineSlot({ label: 'Prefix cache hits', width: 220, height: 30 });

  rootElement.append(hero, spark.root, counts, sectionLabel('savings'), savings);

  let description = 'Prefix cache: waiting for the first sample.';

  const paint = () => {
    const hits = telemetryStore.field('prefix_cache.hits');
    const generations = telemetryStore.field('prefix_cache.lookups');
    const capability = telemetryStore.capability?.('prefix-cache') ?? { available: true };

    const hitRate = deriveHitRate(hits, generations, capability);

    replaceChildren(hero, [
      renderField(hitRate, { label: 'Prefix cache hit rate' }),
      element('span', { className: 'hero-figure__caption', text: 'hit rate' }),
    ]);

    replaceChildren(counts, [
      metricRow('hits', annotateHits(hits, capability)),
      // NOT "lookups". See the file header: the server's counter of that name
      // counts completed generations, and calling it a lookup count here would
      // launder a real counter of the wrong noun into a plausible-looking one.
      metricRow('completed generations', generations, {
        label: 'Completed generations (the server\u2019s "lookups" counter)',
      }),
    ]);

    replaceChildren(savings, [
      metricRow('tokens reused', telemetryStore.field('prefix.tokens_reused')),
      metricRow('prefill skipped', telemetryStore.field('prefix.prefill_tokens_skipped')),
      metricRow('time saved', telemetryStore.field('prefix.time_saved_ms'), {
        format: (value) => formatDuration(value),
      }),
      metricRow('evictions', telemetryStore.field('prefix.evictions')),
    ]);

    const hitSeries = telemetryStore.series('prefix_cache.hits', WINDOW_MS);
    renderSparkline(spark, hitSeries, {
      width: 220,
      height: 30,
      windowMs: WINDOW_MS,
      nowMs: Date.now(),
      label: 'Prefix cache hits',
      unit: 'hits',
    });

    description = buildDescription({ hitRate, hits, generations, capability });
  };

  const scheduler = createRepaintScheduler(rootElement, paint);
  const stopObserving = observeVisibility(rootElement, (visible) => scheduler.setVisible(visible));
  const unsubscribe = telemetryStore.subscribe(() => scheduler.request());
  scheduler.request();

  return {
    unmount() {
      unsubscribe();
      stopObserving();
      scheduler.cancel();
      rootElement.replaceChildren();
    },
    describe() {
      return description;
    },
  };
}

// ── deriving ─────────────────────────────────────────────────────────────────

/**
 * Derive a hit rate, or decline to.
 *
 * Deliberately NOT `telemetryStore.field('prefix.hit_rate')`: the server's own
 * hit_rate is hits ÷ completed-generations and emits a literal 0.0 when the
 * denominator is zero. Computing it here instead of reading it lets the panel
 * distinguish the three cases the wire value collapses into one — no data,
 * a genuine zero, and a code path that structurally cannot report hits.
 *
 * @param {any} hits
 * @param {any} generations
 * @param {{available: boolean, reason?: string}} capability
 * @returns {object}
 */
export function deriveHitRate(hits, generations, capability = { available: true }) {
  // Not-applicability outranks unavailability and is contagious: a rate whose
  // inputs are meaningless on this execution path is itself meaningless, not
  // "not measured yet". Getting this backwards is what made the headline number
  // on this panel apologise on the scatter server while its own supporting rows
  // correctly said n/a.
  const structural =
    capability.state === 'not-applicable' ||
    hits?.state === 'not-applicable' ||
    generations?.state === 'not-applicable';

  if (structural) {
    return {
      value: null,
      state: 'not-applicable',
      source: 'derived',
      unit: '%',
      label: 'Prefix cache hit rate',
      reason: capability.reason ?? hits?.reason ?? PREFIX_REASONS.PATH_PINNED_AT_ZERO,
    };
  }

  if (!capability.available) {
    return {
      value: null,
      state: 'unavailable',
      source: 'derived',
      unit: '%',
      label: 'Prefix cache hit rate',
      reason: capability.reason ?? PREFIX_REASONS.PATH_PINNED_AT_ZERO,
    };
  }

  const hitCount = numericValueOf(hits);
  const generationCount = numericValueOf(generations);

  if (hitCount === null || generationCount === null) {
    return {
      value: null,
      state: 'unavailable',
      source: 'derived',
      unit: '%',
      label: 'Prefix cache hit rate',
      reason: PREFIX_REASONS.RATE_FROM_MISNAMED_DENOMINATOR,
    };
  }

  if (generationCount === 0) {
    // "No generations yet" is PENDING, not unavailable: it resolves itself the
    // moment a scenario runs. Rendering 0% here would be the exact fabrication
    // the server already commits at metrics.rs:301-309.
    return {
      value: null,
      state: 'pending',
      source: 'derived',
      unit: '%',
      label: 'Prefix cache hit rate',
      reason: PREFIX_REASONS.NO_GENERATIONS_YET,
    };
  }

  // The denominator is real arithmetic over a counter that counts the wrong
  // noun. We show the number because it is the best available summary AND we
  // name the defect on hover — the alternative, hiding it, tells the visitor
  // less than telling them exactly what it is.
  return {
    value: (hitCount / generationCount) * 100,
    state: 'ok',
    source: 'derived',
    unit: '%',
    label: 'Prefix cache hit rate',
    derivedFrom: ['prefix_cache.hits', 'prefix_cache.lookups'],
  };
}

/**
 * Attach the path-dependence caveat to a zero hit count.
 *
 * A zero here has two very different meanings — "nothing was reused" and "this
 * code path cannot report reuse" — and only the second one is a caveat the
 * visitor needs. The value stays a real, rendered zero either way; the hover
 * gains a sentence.
 *
 * @param {any} hits
 * @param {{available: boolean, reason?: string}} capability
 * @returns {any}
 */
function annotateHits(hits, capability) {
  if (!isRenderable(hits) || numericValueOf(hits) !== 0 || capability.available !== false) {
    return hits;
  }
  return { ...hits, reason: capability.reason ?? PREFIX_REASONS.PATH_PINNED_AT_ZERO };
}

/**
 * @param {Record<string, any>} fields
 * @returns {string}
 */
function buildDescription(fields) {
  const parts = ['Prefix cache:'];

  if (isRenderable(fields.hitRate)) {
    parts.push(`hit rate ${Number(fields.hitRate.value).toFixed(1)} percent,`);
  } else if (fields.hitRate.state === 'pending') {
    parts.push('no generations have completed yet, so there is no hit rate to report,');
  } else if (fields.hitRate.state === 'not-applicable') {
    // The cache is never consulted on this path, so there is no rate to have.
    // "Not measurable yet" would imply the cache tried and we failed to observe.
    parts.push('there is no hit rate here because the cache is never consulted,');
  } else {
    parts.push('hit rate is not measurable yet,');
  }

  parts.push(`${describeFieldText('hits', fields.hits)}.`);
  parts.push(
    `${describeFieldText('Completed generations', fields.generations)} ` +
      "(this is the server's lookup counter, which counts generations rather than cache consultations).",
  );

  if (fields.capability.available === false) {
    parts.push(fields.capability.reason ?? PREFIX_REASONS.PATH_PINNED_AT_ZERO);
  }
  // The panel ships unconditionally on every engine. It is never hidden to make
  // a demo look stronger, so the description always ends up saying something
  // true rather than nothing at all.
  return parts.join(' ');
}

export { mount };
