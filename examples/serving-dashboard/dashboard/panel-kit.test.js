// Copyright (c) Microsoft Corporation.
//
// Tests for the shared panel rendering helpers.
//
// The centre of gravity is `renderField`. It is the function every displayed
// number on this page passes through, so it is the one place where the
// difference between "the server measured zero" and "the server cannot measure
// this" is either preserved or destroyed. These tests hold that line.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { after, before, describe, it } from 'node:test';

import { flushAnimationFrames, installFakeDom } from './testing/fake-dom.js';

// These exercise scheduler MECHANICS, not AC62. A store publishing no endpoint
// errors keeps the gated-endpoint notice inert so it cannot perturb the counts.
const noGateStoreOptions = {
  telemetryStore: { getSnapshot: () => ({ endpointErrors: {} }) },
};

let uninstallDom;
before(() => {
  uninstallDom = installFakeDom();
});
after(() => uninstallDom());

const {
  bindPanel,
  capabilityNotice,
  createRepaintScheduler,
  describeFieldText,
  formatDuration,
  formatNumber,
  metricRow,
  renderField,
  SOURCE_BADGES,
  sourceBadge,
  UNKNOWN_SOURCE_BADGE,
  UNKNOWN_SOURCE_CLASS,
  withAcronyms,
} = await import('./panel-kit.js');

const { PANELS } = await import('./index.js');
const { createFakeStore, measured } = await import('./testing/fake-store.js');

/**
 * A store that answers every field and rate with a large measured value, so the
 * panels that scale bytes and durations all take their formatted branch.
 *
 * It delegates to the shared `createFakeStore` rather than being hand-written:
 * a bespoke fake diverged from the real store contract twice while this defect
 * was being diagnosed, and a fake that throws on `rate()` mounts nothing and
 * reports a clean page.
 */
function doubleUnitProbeStore() {
  const store = Object.create(createFakeStore());
  const unitFor = (path) => (/uptime|latency|duration|ttft|age|time/i.test(path) ? 'ms' : 'bytes');
  store.field = (path) => measured(5.35 * 1024 ** 3, { unit: unitFor(path), label: path });
  store.rate = (path) => measured(1234.5, { unit: unitFor(path), label: path });
  return store;
}

describe('renderField — a measured zero and an unmeasurable value must look different', () => {
  it('renders a real zero as a full-contrast number, with no apology', () => {
    // demo-ux.md §4.1: "Measured. The value really is zero." `rejections: 0` is
    // real and good news; hiding or dimming it would misrepresent the server.
    const node = renderField({ value: 0, state: 'ok', source: 'server', unit: 'count', label: 'Rejections' });

    assert.equal(node.getAttribute('data-state'), 'measured');
    assert.equal(node.findByClass('value__num').textContent, '0');
    assert.equal(node.findByClass('value__num--unavailable'), null);
  });

  it('renders a real false as literal false, never as an em-dash', () => {
    const node = renderField(measured(false, {
      source: 'server',
      label: 'Feature enabled',
    }));

    assert.equal(node.getAttribute('data-state'), 'measured');
    assert.equal(node.findByClass('value__num').textContent, 'false');
    assert.equal(node.findByClass('value__num--unavailable'), null);
    assert.doesNotMatch(node.textContent, /—/);
  });

  it('renders an unavailable value as an em-dash and never as a number', () => {
    const node = renderField({
      value: null,
      state: 'unavailable',
      source: 'server',
      unit: 'count',
      label: 'Preemptions',
      reason: 'The scheduler performs preemption but keeps no counter for it.',
    });

    assert.equal(node.getAttribute('data-state'), 'unavailable');
    assert.equal(node.findByClass('value__num--unavailable').textContent, '—');
    assert.doesNotMatch(node.textContent, /\d/, 'an unavailable field must contain no digits at all');
  });

  it('keeps the unit when the value is unavailable, because WHICH thing is missing is information', () => {
    const node = renderField({
      value: null,
      state: 'unavailable',
      unit: 'ms',
      label: 'Queue wait',
      reason: 'The browser can only observe sent to first token.',
    });

    // Trimmed: the span carries a leading space so that a value and its unit
    // do not fuse into "12ms" in textContent, which the table view and a
    // clipboard copy both read. What this test cares about is that the unit
    // SURVIVES an absent value -- knowing which of the two is missing is itself
    // information -- not how it is padded.
    assert.equal(node.findByClass('value__unit').textContent.trim(), 'ms');
  });

  it('makes the reason reachable by keyboard and screen reader, not hover only', () => {
    const reason = 'KV page statistics are computed by the engine but not yet exposed over HTTP.';
    const node = renderField({ value: null, state: 'unavailable', label: 'KV blocks', reason });

    // Focusable, but as a stop on a roving cursor rather than its own tab stop
    // (AC29). createRovingGroup promotes one of these to tabindex 0.
    assert.equal(node.getAttribute('tabindex'), '-1', 'the explanation must be focusable');
    assert.equal(node.hasAttribute('data-roving-item'), true, 'must join the roving cursor');
    assert.match(node.getAttribute('aria-label'), /KV blocks: not measurable yet/);
    assert.match(node.getAttribute('aria-label'), /not yet exposed over HTTP/);
    assert.equal(node.getAttribute('title'), reason);
  });

  it('renders pending distinctly from unavailable', () => {
    const node = renderField({ value: null, state: 'pending', unit: 'tok/s', label: 'Throughput' });

    assert.equal(node.getAttribute('data-state'), 'pending');
    assert.equal(node.findByClass('value__num--pending').textContent, '···');
    assert.match(node.getAttribute('aria-label'), /no samples yet/i);
  });

  it('degrades an unknown state to an em-dash rather than printing the value', () => {
    // Non-strict is the browser path: never throw, never print the value.
    const node = renderField(
      { value: 99, state: 'freshly-invented', label: 'Anything' },
      { strict: false },
    );

    assert.equal(node.getAttribute('data-state'), 'unavailable');
    assert.doesNotMatch(node.textContent, /99/);

    // ...and in development the same field stops the build instead.
    assert.throws(() => renderField({ value: 99, state: 'freshly-invented', label: 'Anything' }));
  });

  it('shows a stale value with its age instead of presenting it as current', () => {
    const node = renderField({
      value: 42,
      state: 'stale',
      source: 'server',
      unit: 'count',
      label: 'Queue depth',
      observedAtMs: Date.now() - 8000,
    });

    assert.equal(node.getAttribute('data-stale'), 'true');
    assert.match(node.findByClass('value__stale').textContent, /^8s old$/);
    // The age must reach a screen reader too, not just the pixels.
    assert.match(node.getAttribute('aria-label'), /stale, 8s old/);
  });

  it('will not date a value from an unrecognised property', () => {
    // This test exists because the original one did not. It asserted the age
    // suffix appeared, using a fixture whose timestamp property the renderer
    // did not actually read. Both sides agreed and both were wrong, so every
    // stale value in the real app rendered "0s ago" — an actively false claim
    // of freshness on the exact values AC45 protects.
    //
    // `observedAtMs` and `at` are both accepted now (the ruled envelope and the
    // implemented one disagree). Anything else must still refuse to date it.
    const node = renderField({
      value: 42,
      state: 'stale',
      source: 'server',
      label: 'Queue depth',
      lastSeen: Date.now() - 8000,
    });

    assert.equal(
      node.getAttribute('data-stale'),
      'expired',
      'an undateable stale value must not be shown as a number',
    );
    assert.match(node.getAttribute('aria-label'), /age unknown/);
    assert.doesNotMatch(node.textContent, /42/, 'the number survived without a trustworthy age');
  });
});

describe('renderField — AC7 provenance badges', () => {
  it('badges a server counter, a client measurement and a derivation differently', () => {
    const server = renderField({ value: 1, state: 'ok', source: 'server', label: 'a' });
    const client = renderField({ value: 1, state: 'ok', source: 'client', label: 'b' });
    const derived = renderField({ value: 1, state: 'ok', source: 'derived', label: 'c' });

    assert.equal(server.findByClass('value__src').textContent, 'ˢ');
    assert.equal(client.findByClass('value__src').textContent, 'ᶜ');
    assert.equal(derived.findByClass('value__src').textContent, 'ᴰ');
  });

  it('treats an endpoint path as a server source, so a curl-able origin still badges', () => {
    // The ruling splits source (badge class) from origin (endpoint string), but
    // CONTRACT.md still puts '/v1/status' in `source`. Tolerating a path here
    // means the badge is right under either shape.
    const node = renderField({ value: 7, state: 'ok', source: '/v1/status', label: 'Queue depth' });

    assert.equal(node.findByClass('value__src').textContent, 'ˢ');
    assert.match(node.findByClass('value__src').getAttribute('title'), /\/v1\/status/);
  });

  it('prefixes an estimated value with a tilde so it reads as a different kind of claim', () => {
    const node = renderField({ value: 340, state: 'ok', source: 'estimated', unit: 'ms', label: 'Time saved' });

    assert.match(node.findByClass('value__num').textContent, /^~/);
    assert.equal(node.findByClass('value__src').textContent, 'ᴱ');
  });

  it('spells SIMULATED out in full, because a superscript is easy to miss', () => {
    assert.equal(sourceBadge('simulated').textContent, 'SIM');
    assert.match(sourceBadge('simulated').getAttribute('title'), /not measured at all/i);
  });

  // The four tests below exist because the fallback used to be
  // `SOURCE_BADGES.derived`, so every value the catalogue could not identify was
  // rendered as "Derived by arithmetic on measured inputs" — a positive claim
  // about measurement, handed out on no evidence. It was reachable from three
  // different directions and no test noticed, because a wrong badge still
  // renders a badge.
  it('refuses to answer an unrecognised source class with the derived claim', () => {
    const badge = sourceBadge('guessed');

    assert.equal(badge.textContent, UNKNOWN_SOURCE_BADGE.glyph);
    assert.notEqual(badge.textContent, SOURCE_BADGES.derived.glyph);
    assert.doesNotMatch(badge.getAttribute('title'), /arithmetic on measured inputs/i);
    assert.match(badge.getAttribute('title'), /unknown provenance/i);
  });

  it('renders a field carrying no provenance at all as unknown, not as derived', () => {
    const node = renderField({ value: 41, state: 'measured', label: 'Mystery' });

    assert.equal(node.findByClass('value__src').textContent, UNKNOWN_SOURCE_BADGE.glyph);
  });

  it('still says derived when there is evidence of the arithmetic', () => {
    const node = renderField({
      value: 41,
      state: 'measured',
      label: 'Tokens per second',
      derivedFrom: ['tokens', 'elapsed'],
    });

    assert.equal(node.findByClass('value__src').textContent, SOURCE_BADGES.derived.glyph);
  });

  it('keeps the unknown class out of SOURCE_BADGES, which is used as a validity predicate', () => {
    // `normaliseSourceClass` decides whether a caller-supplied string is a real
    // source class with `x in SOURCE_BADGES`. If the unknown class were a member
    // it would validate as a genuine provenance, which is the exact inversion
    // this constant exists to prevent.
    assert.equal(UNKNOWN_SOURCE_CLASS in SOURCE_BADGES, false);
    assert.equal(sourceBadge(UNKNOWN_SOURCE_CLASS).textContent, UNKNOWN_SOURCE_BADGE.glyph);
  });
});

describe('formatNumber / formatDuration', () => {
  it('picks precision by magnitude so the same quantity reads the same everywhere', () => {
    assert.equal(formatNumber(0), '0');
    assert.equal(formatNumber(7.25), '7.25');
    assert.equal(formatNumber(98.74), '98.7');
    assert.equal(formatNumber(318), '318');
    assert.equal(formatNumber(4602), '4,602');
  });

  it('appends a percent sign rather than a spaced unit', () => {
    assert.equal(formatNumber(62.1, '%'), '62.1%');
  });

  it('returns an em-dash for a non-finite number instead of NaN or Infinity', () => {
    assert.equal(formatNumber(NaN), '—');
    assert.equal(formatNumber(Infinity), '—');
  });

  it('scales duration units at human thresholds', () => {
    assert.equal(formatDuration(0.4), '0.40 ms');
    assert.equal(formatDuration(38), '38 ms');
    assert.equal(formatDuration(1840), '1.84 s');
    assert.equal(formatDuration(124_000), '2 m 04 s');
  });
});

describe('metricRow', () => {
  it('labels the value for a screen reader even when it is an em-dash', () => {
    const row = metricRow('Preemptions', {
      value: null,
      state: 'unavailable',
      reason: 'No counter exists.',
    });

    assert.equal(row.findByClass('metric-row__label').textContent, 'Preemptions');
    assert.match(row.findByClass('value').getAttribute('aria-label'), /^Preemptions: not measurable/);
  });
});

describe('capabilityNotice', () => {
  it('names the fix and reassures that the rest of the page still works', () => {
    const notice = capabilityNotice({
      title: 'KV block introspection is off',
      body: "The engine computes page statistics; this server build doesn't expose them.",
      command: '--enable-debug-endpoints',
    });

    assert.match(notice.textContent, /--enable-debug-endpoints/);
    assert.match(
      notice.textContent,
      /Everything else on this page still works\./,
      'AC20: a missing flag is not a broken dashboard, and the copy must say so',
    );
  });

  it('gives the copy button an accessible name that says what it copies', () => {
    const notice = capabilityNotice({ title: 't', body: 'b', command: '--enable-admin-endpoints' });

    assert.equal(
      notice.findByClass('capability-notice__copy').getAttribute('aria-label'),
      'Copy --enable-admin-endpoints to the clipboard',
    );
  });
});

describe('withAcronyms — AC30', () => {
  it('defines each acronym on its first appearance only', () => {
    const fragment = withAcronyms('TTFT is the time to first token; TTFT is measured per request.', {
      TTFT: 'Time to first token',
    });
    const host = document.createElement('div');
    host.append(fragment);

    const abbreviations = host.findAllByTag('abbr');
    assert.equal(abbreviations.length, 1, 'a definition on every occurrence is noise, not help');
    assert.equal(abbreviations[0].getAttribute('title'), 'Time to first token');
    assert.equal(host.textContent, 'TTFT is the time to first token; TTFT is measured per request.');
  });

  it('leaves text untouched when there are no acronyms to define', () => {
    const host = document.createElement('div');
    host.append(withAcronyms('Plain sentence.', {}));

    assert.equal(host.textContent, 'Plain sentence.');
    assert.equal(host.findAllByTag('abbr').length, 0);
  });
});

describe('createRepaintScheduler — AC23', () => {
  it('coalesces many requests in one frame into a single paint', () => {
    const root = document.createElement('div');
    let paints = 0;
    const scheduler = createRepaintScheduler(root, () => {
      paints += 1;
    }, noGateStoreOptions);

    scheduler.request();
    scheduler.request();
    scheduler.request();
    flushAnimationFrames();

    assert.equal(paints, 1, 'three requests in one frame must produce one paint, not three');
  });

  it('does not paint while the panel is hidden, and catches up when it returns', () => {
    const root = document.createElement('div');
    let paints = 0;
    const scheduler = createRepaintScheduler(root, () => {
      paints += 1;
    }, noGateStoreOptions);

    scheduler.setVisible(false);
    scheduler.request();
    flushAnimationFrames();
    assert.equal(paints, 0, 'an off-screen panel must not burn frames');

    scheduler.setVisible(true);
    flushAnimationFrames();
    assert.equal(paints, 1, 'and it must repaint once it is on screen again');
  });

  it('respects the shell setting `hidden` on a collapsed panel body', () => {
    const root = document.createElement('div');
    root.hidden = true;
    let paints = 0;
    const scheduler = createRepaintScheduler(root, () => {
      paints += 1;
    }, noGateStoreOptions);

    scheduler.request();
    flushAnimationFrames();

    assert.equal(paints, 0);
  });

  it('cancel() prevents a queued paint, so destroy() cannot leave a frame in flight', () => {
    const root = document.createElement('div');
    let paints = 0;
    const scheduler = createRepaintScheduler(root, () => {
      paints += 1;
    }, noGateStoreOptions);

    scheduler.request();
    scheduler.cancel();
    flushAnimationFrames();

    assert.equal(paints, 0);
  });
});

describe('describeFieldText — the sentence panels build describe() from', () => {
  it('says a value is not measurable rather than reporting a number', () => {
    assert.equal(
      describeFieldText('Preemptions', { value: null, state: 'unavailable' }),
      'Preemptions is not measurable yet',
    );
  });

  it('distinguishes "no samples yet" from "not measurable"', () => {
    assert.equal(describeFieldText('Hit rate', { state: 'pending' }), 'Hit rate has no samples yet');
  });

  it('includes the unit for a measured value', () => {
    assert.equal(
      describeFieldText('Queue depth', { value: 3, state: 'ok', unit: 'requests' }),
      'Queue depth 3 requests',
    );
  });
});

// ---------------------------------------------------------------------------
// textContent is a PRODUCT SURFACE, not a test convenience.
//
// This has now been the root cause of two separate defects: a stale value
// fusing with its age into "4112s old", and a value fusing with its unit into
// "3sequences". Both looked perfect in a browser, because the spacing was
// supplied by a CSS `gap` that exists only in layout.
//
// The reason it matters is that the rendered text is read by things that never
// see the CSS: the "view as table" rendering, `describe()` (which becomes the
// chart's accessible name), a screen reader's flat traversal, and anyone who
// selects the number and copies it. So this is pinned rather than left to be
// rediscovered a third time.
// ---------------------------------------------------------------------------

describe('rendered text is legible without stylesheets', () => {
  it('separates a value from its unit in textContent, not merely with a CSS gap', () => {
    const rendered = renderField(
      { value: 3, state: 'ok', unit: 'sequences', label: 'active batch', source: 'server' },
      { label: 'active batch' },
    );

    assert.match(rendered.textContent, /3 sequences/);
    assert.doesNotMatch(
      rendered.textContent,
      /3sequences/,
      'the space is supplied by CSS only; a table view or clipboard copy loses it',
    );
  });

  it('separates both terms of an n-of-m ratio', () => {
    const rendered = renderField(
      { value: 3, state: 'ok', unit: 'of 4', label: 'occupancy', source: 'derived' },
      { label: 'occupancy' },
    );

    assert.match(rendered.textContent, /3 of 4/);
  });
});

describe('renderField — a wire-spelling flip must not blank the dashboard', () => {
  it('renders the SAME number under both ratified spellings of measured', () => {
    // The failure this prevents is silent and inverted: if the panels knew only
    // the spelling that did not ship, every live field would fall to the
    // unknown-state path and the page would fill with em-dashes while the
    // server answered perfectly. Nothing would be logged, and the dashboard's
    // own honesty machinery would be the thing testifying the data is absent.
    const ok = renderField({ value: 41, state: 'ok', label: 'Tokens', unit: 'tok/s' });
    const measured = renderField({ value: 41, state: 'measured', label: 'Tokens', unit: 'tok/s' });

    assert.equal(measured.getAttribute('data-state'), ok.getAttribute('data-state'));
    assert.equal(measured.textContent, ok.textContent);
    assert.match(measured.textContent, /41/);
  });

  it('still renders a genuine zero as a stark 0 under either spelling', () => {
    // The documented-zero trap: a real measured 0 must never be mistaken for a
    // missing value, and that must hold on both sides of the rename.
    for (const state of ['ok', 'measured']) {
      const node = renderField({ value: 0, state, label: 'Cache hits', unit: '%' });
      assert.match(node.textContent, /0/, `a measured zero vanished under state="${state}"`);
      assert.doesNotMatch(node.textContent, /—/, `a measured zero em-dashed under "${state}"`);
    }
  });
});

describe('not-applicable copy — demo-ux.md §20.2', () => {
  // This state is the ONLY one whose explanation renders on screen unprompted,
  // and under two servers it is the NORMAL case rather than the exception. So
  // this copy is read more than any other text the dashboard produces, and it
  // is the sentence carrying the demo's central technical claim.

  // Each of these turns an architectural FACT back into a missing FEATURE.
  // "Not yet" and "coming soon" promise a value that can never arrive on this
  // execution path; "unfortunately" apologises for a design decision; and
  // "currently unavailable" is the exact phrasing of the OTHER state, which
  // collapses the distinction the whole five-state vocabulary exists to draw.
  const BANNED = [/\bnot yet\b/i, /\bcoming soon\b/i, /\bunfortunately\b/i, /\bcurrently unavailable\b/i];

  /** Every reason the provenance table can hand to a not-applicable field. */
  async function bypassedReasons() {
    const { PROVENANCE } = await import('../telemetry-provenance.js');
    /** @type {{key: string, reason: string}[]} */
    const found = [];
    for (const [key, entry] of Object.entries(PROVENANCE)) {
      for (const perOrigin of Object.values(entry?.byOrigin ?? {})) {
        if (perOrigin?.classification === 'STRUCTURALLY_BYPASSED' && perOrigin.reason) {
          found.push({ key, reason: perOrigin.reason });
        }
      }
    }
    return found;
  }

  it('never apologises for a value that was never possible', async () => {
    const reasons = await bypassedReasons();
    assert.ok(reasons.length > 0, 'expected the provenance table to declare bypassed fields');

    for (const { key, reason } of reasons) {
      for (const banned of BANNED) {
        assert.doesNotMatch(
          reason,
          banned,
          `${key}: this wording reframes an architectural fact as a missing feature`,
        );
      }
    }
  });

  it('cites the source that proves the bypass, so a reader can check us', async () => {
    // An unfalsifiable claim about our own architecture is just marketing. The
    // citation is what makes this a teaching surface rather than an assertion.
    for (const { key, reason } of await bypassedReasons()) {
      assert.match(reason, /\.rs:\d+/, `${key}: no file:line citation to check the claim against`);
    }
  });

  it('has a headline sentence that stands alone on screen', async () => {
    // The caption shows the first sentence; the rest is one hover away. If a
    // reason's opening sentence does not itself say the path never asks the
    // question, the on-screen text becomes a fragment and the meaning moves
    // entirely into the tooltip nobody opens.
    for (const { key, reason } of await bypassedReasons()) {
      const node = renderField({ state: 'not-applicable', reason, label: key });
      const caption = node.textContent.replace(/^n\/a/, '');

      assert.ok(caption.length > 20, `${key}: on-screen caption is too short to mean anything`);
      assert.doesNotMatch(
        caption,
        /\.rs:\d+$/,
        `${key}: caption was cut mid-citation — a file:line must not be read as a sentence end`,
      );
      assert.match(
        caption,
        /never|no |bypass/i,
        `${key}: the visible sentence must say the path cannot ask, not merely that a value is absent`,
      );
    }
  });
});

describe('the unit suffix belongs to whoever formatted the number', () => {
  // This defect shipped to the served page: "5.35 GiB bytes" and "3 m 12 s ms".
  // `formatBytes` and `formatDuration` SCALE a value, so they must emit the
  // unit they scaled to — and `renderField` then appended the field's raw unit
  // on top. Every one of the 282 tests passing at the time stayed green,
  // because each checked the number OR the unit and never the assembled string.
  //
  // The rule these tests pin: a custom formatter owns its unit; the default
  // formatter does not, so the suffix is derived from `options.format` and
  // cannot be forgotten at a call site.
  const formatGiB = (value) => `${(value / 1024 ** 3).toFixed(2)} GiB`;
  const measuredBytes = (value = 5.35 * 1024 ** 3) => ({
    value,
    state: 'measured',
    unit: 'bytes',
    source: 'server',
    observedAtMs: Date.now(),
  });

  it('does not append the raw unit when a custom formatter already scaled it', () => {
    const node = renderField(measuredBytes(), { format: formatGiB, label: 'VRAM' });
    const text = node.textContent.replace(/[ˢᵉᶜ]/g, '').trim();

    assert.equal(text, '5.35 GiB');
    assert.doesNotMatch(text, /GiB\s+bytes/, 'the scaled unit and the raw unit both rendered');
  });

  it('still appends the unit when the default formatter is used', () => {
    // The complementary half. If the fix had simply stopped emitting the unit
    // span, every unformatted number on the page would silently lose its unit
    // and this suite would still be green.
    const node = renderField(
      { value: 41, state: 'measured', unit: 'req', source: 'server', observedAtMs: Date.now() },
      { label: 'Queue depth' },
    );

    assert.match(node.textContent, /41 req/);
  });

  it('keeps the unit on the placeholder branches, which never run the formatter', () => {
    // §4.1: WHICH thing is missing is itself information. "— bytes" says
    // strictly more than "—", and there is no collision to avoid here because
    // the formatter is never reached without a value.
    for (const state of ['unavailable', 'pending']) {
      const node = renderField(
        { value: null, state, unit: 'bytes', reason: 'not plumbed' },
        { format: formatGiB, label: 'VRAM' },
      );

      assert.match(
        node.textContent,
        /bytes/,
        `${state} dropped its unit — the reader can no longer tell WHAT is missing`,
      );
    }
  });

  it('makes the same decision in the accessible name as on screen', () => {
    // A screen reader announcing "VRAM: 5.35 GiB bytes" is the identical
    // defect, heard instead of seen — and it was a separate code path.
    const node = renderField(measuredBytes(), { format: formatGiB, label: 'VRAM' });

    assert.equal(node.getAttribute('aria-label'), 'VRAM: 5.35 GiB');
  });

  it('does not double a percent sign in the accessible name', () => {
    // `formatNumber` appends '%' itself, so the aria sentence was appending a
    // second one while the visible text was correct. Found while fixing the
    // scaled-unit case; the two share the suffix decision.
    const node = renderField(
      { value: 42.1, state: 'measured', unit: '%', source: 'server', observedAtMs: Date.now() },
      { label: 'Utilisation' },
    );

    assert.equal(node.getAttribute('aria-label'), 'Utilisation: 42.1%');
  });

  it('applies the same rule to describeFieldText, the table-view route', () => {
    assert.equal(
      describeFieldText('VRAM', measuredBytes(), formatGiB),
      'VRAM 5.35 GiB',
      'the accessible table view re-appended the raw unit',
    );
    assert.equal(
      describeFieldText('Queue', {
        value: 41,
        state: 'measured',
        unit: 'req',
        source: 'server',
        observedAtMs: Date.now(),
      }),
      'Queue 41 req',
      'the unformatted route must keep its unit',
    );
  });

  it('renders no doubled unit anywhere in the real panel registry', () => {
    // The acceptance that actually caught this. The unit tests above all pass
    // against a field I hand-wrote; this one mounts every REGISTERED panel with
    // its real bindings and reads the assembled text, which is the only artefact
    // a visitor sees. It reported 19 doubled renders across 56 value nodes
    // before the fix.
    const seen = { mounted: 0, values: 0 };
    const offenders = [];

    for (const panel of PANELS) {
      const root = document.createElement('div');
      const handle = panel.module.mount(root, doubleUnitProbeStore());
      seen.mounted += 1;
      flushAnimationFrames();

      const walk = (node) => {
        if (node.classList?.contains?.('value')) {
          seen.values += 1;
          const text = node.textContent.replace(/[ˢᵉᶜ~·]/g, '').trim();
          const aria = node.getAttribute?.('aria-label') ?? '';
          if (/(GiB|MiB|KiB|TiB)\s+bytes$/.test(text) || /\bs\s+ms$/.test(text)) {
            offenders.push(`${panel.id}: ${text}`);
          }
          if (/(GiB|MiB|KiB|TiB)\s+bytes|\bs\s+ms\b|%\s+%/.test(aria)) {
            offenders.push(`${panel.id} (aria): ${aria}`);
          }
        }
        (node.children ?? []).forEach(walk);
      };
      walk(root);
      handle?.unmount?.();
    }

    // Anti-vacuity. A probe that mounts nothing reports zero offenders and is
    // indistinguishable from a clean page — this exact false green cost me two
    // iterations while diagnosing the defect.
    assert.ok(seen.mounted >= 5, `only ${seen.mounted} panels mounted; the scan proves nothing`);
    assert.ok(seen.values > 20, `only ${seen.values} value nodes scanned; the scan proves nothing`);
    assert.deepEqual(offenders, [], 'a scaled unit and a raw unit rendered together');
  });
});

// A panel declares its freshness contract once, in `meta.staleCeilingMs`, and
// `bindPanel` carries it to every render call. Three declarations must stay
// distinguishable there:
//
//   omitted -> inherit the 10s default
//   null    -> "I have no freshness contract"; never withhold a number for age
//   number  -> that ceiling
//
// This guard exists because the middle one was UNIMPLEMENTABLE for a while.
// Both call sites resolved the ceiling with `?? DEFAULT_STALE_CEILING_MS`, and
// `??` fires on `null` as well as `undefined` — so a panel that explicitly
// declined a ceiling was handed a 10-second one, and the option documented in
// the panel contract could not be exercised by anybody. `prefix-cache.js` ships
// exactly that declaration, so this was live, not hypothetical; it was invisible
// only because that panel binds no telemetry after the counter cut.
//
// The rule about WHICH panels may declare `null` lives in staleness.test.js.
// This is the other half, and the half that went missing: given the
// declaration, does the renderer OBEY it. A source-level rule cannot answer
// that — the defect was two `??` operators downstream of every meta it checked.
describe('bindPanel: the declared stale ceiling reaches the renderer intact', () => {
  const NOW = 1_000_000;
  // Old enough to be past the default ceiling, so "opted out" and "inherited
  // the default" cannot accidentally agree.
  const staleField = {
    state: 'stale',
    value: 42,
    unit: 'req/s',
    source: 'server',
    observedAtMs: NOW - 60_000,
  };

  const render = (meta) => bindPanel(meta).renderField(staleField, { nowMs: NOW });

  it('honours a null ceiling: the number survives its own age', () => {
    const row = render({ staleCeilingMs: null });
    assert.equal(row.getAttribute('data-state'), 'stale');
    assert.notEqual(
      row.getAttribute('data-stale'),
      'expired',
      'a panel that declared no ceiling had its number withheld for being past one',
    );
    assert.match(row.textContent, /42/, 'the value vanished from a panel with no ceiling');
  });

  it('still tells the reader how old it is — opting out withholds nothing', () => {
    // The escape hatch must not become a way to launder a stale number into a
    // fresh-looking one. No ceiling means the NUMBER stays, not that the age
    // goes away.
    const row = render({ staleCeilingMs: null });
    assert.match(row.textContent, /1m 0s old/);
    assert.equal(row.getAttribute('data-stale'), 'true');
  });

  it('an omitted ceiling still inherits the default — positive control', () => {
    // Without this the suite above would pass just as well against a renderer
    // that had lost the ceiling machinery altogether.
    const row = render({});
    assert.equal(row.getAttribute('data-stale'), 'expired');
    assert.doesNotMatch(row.textContent, /42/);
  });

  it('null and omitted must not render the same row', () => {
    // The single assertion that fails against the original `??` defect: under
    // it, these two were byte-identical.
    assert.notEqual(
      render({ staleCeilingMs: null }).textContent,
      render({}).textContent,
      'the opt-out is indistinguishable from the default — `??` is collapsing null into it',
    );
  });

  it('a declared numeric ceiling is obeyed', () => {
    assert.equal(render({ staleCeilingMs: 3_000 }).getAttribute('data-stale'), 'expired');
    assert.notEqual(render({ staleCeilingMs: 120_000 }).getAttribute('data-stale'), 'expired');
  });

  it('the documented type admits the value a shipping panel passes', () => {
    // The JSDoc typed this `{staleCeilingMs?: number}` while prefix-cache.js
    // passed `null`. A declared type that excludes a live call site is a lie
    // told to every editor and every reader.
    const source = readFileSync(new URL('./panel-kit.js', import.meta.url), 'utf8');
    const signature = source.match(/@param \{\{staleCeilingMs\?:[^}]*\}\} panelMeta/);
    assert.ok(signature, 'the bindPanel @param for panelMeta moved or was renamed');
    assert.match(
      signature[0],
      /null/,
      'bindPanel is documented as taking only a number, but a shipping panel declares null',
    );
  });
});
