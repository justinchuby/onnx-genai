// Copyright (c) Microsoft Corporation.
//
// AC45 — a displayed value must carry its age the moment it stops being live.
//
// The failure this guards against is specific and easy to miss: a frozen chart
// and a saturated one are pixel-identical. Every panel can be individually
// honest and the page still lies, because honesty is enforced per FIELD while
// the failure is per CONNECTION.

import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';

import { installFakeDom } from './testing/fake-dom.js';

let uninstallDom;
before(() => {
  uninstallDom = installFakeDom();
});
after(() => uninstallDom());

const { bindPanel, renderField } = await import('./panel-kit.js');
const { DEFAULT_STALE_CEILING_MS, ageMsOf, formatAge, isPastStaleCeiling, renderStateOf } =
  await import('./field-state.js');
const { adaptStore } = await import('./store-adapter.js');

/** @param {number} ageMs */
const staleField = (ageMs, extra = {}) => ({
  value: 41,
  state: 'stale',
  source: 'server',
  unit: 'requests',
  label: 'Queue depth',
  observedAtMs: Date.now() - ageMs,
  ...extra,
});

describe('AC45(a) — age in text, never colour alone', () => {
  it('states the age in words next to the value', () => {
    const node = renderField(staleField(4000), { staleCeilingMs: 30_000 });
    assert.match(node.findByClass('value__stale').textContent, /4s old/);
  });

  it('announces the age to a screen reader, not only to the pixels', () => {
    // "41" announced bare, while the screen reads "41 · 12s old", hands the
    // number to a screen-reader user stripped of the qualifier that makes it
    // honest.
    const node = renderField(staleField(12_000), { staleCeilingMs: 30_000 });
    assert.match(node.getAttribute('aria-label'), /stale/i);
    assert.match(node.getAttribute('aria-label'), /12s old/);
  });

  it('says the age is unknown rather than implying it is zero', () => {
    assert.equal(ageMsOf({ state: 'stale' }), null);
    assert.equal(formatAge(null), 'age unknown');
    // Zero would read as "observed just now" — the most dangerous thing we
    // could say about a value we cannot date.
    assert.notEqual(formatAge(null), '0s old');
  });
});

describe('the value and its age must not fuse', () => {
  it('separates them with a real character, not just a flex gap', () => {
    // The ruled treatment is `41 · 12s old`. The two spans were previously held
    // apart only by CSS gap, so the screen looked correct while textContent
    // read "4112s old" — and textContent is what the table view, describe()
    // and copy-paste actually consume. A fused figure is worse than an
    // unqualified one: 4112 is a number nobody measured.
    const node = renderField(staleField(12_000), { staleCeilingMs: 30_000 });

    // The unit and source badge sit between the two, so this asserts the age
    // is introduced by the separator and — the part that matters — that no
    // digit of the value ever touches a digit of the age.
    assert.match(node.textContent, /^41\D+·\s*12s old/);
    assert.doesNotMatch(node.textContent, /\d\d\d\d/, 'value and age fused into one number');
  });

  it('separates the em-dash from the age once the value is withheld', () => {
    const node = renderField(staleField(20_000), { staleCeilingMs: 3000 });

    assert.match(node.textContent, /—\s*·\s*20s old/);
    assert.doesNotMatch(node.textContent, /\d\d\d/, 'withheld value left digits fused to the age');
  });
});

describe('the timestamp property the ruling wrote down', () => {
  it('accepts `at` as well as `observedAtMs`, since the two contracts disagree', () => {
    // telemetry-field.js and CONTRACT.md say observedAtMs; the lead's ruled
    // envelope says `at`. Accepting both means neither spelling loses an age.
    const { observedAtMs, ...rest } = staleField(4000);
    const node = renderField({ ...rest, at: observedAtMs }, { staleCeilingMs: 30_000 });

    assert.match(node.textContent, /4s old/);
  });

  it('still refuses to date a value from an unrecognised property', () => {
    // The original bug was a Date.now() fallback that claimed every stale
    // value had just been observed. Unknown property must still withhold.
    const { observedAtMs, ...rest } = staleField(4000);
    const node = renderField({ ...rest, lastSeen: observedAtMs }, { staleCeilingMs: 30_000 });

    assert.doesNotMatch(node.textContent, /41/, 'dated the value from a property nobody publishes');
  });
});

describe('AC45(b) — past the ceiling it stops being a number', () => {
  it('withholds the number once the value is too old', () => {
    const node = renderField(staleField(20_000), { staleCeilingMs: 3000 });
    assert.doesNotMatch(node.textContent, /41/, 'an expired value was still shown as a number');
    assert.equal(node.getAttribute('data-stale'), 'expired');
  });

  it('keeps showing the age after withholding the number', () => {
    // WHY it went away is the useful part; a bare em-dash would lose it.
    const node = renderField(staleField(20_000), { staleCeilingMs: 3000 });
    assert.match(node.textContent, /20s old/);
    assert.match(node.getAttribute('title'), /past this panel's 3s limit/);
  });

  it('still shows a value that is stale but within the ceiling', () => {
    const node = renderField(staleField(2000), { staleCeilingMs: 3000 });
    assert.match(node.textContent, /41/);
    assert.equal(node.getAttribute('data-stale'), 'true');
  });

  it('treats an undateable stale value as past any ceiling', () => {
    // If we cannot say how old it is, we cannot claim it is recent enough.
    assert.equal(isPastStaleCeiling({ state: 'stale', value: 1 }, 60_000), true);
  });

  it('never withholds a value that is not stale at all', () => {
    assert.equal(isPastStaleCeiling({ state: 'ok', value: 1, observedAtMs: 0 }, 1), false);
  });
});

describe('AC45(c) — the ceiling is per-panel, not global', () => {
  it('lets a fast panel expire a value a slow panel still shows', async () => {
    const scheduling = await import('./scheduling.js');
    const system = await import('./system.js');

    assert.ok(
      scheduling.meta.staleCeilingMs < system.meta.staleCeilingMs,
      'a 4Hz occupancy panel must not tolerate the same age as static model identity',
    );

    const field = staleField(8000);
    const fast = bindPanel(scheduling.meta).renderField(field);
    const slow = bindPanel(system.meta).renderField(field);

    assert.doesNotMatch(fast.textContent, /41/, 'the fast panel should have expired this value');
    assert.match(slow.textContent, /41/, 'the slow panel should still show it');
  });

  it('gives every panel an explicit ceiling rather than inheriting the default', async () => {
    const { PANELS } = await import('./index.js');
    for (const panel of PANELS) {
      const { staleCeilingMs, cadence } = panel.module.meta;

      // A panel that polls nothing has no ceiling to declare — staleness is a
      // property of a value that stopped arriving, and this panel has no
      // values. `null` is the explicit, honest declaration for that case;
      // inventing a number would imply a freshness contract it does not have.
      // The escape hatch is gated on `cadence === 0` so it cannot be used to
      // silently drop the ceiling from a panel that DOES poll.
      if (cadence === 0) {
        assert.equal(
          staleCeilingMs,
          null,
          `${panel.id} polls nothing, so its ceiling must be explicitly null`,
        );
        continue;
      }

      assert.equal(
        typeof staleCeilingMs,
        'number',
        `${panel.id} has no declared stale ceiling`,
      );
    }
  });

  it('falls back to the default when a panel declares nothing', () => {
    const { renderField: unbound } = bindPanel({});
    const node = unbound(staleField(DEFAULT_STALE_CEILING_MS + 5000));
    assert.equal(node.getAttribute('data-stale'), 'expired');
  });
});

describe('AC45(d) — a whole-origin stall marks every panel', () => {
  it('pins the connection vocabulary it depends on', async () => {
    // markStalledOrigin fails OPEN: if these names drift, every value keeps
    // rendering as live through a total outage and nothing looks wrong. The
    // field vocabulary already changed under this code once mid-session, so
    // this assertion is not hypothetical.
    const { CONNECTION_STATES } = await import('../telemetry-store.js');
    assert.deepEqual(
      Object.fromEntries(Object.entries(CONNECTION_STATES)),
      {
        CONNECTING: 'connecting',
        CONNECTED: 'connected',
        UNREACHABLE: 'unreachable',
        NO_MODEL: 'no-model',
      },
      'the store changed its connection vocabulary; markStalledOrigin needs review',
    );
  });

  /** @param {string} connectionState */
  function storeWithConnection(connectionState) {
    const snapshot = {
      timestampMs: Date.now(),
      fields: {},
      connection: { state: connectionState },
      endpointErrors: {},
    };
    return {
      field: () => ({
        value: 41,
        state: 'measured',
        source: 'server',
        label: 'Queue depth',
        observedAtMs: Date.now() - 500,
      }),
      subscribe: () => () => {},
      getSnapshot: () => snapshot,
    };
  }

  it('downgrades a live-looking field when the origin stopped answering', () => {
    // The store only marks the field whose poll failed. A field whose endpoint
    // was not polled this cycle keeps state 'measured' and goes on looking live.
    const adapter = adaptStore(storeWithConnection('unreachable'));
    const field = adapter.field('queue.depth');
    assert.equal(renderStateOf(field), 'stale');
    assert.match(field.reason, /stopped answering/);
    adapter.destroy();
  });

  it('explains a loaded-but-modelless server differently from a dead one', () => {
    const adapter = adaptStore(storeWithConnection('no-model'));
    assert.match(adapter.field('queue.depth').reason, /no model loaded/);
    adapter.destroy();
  });

  it('leaves fields alone while the origin is healthy', () => {
    const adapter = adaptStore(storeWithConnection('connected'));
    assert.equal(renderStateOf(adapter.field('queue.depth')), 'measured');
    adapter.destroy();
  });

  it('preserves the original observation time through the downgrade', () => {
    // Re-dating the field on stall would restart its age at zero, which is
    // precisely the lie AC45 exists to prevent.
    const store = storeWithConnection('unreachable');
    const observedAtMs = store.field().observedAtMs;
    const adapter = adaptStore(store);
    assert.equal(adapter.field('queue.depth').observedAtMs, observedAtMs);
    adapter.destroy();
  });
});

describe('AC43 — "not applicable" is not "unavailable"', () => {
  it('renders n/a rather than an em-dash, and never promises a later value', async () => {
    // Continuous batching and paged KV are mutually exclusive, so on the
    // batching origin the KV metrics are not late — they are unreachable by
    // construction. "Not measurable yet" would promise a value that can never
    // arrive, and the mutual-exclusivity story is the most technically
    // interesting thing the demo has to say.
    const { notApplicableField } = await import('../telemetry-field.js');
    const field = notApplicableField(
      'ContinuousBatchManager never touches engine.kv_cache on this server.',
      { label: 'KV pages used' },
    );

    const node = renderField(field, { label: 'KV pages used' });
    assert.equal(node.getAttribute('data-state'), 'not-applicable');
    assert.match(node.textContent, /^n\/a/);

    // THE REASON MUST BE ON SCREEN, NOT BEHIND A HOVER (demo-ux.md §17).
    // A bare `n/a` is indistinguishable from a broken panel, and under two
    // servers this state is the NORMAL case rather than the exception — so
    // hiding the explanation renders the demo's single most interesting
    // finding as a dashboard half-covered in apparent breakage. A fact nobody
    // hovers over is also a fact nobody learns: it is invisible in a
    // screenshot, on a projector, and on touch.
    assert.match(
      node.textContent,
      /never touches engine\.kv_cache/,
      'the not-applicable explanation must render on screen, not only in a tooltip',
    );
    assert.match(node.getAttribute('aria-label'), /not applicable here/);
    assert.match(node.getAttribute('aria-label'), /never touches engine\.kv_cache/);
    assert.doesNotMatch(
      node.getAttribute('aria-label'),
      /yet|waiting|loading/i,
      'this wording promises a value that cannot arrive',
    );
  });

  it('is not renderable as a number', async () => {
    const { isRenderable } = await import('./field-state.js');
    const { notApplicableField } = await import('../telemetry-field.js');
    assert.equal(isRenderable(notApplicableField('structurally unreachable')), false);
  });

  it('does not silently collapse into unavailable', () => {
    assert.equal(renderStateOf({ state: 'not-applicable', value: null }), 'not-applicable');
  });
});
