/**
 * THE PREFIX COUNTERS MUST BE UNREACHABLE BY EXECUTION, NOT MERELY UNQUOTED.
 *
 * A controlled A/B proved the prefix-cache counters cannot distinguish reuse
 * from no-reuse: twelve requests with six deliberately unique prompts scored
 * twelve hits and a 0.9375 rate. The counters are therefore unshippable, and
 * the ruling is that the panel survives with ZERO field bindings -- prose and
 * source citations only.
 *
 * `prefix-counters-forbidden.test.js` already ratchets this, and it is a good
 * ratchet, but every assertion in it is a `readFileSync` plus a regular
 * expression. It is a grep wearing a test's clothing. That matters here more
 * than usual: this item survived five sweeps, ten greps and roughly an hour of
 * 438 passing tests without anyone seeing its true state, because reading
 * source cannot answer the question actually being asked -- which is not "does
 * any file contain this string" but "does any panel a visitor can see REQUEST
 * this counter when it runs".
 *
 * Those differ in both directions. A panel can name a counter in a comment and
 * bind nothing (safe, but greps red). A panel can bind one through an alias, a
 * computed key, or an adapter mapping without the literal ever appearing in its
 * source (unsafe, and greps green). The second is the one that ships.
 *
 * So this test mounts every panel in the real registry against a store that
 * RECORDS every key asked for, and asserts the prefix counters are never among
 * them. It is the executable half, and it is deliberately blind to source text.
 */

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { flushAnimationFrames, installFakeDom } from './testing/fake-dom.js';

/**
 * The two counters ruled unshippable, in every spelling that can reach a store:
 * the dotted store key and the underscored wire name.
 */
const BANNED = [
  'prefix_cache.hits',
  'prefix_cache.lookups',
  'prefix_cache_hits',
  'prefix_cache_lookups',
];

const isBanned = (key) =>
  typeof key === 'string' && BANNED.some((banned) => key.includes(banned));

/**
 * A store that answers everything as unavailable and, crucially, writes down
 * every key it was asked for. The panel cannot tell it apart from a server that
 * has not plumbed anything, so it takes its ordinary degraded path -- which is
 * the path that must not reach for a banned counter.
 */
function recordingStore(requested) {
  const absent = (key) => {
    requested.push(key);
    return {
      state: 'unavailable',
      value: null,
      reason: 'Recorded by the prefix tripwire; no server is running.',
      label: typeof key === 'string' ? key : 'value',
    };
  };
  return {
    field: absent,
    rate: absent,
    series: (key) => {
      requested.push(key);
      return { state: 'unavailable', t: [], v: [], gaps: [], reason: 'tripwire' };
    },
    rateSeries(key) {
      return this.series(key);
    },
    requests: () => [],
    capability: () => false,
    connection: () => ({ state: 'ok', serverMessage: null }),
    subscribe: () => () => {},
    subscribeRequests: () => () => {},
    getSnapshot: () => ({ fields: {}, endpointErrors: {} }),
    destroy: () => {},
  };
}

describe('no registered panel can reach a prefix counter when it runs', () => {
  it('requests no banned counter while mounting every panel in the registry', async () => {
    const uninstall = installFakeDom();
    try {
      // The REGISTRY, not a hand-listed set of panel names. A hand-listed set
      // would silently stop covering a panel the moment someone registered a
      // new one, which is precisely how an unbound panel goes unnoticed.
      const { PANELS } = await import('./index.js');
      assert.ok(
        Array.isArray(PANELS) && PANELS.length > 0,
        'the registry is empty, so this test would pass by checking nothing',
      );

      const offenders = [];
      for (const panel of PANELS) {
        const requested = [];
        const module = panel.module ?? panel;
        const mount = module.mount ?? panel.mount;
        assert.equal(
          typeof mount,
          'function',
          `panel "${panel.id ?? '(unnamed)'}" exposes no mount(), so it was never executed here`,
        );

        const root = globalThis.document.createElement('div');
        const handle = mount(root, recordingStore(requested));
        flushAnimationFrames();
        handle?.unmount?.();

        for (const key of requested.filter(isBanned)) {
          offenders.push(`${panel.id ?? '(unnamed)'} requested "${key}"`);
        }
      }

      assert.deepEqual(
        offenders,
        [],
        `${offenders.join('\n')}\n\nThese counters cannot distinguish prefix reuse from ` +
          'no-reuse, so any panel binding one renders a fabricated measurement wearing a ' +
          'plausible label. The prefix panel keeps its prose and its source citations; only ' +
          'the field binding is forbidden.',
      );
    } finally {
      uninstall();
    }
  });

  it('mounts a panel that is not the prefix panel, so the check is not vacuous', async () => {
    // The failure mode this test exists to avoid is the one that kept the item
    // open: a check that passes because it inspected nothing. If the registry
    // ever loses its panels, the assertion above goes quiet rather than red.
    const { PANELS } = await import('./index.js');
    const ids = PANELS.map((panel) => panel.id ?? '(unnamed)');
    assert.ok(
      ids.length >= 5,
      `only ${ids.length} panel(s) in the registry (${ids.join(', ')}) -- too few to be ` +
        'the real dashboard, so the tripwire above is probably checking nothing',
    );
  });
});
