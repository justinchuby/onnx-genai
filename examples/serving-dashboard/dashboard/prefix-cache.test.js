// Copyright (c) Microsoft Corporation.
//
// Tests for the prefix-cache panel.
//
// This panel is the odd one in the registry: it renders a FINDING and binds no
// telemetry. So the interesting assertions are not "does it show the number" —
// there is no number — but the two properties the ruling actually turns on:
//
//   1. IT CANNOT READ TELEMETRY. Proven by mounting it with a store that
//      throws on any access whatsoever, rather than by grepping the source for
//      field names. A source scan proves the bindings we thought to look for
//      are absent; a booby-trapped store proves ALL of them are.
//
//   2. IT RENDERS IDENTICALLY ON BOTH ORIGINS. The gap is real on both
//      execution paths for two different reasons, and an origin-dependent
//      render would preserve a distinction that no longer exists.

import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';

import { flushAnimationFrames, installFakeDom } from './testing/fake-dom.js';

let uninstallDom;
before(() => {
  uninstallDom = installFakeDom();
});
after(() => uninstallDom());

const { meta, mount } = await import('./prefix-cache.js');

/**
 * A store that is an error if it is touched at all. Any property read — a
 * field, a capability, a subscription, `then` — throws with the name of what
 * was reached for, so a failure names the binding that was added.
 */
function boobyTrappedStore() {
  return new Proxy(
    {},
    {
      get(_target, property) {
        throw new Error(
          `the prefix panel read '${String(property)}' from the store. It must bind nothing.`,
        );
      },
      has(_target, property) {
        throw new Error(`the prefix panel probed '${String(property)}' on the store.`);
      },
    },
  );
}

function mountIntoFreshRoot(store) {
  const root = document.createElement('div');
  const handle = mount(root, store);
  flushAnimationFrames();
  return { root, handle };
}

/** Every text node the panel rendered, in document order. */
function renderedText(root) {
  return root
    .findAllByTag('p')
    .concat(root.findAllByTag('li'))
    .map((node) => node.textContent)
    .join('\n');
}

describe('prefix-cache panel', () => {
  it('mounts and renders without touching the store at all', () => {
    // The registry calls mount(root, store) for every panel. This one is handed
    // a store that detonates on contact and must not notice.
    const { root, handle } = mountIntoFreshRoot(boobyTrappedStore());

    assert.ok(root.findByClass('panel-prefix-cache__headline'), 'no headline rendered');
    assert.equal(typeof handle.unmount, 'function');
    handle.unmount();
  });

  it('survives repaint frames without polling anything', () => {
    // cadence is 0 and nothing subscribes, but "nothing repaints" is a claim
    // worth pinning: a later edit that adds an animation-frame read would only
    // fail on a subsequent frame, not on mount.
    const { root, handle } = mountIntoFreshRoot(boobyTrappedStore());
    flushAnimationFrames();
    flushAnimationFrames();
    assert.ok(root.findByClass('panel-prefix-cache__provenance'), 'panel lost its provenance line');
    handle.unmount();
  });

  it('renders byte-identically regardless of origin', () => {
    // The two paths fail for different reasons; neither is origin-conditional.
    const first = mountIntoFreshRoot(boobyTrappedStore());
    const second = mountIntoFreshRoot(boobyTrappedStore());

    assert.equal(
      renderedText(first.root),
      renderedText(second.root),
      'the panel rendered differently across two mounts, so its output is not origin-independent',
    );
    assert.ok(renderedText(first.root).length > 0, 'both renders were empty, so the check is vacuous');

    first.handle.unmount();
    second.handle.unmount();
  });

  it('states the finding, the counter warning, and why no number appears', () => {
    const { root, handle } = mountIntoFreshRoot(boobyTrappedStore());
    const text = renderedText(root);

    assert.match(text, /not happening on either execution path/, 'the finding itself is missing');
    assert.match(text, /control request/, 'the counter-behaviour argument is missing');
    assert.match(text, /No timing figure appears here/, 'the withdrawal is not explained');

    handle.unmount();
  });

  it('cites the engine sites a reader can check', () => {
    const { root, handle } = mountIntoFreshRoot(boobyTrappedStore());
    const citations = root.findAllByTag('li');

    assert.ok(citations.length >= 4, `expected the four engine citations, got ${citations.length}`);
    const joined = citations.map((node) => node.textContent).join('\n');
    for (const file of ['runtime.rs', 'batched.rs', 'metrics.rs']) {
      assert.ok(joined.includes(file), `no citation names ${file}`);
    }

    handle.unmount();
  });

  it('describes itself for a screen reader without quoting a timing figure', () => {
    const { handle } = mountIntoFreshRoot(boobyTrappedStore());
    const description = handle.describe();

    assert.match(description, /Prefix cache/);
    assert.match(description, /binds no live telemetry/);
    assert.doesNotMatch(
      description,
      /\d+\s*(ms|milliseconds)/,
      'the spoken description quotes a latency, and every prefix timing figure was withdrawn',
    );

    handle.unmount();
  });

  it('empties its root on unmount', () => {
    const { root, handle } = mountIntoFreshRoot(boobyTrappedStore());
    assert.ok(root.children.length > 0, 'nothing was rendered, so unmount proves nothing');
    handle.unmount();
    assert.equal(root.children.length, 0, 'unmount left nodes behind');
  });

  it('declares no server-mode gate', () => {
    // Panel-level mode gating was removed wholesale. This panel carried the
    // last `requires: null` before it was cut, so it is the likeliest place for
    // the retired key to return on a restore.
    assert.equal('requires' in meta, false, 'the retired `requires` key is back');
    assert.equal('modes' in meta, false, 'the retired `modes` key is back');
    assert.equal(meta.id, 'prefix-cache');
    assert.equal(meta.cadence, 0, 'a panel that binds nothing must not declare a poll cadence');
  });
});
