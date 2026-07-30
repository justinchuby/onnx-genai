// Copyright (c) Microsoft Corporation.
//
// Tests for the §31/D89 panel-level `not-applicable` treatment.
//
// The rule under test: when EVERY field in a panel is not-applicable, the panel
// replaces its body with one explanation instead of a column of `n/a` glyphs
// that each repeat the same sentence. When ANY field is in another state, the
// per-field treatment survives untouched.
//
// This file exists separately from panels.test.js because the behaviour is not
// a property of any one panel — it is a property of the repaint path, and it
// has to hold for a panel written next week by someone who has never read this
// file. The last two tests therefore assert it on the REAL panels rather than
// on a fixture, since a helper that works in isolation and is never reached is
// the exact failure this project keeps finding.

import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';

import { flushAnimationFrames, installFakeDom } from './testing/fake-dom.js';
import { createFakeStore, unavailable } from './testing/fake-store.js';
import {
  collapseNotApplicableBody,
  createRepaintScheduler,
  element,
  renderField,
  renderGatedEndpointNotice,
  replaceChildren,
} from './panel-kit.js';
import { PANELS } from './index.js';

let uninstallDom;
before(() => {
  uninstallDom = installFakeDom();
});
after(() => uninstallDom());

const BYPASS_REASON =
  'This engine runs continuous batching, which never consults the paged KV cache. ' +
  'ContinuousBatchManager (batched.rs:101-110) holds no reference to engine.kv_cache.';

const notApplicable = (label, reason = BYPASS_REASON) => ({
  value: null,
  state: 'not-applicable',
  source: 'server',
  unit: '',
  label,
  reason,
});

const measuredField = (label, value) => ({
  value,
  state: 'measured',
  source: 'server',
  unit: '',
  label,
  at: Date.now(),
});

/** Every element carrying a data-state, in tree order. */
function statesIn(root) {
  const found = [];
  const visit = (node) => {
    if (!node || typeof node.getAttribute !== 'function') return;
    const state = node.getAttribute('data-state');
    if (state !== null && state !== undefined) found.push(state);
    for (const child of node.children ?? []) visit(child);
  };
  visit(root);
  return found;
}

// Reads `classList`, which both the test DOM and a real one implement.
// An earlier version of this helper read `node.className` -- undefined on the
// test DOM -- so it returned 0 for every subtree and every `assert.equal(n, 0)`
// in this file passed without inspecting anything. A counting helper that
// cannot count is the same class of defect as the panel it is here to test.
function countClass(root, className) {
  let n = 0;
  const visit = (node) => {
    if (!node || typeof node.getAttribute !== 'function') return;
    if (node.classList?.contains(className)) n += 1;
    for (const child of node.children ?? []) visit(child);
  };
  visit(root);
  return n;
}

/** A panel body holding `fields`, built through the real render path. */
function panelWith(fields) {
  const root = element('div', {});
  replaceChildren(
    root,
    fields.map((field) => renderField(field)),
  );
  return root;
}

describe('panel-level not-applicable (§31/D89)', () => {
  it('collapses a wholly bypassed panel to a single explanation', () => {
    const root = panelWith([
      notApplicable('pages used'),
      notApplicable('pages total'),
      notApplicable('block size'),
    ]);
    assert.equal(statesIn(root).length, 3, 'precondition: three field wrappers');

    assert.equal(collapseNotApplicableBody(root), true);

    assert.equal(countClass(root, 'panel-bypass'), 1, 'exactly one notice');
    assert.equal(
      countClass(root, 'value__num--not-applicable'),
      0,
      'no n/a glyph survives the collapse',
    );
    assert.match(root.textContent, /Not applicable here/);
    assert.match(root.textContent, /batched\.rs:101-110/, 'the citation reaches the screen');
  });

  it('renders the reason IN FULL, not the first-sentence caption', () => {
    // The field-level treatment trims to one sentence because four reasons can
    // stack inside one card. A panel-level notice has the whole body to itself,
    // so the trim is no longer paying for anything -- and the part it drops is
    // the evidence.
    const root = panelWith([notApplicable('pages used'), notApplicable('pages total')]);
    collapseNotApplicableBody(root);

    assert.ok(
      root.textContent.includes(BYPASS_REASON),
      'the untruncated reason, including its citation, is on screen',
    );
  });

  it('does nothing to a panel with no fields at all', () => {
    // kv-memory renders its own capability notice and requests its own empty
    // state. Neither is bypassed -- both are already speaking for themselves,
    // and overwriting that would delete the better message.
    const root = element('div', {
      children: [element('p', { className: 'own-message', text: 'KV block introspection is off' })],
    });

    assert.equal(collapseNotApplicableBody(root), false);
    assert.equal(countClass(root, 'panel-bypass'), 0);
    assert.match(root.textContent, /KV block introspection is off/, 'its own message survives');
  });

  it('does nothing when a single field is still measured (the D83 mixed case)', () => {
    const root = panelWith([
      measuredField('queue depth', 3),
      notApplicable('preemptions'),
      notApplicable('swaps'),
    ]);

    assert.equal(collapseNotApplicableBody(root), false);
    assert.equal(countClass(root, 'panel-bypass'), 0);
    assert.equal(
      countClass(root, 'value__num--not-applicable'),
      2,
      'structurally pinned fields inside a live panel stay per-field',
    );
  });

  it('preserves DISTINCT reasons rather than collapsing them into one', () => {
    const other = 'Preemption is never triggered: the scheduler admits by budget (scheduler.rs:88).';
    const root = panelWith([notApplicable('a'), notApplicable('b', other), notApplicable('c')]);

    collapseNotApplicableBody(root);

    assert.equal(countClass(root, 'panel-bypass__reason'), 2, 'two distinct reasons, two lines');
    assert.ok(root.textContent.includes(BYPASS_REASON));
    assert.ok(root.textContent.includes(other));
  });

  it('says one thing once when seventeen fields carry the same sentence', () => {
    const root = panelWith(Array.from({ length: 17 }, (_, i) => notApplicable(`field ${i}`)));

    collapseNotApplicableBody(root);

    assert.equal(countClass(root, 'panel-bypass__reason'), 1);
    const occurrences = root.textContent.split('ContinuousBatchManager').length - 1;
    assert.equal(occurrences, 1, 'the sentence appears exactly once, not seventeen times');
  });

  describe('wired into the repaint path, not applied once at mount', () => {
    it('re-collapses after every repaint', () => {
      // A collapse applied at mount is erased by the next frame, and panels
      // repaint up to four times a second. That bug would be invisible for the
      // first 250ms and permanent afterwards.
      const root = element('div', {});
      let fields = [notApplicable('a'), notApplicable('b')];
      const paint = () =>
        replaceChildren(
          root,
          fields.map((field) => renderField(field)),
        );

      const scheduler = createRepaintScheduler(root, paint, {
        minIntervalMs: 0,
        telemetryStore: { getSnapshot: () => ({ endpointErrors: {} }) },
      });
      scheduler.request();
      flushAnimationFrames();
      assert.equal(countClass(root, 'panel-bypass'), 1, 'collapsed on first paint');

      scheduler.request();
      flushAnimationFrames();
      assert.equal(countClass(root, 'panel-bypass'), 1, 'still collapsed after a repaint');
      assert.equal(countClass(root, 'value__num--not-applicable'), 0);

      // The visitor switches to the server that CAN measure this. The panel
      // must come back on its own, with no transition-specific code.
      fields = [measuredField('a', 12), measuredField('b', 4)];
      scheduler.request();
      flushAnimationFrames();
      assert.equal(countClass(root, 'panel-bypass'), 0, 'the collapse reverses');
      assert.equal(statesIn(root).filter((s) => s === 'measured').length, 2);

      scheduler.cancel();
    });
  });

  describe('the real panels', () => {
    /** A store that reports every field as not-applicable, as a bypassed server does. */
    const bypassedStore = () => ({
      field: (path) => notApplicable(path),
      rate: (path) => notApplicable(path),
      series: () => ({ state: 'not-applicable', t: [], v: [], gaps: [], reason: BYPASS_REASON }),
      rateSeries() {
        return this.series();
      },
      requests: () => [],
      capability: () => false,
      connection: () => ({ state: 'ok', serverMessage: null }),
      subscribe: () => () => {},
      subscribeRequests: () => () => {},
      destroy: () => {},
    });

    // Measured on this tree BEFORE the collapse existed. These are the counts a
    // visitor would have seen, and they are why this feature exists.
    const BEFORE = { throughput: 17, system: 14, scheduling: 10 };

    for (const name of ['throughput', 'system', 'scheduling']) {
      it(`${name} collapses to one notice (was ${BEFORE[name]} n/a glyphs)`, async () => {
        const panel = await import(`./${name}.js`);
        const root = element('div', {});
        const handle = panel.mount(root, bypassedStore());
        flushAnimationFrames();

        assert.equal(countClass(root, 'panel-bypass'), 1, `${name} should collapse`);
        assert.equal(
          countClass(root, 'value__num--not-applicable'),
          0,
          `${name} should show no n/a glyphs`,
        );
        assert.ok(BEFORE[name] > 1, 'sanity: this panel really did repeat itself');

        handle.unmount();
      });
    }

    it('leaves a panel that renders its own empty state alone', async () => {
      for (const name of ['requests']) {
        const panel = await import(`./${name}.js`);
        const root = element('div', {});
        const handle = panel.mount(root, bypassedStore());
        flushAnimationFrames();

        assert.equal(
          countClass(root, 'panel-bypass'),
          0,
          `${name} binds no telemetry, so it is not bypassed`,
        );
        assert.ok(root.textContent.trim().length > 0, `${name} still says something`);

        handle.unmount();
      }
    });
  });
});

// AC62 — A MISSING FLAG MUST NOT PRESENT AS A BROKEN SERVER.
//
// The debug-gated endpoints answer 404 when the server was launched without
// --enable-debug-endpoints. 404 is the cruellest status available here: a 403
// would say "you need permission" and diagnose itself, but 404 says "this
// endpoint does not exist", which a visitor reads as a wrong URL, a stale
// build, or a broken demo — anything except "add a flag".
//
// Measured before this existed: with /v1/debug/kv 404ing, the KV panel showed
// em-dashes carrying the FIELD's reason — "not exposed to the HTTP layer" —
// which is true of the default build and is the WRONG EXPLANATION here. It
// describes an unfixable limitation when the fix is one restart.
describe('AC62 — a gated 404 names the flag and the whole command', () => {
  const gatedStore = (endpointErrors) => ({
    getSnapshot: () => ({ endpointErrors }),
  });

  const panelWithUnavailableField = () => {
    const root = document.createElement('div');
    root.append(
      element('div', {
        className: 'metric',
        attrs: { 'data-state': 'unavailable', title: 'Not exposed to the HTTP layer.' },
        children: [element('span', { className: 'value__num--unavailable', text: '—' })],
      }),
    );
    return root;
  };

  it('names the flag AND the full command, not just the flag', () => {
    const root = panelWithUnavailableField();
    const rendered = renderGatedEndpointNotice(root, {
      '/v1/debug/kv': '/v1/debug/kv is disabled. Restart with --enable-debug-endpoints.',
    });

    assert.equal(rendered, true);
    const text = root.textContent;
    assert.match(text, /--enable-debug-endpoints/, 'must name the flag');
    assert.match(text, /onnx-genai-server/, 'must show the executable');
    assert.match(text, /--model/, 'a flag with no command is a puzzle, not a fix');
    assert.match(text, /--addr/, '--addr is the flag people get wrong (--port is rejected)');
  });

  it('says nothing when no gated endpoint failed', () => {
    const root = panelWithUnavailableField();
    assert.equal(renderGatedEndpointNotice(root, {}), false);
    assert.equal(renderGatedEndpointNotice(root, undefined), false);
    assert.ok(!root.textContent.includes('--enable-debug-endpoints'));
  });

  it('stays silent on a panel that lost nothing', () => {
    // The endpoint is down, but this panel reads none of it. Repeating the
    // notice on all five panels would train the eye to skip it.
    const root = document.createElement('div');
    root.append(
      element('div', {
        className: 'metric',
        attrs: { 'data-state': 'measured' },
        children: [element('span', { className: 'value__num', text: '7' })],
      }),
    );
    assert.equal(renderGatedEndpointNotice(root, { '/v1/debug/kv': 'disabled' }), false);
  });

  it('is idempotent across repaints, not appended once per frame', () => {
    const root = panelWithUnavailableField();
    const errors = { '/v1/debug/kv': 'disabled' };
    renderGatedEndpointNotice(root, errors);
    renderGatedEndpointNotice(root, errors);
    renderGatedEndpointNotice(root, errors);

    const notices = root.children.filter((child) =>
      child.classList.contains('panel-gated-notice'),
    );
    assert.equal(notices.length, 1, 'three repaints must leave exactly one notice');
  });

  it('disappears once the server is restarted with the flag', () => {
    const root = panelWithUnavailableField();
    renderGatedEndpointNotice(root, { '/v1/debug/kv': 'disabled' });
    assert.ok(root.textContent.includes('--enable-debug-endpoints'));

    // Next poll succeeds: the notice must clear itself rather than linger as a
    // stale instruction to fix something that is no longer broken.
    renderGatedEndpointNotice(root, {});
    assert.ok(!root.textContent.includes('--enable-debug-endpoints'));
  });

  it('REFUSES to build a scheduler with no store, so a panel cannot lose the notice silently', () => {
    // A panel that forgets the store paints perfectly and drops AC62 without a
    // symptom — invisible in exactly the situation the notice exists for.
    assert.throws(
      () => createRepaintScheduler(document.createElement('div'), () => {}),
      /requires \{ telemetryStore \}/,
    );
  });

  it('MOUNTS a real panel and finds the notice, proving the repaint path is wired', () => {
    // The unit tests above call renderGatedEndpointNotice directly, so they all
    // still pass if it is never CALLED by the scheduler. Deleting the call from
    // run() was green across every one of them. This is the only test that
    // fails when the mechanism is disconnected, which is the failure most
    // likely to happen during a refactor.
    const store = createFakeStore({
      fields: {
        'kv.pages_used': unavailable('Not exposed to the HTTP layer.'),
        'kv.pages_total': unavailable('Not exposed to the HTTP layer.'),
      },
    });
    const withGate = {
      ...store,
      // The fake store models fields, not connection state; the notice reads
      // only endpointErrors, so that is all this needs to supply.
      getSnapshot: () => ({ endpointErrors: { '/v1/debug/kv': '/v1/debug/kv is disabled.' } }),
    };

    const kvPanel = PANELS.find((panel) => panel.id === 'kv-memory');
    const root = document.createElement('div');
    const handle = kvPanel.module.mount(root, withGate);
    flushAnimationFrames();

    assert.match(
      root.textContent,
      /--enable-debug-endpoints/,
      'a mounted panel must surface the flag without the panel knowing about AC62',
    );
    assert.match(root.textContent, /onnx-genai-server --model/, 'and the whole command');
    handle.unmount();
  });
});
