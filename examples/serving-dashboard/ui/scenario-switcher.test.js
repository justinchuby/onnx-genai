// Copyright (c) Microsoft Corporation.
//
// The switcher is the control whose entire job is to enumerate what this
// product offers, which makes it the one place a silent substitution does the
// most damage: a visitor who followed a link to a scenario we cut is handed a
// DIFFERENT scenario, rendered perfectly, every field correctly badged.
//
// Every honesty mechanism in this dashboard operates on a FIELD. None of them
// operate on a ROUTE. So a resolver that reports a substitution and a page that
// never displays it are indistinguishable from the visitor's seat -- which is
// why these assertions read the DOM rather than the return value.

import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';

import { installFakeDom } from '../dashboard/testing/fake-dom.js';

let uninstallDom;
before(() => {
  uninstallDom = installFakeDom();
});
after(() => uninstallDom());

const SCATTER = 'http://127.0.0.1:8123';
const DYNAMIC = 'http://127.0.0.1:8124';

/** @returns {Promise<{mountScenarioSwitcher: Function, resolveScenario: Function}>} */
async function load() {
  const { mountScenarioSwitcher } = await import('./scenario-switcher.js');
  const { resolveScenario } = await import('../scenario-origins.js');
  return { mountScenarioSwitcher, resolveScenario };
}

function mountWith(mount, substitution) {
  const root = document.createElement('div');
  mount(root, {
    origins: { scatter: SCATTER, dynamic: DYNAMIC },
    currentScenarioId: 'continuous-batching',
    currentOrigin: SCATTER,
    substitution,
  });
  return root;
}

describe('the switcher states a scenario substitution to the visitor', () => {
  it('renders a notice naming the rejected id and what is shown instead', async () => {
    const { mountScenarioSwitcher, resolveScenario } = await load();
    const { substitution } = resolveScenario(`${SCATTER}/demo?scenario=prefix-cache`, ['scatter']);

    const root = mountWith(mountScenarioSwitcher, substitution);
    const notice = root.findByClass('scenario-switcher__substitution');

    assert.ok(notice, 'the substitution never reached the page');
    assert.match(notice.textContent, /prefix-cache/, 'must quote what the visitor asked for');
    assert.match(notice.textContent, /Continuous batching/, 'must name what it showed instead');
    assert.equal(notice.dataset.substitution, 'cut');
  });

  it('announces it, because the visitor is looking at the panels and not at this', async () => {
    const { mountScenarioSwitcher, resolveScenario } = await load();
    const { substitution } = resolveScenario(`${SCATTER}/demo?scenario=prefix-cache`, ['scatter']);

    const notice = mountWith(mountScenarioSwitcher, substitution).findByClass('scenario-switcher__substitution');

    // `alert` rather than `status`: this contradicts an action the visitor just
    // took. A polite live region waits for a pause that a page loading fresh
    // telemetry never has.
    assert.equal(notice.getAttribute('role'), 'alert');
  });

  it('renders NOTHING when we showed what was asked for', async () => {
    // The anti-vacuity control. A notice that is always present is wallpaper,
    // and wallpaper is what everyone learns to stop reading -- at which point
    // the honest version of this notice is invisible too.
    const { mountScenarioSwitcher } = await load();
    const root = mountWith(mountScenarioSwitcher, null);

    assert.equal(root.findByClass('scenario-switcher__substitution'), null);
    assert.ok(
      root.findByClass('scenario-switcher__tabs'),
      'positive control: the switcher must actually have rendered',
    );
  });

  it('keeps the substitution notice separate from the contradiction notice', async () => {
    // They state different facts -- one says the URL misdescribes the server,
    // the other says we did not render what you asked for. Merging them into
    // one channel is how a reader learns to distrust both.
    const { mountScenarioSwitcher, resolveScenario } = await load();
    const { substitution } = resolveScenario(`${SCATTER}/demo?scenario=nope`, ['scatter']);

    const root = document.createElement('div');
    mountScenarioSwitcher(root, {
      origins: { scatter: SCATTER, dynamic: DYNAMIC },
      currentScenarioId: 'continuous-batching',
      currentOrigin: SCATTER,
      contradiction: 'This URL calls this server dynamic; it is a static-cache build.',
      substitution,
    });

    const sub = root.findByClass('scenario-switcher__substitution');
    const contra = root.findByClass('scenario-switcher__contradiction');
    assert.ok(sub && contra, 'both notices must survive together');
    assert.notEqual(sub.textContent, contra.textContent);
    assert.equal(sub.dataset.substitution, 'unknown', 'a typo is not a cut');
  });

  it('escapes nothing into markup, because the id comes off the query string', async () => {
    const { mountScenarioSwitcher, resolveScenario } = await load();
    const hostile = '<img src=x onerror=alert(1)>';
    const { substitution } = resolveScenario(
      `${SCATTER}/demo?scenario=${encodeURIComponent(hostile)}`,
      ['scatter'],
    );

    const notice = mountWith(mountScenarioSwitcher, substitution).findByClass('scenario-switcher__substitution');

    assert.equal(notice.children.length, 0, 'the rejected id was parsed as markup');
    assert.match(notice.textContent, /img src=x/, 'and it is still quoted back as text');
  });
});
