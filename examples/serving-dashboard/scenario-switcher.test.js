// Copyright (c) Microsoft Corporation.
//
// Tests for the scenario switcher.
//
// The bug these exist to prevent is not a rendering bug. It is that
// planScenario computed `requiresNavigation`, nothing consumed it, and the
// demo shipped with no route to the second server at all. So the assertions
// below are mostly about REACHABILITY rather than markup.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import { SCENARIOS, planScenario } from './scenario-origins.js';
import { describeSwitcher } from './ui/scenario-switcher.js';

const SCATTER = 'http://127.0.0.1:8123';
const DYNAMIC = 'http://127.0.0.1:8124';

const BOTH = { scatter: SCATTER, dynamic: DYNAMIC };
const SCATTER_ONLY = { scatter: SCATTER, dynamic: null };

/**
 * @param {Record<string, string|null>} origins
 * @param {string} currentOrigin
 */
function planAll(origins, currentOrigin) {
  const plans = Object.keys(SCENARIOS).map((id) => ({
    id,
    plan: planScenario(id, origins, currentOrigin),
  }));
  return {
    reachable: plans.filter(({ plan }) => plan.available),
    unreachable: plans.filter(({ plan }) => !plan.available),
  };
}

test('every scenario is reachable when both servers are configured', () => {
  const { reachable, unreachable } = planAll(BOTH, SCATTER);

  assert.equal(unreachable.length, 0);
  assert.equal(reachable.length, Object.keys(SCENARIOS).length);
});

test('scenarios on the other server are marked as requiring navigation', () => {
  const { reachable } = planAll(BOTH, SCATTER);

  const remote = reachable.filter(({ plan }) => plan.requiresNavigation).map(({ id }) => id);

  // Everything backed by the dynamic server, viewed from the scatter server.
  assert.deepEqual(remote.sort(), ['memory-pressure', 'paged-kv', 'prefix-cache']);
});

test('a scenario served by this very origin never navigates', () => {
  const { reachable } = planAll(BOTH, DYNAMIC);

  const local = reachable.filter(({ plan }) => !plan.requiresNavigation).map(({ id }) => id);
  assert.deepEqual(local.sort(), ['memory-pressure', 'paged-kv', 'prefix-cache']);
});

test('an unconfigured peer makes its scenarios unreachable rather than broken links', () => {
  const { reachable, unreachable } = planAll(SCATTER_ONLY, SCATTER);

  assert.deepEqual(
    reachable.map(({ id }) => id),
    ['continuous-batching'],
  );
  assert.equal(unreachable.length, 3);

  // Each carries a reason naming the launcher, because the visitor's fix is to
  // start the demo properly rather than to hunt for a port.
  for (const { plan } of unreachable) {
    assert.match(plan.reason, /run-demo\.sh/);
  }
});

test('describe() names the current scenario and where the others live', () => {
  const { reachable, unreachable } = planAll(BOTH, SCATTER);
  const sentence = describeSwitcher(reachable, unreachable, 'continuous-batching');

  assert.match(sentence, /Continuous batching/);
  assert.match(sentence, /other server/);
  assert.doesNotMatch(sentence, /unavailable/);
});

test('describe() reports unreachable scenarios as a server that is not running', () => {
  const { reachable, unreachable } = planAll(SCATTER_ONLY, SCATTER);
  const sentence = describeSwitcher(reachable, unreachable, 'continuous-batching');

  assert.match(sentence, /3 more are unavailable/);
  assert.match(sentence, /not running/);
});

test('describe() does not claim a scenario when none is reachable', () => {
  const { reachable, unreachable } = planAll({ scatter: null, dynamic: null }, SCATTER);
  const sentence = describeSwitcher(reachable, unreachable, 'continuous-batching');

  assert.match(sentence, /No scenarios are reachable/);
});

// THE MISSING-CSS GUARD. The switcher shipped with NO stylesheet rules at all,
// so the label and the "on the dynamic server" hint rendered glued together as
// "Paged KV block tableon the dynamic server". Every assertion above passed,
// because textContent has no concept of whitespace. In a project with no build
// step nothing else checks that a class the JS emits actually exists in CSS.
test('every class the switcher emits has a rule in shell.css', () => {
  const dir = new URL('./', import.meta.url);
  const source = readFileSync(fileURLToPath(new URL('./ui/scenario-switcher.js', dir)), 'utf8');
  const css = readFileSync(fileURLToPath(new URL('./styles/shell.css', dir)), 'utf8');

  const emitted = new Set(
    [...source.matchAll(/'(scenario-switcher(?:__[a-z-]+)?)'/g)].map((m) => m[1]),
  );
  assert.ok(emitted.size >= 6, `expected several classes, found ${emitted.size}`);

  const missing = [...emitted].filter(
    // The block itself is only a container; it needs no rule of its own.
    (cls) => cls !== 'scenario-switcher' && !css.includes(`.${cls}`),
  );
  assert.deepEqual(missing, [], `classes emitted by the switcher with no CSS rule: ${missing}`);
});

// THE DRIFT GUARD. index.html previously hardcoded `paged-kv-block-table` and
// `prefix-caching`, neither of which is a registry id. Both looked plausible.
// Nothing failed, because nothing read them -- which is precisely how they
// stayed wrong. This test fails the moment the markup names a scenario again.
test('index.html does not name scenario ids, which only the registry may do', () => {
  const html = readFileSync(fileURLToPath(new URL('./index.html', import.meta.url)), 'utf8');

  const scenarioAttributes = [...html.matchAll(/data-scenario="([^"]*)"/g)].map((m) => m[1]);
  assert.deepEqual(
    scenarioAttributes,
    [],
    'index.html must not enumerate scenarios; the switcher renders them from SCENARIOS',
  );

  assert.ok(
    html.includes('id="scenario-switcher"'),
    'index.html must provide the switcher mount point',
  );
});
