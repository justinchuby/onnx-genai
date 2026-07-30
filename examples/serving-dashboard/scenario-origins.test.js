// Copyright (c) Microsoft Corporation.
//
// These tests exist because every failure mode below is SILENT. A wrong origin
// does not throw — it renders another server's structural zeros as if they were
// measurements, which is indistinguishable from working.

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  SERVER_CLASSES,
  SCENARIOS,
  parseOrigin,
  resolveOrigins,
  planScenario,
  reconcileSelfClasses,
  scenarioHref,
  currentScenarioId,
} from './scenario-origins.js';

const SCATTER = 'http://127.0.0.1:8123';
const DYNAMIC = 'http://127.0.0.1:8124';

test('no port is hard-coded anywhere in the module', async () => {
  const source = await import('node:fs/promises').then((fs) =>
    fs.readFile(new URL('./scenario-origins.js', import.meta.url), 'utf8'),
  );
  // Ports live in run-demo.sh and the environment. A literal here would be a
  // copy that drifts the moment someone sets SCATTER_PORT.
  // Strip both comment forms: prose legitimately shows example URLs, and only
  // executable code can actually pin a port.
  const code = source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
  assert.equal(/\b81\d\d\b/.test(code), false, 'found a literal port in executable code');
});

test('a peer origin is never guessed when it was not provided', () => {
  const origins = resolveOrigins({
    href: `${SCATTER}/demo`,
    selfClasses: [SERVER_CLASSES.SCATTER],
  });

  assert.equal(origins.scatter, SCATTER);
  // The dangerous answer is SCATTER. Polling the scatter server for paged-KV
  // fields yields structural zeros that look exactly like measurements.
  assert.equal(origins.dynamic, null);
});

test('an unavailable scenario explains itself and names the launcher', () => {
  const origins = resolveOrigins({
    href: `${SCATTER}/demo`,
    selfClasses: [SERVER_CLASSES.SCATTER],
  });
  const plan = planScenario('paged-kv', origins, SCATTER);

  assert.equal(plan.available, false);
  assert.equal(plan.baseUrl, null);
  assert.match(plan.reason, /run-demo\.sh/);
  assert.match(plan.reason, /dynamic/);
});

test('the launcher can point the page at both servers', () => {
  const origins = resolveOrigins({
    href: `${SCATTER}/demo?scatter-origin=${encodeURIComponent(SCATTER)}&dynamic-origin=${encodeURIComponent(DYNAMIC)}`,
    selfClasses: [SERVER_CLASSES.SCATTER],
  });

  assert.equal(origins.scatter, SCATTER);
  assert.equal(origins.dynamic, DYNAMIC);

  const plan = planScenario('paged-kv', origins, SCATTER);
  assert.equal(plan.available, true);
  assert.equal(plan.baseUrl, DYNAMIC);
  // Cross-server means navigate. The server has no CORS config and needs none.
  assert.equal(plan.requiresNavigation, true);
});

test('ONE server serving both models collapses to a single origin, with no config', () => {
  // This is the topology @d7cf9b84 is investigating. It must cost nothing.
  const origins = resolveOrigins({
    href: `${SCATTER}/demo`,
    selfClasses: [SERVER_CLASSES.SCATTER, SERVER_CLASSES.DYNAMIC],
  });

  assert.equal(origins.scatter, SCATTER);
  assert.equal(origins.dynamic, SCATTER);

  for (const id of Object.keys(SCENARIOS)) {
    const plan = planScenario(id, origins, SCATTER);
    assert.equal(plan.available, true, `${id} should be available`);
    assert.equal(plan.requiresNavigation, false, `${id} should not navigate`);
  }
});

test('switching scenarios carries the topology forward', () => {
  const origins = resolveOrigins({
    href: `${SCATTER}/demo?scatter-origin=${encodeURIComponent(SCATTER)}&dynamic-origin=${encodeURIComponent(DYNAMIC)}`,
    selfClasses: [SERVER_CLASSES.SCATTER],
  });

  const href = scenarioHref('paged-kv', origins);
  const url = new URL(href);

  assert.equal(url.origin, DYNAMIC);
  // The trailing slash is load-bearing: without it relative module specifiers
  // resolve against / instead of /demo/, and every import 404s.
  assert.equal(url.pathname, '/demo/');
  assert.equal(url.searchParams.get('scenario'), 'paged-kv');
  // Without these the next switch back would find no scatter origin and the
  // demo would degrade halfway through.
  assert.equal(url.searchParams.get('scatter-origin'), SCATTER);
  assert.equal(url.searchParams.get('dynamic-origin'), DYNAMIC);
});

test('a javascript: origin is rejected rather than turned into an href', () => {
  // The page is meant to be handed around as a URL, so a link carrying a
  // hostile parameter is the realistic threat.
  assert.equal(parseOrigin('javascript:alert(1)'), null);
  assert.equal(parseOrigin('data:text/html,<script>'), null);
  assert.equal(parseOrigin('not a url'), null);
  assert.equal(parseOrigin(''), null);
  assert.equal(parseOrigin(null), null);
  assert.equal(parseOrigin('http://127.0.0.1:8124/some/path'), 'http://127.0.0.1:8124');
});

test('a hostile origin parameter cannot survive into a resolved origin', () => {
  const origins = resolveOrigins({
    href: `${SCATTER}/demo?dynamic-origin=javascript:alert(1)`,
    selfClasses: [SERVER_CLASSES.SCATTER],
  });
  assert.equal(origins.dynamic, null);
});

test('the default scenario is one THIS server can actually serve', () => {
  // A visitor landing on the dynamic server must not open into an unavailable
  // batching panel and conclude the demo is broken.
  assert.equal(
    currentScenarioId(`${DYNAMIC}/demo`, [SERVER_CLASSES.DYNAMIC]),
    'paged-kv',
  );
  assert.equal(
    currentScenarioId(`${SCATTER}/demo`, [SERVER_CLASSES.SCATTER]),
    'continuous-batching',
  );
});

test('an explicit scenario request is honoured, and a bogus one is not', () => {
  assert.equal(
    currentScenarioId(`${SCATTER}/demo?scenario=prefix-cache`, [SERVER_CLASSES.SCATTER]),
    'prefix-cache',
  );
  assert.equal(
    currentScenarioId(`${SCATTER}/demo?scenario=../../etc/passwd`, [SERVER_CLASSES.SCATTER]),
    'continuous-batching',
  );
});

test('every scenario names a real server class and says why', () => {
  const classes = new Set(Object.values(SERVER_CLASSES));
  for (const [id, scenario] of Object.entries(SCENARIOS)) {
    assert.equal(scenario.id, id, `${id} id mismatch`);
    assert.ok(classes.has(scenario.serverClass), `${id} has an unknown server class`);
    // The `why` is rendered to the visitor when the scenario is unavailable.
    assert.ok(scenario.why.length > 20, `${id} needs a real explanation`);
  }
});

// ---------------------------------------------------------------------------
// reconcileSelfClasses -- detection beats selection.
//
// The URL is an assertion; the model id is a fact. Believing a URL that
// misdescribes its server yields a fully rendered, entirely wrong dashboard:
// prefix-cache fields marked structurally-not-applicable on a server that
// genuinely measures them, and batching panels mounted where batching cannot
// happen. Nothing on screen would look broken.
// ---------------------------------------------------------------------------

test('the server overrules a URL that misdescribes it', () => {
  const result = reconcileSelfClasses({
    declared: ['scatter'],
    observed: ['dynamic'],
    observedModelId: 'qwen2.5-0.5b',
    origin: 'http://127.0.0.1:8124',
  });

  assert.deepEqual(result.classes, ['dynamic']);
  assert.ok(result.contradiction, 'a disagreement must be surfaced, not silently corrected');
  assert.match(result.contradiction, /qwen2\.5-0\.5b/);
  assert.match(result.contradiction, /scatter/);
});

test('an agreeing URL produces no notice', () => {
  const result = reconcileSelfClasses({
    declared: ['scatter'],
    observed: ['scatter'],
    observedModelId: 'qwen2.5-0.5b-scatter-v2',
    origin: 'http://127.0.0.1:8123',
  });

  assert.deepEqual(result.classes, ['scatter']);
  assert.equal(result.contradiction, null);
});

test('an unreachable server leaves the launcher declaration standing', () => {
  // Refusing the declaration here would mount nothing at all, which is worse
  // than trusting the process that bound the ports.
  const result = reconcileSelfClasses({
    declared: ['dynamic'],
    observed: [],
    observedModelId: null,
    origin: 'http://127.0.0.1:8124',
  });

  assert.deepEqual(result.classes, ['dynamic']);
  assert.equal(result.contradiction, null);
});

test('a hand-started server with no URL parameters is detected from its model', () => {
  const result = reconcileSelfClasses({
    declared: [],
    observed: ['scatter'],
    observedModelId: 'qwen2.5-0.5b-scatter-v2',
    origin: 'http://127.0.0.1:9999',
  });

  assert.deepEqual(result.classes, ['scatter']);
  assert.equal(result.contradiction, null);
});

test('a disproved declaration is reported so it can be dropped from the map', () => {
  const result = reconcileSelfClasses({
    declared: ['scatter'],
    observed: ['dynamic'],
    observedModelId: 'qwen2.5-0.5b',
    origin: 'http://127.0.0.1:8124',
  });

  // Explaining the contradiction is not enough: the scatter origin must stop
  // pointing here, or the page offers a batching tab on a server that cannot
  // batch while simultaneously explaining that it cannot.
  assert.deepEqual(result.discredited, ['scatter']);
});

test('agreement discredits nothing', () => {
  const result = reconcileSelfClasses({
    declared: ['dynamic'],
    observed: ['dynamic'],
    observedModelId: 'qwen2.5-0.5b',
    origin: 'http://127.0.0.1:8124',
  });

  assert.deepEqual(result.discredited, []);
});
