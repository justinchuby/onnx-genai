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
  resolveScenario,
  describeSubstitution,
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
  const HOST = '127.0.0.1';
  assert.equal(parseOrigin('javascript:alert(1)', HOST), null);
  assert.equal(parseOrigin('data:text/html,<script>', HOST), null);
  assert.equal(parseOrigin('not a url', HOST), null);
  assert.equal(parseOrigin('', HOST), null);
  assert.equal(parseOrigin(null, HOST), null);
  assert.equal(parseOrigin('http://127.0.0.1:8124/some/path', HOST), 'http://127.0.0.1:8124');
});

test('a well-formed origin on ANOTHER host is rejected', () => {
  // The scheme check passes here. This is the gap it left: a third-party
  // origin would be polled and its numbers rendered under our chrome, each
  // one carrying the provenance badge that says a named server measured it.
  // The values would be real, correctly labelled, and somebody else's --
  // which is why no value-level honesty check can detect this.
  const HOST = '127.0.0.1';
  assert.equal(parseOrigin('http://evil.example', HOST), null);
  assert.equal(parseOrigin('https://evil.example:8124', HOST), null);
  // A hostname that merely CONTAINS ours must not pass either.
  assert.equal(parseOrigin('http://127.0.0.1.evil.example', HOST), null);
  assert.equal(parseOrigin('http://evil.example/?x=127.0.0.1', HOST), null);
  // A stranger that BEGINS with a loopback name is still a stranger. This is
  // the arm the loopback widening below could plausibly have broken.
  assert.equal(parseOrigin('http://localhost.evil.example', HOST), null);
});

test('loopback spellings name ONE machine, and the peer must survive all of them', () => {
  // C13. The launcher prints 127.0.0.1 (BIND_HOST); operators type localhost.
  // Comparing spellings as strings dropped the peer origin, and the failure
  // was invisible: the panel rendered a correct, truthful `unavailable` for a
  // server that was up and answering 200 on the same machine. A true statement
  // concealing a broken system is the one defect our honesty layer certifies
  // instead of catching -- nobody debugs a panel behaving as designed.
  for (const page of ['127.0.0.1', 'localhost', '::1', '[::1]']) {
    for (const raw of [
      'http://127.0.0.1:8133',
      'http://localhost:8133',
      'http://[::1]:8133',
    ]) {
      assert.equal(
        parseOrigin(raw, page),
        new URL(raw).origin,
        `loopback ${raw} must be reachable from a page at ${page}`,
      );
    }
    // BOTH ARMS IN ONE TEST. Without this the widening is untested in the only
    // direction that can hurt us: a third-party host is not loopback under any
    // spelling, so it stays rejected from every loopback page.
    assert.equal(parseOrigin('http://evil.example.com:8133', page), null);
    assert.equal(parseOrigin('https://attacker.test/', page), null);
    assert.equal(parseOrigin('javascript:alert(1)', page), null);
  }
});

test('the widening does NOT make non-loopback hosts interchangeable', () => {
  // Exact match is still the rule everywhere off loopback: two real hosts that
  // happen to be the same machine are not our problem and must not be guessed.
  assert.equal(parseOrigin('http://demo-b.internal:8133', 'demo-a.internal'), null);
  assert.equal(
    parseOrigin('http://demo-a.internal:8133', 'demo-a.internal'),
    'http://demo-a.internal:8133',
  );
});

test('a differing PORT on our own host is allowed -- that is the real topology', () => {
  const HOST = '127.0.0.1';
  assert.equal(parseOrigin('http://127.0.0.1:9999', HOST), 'http://127.0.0.1:9999');
});

test('parseOrigin THROWS when given no hostname to compare against', () => {
  // Fail closed. A caller that omits it would otherwise silently get the old
  // scheme-only behaviour -- a security check that fails OPEN is
  // indistinguishable from one that passed.
  assert.throws(() => parseOrigin('http://evil.example'), TypeError);
  assert.throws(() => parseOrigin('http://evil.example', ''), TypeError);
});

test('a hostile origin parameter cannot survive into a resolved origin', () => {
  const origins = resolveOrigins({
    href: `${SCATTER}/demo?dynamic-origin=javascript:alert(1)`,
    selfClasses: [SERVER_CLASSES.SCATTER],
  });
  assert.equal(origins.dynamic, null);
});

test('a third-party origin parameter cannot survive into a resolved origin', () => {
  // The end-to-end version of the check above: this is the URL an attacker
  // would actually hand a visitor.
  const origins = resolveOrigins({
    href: `${SCATTER}/demo?dynamic-origin=${encodeURIComponent('http://evil.example:8124')}`,
    selfClasses: [SERVER_CLASSES.SCATTER],
  });
  // null, not the hostile origin, and NOT silently swapped for our own origin
  // either -- guessing the peer is here is the other silent failure this
  // module exists to prevent.
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
    currentScenarioId(`${SCATTER}/demo?scenario=paged-kv`, [SERVER_CLASSES.SCATTER]),
    'paged-kv',
  );
  assert.equal(
    currentScenarioId(`${SCATTER}/demo?scenario=../../etc/passwd`, [SERVER_CLASSES.SCATTER]),
    'continuous-batching',
  );
});

test('a CUT scenario is not addressable, even by hand-typed URL', () => {
  // Removing the tab is not enough. The id was public for a while -- it is in
  // chat logs, in the design doc and possibly in a bookmark -- so the URL has to
  // stop resolving too, or the capability is still reachable by anyone who kept
  // the link. Prefix reuse was measured and found ABSENT, so this route would be
  // a navigable promise of a feature the engine does not have.
  assert.equal(
    currentScenarioId(`${SCATTER}/demo?scenario=prefix-cache`, [SERVER_CLASSES.SCATTER]),
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

// ---------------------------------------------------------------------------
// The substitution is stated, not performed in silence.
//
// Resolving a cut id to a real one was always CORRECT -- the page must render
// something. The defect was that it happened invisibly: an operator following a
// link to `prefix-cache` got paged KV, drawn perfectly, every field correctly
// badged, with nothing saying they were looking at a different scenario. That
// is a confident answer to a question nobody asked, which is the exact failure
// this dashboard exists to refuse -- one layer above where the field-level
// honesty apparatus can see it.
// ---------------------------------------------------------------------------

test('a substituted scenario reports the substitution instead of hiding it', () => {
  const cut = resolveScenario(`${SCATTER}/demo?scenario=prefix-cache`, [SERVER_CLASSES.SCATTER]);

  assert.equal(cut.id, 'continuous-batching', 'must still render something');
  assert.equal(cut.requested, 'prefix-cache');
  assert.ok(cut.substitution, 'the fact that we substituted must survive the resolver');
  assert.equal(cut.substitution.kind, 'cut');
  assert.equal(cut.substitution.shown, 'continuous-batching');
  assert.ok(
    cut.substitution.reason,
    'a CUT scenario has a recorded reason and the visitor is entitled to it',
  );
});

test('a cut scenario and a typo are told apart, because they are different mistakes', () => {
  // Previously indistinguishable: both took the same silent fallback, so the
  // application could not tell "we withdrew this" from "you misspelled it".
  const cut = resolveScenario(`${SCATTER}/demo?scenario=prefix-cache`, [SERVER_CLASSES.SCATTER]);
  const typo = resolveScenario(`${SCATTER}/demo?scenario=paged-kvv`, [SERVER_CLASSES.SCATTER]);

  assert.equal(cut.substitution.kind, 'cut');
  assert.equal(typo.substitution.kind, 'unknown');
  assert.equal(typo.substitution.reason, null, 'we have no explanation for a typo, and say so');
});

test('a scenario we DID render reports no substitution at all', () => {
  // The anti-vacuity control for the two tests above: if `substitution` were
  // truthy on the happy path, every visitor would see an apology and the
  // notice would be trained out of everyone within a day.
  for (const href of [`${SCATTER}/demo?scenario=paged-kv`, `${SCATTER}/demo`, `${DYNAMIC}/demo`]) {
    const resolved = resolveScenario(href, [SERVER_CLASSES.SCATTER, SERVER_CLASSES.DYNAMIC]);
    assert.equal(resolved.substitution, null, `${href} must not claim a substitution`);
    assert.ok(Object.hasOwn(SCENARIOS, resolved.id));
  }
});

test('the substitution sentence names the rejected id AND what is shown instead', () => {
  // "Showing something else" is only honest if the visitor can tell WHAT else.
  const { substitution } = resolveScenario(
    `${DYNAMIC}/demo?scenario=prefix-cache`,
    [SERVER_CLASSES.DYNAMIC],
  );
  const sentence = describeSubstitution(substitution);

  assert.match(sentence, /prefix-cache/, 'must quote what it rejected');
  assert.match(sentence, /Paged KV block table/, 'must name the scenario by its VISIBLE label');
  assert.match(sentence, /cut/, 'must say it was withdrawn rather than imply a typo');
});

test('an unbounded scenario id cannot push the panels off the page', () => {
  // The value is attacker-controlled -- `?scenario=../../etc/passwd` is already
  // a test above. Rendering is textContent everywhere, so this is a LAYOUT
  // bound, not an injection defence.
  const huge = 'z'.repeat(5000);
  const { substitution } = resolveScenario(
    `${SCATTER}/demo?scenario=${huge}`,
    [SERVER_CLASSES.SCATTER],
  );

  assert.ok(
    substitution.requested.length < 100,
    `quoted ${substitution.requested.length} chars of a 5000-char id`,
  );
  assert.ok(describeSubstitution(substitution).length < 300);
});

test('currentScenarioId stays a projection of resolveScenario, never a second resolver', () => {
  // Two resolvers that agree today are a divergence waiting to happen, and this
  // one decides which page a visitor sees.
  for (const href of [
    `${SCATTER}/demo`,
    `${DYNAMIC}/demo`,
    `${SCATTER}/demo?scenario=paged-kv`,
    `${SCATTER}/demo?scenario=prefix-cache`,
    `${SCATTER}/demo?scenario=../../etc/passwd`,
  ]) {
    for (const classes of [[SERVER_CLASSES.SCATTER], [SERVER_CLASSES.DYNAMIC], []]) {
      assert.equal(
        currentScenarioId(href, classes),
        resolveScenario(href, classes).id,
        `${href} disagreed for ${JSON.stringify(classes)}`,
      );
    }
  }
});
