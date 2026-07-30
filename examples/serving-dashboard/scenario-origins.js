// Copyright (c) Microsoft Corporation.
//
// WHICH SERVER BACKS WHICH SCENARIO — the single config point.
//
// Continuous batching and paged KV are mutually exclusive in this runtime
// (ContinuousBatchManager never touches engine.kv_cache,
// crates/onnx-genai-engine/src/engine/batched.rs:101-110), so today the demo
// runs two servers and each scenario reads from the one that can actually
// measure it.
//
// That may not stay true. If one server learns to hold both models and
// attribute telemetry per model, the topology collapses to a single origin.
// THE UI IS IDENTICAL EITHER WAY, so this module is the only place that knows
// the difference — and the collapse costs nothing, because a server that
// advertises both classes resolves both to its own origin automatically.
//
// Two rules this module exists to enforce:
//
//   1. NO PORT IS HARD-CODED. The launcher binds the ports, so the launcher
//      passes them in the URL it prints. A constant here would be a fourth
//      copy of a number that already lives in run-demo.sh and the environment.
//
//   2. AN UNKNOWN ORIGIN IS NEVER GUESSED. Falling back to "just use this
//      origin" would silently poll the scatter server for paged-KV fields and
//      render its structural zeros as measurements. That is precisely the
//      fabrication this codebase is built to prevent, so an unresolved origin
//      resolves to null and the scenario degrades with a reason.

/** The two engine configurations. A server is one, the other, or both. */
export const SERVER_CLASSES = Object.freeze({
  /** Static-cache model. Continuous batching engages; the page table does not. */
  SCATTER: 'scatter',
  /** Dynamic model. Paged KV and the prefix cache engage; batching does not. */
  DYNAMIC: 'dynamic',
});

/**
 * Every scenario, and the server class that can genuinely measure it.
 *
 * This table is the answer to "why is this panel talking to that port". Adding
 * a scenario means adding a row here and nowhere else.
 */
export const SCENARIOS = Object.freeze({
  'continuous-batching': Object.freeze({
    id: 'continuous-batching',
    label: 'Continuous batching',
    serverClass: SERVER_CLASSES.SCATTER,
    why: 'Continuous batching engages only on static-cache models.',
  }),
  'paged-kv': Object.freeze({
    id: 'paged-kv',
    label: 'Paged KV block table',
    serverClass: SERVER_CLASSES.DYNAMIC,
    why: 'The block allocator runs only on the dynamic path; static-cache models use runtime-owned in-place buffers.',
  }),
  'memory-pressure': Object.freeze({
    id: 'memory-pressure',
    label: 'Memory pressure',
    serverClass: SERVER_CLASSES.DYNAMIC,
    why: 'Pressure is created by genuinely filling the KV pool, which only the dynamic path allocates from.',
  }),
});

/**
 * Scenarios that were CUT, and why. Deliberately kept as a record rather than
 * deleted, so the next person to reach for the id learns the finding instead of
 * re-adding the tab.
 *
 * A tab is not a panel. A panel displays a value and can honestly say "n/a"; a
 * TAB ADVERTISES A CAPABILITY — a labelled, clickable promise that the product
 * does this, made before the visitor has seen a single number. Prefix reuse was
 * measured and found ABSENT on both execution paths (QA, n=6 per arm with a
 * sensitivity control: shared-prefix requests ran 7% SLOWER than a zero-sharing
 * control, while the engine's hit counter fired on every request including all
 * six controls). So a reachable `?scenario=prefix-cache` route is a navigable
 * link to a feature we proved is not there.
 *
 * The finding itself still ships — it is the most credible thing the demo has
 * to say — but it ships as evidence inside a panel, not as a capability tab.
 */
export const CUT_SCENARIOS = Object.freeze({
  // Keyed only, with no `id:` field, and that is deliberate: an `id` is what
  // makes a scenario addressable, and this one must not be addressable. It also
  // keeps the page-claims guard honest, which scans for `id:`/`label:`
  // declarations and cannot otherwise tell a record from a route.
  'prefix-cache': Object.freeze({
    reason:
      'Prefix reuse was measured and found absent on both execution paths, so a tab ' +
      'advertising it would promise a capability the engine does not have. The null ' +
      'result is rendered as evidence instead.',
  }),
});

/**
 * Which classes THIS server can serve, when the launcher did not say.
 *
 * The server exposes no cache-type field — DebugConfigResponse
 * (routes/mod.rs:144-151) carries model_id, pipeline and context length, and
 * nothing that distinguishes a static-cache model from a dynamic one. So there
 * is no authoritative signal, and the launcher's query parameters are the only
 * declaration. This is the fallback for a hand-started server.
 *
 * The inference is deliberately narrow: it decides only WHICH LOCAL SCENARIOS
 * to offer. It can never invent a peer origin, so a wrong guess cannot make one
 * server's numbers appear under the other server's name — the worst case is
 * that a scenario is offered which then reports honest structural zeros.
 *
 * @param {string|null} modelId  From /health, which is never gated.
 * @returns {{classes: string[], confidence: 'inferred'|'unknown', reason: string}}
 */
export function selfClassesFromModelId(modelId) {
  if (typeof modelId !== 'string' || modelId === '') {
    return {
      classes: [],
      confidence: 'unknown',
      reason: 'The server did not report a model id, so its engine configuration is unknown.',
    };
  }

  // Continuous batching engages only on static-cache models, which are built
  // and named with a -scatter suffix. The documented launch command and
  // run-demo.sh both use it.
  const isScatter = /scatter/i.test(modelId);
  return {
    classes: [isScatter ? SERVER_CLASSES.SCATTER : SERVER_CLASSES.DYNAMIC],
    confidence: 'inferred',
    reason:
      `Inferred from the model id "${modelId}" because the server exposes no cache-type field. ` +
      'Start the demo with run-demo.sh to declare this explicitly.',
  };
}

/**
 * Reconcile what the URL DECLARES about this server with what the server
 * itself REPORTS.
 *
 * DETECTION BEATS SELECTION. run-demo.sh declares the classes in the URL it
 * prints, because it is the process that bound the ports -- but a URL is an
 * ASSERTION and the model id is a FACT. A visitor who points this page at
 * their own server, or edits a port in a copied link, can trivially declare
 * "scatter" over a dynamic server. Believing the URL there produces a fully
 * rendered and entirely wrong dashboard: prefix-cache fields classified
 * structurally-not-applicable on a server that genuinely measures them, and
 * batching panels mounted on a server that cannot batch.
 *
 * When the two disagree the SERVER WINS, and the disagreement is returned so
 * the page can say so. Silently correcting leaves the visitor with a working
 * page and a link that keeps lying to the next person they send it to.
 *
 * @param {object} input
 * @param {string[]} input.declared  Classes the URL claims for this origin.
 * @param {string[]} input.observed  Classes inferred from the model id.
 * @param {string|null} input.observedModelId
 * @param {string} input.origin      This page's own origin, for the message.
 * @returns {{classes: string[], declared: string[], contradiction: string|null}}
 */
export function reconcileSelfClasses({ declared, observed, observedModelId, origin }) {
  if (declared.length === 0) {
    return { classes: observed, declared: [], discredited: [], contradiction: null };
  }

  // Nothing to check against. An unreachable or unidentifiable server leaves
  // the declaration as the only information available, and refusing it would
  // mount nothing at all -- a worse outcome than trusting the launcher.
  if (observed.length === 0) {
    return { classes: declared, declared, discredited: [], contradiction: null };
  }

  if (observed.some((serverClass) => declared.includes(serverClass))) {
    return { classes: declared, declared, discredited: [], contradiction: null };
  }

  return {
    classes: observed,
    declared,
    // The classes the URL claimed for THIS origin that the server has now
    // disproved. They must be dropped from the origins map: leaving them means
    // offering a tab that navigates here promising a capability this server
    // provably does not have.
    discredited: declared.filter((serverClass) => !observed.includes(serverClass)),
    contradiction:
      `This page was opened with a URL declaring the ${declared.join(' and ')} server, but ` +
      `${origin} reports the model "${observedModelId}", which is a ${observed.join(' and ')} ` +
      "server. The server's own answer is being used, because a URL parameter is an assertion " +
      'and the model id is a fact. Check the origin parameters against the servers that are ' +
      'actually running.',
  };
}

/**
 * Translation to the dashboard registry's vocabulary.
 *
 * @c8d9a40e's dashboard/index.js names the two engine configurations by what
 * they can DEMONSTRATE ('batching', 'paged'); this module names them by what
 * they ARE ('scatter', 'dynamic'), because the model's cache type is the root
 * cause of the exclusivity and stays meaningful if the two servers are ever
 * collapsed into one multi-model server.
 *
 * Rather than either of us silently renaming the other's strings, the seam is
 * this one map. It is the only place the two vocabularies meet.
 *
 * @type {Readonly<Record<string, 'batching'|'paged'>>}
 */
export const SERVER_MODE_BY_CLASS = Object.freeze({
  [SERVER_CLASSES.SCATTER]: 'batching',
  [SERVER_CLASSES.DYNAMIC]: 'paged',
});

/**
 * Query parameter naming the origin for a server class, e.g.
 * `?dynamic-origin=http://host:port`. run-demo.sh prints a URL carrying these
 * because it is the process that actually bound the ports.
 */
function paramNameFor(serverClass) {
  return `${serverClass}-origin`;
}

/**
 * Reject anything that is not a plain http(s) origin.
 *
 * The value arrives from the query string, so it is attacker-controllable in
 * the sense that a link can carry anything. It is used to build fetch URLs and
 * to populate an href, so a `javascript:` value would be an XSS vector on a
 * page whose whole purpose is to be handed around as a URL.
 *
 * @param {string} raw
 * @returns {string|null} A normalised origin, or null if unusable.
 */
export function parseOrigin(raw) {
  if (typeof raw !== 'string' || raw === '') return null;
  let url;
  try {
    url = new URL(raw);
  } catch {
    return null;
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
  return url.origin;
}

/**
 * Resolve every server class to the origin that serves it.
 *
 * @param {object} options
 * @param {string} options.href            The page's current URL.
 * @param {string[]} options.selfClasses   Classes THIS server can serve. One
 *   entry today; both once a single server can hold both models, which is the
 *   whole reason this is a list.
 * @returns {Record<string, string|null>} class -> origin, null when unknown.
 */
export function resolveOrigins({ href, selfClasses = [] }) {
  const url = new URL(href);
  const resolved = {};

  for (const serverClass of Object.values(SERVER_CLASSES)) {
    // An explicit parameter wins: it is how the launcher tells the page where
    // its peer landed, and how a developer points at a hand-started server.
    const fromQuery = parseOrigin(url.searchParams.get(paramNameFor(serverClass)));
    if (fromQuery) {
      resolved[serverClass] = fromQuery;
      continue;
    }
    // Otherwise, only this origin, and only if this server really is that
    // class. Never assume the peer is here.
    resolved[serverClass] = selfClasses.includes(serverClass) ? url.origin : null;
  }

  return resolved;
}

/**
 * Where a scenario should read telemetry from, and whether it can run at all.
 *
 * @param {string} scenarioId
 * @param {Record<string, string|null>} origins  From resolveOrigins.
 * @param {string} currentOrigin                 The page's own origin.
 * @returns {{
 *   scenario: object,
 *   serverClass: string,
 *   baseUrl: string|null,
 *   available: boolean,
 *   requiresNavigation: boolean,
 *   reason: string|null
 * }}
 */
export function planScenario(scenarioId, origins, currentOrigin) {
  const scenario = SCENARIOS[scenarioId];
  if (!scenario) throw new Error(`unknown scenario: ${scenarioId}`);

  const baseUrl = origins[scenario.serverClass] ?? null;

  if (!baseUrl) {
    return {
      scenario,
      serverClass: scenario.serverClass,
      baseUrl: null,
      available: false,
      requiresNavigation: false,
      // Naming the launcher matters: the visitor's fix is to start the demo
      // the documented way, not to hunt for a port.
      reason:
        `No ${scenario.serverClass} server is configured. ${scenario.why} ` +
        'Start the demo with examples/serving-dashboard/run-demo.sh, which starts both ' +
        'servers and prints a URL that carries their addresses.',
    };
  }

  return {
    scenario,
    serverClass: scenario.serverClass,
    baseUrl,
    available: true,
    // Each page talks only to the origin that served it, so reaching another
    // server is a navigation, not a cross-origin fetch. That is why the server
    // needs no CORS configuration and has none.
    requiresNavigation: baseUrl !== currentOrigin,
    reason: null,
  };
}

/**
 * The URL to navigate to for a scenario, carrying the topology forward.
 *
 * Without this the origin parameters would be lost on the first scenario
 * switch and the demo would degrade halfway through.
 *
 * @param {string} scenarioId
 * @param {Record<string, string|null>} origins
 * @returns {string|null} An absolute URL, or null if the scenario is unavailable.
 */
export function scenarioHref(scenarioId, origins) {
  const scenario = SCENARIOS[scenarioId];
  if (!scenario) throw new Error(`unknown scenario: ${scenarioId}`);

  const baseUrl = origins[scenario.serverClass] ?? null;
  if (!baseUrl) return null;

  const target = new URL('/demo/', baseUrl);
  for (const [serverClass, origin] of Object.entries(origins)) {
    if (origin) target.searchParams.set(paramNameFor(serverClass), origin);
  }
  target.searchParams.set('scenario', scenarioId);
  return target.toString();
}

/**
 * The scenario the current URL selects, defaulting to one this server can
 * actually serve rather than a fixed first entry — otherwise a visitor landing
 * on the dynamic server opens straight into an unavailable batching panel.
 *
 * @param {string} href
 * @param {string[]} selfClasses
 * @returns {string}
 */
export function currentScenarioId(href, selfClasses = []) {
  const requested = new URL(href).searchParams.get('scenario');
  if (requested && Object.hasOwn(SCENARIOS, requested)) return requested;

  const local = Object.values(SCENARIOS).find((s) => selfClasses.includes(s.serverClass));
  return local ? local.id : 'continuous-batching';
}
