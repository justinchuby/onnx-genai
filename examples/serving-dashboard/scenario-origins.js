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
  'prefix-cache': Object.freeze({
    id: 'prefix-cache',
    label: 'Prefix caching',
    serverClass: SERVER_CLASSES.DYNAMIC,
    why: 'The batching path never consults the prefix trie, which the engine tests assert (batched_static_decode.rs:53).',
  }),
  'memory-pressure': Object.freeze({
    id: 'memory-pressure',
    label: 'Memory pressure',
    serverClass: SERVER_CLASSES.DYNAMIC,
    why: 'Pressure is created by genuinely filling the KV pool, which only the dynamic path allocates from.',
  }),
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
