// Copyright (c) Microsoft Corporation.
//
// The telemetry store: ONE polling loop, ONE shared snapshot, many subscribers.
//
// Every panel in the page reads from this store and nothing else. There is
// exactly one poll cycle no matter how many panels are mounted, so adding a
// panel costs zero extra HTTP traffic and every panel on screen is showing the
// same instant in time — panels that poll independently drift apart and the
// dashboard starts contradicting itself.
//
// The store's other job is honesty. It never hands out a bare number. Every
// value leaves here wrapped in a TelemetryField carrying its provenance, and
// fields classified DOCUMENTED_ZERO or NOT_PLUMBED in telemetry-provenance.js
// are forced to `unavailable` regardless of what the server sent. See
// telemetry-field.js for why that is load-bearing rather than fussy.
//
// See CONTRACT.md for the panel-facing contract.

import {
  measuredField,
  unavailableField,
  staleField,
  FIELD_STATES,
} from './telemetry-field.js';
import {
  ENDPOINTS,
  DEBUG_GATED_ENDPOINTS,
  DEBUG_ENDPOINTS_FLAG,
  PROVENANCE,
  NEVER_MEASURED_CLASSIFICATIONS,
  allFieldKeys,
} from './telemetry-provenance.js';

/**
 * Connection state. These drive the two BLOCKING full-stage failure states
 * required by the spec; they are deliberately distinct because they are
 * different problems with different fixes.
 *
 * @typedef {'connecting' | 'connected' | 'unreachable' | 'no-model'} ConnectionState
 */
export const CONNECTION_STATES = Object.freeze({
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  UNREACHABLE: 'unreachable',
  NO_MODEL: 'no-model',
});

/** Poll cadence bounds. The spec fixes the dashboard at 250-500 ms. */
export const MIN_POLL_INTERVAL_MS = 100;
export const DEFAULT_POLL_INTERVAL_MS = 250;

/** Backoff for reconnection attempts while unreachable, in milliseconds. */
const RECONNECT_BACKOFF_MS = Object.freeze([500, 1000, 2000, 4000, 4000, 8000]);

/**
 * @typedef {object} ConnectionStatus
 * @property {ConnectionState} state
 * @property {string} origin              The origin we are polling.
 * @property {string|null} serverMessage  The server's OWN error text, verbatim,
 *                                        never paraphrased. Null if the failure
 *                                        was transport-level.
 * @property {string|null} transportError The browser's fetch error text, verbatim.
 * @property {number} consecutiveFailures
 * @property {number|null} lastSuccessAtMs
 * @property {number|null} nextRetryAtMs  For the visible reconnect countdown.
 */

/**
 * @typedef {object} TelemetrySnapshot
 * @property {number} timestampMs        When this snapshot was assembled.
 * @property {ConnectionStatus} connection
 * @property {Readonly<Record<string, import('./telemetry-field.js').TelemetryField>>} fields
 * @property {Readonly<Record<string, string|null>>} endpointErrors
 *   Endpoint path -> why it failed this cycle (null when it succeeded).
 */

/**
 * Create the telemetry store.
 *
 * The store is inert until `start()` is called, which makes it testable without
 * a server and lets the page decide when polling begins.
 *
 * @param {object} [options]
 * @param {string} [options.baseUrl]
 *   Origin of the server. Defaults to the page's own origin, because the server
 *   serves this demo at GET /demo — same-origin, no CORS, nothing to configure.
 * @param {number} [options.pollIntervalMs] 250-500 ms. Clamped to >= 100 ms.
 * @param {typeof fetch} [options.fetchImpl] Injected for tests.
 * @param {() => number} [options.now] Injected for tests.
 * @returns {TelemetryStore}
 */
export function createTelemetryStore({
  baseUrl = typeof location !== 'undefined' ? location.origin : 'http://127.0.0.1:8123',
  pollIntervalMs = DEFAULT_POLL_INTERVAL_MS,
  fetchImpl = typeof fetch !== 'undefined' ? fetch.bind(globalThis) : undefined,
  now = () => Date.now(),
} = {}) {
  if (!fetchImpl) {
    throw new TypeError(
      'createTelemetryStore() needs a fetch implementation. Pass `fetchImpl` when running ' +
        'outside a browser.',
    );
  }

  const interval = Math.max(MIN_POLL_INTERVAL_MS, pollIntervalMs);

  /** @type {Set<(snapshot: TelemetrySnapshot) => void>} */
  const subscribers = new Set();

  /** @type {TelemetrySnapshot} */
  let snapshot = initialSnapshot(baseUrl, now());

  let running = false;
  /** True while a poll cycle is in flight. Guarantees at most one outstanding
   *  request set — a slow server must never queue up polls behind it. */
  let pollInFlight = false;
  /** @type {ReturnType<typeof setTimeout>|null} */
  let timerHandle = null;
  let consecutiveFailures = 0;

  /** @type {TelemetryStore} */
  const store = {
    get pollIntervalMs() {
      return interval;
    },

    get baseUrl() {
      return baseUrl;
    },

    start() {
      if (running) return;
      running = true;
      void runPollCycle();
    },

    stop() {
      running = false;
      if (timerHandle !== null) {
        clearTimeout(timerHandle);
        timerHandle = null;
      }
    },

    get isRunning() {
      return running;
    },

    getSnapshot() {
      return snapshot;
    },

    field(key) {
      const existing = snapshot.fields[key];
      if (existing) return existing;
      return unavailableField(
        `No field named "${key}" is published by this server build. Check the key against ` +
          'telemetry-provenance.js.',
        { source: 'unknown' },
      );
    },

    subscribe(listener) {
      if (typeof listener !== 'function') {
        throw new TypeError('subscribe() expects a function.');
      }
      subscribers.add(listener);
      // Deliver the current snapshot immediately so a panel mounting mid-run
      // paints real state on its first frame instead of an empty one. Guarded
      // for the same reason publish() is: a panel that throws on its first
      // render must not take down whoever mounted it.
      deliver(listener, snapshot);
      return () => {
        subscribers.delete(listener);
      };
    },

    async pollOnce() {
      await runPollCycle({ scheduleNext: false });
      return snapshot;
    },
  };

  return store;

  // ------------------------------------------------------------------ internals

  /**
   * One full poll cycle: fetch every source, rebuild the field map, publish.
   *
   * @param {object} [options]
   * @param {boolean} [options.scheduleNext]
   */
  async function runPollCycle({ scheduleNext = true } = {}) {
    if (pollInFlight) return;
    pollInFlight = true;
    try {
      const results = await fetchAllSources();
      publish(buildSnapshot(results));
    } catch (error) {
      // A throw here is a bug in the store, not a server failure — server
      // failures are captured per-endpoint inside fetchAllSources(). Surface it
      // rather than silently stopping the loop.
      console.error('[telemetry-store] poll cycle failed unexpectedly', error);
    } finally {
      pollInFlight = false;
      if (scheduleNext && running) {
        timerHandle = setTimeout(() => {
          void runPollCycle();
        }, nextDelayMs());
      }
    }
  }

  /** Poll fast when healthy; back off while unreachable so we never flood. */
  function nextDelayMs() {
    if (snapshot.connection.state !== CONNECTION_STATES.UNREACHABLE) {
      return interval;
    }
    const index = Math.min(consecutiveFailures - 1, RECONNECT_BACKOFF_MS.length - 1);
    return RECONNECT_BACKOFF_MS[Math.max(0, index)];
  }

  /**
   * Fetch every source concurrently. Each source degrades independently: a
   * disabled debug endpoint must not take down the panels fed by /v1/status.
   *
   * @returns {Promise<Record<string, SourceResult>>}
   */
  async function fetchAllSources() {
    const paths = [
      ENDPOINTS.HEALTH,
      ENDPOINTS.STATUS,
      ENDPOINTS.DEBUG_KV,
      ENDPOINTS.DEBUG_CONFIG,
    ];
    const settled = await Promise.all(paths.map((path) => fetchJson(path)));
    /** @type {Record<string, SourceResult>} */
    const byPath = {};
    paths.forEach((path, index) => {
      byPath[path] = settled[index];
    });
    return byPath;
  }

  /**
   * @typedef {object} SourceResult
   * @property {boolean} ok
   * @property {any} body
   * @property {number|null} status        HTTP status, null on transport failure.
   * @property {string|null} serverMessage The server's own error text, verbatim.
   * @property {string|null} transportError
   */

  /**
   * @param {string} path
   * @returns {Promise<SourceResult>}
   */
  async function fetchJson(path) {
    try {
      const response = await fetchImpl(`${baseUrl}${path}`, {
        headers: { accept: 'application/json' },
        cache: 'no-store',
      });
      if (!response.ok) {
        return {
          ok: false,
          body: null,
          status: response.status,
          serverMessage: await readServerMessage(response),
          transportError: null,
        };
      }
      return {
        ok: true,
        body: await response.json(),
        status: response.status,
        serverMessage: null,
        transportError: null,
      };
    } catch (error) {
      return {
        ok: false,
        body: null,
        status: null,
        serverMessage: null,
        transportError: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Extract the server's error message VERBATIM.
   *
   * The server's error body is `{ "error": { "message": "...", "type": "..." } }`
   * (crates/onnx-genai-server/src/routes/mod.rs:444-449) and its messages are
   * written in a what/why/how style that is better than anything we would
   * paraphrase. We show it unedited.
   *
   * @param {Response} response
   * @returns {Promise<string|null>}
   */
  async function readServerMessage(response) {
    try {
      const text = await response.text();
      if (!text) return null;
      try {
        const parsed = JSON.parse(text);
        return parsed?.error?.message ?? text;
      } catch {
        return text;
      }
    } catch {
      return null;
    }
  }

  /**
   * Turn raw source results into the published snapshot.
   *
   * @param {Record<string, SourceResult>} sources
   * @returns {TelemetrySnapshot}
   */
  function buildSnapshot(sources) {
    const timestampMs = now();
    const health = sources[ENDPOINTS.HEALTH];
    const status = sources[ENDPOINTS.STATUS];

    // Unreachable means TRANSPORT failure — the process is not there. An HTTP
    // error is a reachable server telling us something, which is a different
    // state with a different fix.
    const transportDown = health.transportError !== null && status.transportError !== null;

    if (transportDown) {
      consecutiveFailures += 1;
      return {
        timestampMs,
        connection: {
          state: CONNECTION_STATES.UNREACHABLE,
          origin: baseUrl,
          serverMessage: null,
          transportError: health.transportError ?? status.transportError,
          consecutiveFailures,
          lastSuccessAtMs: snapshot.connection.lastSuccessAtMs,
          nextRetryAtMs: timestampMs + nextDelayMs(),
        },
        fields: ageFields(
          snapshot.fields,
          `The server at ${baseUrl} stopped responding, so this value is frozen at its last ` +
            'measured reading.',
        ),
        endpointErrors: allEndpointsFailed(health.transportError ?? status.transportError),
      };
    }

    consecutiveFailures = 0;
    const noModel = detectNoModel(sources);
    if (noModel) {
      return {
        timestampMs,
        connection: {
          state: CONNECTION_STATES.NO_MODEL,
          origin: baseUrl,
          serverMessage: noModel.serverMessage,
          transportError: null,
          consecutiveFailures: 0,
          lastSuccessAtMs: timestampMs,
          nextRetryAtMs: timestampMs + interval,
        },
        // No model means no runtime state to report. Everything is unavailable
        // for one clear reason, not frozen at some earlier reading.
        fields: allUnavailable(
          'The server is running but has no model loaded, so it has no runtime state to report.',
        ),
        endpointErrors: {},
      };
    }

    return {
      timestampMs,
      connection: {
        state: CONNECTION_STATES.CONNECTED,
        origin: baseUrl,
        serverMessage: null,
        transportError: null,
        consecutiveFailures: 0,
        lastSuccessAtMs: timestampMs,
        nextRetryAtMs: timestampMs + interval,
      },
      fields: buildFields(sources, timestampMs),
      endpointErrors: collectEndpointErrors(sources),
    };
  }

  /**
   * "Reachable but no model" detection.
   *
   * `/v1/status.healthy` is `registry.default_id().is_some()`
   * (routes/admin.rs:47-51), which is precisely "a model is registered". A
   * model-dependent endpoint also answers 500 `"no model loaded"`
   * (routes/admin.rs:110-115 via ApiError::internal) — we prefer the server's
   * own sentence when we have it.
   *
   * @param {Record<string, SourceResult>} sources
   * @returns {{serverMessage: string|null}|null}
   */
  function detectNoModel(sources) {
    const status = sources[ENDPOINTS.STATUS];
    const debugConfig = sources[ENDPOINTS.DEBUG_CONFIG];
    const healthyFlag = status.ok ? status.body?.healthy : undefined;
    if (healthyFlag !== false) {
      return null;
    }
    // Prefer a message the server actually wrote over one we invent.
    const serverMessage =
      debugConfig.status === 500 && debugConfig.serverMessage ? debugConfig.serverMessage : null;
    return { serverMessage };
  }

  /**
   * Build the field map for a healthy cycle.
   *
   * This is where the provenance table becomes enforcement: a DOCUMENTED_ZERO
   * or NOT_PLUMBED field is emitted as `unavailable` even when the response
   * carried a perfectly parseable number for it.
   *
   * @param {Record<string, SourceResult>} sources
   * @param {number} timestampMs
   */
  function buildFields(sources, timestampMs) {
    /** @type {Record<string, import('./telemetry-field.js').TelemetryField>} */
    const fields = {};
    for (const key of allFieldKeys()) {
      const entry = PROVENANCE[key];

      if (NEVER_MEASURED_CLASSIFICATIONS.includes(entry.classification)) {
        fields[key] = unavailableField(entry.reason, { source: entry.source, unit: entry.unit });
        continue;
      }

      const source = sources[entry.source];
      if (!source) {
        fields[key] = unavailableField(
          `This demo does not poll ${entry.source} yet.`,
          { source: entry.source, unit: entry.unit },
        );
        continue;
      }

      if (!source.ok) {
        fields[key] = unavailableField(describeSourceFailure(entry.source, source), {
          source: entry.source,
          unit: entry.unit,
        });
        continue;
      }

      const rawValue = readPath(source.body, entry.path);
      if (rawValue === undefined || rawValue === null) {
        fields[key] = unavailableField(
          `${entry.source} responded, but carried no "${entry.path}" field. This server build ` +
            'may predate that field.',
          { source: entry.source, unit: entry.unit },
        );
        continue;
      }

      fields[key] = measuredField(rawValue, {
        source: entry.source,
        unit: entry.unit,
        observedAtMs: timestampMs,
      });
    }
    return Object.freeze(fields);
  }

  /**
   * Human explanation for a failed source, naming the fix where there is one.
   * A 404 on a debug-gated endpoint is a missing flag, not a broken server, and
   * saying so is the difference between "this is broken" and "I missed a step".
   *
   * @param {string} path
   * @param {SourceResult} source
   */
  function describeSourceFailure(path, source) {
    if (source.transportError) {
      return `${path} could not be reached: ${source.transportError}`;
    }
    if (source.status === 404 && DEBUG_GATED_ENDPOINTS.includes(path)) {
      return (
        `${path} is disabled. Restart the server with ${DEBUG_ENDPOINTS_FLAG} to enable it ` +
        '(the rest of the dashboard keeps working without it).'
      );
    }
    if (source.serverMessage) {
      return `${path} returned ${source.status}: ${source.serverMessage}`;
    }
    return `${path} returned HTTP ${source.status}.`;
  }

  /** @param {Record<string, SourceResult>} sources */
  function collectEndpointErrors(sources) {
    /** @type {Record<string, string|null>} */
    const errors = {};
    for (const [path, source] of Object.entries(sources)) {
      errors[path] = source.ok ? null : describeSourceFailure(path, source);
    }
    return Object.freeze(errors);
  }

  /** @param {string|null} error */
  function allEndpointsFailed(error) {
    /** @type {Record<string, string|null>} */
    const errors = {};
    for (const path of Object.values(ENDPOINTS)) {
      errors[path] = error;
    }
    return Object.freeze(errors);
  }

  /** @param {string} reason */
  function allUnavailable(reason) {
    /** @type {Record<string, import('./telemetry-field.js').TelemetryField>} */
    const fields = {};
    for (const key of allFieldKeys()) {
      const entry = PROVENANCE[key];
      fields[key] = unavailableField(reason, { source: entry.source, unit: entry.unit });
    }
    return Object.freeze(fields);
  }

  /**
   * @param {Readonly<Record<string, import('./telemetry-field.js').TelemetryField>>} fields
   * @param {string} reason
   */
  function ageFields(fields, reason) {
    /** @type {Record<string, import('./telemetry-field.js').TelemetryField>} */
    const aged = {};
    for (const [key, field] of Object.entries(fields)) {
      aged[key] = staleField(field, reason);
    }
    return Object.freeze(aged);
  }

  /** @param {TelemetrySnapshot} next */
  function publish(next) {
    snapshot = Object.freeze(next);
    for (const listener of subscribers) {
      deliver(listener, snapshot);
    }
  }

  /**
   * Deliver a snapshot to one subscriber. One panel throwing must never stop
   * the other panels updating, so every delivery path goes through here.
   *
   * @param {(snapshot: TelemetrySnapshot) => void} listener
   * @param {TelemetrySnapshot} value
   */
  function deliver(listener, value) {
    try {
      listener(value);
    } catch (error) {
      console.error('[telemetry-store] subscriber threw; other panels continue', error);
    }
  }
}

/**
 * @param {string} baseUrl
 * @param {number} timestampMs
 * @returns {TelemetrySnapshot}
 */
function initialSnapshot(baseUrl, timestampMs) {
  /** @type {Record<string, import('./telemetry-field.js').TelemetryField>} */
  const fields = {};
  for (const key of allFieldKeys()) {
    const entry = PROVENANCE[key];
    fields[key] = unavailableField('Waiting for the first poll to complete.', {
      source: entry.source,
      unit: entry.unit,
    });
  }
  return Object.freeze({
    timestampMs,
    connection: Object.freeze({
      state: CONNECTION_STATES.CONNECTING,
      origin: baseUrl,
      serverMessage: null,
      transportError: null,
      consecutiveFailures: 0,
      lastSuccessAtMs: null,
      nextRetryAtMs: null,
    }),
    fields: Object.freeze(fields),
    endpointErrors: Object.freeze({}),
  });
}

/**
 * Read a dotted path out of a parsed JSON body.
 *
 * @param {any} body
 * @param {string} path
 */
function readPath(body, path) {
  return path.split('.').reduce((node, segment) => {
    if (node === null || node === undefined) return undefined;
    return node[segment];
  }, body);
}

/**
 * @typedef {object} TelemetryStore
 * @property {number} pollIntervalMs
 * @property {string} baseUrl
 * @property {boolean} isRunning
 * @property {() => void} start
 * @property {() => void} stop
 * @property {() => TelemetrySnapshot} getSnapshot
 * @property {(key: string) => import('./telemetry-field.js').TelemetryField} field
 * @property {(listener: (snapshot: TelemetrySnapshot) => void) => () => void} subscribe
 * @property {() => Promise<TelemetrySnapshot>} pollOnce
 */

export { FIELD_STATES };
