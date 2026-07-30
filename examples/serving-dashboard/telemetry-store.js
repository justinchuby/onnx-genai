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
  notApplicableField,
  pendingField,
  staleField,
  FIELD_STATES,
  SOURCE_CLASSES,
} from './telemetry-field.js';
import {
  ENDPOINTS,
  resolveForOrigin,
  DEBUG_GATED_ENDPOINTS,
  DEBUG_ENDPOINTS_FLAG,
  TEXT_ENDPOINTS,
  FEATURE_GATED_ENDPOINTS,
  PROVENANCE,
  NEVER_MEASURED_CLASSIFICATIONS,
  matchesStub,
  allFieldKeys,
} from './telemetry-provenance.js';
import { parsePrometheusText, scalarOf, histogramMean } from './prometheus-parse.js';
import {
  DEFAULT_REQUEST_TIMEOUT_MS,
  RequestTimeoutError,
  fetchWithDeadline,
} from './request-deadline.js';
import { selfClassesFromModelId } from './scenario-origins.js';

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

/**
 * PER-ENDPOINT POLL CADENCE.
 *
 * Polling six endpoints every 250 ms is 24 requests/second, which eats into the
 * server's <2% overhead budget (AC33) to re-fetch values that cannot have
 * changed. Model context length is fixed for the life of the process; resource
 * LIMITS are configuration. Only the live counters need the full rate.
 *
 * A cadence of 0 means "every poll". Anything else is the minimum gap between
 * requests to that endpoint; in between, the last response is reused and its
 * fields keep the timestamp of the request that actually produced them, so a
 * reused body never claims to be fresher than it is.
 *
 * @type {Readonly<Record<string, number>>}
 */
const ENDPOINT_CADENCE_MS = Object.freeze({
  [ENDPOINTS.HEALTH]: 0,      // liveness probe; must not lag a server going away
  [ENDPOINTS.STATUS]: 0,      // queue depth moves per request
  [ENDPOINTS.DEBUG_KV]: 0,    // batch and KV occupancy are the live signal
  [ENDPOINTS.METRICS]: 500,   // by far the largest payload; still within spec
  [ENDPOINTS.RESOURCES]: 1000,
  [ENDPOINTS.DEBUG_CONFIG]: 30000, // model_max_context/pipeline cannot change
  [ENDPOINTS.MODELS]: 30000,       // the loaded model set, for attribution
});

/** Poll cadence bounds. The spec fixes the dashboard at 250-500 ms. */
export const MIN_POLL_INTERVAL_MS = 100;
export const DEFAULT_POLL_INTERVAL_MS = 250;

/**
 * How long a single request may stay silent before we give up on it.
 *
 * A SERVER THAT FAILS AND A SERVER THAT STALLS ARE COMPLETELY DIFFERENT
 * FAILURES, AND ONLY ONE OF THEM WAS HANDLED. If the server refuses the socket,
 * `fetch` rejects, the cycle records a `transportError`, and the reconnect
 * ladder does its job. If the server ACCEPTS the socket and then never replies
 * — which is precisely what an inference server does mid-long-generation — the
 * promise never settles, `pollInFlight` never clears, and every later tick
 * returns early. The page then shows its last good numbers, with their original
 * timestamps, FOREVER. Not stale, not unavailable: stopped, while looking
 * healthy. That is the one rendering this product must never produce, and it
 * was the only absence state we had no vocabulary for.
 *
 * The timeout does not invent a new failure mode; it converts an unhandled one
 * into the handled one. Two seconds is deliberately many multiples of the
 * poll interval: this exists to break a hang, not to police a slow server.
 *
 * Re-exported, not redeclared. Two constants of the same name in two modules is
 * how a duplicate `batch.capacity` key silently picked a misnomer tonight: JS
 * gives no error, it just picks one. There is one deadline in this product.
 */
export { DEFAULT_REQUEST_TIMEOUT_MS };

/** Backoff for reconnection attempts while unreachable, in milliseconds. */
const RECONNECT_BACKOFF_MS = Object.freeze([500, 1000, 2000, 4000, 4000, 8000]);

/**
 * How long to stop re-requesting an endpoint that answered with an HTTP error.
 *
 * A disabled debug endpoint (404) or a model-less server (500) returns the same
 * error on every poll. At the 250 ms dashboard cadence that is four failed
 * requests per second forever — a console flood that buries real errors and
 * makes the page look broken to anyone with DevTools open.
 *
 * The endpoint is retried on this slower cadence instead, so the page still
 * recovers by itself once the operator fixes the cause (which the failure
 * states promise it will) without hammering an endpoint we know is failing.
 * Transport failures are not suppressed here — those back the whole cycle off
 * via RECONNECT_BACKOFF_MS.
 */
const ENDPOINT_ERROR_RETRY_MS = 10_000;

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
 * @param {number} [options.requestTimeoutMs]
 *   How long one request may stay silent before it is aborted and reported as a
 *   transport error. See DEFAULT_REQUEST_TIMEOUT_MS.
 * @param {typeof fetch} [options.fetchImpl] Injected for tests.
 * @param {() => number} [options.now] Injected for tests.
 * @returns {TelemetryStore}
 */
export function createTelemetryStore({
  baseUrl = typeof location !== 'undefined' ? location.origin : 'http://127.0.0.1:8123',
  // Which server this store polls. The demo runs TWO — a scatter server that
  // batches and a dynamic server that pages KV — because those two features are
  // mutually exclusive in this runtime. Every field carries this, so a number
  // can never be shown without saying which server produced it.
  origin = 'scatter',
  pollIntervalMs = DEFAULT_POLL_INTERVAL_MS,
  requestTimeoutMs = DEFAULT_REQUEST_TIMEOUT_MS,
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
  let snapshot = initialSnapshot(baseUrl, now(), origin);

  let running = false;
  /** True while a poll cycle is in flight. Guarantees at most one outstanding
   *  request set — a slow server must never queue up polls behind it. */
  let pollInFlight = false;
  /** @type {ReturnType<typeof setTimeout>|null} */
  let timerHandle = null;
  let consecutiveFailures = 0;
  /**
   * Endpoints that answered with an HTTP error, and the last failure we saw
   * from each. While suppressed, the cached failure is replayed instead of
   * issuing a request, so panels keep their explanation without the network
   * noise. @type {Map<string, {retryAtMs: number, result: SourceResult}>}
   */
  const suppressedEndpoints = new Map();

  // Last SUCCESSFUL response per endpoint, for cadence reuse. Failures are not
  // cached here -- they go through suppressedEndpoints, which has its own retry
  // window and its own explanation for the visitor.
  /** @type {Map<string, SourceResult>} */
  const lastSuccessfulFetch = new Map();

  /**
   * WHICH ENGINE PRODUCED THIS NUMBER, as asserted by the server itself.
   *
   * `origin` is the client's belief about which server it is talking to. This
   * is the server's own answer. The distinction is not pedantic: if someone
   * starts the two servers with the ports swapped, or the single --models-dir
   * server lands, a client-side label becomes a confident lie while the
   * server-reported model id stays true. Under our own rules a client-inferred
   * attribution would be sourceClass 'client', and attribution is the one field
   * that must not be a guess.
   *
   * @type {{modelIds: string[], primary: string|null}}
   */
  let attribution = { modelIds: [], primary: null };

  // NOTE: mutable closure state lives HERE, above every function that reads it.
  // Declaring `let` further down puts it in the temporal dead zone for the
  // pending-field build that runs during construction -- which throws at
  // runtime while every unit test that stubs the build still passes.
  /** Warn once if the server is not the kind of server we think it is. */
  let mismatchWarned = false;

  /**
   * Fields whose provenance classification the wire has contradicted, keyed by
   * field path. Survives across polls so the AC10 honesty footer can list them.
   * @type {Map<string, string>}
   */
  const provenanceWarnings = new Map();

  /**
   * Previous sample of the cumulative token counter, for deriving a rate.
   * @type {{total: number, atMs: number}|null}
   */
  let lastTokenSample = null;

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

    /**
     * Fields where the wire contradicted this build's provenance audit, keyed
     * by field path.
     *
     * The AC10 honesty footer renders these. An empty map is the normal case
     * and means the audit still matches the server; a non-empty one means
     * telemetry-provenance.js needs editing, and says which rows.
     *
     * @returns {Record<string, string>}
     */
    provenanceWarnings() {
      return Object.freeze(Object.fromEntries(provenanceWarnings));
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
    // While unreachable, probe only the two ungated endpoints that determine
    // the connection state. Asking a server we cannot reach for detail it
    // cannot give quadruples the failed-request noise for no information.
    const paths =
      snapshot.connection.state === CONNECTION_STATES.UNREACHABLE
        ? [ENDPOINTS.HEALTH, ENDPOINTS.STATUS]
        : [
            ENDPOINTS.HEALTH,
            ENDPOINTS.STATUS,
            ENDPOINTS.DEBUG_KV,
            ENDPOINTS.DEBUG_CONFIG,
            ENDPOINTS.METRICS,
            ENDPOINTS.RESOURCES,
            ENDPOINTS.MODELS,
          ];

    const settled = await Promise.all(paths.map((path) => fetchSource(path)));
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
   * Fetch one endpoint. JSON endpoints yield a parsed object in `body`; text
   * endpoints (currently only /metrics) yield a parsed Map of metric families,
   * so downstream code reads them via `metric` rather than `path`.
   *
   * @param {string} path
   * @returns {Promise<SourceResult>}
   */
  async function fetchSource(path) {
    const isText = TEXT_ENDPOINTS.includes(path);

    // Not due yet: reuse the last response rather than asking again for a value
    // that cannot have changed. The cached result keeps its original
    // fetchedAtMs, so fields built from it report their true observation time.
    const cadenceMs = ENDPOINT_CADENCE_MS[path] ?? 0;
    const cached = lastSuccessfulFetch.get(path);
    if (cached && cadenceMs > 0 && now() - cached.fetchedAtMs < cadenceMs) {
      return cached;
    }

    // Replay a recent HTTP failure rather than re-issuing a request we already
    // know will fail. The visitor still sees the same explanation; DevTools
    // does not fill with identical errors.
    const suppressed = suppressedEndpoints.get(path);
    if (suppressed && now() < suppressed.retryAtMs) {
      return suppressed.result;
    }

    // A stalled server is indistinguishable from a working one until you give
    // up on it, so every request carries its own deadline. The idiom lives in
    // request-deadline.js because it previously lived here ONLY, and the boot
    // probe in app.js never received it.
    try {
      const response = await fetchWithDeadline(`${baseUrl}${path}`, {
        fetchImpl,
        timeoutMs: requestTimeoutMs,
        headers: { accept: isText ? 'text/plain' : 'application/json' },
        cache: 'no-store',
      });
      if (!response.ok) {
        const result = {
          ok: false,
          body: null,
          status: response.status,
          serverMessage: await readServerMessage(response),
          transportError: null,
        };
        suppressedEndpoints.set(path, {
          retryAtMs: now() + ENDPOINT_ERROR_RETRY_MS,
          result,
        });
        return result;
      }
      suppressedEndpoints.delete(path);
      const result = {
        ok: true,
        body: isText ? parsePrometheusText(await response.text()) : await response.json(),
        status: response.status,
        serverMessage: null,
        transportError: null,
        fetchedAtMs: now(),
      };
      lastSuccessfulFetch.set(path, result);
      return result;
    } catch (error) {
      // Transport failures are handled by the whole-cycle backoff, not here:
      // suppressing them per-endpoint would delay recovery when the server
      // comes back.
      suppressedEndpoints.delete(path);
      return {
        ok: false,
        body: null,
        status: null,
        serverMessage: null,
        // A timeout gets its own sentence rather than the runtime's abort text.
        // "This operation was aborted" reads like the dashboard did something
        // wrong; the visitor needs to know the SERVER went quiet, and for how
        // long, because that distinguishes a hung generation from a dead port.
        // The distinction is carried by the error TYPE, so nothing here has to
        // string-match a runtime's wording.
        transportError:
          error instanceof RequestTimeoutError
            ? error.message
            : error instanceof Error
              ? error.message
              : String(error),
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
   * Read the loaded model set from whichever source answered.
   *
   * /v1/models is ungated (lib.rs:65 sits outside the admin gate) and lists
   * every loaded model, which is also how a single multi-model server announces
   * itself. /health.model is the fallback and carries the same id for the
   * single-model case (routes/mod.rs:105-108).
   *
   * @param {Record<string, SourceResult>} sources
   */
  function readAttribution(sources) {
    const health = sources[ENDPOINTS.HEALTH];
    const fromHealth = health?.ok ? health.body?.model : null;
    const served = typeof fromHealth === 'string' && fromHealth !== '' ? fromHealth : null;

    const models = sources[ENDPOINTS.MODELS];
    if (models?.ok && Array.isArray(models.body?.data)) {
      const entries = models.body.data;
      const ids = entries.map((entry) => entry?.id).filter((id) => typeof id === 'string');
      if (ids.length > 0) {
        // WHICH ID IS PRIMARY IS NOT A FORMATTING DETAIL -- IT DECIDES WHICH
        // MODEL EVERY NUMBER ON THE PAGE IS ATTRIBUTED TO.
        //
        // This previously took ids[0]. /v1/models is emitted from a registry
        // whose ids are SORTED, so ids[0] is the alphabetically-first model,
        // which is not necessarily the one this server is serving. In the
        // two-model demo "qwen2.5-0.5b" sorts before "qwen2.5-0.5b-scatter-v2",
        // so a scatter server would have attributed every KV and batching
        // number to the DYNAMIC model -- the exact silent misattribution the
        // two-server design exists to make impossible, reintroduced on the
        // client.
        //
        // /health.model is what THIS process is serving, so it wins. The list
        // is still the right source for the full id set.
        const primary =
          (served && ids.includes(served) ? served : null) ??
          entries.find((entry) => entry?.is_default === true)?.id ??
          (ids.length === 1 ? ids[0] : null) ??
          served;
        return { modelIds: ids, primary: primary ?? null };
      }
    }

    if (served) return { modelIds: [served], primary: served };
    return { modelIds: [], primary: null };
  }

    // ⛔ `projectServedModel()` STOOD HERE AND IS DELETED ON PURPOSE.
  //
  // Its entire job was to copy the served /v1/models entry onto
  // `models.body.served` so a PROVENANCE row could address `served.*`. Exactly
  // ONE row ever did: `server.model_path`. That row is now a ban in
  // NEVER_BIND, because the directory is TRUE and must still never be shown --
  // on loopback it is the operator's absolute home path.
  //
  // With the row gone this function had no consumer, and keeping it would have
  // been worse than useless: it LIFTED the absolute path out of a list nobody
  // addresses and pinned it at a fixed, guessable location on the parsed body,
  // where the next panel to read the raw body would find it sitting ready. The
  // ban stops a field being BOUND; deleting this stops it being ADDRESSABLE.
  //
  // Nothing about model attribution was lost. `readAttribution()` above is a
  // separate function, still called on every cycle, and it is what carries
  // @12e42da8's finding that the implicit default is the ALPHABETICALLY FIRST
  // id -- so a reader of index 0 describes the wrong model on the scatter
  // server while looking entirely correct. That knowledge is load-bearing and
  // it did not live here.

  /**
   * Explain, in terms a developer can act on, that this file's audit is stale.
   *
   * @param {string} key
   * @param {import('./telemetry-provenance.js').ProvenanceEntry} entry
   * @param {unknown} observed
   * @returns {string}
   */
  function describeStaleProvenance(key, entry, observed) {
    const expected = 'stubValue' in entry ? JSON.stringify(entry.stubValue) : 'no value at all';
    return (
      `"${key}" is classified ${entry.classification} in telemetry-provenance.js, which expects ` +
      `${expected} from ${entry.source}. The server sent ${JSON.stringify(observed)}. ` +
      'That means this field is now genuinely measured and the provenance table is out of date. ' +
      `Update its classification and re-read the evidence citation (${entry.evidence}). ` +
      'The value is being displayed rather than hidden, because suppressing a real measurement ' +
      'is the failure this check exists to catch -- but it is flagged until the table is fixed.'
    );
  }

  /** Record a stale-provenance finding, warning once per field rather than per poll. */
  function recordProvenanceWarning(key, warning) {
    if (provenanceWarnings.has(key)) return;
    provenanceWarnings.set(key, warning);
    console.warn(`[telemetry-store] provenance table is stale: ${warning}`);
  }

  /**
   * The one case where per-origin classification can go silently wrong.
   *
   * `origin` decides whether a prefix-cache zero is a real measurement or
   * structurally impossible. If the two servers are started with their ports
   * swapped, or a URL is shared with the wrong origin parameter, that decision
   * inverts and the page renders confident nonsense with no error anywhere.
   * The server's own model id is the check.
   */
  function warnOnAttributionMismatch() {
    if (mismatchWarned || !attribution.primary || !origin) return;
    const inferred = selfClassesFromModelId(attribution.primary).classes;
    if (inferred.length === 0 || inferred.includes(origin)) return;
    mismatchWarned = true;
    console.warn(
      `[telemetry-store] This store is configured as the "${origin}" server, but ` +
        `${baseUrl} reports model "${attribution.primary}", which looks like a ` +
        `"${inferred[0]}" server. Per-server classification is now probably inverted: ` +
        'prefix-cache and paged-KV fields will be labelled with the wrong kind of zero. ' +
        'Check the origin parameters in the URL against the servers that are actually running.',
    );
  }

  /**
   * The provenance metadata every field carries, regardless of its state.
   *
   * An unavailable field needs this just as much as a measured one: the audit
   * table and the hover must be able to say WHICH server and WHICH endpoint a
   * missing value would have come from, or "unavailable" is unfalsifiable.
   *
   * @param {object} entry A PROVENANCE entry.
   */
  function fieldMeta(entry) {
    return {
      source: entry.source,
      sourceClass: entry.derived ? SOURCE_CLASSES.DERIVED : SOURCE_CLASSES.SERVER,
      origin,
      // Server-asserted, never inferred. Null when the server has not answered
      // yet, which is honest -- an unattributed number is better than a wrong
      // attribution.
      originModelId: attribution.primary,
      label: entry.label,
      unit: entry.unit,
    };
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
    // Resolved BEFORE the loop so every field in this snapshot carries the same
    // attribution, rather than some fields having it and others not.
    attribution = readAttribution(sources);
    warnOnAttributionMismatch();

    /** @type {Record<string, import('./telemetry-field.js').TelemetryField>} */
    const fields = {};
    for (const key of allFieldKeys()) {
      // Resolve per-server overrides first: the same field can be a genuine
      // measurement on one server and structurally meaningless on the other.
      const entry = resolveForOrigin(PROVENANCE[key], origin);

      // Client-derived fields are produced after this loop, from other fields.
      if (entry.derived) continue;

      const suppressed = NEVER_MEASURED_CLASSIFICATIONS.includes(entry.classification);

      if (suppressed) {
        // Do NOT return here without looking at the wire first. This table is a
        // snapshot of server source, and the server team's job is to invalidate
        // it. If we suppress without checking, the day a field becomes real is
        // the day we start hiding a genuine measurement behind an em-dash --
        // and that failure is invisible, because it looks like caution.
        const source = sources[entry.source];
        const observed = source?.ok ? readEntryValue(source.body, entry) : undefined;

        // Absence is NOT evidence that a field became real: a proxy serving
        // HTML at /metrics, or a build predating the field, both read as
        // undefined. Only a value that is actually present can contradict the
        // table -- and treating absence as a contradiction crashes the poll
        // cycle, because there is nothing to render.
        //
        // AND AN EMPTY STRING IS AN ABSENCE, NOT A CONTRADICTION. This is the
        // one absence a `!== undefined && !== null` test cannot see, and the
        // consequence is the worst shape we have: `''` would satisfy `present`,
        // reach `measuredField`, and render the field's label followed by
        // NOTHING -- carrying `data-state="measured"`, which tells a visitor
        // that the blank IS the reading. An absence rendered as a trusted
        // value is the exact defect this whole layer exists to prevent, and it
        // would have arrived through the branch built to catch it.
        //
        // It is also not hypothetical upstream: `model_path_for_display`
        // (crates/onnx-genai-server/src/routes/admin.rs:31) ends in
        // `unwrap_or_default()`, so a path whose `file_name()` is `None` -- one
        // ending in `..` or a root -- emits exactly `""`. That fallback is
        // CORRECT; it fails closed rather than disclosing the full path. This
        // is the client honouring it instead of promoting its output.
        //
        // Note `0` and `false` are deliberately still present: they are real
        // readings for a numeric or boolean field, and suppressing them would
        // recreate the absent-versus-zero defect one layer down.
        const present = observed !== undefined && observed !== null && observed !== '';
        if (source?.ok && present && !matchesStub(entry, observed)) {
          const warning = describeStaleProvenance(key, entry, observed);
          recordProvenanceWarning(key, warning);
          // Show the value. Hiding a real number is the exact failure this
          // branch exists to catch -- but it is shown carrying the
          // disagreement, so it can never pass as an ordinary measurement.
          fields[key] = measuredField(observed, {
            ...fieldMeta(entry),
            observedAtMs: source.fetchedAtMs ?? timestampMs,
            provenanceWarning: warning,
          });
          continue;
        }

        fields[key] = neverMeasuredField(entry, fieldMeta(entry));
        continue;
      }

      const source = sources[entry.source];
      if (!source) {
        fields[key] = unavailableField(`This demo does not poll ${entry.source} yet.`, fieldMeta(entry));
        continue;
      }

      if (!source.ok) {
        fields[key] = unavailableField(describeSourceFailure(entry.source, source), fieldMeta(entry));
        continue;
      }

      const rawValue = readEntryValue(source.body, entry);
      if (rawValue === undefined || rawValue === null) {
        fields[key] = unavailableField(
          `${entry.source} responded, but carried no "${entry.metric ?? entry.path}" value. ` +
            'This server build may predate it.',
          fieldMeta(entry),
        );
        continue;
      }

      // A reused body must not claim this poll's timestamp -- otherwise a value
      // fetched 30 s ago looks as fresh as one fetched now, and staleness
      // becomes unrepresentable for exactly the fields most likely to be stale.
      fields[key] = measuredField(rawValue, {
        ...fieldMeta(entry),
        observedAtMs: source.fetchedAtMs ?? timestampMs,
      });
    }
    addDerivedThroughput(fields, timestampMs);
    suppressUndefinedHitRate(fields);
    return Object.freeze(fields);
  }

  /**
   * A hit rate computed from zero lookups is emitted as a literal 0.0 by BOTH
   * endpoints that report it — metrics.rs:301-305 and routes/admin.rs:126-130
   * each do `if lookups == 0 { 0.0 }`. So "nothing has been looked up yet" and
   * "we looked and hit nothing" are the SAME BYTES on the wire.
   *
   * Nothing downstream can recover the difference, so it has to be corrected
   * here, where the denominator is still in scope. Rendering an unqualified
   * "0% hit rate" for an idle server is a fabricated measurement.
   *
   * @param {Record<string, import('./telemetry-field.js').TelemetryField>} fields
   */
  function suppressUndefinedHitRate(fields) {
    const rate = fields['prefix_cache.hit_rate'];
    if (!rate || rate.state !== FIELD_STATES.MEASURED) return;

    const denominator = fields['metrics.prefix_cache_lookups'] ?? fields['prefix_cache.lookups'];
    if (!denominator || denominator.state !== FIELD_STATES.MEASURED) return;
    if (Number(denominator.value) !== 0) return;

    fields['prefix_cache.hit_rate'] = unavailableField(
      'The server reports a 0.0 hit rate, but nothing has been looked up yet, and it emits the ' +
        'same 0.0 in both cases (metrics.rs:301-305). A rate with a zero denominator is ' +
        'undefined, not zero.',
      {
        source: rate.source,
        sourceClass: rate.sourceClass,
        origin: rate.origin,
        label: rate.label,
        unit: rate.unit,
      },
    );
  }

  /**
   * Throughput, derived client-side by differentiating the cumulative token
   * counter between two polls.
   *
   * The server hardcodes `tokens_per_second: 0.0` because it records totals but
   * no rate (see telemetry-provenance.js). The totals are real, though, so the
   * rate is genuinely recoverable here — this is a real measurement of the
   * server, computed by us, and is labelled as derived rather than reported.
   *
   * On the first poll there is no previous sample, so the field is `pending`,
   * not `unavailable`: it resolves on the next tick, and an em-dash would
   * wrongly promise a number that is never coming.
   *
   * @param {Record<string, import('./telemetry-field.js').TelemetryField>} fields
   * @param {number} timestampMs
   */
  function addDerivedThroughput(fields, timestampMs) {
    const total = fields['metrics.tokens_generated_total'];
    const options = {
      source: ENDPOINTS.METRICS,
      sourceClass: SOURCE_CLASSES.DERIVED,
      origin,
      label: PROVENANCE['throughput.observed'].label,
      unit: 'tokens/s',
    };

    if (!total || total.state !== FIELD_STATES.MEASURED) {
      lastTokenSample = null;
      fields['throughput.observed'] = unavailableField(
        'Derived from the cumulative token counter on /metrics, which is not available.',
        options,
      );
      return;
    }

    const previous = lastTokenSample;
    lastTokenSample = { total: total.value, atMs: timestampMs };

    if (!previous) {
      fields['throughput.observed'] = pendingField(
        'Waiting for a second sample of the token counter to measure a rate.',
        options,
      );
      return;
    }

    const elapsedS = (timestampMs - previous.atMs) / 1000;
    const delta = total.value - previous.total;
    // A restarted server resets the counter; a negative delta is not a rate.
    if (elapsedS <= 0 || delta < 0) {
      fields['throughput.observed'] = pendingField(
        'The token counter reset, most likely a server restart. Re-measuring.',
        options,
      );
      return;
    }

    fields['throughput.observed'] = measuredField(delta / elapsedS, {
      ...options,
      observedAtMs: timestampMs,
      derivedFrom: ['metrics.tokens_generated_total'],
    });
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
    // A feature-gated endpoint is missing at BUILD time, so telling the visitor
    // to pass a flag would send them down a dead end — the fix is a rebuild.
    if (source.status === 404 && FEATURE_GATED_ENDPOINTS[path]) {
      return (
        `${path} is not compiled into this server. It is on by default; this build used ` +
        `--no-default-features without re-enabling the "${FEATURE_GATED_ENDPOINTS[path]}" ` +
        'feature. Rebuild with it to see these metrics.'
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
      // Resolve per-server overrides first: the same field can be a genuine
      // measurement on one server and structurally meaningless on the other.
      const entry = resolveForOrigin(PROVENANCE[key], origin);
      // catalogueMeta, not a hand-rolled `{source, unit}`: this frame dropped
      // `label` (so every field rendered as the literal word "value"), and it
      // defaulted `sourceClass` to SERVER, which made a DERIVED field claim the
      // server reported it. The second one is a provenance lie, not a caption
      // bug, and it is the reason this is the same helper the other frames use.
      fields[key] = unavailableField(reason, catalogueMeta(entry, origin));
    }
    return Object.freeze(fields);
  }

  /**
   * Age the field map because the server went away.
   *
   * A field that HAS a value goes stale, keeping its last reading and its
   * original timestamp. A field that has NO value gets its reason re-derived
   * rather than inherited: an unavailable field carried forward from an earlier
   * connection state keeps that state's explanation, so a visitor hovering
   * `queue.depth` on a dead server could be told "the server has no model
   * loaded" — a confident, specific, wrong answer, which is worse than none.
   *
   * @param {Readonly<Record<string, import('./telemetry-field.js').TelemetryField>>} fields
   * @param {string} reason
   */
  function ageFields(fields, reason) {
    /** @type {Record<string, import('./telemetry-field.js').TelemetryField>} */
    const aged = {};
    for (const key of allFieldKeys()) {
      // Resolve per-server overrides first: the same field can be a genuine
      // measurement on one server and structurally meaningless on the other.
      const entry = resolveForOrigin(PROVENANCE[key], origin);
      const field = fields[key];

      if (field && (field.state === FIELD_STATES.MEASURED || field.state === FIELD_STATES.STALE)) {
        aged[key] = staleField(field, reason);
        continue;
      }

      // Never-measurable fields keep their permanent explanation — it is true
      // regardless of whether the server is up.
      aged[key] = NEVER_MEASURED_CLASSIFICATIONS.includes(entry.classification)
        ? neverMeasuredField(entry, catalogueMeta(entry, origin))
        : pendingField(
            `The server at ${baseUrl} is not responding, so no measurement has arrived for this ` +
              'field yet. It will fill in when the server returns.',
            catalogueMeta(entry, origin),
          );
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
 * Build the field for a classification that can never yield a number.
 *
 * The mapping from classification to state lived in three places and had
 * already drifted in two of them, which is exactly the failure this collapses:
 * a bypassed field rendered `not-applicable` after a poll, `unavailable` on the
 * first frame, and `pending` when the server was down -- three answers for one
 * unchanging architectural fact, depending only on WHEN you looked.
 *
 * The distinction it encodes is the one @0837fdf9 and the Lead both insist on:
 * `unavailable` is a PROMISE (someone will plumb this), `not-applicable` is a
 * FACT (this path never consults that subsystem). Same em-dash, different
 * sentence, and only one of them is owed future work.
 *
 * The catalogue metadata every field carries on a frame with no server answer.
 *
 * The store has a `fieldMeta` for the healthy path, but it is a closure over
 * `attribution`, so the frames that exist BEFORE or WITHOUT a response each
 * hand-rolled `{source, unit}` instead: the first frame, the offline frame,
 * the never-measurable fields on both, and -- found last, a fix that did not
 * travel -- the NO_MODEL frame in `allUnavailable`. All of them dropped
 * `label`, and `renderField` falls back to the literal word "value".
 *
 * The count is deliberately not written here. Three sites were fixed and a
 * fourth survived for hours because the sentence said "three" and reviewers
 * read the sentence instead of the callers. `frame-metadata.test.js` censuses
 * every frame the store can build, so the number lives in a test that fails
 * rather than in prose that ages.
 *
 * That is the frame a visitor always sees and the one no test covered. Worse
 * for the never-measurable fields: they have no second frame to correct it, so
 * a documented zero announced itself as "value" permanently.
 *
 * `originModelId` is null rather than omitted: no server has spoken yet, and an
 * unattributed value is honest where a guessed attribution is not.
 *
 * @param {object} entry A PROVENANCE entry, already resolved for the origin.
 * @param {string} origin
 */
function catalogueMeta(entry, origin) {
  return {
    source: entry.source,
    sourceClass: entry.derived ? SOURCE_CLASSES.DERIVED : SOURCE_CLASSES.SERVER,
    origin,
    originModelId: null,
    label: entry.label,
    unit: entry.unit,
  };
}

/**
 * @param {{classification: string, reason: string}} entry
 * @param {object} meta
 */
function neverMeasuredField(entry, meta) {
  return entry.classification === 'STRUCTURALLY_BYPASSED'
    ? notApplicableField(entry.reason, meta)
    : unavailableField(entry.reason, meta);
}

/**
 * @param {string} baseUrl
 * @param {number} timestampMs
 * @param {string} origin Capability profile of the server this store polls.
 *   Required: classification is keyed on (field, profile), so a snapshot built
 *   without it mis-classifies every per-profile field on the FIRST frame — the
 *   one frame a visitor always sees.
 * @returns {TelemetrySnapshot}
 */
function initialSnapshot(baseUrl, timestampMs, origin) {
  /** @type {Record<string, import('./telemetry-field.js').TelemetryField>} */
  const fields = {};
  for (const key of allFieldKeys()) {
    const entry = resolveForOrigin(PROVENANCE[key], origin);
    // A measurable field before the first poll is PENDING — it will resolve on
    // its own. A documented zero is UNAVAILABLE from the very first frame,
    // because no amount of waiting will ever produce a value for it, and
    // showing a spinner for it would promise something that is never coming.
    fields[key] = NEVER_MEASURED_CLASSIFICATIONS.includes(entry.classification)
      ? neverMeasuredField(entry, catalogueMeta(entry, origin))
      : pendingField('Waiting for the first poll to complete.', catalogueMeta(entry, origin));
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
 * Read one provenance entry's raw value out of its source body, dispatching on
 * whether that source is JSON or parsed Prometheus text.
 *
 * Returns `undefined` when the value is absent so that `buildFields` can
 * explain the gap. It must never fall back to 0 — that is the failure mode the
 * whole provenance table exists to prevent.
 *
 * @param {any} body
 * @param {object} entry A PROVENANCE entry.
 * @returns {number|string|boolean|undefined}
 */
function readEntryValue(body, entry) {
  if (!entry.metric) {
    return entry.path ? readPath(body, entry.path) : undefined;
  }
  // Prometheus sources parse to a Map; anything else means the endpoint
  // returned a shape we did not expect, and guessing would be worse than a gap.
  if (!(body instanceof Map)) return undefined;

  if (entry.kind === 'histogram_mean') {
    const observed = histogramMean(body, entry.metric);
    // `null` here means zero observations recorded — a real state, but not a
    // measurement. Reporting 0s of latency for an idle server would be a lie.
    return observed === null ? undefined : observed.mean;
  }
  const value = scalarOf(body, entry.metric);
  return value === null ? undefined : value;
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
