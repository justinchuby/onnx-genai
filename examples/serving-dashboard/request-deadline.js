// Every request this page makes carries a deadline, and the reason is a
// failure mode that none of our other machinery can see.
//
// A server that REFUSES a connection rejects, and the reconnect ladder handles
// it correctly -- attempts climb 9, 11, 11, 13. A server that ACCEPTS the
// socket and never answers does not reject. It pends. The `await` never
// returns, the `finally` never runs, the in-flight flag stays set, and the
// poll loop never re-arms: attempts freeze at 7, 7, 7, 7, forever. The page
// keeps showing its last numbers as if they were current, which is the one
// absence our five honest states cannot express.
//
// An inference server mid-long-generation is exactly the second kind. So the
// reconnect ladder is well built and structurally unreachable for the failure
// most likely to happen on stage.
//
// A deadline converts that hang into the transport error the existing
// machinery already handles correctly. It does not add a new failure path; it
// makes an existing one reachable.
//
// WHY THIS IS A MODULE AND NOT AN IDIOM. It was written once, correctly, in
// telemetry-store.js, and the other caller never heard about it -- `app.js`
// probed /health with a bare fetch on the boot path, before any panel mounts.
// A fix that cannot be imported applies exactly once, and this one had already
// failed to travel across a single repository. The census test in
// request-deadline.test.js is the other half: it fails if any shipped module
// grows a fetch that does not come through here.

/**
 * How long any single request may take before we call it a hang.
 *
 * Two seconds is longer than any healthy endpoint on this dashboard and far
 * shorter than a visitor's patience. The point is not to be precise; it is to
 * be FINITE, because the alternative is unbounded.
 */
export const DEFAULT_REQUEST_TIMEOUT_MS = 2_000;

/**
 * Raised when a request passed its deadline, so callers can tell "the server
 * went quiet" from "the network refused us" without parsing anybody's prose.
 *
 * The runtime's own abort text ("This operation was aborted") reads like the
 * dashboard did something wrong. A visitor needs to know the SERVER went
 * quiet, and for how long, because that is what distinguishes a hung
 * generation from a dead port.
 */
export class RequestTimeoutError extends Error {
  /** @param {number} timeoutMs */
  constructor(timeoutMs) {
    super(
      `no response within ${timeoutMs} ms — the server accepted the connection but never replied`,
    );
    this.name = 'RequestTimeoutError';
    this.timeoutMs = timeoutMs;
  }
}

/**
 * `fetch` with a deadline. Identical to `fetch` on every path except one: a
 * request that never settles rejects with RequestTimeoutError instead of
 * pending forever.
 *
 * @param {string|URL} input
 * @param {{
 *   fetchImpl?: typeof fetch,
 *   timeoutMs?: number,
 *   signal?: AbortSignal,
 *   headers?: Record<string, string>,
 *   cache?: RequestCache,
 * }} [options] `fetchImpl` and `timeoutMs` are consumed here; everything else
 *   is passed through to the underlying fetch untouched — EXCEPT `signal`,
 *   which is COMPOSED rather than forwarded or replaced. A caller's signal and
 *   this function's deadline both remain able to abort the request, and
 *   whichever fires first wins. Cancelling via your own signal raises the
 *   underlying abort error; only the deadline produces RequestTimeoutError.
 *
 *   Composition, not replacement, because the two signals answer different
 *   questions -- "the server went quiet" and "the caller stopped caring" --
 *   and a request can be subject to both at once. Dropping either one loses a
 *   guarantee somebody is relying on, silently and only under load.
 *
 *   Requires `AbortSignal.any` (Node 20+, current browsers), and only on the
 *   path where a caller actually passes a signal. An unsupporting runtime
 *   throws immediately and by name rather than degrading to the old behaviour,
 *   because the old behaviour was to discard the caller's signal without
 *   saying so, which is the defect this documents.
 * @returns {Promise<Response>}
 */
export async function fetchWithDeadline(input, options = {}) {
  const {
    fetchImpl = globalThis.fetch,
    timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS,
    signal: callerSignal,
    ...init
  } = options;

  const controller = new AbortController();
  // Tracked explicitly rather than read back off the abort reason: the reason
  // is a moving target across runtimes, and this decides a visitor-facing
  // sentence. A caller must never have to string-match a runtime's wording.
  let timedOut = false;
  const deadline = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  // Composed, never replaced. `{ ...init, signal: controller.signal }` reads as
  // correct and silently discards a caller's signal, because a spread whose
  // last writer is us wins every key we happen to name.
  //
  // `timedOut` is set ONLY by our own timer above, so a caller-initiated abort
  // still surfaces as the runtime's abort error rather than being relabelled a
  // server timeout. Attribution follows whoever actually fired.
  const signal = callerSignal
    ? AbortSignal.any([callerSignal, controller.signal])
    : controller.signal;

  try {
    return await fetchImpl(input, { ...init, signal });
  } catch (error) {
    if (timedOut) throw new RequestTimeoutError(timeoutMs);
    throw error;
  } finally {
    // Cleared on every path, including the successful ones that return from
    // inside the try.
    //
    // HONEST SCOPE OF THIS LINE, because it looks like it prevents more than
    // it does: a leaked timer would NOT abort a later request. Each call owns
    // its controller, and by the time a leaked timer fired, its signal would
    // already be detached from a settled fetch, so the abort would be inert.
    // What this prevents is timer ACCUMULATION over a long-running page.
    // No behavioural test can distinguish its presence from its absence, so
    // the limit is written here rather than asserted somewhere it would look
    // stronger than it is.
    clearTimeout(deadline);
  }
}
