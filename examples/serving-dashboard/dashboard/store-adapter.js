// Copyright (c) Microsoft Corporation.
//
// The store adapter — dashboard-side, and deliberately not part of the store.
//
// CONTRACT.md gives every panel `field(key)`, `subscribe(fn)`, `getSnapshot()`.
// Panels additionally need three things the store does not provide, and should
// not:
//
//   • series(key, windowMs) — sparkline history.
//   • requests()            — the client-observed per-request table.
//   • capability(name)      — "can this panel populate at all right now?"
//
// WHY HISTORY LIVES HERE RATHER THAN IN THE STORE
// Only sparklines need history, sparklines are dashboard-owned, and the cost of
// keeping it is a ring buffer per key that grows with the number of panels
// mounted. Putting it in the store would make every consumer pay for a feature
// one consumer uses, and would put the retention policy — the thing most likely
// to need tuning — on the far side of an ownership boundary. The store stays a
// pure projection of the latest poll, which is also what makes its snapshots
// safely freezable.
//
// WHAT THE ADAPTER REFUSES TO DO
// It never invents a sample. A poll where a field is unavailable, pending or
// stale records a GAP, not a value and not a zero. A sparkline drawn across a
// gap would assert continuity nobody measured — the chart equivalent of
// rendering a documented zero, and harder to spot because the eye reads a line
// as evidence.

import { derivedField, hasValue, numericValueOf, pendingField } from '../telemetry-field.js';
import { CONNECTION_STATES } from '../telemetry-store.js';
import { RENDER_STATES, normaliseState } from './field-state.js';

/** Samples retained per key. At 250 ms that is ~5 minutes of history. */
const DEFAULT_CAPACITY = 1200;

/**
 * @typedef {object} Series
 * @property {'ok'|'unavailable'} state
 * @property {number[]} t Sample timestamps, ascending.
 * @property {number[]} v Sample values, index-aligned with `t`.
 * @property {Array<[number, number]>} gaps Half-open [startMs, endMs) spans with no data.
 * @property {string} [reason] Why the series is unavailable.
 */

/**
 * Wrap a TelemetryStore in the interface the dashboard panels consume.
 *
 * The adapter subscribes to the store exactly once no matter how many panels
 * mount, so history is recorded once and every panel sees the same instants.
 * Panels that disagree about what time it is make the dashboard contradict
 * itself, which is the same defect as two independent polling loops.
 *
 * @param {object} telemetryStore The real store from telemetry-store.js.
 * @param {object} [options]
 * @param {number} [options.capacity] Samples retained per key.
 * @param {() => Array<object>} [options.requests] Supplies the client-observed
 *   request table. The scenario runner owns that data; when it is absent the
 *   requests panel says so rather than showing an empty table, because an empty
 *   table looks like "no traffic" when the truth is "not wired up".
 * @returns {object}
 */
export function adaptStore(telemetryStore, options = {}) {
  const capacity = options.capacity ?? DEFAULT_CAPACITY;
  /** @type {Map<string, {t: number[], v: number[], gaps: Array<[number, number]>, lastAt: number|null}>} */
  const histories = new Map();
  /** Observed gaps between snapshots, so the poll interval is measured rather than declared. */
  const observedIntervalsMs = [];
  let lastSnapshotAtMs = null;
  /** @type {Set<(snapshot: object) => void>} */
  const listeners = new Set();

  const unsubscribeFromStore = telemetryStore.subscribe((snapshot) => {
    record(snapshot);
    for (const listener of [...listeners]) {
      // Mirrors the store's own guarantee: one panel throwing must not stop the
      // others from updating. A safety net, not a licence.
      try {
        listener(snapshot);
      } catch (error) {
        console.error('[dashboard] panel subscriber threw', error);
      }
    }
  });

  /** @param {object} snapshot */
  function record(snapshot) {
    const at = snapshot?.timestampMs ?? Date.now();

    if (lastSnapshotAtMs !== null && at > lastSnapshotAtMs) {
      // The OBSERVED gap, not the configured one. The configured interval is a
      // statement of intent; under load the two diverge, and the whole point of
      // showing this number is to reveal that divergence.
      observedIntervalsMs.push(at - lastSnapshotAtMs);
      if (observedIntervalsMs.length > 32) observedIntervalsMs.shift();
    }
    lastSnapshotAtMs = at;
    for (const [key, field] of Object.entries(snapshot?.fields ?? {})) {
      const numeric = numericValueOf(field);
      if (numeric === null || !hasValue(field)) {
        markGap(key, at);
        continue;
      }
      if (field.state === 'stale') {
        // A stale field is the same reading arriving again. Plotting it would
        // draw a flat line the server never reported — the visual claim that
        // nothing changed, when the truth is that we stopped hearing.
        markGap(key, at);
        continue;
      }
      push(key, at, numeric);
    }
  }

  /** @param {string} key */
  function historyFor(key) {
    let history = histories.get(key);
    if (!history) {
      history = { t: [], v: [], gaps: [], lastAt: null };
      histories.set(key, history);
    }
    return history;
  }

  /** @param {string} key @param {number} at @param {number} value */
  function push(key, at, value) {
    const history = historyFor(key);
    history.t.push(at);
    history.v.push(value);
    history.lastAt = at;
    while (history.t.length > capacity) {
      history.t.shift();
      history.v.shift();
    }
  }

  /** @param {string} key @param {number} at */
  function markGap(key, at) {
    const history = historyFor(key);
    const previous = history.gaps[history.gaps.length - 1];
    if (previous && previous[1] >= (history.lastAt ?? -Infinity) && previous[1] <= at) {
      // Extend the open gap rather than accumulating one per poll, so a server
      // that is down for a minute produces one span and not 240.
      previous[1] = at;
      return;
    }
    history.gaps.push([history.lastAt ?? at, at]);
  }

  /**
   * Differentiate a cumulative counter over `windowMs` of recorded history.
   * @param {string} key
   * @param {number} windowMs
   * @returns {number|null} null when there is not yet enough evidence.
   */
  function computeRate(key, windowMs) {
    const history = histories.get(key);
    if (!history || history.t.length < 2) return null;

    const latestAt = history.t[history.t.length - 1];
    const cutoff = latestAt - windowMs;
    let first = history.t.length - 1;
    while (first > 0 && history.t[first - 1] >= cutoff) first -= 1;
    if (first === history.t.length - 1) return null;

    const elapsedMs = latestAt - history.t[first];
    if (elapsedMs <= 0) return null;

    const delta = history.v[history.v.length - 1] - history.v[first];
    // A counter that went backwards means the server restarted. A negative
    // token rate is nonsense and clamping to zero would claim an idleness we
    // never observed, so we decline to answer this cycle.
    if (delta < 0) return null;
    return (delta / elapsedMs) * 1000;
  }

  return {
    /** @param {string} key */
    field(key) {
      // client.* never reaches the store. The store answers an unknown key with
      // 'No field named "x" is published by this server build', which for a
      // CLIENT-measured value is a false accusation: it blames the server for a
      // number the browser owns and the server has never heard of. Same failure
      // as coalescing a missing scenario runner to an empty list — the dashboard
      // reporting its own gap as someone else's.
      if (key.startsWith('client.')) {
        return clientField(key, observedIntervalsMs);
      }
      return markStalledOrigin(telemetryStore.field(key), telemetryStore.getSnapshot().connection);
    },

    getSnapshot() {
      return telemetryStore.getSnapshot();
    },

    /**
     * @param {string} key
     * @param {number} [windowMs] Only samples newer than this are returned.
     * @returns {Series}
     */
    series(key, windowMs) {
      const history = histories.get(key);
      if (!history || history.t.length === 0) {
        const field = telemetryStore.field(key);
        // An empty history means three different things and they must not share
        // one treatment. If the metric is structurally bypassed the chart says
        // so and stops apologising; if the server cannot report it the chart is
        // hatched; and if it is simply too early the chart is PENDING, not
        // unavailable — a healthy metric that has not accumulated a sample yet
        // must never be drawn as unmeasurable.
        const state =
          field.state === 'not-applicable' || field.state === 'unavailable'
            ? field.state
            : 'pending';
        return {
          state,
          t: [],
          v: [],
          gaps: [],
          reason:
            state === 'pending'
              ? 'No samples have been recorded yet for this metric.'
              : field.reason,
        };
      }
      if (!windowMs) {
        return { state: 'ok', t: [...history.t], v: [...history.v], gaps: [...history.gaps] };
      }
      const cutoff = (history.t[history.t.length - 1] ?? 0) - windowMs;
      const from = history.t.findIndex((time) => time >= cutoff);
      const start = from < 0 ? history.t.length : from;
      return {
        state: 'ok',
        t: history.t.slice(start),
        v: history.v.slice(start),
        gaps: history.gaps.filter(([, end]) => end >= cutoff),
      };
    },

    /**
     * The per-second rate of change of a cumulative counter.
     *
     * This exists because the honest tokens/sec is not on the wire.
     * `throughput.tokens_per_second` is a DOCUMENTED_ZERO — the server emits a
     * literal 0.0 — while `metrics.tokens_generated_total` is genuinely
     * measured. Differentiating the counter we do have beats reading the number
     * we were handed, and the result is badged `derived` so nobody mistakes it
     * for something the server computed.
     *
     * Returns `pending` rather than 0 when there is not yet enough history:
     * two samples over a real interval are the minimum evidence for a rate, and
     * a rate reported from one sample is a guess wearing a unit.
     *
     * @param {string} key
     * @param {object} [rateOptions]
     * @param {number} [rateOptions.windowMs] History window to differentiate over.
     * @param {string} [rateOptions.unit]
     * @returns {object} A TelemetryField.
     */
    rate(key, rateOptions = {}) {
      const windowMs = rateOptions.windowMs ?? 5000;
      const unit = rateOptions.unit ?? null;
      const field = telemetryStore.field(key);

      // Contagion first: if the counter itself is unavailable or pending, the
      // rate inherits that and says which input it is waiting on.
      const gated = derivedField({ [key]: field }, () => 0, { unit });
      // not-applicable included: without it a structurally-bypassed counter
      // reaches computeRate and a rate is manufactured for a subsystem this
      // execution path never consults.
      const gatedState = normaliseState(gated.state);
      if (
        gatedState === RENDER_STATES.UNAVAILABLE ||
        gatedState === RENDER_STATES.PENDING ||
        gatedState === RENDER_STATES.NOT_APPLICABLE
      ) {
        return gated;
      }

      const computed = computeRate(key, windowMs);
      if (computed === null) {
        // Deliberately pending, not unavailable: history accumulates on its own,
        // so this resolves within a few polls. Telling a visitor to give up on a
        // number that is seconds away is as wrong as telling them to wait for
        // one that is never coming.
        return pendingField(
          'Not enough history yet to differentiate this counter. A rate needs two samples over ' +
            'a real interval; one sample is a guess wearing a unit.',
          { source: 'derived', unit },
        );
      }
      return derivedField({ [key]: field }, () => computed, { unit });
    },

    /**
     * The rate series matching `rate()` — pointwise deltas of a cumulative
     * counter, so a sparkline under a tok/s figure plots tok/s and not the
     * monotonically rising counter it was derived from. A rising line labelled
     * with a rate unit is a mislabelled chart, and the eye trusts the shape
     * long before it reads the axis.
     *
     * @param {string} key
     * @param {number} [windowMs]
     * @returns {Series}
     */
    rateSeries(key, windowMs) {
      const base = this.series(key, windowMs);
      if (normaliseState(base.state) !== RENDER_STATES.OK || base.t.length < 2) {
        return {
          state: 'unavailable',
          t: [],
          v: [],
          gaps: base.gaps ?? [],
          reason:
            base.reason ??
            'A rate series needs at least two samples; one sample has no slope to plot.',
        };
      }
      const t = [];
      const v = [];
      for (let index = 1; index < base.t.length; index += 1) {
        const elapsedMs = base.t[index] - base.t[index - 1];
        const delta = base.v[index] - base.v[index - 1];
        // Skip rather than plot: a zero-length interval has no rate, and a
        // counter reset has no meaningful one.
        if (elapsedMs <= 0 || delta < 0) continue;
        t.push(base.t[index]);
        v.push((delta / elapsedMs) * 1000);
      }
      if (t.length === 0) {
        return {
          state: 'unavailable',
          t: [],
          v: [],
          gaps: base.gaps ?? [],
          reason: 'No interval in the window yields a usable rate.',
        };
      }
      return { state: 'ok', t, v, gaps: base.gaps ?? [] };
    },

    requests() {
      return options.requests ? options.requests() : null;
    },

    connection() {
      return telemetryStore.getSnapshot().connection;
    },

    /**
     * Whether a panel's data can arrive at all.
     *
     * Derived from the field envelope rather than configured separately: a
     * capability that has to be kept in sync with the fields it describes will
     * eventually disagree with them, and the fields are the ones telling the
     * truth.
     *
     * `state` distinguishes the two ways a capability can be absent, because
     * they deserve opposite voices: 'unavailable' is apologetic and resolves
     * itself when someone does the plumbing, while 'not-applicable' means the
     * metric is meaningless on this execution path by design and no amount of
     * work will produce it. Callers must branch on `state`, never on the prose
     * in `reason` — the reason exists to be displayed, not parsed.
     *
     * @param {string} name
     * @returns {{available: boolean, state: string, reason: string}}
     */
    capability(name) {
      const keys = CAPABILITY_KEYS[name];
      if (!keys) {
        return { available: true, state: 'ok', reason: '' };
      }

      let sawNotApplicable = false;
      for (const key of keys) {
        const field = telemetryStore.field(key);
        if (field.state === 'not-applicable') {
          sawNotApplicable = true;
          continue;
        }
        if (field.state !== 'unavailable') {
          return { available: true, state: 'ok', reason: '' };
        }
      }

      // A capability whose every field is structurally bypassed is NOT the same
      // as one nobody has plumbed. Before this distinction existed, the prefix
      // cache panel on the scatter server rendered the apologetic treatment for
      // a metric that is correctly and permanently absent there.
      const explaining = sawNotApplicable
        ? keys.find((key) => telemetryStore.field(key).state === 'not-applicable')
        : keys[0];

      return {
        available: false,
        state: sawNotApplicable ? 'not-applicable' : 'unavailable',
        reason:
          telemetryStore.field(explaining).reason ?? 'This server build does not report it.',
      };
    },

    /** @param {(snapshot: object) => void} listener */
    subscribe(listener) {
      listeners.add(listener);
      const snapshot = telemetryStore.getSnapshot();
      // Match the store: deliver immediately so a panel paints real state on
      // its first frame rather than an empty one.
      try {
        listener(snapshot);
      } catch (error) {
        console.error('[dashboard] panel subscriber threw on first delivery', error);
      }
      return () => listeners.delete(listener);
    },

    /** @param {(requests: Array<object>|null) => void} listener */
    subscribeRequests(listener) {
      return this.subscribe(() => listener(this.requests()));
    },

    /** Release the single upstream subscription. Call when unmounting the page. */
    destroy() {
      listeners.clear();
      histories.clear();
      unsubscribeFromStore();
    },
  };
}

/**
 * Which field keys prove a capability is live. A panel is shown if ANY of its
 * keys is something other than `unavailable` — one working metric is worth a
 * panel, and `pending` must count as available or every panel would hide itself
 * for the first 250 ms and flicker into existence.
 */
const CAPABILITY_KEYS = Object.freeze({
  'kv-pages': ['kv.pages_used', 'kv.pages_total', 'kv.introspection'],
  // 'prefix-cache' is DELIBERATELY ABSENT. Its panel binds no telemetry at all
  // — the counters were ruled unshippable because the hit counter cannot
  // distinguish reuse from no-reuse: twelve requests, six of them deliberately
  // unique, produced +12 hits, one per completed generation. (That argument
  // needs no stopwatch. An earlier timing A/B was also cited here; its author
  // withdrew it after the re-run came back with the opposite sign, so it is
  // gone from this comment on purpose.) There is no capability to detect, and
  // listing it here would gate a static, always-true finding behind a live
  // probe — the panel would vanish exactly when the server is unreachable,
  // which is when its explanation is most useful.
  'batch-occupancy': ['batch.utilization', 'batch.active_size'],
  throughput: ['throughput.tokens_per_second', 'metrics.tokens_generated_total'],
});

/**
 * AC45(d) — a whole-origin stall marks EVERY field from that origin.
 *
 * The store marks a field stale when ITS poll fails. But a field whose endpoint
 * was not polled this cycle keeps its `ok` state and goes on looking live, so
 * every field can be individually honest while the page as a whole lies. This
 * is AC6 arriving through the transport layer rather than through a panel:
 * honesty was enforced per-field, and the failure is per-connection.
 *
 * It matters most with two origins. One server can die while the other stays
 * live, and without this the dead half keeps presenting its last good frame
 * indefinitely, right next to a half that is genuinely updating.
 *
 * Applied in the adapter rather than in each panel because a panel that forgot
 * would look completely normal — the values would simply be wrong.
 *
 * @param {any} field
 * @param {{state?: string, serverMessage?: string|null}|null|undefined} connection
 * @returns {any}
 */
function markStalledOrigin(field, connection) {
  const state = connection?.state;
  // Bound to the store's exported symbols rather than to string literals: the
  // field vocabulary already changed under this code once mid-session, and a
  // silently non-matching literal here fails OPEN — every value would keep
  // rendering as live through a total outage.
  if (!state || state === CONNECTION_STATES.CONNECTED || state === CONNECTION_STATES.CONNECTING) {
    return field;
  }
  // Only a currently-live value can be downgraded. Anything already stale keeps
  // its ORIGINAL observation time, and unavailable/pending/not-applicable are
  // saying something truer than "stale" already.
  if (normaliseState(field?.state) !== RENDER_STATES.OK) {
    return field;
  }
  return {
    ...field,
    state: 'stale',
    reason:
      state === CONNECTION_STATES.NO_MODEL
        ? 'The server is running but has no model loaded, so nothing is refreshing this value.'
        : 'The server stopped answering, so this is the last value we received rather than a current one.',
  };
}

/**
 * Values the BROWSER measures about itself. The server has no opinion on any of
 * these and must never be blamed for their absence.
 *
 * @param {string} key
 * @param {number[]} observedIntervalsMs
 * @returns {object}
 */
function clientField(key, observedIntervalsMs) {
  if (key === 'client.poll_interval_ms') {
    if (observedIntervalsMs.length === 0) {
      return {
        value: null,
        state: 'pending',
        source: 'client',
        unit: 'ms',
        label: 'Poll interval',
        reason: 'Two polls are needed before an interval can be observed.',
        observedAtMs: null,
      };
    }
    // Median, not mean: one long stall would drag an average and misreport the
    // typical cadence as worse than it is.
    const sorted = [...observedIntervalsMs].sort((a, b) => a - b);
    return {
      value: sorted[Math.floor(sorted.length / 2)],
      state: 'ok',
      source: 'client',
      unit: 'ms',
      label: 'Poll interval',
      reason: 'Measured in the browser as the median gap between polls.',
      observedAtMs: Date.now(),
    };
  }

  // Honestly unavailable, and the reason names the DASHBOARD as the gap. An
  // em-dash here is correct; blaming "this server build" for it was not.
  const reasons = {
    'client.poll_rtt_ms':
      'The dashboard does not yet time individual requests — the store does not expose per-request duration.',
    'client.dropped_frames':
      'The dashboard does not yet instrument repaint drops.',
  };
  return {
    value: null,
    state: 'unavailable',
    source: 'client',
    unit: null,
    label: key,
    reason: reasons[key] ?? 'This is a client-side value the dashboard does not measure yet.',
    observedAtMs: null,
  };
}
