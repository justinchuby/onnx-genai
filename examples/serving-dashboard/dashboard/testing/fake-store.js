// Copyright (c) Microsoft Corporation.
//
// A fake telemetry store for panel tests.
//
// Panels are pure consumers of the store contract (demo-ux.md §3.2), so this
// fake is all a panel test needs — and building panels against a fake first is
// how we found out early that the contract has to be total: `field()` must
// always return a Field and never `undefined`, or every panel needs a null
// check at every call site.
//
// This is a TEST HELPER, never loaded by the browser.

import { FIELD_STATES } from '../../telemetry-field.js';

/**
 * Build a measured field.
 *
 * @param {number|string} value
 * @param {object} [options]
 * @param {string} [options.source]
 * @param {string} [options.unit]
 * @param {string} [options.label]
 * @returns {object}
 */
export function measured(value, options = {}) {
  return {
    value,
    state: FIELD_STATES.MEASURED,
    source: options.source ?? 'server',
    unit: options.unit ?? '',
    label: options.label ?? '',
    at: Date.now(),
  };
}

/**
 * Build an unavailable field.
 *
 * @param {string} reason
 * @param {object} [options]
 * @param {string} [options.unit]
 * @param {string} [options.label]
 * @returns {object}
 */
export function unavailable(reason, options = {}) {
  return {
    value: null,
    state: 'unavailable',
    source: options.source ?? 'server',
    unit: options.unit ?? '',
    label: options.label ?? '',
    reason,
  };
}

/**
 * Build a series.
 *
 * @param {Array<[number, number]>} samples [timestampMs, value]
 * @param {object} [options]
 * @returns {object}
 */
export function series(samples, options = {}) {
  return {
    state: options.state ?? 'ok',
    t: samples.map(([time]) => time),
    v: samples.map(([, value]) => value),
    gaps: options.gaps ?? [],
    unit: options.unit ?? '',
    label: options.label ?? '',
    reason: options.reason,
  };
}

/**
 * Create a fake store.
 *
 * Unknown field paths resolve to an unavailable field rather than `undefined`,
 * exactly as the real store contract requires — so a panel asking for a field
 * nobody has plumbed yet renders an em-dash instead of crashing the page.
 *
 * @param {object} [spec]
 * @param {Record<string, object>} [spec.fields]
 * @param {Record<string, object>} [spec.series]
 * @param {Array<object>} [spec.requests]
 * @param {Record<string, {available: boolean, reason: string, fix?: string}>} [spec.capabilities]
 * @param {object} [spec.connection]
 * @returns {object}
 */
export function createFakeStore(spec = {}) {
  const fields = { ...(spec.fields ?? {}) };
  const seriesByPath = { ...(spec.series ?? {}) };
  const capabilities = { ...(spec.capabilities ?? {}) };
  /** @type {Array<() => void>} */
  const subscribers = [];

  return {
    field(path) {
      return (
        fields[path] ??
        unavailable(`No field is registered at "${path}".`, { label: path })
      );
    },
    series(path) {
      return (
        seriesByPath[path] ?? {
          state: 'unavailable',
          t: [],
          v: [],
          gaps: [],
          reason: `No series is registered at "${path}".`,
        }
      );
    },
    /**
     * Rate of a cumulative counter. The fake takes the answer directly via
     * `spec.rates`, because the panels' contract is "ask for a rate and get a
     * field" — re-deriving it here would test the adapter's arithmetic twice
     * and the panel's behaviour not at all.
     * @param {string} path
     */
    rate(path) {
      return (
        (spec.rates ?? {})[path] ??
        unavailable(`No rate is registered at "${path}".`, { label: path })
      );
    },
    /** @param {string} path */
    rateSeries(path) {
      return this.series(path);
    },
    requests() {
      // null, not [] — matching the adapter, which returns null when no scenario
      // runner is wired up. A fake that always hands back an array would let a
      // panel conflate "idle" with "unwired" and still pass every test.
      return spec.requests ?? null;
    },
    connection() {
      return spec.connection ?? { state: 'live', rttMs: 4, lastOkAt: Date.now(), attempt: 0 };
    },
    capability(name) {
      return capabilities[name] ?? { available: true, reason: '' };
    },
    subscribe(callback) {
      subscribers.push(callback);
      return () => {
        const index = subscribers.indexOf(callback);
        if (index >= 0) {
          subscribers.splice(index, 1);
        }
      };
    },
    subscribeRequests(callback) {
      return this.subscribe(callback);
    },

    // ── test controls ────────────────────────────────────────────────────────

    /** @param {string} path @param {object} field */
    setField(path, field) {
      fields[path] = field;
    },
    /** @param {string} path @param {object} value */
    setSeries(path, value) {
      seriesByPath[path] = value;
    },
    /** Notify every subscriber, as a poll tick would. */
    tick() {
      for (const callback of [...subscribers]) {
        callback({ at: Date.now() });
      }
    },
    /** How many subscriptions are still live — the leak check for destroy(). */
    subscriberCount() {
      return subscribers.length;
    },
  };
}
