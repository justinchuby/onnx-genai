// Copyright (c) Microsoft Corporation.
//
// The panel registry — the dashboard's only export to the page shell.
//
// The shell imports PANELS and mounts what it is given. It does not name panels
// individually, which is what lets a panel be added, reordered or made
// conditional without editing shared, two-owner code.
//
// WHY NO PANEL IS HIDDEN BY SERVER MODE
// This registry used to filter panels by server mode, so `scheduling` was
// removed from the DOM entirely on the paged server. That is gone, on a ruling
// and on measurement, and both halves matter.
//
// The MEASUREMENT, because it is the part that was never checked: the two
// servers serve an IDENTICAL field set. `/v1/status` on the paged origin
// returns the same keys as on the batching origin -- queue_depth,
// batch_utilization, batch_in_flight, batch_capacity -- and the same
// server-declared absences. So the scheduling panel does not merely have "an
// explanation to offer" on the paged server; it has the SAME NUMBERS it shows
// on the batching one. The gate was hiding measured data.
//
// The RULING: empty means a grid of unexplained em-dashes. A panel whose
// content is a sentence explaining why this architecture cannot produce a
// number is the opposite of empty -- it is the most informative thing on the
// page, and `collapseNotApplicableBody` renders exactly that when every field
// is structurally bypassed. Hiding replaces the demo's central technical claim
// with SILENCE, and silence reads as a smaller dashboard rather than as
// something withheld.
//
// The gate survived because a stale mental model outlived the vocabulary
// change and kept enforcing itself through a green test. It was justified by
// four-state reasoning -- em-dashes promise data that is not coming -- which is
// precisely the confusion the fifth state, `not-applicable`, was introduced to
// eliminate. The test was green because it faithfully tested the old
// vocabulary.
//
// Nothing replaces the mechanism. A second mechanism enforcing an invariant we
// already hold structurally is not defence in depth, it is a divergence waiting
// to happen -- which is literally what this was, two gating mechanisms
// disagreeing while both suites stayed green, because each was only ever tested
// against itself.
//
// This is also why `origin` is per-scenario configuration and no port is ever
// written down in this directory.

import { createRovingGroup, setPanelView } from './panel-kit.js';
import { adaptStore } from './store-adapter.js';

import * as kvMemory from './kv-memory.js';
import * as requests from './requests.js';
import * as scheduling from './scheduling.js';
import * as system from './system.js';
import * as throughput from './throughput.js';

/**
 * @typedef {object} PanelModule
 * @property {object} meta
 * @property {(root: HTMLElement, store: object) => {destroy: () => void}} mount
 */

/**
 * @typedef {object} RegisteredPanel
 * @property {string} id
 * @property {string} title Human-readable, for the roving group's accessible name.
 * @property {PanelModule} module
 */

/**
 * Panels in DOM order. The order is editorial, not alphabetical: a visitor
 * reads down the column, so it runs from the outcome (throughput) to the
 * mechanism that produced it (scheduling, memory, cache) and ends with the
 * evidence (per-request table) and the environment (system).
 *
 * @type {ReadonlyArray<RegisteredPanel>}
 */
export const PANELS = Object.freeze(
  [throughput, scheduling, kvMemory, requests, system].map((module) =>
    // `title` is lifted out of `meta` deliberately. The shell naturally
    // reaches for `panel.title`, and it silently read `undefined` for a while:
    // every roving group was built with no accessible name, which is invisible
    // on screen and only audible to a screen-reader user. Carrying the field
    // the callers actually reach for is cheaper than expecting each of them to
    // remember the extra hop through `.module.meta`.
    Object.freeze({
      id: module.meta.id,
      title: module.meta.title,
      module,
    }),
  ),
);

/**
 * Look up one registered panel.
 * @param {string} id
 * @returns {RegisteredPanel|undefined}
 */
export function panelById(id) {
  return PANELS.find((panel) => panel.id === id);
}

/**
 * Mount every registered panel.
 *
 * Every panel, unconditionally: the server mode does not decide what mounts.
 * `resolveRoot` returning null is the only way a panel is skipped, and that is
 * the shell saying "I have nowhere to put this", not the dashboard deciding a
 * visitor should not see it.
 *
 * The shell calls this instead of mounting panels itself, which is what
 * guarantees the store is adapted exactly once. Mounting panels against the raw
 * store would appear to work — `field()` and `subscribe()` are both there — and
 * then every sparkline would be permanently empty, because history is recorded
 * by the adapter and nowhere else. That is a failure with no error message, so
 * the API is shaped to make it hard to reach.
 *
 * @param {object} options
 * @param {object} options.telemetryStore The raw store from telemetry-store.js.
 * @param {(panel: RegisteredPanel) => HTMLElement|null} options.resolveRoot
 *   Given a panel, return the element it should own. Return null to skip it.
 * @param {() => Array<object>} [options.requests] Client-observed request table.
 * @returns {{unmount: () => void, mounted: ReadonlyArray<string>}}
 */
export function mountDashboard({ telemetryStore, resolveRoot, requests }) {
  const adapter = adaptStore(telemetryStore, { requests });
  /** @type {Array<{id: string, handle: {unmount: () => void}, roving: {destroy: () => void}}>} */
  const mounted = [];

  for (const panel of PANELS) {
    const root = resolveRoot(panel);
    if (!root) continue;
    const handle = panel.module.mount(root, adapter);
    // AC29 is applied centrally rather than panel by panel: a panel author who
    // forgets makes their whole panel a wall of tab stops, and nothing on
    // screen would look wrong. One group per panel means the dashboard costs a
    // keyboard user one Tab per panel, with arrows to read within it.
    const roving = createRovingGroup(root, { label: panel.title });
    mounted.push({ id: panel.id, handle, roving, root });
  }

  return {
    mounted: Object.freeze(mounted.map((entry) => entry.id)),

    /**
     * Drive the shell's uniform "view as table" toggle (AC28).
     *
     * The shell owns the control; the data lives in the panels, so it cannot
     * build the table itself from describe() alone.
     *
     * @param {'chart'|'table'} view
     * @param {string} [panelId] Omit to switch every panel at once.
     * @returns {number} Charts switched, so the shell can hide a toggle that
     *   would do nothing on a panel with no charts.
     */
    setView(view, panelId) {
      let switched = 0;
      for (const entry of mounted) {
        if (panelId && entry.id !== panelId) continue;
        switched += setPanelView(entry.root, view);
      }
      return switched;
    },
    unmount() {
      for (const entry of mounted) {
        // One panel failing to unmount must not strand the subscriptions of the
        // rest — an unmount path that gives up halfway is how a single-page app
        // leaks listeners across navigations.
        try {
          entry.roving.destroy();
          entry.handle.unmount();
        } catch (error) {
          console.error(`[dashboard] ${entry.id} threw during unmount`, error);
        }
      }
      mounted.length = 0;
      adapter.destroy();
    },
  };
}

export { kvMemory, requests, scheduling, system, throughput };
