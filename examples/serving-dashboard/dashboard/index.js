// Copyright (c) Microsoft Corporation.
//
// The panel registry — the dashboard's only export to the page shell.
//
// The shell imports PANELS and mounts what it is given. It does not name panels
// individually, which is what lets a panel be added, reordered or made
// conditional without editing shared, two-owner code.
//
// WHY PANELS ARE FILTERED BY SERVER MODE
// Continuous batching and paged KV are mutually exclusive in this runtime:
// ContinuousBatchManager never touches engine.kv_cache, so a KV panel pointed at
// the batching server is not merely empty, it is structurally incapable of ever
// filling. The demo therefore runs two servers and the scenario switcher is the
// server switcher.
//
// We could have shown every panel everywhere and let the unavailable treatment
// explain the empties. We deliberately do not. A panel that can never populate
// on the server you are looking at is not missing data — it is the wrong panel,
// and an em-dash would imply "not yet" when the truth is "not here". Honest
// emptiness is for a metric that could arrive; absence is for one that cannot.
//
// This is also why `origin` is per-scenario configuration and no port is ever
// written down in this directory. If the two servers are later collapsed into
// one multi-model server, `modes` becomes the only thing that changes and no
// panel is touched.

import { createRovingGroup, setPanelView } from './panel-kit.js';
import { adaptStore } from './store-adapter.js';

import * as kvMemory from './kv-memory.js';
import * as requests from './requests.js';
import * as scheduling from './scheduling.js';
import * as system from './system.js';
import * as throughput from './throughput.js';

/**
 * @typedef {'batching'|'paged'} ServerMode
 *   'batching' — the scatter server (Scenario A: continuous batching).
 *   'paged'    — the dynamic server (Scenarios B and C: paged KV, prefix reuse).
 */

/**
 * @typedef {object} PanelModule
 * @property {object} meta
 * @property {(root: HTMLElement, store: object) => {destroy: () => void}} mount
 */

/**
 * @typedef {object} RegisteredPanel
 * @property {string} id
 * @property {PanelModule} module
 * @property {ReadonlyArray<ServerMode>} modes Server modes this panel can populate on.
 */

const BOTH = Object.freeze(['batching', 'paged']);

/**
 * Which server modes a panel can mount on, DERIVED from its own `meta.requires`.
 *
 * This used to be declared here by hand, separately from each panel's
 * `meta.requires`, and the two drifted -- silently, and in the worst possible
 * direction. `kv-memory` declared `requires: null` ("I ship everywhere") and
 * this table gated it to `['paged']`. The table won, because it is what
 * `panelsForMode` filters on.
 *
 * The consequence was that on the batching server -- the profile the default
 * model runs -- neither panel mounted AT ALL. A visitor never saw the prefix
 * cache, never saw the KV panel, and therefore never encountered the demo's
 * central technical claim: that continuous batching and the paged KV cache are
 * mutually exclusive execution paths. That claim is taught by those panels
 * rendering `not-applicable` WITH THEIR REASON. Hiding them replaces the
 * lesson with silence, and silence looks like a smaller dashboard rather than
 * like something withheld.
 *
 * Both suites stayed green throughout, because each mechanism was tested
 * against itself. Deriving removes the second mechanism rather than adding a
 * test to reconcile them: a panel now answers "where do I belong" exactly once,
 * in the file that also implements its behaviour.
 *
 * @param {PanelModule} module
 * @returns {ReadonlyArray<ServerMode>}
 */
export function modesFor(module) {
  switch (module.meta.requires) {
    case 'continuous-batch':
      return Object.freeze(['batching']);
    case 'paged-kv':
      return Object.freeze(['paged']);
    case null:
    case undefined:
      // Universal. Note this is the answer for every panel that ADAPTS rather
      // than disappears -- kv-memory draws a paged block table on one profile
      // and decode-row occupancy on the other.
      return BOTH;
    default:
      throw new Error(
        `${module.meta.id}: unknown meta.requires ${JSON.stringify(module.meta.requires)}. ` +
          'Refusing to guess where this panel belongs — guessing wrong hides it on a server ' +
          'where it had something to say, with no error anywhere.',
      );
  }
}

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
    Object.freeze({ id: module.meta.id, module, modes: modesFor(module) }),
  ),
);

/**
 * The panels that can genuinely populate on a given server mode, in DOM order.
 *
 * An unknown mode returns every panel rather than none: a registry that hides
 * the whole dashboard because a config string was misspelled fails in a way
 * that looks like a broken server, and a visible panel explaining its own
 * emptiness is far easier to diagnose than an empty page.
 *
 * @param {ServerMode|string} mode
 * @returns {ReadonlyArray<RegisteredPanel>}
 */
export function panelsForMode(mode) {
  if (mode !== 'batching' && mode !== 'paged') {
    return PANELS;
  }
  return PANELS.filter((panel) => panel.modes.includes(mode));
}

/**
 * Look up one registered panel.
 * @param {string} id
 * @returns {RegisteredPanel|undefined}
 */
export function panelById(id) {
  return PANELS.find((panel) => panel.id === id);
}

/**
 * Mount every panel that can populate on `mode`.
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
 * @param {ServerMode|string} [options.mode]
 * @param {() => Array<object>} [options.requests] Client-observed request table.
 * @returns {{unmount: () => void, mounted: ReadonlyArray<string>}}
 */
export function mountDashboard({ telemetryStore, resolveRoot, mode, requests }) {
  const adapter = adaptStore(telemetryStore, { requests });
  /** @type {Array<{id: string, handle: {unmount: () => void}, roving: {destroy: () => void}}>} */
  const mounted = [];

  for (const panel of panelsForMode(mode)) {
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
