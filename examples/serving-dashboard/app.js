// Copyright (c) Microsoft Corporation.
//
// Page shell bootstrap.
//
// Responsibilities, and deliberately nothing more:
//   1. refuse to run from file:// with a useful message rather than a blank page
//   2. create the ONE telemetry store the whole page shares
//   3. mount the shell chrome — model card, connection indicator, failure states
//   4. render the provenance footer from the provenance table itself
//
// It does NOT draw dashboard panels. Those are the dashboard developer's
// modules, mounted into [data-panel] elements via mount(rootElement,
// telemetryStore). Keeping the shell ignorant of panel internals is what lets
// both of us work in the same page without merge pain.

import { createTelemetryStore } from './telemetry-store.js';
import { CONNECTION_STATES } from './telemetry-store.js';
import { mountDashboard } from './dashboard/index.js';
import {
  SERVER_MODE_BY_CLASS,
  currentScenarioId,
  planScenario,
  reconcileSelfClasses,
  resolveOrigins,
  selfClassesFromModelId,
} from './scenario-origins.js';
import { mountFailureStates } from './ui/failure-state.js';
import { mountModelCard } from './ui/model-card.js';
import { mountScenarioSwitcher } from './ui/scenario-switcher.js';
import { PROVENANCE, allFieldKeys } from './telemetry-provenance.js';
import { FIELD_STATES } from './telemetry-field.js';

/** Poll cadence. The spec fixes the dashboard between 250 and 500 ms. */
const POLL_INTERVAL_MS = 250;

/** Colour is never the only channel — each state has a distinct glyph. */
const CONNECTION_GLYPHS = Object.freeze({
  [CONNECTION_STATES.CONNECTING]: '◍',
  [CONNECTION_STATES.CONNECTED]: '●',
  [CONNECTION_STATES.UNREACHABLE]: '✕',
  [CONNECTION_STATES.NO_MODEL]: '○',
});

const CLASSIFICATION_TEXT = Object.freeze({
  MEASURED: 'Measured by the server',
  DOCUMENTED_ZERO: 'Server sends a documented zero — shown as unavailable',
  NOT_PLUMBED: 'Exists in the process, not yet exposed over HTTP',
});

async function main() {
  const fileGuard = document.getElementById('file-protocol-guard');
  const app = document.getElementById('app');

  // The page is served BY the server it talks to, so a file:// origin means the
  // visitor opened index.html from disk. Nothing below would work, and every
  // fetch would fail with an opaque CORS error, so stop here with the fix.
  if (location.protocol === 'file:') {
    return;
  }

  if (fileGuard) fileGuard.hidden = true;
  if (app) app.hidden = false;

  // WHICH SERVER IS THIS? It has to be settled before the store exists, because
  // the provenance table classifies several fields differently per server --
  // prefix_cache.hashes is a genuine measurement on the dynamic server and
  // structurally not-applicable on the scatter one, from identical bytes.
  const detection = await determineSelfClasses();
  const selfClasses = detection.classes;
  const origins = resolveOrigins({ href: location.href, selfClasses });

  // Drop declarations this server has DISPROVED about itself. Without this the
  // page shows the contradiction notice and STILL offers a tab navigating back
  // here for a capability we now know is absent -- explaining a lie while
  // continuing to offer it.
  for (const serverClass of detection.discredited) {
    if (origins[serverClass] === location.origin) origins[serverClass] = null;
  }
  const scenarioId = currentScenarioId(location.href, selfClasses);
  const plan = planScenario(scenarioId, origins, location.origin);

  // Panels only ever read the server that served this page. A scenario backed
  // by the other server is a NAVIGATION, which is why no CORS config exists.
  const serverClass = selfClasses[0] ?? null;

  const telemetryStore = createTelemetryStore({
    pollIntervalMs: POLL_INTERVAL_MS,
    origin: serverClass,
  });

  mountFailureStates(requireElement('failure-state'), telemetryStore);
  mountModelCard(requireElement('model-card'), telemetryStore);
  // The switcher is the ONLY route to the other server's scenarios. Without it
  // planScenario's work reached nothing: `plan` was computed, stashed on
  // globalThis and never rendered, so a visitor on the scatter server had no
  // way to reach paged KV or prefix caching at all.
  const scenarioSwitcher = mountScenarioSwitcher(requireElement('scenario-switcher'), {
    origins,
    currentScenarioId: scenarioId,
    currentOrigin: location.origin,
    contradiction: detection.contradiction,
  });
  mountConnectionIndicator(requireElement('connection-indicator'), telemetryStore);
  renderProvenanceFooter(requireElement('provenance-table'));

  const dashboard = mountPanels(telemetryStore, serverClass);

  telemetryStore.start();

  // Exposed for the dashboard developer's modules and for manual poking in
  // DevTools. One store for the page -- panels must never create their own.
  globalThis.onnxGenAiDemo = {
    telemetryStore,
    origins,
    plan,
    serverClass,
    dashboard,
    scenarioSwitcher,
  };
}

/**
 * Which engine configurations this origin can serve.
 *
 * The IO half only: fetch what the URL claims and what the server reports,
 * then hand both to reconcileSelfClasses, which owns the decision and is
 * unit-tested there.
 *
 * @returns {Promise<{classes: string[], declared: string[], contradiction: string|null}>}
 */
async function determineSelfClasses() {
  // Parameters only: passing no selfClasses means same-origin defaults are not
  // applied yet, so this reads exactly what the launcher declared.
  const declaredOrigins = resolveOrigins({ href: location.href, selfClasses: [] });
  const declared = Object.entries(declaredOrigins)
    .filter(([, origin]) => origin === location.origin)
    .map(([serverClass]) => serverClass);

  let observed = [];
  let observedModelId = null;
  try {
    // /health is never gated and is the only ungated identity endpoint.
    const response = await fetch(new URL('/health', location.origin), {
      headers: { accept: 'application/json' },
    });
    if (response.ok) {
      const body = await response.json();
      observedModelId = body?.model ?? null;
      observed = selfClassesFromModelId(observedModelId).classes;
    }
  } catch {
    // Unreachable server. The failure states own this case and render the
    // launch command; mounting no panels is correct until it comes back.
  }

  return reconcileSelfClasses({
    declared,
    observed,
    observedModelId,
    origin: location.origin,
  });
}

/**
 * Create a host element per registered panel and mount into it.
 *
 * Hosts are generated rather than written in index.html so the shell cannot
 * disagree with the registry about which panels exist -- it previously did.
 *
 * @param {object} telemetryStore
 * @param {string|null} serverClass
 */
function mountPanels(telemetryStore, serverClass) {
  const grid = document.getElementById('panel-grid');
  if (!grid) return null;

  return mountDashboard({
    telemetryStore,
    mode: serverClass ? SERVER_MODE_BY_CLASS[serverClass] : undefined,
    resolveRoot(panel) {
      const host = document.createElement('div');
      host.dataset.panel = panel.id;
      grid.append(host);
      return host;
    },
  });
}

/**
 * The always-visible connection state. Distinct from the blocking failure
 * layer: this is the persistent "am I live right now" signal, including while
 * everything is fine, so its absence is never how a visitor learns something
 * broke.
 *
 * @param {HTMLElement} rootElement
 * @param {import('./telemetry-store.js').TelemetryStore} telemetryStore
 */
function mountConnectionIndicator(rootElement, telemetryStore) {
  const dot = document.createElement('span');
  dot.className = 'connection-indicator__dot';

  const label = document.createElement('span');
  label.className = 'connection-indicator__label';

  rootElement.append(dot, label);

  return telemetryStore.subscribe((snapshot) => {
    const { state, lastSuccessAtMs } = snapshot.connection;
    rootElement.dataset.state = state;
    // Shape as well as colour: a filled/hollow/slashed marker carries the same
    // information for a visitor who cannot distinguish the colours.
    dot.textContent = CONNECTION_GLYPHS[state] ?? '?';
    label.textContent = connectionLabel(state, lastSuccessAtMs);
  });
}

/** Colour is never the only channel — each state has a distinct glyph. */
/**
 * @param {string} state
 * @param {number|null} lastSuccessAtMs
 */
function connectionLabel(state, lastSuccessAtMs) {
  switch (state) {
    case CONNECTION_STATES.CONNECTED:
      return 'Live';
    case CONNECTION_STATES.CONNECTING:
      return 'Connecting…';
    case CONNECTION_STATES.NO_MODEL:
      return 'No model loaded';
    case CONNECTION_STATES.UNREACHABLE:
      return lastSuccessAtMs
        ? `Disconnected — last reading ${formatAge(Date.now() - lastSuccessAtMs)} ago`
        : 'Disconnected';
    default:
      return state;
  }
}

/** @param {number} ageMs */
function formatAge(ageMs) {
  const seconds = Math.max(0, Math.round(ageMs / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

/**
 * The footer provenance table, generated from telemetry-provenance.js rather
 * than hand-written.
 *
 * A hand-maintained honesty table drifts from the code and then becomes a more
 * confident lie than the thing it was written to prevent. Generating it means
 * the page cannot claim a field is measured unless the store would actually
 * treat it that way.
 *
 * @param {HTMLElement} rootElement
 */
function renderProvenanceFooter(rootElement) {
  const table = document.createElement('table');
  table.className = 'provenance-table';

  const caption = document.createElement('caption');
  caption.textContent =
    'Every value this page can display, where it comes from, and whether the server ' +
    'genuinely measures it. Generated from the source, not written by hand.';
  table.append(caption);

  table.append(
    tableRow('th', ['Metric', 'Source', 'Status', 'Evidence']),
  );

  const body = document.createElement('tbody');
  for (const key of allFieldKeys()) {
    const entry = PROVENANCE[key];
    const row = tableRow('td', [
      entry.label,
      entry.source,
      CLASSIFICATION_TEXT[entry.classification] ?? entry.classification,
      entry.evidence,
    ]);
    row.dataset.classification = entry.classification;
    row.dataset.renderState =
      entry.classification === 'MEASURED' ? FIELD_STATES.MEASURED : FIELD_STATES.UNAVAILABLE;
    body.append(row);
  }
  table.append(body);
  rootElement.replaceChildren(table);
}

/**
 * @param {'th'|'td'} cellTag
 * @param {string[]} values
 */
function tableRow(cellTag, values) {
  const row = document.createElement('tr');
  for (const value of values) {
    const cell = document.createElement(cellTag);
    cell.textContent = value;
    row.append(cell);
  }
  return row;
}

/** @param {string} id */
function requireElement(id) {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(
      `The page shell is missing #${id}. index.html and app.js have drifted apart — ` +
        'app.js mounts into ids declared in index.html.',
    );
  }
  return element;
}

// Called last: every module-level `const` above must be initialised before the
// shell mounts. Calling main() at the top of the file put the constants in the
// temporal dead zone and threw on first render.
// main() is async because the server's engine configuration must be known
// before the store is created. An unhandled rejection here would leave the page
// on the loading state with nothing in the console.
main().catch((error) => {
  console.error('[shell] the dashboard failed to start', error);
});
