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
import { mountFailureStates } from './ui/failure-state.js';
import { mountModelCard } from './ui/model-card.js';
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

function main() {
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

  const telemetryStore = createTelemetryStore({ pollIntervalMs: POLL_INTERVAL_MS });

  mountFailureStates(requireElement('failure-state'), telemetryStore);
  mountModelCard(requireElement('model-card'), telemetryStore);
  mountConnectionIndicator(requireElement('connection-indicator'), telemetryStore);
  renderProvenanceFooter(requireElement('provenance-table'));

  telemetryStore.start();

  // Exposed for the dashboard developer's modules and for manual poking in
  // DevTools. One store for the page — panels must never create their own.
  globalThis.onnxGenAiDemo = { telemetryStore };
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
main();
