// Copyright (c) Microsoft Corporation.
//
// The model card: which model the visitor is actually watching.
//
// Every number on this page reads differently depending on the answer, so the
// card sits in the header where it cannot be missed. Two of its fields — model
// directory and execution provider — are not exposed by any endpoint yet, so
// they render an em-dash with a hover explaining exactly why. That is the
// honest state, not a gap to be papered over with a guess: inferring "cpu"
// because it is the common case would be a fabricated fact about the machine
// the visitor is running on.
//
// This is a shell component, not a dashboard panel, but it follows the same
// mount(rootElement, telemetryStore) contract so there is one shape to learn.

import { FIELD_STATES } from '../telemetry-field.js';
import { formatField } from '../format.js';

/**
 * Fields shown on the card, in reading order: identity first, then the two
 * facts that change how every latency number should be interpreted.
 */
const CARD_FIELDS = Object.freeze([
  { key: 'server.model_id', label: 'Model' },
  { key: 'server.model_path', label: 'Directory' },
  { key: 'server.context_length', label: 'Context', format: formatTokenCount },
  { key: 'server.execution_provider', label: 'Execution provider' },
]);

/**
 * @param {HTMLElement} rootElement
 * @param {import('../telemetry-store.js').TelemetryStore} telemetryStore
 * @returns {{ unmount: () => void }}
 */
export function mountModelCard(rootElement, telemetryStore) {
  rootElement.className = 'model-card';
  rootElement.replaceChildren();

  const list = document.createElement('dl');
  list.className = 'model-card__fields';

  /** @type {Map<string, {value: HTMLElement, term: HTMLElement}>} */
  const cells = new Map();

  for (const { key, label } of CARD_FIELDS) {
    const term = document.createElement('dt');
    term.className = 'model-card__label';
    term.textContent = label;

    const value = document.createElement('dd');
    value.className = 'model-card__value';
    value.dataset.field = key;
    value.textContent = '···';

    list.append(term, value);
    cells.set(key, { value, term });
  }

  rootElement.append(list);

  const unsubscribe = telemetryStore.subscribe(() => {
    for (const { key, format } of CARD_FIELDS) {
      const cell = cells.get(key);
      if (!cell) continue;
      renderCardField(cell.value, telemetryStore.field(key), format);
    }
  });

  return {
    unmount() {
      unsubscribe();
      rootElement.replaceChildren();
    },
  };
}

/**
 * Render one card field.
 *
 * `data-state` is what the stylesheet hooks, so the em-dash and the pending
 * ellipsis look identical here and in every dashboard panel — one visual
 * language for absence across the whole page.
 *
 * @param {HTMLElement} element
 * @param {import('../telemetry-field.js').TelemetryField} field
 * @param {(value: any) => string} [format]
 */
function renderCardField(element, field, format) {
  const rendered = format ? formatField(field, { format }) : formatField(field);
  element.textContent = rendered.text;
  element.dataset.state = field.state;

  element.title = rendered.title;
  element.setAttribute('aria-label', rendered.title);

  // Absence is never silent: mark it so a reviewer can grep the DOM for any
  // field claiming a value it does not have.
  if (field.state === FIELD_STATES.UNAVAILABLE) {
    element.dataset.unavailableReason = field.reason ?? '';
  } else {
    delete element.dataset.unavailableReason;
  }
}

/**
 * @param {number} value
 * @returns {string}
 */
function formatTokenCount(value) {
  return `${Number(value).toLocaleString()} tokens`;
}
