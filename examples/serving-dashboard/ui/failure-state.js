// Copyright (c) Microsoft Corporation.
//
// The two BLOCKING failure states.
//
// These are the most likely first thing a visitor ever sees, so they get the
// same care as the scenarios. A visitor who sees empty panels concludes the
// PRODUCT is broken; a visitor who sees a clear instruction concludes they
// missed a step. Same underlying condition, opposite impressions — and the
// second one is both truer and recoverable.
//
// They are full-stage, never a toast over dead charts, and they are kept
// DISTINCT from each other because they are different problems with different
// fixes:
//
//   unreachable — the server process is not there.       Fix: start it.
//   no-model    — the process is there with nothing loaded. Fix: pass --model.
//
// Collapsing them into one "something went wrong" panel would send the visitor
// looking in the wrong place.
//
// RULE: server-authored text is rendered VERBATIM via textContent, never
// paraphrased and never through innerHTML. The server's messages are written in
// a what/why/how style that is better than anything we would substitute, and a
// message the visitor can grep for is worth more than one we made prettier.

import { CONNECTION_STATES } from '../telemetry-store.js';
import { LAUNCH_COMMAND, DEMO_URL, MODEL_CHOICE_NOTE } from './launch-command.js';

/**
 * Mount the blocking-failure layer. It owns the whole stage when active and is
 * completely absent from the accessibility tree when not.
 *
 * @param {HTMLElement} rootElement
 * @param {import('../telemetry-store.js').TelemetryStore} telemetryStore
 * @returns {{ unmount: () => void }}
 */
export function mountFailureStates(rootElement, telemetryStore) {
  rootElement.hidden = true;
  rootElement.className = 'failure-state';
  // `alert` rather than `status`: this is a blocking condition the visitor must
  // act on, and it must interrupt a screen reader rather than wait its turn.
  rootElement.setAttribute('role', 'alert');
  rootElement.setAttribute('aria-live', 'assertive');

  /** @type {number|null} */
  let countdownTimer = null;
  let lastRenderedState = null;

  const unsubscribe = telemetryStore.subscribe((snapshot) => {
    render(snapshot);
  });

  return {
    unmount() {
      unsubscribe();
      stopCountdown();
      rootElement.replaceChildren();
    },
  };

  /** @param {import('../telemetry-store.js').TelemetrySnapshot} snapshot */
  function render(snapshot) {
    const { state } = snapshot.connection;

    if (state === CONNECTION_STATES.CONNECTED) {
      hide();
      return;
    }
    if (state === CONNECTION_STATES.CONNECTING) {
      // Never flash a scary error during normal startup. The shell shows its
      // own quiet loading state until the first poll resolves.
      hide();
      return;
    }

    // Rebuild only on an actual state change: the countdown updates itself, and
    // rebuilding the subtree 4x/second would steal focus from the copy button
    // and re-announce the alert to a screen reader on every poll.
    if (state !== lastRenderedState) {
      lastRenderedState = state;
      rootElement.replaceChildren(
        state === CONNECTION_STATES.UNREACHABLE
          ? buildUnreachable(snapshot)
          : buildNoModel(snapshot),
      );
      rootElement.hidden = false;
      document.body.dataset.blocked = state;
    }

    if (state === CONNECTION_STATES.UNREACHABLE) {
      updateCountdown(snapshot);
    }
  }

  function hide() {
    if (lastRenderedState === null) return;
    lastRenderedState = null;
    stopCountdown();
    rootElement.hidden = true;
    rootElement.replaceChildren();
    delete document.body.dataset.blocked;
  }

  /**
   * State (a): the server is not there.
   *
   * @param {import('../telemetry-store.js').TelemetrySnapshot} snapshot
   */
  function buildUnreachable(snapshot) {
    const { origin, transportError } = snapshot.connection;

    const panel = element('div', 'failure-state__panel');
    panel.append(
      element('p', 'failure-state__eyebrow', 'Not connected'),
      element('h1', 'failure-state__title', `Can't reach the onnx-genai server at ${origin}.`),
      element(
        'p',
        'failure-state__lead',
        'This page is served by the onnx-genai server itself, so if you can read this the ' +
          'server was running a moment ago and has since stopped or restarted.',
      ),
    );

    if (transportError) {
      panel.append(
        labelledVerbatim('Browser reported', transportError, 'failure-state__verbatim'),
      );
    }

    panel.append(
      element('h2', 'failure-state__subtitle', 'Start the server'),
      copyableCommand(LAUNCH_COMMAND),
      element('p', 'failure-state__note', MODEL_CHOICE_NOTE),
      element('p', 'failure-state__note', `Then open ${DEMO_URL}`),
      buildRetryIndicator(),
    );
    return panel;
  }

  /**
   * State (b): the server is there, with nothing loaded.
   *
   * @param {import('../telemetry-store.js').TelemetrySnapshot} snapshot
   */
  function buildNoModel(snapshot) {
    const { origin, serverMessage } = snapshot.connection;

    const panel = element('div', 'failure-state__panel');
    panel.append(
      element('p', 'failure-state__eyebrow', 'No model loaded'),
      element('h1', 'failure-state__title', 'The server is running but has no model loaded.'),
      element(
        'p',
        'failure-state__lead',
        `The onnx-genai server at ${origin} is answering requests, so this is not a connection ` +
          'problem. It was started without a model directory, so there is nothing to generate ' +
          'with and no runtime state to show.',
      ),
    );

    if (serverMessage) {
      panel.append(
        labelledVerbatim('The server says', serverMessage, 'failure-state__verbatim'),
      );
    }

    panel.append(
      element('h2', 'failure-state__subtitle', 'Restart it with a model directory'),
      copyableCommand(LAUNCH_COMMAND),
      element('p', 'failure-state__note', MODEL_CHOICE_NOTE),
      element(
        'p',
        'failure-state__note',
        'This page will recover on its own once a model is loaded — no refresh needed.',
      ),
    );
    return panel;
  }

  /** The visible auto-retry countdown. A silent retry looks like a hang. */
  function buildRetryIndicator() {
    const wrapper = element('p', 'failure-state__retry');
    wrapper.append(
      element('span', 'failure-state__retry-dot'),
      element('span', 'failure-state__retry-text', 'Reconnecting…'),
    );
    wrapper.dataset.role = 'retry-indicator';
    return wrapper;
  }

  /**
   * Tick the countdown once a second. Only the text node changes, so focus and
   * the screen-reader alert are never disturbed.
   *
   * @param {import('../telemetry-store.js').TelemetrySnapshot} snapshot
   */
  function updateCountdown(snapshot) {
    const text = rootElement.querySelector('.failure-state__retry-text');
    if (!text) return;
    stopCountdown();

    const tick = () => {
      const nextRetryAtMs = telemetryStore.getSnapshot().connection.nextRetryAtMs;
      const secondsLeft = nextRetryAtMs
        ? Math.max(0, Math.ceil((nextRetryAtMs - Date.now()) / 1000))
        : 0;
      const attempts = telemetryStore.getSnapshot().connection.consecutiveFailures;
      text.textContent =
        secondsLeft > 0
          ? `Reconnecting in ${secondsLeft}s — attempt ${attempts + 1}`
          : `Reconnecting… (attempt ${attempts + 1})`;
    };

    tick();
    countdownTimer = setInterval(tick, 1000);
    void snapshot;
  }

  function stopCountdown() {
    if (countdownTimer !== null) {
      clearInterval(countdownTimer);
      countdownTimer = null;
    }
  }
}

/**
 * A command block with a copy button.
 *
 * The command is set via textContent from the single LAUNCH_COMMAND constant,
 * so what the visitor copies is byte-identical to what the README documents.
 *
 * @param {string} command
 */
function copyableCommand(command) {
  const wrapper = element('div', 'failure-state__command');

  const pre = element('pre', 'failure-state__command-text');
  pre.tabIndex = 0;
  pre.textContent = command;
  pre.setAttribute('aria-label', 'Server launch command');

  const button = element('button', 'failure-state__copy', 'Copy');
  button.type = 'button';
  button.addEventListener('click', async () => {
    const copied = await copyToClipboard(command);
    // Say what actually happened. A "Copied!" that lied would be a small
    // version of the exact sin this demo is built to avoid.
    button.textContent = copied ? 'Copied' : 'Press ⌘/Ctrl+C';
    if (!copied) selectText(pre);
    setTimeout(() => {
      button.textContent = 'Copy';
    }, 2000);
  });

  wrapper.append(pre, button);
  return wrapper;
}

/**
 * Render text the server or browser authored, VERBATIM and visibly quoted, so
 * the visitor can tell our words from the system's.
 *
 * @param {string} label
 * @param {string} text
 * @param {string} className
 */
function labelledVerbatim(label, text, className) {
  const wrapper = element('figure', className);
  const caption = element('figcaption', `${className}-label`, label);
  const body = element('pre', `${className}-text`);
  body.textContent = text;
  wrapper.append(caption, body);
  return wrapper;
}

/**
 * @param {string} tagName
 * @param {string} className
 * @param {string} [text]
 */
function element(tagName, className, text) {
  const node = document.createElement(tagName);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * Clipboard write with a real fallback. `navigator.clipboard` is undefined on
 * insecure non-localhost origins, which is exactly where someone demoing from
 * another machine will be.
 *
 * @param {string} text
 * @returns {Promise<boolean>}
 */
async function copyToClipboard(text) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // Fall through to selection.
  }
  return false;
}

/** @param {HTMLElement} node */
function selectText(node) {
  const range = document.createRange();
  range.selectNodeContents(node);
  const selection = window.getSelection();
  selection?.removeAllRanges();
  selection?.addRange(range);
  node.focus();
}
