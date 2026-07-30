// Copyright (c) Microsoft Corporation.
//
// The scenario switcher: the only way to reach the other server.
//
// WHY THIS IS NAVIGATION AND NOT A CLICK HANDLER.
// The two engine configurations are mutually exclusive, so the demo runs two
// servers on two ports. Two ports are two ORIGINS, and this page has no CORS
// layer to cross between them -- deliberately, because same-origin serving was
// chosen precisely so none would be needed. A scenario backed by the other
// server therefore cannot be fetched; it can only be NAVIGATED to.
//
// That constraint buys something rather than costing something. A full page
// load cannot carry one server's numbers under the other server's heading,
// because nothing survives it. The rule holds by construction instead of by a
// teardown routine somebody has to remember to write.
//
// WHY UNREACHABLE SCENARIOS ARE NOT RENDERED AS DISABLED TABS.
// A greyed-out tab reads as an invitation the visitor is being refused. These
// scenarios are not refused, they are simply on a server that is not running --
// a launcher problem with a launcher fix. So they collapse into one note that
// says what is missing and the command that supplies it.

import { SCENARIOS, planScenario, scenarioHref } from '../scenario-origins.js';

/**
 * Render the scenario tabs and the note covering whatever is unreachable.
 *
 * @param {HTMLElement} rootElement
 * @param {object} options
 * @param {Record<string, string|null>} options.origins  From resolveOrigins.
 * @param {string} options.currentScenarioId
 * @param {string} options.currentOrigin
 * @param {string|null} [options.contradiction]  Set when the URL misdescribes
 *   the server it points at; the server's answer has already won.
 * @returns {{ unmount: () => void, describe: () => string }}
 */
export function mountScenarioSwitcher(rootElement, options) {
  const { origins, currentScenarioId, currentOrigin, contradiction = null } = options;

  rootElement.className = 'scenario-switcher';
  rootElement.replaceChildren();

  // A URL that misdescribes the server it points at is shown BEFORE the tabs.
  // The page has already overruled it -- capability comes from the server's
  // model id, not from a query parameter -- but silently correcting a bad link
  // leaves the visitor with a working page and a link that keeps lying.
  if (contradiction) {
    rootElement.append(buildContradictionNotice(contradiction));
  }

  const plans = Object.keys(SCENARIOS).map((id) => ({
    id,
    plan: planScenario(id, origins, currentOrigin),
  }));

  const reachable = plans.filter(({ plan }) => plan.available);
  const unreachable = plans.filter(({ plan }) => !plan.available);

  const list = document.createElement('ul');
  list.className = 'scenario-switcher__tabs';
  // Tabs are links, not an ARIA tablist: activating one loads a document.
  // Announcing them as tabs would promise in-place panel switching that this
  // control genuinely does not do.
  list.setAttribute('aria-label', 'Scenarios');

  for (const { id, plan } of reachable) {
    list.append(buildTab(id, plan, origins, currentScenarioId));
  }
  rootElement.append(list);

  if (unreachable.length > 0) {
    rootElement.append(buildUnreachableNote(unreachable));
  }

  return {
    unmount() {
      rootElement.replaceChildren();
    },
    describe() {
      return describeSwitcher(reachable, unreachable, currentScenarioId);
    },
  };
}

/**
 * The notice shown when the URL's declaration lost to the server's own answer.
 *
 * role="status" rather than "alert": nothing is broken and nothing is being
 * lost, so this must not interrupt a screen-reader user mid-sentence.
 *
 * @param {string} contradiction
 * @returns {HTMLElement}
 */
function buildContradictionNotice(contradiction) {
  const notice = document.createElement('p');
  notice.className = 'scenario-switcher__contradiction';
  notice.dataset.state = 'stale';
  notice.setAttribute('role', 'status');
  notice.textContent = contradiction;
  return notice;
}

/**
 * One tab. A same-origin scenario is a plain link within this page; a
 * cross-origin one is labelled as changing servers, because a visitor who
 * clicks and sees every number change deserves to have been told why.
 *
 * @param {string} id
 * @param {ReturnType<typeof planScenario>} plan
 * @param {Record<string, string|null>} origins
 * @param {string} currentScenarioId
 * @returns {HTMLLIElement}
 */
function buildTab(id, plan, origins, currentScenarioId) {
  const item = document.createElement('li');
  // No class: the styling hooks are data-remote and aria-current, so a class
  // here would be a name nothing reads.

  const link = document.createElement('a');
  link.className = 'scenario-switcher__tab';
  link.href = scenarioHref(id, origins) ?? '#';
  link.dataset.scenario = id;
  link.dataset.serverClass = plan.serverClass;

  const label = document.createElement('span');
  label.className = 'scenario-switcher__label';
  label.textContent = plan.scenario.label;
  link.append(label);

  if (id === currentScenarioId) {
    link.setAttribute('aria-current', 'page');
    item.dataset.current = 'true';
  }

  if (plan.requiresNavigation) {
    item.dataset.remote = 'true';

    // Marked in TEXT, not colour alone: the visitor is about to leave this
    // server, and that is the single most surprising thing this control does.
    const hint = document.createElement('span');
    hint.className = 'scenario-switcher__hint';
    hint.textContent = `on the ${plan.serverClass} server`;
    link.append(hint);

    link.title =
      `Opens the ${plan.serverClass} server at ${plan.baseUrl}. ` +
      `${plan.scenario.why} The two servers are separate processes, so this is a ` +
      'page load rather than a panel switch.';
  } else {
    link.title = plan.scenario.why;
  }

  item.append(link);
  return item;
}

/**
 * The one note covering every scenario whose server is not running.
 *
 * @param {{id: string, plan: ReturnType<typeof planScenario>}[]} unreachable
 * @returns {HTMLElement}
 */
function buildUnreachableNote(unreachable) {
  const note = document.createElement('aside');
  note.className = 'scenario-switcher__note';
  // Informational, not an alert: nothing is broken. The other server is simply
  // not running, and that is a choice the operator can make.
  note.dataset.state = 'not-applicable';

  const heading = document.createElement('h3');
  heading.className = 'scenario-switcher__note-heading';
  heading.textContent =
    unreachable.length === 1
      ? 'One scenario needs the other server'
      : `${unreachable.length} scenarios need the other server`;
  note.append(heading);

  const list = document.createElement('ul');
  list.className = 'scenario-switcher__note-list';
  for (const { plan } of unreachable) {
    const item = document.createElement('li');
    const name = document.createElement('strong');
    name.textContent = plan.scenario.label;
    item.append(name, document.createTextNode(` — ${plan.scenario.why}`));
    list.append(item);
  }
  note.append(list);

  const fix = document.createElement('p');
  fix.className = 'scenario-switcher__note-fix';
  fix.textContent =
    'Start the demo with examples/serving-dashboard/run-demo.sh. It starts both ' +
    'servers and prints a URL carrying their addresses, which is what makes these ' +
    'scenarios reachable from this page.';
  note.append(fix);

  return note;
}

/**
 * A plain-English sentence describing the switcher, for assistive technology
 * and for the shell's text view.
 *
 * @param {{id: string, plan: ReturnType<typeof planScenario>}[]} reachable
 * @param {{id: string, plan: ReturnType<typeof planScenario>}[]} unreachable
 * @param {string} currentScenarioId
 * @returns {string}
 */
export function describeSwitcher(reachable, unreachable, currentScenarioId) {
  if (reachable.length === 0) {
    return 'No scenarios are reachable, because no server is configured to serve one.';
  }

  const current = SCENARIOS[currentScenarioId];
  const currentLabel = current ? current.label : currentScenarioId;
  const local = reachable.filter(({ plan }) => !plan.requiresNavigation).length;
  const remote = reachable.length - local;

  const parts = [
    `Showing ${currentLabel}.`,
    `${reachable.length} ${reachable.length === 1 ? 'scenario is' : 'scenarios are'} available`,
  ];
  parts.push(
    remote === 0
      ? 'on this server.'
      : `, ${local} on this server and ${remote} on the other server, which opens in a new page.`,
  );
  if (unreachable.length > 0) {
    parts.push(
      `${unreachable.length} more ${unreachable.length === 1 ? 'is' : 'are'} unavailable ` +
        'because the server that measures them is not running.',
    );
  }
  return parts.join(' ').replace(/\s+,/g, ',');
}
