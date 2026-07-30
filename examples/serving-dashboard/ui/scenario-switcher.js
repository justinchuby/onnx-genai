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

import { SCENARIOS, planScenario, scenarioHref, describeSubstitution } from '../scenario-origins.js';
import { SERVERS, launchCommandFor } from './launch-command.js';

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
  const {
    origins,
    currentScenarioId,
    currentOrigin,
    contradiction = null,
    substitution = null,
  } = options;

  rootElement.className = 'scenario-switcher';
  rootElement.replaceChildren();

  // A URL that misdescribes the server it points at is shown BEFORE the tabs.
  // The page has already overruled it -- capability comes from the server's
  // model id, not from a query parameter -- but silently correcting a bad link
  // leaves the visitor with a working page and a link that keeps lying.
  if (contradiction) {
    rootElement.append(buildContradictionNotice(contradiction));
  }

  // Rendered FIRST, above the tabs, and deliberately not folded into the
  // contradiction channel. They are different facts: a contradiction says the
  // URL misdescribes the server, this says we did not render what you asked
  // for. One string carrying two meanings is how the next reader learns to
  // distrust both.
  if (substitution) {
    rootElement.append(buildSubstitutionNotice(substitution));
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
 * The notice a visitor gets when their URL asked for a scenario we did not
 * render. `textContent`, because the rejected id is straight off the query
 * string and is quoted back on purpose.
 *
 * @param {import('../scenario-origins.js').ScenarioSubstitution} substitution
 */
function buildSubstitutionNotice(substitution) {
  const notice = document.createElement('p');
  notice.className = 'scenario-switcher__substitution';
  notice.dataset.substitution = substitution.kind;
  // `alert`, not `status`: this one contradicts an action the visitor just
  // took, so it should interrupt rather than wait for a pause in speech.
  notice.setAttribute('role', 'alert');
  notice.textContent = describeSubstitution(substitution);
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
  // Name the server in the heading when every unreachable scenario needs the
  // SAME one, which is the common case in a two-server demo. "The other server"
  // is vague the moment the body text below names it specifically, and a
  // heading that is less precise than the paragraph under it reads as a
  // different, larger problem than the one being described.
  const missingClasses = [...new Set(unreachable.map(({ plan }) => plan.scenario.serverClass))];
  const onlyServer = missingClasses.length === 1 ? SERVERS[missingClasses[0]] : null;
  const serverPhrase = onlyServer ? `the ${onlyServer.label}` : 'another server';
  heading.textContent =
    unreachable.length === 1
      ? `One scenario needs ${serverPhrase}`
      : `${unreachable.length} scenarios need ${serverPhrase}`;
  note.append(heading);

  const list = document.createElement('ul');
  list.className = 'scenario-switcher__note-list';
  for (const { plan } of unreachable) {
    const item = document.createElement('li');
    const name = document.createElement('strong');
    name.textContent = plan.scenario.label;
    const server = SERVERS[plan.scenario.serverClass];
    // Name the SPECIFIC server. "The other server" is unactionable once there
    // is more than one thing a visitor could start, and demo-spec.md:120
    // requires the dynamic-model server to be named as such.
    const needs = server ? ` — needs the ${server.label}. ` : ' — ';
    item.append(name, document.createTextNode(`${needs}${plan.scenario.why}`));
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

  // The exact command for each missing server, deduplicated: two scenarios
  // needing the same server is one thing to start, not two. Shown because
  // "run run-demo.sh" does not help a visitor who is deliberately running one
  // server by hand, which is the case where this note actually appears.
  for (const serverClass of missingClasses) {
    const server = SERVERS[serverClass];
    if (!server) continue;

    const heading2 = document.createElement('p');
    heading2.className = 'scenario-switcher__note-fix';
    heading2.textContent = `Or start the ${server.label} yourself — it demonstrates ${server.demonstrates}:`;
    note.append(heading2);

    const pre = document.createElement('pre');
    pre.className = 'scenario-switcher__note-command';
    pre.textContent = launchCommandFor(serverClass);
    note.append(pre);
  }

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
  // Name the server rather than saying "the other server". A screen-reader user
  // gets this string INSTEAD of the visual note, not in addition to it, so any
  // fact the note names must be here too or the text view is a downgrade.
  const remoteClasses = [
    ...new Set(
      reachable.filter(({ plan }) => plan.requiresNavigation).map(({ plan }) => plan.scenario.serverClass),
    ),
  ];
  const remoteServer = remoteClasses.length === 1 ? SERVERS[remoteClasses[0]] : null;
  const remotePhrase = remoteServer ? `the ${remoteServer.label}` : 'another server';
  parts.push(
    remote === 0
      ? 'on this server.'
      : `, ${local} on this server and ${remote} on ${remotePhrase}, which opens in a new page.`,
  );
  if (unreachable.length > 0) {
    const missingClasses = [...new Set(unreachable.map(({ plan }) => plan.scenario.serverClass))];
    const missingServer = missingClasses.length === 1 ? SERVERS[missingClasses[0]] : null;
    const missingPhrase = missingServer ? `the ${missingServer.label} is` : 'the servers that measure them are';
    parts.push(
      `${unreachable.length} more ${unreachable.length === 1 ? 'is' : 'are'} unavailable ` +
        `because ${missingPhrase} not running.`,
    );
  }
  return parts.join(' ').replace(/\s+,/g, ',');
}
