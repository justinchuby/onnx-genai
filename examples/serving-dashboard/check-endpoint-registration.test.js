// THE CONSEQUENCE CHECK: every endpoint the dashboard polls must actually be
// REGISTERED by the launch command we document.
//
// WHY THIS EXISTS, AND WHY IT IS NOT A STRING MATCH. A proposal to drop
// `--enable-debug-endpoints` from run-demo.sh was verified by curling `/demo`
// and `/v1/status` -- both of which answer with or without the flag -- and
// nearly landed. The flag governs a THIRD set of routes: lib.rs registers
// `/v1/debug/*` only INSIDE `if state.config.enable_debug_endpoints`. Not
// "returns 403", not "returns empty" -- NOT REGISTERED, so a 404.
//
// The failure mode that produces is the worst one available to this project:
// the server starts, `/demo` serves the page, `/v1/status` answers, every
// smoke test passes -- AND EVERY KV AND MODEL-CARD FIELD ON BOTH SERVERS
// DEGRADES TO UNAVAILABLE. A green suite and a page full of em-dashes.
//
// So this file does NOT assert that a flag string appears somewhere. A test
// that greps for `--enable-debug-endpoints` in a document proves nothing about
// whether the demo works; it would have passed on the day the flag was dropped
// from the launcher, and it would pass today if the launcher were correct and
// the server's gate were renamed. THE PROPERTY WORTH ENFORCING IS THE
// CONSEQUENCE: for each endpoint the dashboard polls, is there a launch flag
// in the documented command that causes that route to be registered?
//
// Three independent artefacts, none of which is allowed to vouch for another:
//   telemetry-provenance.js  what the dashboard polls    (the demand)
//   crates/.../lib.rs        what each flag registers    (the supply)
//   run-demo.sh + README     what we tell people to run  (the claim)

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..', '..');
const LIB_RS = join(REPO, 'crates', 'onnx-genai-server', 'src', 'lib.rs');

const LAUNCHER = readFileSync(join(HERE, 'run-demo.sh'), 'utf8');
const LIB = readFileSync(LIB_RS, 'utf8');

// THE DEMAND. Read out of the module rather than restated, so adding an
// endpoint to the dashboard automatically extends this check. A hardcoded list
// here would silently stop covering whatever was added last.
const { ENDPOINTS, FEATURE_GATED_ENDPOINTS } = await import('./telemetry-provenance.js');

// THE SUPPLY. Parse lib.rs into (route -> gating flag or null). A route inside
// `if state.config.enable_X_endpoints { ... }` needs `--enable-X-endpoints`;
// one outside every such block is unconditional.
function routeGates(rust) {
  const lines = rust.split('\n');
  const gates = new Map();
  let currentGate = null;
  let depth = 0;

  for (const line of lines) {
    const opening = line.match(/if\s+state\.config\.enable_([a-z_]+)_endpoints\s*\{/);
    if (opening) {
      currentGate = `--enable-${opening[1].replace(/_/g, '-')}-endpoints`;
      depth = 1;
      continue;
    }
    if (currentGate) {
      // Track brace depth so the gate closes at the right place rather than at
      // the first `}` belonging to a nested builder call.
      depth += (line.match(/\{/g) ?? []).length;
      depth -= (line.match(/\}/g) ?? []).length;
      if (depth <= 0) {
        currentGate = null;
        continue;
      }
    }
    const route = line.match(/\.route\(\s*"([^"]+)"/);
    if (route) gates.set(route[1], currentGate);
  }
  return gates;
}

// THE CLAIM. Only flags actually passed on a server-launch line count.
//
// ⚠️ THE FIRST VERSION OF THIS TESTED `LAUNCHER.includes(gate)` AND WAS GREEN
// ON THE EXACT DEFECT IT WAS BUILT TO CATCH. Removing
// `--enable-debug-endpoints` from BOTH server invocations left the check
// passing, because run-demo.sh MENTIONS the flag in two comments explaining
// what it does. The check proved the string was in the file; the property that
// matters is that the flag reaches the server. That is precisely the
// string-versus-property error this file exists to correct -- I made it inside
// the fix for it.
function launchFlags(script) {
  const lines = script.split('\n');
  const IN_COMMAND_POSITION =
    /^\s*(?:[A-Za-z_][A-Za-z0-9_]*=(?:"[^"]*"|'[^']*'|\S*)\s+)*"?\$\{?SERVER_BIN\}?"?(\s|$)/;
  const perLaunch = [];
  for (let i = 0; i < lines.length; i += 1) {
    if (/^\s*#/.test(lines[i])) continue;
    if (!IN_COMMAND_POSITION.test(lines[i])) continue;
    const block = [];
    let j = i;
    // Strip trailing comments so a `# --enable-x` note never counts as a flag.
    for (;;) {
      block.push(lines[j].replace(/#.*$/, ''));
      if (!/\\\s*$/.test(lines[j]) || j + 1 >= lines.length) break;
      j += 1;
    }
    perLaunch.push(block.join('\n'));
  }
  return perLaunch;
}

const LAUNCHES = launchFlags(LAUNCHER);
const GATES = routeGates(LIB);

test('the route table parses — the checker has something to check', () => {
  // THE EMPTY-INPUT GUARD. A checker that enumerates nothing passes everything,
  // silently, and grows more trusted with every green run. That is the purest
  // false green available: the other tests in this file iterate over ENDPOINTS
  // and assert per item, so an empty parse makes them vacuously true.
  assert.ok(
    GATES.size >= 15,
    `Only ${GATES.size} routes parsed out of lib.rs. The .route() matcher has ` +
      `stopped matching — every other assertion in this file iterates and is ` +
      `now vacuously true. Fix the parser before trusting a green run here.`,
  );
  // The launch parser needs its own empty guard. If it finds no launches,
  // `LAUNCHES.filter(...).length` is 0 for every flag and the main test passes
  // while checking nothing -- the same vacuous green, one artefact over.
  assert.ok(
    LAUNCHES.length >= 2,
    `Only ${LAUNCHES.length} server launch(es) parsed out of run-demo.sh. The ` +
      `demo runs two. With zero launches the flag assertion is vacuously true ` +
      `for every endpoint, so this file would approve a launcher that passes ` +
      `no flags at all.`,
  );

  assert.ok(
    [...GATES.values()].some((g) => g !== null),
    'No GATED routes parsed out of lib.rs — every route came back ' +
      'unconditional. The `if state.config.enable_*_endpoints` matcher is ' +
      'broken, so this file would happily approve a launch command with no ' +
      'flags at all.',
  );
});

test('every endpoint the dashboard polls is registered by the documented launch command', () => {
  const missing = [];

  for (const [name, path] of Object.entries(ENDPOINTS)) {
    // Feature-gated endpoints are a BUILD choice, not a launch flag; a 404
    // there is answered with a rebuild and the dashboard says so.
    if (path in FEATURE_GATED_ENDPOINTS) continue;

    if (!GATES.has(path)) {
      missing.push(
        `${name} (${path}) — the dashboard polls it, but NO route with that ` +
          `path is registered anywhere in lib.rs. Either the server route was ` +
          `renamed or the dashboard is polling a path that has never existed.`,
      );
      continue;
    }

    const gate = GATES.get(path);
    if (gate === null) continue; // unconditional; always registered

    // EVERY launch, not merely one of them. The demo runs two servers and the
    // dashboard polls both; a flag on only the first yields a half-dark page,
    // which reads as "that panel is broken" rather than "that server is
    // misconfigured".
    const without = LAUNCHES.filter((l) => !l.includes(gate)).length;
    if (without > 0) {
      missing.push(
        `${name} (${path}) is registered ONLY under \`${gate}\`, and ` +
          `${without} of ${LAUNCHES.length} server launches in run-demo.sh do ` +
          `not pass it. Those servers will start, /demo will serve, ` +
          `/v1/status will answer — and this endpoint will 404, so every ` +
          `field sourced from it degrades to unavailable.`,
      );
    }
  }

  assert.deepEqual(
    missing,
    [],
    `The documented launch command does not register ${missing.length} ` +
      `endpoint(s) the dashboard depends on:\n\n  ${missing.join('\n\n  ')}\n\n` +
      `A launch that 404s these still passes a /demo + /v1/status smoke test. ` +
      `That is a green suite and a page full of em-dashes.`,
  );
});

test('the debug-gated list in telemetry-provenance.js matches what lib.rs actually gates', async () => {
  // The dashboard tells the visitor WHICH FLAG to add when a route 404s. If
  // that list drifts from the server's real gating, the advice is confidently
  // wrong -- the visitor adds a flag and the field stays dark.
  const { DEBUG_GATED_ENDPOINTS, DEBUG_ENDPOINTS_FLAG } = await import(
    './telemetry-provenance.js'
  );

  for (const path of DEBUG_GATED_ENDPOINTS) {
    assert.equal(
      GATES.get(path),
      DEBUG_ENDPOINTS_FLAG,
      `telemetry-provenance.js says ${path} is behind ` +
        `${DEBUG_ENDPOINTS_FLAG}, but lib.rs registers it under ` +
        `${GATES.get(path) ?? 'no flag at all'}. The dashboard would tell a ` +
        `visitor to add a flag that does not fix their 404.`,
    );
  }

  // And the converse: an endpoint the dashboard polls that lib.rs puts behind
  // the debug flag must be DECLARED as debug-gated, or the dashboard will
  // report its 404 without telling the visitor how to fix it.
  for (const path of Object.values(ENDPOINTS)) {
    if (GATES.get(path) === DEBUG_ENDPOINTS_FLAG) {
      assert.ok(
        DEBUG_GATED_ENDPOINTS.includes(path),
        `lib.rs gates ${path} behind ${DEBUG_ENDPOINTS_FLAG}, but ` +
          `telemetry-provenance.js does not list it in DEBUG_GATED_ENDPOINTS. ` +
          `Its 404 would be reported to the visitor with no remedy.`,
      );
    }
  }
});
