// Copyright (c) Microsoft Corporation.
//
// AC40: the launch command in the docs and the launch command in the UI must be
// the same command, so they cannot drift.
//
// (This was AC38 before the spec reconstruction renumbered it. The substance is
// unchanged; the number is not. Verified against the spec rather than carried
// over from memory -- a citation nobody re-opens is how a test ends up
// enforcing a criterion that has moved.)
//
// The command necessarily appears in four places, each of which needs a
// different form, so a literal string comparison is not available:
//
//   run-demo.sh          the executable form, parameterised by environment
//   README.md            the human form, shown with concrete values
//   index.html           the file:// guard — must be inline static markup,
//                        because module scripts never run on a file:// URL
//   ui/launch-command.js the runtime form, rendered into the failure states
//
// So this test asserts the invariants that actually matter: same port, same
// model directory, same flags, and no accidental appearance of a flag we
// deliberately excluded. Those are the things a rename breaks silently.
//
// Run: node --test examples/serving-dashboard/check-launch-command.test.js

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { assertShippingTree } from './shipping-tree.mjs';

import {
  DEFAULT_SERVER_ADDRESS,
  RECOMMENDED_MODEL_DIR,
  LAUNCH_COMMAND,
  DEMO_URL,
  SERVERS,
  launchCommandFor,
} from './ui/launch-command.js';

// Provenance before content. Every path below is resolved from import.meta.url,
// so this file would read a parked worktree self-consistently and pass. Assert
// which tree we are in BEFORE asserting anything about what is in it.
assertShippingTree();

const demoDir = dirname(fileURLToPath(import.meta.url));
const read = (relativePath) => readFileSync(join(demoDir, relativePath), 'utf8');

const runDemoScript = read('run-demo.sh');
const readme = read('README.md');
const indexHtml = read('index.html');

/**
 * The three sources that spell the command out literally. `run-demo.sh` is
 * excluded here on purpose: it is parameterised by environment variables, so it
 * is asserted separately against the same values.
 */
const literalSources = {
  'README.md': readme,
  'index.html': indexHtml,
  'ui/launch-command.js': LAUNCH_COMMAND,
};

/**
 * Strip comments so an explanatory note about a flag is not mistaken for the
 * script passing it. `run-demo.sh` documents why it omits
 * `--enable-admin-endpoints`, and that sentence must not read as usage.
 */
function withoutShellComments(script) {
  return script
    .split('\n')
    .filter((line) => !/^\s*#/.test(line))
    .join('\n');
}

const runDemoCode = withoutShellComments(runDemoScript);

/** The model directory name, which survives every path form the sources use. */
const modelDirectoryName = RECOMMENDED_MODEL_DIR.split('/').pop();

test('the scatter server address is identical everywhere it appears', () => {
  const [, port] = DEFAULT_SERVER_ADDRESS.split(':');

  assert.match(
    runDemoCode,
    new RegExp(`SCATTER_PORT:-${port}\\b`),
    `run-demo.sh must default SCATTER_PORT to ${port} to match ui/launch-command.js`,
  );

  for (const [name, contents] of Object.entries(literalSources)) {
    assert.ok(
      contents.includes(DEFAULT_SERVER_ADDRESS),
      `${name} must contain the address ${DEFAULT_SERVER_ADDRESS}`,
    );
  }
});

test('the demo URL the visitor is told to open is the same in every source', () => {
  for (const [name, contents] of Object.entries(literalSources)) {
    if (name === 'ui/launch-command.js') continue; // it exports DEMO_URL separately
    assert.ok(
      contents.includes(DEMO_URL),
      `${name} must tell the visitor to open exactly ${DEMO_URL}`,
    );
  }

  // Trailing slash: this assertion previously demanded the SHORT form while
  // the trailing-slash test below demanded the long one, so the two could not
  // both pass. The long form is the ruled one.
  assert.equal(DEMO_URL, `http://${DEFAULT_SERVER_ADDRESS}/demo/`);

  // run-demo.sh builds its printed URL from the bound host and port rather
  // than a literal, via SCATTER_ORIGIN. Assert the DERIVATION, not one
  // particular spelling of it -- the script also appends the topology query
  // parameters that tell the page where the peer server landed.
  assert.ok(
    runDemoCode.includes('SCATTER_ORIGIN="http://${BIND_HOST}:${SCATTER_PORT}"'),
    'run-demo.sh must build its origin from the host and port it actually bound',
  );
  assert.ok(
    runDemoCode.includes('${SCATTER_ORIGIN}/demo/'),
    'run-demo.sh must print the /demo/ URL built from that origin',
  );
});

test('the static-cache model directory is identical everywhere', () => {
  // The batching scenario is silently wrong on any other model, so this string
  // drifting is a demo-breaking, non-obvious failure.
  for (const [name, contents] of Object.entries(literalSources)) {
    assert.ok(
      contents.includes(RECOMMENDED_MODEL_DIR),
      `${name} must reference the model directory ${RECOMMENDED_MODEL_DIR}`,
    );
  }

  assert.ok(
    runDemoCode.includes(modelDirectoryName),
    `run-demo.sh must resolve the static-cache model to ${modelDirectoryName}`,
  );
});

test('every source enables the debug endpoints', () => {
  // Without this flag the KV, prefix-cache and context-length fields degrade to
  // unavailable. A copy that omits it produces a correct but much emptier demo.
  for (const [name, commands] of Object.entries(copyPasteableCommands)) {
    assert.ok(
      commands.includes('--enable-debug-endpoints'),
      `${name} must pass --enable-debug-endpoints`,
    );
  }
});

/**
 * The text a visitor can actually copy and paste: fenced code blocks in the
 * README, `<pre>` in the guard markup, and the exported command itself. Prose
 * *about* a flag is not usage of it — both the README and `run-demo.sh`
 * deliberately explain why the admin flag is absent, and that explanation must
 * not be mistaken for the flag being present.
 */
const copyPasteableCommands = {
  'README.md': [...readme.matchAll(/```[a-z]*\n([\s\S]*?)```/g)].map((m) => m[1]).join('\n'),
  'index.html': [...indexHtml.matchAll(/<pre[^>]*>([\s\S]*?)<\/pre/g)].map((m) => m[1]).join('\n'),
  'ui/launch-command.js': LAUNCH_COMMAND,
  'run-demo.sh': runDemoCode,
};

test('no copy-pasteable command enables the admin endpoints', () => {
  // Deliberately excluded: the demo never calls /v1/admin/*, and the server has
  // no authentication. This test is what stops the flag reappearing by copy
  // paste from an older draft of the command.
  for (const [name, commands] of Object.entries(copyPasteableCommands)) {
    assert.ok(
      commands.length > 0,
      `${name} must contain at least one copy-pasteable command block`,
    );
    assert.ok(
      !commands.includes('--enable-admin-endpoints'),
      `${name} must not pass --enable-admin-endpoints: the demo does not use it, ` +
        'and the server ships without authentication',
    );
  }
});

test('every copy-pasteable command passes --demo-assets-dir', () => {
  // The server resolves demo assets relative to its WORKING DIRECTORY
  // (crates/onnx-genai-server/src/demo_assets.rs), so a documented command
  // without this flag only works when pasted from the repository root. That is
  // exactly the kind of silent, environment-dependent instruction this test
  // exists to stop shipping.
  //
  // NOW SCOPED TO ALL FOUR SOURCES. This assertion was deliberately narrowed to
  // the two doc-owned sources while index.html's file:// guard and
  // ui/launch-command.js still omitted the flag -- those belong to the demo
  // developer, and asserting it here would have broken their suite with a
  // defect they had not been told about yet. @bb2ee824 has since fixed both
  // (verified independently here, not taken on report), so the exemption is
  // spent and the narrow scope is now the only thing that could let it return.
  //
  // An exemption is a promise to come back. Left in place after its reason
  // expires it becomes indistinguishable from an oversight, and it is silent
  // either way -- the check stays green while covering less than its name says.
  //
  // Empty-input floor: `Object.entries({})` iterates zero times and this test
  // would pass without checking anything, growing more trusted with every green
  // run. Naming the four sources explicitly also makes DELETING one a failure
  // rather than a silent reduction in coverage.
  assert.deepEqual(
    Object.keys(copyPasteableCommands).sort(),
    ['README.md', 'index.html', 'run-demo.sh', 'ui/launch-command.js'],
    'the set of copy-pasteable command sources changed; this test must cover all of them',
  );

  for (const [name, commands] of Object.entries(copyPasteableCommands)) {
    assert.ok(
      commands.includes('--demo-assets-dir'),
      `${name} must pass --demo-assets-dir so the command works from any directory`,
    );
  }
});

/**
 * Every `onnx-genai-server` invocation in the README, as its own command.
 *
 * The test above asks whether the FILE mentions `--demo-assets-dir`. That is a
 * broader question than the one that matters: a README showing two servers and
 * flagging only the first passes it while shipping the exact defect. The two
 * come apart because a flag is a property of an INVOCATION and the old check
 * measured a DOCUMENT.
 */
function readmeServerInvocations() {
  const lines = readme.split('\n');
  const invocations = [];
  for (let i = 0; i < lines.length; i += 1) {
    if (!/onnx-genai-server\b/.test(lines[i])) continue;
    // Prose mentions and cargo build lines are not launches.
    if (!/\\\s*$/.test(lines[i])) continue;
    const command = [lines[i]];
    while (/\\\s*$/.test(command[command.length - 1]) && i + 1 < lines.length) {
      i += 1;
      command.push(lines[i]);
    }
    invocations.push(command.join('\n'));
  }
  return invocations;
}

test('every README server invocation passes --demo-assets-dir, absolutely', () => {
  const invocations = readmeServerInvocations();

  // Empty-input floor. A regex that stops matching would otherwise make this
  // test pass by finding nothing to check.
  assert.equal(
    invocations.length,
    2,
    'the README should show exactly two server launches (scatter and dynamic); ' +
      'if that changed deliberately, update this count',
  );

  for (const command of invocations) {
    const flag = /--demo-assets-dir\s+("?[^"\s\\]+"?)/.exec(command);

    // Under our navigation model a scenario switch LOADS the other server's own
    // /demo. A server started without this flag therefore serves the
    // missing-assets page rather than the dashboard -- silently at launch, and
    // visibly on the first scenario switch, which is while the demo is being
    // watched. Flagging one server proves nothing about the other.
    assert.ok(
      flag,
      `a README server invocation omits --demo-assets-dir:\n${command}`,
    );

    const value = flag[1].replace(/"/g, '');
    assert.ok(
      value.startsWith('/') || value.startsWith('${') || value.startsWith('$('),
      `--demo-assets-dir must be absolute (or a variable resolved to one), got ` +
        `"${value}". A relative value only works from one directory, and the ` +
        `failure is a working API with a broken /demo -- which reads as a ` +
        `corrupt build, not a wrong path.`,
    );
  }
});

/**
 * Documented `node --test` invocations, by source document.
 *
 * `node --test` treats "no files matched" as SUCCESS. So a documented command
 * that globs the wrong directory does not error -- it prints a clean green in
 * milliseconds. CONTRACT.md shipped one glob and no working directory, which
 * ran 279 of 582 tests from the repository root and ZERO from the directory the
 * document itself lives in, exiting 0 both times.
 */
function documentedTestCommands() {
  const found = [];
  for (const name of ['README.md', 'CONTRACT.md']) {
    for (const line of read(name).split('\n')) {
      if (line.includes('node --test')) found.push({ name, line });
    }
  }
  return found;
}

test('every documented full-suite test command reaches the dashboard directory', () => {
  const commands = documentedTestCommands();

  assert.ok(
    commands.length >= 2,
    'no documented `node --test` command found; this check is inspecting nothing',
  );

  // Only the BROAD root glob is a full-suite claim. A deliberately narrow
  // selector like `check-*.test.js` is honest about covering a subset, so the
  // leading delimiter here is what separates "all root tests" from "some".
  const broadRootGlob = /(?:^|['"/])\*\.test\.js/;

  // NOT `line.includes('dashboard/')`. The demo lives in `serving-dashboard/`,
  // so that substring is already present in the very path being globbed --
  // `examples/serving-dashboard/*.test.js` "contains dashboard/" while globbing
  // nothing inside it. The first version of this test used that check, and a
  // landed mutation restoring the one-glob form left it GREEN. The leading
  // delimiter is what makes this a directory segment rather than a word ending.
  const panelGlob = /(?:^|['"/])dashboard\/\*\.test\.js/;

  let fullSuiteClaims = 0;
  for (const { name, line } of commands) {
    if (!broadRootGlob.test(line)) continue;
    fullSuiteClaims += 1;
    assert.ok(
      panelGlob.test(line),
      `${name} documents a full-suite command that never globs dashboard/, ` +
        `so it silently omits roughly half the suite AND EXITS 0:\n  ${line.trim()}`,
    );
  }

  assert.ok(
    fullSuiteClaims >= 2,
    `only ${fullSuiteClaims} full-suite command(s) matched the glob pattern; ` +
      `the matcher has probably drifted and this test is passing over an empty set`,
  );
});

test('run-demo.sh starts both servers, on distinct ports', () => {  const scatterPort = DEFAULT_SERVER_ADDRESS.split(':')[1];
  const dynamicPort = runDemoCode.match(/DYNAMIC_PORT:-(\d+)/)?.[1];

  assert.ok(dynamicPort, 'run-demo.sh must define a DYNAMIC_PORT default');
  assert.notEqual(
    dynamicPort,
    scatterPort,
    'the two servers must bind different ports; they are separate origins by design',
  );
  assert.ok(
    runDemoCode.includes('DYNAMIC_MODEL'),
    'run-demo.sh must start a dynamic-cache server for the paged KV and prefix scenarios',
  );
});

test('both servers bind loopback by default', () => {
  // --enable-debug-endpoints widens the surface of a server that has no auth.
  assert.match(
    runDemoCode,
    /BIND_HOST:-127\.0\.0\.1/,
    'run-demo.sh must bind loopback by default',
  );
});

test('the README documents the deliberate divergence from diffusion-demo', () => {
  // AC40 requires this to read as a decision rather than an oversight, so a
  // future contributor does not "fix" it by adding a bundler.
  assert.match(readme, /diffusion-demo/, 'README must name examples/diffusion-demo');
  assert.match(
    readme,
    /bundler/i,
    'README must explain why there is no bundler',
  );
});

// ---------------------------------------------------------------------------
// Every flag we document must actually exist in the server's CLI.
//
// This is the check that catches the most expensive kind of documentation bug:
// a flag that was real when it was written and has since been renamed or cut.
// Prose describing a deleted flag reads exactly like prose describing a live
// one, so nothing but a mechanical comparison against the CLI will find it.
//
// It is deliberately one-directional. Flags the server has and we do not
// document are fine — most of them are irrelevant to the demo. Flags we
// document and the server does not have are always a bug.
// ---------------------------------------------------------------------------

const repoRoot = join(demoDir, '..', '..');
const cliSource = readFileSync(
  join(repoRoot, 'crates/onnx-genai-server/src/cli.rs'),
  'utf8',
);

/**
 * clap names a flag after its field unless `long = "..."` overrides it. The two
 * must not both be accepted: treating the field name as valid when an override
 * exists would let a misspelled flag pass, which is precisely the drift this
 * test is for.
 */
const cliFlags = new Set();
for (const [, attr, field] of cliSource.matchAll(
  /#\[arg\(([\s\S]*?)\)\]\s*pub\s+([a-z0-9_]+)\s*:/g,
)) {
  const override = attr.match(/long\s*=\s*"([a-z0-9-]+)"/);
  cliFlags.add(`--${override ? override[1] : field.replaceAll('_', '-')}`);
}

/**
 * Yields only the commands that invoke the server binary, joining shell line
 * continuations first so a flag on its own line stays attached to its command.
 *
 * Scoping matters: the README also documents a Mobius model-build command whose
 * flags belong to a different tool entirely. Checking those against the server's
 * CLI would fail for a reason that has nothing to do with drift.
 */
function serverInvocations(text) {
  return text
    .replaceAll(/\\\n\s*/g, ' ')
    .split('\n')
    .filter((line) => /onnx-genai-server|SERVER_BIN|cargo run/.test(line))
    // `cargo build -p onnx-genai-server` names the binary but runs none of it,
    // so its flags belong to cargo rather than to the server.
    .filter((line) => !/\bcargo\s+(build|install)\b/.test(line))
    // `cargo run --release -p … -- --model …` carries cargo's own flags before
    // the `--` separator. Only what follows it is passed to the server.
    .map((line) => (line.includes(' -- ') ? line.slice(line.indexOf(' -- ') + 4) : line));
}

test('the CLI flag list was parsed at all', () => {
  // Guards the assertions below: if cli.rs moves or its shape changes, the
  // extraction would silently yield an empty set and every check would pass
  // vacuously. A test that cannot fail is worse than no test.
  assert.ok(
    cliFlags.size > 5,
    `expected to parse several flags from cli.rs, found ${cliFlags.size}`,
  );
  assert.ok(
    cliFlags.has('--model'),
    'expected --model among the parsed CLI flags; the extraction is wrong',
  );
});

test('every flag in a copy-pasteable command exists in the server CLI', () => {
  let checked = 0;
  for (const [name, text] of Object.entries(copyPasteableCommands)) {
    for (const invocation of serverInvocations(text)) {
      for (const [, flag] of invocation.matchAll(/(?<![\w-])(--[a-z][a-z0-9-]+)/g)) {
        checked += 1;
        assert.ok(
          cliFlags.has(flag),
          `${name} documents ${flag}, which does not exist in ` +
            'crates/onnx-genai-server/src/cli.rs. Either the flag was renamed ' +
            'or removed, or the docs invented it.',
        );
      }
    }
  }
  assert.ok(checked > 0, 'no server flags were checked; the extraction is wrong');
});

// ---------------------------------------------------------------------------
// The demo URL is `/demo/`, with the trailing slash.
//
// `/demo` is a *temporary redirect* to `/demo/` (lib.rs:82-84), so the short
// form works — which is exactly why it drifts back. It costs an extra
// round-trip on every scenario switch, and any test asserting a URL
// byte-for-byte will disagree with a document that omits it.
// ---------------------------------------------------------------------------

test('every documented demo URL carries the trailing slash', () => {
  const sources = {
    'README.md': readme,
    'run-demo.sh': runDemoScript,
    'ui/launch-command.js': DEMO_URL,
  };
  for (const [name, text] of Object.entries(sources)) {
    const short = [...text.matchAll(/https?:\/\/[^\s`'")>]*\/demo(?![/\w-])/g)];
    assert.equal(
      short.length,
      0,
      `${name} points at /demo without the trailing slash ` +
        `(${short.map((m) => m[0]).join(', ')}). Use /demo/ — the short form ` +
        'is a temporary redirect, so it works while costing a round-trip and ' +
        'breaking any byte-exact comparison.',
    );
  }
});

// AC40, added after the spec reconstruction surfaced a clause this suite did
// not yet enforce: "No server command in any document, UI string, or copy
// button may show a `genai_config.json` path."
//
// The trap is built by our own tooling. `onnx-genai generate ./m/genai_config.json`
// works, because resolve_model_dir() (cli/src/lib.rs:674) coerces a file to its
// parent. The server has no equivalent, so the CLI teaches a habit the server
// rejects. A config-file path in a copy-pasteable server command is therefore
// not a typo -- it is the single most likely first-run failure, pasted.
test('no copy-pasteable server command points --model at a config file', () => {
  for (const [name, text] of Object.entries(copyPasteableCommands)) {
    for (const invocation of serverInvocations(text)) {
      assert.doesNotMatch(
        invocation,
        /--model[= ]\S*\.json\b/,
        `${name} shows a server command pointing --model at a config file:\n` +
          `  ${invocation.trim()}\n` +
          `--model takes a model DIRECTORY. The onnx-genai CLI accepts a ` +
          `config-file path and coerces it to the parent; the server does not.`,
      );
    }
  }
});

// Release-gate check, requested by the lead after `--max-batch` circulated as
// "approved and surfaced" and was reasoned from twice before anyone grepped
// cli.rs for it: EVERY flag named anywhere in the demo's documents and UI
// strings must exist in the server CLI -- not merely the flags inside
// copy-pasteable commands.
//
// The earlier check only read pasteable blocks, so a flag discussed in prose
// could be invented, renamed, or retired with nothing to catch it. Prose is
// where flags are justified, which is exactly where a stale one is most
// convincing.
//
// Flags belonging to other tools are classified explicitly rather than
// skipped by pattern. The map is fail-closed: an unrecognised flag is a
// failure, so a new foreign flag must be named here rather than silently
// tolerated. An allowlist that swallows unknowns is the superset-of-reality
// bug that let an earlier version of this suite pass its own mutation.
const FOREIGN_FLAGS = new Map([
  ['--release', 'cargo'],
  ['--test', 'cargo / node --test'],
  ['--runtime', 'mobius build'],
  ['--fail', 'curl'],
  ['--silent', 'curl'],
  ['--max-time', 'curl'],
  ['--show-toplevel', 'git rev-parse'],
]);

const flagBearingDocuments = {
  'README.md': readme,
  'run-demo.sh': runDemoCode,
  'index.html': indexHtml,
  'ui/launch-command.js': LAUNCH_COMMAND,
  ...(existsSync(join(demoDir, 'QA-PLAN.md'))
    ? { 'QA-PLAN.md': read('QA-PLAN.md') }
    : {}),
};

test('every flag named in any demo document exists in the server CLI', () => {
  let checked = 0;
  for (const [name, text] of Object.entries(flagBearingDocuments)) {
    for (const [, flag] of text.matchAll(/(?<![\w-])(--[a-z][a-z0-9-]+)/g)) {
      if (FOREIGN_FLAGS.has(flag)) continue;
      checked += 1;
      assert.ok(
        cliFlags.has(flag),
        `${name} names ${flag}, which does not exist in ` +
          `crates/onnx-genai-server/src/cli.rs.\n` +
          `Either it was renamed or removed, or it was never built -- a flag ` +
          `described as "approved" reads exactly like a flag that shipped.\n` +
          `If it belongs to another tool, add it to FOREIGN_FLAGS with its owner.`,
      );
    }
  }
  assert.ok(checked > 10, `expected to check many flags, checked ${checked}`);
});

// Every `?scenario=` value the README publishes must be a real SCENARIOS key.
//
// This one has no symptom. `currentScenarioId` (scenario-origins.js:275)
// validates the parameter and SILENTLY FALLS BACK to a locally-servable
// scenario when it does not recognise it -- correct behaviour for a visitor
// typing a URL, and precisely why a wrong id in the README is dangerous: the
// link still loads, still renders a working dashboard, and quietly shows a
// DIFFERENT scenario than the prose around it describes. No error, no console
// warning, nothing to notice.
//
// The ids are also easy to get wrong because index.html's `data-scenario`
// attributes are a separate namespace that only partly coincides with them
// (`paged-kv-block-table` vs `paged-kv`). The URL keys are the ones that must
// match, so they are the ones asserted here, against the module itself rather
// than a copy of the list.
test('every ?scenario= value in the README is a real scenario id', async () => {
  const { SCENARIOS } = await import('./scenario-origins.js');
  const ids = Object.keys(SCENARIOS);
  assert.ok(ids.length >= 3, `expected several scenarios, got ${ids.length}`);

  const cited = [...readme.matchAll(/[?&]scenario=([a-z0-9-]+)/g)].map((m) => m[1]);
  assert.ok(cited.length > 0, 'expected the README to publish at least one scenario URL');

  for (const id of cited) {
    assert.ok(
      Object.hasOwn(SCENARIOS, id),
      `README publishes ?scenario=${id}, which is not a scenario id ` +
        `(${ids.join(', ')}).\nThe link will still load and show a working ` +
        `dashboard -- just not the scenario the surrounding prose describes.`,
    );
  }
});

// --- The second server ------------------------------------------------------
//
// The demo runs TWO servers because continuous batching and paged KV are
// mutually exclusive in this runtime. A single set of constants could only
// describe one of them, which is why the unreachable-scenario note used to say
// "the other server" -- unactionable once there is more than one thing a
// visitor could start.

test('both servers are defined, and they differ in every field that matters', () => {
  const { scatter, dynamic } = SERVERS;

  assert.notEqual(scatter.address, dynamic.address, 'two servers cannot share a port');
  assert.notEqual(scatter.modelDir, dynamic.modelDir, 'the model IS the difference between them');
  assert.notEqual(scatter.modelId, dynamic.modelId, 'attribution keys on the model id');

  // The scatter model must be a static-cache build or batching silently falls
  // back to the per-request path, and the demo disproves its own headline.
  assert.match(scatter.modelDir, /-scatter/, 'batching needs a -scatter model');
  assert.doesNotMatch(dynamic.modelDir, /-scatter/, 'paged KV needs a NON-scatter model');
});

test('each launch command names its own server, not the other one', () => {
  for (const [serverClass, server] of Object.entries(SERVERS)) {
    const command = launchCommandFor(serverClass);
    assert.ok(command.includes(`--model ${server.modelDir}`), `${serverClass}: model dir`);
    assert.ok(command.includes(`--addr ${server.address}`), `${serverClass}: address`);
    assert.ok(command.includes(`--model-id ${server.modelId}`), `${serverClass}: model id`);
    assert.ok(
      command.includes('--enable-debug-endpoints'),
      `${serverClass}: without this the KV and prefix-cache panels go unavailable`,
    );
    assert.ok(
      !command.includes('--enable-admin-endpoints'),
      `${serverClass}: nothing calls /v1/admin/* and the server has no auth`,
    );
  }
});

test('an unknown server class throws rather than rendering an empty command', () => {
  // A failure state that renders a blank <pre> looks like the page is still
  // loading, so the visitor waits instead of acting. Fail loudly at the source.
  assert.throws(() => launchCommandFor('nonexistent'), /Unknown server class/);
});

test('the scatter launch command is still the one the README and index.html quote', () => {
  // The refactor to a two-server registry must not have moved the string the
  // other three copies are checked against.
  assert.equal(LAUNCH_COMMAND, launchCommandFor('scatter'));
  assert.equal(DEFAULT_SERVER_ADDRESS, SERVERS.scatter.address);
  assert.equal(RECOMMENDED_MODEL_DIR, SERVERS.scatter.modelDir);
});

test('no placeholder path is presented as the command to paste', () => {
  // AC38: the command a visitor is told to paste must work VERBATIM. The
  // explanatory "under the hood" block legitimately carries a
  // `/path/to/onnx-genai/models` placeholder, because models are gitignored and
  // their location genuinely varies per machine -- but the CANONICAL command
  // must never contain one, or the rule that makes AC38 checkable dissolves.
  //
  // MUTATION: put `/path/to/...` into LAUNCH_COMMAND -> red.
  assert.ok(
    !/\/path\/to\//.test(LAUNCH_COMMAND),
    `The canonical launch command contains a placeholder path and therefore ` +
      `cannot be pasted as-is: ${LAUNCH_COMMAND}`,
  );

  // Every placeholder that DOES appear must be accompanied, within the five
  // lines above it, by an instruction to substitute something real. An
  // unexplained placeholder reads as a literal path.
  const lines = readme.split('\n');
  const unexplained = [];
  lines.forEach((line, i) => {
    if (!line.includes('/path/to/')) return;
    // Five lines, because the explaining prose usually sits above the ``` fence
    // and a blank line, not immediately above the command itself.
    const context = lines.slice(Math.max(0, i - 5), i).join(' ');
    if (
      !/point (this|the script|it) at|whichever|replace|substitute|your own|actually hold/i.test(
        context,
      )
    ) {
      unexplained.push(`README.md:${i + 1}  ${line.trim()}`);
    }
  });

  assert.deepEqual(
    unexplained,
    [],
    `These placeholder paths are not explained in the three lines above them, ` +
      `so a reader may paste them literally:\n  ${unexplained.join('\n  ')}`,
  );
});
