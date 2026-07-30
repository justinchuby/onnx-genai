// Copyright (c) Microsoft Corporation.
//
// AC38: the launch command in the docs and the launch command in the UI must be
// the same command, so they cannot drift.
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
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  DEFAULT_SERVER_ADDRESS,
  RECOMMENDED_MODEL_DIR,
  LAUNCH_COMMAND,
  DEMO_URL,
} from './ui/launch-command.js';

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

  assert.equal(DEMO_URL, `http://${DEFAULT_SERVER_ADDRESS}/demo`);
  assert.ok(
    runDemoCode.includes('${BIND_HOST}:${SCATTER_PORT}/demo'),
    'run-demo.sh must print the /demo URL built from the same host and port',
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
  // Scoped to the two sources owned by the docs. index.html's file:// guard and
  // ui/launch-command.js are owned by the demo developer and still omit the
  // flag; widening this assertion to cover them is tracked with that owner, and
  // asserting it here today would only break their suite with their own bug.
  const owned = {
    'README.md': copyPasteableCommands['README.md'],
    'run-demo.sh': copyPasteableCommands['run-demo.sh'],
  };

  for (const [name, commands] of Object.entries(owned)) {
    assert.ok(
      commands.includes('--demo-assets-dir'),
      `${name} must pass --demo-assets-dir so the command works from any directory`,
    );
  }
});

test('run-demo.sh starts both servers, on distinct ports', () => {
  const scatterPort = DEFAULT_SERVER_ADDRESS.split(':')[1];
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
  // AC38 requires this to read as a decision rather than an oversight, so a
  // future contributor does not "fix" it by adding a bundler.
  assert.match(readme, /diffusion-demo/, 'README must name examples/diffusion-demo');
  assert.match(
    readme,
    /bundler/i,
    'README must explain why there is no bundler',
  );
});
