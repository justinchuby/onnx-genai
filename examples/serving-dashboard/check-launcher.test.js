// Guards the promises the README makes ABOUT run-demo.sh.
//
// Why this file exists. The README tells a visitor that `run-demo.sh` "works
// from any directory," and that promise rests entirely on one flag being
// present on every server launch with an ABSOLUTE value. The server's default
// for --demo-assets-dir is the RELATIVE path ./examples/serving-dashboard, so
// a launch that omits the flag still produces a perfectly healthy API on a
// port that answers -- and a dead /demo. With several worktrees on one machine
// that is not a hypothetical: most candidate roots yield a working server and
// a 404 page, which is the failure mode this whole demo exists to argue
// against, since a healthy-looking server is indistinguishable from a correct
// one until you load the page.
//
// Nothing else in the suite covers the launcher. Every other check reads the
// README against the Rust and the JavaScript; this one reads it against the
// shell script, which is the only artefact a visitor actually executes.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { assertShippingTree } from './shipping-tree.mjs';

// Provenance before content. Every path below is resolved from import.meta.url,
// so this file would read a parked worktree self-consistently and pass. Assert
// which tree we are in BEFORE asserting anything about what is in it.
assertShippingTree();

const HERE = dirname(fileURLToPath(import.meta.url));
const LAUNCHER = readFileSync(join(HERE, 'run-demo.sh'), 'utf8');
const README = readFileSync(join(HERE, 'README.md'), 'utf8');

// A server launch is a line that INVOKES the binary, plus its backslash
// continuations. Parsed structurally rather than by counting: if someone adds
// a third server, this finds it without the file being edited.
//
// ⚠️ "Mentions SERVER_BIN" is NOT the same as "launches SERVER_BIN", and the
// first version of this parser conflated them. It reported run-demo.sh:207 --
// `if [[ ! -x "${SERVER_BIN}" ]]` -- as a server started without
// --demo-assets-dir. That is an existence TEST, and acting on the report would
// have meant adding a server flag to a bracket expression: a checker
// manufacturing a real defect in correct code, carrying a failing test's
// authority. So the rule is command position: SERVER_BIN must be preceded on
// its line by nothing but environment assignments.
function serverLaunches(script) {
  const lines = script.split('\n');
  const IN_COMMAND_POSITION =
    /^\s*(?:[A-Za-z_][A-Za-z0-9_]*=(?:"[^"]*"|'[^']*'|\S*)\s+)*"?\$\{?SERVER_BIN\}?"?(\s|$)/;
  const launches = [];
  for (let i = 0; i < lines.length; i += 1) {
    if (/^\s*#/.test(lines[i])) continue;
    if (!IN_COMMAND_POSITION.test(lines[i])) continue;
    const block = [lines[i]];
    let j = i;
    while (/\\\s*$/.test(lines[j]) && j + 1 < lines.length) {
      j += 1;
      block.push(lines[j]);
    }
    launches.push({ line: i + 1, text: block.join('\n') });
  }
  return launches;
}

test('run-demo.sh actually launches the servers the README says it does', () => {
  const launches = serverLaunches(LAUNCHER);
  assert.ok(
    launches.length >= 2,
    `Expected run-demo.sh to launch at least two servers -- the README's whole ` +
      `premise is that two configurations run side by side -- but found ` +
      `${launches.length} invocation(s) of SERVER_BIN. Either the launcher ` +
      `regressed or this parser stopped recognising a launch; check both before ` +
      `assuming the second.`,
  );
});

test('every server launch passes --demo-assets-dir', () => {
  for (const launch of serverLaunches(LAUNCHER)) {
    assert.match(
      launch.text,
      /--demo-assets-dir/,
      `run-demo.sh:${launch.line} starts a server WITHOUT --demo-assets-dir. ` +
        `That server will answer its API perfectly and serve a 404 at /demo, ` +
        `because the flag's default is the RELATIVE path ./examples/serving-dashboard. ` +
        `README.md tells the visitor the launcher "works from any directory"; ` +
        `without this flag on EVERY launch that sentence is false for one of them.`,
    );
  }
});

test('the --demo-assets-dir value is absolute, not relative', () => {
  // SCRIPT_DIR is the only sanctioned value. It is built with `cd ... && pwd`,
  // which is what makes it absolute -- so the derivation is asserted too. A
  // variable with the right NAME and a relative value would pass a grep for
  // the name alone, and this check exists precisely because a plausible name
  // is not evidence about a value.
  assert.match(
    LAUNCHER,
    /SCRIPT_DIR="\$\(cd "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)" && pwd\)"/,
    `run-demo.sh no longer derives SCRIPT_DIR with 'cd ... && pwd'. That ` +
      `derivation is the ONLY reason --demo-assets-dir is absolute. If the ` +
      `derivation changed, the flag may still be present and still be wrong.`,
  );

  for (const launch of serverLaunches(LAUNCHER)) {
    const value = launch.text.match(/--demo-assets-dir\s+"?([^"\s\\]+)"?/);
    assert.ok(value, `run-demo.sh:${launch.line} has --demo-assets-dir with no parseable value.`);
    assert.ok(
      value[1].includes('SCRIPT_DIR') || value[1].startsWith('/'),
      `run-demo.sh:${launch.line} passes --demo-assets-dir ${value[1]}, which is ` +
        `not absolute. A relative value works from the repo root and 404s ` +
        `everywhere else -- the launch SUCCEEDS and the page is gone, which is ` +
        `the hardest version of this bug to notice.`,
    );
  }
});

test('the README still makes the claim this file is guarding', () => {
  // If the README stops promising directory-independence, the three checks
  // above are guarding nothing and should be reconsidered rather than left
  // running. A check whose premise has been deleted is not free: it still
  // passes, and it still looks like coverage.
  assert.match(
    README,
    /run-demo\.sh[^.]*always passes it, so it works from any directory/,
    `README.md no longer claims run-demo.sh "works from any directory". This ` +
      `file exists to hold that specific sentence to account. If the claim was ` +
      `deliberately dropped, delete these tests; do not leave them asserting a ` +
      `promise the documentation no longer makes.`,
  );
});

// Every `scenario=<id>` URL the launcher or the README hands a visitor must be
// a scenario that actually exists.
//
// Why this is a separate hazard from a broken link, and why it is worse. An
// unrecognised id does not 404 and does not warn: `currentScenarioId` falls
// back to the local server's own scenario and renders it perfectly. That
// fallback is CORRECT and is deliberately asserted in scenario-origins.test.js
// -- the prefix-cache id was public for a while, so a bookmark that outlived
// the cut has to land somewhere sane rather than stay reachable.
//
// But a fallback written for a STALE link is not a licence to print a FRESH
// one. run-demo.sh advertised `?scenario=prefix-cache` in its success banner
// -- the first thing an operator reads -- for a scenario deliberately moved to
// CUT_SCENARIOS with no `id:`, precisely so it could not be addressed. The
// visitor clicked "prefix caching" and got a flawless, fully honest page about
// paged KV, with nothing anywhere saying they were shown something else. A 404
// would have been kinder: it is the one failure our five-state vocabulary
// cannot narrate, because every field on that page IS correctly labelled.
//
// The cut was enforced in JavaScript and undone in a printf. Nothing reconciled
// the two, because every honesty mechanism we own inspects a FIELD -- none
// inspects a ROUTE, a LINK, or a launch banner, and the visitor's CHOICE is
// made before a single field is read.
test('every scenario URL we hand a visitor names a scenario that exists', async () => {
  const { SCENARIOS, CUT_SCENARIOS } = await import('./scenario-origins.js');

  const cited = [];
  for (const [label, source] of [
    ['run-demo.sh', LAUNCHER],
    ['README.md', README],
  ]) {
    for (const m of source.matchAll(/scenario=([a-z0-9-]+)/g)) {
      cited.push({ label, id: m[1] });
    }
  }

  // Anti-vacuity. If the extractor stops matching -- the URLs get templated,
  // the query key is renamed -- this check would pass by finding nothing, which
  // is the failure mode it is least able to report on itself.
  assert.ok(
    cited.length > 0,
    'no scenario= URLs found in run-demo.sh or README.md. Either both stopped ' +
      'advertising scenarios, or this extractor no longer matches how they are ' +
      'written. Both mean this check is guarding nothing.',
  );

  const bad = cited
    .filter(({ id }) => !Object.hasOwn(SCENARIOS, id))
    .map(({ label, id }) => {
      const why = Object.hasOwn(CUT_SCENARIOS, id)
        ? `'${id}' is in CUT_SCENARIOS -- it was deliberately made unaddressable`
        : `'${id}' is not a scenario at all -- likely a typo`;
      return `${label} advertises ?scenario=${id}. ${why}. The page will NOT ` +
        `error: it silently renders a different scenario, correctly labelled, ` +
        `and the visitor has no way to tell. Remove the link, or register the id.`;
    });

  assert.deepEqual(bad, []);
});

// ---------------------------------------------------------------------------
// The tokenizer-asset preflight, tested by RUNNING the launcher.
//
// Every other test in this file reads run-demo.sh as text. That is the wrong
// instrument for a guard: a regex over a shell script establishes that a line
// was typed, not that it executes, not that it executes BEFORE the thing it
// protects, and not that it stays quiet on a good input. So these arms spawn
// the real script against synthetic model directories and read its exit status.
//
// What the guard is for. run-demo.sh's require_model tests `[[ -d ]]` and
// nothing more, so it accepts a model directory that scripts/build_qwen.sh
// would have REJECTED -- build_qwen.sh names tokenizer.json and
// tokenizer_config.json in REQUIRED_ARTIFACTS for both runtime targets. Models
// are gitignored, so hand-assembled directories, directories built before that
// check existed, and a MODELS_DIR aimed at another checkout are the ordinary
// ways to get one.
//
// And the server does not object. ChatTemplate::from_model_dir
// (crates/onnx-genai-ort/src/chat_template.rs:150) returns Ok with a generic
// DEFAULT_CHAT_TEMPLATE when tokenizer_config.json is absent; load_eos_token_ids
// (crates/onnx-genai-ort/src/tokenizer.rs:103) reads stop ids from
// generation_config.json and then tokenizer_config.json and treats both as
// optional. The run therefore goes green end to end -- server up, /health 200,
// `ready` printed, dashboard populated -- and only the replies are wrong. That
// is the same shape as the static-cache defect the neighbouring guard exists
// for: nothing errors, nothing is fabricated, and the demo cannot demonstrate
// the thing it exists to demonstrate.

const SCATTER_DIR = 'qwen2.5-0.5b-scatter-v2';
const DYNAMIC_DIR = 'qwen2.5-0.5b';

// Build a models directory and run the launcher's preflight against it.
//
// Every arm below is engineered to fail during preflight, which is what keeps
// this bounded: run-demo.sh checks models before ports and ports before
// `cargo build --release`, so no arm here ever reaches a compiler or a socket.
function runPreflight({ scatter = [], dynamic = [], staticCache = true }) {
  const root = mkdtempSync(join(tmpdir(), '1cb-launcher-'));
  try {
    for (const [dir, assets] of [[SCATTER_DIR, scatter], [DYNAMIC_DIR, dynamic]]) {
      mkdirSync(join(root, dir), { recursive: true });
      for (const asset of assets) writeFileSync(join(root, dir, asset), '{}');
    }
    writeFileSync(
      join(root, SCATTER_DIR, 'inference_metadata.yaml'),
      staticCache ? 'model:\n  io:\n    static_cache: {}\n' : 'model: {}\n',
    );

    const run = spawnSync('bash', [join(HERE, 'run-demo.sh')], {
      cwd: HERE,
      encoding: 'utf8',
      env: { ...process.env, MODELS_DIR: root },
    });
    return { status: run.status, output: `${run.stdout ?? ''}${run.stderr ?? ''}`, root };
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

const BOTH_ASSETS = ['tokenizer.json', 'tokenizer_config.json'];

// The positive arm, once per asset and once per model, because "the check
// fires" and "the check fires for the directory that is actually broken" are
// different claims -- a guard wired to only one of two model paths passes the
// first and fails the second, and the dynamic model is the one nothing else in
// the preflight inspects at all.
for (const [label, dir, present] of [
  ['scatter', SCATTER_DIR, { scatter: ['tokenizer.json'], dynamic: BOTH_ASSETS }],
  ['dynamic', DYNAMIC_DIR, { scatter: BOTH_ASSETS, dynamic: ['tokenizer.json'] }],
]) {
  test(`run-demo.sh refuses to launch when the ${label} model has no tokenizer_config.json`, () => {
    const { status, output } = runPreflight(present);

    assert.equal(
      status,
      1,
      `run-demo.sh exited ${status} for a ${label} model with no ` +
        `tokenizer_config.json. It must refuse: the server would start, answer ` +
        `/health, and stream untemplated replies that never stop.\n${output}`,
    );
    assert.match(output, /missing tokenizer assets/);
    assert.match(output, /tokenizer_config\.json/);
    assert.ok(
      output.includes(dir),
      `the refusal must name the offending directory (${dir}) -- with two ` +
        `models and a MODELS_DIR override, "a model" is not an actionable ` +
        `answer.\n${output}`,
    );
  });
}

test('run-demo.sh refuses to launch when a model has no tokenizer.json either', () => {
  const { status, output } = runPreflight({ scatter: [], dynamic: BOTH_ASSETS });

  assert.equal(status, 1, output);
  assert.match(output, /tokenizer\.json/);
  assert.match(output, /tokenizer_config\.json/);
});

// ⚠️ This arm pins a defect the first draft of the guard actually shipped, and
// it is the one worth keeping. The refusal offers a rebuild command, and a
// single generic `OUT_DIR=... scripts/build_qwen.sh` is WRONG for the scatter
// model: without STATIC_CACHE=1 that command overwrites the static-cache export
// with a dynamic one. The user would follow correct-looking instructions, the
// directory would refill with the missing files, this check would go quiet, and
// require_static_cache would then reject the result -- or worse, an older
// inference_metadata.yaml would survive and the demo would batch nothing.
//
// A remedy that destroys the artefact it claims to repair is worse than no
// remedy, because it carries the authority of the tool that diagnosed it.
test('the rebuild command offered for the scatter model preserves the static cache', () => {
  const { output } = runPreflight({ scatter: ['tokenizer.json'], dynamic: BOTH_ASSETS });

  const line = output.split('\n').find((l) => l.includes('scripts/build_qwen.sh'));
  assert.ok(line, `the refusal must offer a rebuild command.\n${output}`);
  assert.match(
    line,
    /STATIC_CACHE=1/,
    `the scatter model's rebuild command omits STATIC_CACHE=1, so following it ` +
      `would replace the static-cache export with a dynamic one and silently ` +
      `disable the batching scenario this demo exists to show:\n  ${line}`,
  );
});

// The negative control, and the reason the arms above mean anything.
//
// A guard that refuses EVERY model directory passes all four tests above while
// making the demo unlaunchable. This arm gives both models complete tokenizer
// assets and breaks the NEXT check instead: the run must fall through to
// require_static_cache's message and must never mention tokenizer assets.
//
// It is doing two jobs. It proves the guard stays quiet on a good directory,
// and it proves the guard is REACHED and passed rather than skipped -- because
// the static-cache error is emitted from a line that runs after it. A check
// deleted from the script entirely would also produce silence here; only the
// pairing of this arm with the positive ones tells silence apart from absence.
test('a model directory with complete tokenizer assets passes the preflight', () => {
  const { status, output } = runPreflight({
    scatter: BOTH_ASSETS,
    dynamic: BOTH_ASSETS,
    staticCache: false,
  });

  assert.equal(status, 1, output);
  assert.doesNotMatch(
    output,
    /missing tokenizer assets/,
    `the tokenizer preflight fired on a directory holding every asset it asks ` +
      `for. A guard that cannot be satisfied gets deleted, not fixed.\n${output}`,
  );
  assert.match(
    output,
    /no static-cache declaration/,
    `expected the run to reach require_static_cache, which runs AFTER the ` +
      `tokenizer preflight. Not reaching it means this control proves nothing ` +
      `about whether the preflight was executed at all.\n${output}`,
  );

  // The claim above -- "which runs AFTER the tokenizer preflight" -- is the
  // only reason this arm demonstrates reachability, and it is a claim about
  // the script's ORDER, which no arm here can observe. Moving the preflight
  // below require_static_cache leaves all five arms green and quietly turns
  // that sentence into a false one. So pin it. This is a source assertion and
  // is deliberately the only one in this section: it guards the argument, not
  // the behaviour.
  const preflightAt = LAUNCHER.indexOf('\nrequire_tokenizer_assets "');
  const staticCacheAt = LAUNCHER.indexOf('\nrequire_static_cache "');
  assert.ok(preflightAt > 0 && staticCacheAt > 0, 'both preflight calls must exist');
  assert.ok(
    preflightAt < staticCacheAt,
    'require_tokenizer_assets must be CALLED before require_static_cache. The ' +
      'control above infers "the preflight ran" from seeing the static-cache ' +
      'error, and that inference is only valid in this order.',
  );
});
