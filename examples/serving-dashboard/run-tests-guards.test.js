/**
 * THE RUNNER IS THE INSTRUMENT EVERY OTHER MEASUREMENT IN THIS REPOSITORY IS
 * TAKEN WITH, AND UNTIL THIS FILE EXISTED NOTHING MEASURED IT.
 *
 * `run-tests.sh` carries nine checks whose entire purpose is to refuse a green
 * that was never earned. Every one of them was written in response to a real
 * incident, and not one of them had ever been observed to FIRE. A guard that has
 * never fired and a guard that cannot fire produce identical output on a healthy
 * tree -- which is the exact defect the script exists to prevent, one level up.
 *
 * So this suite breaks the tree in nine specific ways and demands a RED each
 * time. It is the step every previous version of the runner proposal omitted.
 *
 * HOW IT WORKS, AND WHY IT DOES NOT RUN THE REAL SUITE: each case builds a
 * throwaway git repository in a temp directory, puts a tiny four-test fixture
 * suite in it, copies the REAL `run-tests.sh` in unmodified, and runs it there.
 * The script under test is the shipped byte-for-byte file, not a copy of its
 * logic -- a reimplementation would prove only that this file agrees with
 * itself.
 *
 * THE CONTROL IS THE FIRST TEST AND IT IS NOT A FORMALITY. If the fixture
 * repository cannot produce a PASS, then every "it exited 1" below is satisfied
 * by a scratch repo that was broken from the start, and this whole file would be
 * nine confident greens over a harness that never worked. It has already earned
 * its place once: the first draft's fixtures used bare `test()` with no
 * `describe`, so Node reported 0 suites, the `suites < discovered` check fired
 * in EVERY case, and eight of the nine arms passed for a reason that had nothing
 * to do with what they claimed to test.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync, cpSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const RUNNER = join(HERE, 'run-tests.sh');

/** A fixture test file with one `describe` so Node reports one suite per file. */
function fixtureSuite(name, count = 4) {
  const cases = Array.from(
    { length: count },
    (_, i) => `  it('${name} case ${i}', () => { assert.equal(1, 1); });`,
  ).join('\n');
  return [
    "import { describe, it } from 'node:test';",
    "import assert from 'node:assert/strict';",
    `describe('${name}', () => {`,
    cases,
    '});',
    '',
  ].join('\n');
}

const FAILING_SUITE = [
  "import { describe, it } from 'node:test';",
  "import assert from 'node:assert/strict';",
  "describe('deliberately failing fixture', () => {",
  "  it('a uniquely named failing case', () => { assert.equal('a', 'b'); });",
  '});',
  '',
].join('\n');

/**
 * A throwaway git repository containing `files`, the real runner, and nothing
 * else. Everything is committed, so a clean run has zero untracked files and a
 * complete checkout.
 *
 * @param {Record<string, string>} files
 * @returns {string} the repository path
 */
function scratchRepo(files) {
  const root = mkdtempSync(join(tmpdir(), 'bb-runner-'));
  const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' });
  git('init', '-q', '-b', 'main');
  git('config', 'user.email', 'runner-guard@example.invalid');
  git('config', 'user.name', 'runner guard fixture');
  // package.json so Node treats the fixtures as ES modules, exactly as the real
  // dashboard does.
  writeFileSync(join(root, 'package.json'), '{"type":"module"}\n');
  cpSync(RUNNER, join(root, 'run-tests.sh'));
  for (const [name, text] of Object.entries(files)) {
    const full = join(root, name);
    mkdirSync(dirname(full), { recursive: true });
    writeFileSync(full, text);
  }
  git('add', '-A');
  git('commit', '-q', '-m', 'fixture');
  return root;
}

/**
 * Run the real runner inside a scratch repo with the floors lowered to suit a
 * fixture-sized suite.
 *
 * @param {string} root
 * @param {{args?: string[], minTests?: number, minFiles?: number}} [options]
 */
function runRunner(root, options = {}) {
  const { args = [], minTests = 4, minFiles = 1 } = options;
  // NODE_TEST_CONTEXT IS INHERITED AND IT DISARMS THE CHILD. Node sets it for
  // every process it runs tests in; a nested `node --test` sees it, prints
  // "run() is being called recursively ... skipping running files", and emits NO
  // summary at all. The runner then correctly refuses to parse one -- so every
  // case here exited 1 for a reason that had nothing to do with the guard under
  // test, and four of them asserted only on exit codes and went GREEN.
  // The CAN RUN control is what caught it.
  const env = { ...process.env, MIN_TESTS: String(minTests), MIN_FILES: String(minFiles) };
  delete env.NODE_TEST_CONTEXT;
  const result = spawnSync('bash', ['./run-tests.sh', ...args], {
    cwd: root,
    encoding: 'utf8',
    env,
  });
  return { status: result.status, out: `${result.stdout ?? ''}${result.stderr ?? ''}` };
}

/** Two healthy files, eight passing tests. The shape every case starts from. */
function healthyFiles() {
  return {
    'alpha.test.js': fixtureSuite('alpha'),
    'nested/beta.test.js': fixtureSuite('beta'),
    'README.md': '# fixture\n',
  };
}

const created = [];
function repo(files) {
  const root = scratchRepo(files);
  created.push(root);
  return root;
}

process.on('exit', () => {
  for (const root of created) rmSync(root, { recursive: true, force: true });
});

test('CAN RUN: a healthy fixture repository passes, so a red below means something', () => {
  const { status, out } = runRunner(repo(healthyFiles()));
  assert.equal(status, 0, `expected a clean fixture to PASS, got:\n${out}`);
  assert.match(out, /PASS: 8 tests across 2 suites, 0 failures\./);
  assert.match(out, /discovered: 2 test files/);
});

test('the floors are printed, so a lowered floor cannot pass unnoticed', () => {
  const { out } = runRunner(repo(healthyFiles()));
  assert.match(out, /floors: 4 tests \/ 1 files/);
});

test('a repository with no test files at all FAILS instead of reporting zero', () => {
  const { status, out } = runRunner(repo({ 'README.md': '# nothing here\n' }));
  assert.equal(status, 1);
  assert.match(out, /discovered no test files at all/);
  assert.match(out, /the failure that looks like success/);
});

test('an UNTRACKED test file fails the run, because a clean clone does not have it', () => {
  const root = repo(healthyFiles());
  writeFileSync(join(root, 'sneaky.test.js'), fixtureSuite('sneaky'));
  const { status, out } = runRunner(root);
  assert.equal(status, 1);
  assert.match(out, /ran here but are not committed/);
  assert.match(out, /sneaky\.test\.js/);
});

test('--allow-untracked downgrades that to a WARNING and says the total is desk-scoped', () => {
  const root = repo(healthyFiles());
  writeFileSync(join(root, 'sneaky.test.js'), fixtureSuite('sneaky'));
  const { status, out } = runRunner(root, { args: ['--allow-untracked'] });
  assert.equal(status, 0, `expected the escape hatch to pass, got:\n${out}`);
  assert.match(out, /WARN: 1 untracked test file\(s\) were INCLUDED/);
  assert.match(out, /NOT the branch/);
});

test('a deleted TEST file is caught by the incomplete-checkout abort, not by the check written for it', () => {
  const root = repo(healthyFiles());
  rmSync(join(root, 'nested/beta.test.js'));
  const { status, out } = runRunner(root);
  assert.equal(status, 1);
  // THE FINDING: `run-tests.sh` carries a check whose message is
  // "tracked at HEAD but missing from disk", written specifically for this case.
  // It is UNREACHABLE. A missing tracked test file is a missing tracked file,
  // and the incomplete-checkout abort -- added later, broader, and running
  // earlier -- takes it every time. This pins which guard actually does the
  // work, so nobody trusts the dead one.
  assert.match(out, /this checkout is INCOMPLETE/);
  assert.match(out, /beta\.test\.js/);
  assert.match(out, /NO TESTS WERE RUN/);
  assert.doesNotMatch(out, /tracked at HEAD but missing from disk/);
});

test('a discovered file that contributes NO tests fails the suites-vs-discovered check', () => {
  const root = repo({ ...healthyFiles(), 'hollow.test.js': fixtureSuite('hollow') });
  // Emptied after commit: still discovered, still loads, contributes nothing.
  writeFileSync(join(root, 'hollow.test.js'), '// every suite here was commented out\n');
  const { status, out } = runRunner(root, { args: ['--allow-untracked'] });
  assert.equal(status, 1);
  assert.match(out, /found 3 test files but only 2 suites ran/);
  assert.match(out, /contributed no tests at all/);
});

test('a suite below the test floor fails, and says a shrunk suite and a stopped runner look alike', () => {
  const { status, out } = runRunner(repo(healthyFiles()), { minTests: 9 });
  assert.equal(status, 1);
  assert.match(out, /8 tests is below the 9 floor/);
  assert.match(out, /A check that stopped\s+looking and a check that found nothing print the same green/);
});

test('an INCOMPLETE checkout refuses to run the suite at all, rather than reporting reds from it', () => {
  const root = repo(healthyFiles());
  rmSync(join(root, 'README.md')); // tracked, not a test file, imported by nothing
  const { status, out } = runRunner(root);
  assert.equal(status, 1);
  assert.match(out, /this checkout is INCOMPLETE/);
  assert.match(out, /NO TESTS WERE RUN/);
  // The ordering is the whole point of that check: it must abort BEFORE Node
  // runs, so a half-written tree cannot produce misleading reds first.
  assert.doesNotMatch(out, /reconciliation/);
});

test('a failing test is named LAST, so a piped run keeps the diagnosis', () => {
  const root = repo({ ...healthyFiles(), 'broken.test.js': FAILING_SUITE });
  const { status, out } = runRunner(root);
  assert.equal(status, 1);
  assert.match(out, /--- the 1 failing test\(s\), by name ---/);
  const tail = out.trimEnd().split('\n').slice(-3).join('\n');
  assert.match(
    tail,
    /a uniquely named failing case/,
    `the failing test name must survive \`| tail\`, but the last lines were:\n${tail}`,
  );
});
