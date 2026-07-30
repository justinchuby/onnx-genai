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
import { mkdtempSync, writeFileSync, readFileSync, existsSync, rmSync, cpSync, mkdirSync } from 'node:fs';
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
 * `baseline` seeds the count ratchet's baseline file before the run. It is the
 * ONLY way to reach the ratchet's drop path from a test: without it every
 * scratch repo starts with no baseline, which is the seed-and-pass case. An
 * earlier design disabled the ratchet whenever the floors were overridden --
 * which is every case in this file -- and so the ratchet was never once
 * executed by the suite that exists to execute it.
 *
 * @param {string} root
 * @param {{args?: string[], minTests?: number, minFiles?: number,
 *          baseline?: string, allowDrop?: string}} [options]
 */
function runRunner(root, options = {}) {
  const { args = [], minTests = 4, minFiles = 1, baseline, allowDrop } = options;
  // NODE_TEST_CONTEXT IS INHERITED AND IT DISARMS THE CHILD. Node sets it for
  // every process it runs tests in; a nested `node --test` sees it, prints
  // "run() is being called recursively ... skipping running files", and emits NO
  // summary at all. The runner then correctly refuses to parse one -- so every
  // case here exited 1 for a reason that had nothing to do with the guard under
  // test, and four of them asserted only on exit codes and went GREEN.
  // The CAN RUN control is what caught it.
  const env = { ...process.env, MIN_TESTS: String(minTests), MIN_FILES: String(minFiles) };
  delete env.NODE_TEST_CONTEXT;
  if (baseline !== undefined) writeFileSync(join(root, 'test-count.baseline'), `${baseline}\n`);
  if (allowDrop !== undefined) env.ALLOW_TEST_COUNT_DROP = allowDrop;
  const result = spawnSync('bash', ['./run-tests.sh', ...args], {
    cwd: root,
    encoding: 'utf8',
    env,
  });
  return { status: result.status, out: `${result.stdout ?? ''}${result.stderr ?? ''}` };
}

/** The baseline the runner wrote into a scratch repo, or null if it wrote none. */
function baselineIn(root) {
  const path = join(root, 'test-count.baseline');
  return existsSync(path) ? readFileSync(path, 'utf8').trim() : null;
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

test('a deleted TEST file is caught by the incomplete-checkout abort, not by the tracked-but-not-run check', () => {
  const root = repo(healthyFiles());
  rmSync(join(root, 'nested/beta.test.js'));
  const { status, out } = runRunner(root);
  assert.equal(status, 1);
  // WHICH GUARD DOES THE WORK MATTERS, AND I GOT THIS WRONG ONCE. A deleted
  // tracked test file is caught by the incomplete-checkout abort, which runs
  // first and refuses to start Node at all. From that I concluded the
  // tracked-but-not-run check below it was dead code. It is not: it is the only
  // thing that catches a file which is PRESENT and simply never discovered --
  // a narrowed glob -- and the incomplete-checkout abort is correctly silent
  // there. See the comment at that branch in `run-tests.sh`.
  //
  // So this pins the division of labour, not a redundancy.
  assert.match(out, /this checkout is INCOMPLETE/);
  assert.match(out, /beta\.test\.js/);
  assert.match(out, /NO TESTS WERE RUN/);
  assert.doesNotMatch(out, /tracked at HEAD but were NOT RUN/);
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

/** A fixture whose FIRST test commits to the repository it is running in. */
const HEAD_MOVING_SUITE = [
  "import { describe, it } from 'node:test';",
  "import assert from 'node:assert/strict';",
  "import { execFileSync } from 'node:child_process';",
  "import { writeFileSync } from 'node:fs';",
  "describe('a suite that moves HEAD underneath itself', () => {",
  "  it('commits while the suite is running', () => {",
  "    writeFileSync('landed-mid-run.txt', 'another agent committed\\n');",
  "    const git = (...a) => execFileSync('git', a, { encoding: 'utf8' });",
  "    git('add', 'landed-mid-run.txt');",
  "    git('commit', '-q', '-m', 'a commit that lands mid-run');",
  '    assert.ok(true);',
  '  });',
  '});',
  '',
].join('\n');

test('a commit landing MID-RUN fails the whole run, because no single tree was measured', () => {
  const root = repo({ ...healthyFiles(), 'mover.test.js': HEAD_MOVING_SUITE });
  const { status, out } = runRunner(root, { minTests: 5 });
  assert.equal(status, 1);
  assert.match(out, /HEAD MOVED WHILE THE SUITE WAS RUNNING/);
  assert.match(out, /graded against DIFFERENT TREES/);
  assert.match(out, /MOVED MID-RUN/);
  // The point is that EVERY OTHER SIGNAL LOOKED FINE. This run had zero failing
  // tests, a complete checkout and a clean start; the only thing wrong with it
  // was that its subject changed halfway through, which no other check can see.
  assert.match(out, /failed {11}: 0/);
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

// ---------------------------------------------------------------------------
// THE TEST-COUNT RATCHET.
//
// The floor (MIN_TESTS) is an anti-vacuity guard and is blind to silent
// shrinkage: it passes at 595 when the true count is 600. The ratchet is the
// guard for that, and these are the arms that prove it can actually say no.
//
// Every case below seeds a baseline explicitly. That is deliberate and it is
// the whole reason this section exists: the first version of the ratchet
// switched itself off whenever the floors were overridden, and since this
// harness overrides the floors in every case, the guard was unreachable from
// here. It would have shipped green forever without executing once.
// ---------------------------------------------------------------------------

test('RATCHET CAN RUN: with no baseline the runner seeds one and passes', () => {
  const root = repo(healthyFiles());
  const { status, out } = runRunner(root);
  assert.equal(status, 0, out);
  assert.match(out, /no baseline yet; seeded/);
  // The seeded number must be the count actually observed, not the floor.
  assert.equal(baselineIn(root), '8');
});

test('a DROP below the baseline fails the run even though every test passed', () => {
  const root = repo(healthyFiles());
  const { status, out } = runRunner(root, { baseline: '99' });
  assert.equal(status, 1, out);
  assert.match(out, /test count DROPPED 99 -> 8 \(91 fewer\)/);
});

test('a drop with ZERO failures says the tests never RAN, not that a test broke', () => {
  // The two diagnoses are different and only this one points at a load failure,
  // which is the defect the ratchet exists for: 588 -> 435 with nothing red.
  const { out } = runRunner(repo(healthyFiles()), { baseline: '99' });
  assert.match(out, /NO TEST FAILED/);
  assert.match(out, /tests\s+that never ran/);
  assert.doesNotMatch(out, /Do not assume one explains the/);
});

test('a drop WITH failures refuses to let one number explain the other', () => {
  const root = repo({ ...healthyFiles(), 'broken.test.js': FAILING_SUITE });
  const { status, out } = runRunner(root, { baseline: '99', args: ['--allow-untracked'] });
  assert.equal(status, 1, out);
  assert.match(out, /Do not assume one explains the/);
  assert.doesNotMatch(out, /NO TEST FAILED/);
});

test('the drop override must name the EXACT new count; a truthy value is refused', () => {
  // ALLOW_TEST_COUNT_DROP=1 would be typed once and then wave through every
  // later drop. Naming the count makes the permission expire by itself.
  const { status, out } = runRunner(repo(healthyFiles()), { baseline: '99', allowDrop: '1' });
  assert.equal(status, 1, out);
  assert.match(out, /test count DROPPED/);
});

test('the drop override named exactly accepts the drop and lowers the baseline', () => {
  const root = repo(healthyFiles());
  const { status, out } = runRunner(root, { baseline: '99', allowDrop: '8' });
  assert.equal(status, 0, out);
  assert.match(out, /allowed by/);
  assert.equal(baselineIn(root), '8');
});

test('a baseline file that holds no number FAILS rather than becoming no ratchet', () => {
  const root = repo(healthyFiles());
  const { status, out } = runRunner(root, { baseline: 'not-a-number' });
  assert.equal(status, 1, out);
  assert.match(out, /holds no number/);
});

test('a RED run never seeds a baseline, and says out loud that it declined to', () => {
  // A guard that declined to arm itself must not look like one that armed and
  // was satisfied.
  const root = repo({ ...healthyFiles(), 'broken.test.js': FAILING_SUITE });
  const { status, out } = runRunner(root, { args: ['--allow-untracked'] });
  assert.equal(status, 1, out);
  assert.match(out, /ratchet: NOT seeded/);
  assert.equal(baselineIn(root), null);
});
