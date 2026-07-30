import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync, readFileSync, mkdirSync, copyFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));

/**
 * F12 — `repair-citations.mjs` SHIPPED WITH ZERO TESTS, AND IT IS THE ONLY
 * TOOL ON THE BRANCH THAT REWRITES A DOCUMENT REVIEWERS READ.
 *
 * These are BLACK-BOX tests against a throwaway git repository rather than
 * unit tests against exported functions, for one deliberate reason: the whole
 * safety argument for this tool is about WHICH TREE IT READS. A unit test that
 * hands the parser a string proves nothing about that, and would have passed
 * on the version of the script that read the working copy. The property under
 * test only exists in the presence of a real HEAD, a real index and a real
 * dirty file, so the fixture is a real repository.
 *
 * Every test below has been mutation-checked: reverting the behaviour it
 * describes turns it red.
 */
describe('repair-citations reads the shipping tree, not the desk', () => {
  let repo;

  const git = (...args) =>
    execFileSync('git', args, { cwd: repo, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });

  // The tool exits non-zero when any citation needs a human -- that is its
  // contract, not a crash -- so capture stdout regardless of exit status.
  const run = (...args) => {
    try {
      return execFileSync('node', [join(repo, 'tool', 'repair-citations.mjs'), ...args], {
        cwd: repo,
        encoding: 'utf8',
      });
    } catch (e) {
      if (e.stdout === undefined) throw e;
      return e.stdout;
    }
  };

  const README = () => join(repo, 'tool', 'README.md');

  before(() => {
    repo = mkdtempSync(join(tmpdir(), 'repair-citations-'));
    git('init', '-q');
    git('config', 'user.email', 'test@example.com');
    git('config', 'user.name', 'test');

    mkdirSync(join(repo, 'tool'), { recursive: true });
    mkdirSync(join(repo, 'src'), { recursive: true });
    copyFileSync(join(HERE, 'repair-citations.mjs'), join(repo, 'tool', 'repair-citations.mjs'));

    // The cited symbol sits on line 4. A correct tool proposes :4.
    writeFileSync(
      join(repo, 'src', 'widget.rs'),
      ['// one', '// two', '// three', 'pub fn assemble_widget() {}', '// five', ''].join('\n'),
    );
    writeFileSync(
      README(),
      ['# Doc', '', 'The `assemble_widget()` entry point lives at `src/widget.rs:99`.', ''].join(
        '\n',
      ),
    );
    git('add', '-A');
    git('commit', '-q', '-m', 'fixture');
  });

  after(() => {
    if (repo) rmSync(repo, { recursive: true, force: true });
  });

  it('proposes the definition line from a clean tree', () => {
    const out = run();
    assert.match(out, /WOULD REPAIR/, `expected a repair proposal, got:\n${out}`);
    assert.match(
      out,
      /-> :4\b/,
      `expected the definition on line 4 of the COMMITTED file; got:\n${out}`,
    );
  });

  it('does not touch the document without --write', () => {
    const before = readFileSync(README(), 'utf8');
    run();
    assert.equal(readFileSync(README(), 'utf8'), before, 'dry run modified the README');
  });

  it('REFUSES a cited file with uncommitted changes', () => {
    // THE CORE REGRESSION. Before the fix the tool counted lines on disk, so
    // this edit -- which no reviewer will ever see -- silently moved the
    // proposed citation from :4 to :6, and `--write` would have committed it.
    const path = join(repo, 'src', 'widget.rs');
    const pristine = readFileSync(path, 'utf8');
    writeFileSync(path, ['// injected', '// injected', pristine].join('\n'));
    try {
      const out = run();
      assert.match(out, /DECLINED/, `expected a refusal on a dirty cited file, got:\n${out}`);
      assert.match(out, /uncommitted changes/, `expected the reason to name the cause:\n${out}`);
      assert.doesNotMatch(
        out,
        /-> :6\b/,
        'the tool anchored to a line number that only exists on this desk',
      );
    } finally {
      writeFileSync(path, pristine);
    }
  });

  it('REFUSES to resolve a citation to an untracked file', () => {
    // `existsSync` used to accept this: the file is on disk, so the tool read
    // it and anchored a citation to a file no reviewer has. A citation is a
    // promise to someone holding the shipped tree.
    writeFileSync(join(repo, 'src', 'ghost.rs'), 'pub fn haunt_me() {}\n');
    const readme = readFileSync(README(), 'utf8');
    writeFileSync(README(), `${readme}\nThe \`haunt_me()\` helper is at \`src/ghost.rs:1\`.\n`);
    try {
      const out = run();
      assert.match(
        out,
        /ghost\.rs:1` — cited path does not resolve/,
        `expected the untracked citation to be declined, got:\n${out}`,
      );
    } finally {
      writeFileSync(README(), readme);
      rmSync(join(repo, 'src', 'ghost.rs'), { force: true });
    }
  });

  it('applies the repair under --write, and only then', () => {
    assert.match(readFileSync(README(), 'utf8'), /widget\.rs:99/, 'fixture drifted');
    run('--write');
    const after = readFileSync(README(), 'utf8');
    assert.match(after, /widget\.rs:4`/, `--write did not apply the repair:\n${after}`);
    assert.doesNotMatch(after, /widget\.rs:99/, 'the stale citation survived --write');
    // Restore so the suite is order-independent.
    writeFileSync(README(), readFileSync(README(), 'utf8').replace('widget.rs:4`', 'widget.rs:99`'));
    execFileSync('git', ['checkout', '--', 'tool/README.md'], { cwd: repo });
  });

  it('finds citations at all — anti-vacuity', () => {
    // A regex that matched nothing would print no proposals and no declines,
    // and would be byte-identical to a document with no defects.
    const out = run();
    assert.ok(
      /WOULD REPAIR|DECLINED/.test(out),
      `the tool reported neither repairs nor declines; it is parsing nothing:\n${out}`,
    );
  });
});
