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
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

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
