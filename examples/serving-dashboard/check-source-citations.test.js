// The README argues from source. Nearly every claim it makes about what the
// server ACTUALLY does is anchored to a `file.rs:LINE` citation, because "the
// server counts this" is worth nothing without somewhere to go and check.
//
// Those anchors rot silently and constantly. Three were wrong in a single hour
// tonight: `metrics.rs:111` (the increment is at :112), `admin.rs:86` (the
// clamp is at :85), and a `cors.rs` citation that survived the file being
// DELETED. Nothing about a stale line number looks stale -- it is still a
// plausible file and a plausible number, and the only way to catch it is to go
// and look, which is exactly what a reader will not do. A citation that cannot
// be followed is worse than no citation, because it converts "I should check
// this" into "someone already did".
//
// This cannot verify that a line still says what we claim -- that needs a human
// -- but it catches the two failures that actually happen: the file is gone,
// and the file is now too short to contain the line.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join, basename } from 'node:path';

const demoDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = execFileSync('git', ['rev-parse', '--show-toplevel'], {
  cwd: demoDir,
  encoding: 'utf8',
}).trim();

const readme = readFileSync(join(demoDir, 'README.md'), 'utf8');

const trackedRustFiles = execFileSync('git', ['ls-files', '*.rs'], {
  cwd: repoRoot,
  encoding: 'utf8',
  maxBuffer: 32 * 1024 * 1024,
})
  .split('\n')
  .filter(Boolean);

const byBasename = new Map();
for (const path of trackedRustFiles) {
  const key = basename(path);
  if (!byBasename.has(key)) byBasename.set(key, []);
  byBasename.get(key).push(path);
}

const lineCounts = new Map();
function lineCountOf(path) {
  if (!lineCounts.has(path)) {
    lineCounts.set(path, readFileSync(join(repoRoot, path), 'utf8').split('\n').length);
  }
  return lineCounts.get(path);
}

const CITATION = /`([a-z_]+(?:\/[a-z_]+)*\.rs):(\d+)(?:-(\d+))?`/g;

function citations() {
  const found = [];
  for (const match of readme.matchAll(CITATION)) {
    found.push({
      text: match[0],
      file: basename(match[1]),
      // The last line the citation claims to reference.
      line: Number(match[3] ?? match[2]),
    });
  }
  return found;
}

test('the README cites source at all', () => {
  // If this ever hits zero, the regex stopped matching and every assertion
  // below would pass over an empty list -- a green suite proving nothing.
  assert.ok(
    citations().length >= 10,
    `Only ${citations().length} source citations found in README.md. The README ` +
      `argues from source throughout, so this almost certainly means the ` +
      `citation format changed and this check is now inspecting nothing.`,
  );
});

test('every file the README cites still exists in the repository', () => {
  for (const { text, file } of citations()) {
    assert.ok(
      byBasename.has(file),
      `README.md cites ${text}, but no tracked file named '${file}' exists ` +
        `anywhere in the repository. It was renamed or deleted -- a citation ` +
        `pointing at a file that is gone reads exactly like one that is fine. ` +
        `(This project deleted cors.rs while docs still cited it.)`,
    );
  }
});

test('every line the README cites still exists in the file', () => {
  for (const { text, file, line } of citations()) {
    if (!byBasename.has(file)) continue; // reported by the test above

    const candidates = byBasename.get(file);
    const longest = Math.max(...candidates.map(lineCountOf));

    assert.ok(
      line <= longest,
      `README.md cites ${text}, but the longest file named '${file}' has only ` +
        `${longest} lines (${candidates.join(', ')}). The file shrank and the ` +
        `citation now points past the end of it.`,
    );
  }
});
