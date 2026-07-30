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

// Citations are resolved by BASENAME, so the inventory must cover every kind of
// file the README argues from -- not just Rust. An earlier version tracked only
// `*.rs`, which meant the JS citations were silently unverifiable.
const CITED_EXTENSIONS = ['rs', 'js', 'css', 'html', 'sh'];

const trackedSourceFiles = execFileSync(
  'git',
  ['ls-files', ...CITED_EXTENSIONS.map((ext) => `*.${ext}`)],
  {
    cwd: repoRoot,
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
  },
)
  .split('\n')
  .filter(Boolean);

const byBasename = new Map();
for (const path of trackedSourceFiles) {
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

// Path segments contain HYPHENS (`crates/onnx-genai-server/...`) and digits.
// The original class was `[a-z_]+`, which silently excluded every citation into
// the server and engine crates -- 7 of 27, including all the JS ones. A regex
// that under-matches produces a green test over a shrinking sample, which is
// the exact failure this file exists to prevent.
const CITATION = new RegExp(
  '`([A-Za-z0-9_.-]+(?:\\/[A-Za-z0-9_.-]+)*\\.(?:' +
    CITED_EXTENSIONS.join('|') +
    ')):(\\d+)(?:-(\\d+))?`',
  'g',
);

function citations() {
  const found = [];
  for (const match of readme.matchAll(CITATION)) {
    found.push({
      text: match[0],
      path: match[1],
      file: basename(match[1]),
      // The last line the citation claims to reference.
      line: Number(match[3] ?? match[2]),
    });
  }
  return found;
}

// Resolve a citation to the actual file(s) it could mean.
//
// Basename alone is far too loose for this repo: there are 40 files named
// `lib.rs`, and checking a line number against the LONGEST of them means a
// citation to the 200-line server lib.rs is "verified" against a 2395-line one.
// That is an existence check standing in for an identity check -- the exact
// defect class this suite is here to catch -- so when the citation carries a
// path, require the tracked file to end with it.
function resolve({ path, file }) {
  if (path.includes('/')) {
    const exact = trackedSourceFiles.filter((p) => p === path || p.endsWith('/' + path));
    if (exact.length > 0) return exact;
  }
  return byBasename.get(file) ?? [];
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

test('no citation in the README escapes the matcher', () => {
  // THE BUG THIS EXISTS FOR. The three tests above are only as good as the
  // regex feeding them, and a regex that matches nothing -- or matches most
  // things -- reports green either way. The count floor above does not help:
  // it stayed comfortably over 10 while 7 real citations were invisible.
  //
  // So compare the strict matcher against a deliberately loose one. Anything
  // shaped like `path/file.ext:123` that the strict matcher misses is a
  // citation nobody is checking.
  //
  // MUTATION: reverted CITATION's character class to `[a-z_]+` -> red, naming
  // all 7 previously-invisible citations.
  const LOOSE = /`([^`\s]+\.[A-Za-z]{1,5}):(\d+)(?:-(\d+))?`/g;

  const strict = new Set(citations().map((c) => c.text));
  const escaped = [];
  for (const match of readme.matchAll(LOOSE)) {
    const ext = match[1].split('.').pop();
    if (!CITED_EXTENSIONS.includes(ext)) continue; // e.g. `Cargo.toml:187`
    if (!strict.has(match[0])) escaped.push(match[0]);
  }

  assert.deepEqual(
    escaped,
    [],
    `These look like source citations but the CITATION regex does not match ` +
      `them, so no test verifies they point anywhere real:\n  ${escaped.join('\n  ')}`,
  );
});

test('every file the README cites still exists in the repository', () => {
  for (const citation of citations()) {
    assert.ok(
      resolve(citation).length > 0,
      `README.md cites ${citation.text}, but no tracked file matches that path ` +
        `anywhere in the repository. It was renamed, moved or deleted -- a citation ` +
        `pointing at a file that is gone reads exactly like one that is fine. ` +
        `(This project deleted cors.rs while docs still cited it.)`,
    );
  }
});

test('every line the README cites still exists in the file', () => {
  for (const citation of citations()) {
    const candidates = resolve(citation);
    if (candidates.length === 0) continue; // reported by the test above

    const longest = Math.max(...candidates.map(lineCountOf));

    assert.ok(
      citation.line <= longest,
      `README.md cites ${citation.text}, but that file has only ${longest} lines ` +
        `(${candidates.join(', ')}). The file shrank and the citation now points ` +
        `past the end of it.`,
    );
  }
});
