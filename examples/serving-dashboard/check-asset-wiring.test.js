// Asset wiring: every stylesheet that exists on disk must be linked by the page
// exactly once.
//
// This exists because `styles/panels.css` shipped fetchable but unlinked. The
// server served it with a 200, every module test passed, and every panel
// rendered unstyled with nothing in the console. Reported four times by humans
// and fixed zero times, which is the argument for a test rather than a report.
//
// The failure is invisible to every other check we have, and the reason is
// structural: a test that imports an artifact by path proves the artifact is
// good, never that anything references it. Existing, wired, and committed are
// three separate claims. This file checks the second one.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const demoDir = dirname(fileURLToPath(import.meta.url));
const indexHtml = readFileSync(join(demoDir, 'index.html'), 'utf8');

const IGNORED_DIRS = new Set(['node_modules', '.git', 'design']);

function stylesheetsOnDisk(dir = demoDir) {
  const found = [];
  for (const entry of readdirSync(dir)) {
    if (IGNORED_DIRS.has(entry)) continue;
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      found.push(...stylesheetsOnDisk(full));
    } else if (entry.endsWith('.css')) {
      found.push(relative(demoDir, full));
    }
  }
  return found;
}

// Counts <link rel="stylesheet"> hrefs only. A stylesheet named in a comment or
// in prose is not wired, and must not be allowed to satisfy this check --
// asserting a superset of reality is how a drift test passes its own mutation.
function linkCount(cssPath) {
  const hrefs = [...indexHtml.matchAll(/<link\b[^>]*\brel=["']stylesheet["'][^>]*>/gi)]
    .map((tag) => tag[0].match(/\bhref=["']([^"']+)["']/i))
    .filter(Boolean)
    .map((href) => href[1].replace(/^\.\//, '').split(/[?#]/)[0]);
  return hrefs.filter((href) => href === cssPath.split('\\').join('/')).length;
}

test('the stylesheet scan finds files at all', () => {
  // Guard against the check silently passing because it found nothing. A test
  // that cannot fail is a hypothesis, not a check.
  assert.ok(
    stylesheetsOnDisk().length >= 2,
    'expected to find stylesheets on disk; the scan itself is broken',
  );
});

test('the link scan finds links at all', () => {
  const anyLinks = /<link\b[^>]*\brel=["']stylesheet["']/i.test(indexHtml);
  assert.ok(anyLinks, 'index.html has no stylesheet links at all; the scan is broken');
});

for (const css of stylesheetsOnDisk()) {
  test(`${css} is linked exactly once by index.html`, () => {
    const count = linkCount(css);
    assert.equal(
      count,
      1,
      count === 0
        ? `${css} exists on disk but index.html never links it. It will be ` +
          `served with a 200 and applied to nothing: no 404, no console error, ` +
          `just unstyled markup. Add a <link rel="stylesheet"> for it, or delete ` +
          `the file if it is genuinely dead.`
        : `${css} is linked ${count} times. Duplicate links let two copies of a ` +
          `rule set drift apart with nothing to announce it.`,
    );
  });
}
