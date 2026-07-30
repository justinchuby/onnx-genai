// NO MODULE MAY NAME THE PREFIX-CACHE COUNTERS. (@12e42da8, RULED, FINAL.)
//
// WHY THIS IS THE MOST IMPORTANT TRIPWIRE IN THE TREE:
// @fc8b5d97 measured prefix reuse with a control arm and it is ABSENT.
//   - shared ~900-token prefix x6, warm TTFT      1341 ms
//   - six prefixes differing FROM TOKEN 0         1254 ms   <- 7.0% FASTER
//   - sensitivity: prefill is ~90% of TTFT, so a working cache would have
//     collapsed 1380 ms -> ~140 ms. Observed: +7.0%.
// Meanwhile `prefix_cache_hits` reported 19/20 = 95%, because it increments on
// ANY nonzero token match and every /v1/chat/completions request shares the
// chat-template preamble. The counter reads ~95% from the first request and
// never moves.
//
// So this is not a stub and not a misnamed-but-real number. It is a
// PRECISELY-COMPUTED, BEAUTIFULLY-BEHAVED, ENTIRELY FALSE value -- and it
// passes every other safeguard we built, all of which hunt fabricated ZEROS.
// A 95% invites no scrutiny at all. That is the whole danger.
//
// THIS TEST IS A RATCHET, NOT A GATE. Four modules still reference these
// fields; deleting their bindings is owned by the panel authors. The allowlist
// below is the debt, enumerated so it cannot be forgotten, and it may only ever
// SHRINK. Any NEW module naming these fields fails immediately.
//
// MUTATION PROVING IT FAILS: add `prefix_cache_hits` to any .js file not on the
// allowlist -- e.g. `echo "// prefix_cache_hits" >> dashboard/throughput.js`.
import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { join, relative } from 'node:path';

const ROOT = fileURLToPath(new URL('.', import.meta.url));

/** The counters ruled unshippable. `hit_rate` is `hits / completed generations`. */
const FORBIDDEN = ['prefix_cache_hits', 'prefix_cache_lookups', 'prefix_cache_hit_rate'];

/**
 * Modules that still reference the counters, with the reason. THIS LIST MAY
 * ONLY SHRINK. A file leaves the list by deleting its references, never by
 * being added back.
 */
const ALLOWLIST = new Map([
  // PERMANENT. The provenance audit's job is to RECORD that these fields exist
  // and must never be bound. Naming a forbidden field in the register that
  // forbids it is the one legitimate use.
  ['telemetry-provenance.js', 'permanent: the register that forbids them'],
  // DEBT — owned by the panel authors, must reach zero before release.
  ['telemetry-store.js', 'debt: store still projects the field'],
  ['dashboard/prefix-cache.js', 'debt: the panel itself is being removed'],
  ['app.js', 'debt: audit-view wiring'],
]);

/** @returns {string[]} every .js file in the demo, excluding tests and deps */
function sourceFiles(dir = ROOT, found = []) {
  for (const entry of readdirSync(dir)) {
    if (entry === 'node_modules' || entry === '.git' || entry === 'design') continue;
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) sourceFiles(full, found);
    else if (entry.endsWith('.js') && !entry.endsWith('.test.js')) found.push(full);
  }
  return found;
}

describe('the prefix-cache counters are unnameable', () => {
  it('is not referenced by any module outside the shrinking allowlist', () => {
    /** @type {string[]} */
    const violations = [];
    for (const file of sourceFiles()) {
      const rel = relative(ROOT, file);
      if (ALLOWLIST.has(rel)) continue;
      const source = readFileSync(file, 'utf8');
      for (const field of FORBIDDEN) {
        if (source.includes(field)) violations.push(`${rel} names ${field}`);
      }
    }
    assert.deepEqual(
      violations,
      [],
      'A module has bound a prefix-cache counter. These report ~95% hits while ' +
        'a control arm shows shared prefixes are 7.0% SLOWER than unshared ones ' +
        '(@fc8b5d97, n=20) -- the number is precisely computed and entirely ' +
        `false. Ruled unshippable in any form:\n  ${violations.join('\n  ')}`,
    );
  });

  it('has an allowlist that only shrinks', () => {
    // Pins the debt. Deleting a binding means deleting its entry here too, and
    // that edit is the thing a reviewer can actually see.
    assert.equal(
      ALLOWLIST.size,
      4,
      'The allowlist changed size. It may only SHRINK -- if you removed a ' +
        "binding, drop its entry and lower this number. If you added one, don't.",
    );
  });

  it('never lets a panel present these as a rate', () => {
    // The deepest trap: `hit_rate` is hits / COMPLETED GENERATIONS. Even once
    // the counters are gone, the words remain inviting and the field stays on
    // the wire, so the vocabulary is banned from UI copy too.
    for (const file of sourceFiles()) {
      const rel = relative(ROOT, file);
      if (ALLOWLIST.has(rel)) continue;
      const source = readFileSync(file, 'utf8');
      assert.equal(
        /['"`][^'"`]*\b(cache hit rate|prefix hit rate|cache lookups)\b/i.test(source),
        false,
        `${rel} contains UI copy naming a prefix cache rate. Nothing may be ` +
          "labelled 'hit rate' from a division whose denominator counts " +
          'generations, and no prefix panel ships in any form.',
      );
    }
  });
});
