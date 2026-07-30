// NO MODULE MAY NAME THE PREFIX-CACHE COUNTERS. (@12e42da8, RULED, FINAL.)
//
// WHY THIS IS THE MOST IMPORTANT TRIPWIRE IN THE TREE:
// The counter is disqualified on its own arithmetic, and this argument needs
// no stopwatch, so no re-run can withdraw it:
//   - twelve requests: six repeated, SIX DELIBERATELY UNIQUE
//   - +12 hits -- one per completed generation, unique prompts included
// A counter that reads the same whether prefixes are reused or not cannot
// distinguish the two cases, so it is not measuring reuse.
//
// QUOTE THE DELTA, NEVER A RATE. `prefix_cache_hits` and `prefix_cache_lookups`
// are CUMULATIVE SINCE BOOT, so their ratio is a property of the process rather
// than of the experiment: diluted by warm-up and tunable to any value by sending
// more traffic. Four different rates for this one finding appear across our
// documents (0.875, 0.9375, 0.95, 0.96875), every one honestly transcribed and
// not one of them evidence. The delta is immune -- no denominator, no baseline,
// and no amount of unrelated traffic can move it.
//
// (An earlier version of this comment said the rate "never left ~0.94 (it read
// 15/16 before the experiment began)". BOTH HALVES WERE WRONG: 0.9375 is 15/16,
// which is request EIGHT OF TWELVE -- a mid-block snapshot, not a baseline. The
// before-rate was 7/8 = 0.875 and the block ended at 19/20 = 0.95, so the rate
// did move. @376a0297 caught it by checking the arithmetic in
// prefix-cache-verification.md:91 and :127 -- 7/8 + 12/12 = 19/20 exactly.)
// (An earlier timing arm reported shared prefixes slower. ITS OWN AUTHOR
// WITHDREW IT: the interleaved warm re-run came back with the opposite sign,
// on a box whose MEASURED null-A/B noise floor (perf-baseline.md §8.1, true
// delta ZERO by construction) reaches +52.30% / -40.17% between paired arms.
// (An earlier draft cited a 9.8% swing here; §6f retracted that figure as
// evidence because its window overlapped two CPU-heavy ONNX exports, so the
// swing had a cause and was not ambient. The real floor is ~5x larger, which
// makes this argument STRONGER, not weaker.)
// We ship no prefix timing number. Do not cite one.)
//
// `prefix_cache_hits` reads ~95% because it increments on
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
// BOTH SPELLINGS. The wire format is underscored (`prefix_cache_hits`); the
// dashboard's internal field keys are DOTTED (`prefix_cache.hits`). This list
// originally held only the wire spelling, so `dashboard/store-adapter.js:428`
// bound `prefix_cache.hits` in CAPABILITY_KEYS -- unallowlisted -- and the
// tripwire passed. A ban that greps for one spelling of a two-spelling field
// guards the half nobody was going to use.
const FORBIDDEN = [
  'prefix_cache_hits',
  'prefix_cache_lookups',
  'prefix_cache_hit_rate',
  'prefix_cache.hits',
  'prefix_cache.lookups',
  'prefix_cache.hit_rate',
];

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
  // ⚠️ THIS CHECK PASSED WITH THE ENTIRE DASHBOARD EMPTIED AND COMMITTED.
  //
  // Found by mutation: every tracked non-test file in examples/serving-dashboard
  // was truncated to zero bytes and committed in a throwaway worktree. Of the
  // six checks probed that way, five reddened. This one stayed 3 pass / 0 fail.
  //
  // The mechanism is the whole reason this test exists in the first place. Both
  // scanning tests below assert an ABSENCE over an enumerated set, and neither
  // put a floor under the enumeration -- so "no module names a forbidden
  // counter" and "there are no modules" produce the identical green. That is
  // absent-versus-zero, reported by the guard whose subject IS absent-versus-
  // zero, and it is the direction every broken checker fails in: toward passing.
  //
  // So this arm runs first and asserts the scan can see. It is not a test of
  // the product; it is the instrument's own calibration, and a red here
  // invalidates the two arms below rather than reporting a defect in the code.
  it('CAN RUN: the scan reaches real source and the scanner still fires', () => {
    const files = sourceFiles();

    // A count alone would not have caught the mutation that found this: the
    // files were all still THERE, they were merely empty. Bytes are the
    // property that actually went to zero.
    const bytes = files.reduce((total, file) => total + readFileSync(file, 'utf8').length, 0);
    assert.ok(
      files.length >= 20 && bytes >= 100_000,
      `the scan reached ${files.length} files totalling ${bytes} bytes. That is ` +
        `too little to be this dashboard, so the two absence checks below would ` +
        `pass by inspecting nothing. Fix the enumerator, not the ceiling.`,
    );

    // POSITIVE CONTROL, anchored on this file's OWN declared invariant rather
    // than on a string someone might delete: telemetry-provenance.js is in the
    // allowlist as PERMANENT, precisely because the register that forbids these
    // fields must name them. So the scanner MUST find one there. If it does
    // not, either the register stopped recording them -- which is a real
    // finding -- or `FORBIDDEN` no longer matches how they are spelled, which
    // would make every green below meaningless.
    const register = readFileSync(join(ROOT, 'telemetry-provenance.js'), 'utf8');
    const found = FORBIDDEN.filter((field) => register.includes(field));
    assert.ok(
      found.length > 0,
      'the scanner found NONE of the forbidden spellings in telemetry-provenance.js, ' +
        'which the allowlist calls "permanent: the register that forbids them". ' +
        `Either the register no longer names them, or FORBIDDEN (${FORBIDDEN.join(', ')}) ` +
        'has drifted from the spellings in use -- and a ban that matches no ' +
        'spelling in use is the exact defect that let store-adapter.js bind ' +
        'prefix_cache.hits while this test passed.',
    );

    // And the UI-copy regex, both directions: it must catch the banned phrase
    // and must not catch an innocent one. A regex that matches nothing and a
    // codebase that says nothing are indistinguishable from the assertion site.
    const RATE_COPY = /['"`][^'"`]*\b(cache hit rate|prefix hit rate|cache lookups)\b/i;
    assert.equal(RATE_COPY.test(`const label = 'Prefix hit rate';`), true, 'the UI-copy regex no longer fires');
    assert.equal(RATE_COPY.test(`const label = 'Tokens generated';`), false, 'the UI-copy regex flags everything');
  });

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
      'A module has bound a prefix-cache counter. These report ~95% hits on ' +
        'input that CANNOT produce reuse: twelve requests -- six repeated, six ' +
        'deliberately unique -- gave +12 hits, one per completed generation. The number is ' +
        `precisely computed and entirely false. Ruled unshippable in any form:\n  ${violations.join('\n  ')}`,
    );
  });

  it('has an allowlist that only shrinks', () => {
    // Pins the debt. Deleting a binding means deleting its entry here too, and
    // that edit is the thing a reviewer can actually see.
    assert.equal(
      ALLOWLIST.size,
      // 4 -> 5 on 2026-xx by @e00032a4. THIS IS NOT NEW DEBT AND THE RATCHET IS
      // NOT BROKEN. `dashboard/store-adapter.js` was ALREADY binding the fields;
      // it was invisible because FORBIDDEN held only the underscored wire
      // spelling and the store adapter uses the dotted key spelling. Widening
      // the ban DISCLOSED an existing binding rather than permitting a new one.
      // A ratchet may rise exactly once for this reason, and the reason must be
      // written down -- otherwise "it only shrinks" is enforced by a number
      // anyone can edit with a plausible excuse. Resumes shrink-only from 5.
      //
      // 5 -> 3 by @c8d9a40e: both dashboard entries are PAID, not excused.
      // `dashboard/prefix-cache.js` no longer binds or names any of these
      // fields -- the panel now renders the finding itself (the control arm,
      // the sensitivity check and the engine citations) with no telemetry
      // binding at all. `dashboard/store-adapter.js` lost its CAPABILITY_KEYS
      // entry, since a panel that reads nothing needs no capability probe.
      //
      // 3 -> 2 by @bb2ee824: app.js is PAID. Its per-origin comment cited
      // prefix_cache_hits purely as an EXAMPLE of one wire value classified two
      // ways; prefix_cache.hashes makes the same point and is not forbidden.
      //
      // telemetry-store.js is the last entry and it is BLOCKED, not forgotten.
      // The store names the field only in suppressUndefinedHitRate(), which
      // rewrites a 0.0 hit rate to unavailable when the denominator is 0 --
      // real, tested behaviour. It cannot go until the field stops being
      // published, and that is a ruling, not a refactor: see the dynamic-origin
      // finding recorded beside PROVENANCE['prefix_cache.hits'].
      2,
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
