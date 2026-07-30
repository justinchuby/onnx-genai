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
import { assertShippingTree, SHIPPING_REF, announceShippingRef } from './shipping-tree.mjs';

announceShippingRef();

// Provenance before content. Every path below is resolved from import.meta.url,
// so this file would read a parked worktree self-consistently and pass. Assert
// which tree we are in BEFORE asserting anything about what is in it.
assertShippingTree();

const demoDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = execFileSync('git', ['rev-parse', '--show-toplevel'], {
  cwd: demoDir,
  encoding: 'utf8',
}).trim();

// Every claim here is a claim about WHAT SHIPS, so read the committed bytes.
// This used to `readFileSync` the working tree, which reads correctly and means
// the wrong thing -- and the two are indistinguishable whenever the tree is
// clean, which is exactly when you are most likely to trust the result. The
// permitted failure is one-directional and it is the bad direction: a broken
// citation still in HEAD but repaired only on disk scores GREEN, and the repair
// evaporates on the next checkout. A reviewer clones HEAD. So does CI.
//
// It matters more here than almost anywhere else in this suite, because THIS
// tree is shared and usually dirty with other agents' work: resolving a
// citation against someone else's uncommitted edit can certify a line number
// that has never existed on the branch.
function shippedFile(relFromRoot) {
  // SHIPPING_REF rather than the literal `HEAD`. This file resolves a citation
  // in one file against a line number in another, so the two reads MUST come
  // from the same tree; a moving pointer can certify or condemn a line pairing
  // that never coexisted in any commit.
  return execFileSync('git', ['show', `${SHIPPING_REF}:${relFromRoot}`], {
    cwd: repoRoot,
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  });
}

const readme = shippedFile('examples/serving-dashboard/README.md');

// Citations are resolved by BASENAME, so the inventory must cover every kind of
// file the README argues from -- not just Rust. An earlier version tracked only
// `*.rs`, which meant the JS citations were silently unverifiable.
const CITED_EXTENSIONS = ['rs', 'js', 'css', 'html', 'sh'];

const trackedSourceFiles = execFileSync(
  'git',
  // `ls-tree`, not `ls-files`: the index can contain a file that the shipping
  // ref does not, so a citation into a newly-added-but-uncommitted file would
  // resolve.
  //
  // And SHIPPING_REF, not the literal `HEAD` this used to spell. `lineCountOf`
  // and `shippedFile` read the shipping ref, so a literal here builds the
  // candidate inventory from a different commit whenever a reviewer pins one
  // via SHIPPING_TREE_REF or REVIEW_SHA -- the exact mixing the comment on
  // `shippedFile` forbids. Identical when neither is set.
  ['ls-tree', '-r', '--name-only', SHIPPING_REF],
  {
    cwd: repoRoot,
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
  },
)
  .split('\n')
  .filter(Boolean)
  .filter((f) => CITED_EXTENSIONS.some((ext) => f.endsWith(`.${ext}`)));

const byBasename = new Map();
for (const path of trackedSourceFiles) {
  const key = basename(path);
  if (!byBasename.has(key)) byBasename.set(key, []);
  byBasename.get(key).push(path);
}

const lineCounts = new Map();
function lineCountOf(path) {
  if (!lineCounts.has(path)) {
    lineCounts.set(path, shippedFile(path).split('\n').length);
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
    // A citation that spells out a DIRECTORY is making a stronger claim than a
    // bare filename, and it must be graded against that stronger claim. This
    // deliberately does NOT fall through to the basename lookup below: doing so
    // silently rescued `.../src/engine/batched.rs` -- a path with a directory
    // segment that does not exist -- because a file named `batched.rs` lives
    // elsewhere in the tree. The citation resolved, the test stayed green, and
    // the reader following the path landed nowhere.
    //
    // A wrong full path is WORSE than a bare filename, not better: it looks
    // authoritative, it is what a reader copies, and the basename fallback
    // graded it as if it had claimed nothing.
    return trackedSourceFiles.filter((p) => p === path || p.endsWith('/' + path));
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

// ---------------------------------------------------------------------------
// The gap this closes, and why the two tests above could never close it.
//
// Both tests above ask whether a citation is *resolvable*: does the file exist,
// is the line inside it. Neither asks whether the line still holds the thing the
// sentence says it holds. So a citation degrades in total silence -- code moves
// down twelve lines, `metrics.rs:112` keeps pointing inside `metrics.rs`, every
// test stays green, and the number now indicates an unrelated statement.
//
// That is this session's defect class exactly: the instrument is healthy, its
// output is accurate, and it does not mean what it appears to mean. The check
// inspects a STATE (line 112 exists) while the defect lives in a RELATIONSHIP
// (line 112 vs. the claim attached to it).
//
// Two real drifts were caught by hand in this README -- `metrics.rs:112`/`:145`
// had moved to `:115`/`:156`, and `state.rs:201-206` to `:205`. Hand-catching is
// not a strategy: it found these two because someone happened to re-read that
// paragraph. So it becomes mechanical here.
//
// The check: when the prose immediately around a citation names a code symbol in
// backticks, that symbol must appear WITHIN A FEW LINES of the cited one. This
// is deliberately conservative -- it verifies an anchor, not a claim -- because
// a checker with false positives gets deleted, and a deleted checker catches
// nothing. Citations with no nearby symbol are skipped rather than guessed at,
// and the skip count is asserted below so the check cannot quietly erode into
// covering nothing.
const ANCHOR_WINDOW = 6;

// How far back to read for a symbol. Roughly a sentence; long enough to reach
// "`GenerationMetrics::start()` ... (`metrics.rs:115`)", short enough not to
// wander into the previous claim.
const PROSE_LOOKBEHIND = 260;

// Same span as the lookbehind; both are clipped at the claim boundary, which is
// what actually stops an anchor being borrowed from a neighbouring claim. A raw
// character count was tried first and was the wrong instrument.
const PROSE_LOOKAHEAD = 260;

// Words that appear in backticks constantly and anchor nothing.
const NOT_AN_ANCHOR = new Set([
  'true', 'false', 'null', 'none', 'some', 'ok', 'err', 'self', 'let', 'const',
  'if', 'else', 'return', 'fn', 'pub', 'mut', 'measured', 'pending', 'stale',
  'unavailable', 'not-applicable', 'n', 'm', '0', '1',
]);

// A claim is the text between one citation and the next.
//
// This is the boundary rule that finally worked, after a character count and
// then a sentence-terminator regex both failed. Both failed the same way: they
// tried to detect where a THOUGHT ends, using punctuation as a proxy. But the
// README's prose routinely runs two cited facts into one sentence -- "the server
// builds its config at `cli.rs:127-133` ... so `allow_runtime_override` is
// always false (`config.rs:602`)" -- and no punctuation rule separates those,
// because grammatically they are one sentence.
//
// Citations delimit claims better than punctuation does, and for a reason worth
// keeping: a citation IS the boundary of a claim, by definition. If a second
// citation stands between the prose and the number, that prose is answering to
// the other number. Using the thing we are checking as its own delimiter is not
// circular here -- it is the only marker that tracks the unit of attribution
// rather than the unit of grammar.
const ANY_CITATION = new RegExp(CITATION.source, 'g');

// A symbol-anchored citation: a full path, then the symbol, e.g.
//   `crates/onnx-genai-engine/src/batched.rs`, `struct ContinuousBatchManager`
// This is the form we are migrating TO, so it has to be verifiable -- an
// unverified anchor that merely LOOKS durable is the worst of both.
const SYMBOL_ANCHORED =
  // The symbol must START WITH AN IDENTIFIER CHARACTER. Without that the
  // pattern also matched `path.rs`, `:156` -- a line-anchored CONTINUATION,
  // where the second backtick group is another line number, not a symbol. It
  // reported `:156` as a missing symbol: a real-looking failure produced
  // entirely by the matcher, on a citation that was never malformed.
  /`([A-Za-z0-9_./-]+\.(?:rs|js|css|html|sh))`,\s*`([A-Za-z_][^`\n]*)`/g;

function clipBefore(text) {
  const last = [...text.matchAll(ANY_CITATION)].pop();
  const fromCitation = last ? last.index + last[0].length : 0;
  const breaks = [...text.matchAll(/\n\n|\|\s/g)].pop();
  const fromBreak = breaks ? breaks.index + breaks[0].length : 0;
  return text.slice(Math.max(fromCitation, fromBreak));
}

function clipAfter(text) {
  const first = text.match(ANY_CITATION);
  const toCitation = first ? text.indexOf(first[0]) : text.length;
  const brk = text.match(/\n\n|\s\|/);
  const toBreak = brk ? brk.index : text.length;
  return text.slice(0, Math.min(toCitation, toBreak));
}

// The WIDE scope: the whole paragraph or table cell, ignoring citation
// boundaries entirely.
//
// Needed because the tight rule has one systematic false positive, and it is a
// pattern the README uses constantly and SHOULD use: one claim carrying two
// citations, a definition site and a call site. "`PrefixCache::evict_lru`
// (`prefix_cache.rs:151`), called from `paged_decode.rs:53`" is a single claim
// with the symbol named once and cited twice. Delimiting on citations splits it
// and leaves the second number with no anchor of its own -- or worse, with the
// NEXT claim's symbol, which is how this check first demanded that
// `evict_lru_hot` appear in a file that correctly contains `evict_lru`.
//
// So: tight first, wide as a fallback, and fail only when NOTHING anchors. That
// ordering matters. Reporting the tight anchors in the failure keeps the message
// specific, while deciding on the wide set keeps the check quiet when the prose
// is merely phrased in a way the parser did not anticipate.
//
// The bias is deliberate and it is the opposite of this project's usual one.
// Everywhere else we prefer to fail loudly rather than pass quietly. HERE A
// FALSE POSITIVE IS THE MORE EXPENSIVE ERROR, because the "fix" it demands is to
// change a CORRECT citation into a wrong one -- the checker would author the
// exact defect it was built to catch, with the authority of a failing test
// behind it. A missed drift costs one stale number; a manufactured one costs
// the credibility of every number beside it.
function clipWide(text, side) {
  const brk =
    side === 'before'
      ? [...text.matchAll(/\n\n|\|\s/g)].pop()
      : text.match(/\n\n|\s\|/);
  if (!brk) return text;
  return side === 'before' ? text.slice(brk.index + brk[0].length) : text.slice(0, brk.index);
}

function anchorsFor(citation, { tight }) {
  const at = readme.indexOf(citation.text);

  // Clip the lookbehind to the CURRENT claim. Without this, a citation inherits
  // symbols from the sentence before it and the check invents failures: the
  // README's line about `EngineConfig::from_yaml` sits directly in front of an
  // unrelated `cli.rs:127-133`, and the raw lookbehind demanded `from_yaml`
  // appear in cli.rs, which it never did and never should.
  //
  // Worth stating plainly, since this check exists to catch exactly this shape
  // of error in the README itself: A FALSE POSITIVE HERE IS NOT A HARMLESS
  // OVER-REPORT. It would have pushed a correct citation to be "fixed" into a
  // wrong one, so the checker would have MANUFACTURED the defect it screens for.
  // An anchor is only evidence when it belongs to the same claim as the number.
  const rawBefore = readme.slice(Math.max(0, at - PROSE_LOOKBEHIND), at);
  const before = tight ? clipBefore(rawBefore) : clipWide(rawBefore, 'before');

  // Prose usually names the symbol first ("`start()` (`metrics.rs:115`)"), but a
  // table cell reverses it ("`driver.rs:755` `handle_driver_command`"). Reading
  // only backwards made the first run attribute a row's failure to the symbol
  // from the row ABOVE -- it still caught the broken citation, but named the
  // wrong anchor, and a checker that misreports WHICH claim broke is one people
  // learn to distrust. The lookahead is short so it cannot reach the next claim.
  // Symmetric: clip the lookahead at the first claim boundary too, rather than
  // at an arbitrary character count. Anchors are only evidence when they belong
  // to the same claim as the number, and that is true on both sides of it.
  //
  // The run that forced this is worth recording, because the checker turned out
  // to be blind in EXACTLY THE WAY THE CODE IT CHECKS IS BLIND. The README says
  // `prefix_cache_hit_len` is passed as a hardcoded `0` at `batched.rs:262` --
  // and that citation is CORRECT. The symbol is absent from the cited line for
  // the very reason the sentence exists: it is a POSITIONAL argument, so the
  // parameter's name appears in the signature and nowhere near the call. A
  // check that looks for a name near a line cannot see an argument that has no
  // name at the call site, which is the same seam that let the bug live there.
  // The anchor that does work is `with_rng`, and it sits just past the old
  // 60-character horizon.
  const rawAfter = readme.slice(at + citation.text.length, at + citation.text.length + PROSE_LOOKAHEAD);
  const after = tight ? clipAfter(rawAfter) : clipWide(rawAfter, 'after');

  const anchors = new Set();
  for (const [, span] of (before + '\n' + after).matchAll(/`([^`\n]+)`/g)) {
    // Skip anything that is itself a file citation or a path.
    if (/\.(rs|js|css|html|sh)\b/.test(span)) continue;

    const symbol = span.trim().replace(/^\.+/, '').replace(/\(\)$/, '');

    // Route paths anchor better than symbols do, because they appear verbatim
    // in the source as a string literal. Excluding them cost a real defect: the
    // README documented `POST /v1/admin/vram-limit` when the registered route is
    // `/v1/admin/resources/vram-limit`, so the one instruction on that page a
    // reader would actually paste into a terminal returned 404. Every other
    // check passed it -- the FILE existed and the LINE existed; only the string
    // was wrong, and nothing was comparing the string.
    const route = symbol.replace(/^(?:GET|POST|PUT|PATCH|DELETE)\s+/i, '');
    if (/^\/[A-Za-z0-9/_{}.-]+$/.test(route)) {
      anchors.add(route);
      continue;
    }
    if (!/^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*$/.test(symbol)) continue;

    // For `Type::method`, the method name is what appears at the definition.
    const tail = symbol.split('::').pop();
    if (NOT_AN_ANCHOR.has(tail.toLowerCase()) || tail.length < 4) continue;
    anchors.add(tail);
  }
  return [...anchors];
}

test('a cited line still sits beside the symbol the prose names', () => {
  let checked = 0;

  const failures = [];
  const drifted = [];
  for (const citation of citations()) {
    const candidates = resolve(citation);
    if (candidates.length === 0) continue; // reported by an earlier test

    // An unqualified basename can resolve to many files -- there are 40 named
    // `lib.rs` -- and the first version SKIPPED those outright. That silently
    // exempted the loosest citations in the README from the strictest check,
    // which is precisely backwards: a bare `state.rs:80` is less trustworthy
    // than a path-qualified one, not more.
    //
    // Pass if the anchor lands near the cited line in ANY candidate. That is
    // weaker than the single-candidate case and honestly so -- it cannot tell
    // you WHICH state.rs you meant -- but it still catches a line number that
    // fits no candidate at all, and it turns a skip into coverage.

    const tightAnchors = anchorsFor(citation, { tight: true });
    const wideAnchors = anchorsFor(citation, { tight: false });
    if (wideAnchors.length === 0) continue;

    const perFile = candidates.map((c) => {
      // shippedFile, NOT readFileSync. This was the last call site still
      // reading the desk after the rest of the file was converted, and it is
      // the one that decides the verdict: with the symbol renamed in the
      // working copy and untouched in HEAD, this check reported that it
      // "appears ANYWHERE" in a file where it appears twice.
      const lines = shippedFile(c).split('\n');
      return { c, lines };
    });

    // THE GATE IS IDENTITY, NOT PROXIMITY.
    //
    // This previously required the symbol to appear within ANCHOR_WINDOW lines
    // of the cited number, which made it a line checker wearing a symbol's
    // name. @1cb42f0e disproved it with the acceptance criterion this check was
    // commissioned under: insert twenty blank lines above a cited symbol,
    // change nothing else, and it went RED. Nothing moved, nothing was renamed,
    // no claim in the README became false -- and the check failed. That is the
    // failure mode we are supposed to be eliminating, not reproducing: a
    // position-addressed assertion about an artefact that is still moving.
    //
    // So the pass/fail question is now the one that actually maps to a false
    // claim in the document: DOES THE SYMBOL THE PROSE NAMES EXIST IN THE FILE
    // THE CITATION NAMES? If it does, the reader can find it and the sentence
    // is true. If it does not, the citation is stale in the only way that
    // misleads -- it points at something that is not there.
    //
    // Line drift is still reported, with the exact current line numbers, but it
    // DOES NOT FAIL. Per the ruling: name the symbol, quote the text; the line
    // number may accompany both and may never substitute for either. A stale
    // line number is a stale hint next to a true statement. A stale symbol is a
    // false statement.
    const found = perFile.some(({ lines }) =>
      wideAnchors.some((a) => lines.some((l) => l.includes(a))),
    );

    if (found) {
      checked += 1;

      // Report drift without failing, so citations can be repaired in a batch
      // rather than blocking the suite on a number nobody reads.
      const from = Math.max(0, citation.line - 1 - ANCHOR_WINDOW);
      const near = perFile[0].lines
        .slice(from, citation.line + ANCHOR_WINDOW)
        .join('\n');
      if (!wideAnchors.some((a) => near.includes(a))) {
        const anchors = tightAnchors.length > 0 ? tightAnchors : wideAnchors;
        for (const anchor of anchors) {
          const at = [];
          perFile[0].lines.forEach((l, i) => {
            if (l.includes(anchor)) at.push(i + 1);
          });
          if (at.length > 0) {
            drifted.push(
              `${citation.text} -> \`${anchor}\` now at :${at.join(', :')} in ${perFile[0].c}`,
            );
          }
        }
      }
      continue;
    }

    const { c: reportFile, lines } = perFile[0];

    const anchors = tightAnchors.length > 0 ? tightAnchors : wideAnchors;
    // The symbol is in NO candidate file. Report where it does live, if
    // anywhere, rather than asserting what the citation should have said.
    //
    // The first draft named only `findIndex`'s hit and phrased it as "the code
    // moved: X is now at :N". On its first real failure that sentence pointed at
    // a CALL SITE rather than the definition -- confidently, and wrongly. This
    // checker exists to stop the README asserting a line number it has not
    // earned; it must not do that itself in its own error message. So it lists
    // the candidates and says "appears at", leaving the judgement where it
    // belongs. AN AUTOMATED SUGGESTION IS A CLAIM, AND IT INHERITS EVERY
    // OBLIGATION THAT APPLIES TO THE CLAIMS IT POLICES.
    const elsewhere = [];
    for (const anchor of anchors) {
      const at = [];
      lines.forEach((l, i) => {
        if (l.includes(anchor)) at.push(':' + (i + 1));
      });
      if (at.length > 0) elsewhere.push(`${anchor} appears at ${at.join(', ')}`);
    }

    // Collected rather than thrown, so ONE run reports EVERY stale anchor.
    // Failing on the first turns an n-defect report into a 1-defect report and
    // sends the reader away believing they are done; on a branch where the
    // tree moves under you, that costs a full edit-run cycle per citation.
    failures.push(
      `README.md cites ${citation.text} while the surrounding prose names ` +
        `${anchors.map((a) => '`' + a + '`').join(', ')} — but no such symbol ` +
        `appears ANYWHERE in ` +
        `${reportFile}${candidates.length > 1 ? ` (or ${candidates.length - 1} other file(s) of that name)` : ''}.\n` +
        (elsewhere.length > 0
          ? `  The code moved: ${elsewhere.join('; ')}. Update the citation.\n`
          : `  The symbol is not in that file at all — the citation may name ` +
            `the wrong file.\n`) +
        `  A citation that resolves is not a citation that is correct; this is ` +
        `the difference.`,
    );
  }

  if (drifted.length > 0) {
    console.log(
      `\n  note: ${drifted.length} citation line number(s) have drifted. The ` +
        `symbols still exist, so no claim is false; repair at leisure:\n    ` +
        drifted.join('\n    ') +
        '\n',
    );
  }

  assert.ok(
    failures.length === 0,
    `${failures.length} citation(s) name a symbol that is no longer at the ` +
      `cited line:\n\n${failures.join('\n\n')}`,
  );

  // SYMBOL-ANCHORED citations count toward coverage too, and this is not
  // bookkeeping -- without it THIS GUARD ACTIVELY OPPOSES THE FIX IT EXISTS TO
  // ENCOURAGE. Converting `batched.rs:101-110` to (`.../batched.rs`,
  // `struct ContinuousBatchManager`) removes a line number that rots and
  // replaces it with an anchor that survives every rebase -- strictly better,
  // and it dropped `checked` from 36 to 24 and turned the floor RED. The
  // prescribed remedy was "lower the floor and say why", which would have
  // ratcheted the guard weaker every time the docs got stronger.
  //
  // A symbol anchor is the EASIER thing to verify, not the harder one: the
  // symbol either occurs in the shipped file or it does not, with no window
  // and no proximity heuristic. So verify it and count it.
  // NOT `failures`: that array was already asserted above, so pushing to it
  // here would be silently dead -- a check that runs, finds a defect, and
  // reports nothing. Caught by asking where the array is read, not whether it
  // is written.
  const symbolFailures = [];
  for (const m of readme.matchAll(SYMBOL_ANCHORED)) {
    const [, citedPath, symbol] = m;
    const candidates = trackedSourceFiles.filter(
      (f) => f === citedPath || f.endsWith(`/${citedPath}`),
    );
    if (candidates.length !== 1) continue; // uniqueness is the other test's job
    if (!shippedFile(candidates[0]).includes(symbol)) {
      symbolFailures.push(
        `${citedPath} is cited for \`${symbol}\`, but that symbol does not ` +
          `occur in the file at HEAD. A symbol anchor that does not resolve is ` +
          `worse than a stale line number: it reads as durable.`,
      );
    }
    checked += 1;
  }

  assert.deepEqual(
    symbolFailures,
    [],
    `${symbolFailures.length} symbol-anchored citation(s) name a symbol that ` +
      `is not in the file at HEAD:\n  ` +
      symbolFailures.join('\n  '),
  );

  // A coverage floor, because this check degrades silently in the one way that
  // matters: tighten the anchor rules or reword the prose and it starts
  // skipping everything while still reporting green.
  //
  // The floor is set just under the CURRENT count (36), not at a token value.
  // It was first written as `>= 8`, which was worse than useless: coverage could
  // have collapsed by three quarters and still passed. That is the same mistake
  // this suite already caught once -- a floor asks whether the matcher found
  // SOMETHING, so it detects a matcher that DIES and is structurally blind to
  // one that quietly NARROWS. A floor far below actual coverage is not a weak
  // guard, it is a guard shaped like the incident rather than like the fault.
  //
  // Set it near the real number and let it be re-raised deliberately. If prose
  // edits legitimately drop a few citations, lower it in the same commit and
  // say why -- a floor you have to touch is a floor you have to think about.
  //
  // MUTATION: floor at 999 -> red, reporting 36 checked. Route mutated to
  // `/v1/admin/vram-limit` -> red. Citation `driver.rs:755` -> `:813` -> red,
  // listing every line the symbol actually occupies.
  assert.ok(
    checked >= 30,
    `Only ${checked} citations were anchor-checked, down from 36. This test ` +
      `verifies nothing it cannot anchor, so a drop here means coverage ` +
      `evaporated rather than that the README improved. NOTE: converting a ` +
      `line-anchored citation to a symbol anchor does NOT reduce this count -- ` +
      `symbol anchors are counted above. A real drop here means citations were ` +
      `DELETED or the matchers stopped matching.`,
  );
});

// A citation that resolves to the WRONG FILE is worse than one that resolves to
// nothing. A dead link tells the reader "this is stale" in one click; a bare
// basename that matches several tracked files lands them on REAL CODE IN THE
// WRONG CRATE, where every line looks plausible and nothing announces the
// error. They then conclude the DOCS ARE WRONG ABOUT THE DESIGN rather than
// that the docs are stale -- a strictly worse outcome that costs strictly more
// to discover.
//
// The existing checks in this file resolve by basename and are satisfied by ANY
// match, so they were green over ten citations that named a file in a crate the
// sentence was not about. `governor.rs` was the sharp one: two crates have one,
// both plausible in context.
//
// Reported by @1cb42f0e from @e00032a4's checker; re-measured here rather than
// taken on the report, and the report's one DEAD path had already been fixed by
// the time I looked -- which is itself the argument for a standing guard
// instead of a periodic sweep.
test('every source path the README cites resolves to exactly one tracked file', () => {
  const ambiguous = [];
  const dead = [];
  const seen = new Set();

  const inspect = (cited) => {
    if (!cited || seen.has(cited)) return;
    seen.add(cited);
    const candidates = cited.includes('/')
      ? trackedSourceFiles.filter((f) => f === cited || f.endsWith(`/${cited}`))
      : trackedSourceFiles.filter((f) => basename(f) === cited);
    if (candidates.length === 0) dead.push(cited);
    else if (candidates.length > 1) {
      ambiguous.push(`${cited} -> ${candidates.length} files: ${candidates.slice(0, 3).join(', ')}`);
    }
  };

  for (const match of readme.matchAll(ANY_CITATION)) inspect(match[1]);

  // SYMBOL-ANCHORED CITATIONS NAME A PATH TOO, AND IT WAS NOT BEING CHECKED.
  //
  // This loop was originally fed by ANY_CITATION alone, which matches only the
  // line-anchored `path:NNN` form. That made the floor below shrink every time
  // anyone did the exact thing the rest of this file tells them to do: convert
  // a line anchor into a symbol anchor. The population is DISTINCT SPELLINGS,
  // so replacing a bare `admin.rs:80` with a full path that is already cited
  // elsewhere removes a spelling and adds none -- the count falls because the
  // document got BETTER. A guard whose floor drops when you follow its own
  // advice will be read as drift, and the cheap way to silence it is to lower
  // the floor or revert the improvement. Both are wrong.
  //
  // The honest repair is more coverage, not a lower bar: the path in a symbol
  // anchor must resolve to exactly one tracked file for the same reason the
  // path in a line anchor must, and until now nothing asserted that at all.
  for (const match of readme.matchAll(SYMBOL_ANCHORED)) inspect(match[1]);

  // VACUITY FLOOR. Every list below is empty if the matcher stops matching, and
  // an empty offender list is exactly what success looks like. Prove we
  // inspected something before believing we found nothing.
  assert.ok(
    seen.size >= 20,
    `only ${seen.size} distinct source paths were extracted from the README; ` +
      'the citation matcher has drifted and this test is passing vacuously.',
  );

  assert.deepEqual(
    dead,
    [],
    `${dead.length} README citation(s) name a file that is not tracked at HEAD. ` +
      'A reviewer clicking these gets nothing:\n  ' +
      dead.join('\n  '),
  );
  assert.deepEqual(
    ambiguous,
    [],
    `${ambiguous.length} README citation(s) use a BARE BASENAME that matches ` +
      'more than one tracked file, so a reader cannot tell which crate is ' +
      'meant and will land in the wrong one without being told.\n  ' +
      ambiguous.join('\n  ') +
      '\n\nReplacement: write the FULL PATH from the repository root, and let ' +
      'the prose name the symbol. Do NOT fix this by correcting the line ' +
      'number -- the line number is the part that rots, and a repaired one is ' +
      'stale again by the next commit. The path plus a symbol name survives ' +
      'every rebase.',
  );
});

// RATCHET. This is not a claim about the README; it is a claim about THIS FILE,
// and it exists because the defect it forbids has now been introduced twice.
//
// Every content read here must go through `shippedFile`, because a citation is
// a promise to someone holding the shipped tree. The conversion away from
// `readFileSync` left one call site behind -- the one inside the symbol-identity
// check, which is the arm that decides the verdict -- and the result was
// demonstrable rather than theoretical: renaming a cited symbol in the working
// copy while leaving HEAD untouched made this file report that the symbol
// "appears ANYWHERE" in a file where it appears twice.
//
// It fails in both directions, and the second is the dangerous one. A symbol
// present on a desk and absent from the shipping tree turns this check GREEN
// over a README that is false for every reviewer who clones it.
test('RATCHET: this checker reads no source from the working tree', () => {
  const selfPath = fileURLToPath(import.meta.url);
  const self = readFileSync(selfPath, 'utf8');

  // Anti-vacuity: an empty or unreadable self is indistinguishable from a
  // clean one to every assertion below.
  assert.ok(
    self.length > 10_000,
    `read only ${self.length} bytes of this file; the scan has no subject`,
  );

  // A disk read of a path built from the repository root. The guard's own read
  // above is deliberately NOT of this shape -- its subject is the code about to
  // run, not a claim about what is published.
  const DESK_READ = /readFileSync\(\s*join\(\s*repoRoot/;

  // Positive control on the regex itself. A pattern that can no longer match
  // the construct it bans is indistinguishable, from the assertion below, from
  // a file that does not contain it.
  //
  // Assembled from fragments rather than written out: a literal specimen would
  // be found by the scan below in this very file, and the guard would report
  // its own probe as the violation.
  const SPECIMEN = 'const lines = readFileSync(' + "join(" + "repoRoot, c), 'utf8');";
  assert.match(
    SPECIMEN,
    DESK_READ,
    'the ban pattern no longer matches the exact line it was written to forbid',
  );

  assert.doesNotMatch(
    self,
    DESK_READ,
    'a working-tree read is back in this file. It reads correctly on your desk ' +
      'and grades the README against a tree no reviewer has. Use shippedFile().',
  );

  // Positive control on the replacement: the ban is only meaningful if the
  // sanctioned vocabulary is actually in use here.
  const shippedReads = (self.match(/\bshippedFile\(/g) ?? []).length;
  assert.ok(
    shippedReads >= 4,
    `only ${shippedReads} shippedFile() call(s); this file has stopped reading ` +
      'the shipping tree rather than started reading it correctly',
  );
});
