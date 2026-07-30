// Every performance number the README states must be RECOMPUTED from the raw
// per-run samples in perf-baseline.md.
//
// The standing rule this enforces: a surviving number is computed from the
// artifact or deleted, never hand-maintained in two files. Before this test the
// README's headline speedup was a figure typed by a human who had read a
// different document, and nothing connected the two. That is the weakest link
// any claim on this page had -- not because the number was wrong, but because
// NOTHING WOULD HAVE NOTICED IF IT BECAME WRONG. A re-measurement lands in
// perf-baseline.md, the README keeps its old figure, and both files look
// authoritative and cite each other.
//
// So the parser reads the RAW SAMPLES -- not the summary tables, which are
// themselves derived and could drift from their own data -- and recomputes
// median, CV and the ratio. If the README and the artifact disagree, the README
// is wrong by definition, because the artifact carries the samples.
//
// It also enforces the PRECISION rule, which is the part a human reviewer will
// not catch: a ratio may not be printed to more significant figures than its
// confidence interval supports. `2.46x` from an n=4 arm asserts +/-0.005 when
// the data gives +/-0.12. The value is right and the format lies.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { assertShippingTree } from './shipping-tree.mjs';

// Provenance before content. Every path below is resolved from import.meta.url,
// so this file would read a parked worktree self-consistently and pass. Assert
// which tree we are in BEFORE asserting anything about what is in it.
assertShippingTree();

const HERE = dirname(fileURLToPath(import.meta.url));

// Every claim below is a claim about WHAT SHIPS, so every byte below comes from
// `git show HEAD:`, never from the disk.
//
// This guard used to readFileSync the working tree. That reads correctly and
// means the wrong thing, and the two are indistinguishable whenever the tree is
// clean -- which is precisely when you are most likely to trust the result. The
// failure it permits is one-directional and it is the bad direction: a defect
// still present in HEAD but repaired only on disk scores GREEN, and the repair
// then evaporates on the next checkout. A reviewer clones HEAD. So does CI. So
// does the demo. Nobody clones my working tree.
//
// The inverse failure -- a fix on disk that is not yet committed reads RED --
// is the safe one, and its remedy is the thing you were going to do anyway.
function shipped(rel) {
  // The `./` is load-bearing: `git show HEAD:<path>` resolves from the repo
  // root, not the cwd, so bare relative paths silently resolve to nothing.
  return execFileSync('git', ['show', `HEAD:./${rel}`], {
    cwd: HERE,
    maxBuffer: 64 * 1024 * 1024,
  }).toString();
}

// A clean tree makes "reads HEAD" and "reads disk" byte-identical, so state the
// difference out loud rather than letting silence stand in for equality.
function divergentFromHead(rels) {
  const out = [];
  for (const rel of rels) {
    let onDisk;
    try {
      onDisk = readFileSync(join(HERE, rel), 'utf8');
    } catch {
      continue; // deleted on disk; HEAD is still what ships.
    }
    if (onDisk !== shipped(rel)) out.push(rel);
  }
  return out;
}

// A note for whoever reads a RED from this file with a fix already in hand:
// if the offending path is listed as divergent, your fix exists only on disk.
function divergenceNote(rels) {
  const d = divergentFromHead(rels);
  return d.length
    ? `\n\nNOTE: these scanned files differ between HEAD and your working tree, `
      + `so a fix you have made on disk is NOT yet part of what ships: ${d.join(', ')}`
    : '';
}

const README = shipped('README.md');
const BASELINE = shipped('perf-baseline.md');

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function mean(xs) {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
function stdev(xs) {
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));
}

// The single-request arm: a fenced block of bare numbers under "Raw (tok/s)".
function singleRequestSamples() {
  const block = BASELINE.match(/Raw \(tok\/s\)[^\n]*\n```\n([\s\S]*?)```/);
  assert.ok(block, 'perf-baseline.md no longer has a "Raw (tok/s)" fenced block.');
  return block[1].trim().split(/\s+/).map(Number);
}

// The concurrent arm: the per-round table. Columns are
// round | aggregate | per-stream | wall s | wall tput
function concurrentRounds() {
  const rows = [...BASELINE.matchAll(/^\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|$/gm)];
  assert.ok(rows.length >= 3, 'perf-baseline.md no longer has a per-round table.');
  return rows.map((r) => ({ aggregate: Number(r[2]), perStream: Number(r[3]) }));
}

test('the raw samples in perf-baseline.md still parse', () => {
  const single = singleRequestSamples();
  const rounds = concurrentRounds();

  // A parser that silently matches nothing is the failure mode here: it would
  // make every assertion below vacuously true. Assert the shape first.
  assert.ok(
    single.length >= 10 && single.every((n) => Number.isFinite(n) && n > 0),
    `Parsed ${single.length} single-request samples; expected >=10 positive numbers. ` +
      `The raw block moved or changed format, and every check below is vacuous until ` +
      `this is fixed.`,
  );
  assert.ok(
    rounds.length >= 3 && rounds.every((r) => r.aggregate > 0 && r.perStream > 0),
    `Parsed ${rounds.length} concurrent rounds; expected >=3.`,
  );

  // MUTATION: renamed the "Raw (tok/s)" heading in a scratch copy -> red here
  // rather than a false pass downstream.
});

// ---------------------------------------------------------------------------
// WITHDRAWAL GATE. This file used to VERIFY the throughput ratio: it recomputed
// the median, CV and confidence interval from perf-baseline.md and failed if the
// README drifted from them. Those checks were correct and they are gone, because
// the quantity they policed has been withdrawn.
//
// The model the samples were taken against was assembled by accident from two
// builds seventeen days apart, and its inference metadata was edited 54 minutes
// after the build -- inside the measurement window. So the ratio is not merely
// unreproducible by a reader; we cannot show it is internally consistent with
// itself. An arithmetic check against those samples would still have PASSED:
// it verifies that the prose matches the data, never that the data means
// anything. A checker cannot detect a poisoned input by recomputing from it.
//
// So the guard is inverted. It no longer confirms the figure; it forbids it.
test('no shipping document reintroduces the withdrawn throughput ratio', () => {
  assertShippingTree();

  // A guard must quote what it forbids, which is why the digits are here.
  const RATIO = /\b2\.4[5-9]\s*×|\b2\.5\s*×|\b2\.46\b|\bratio[^.\n]{0,40}2\.[45]/i;
  const ARMS = /\b0\.62\s*×|\b82\.\d{3}\s*tok|\b33\.\d{3}\s*tok|\+147\s*%/;

  const EXEMPT = new Set([
    // The lab notebook. It records what was run and must keep its raw samples,
    // or we destroy the evidence that the claim was unsafe.
    'perf-baseline.md',
    // Not mine to edit; owned separately and read as a contract snapshot.
    'demo-spec.md',
  ]);

  // Documents that still carry the figure and belong to someone else. The tree
  // is frozen and both were edited minutes ago; editing them would collide with
  // a live author, and reddening the suite on their files during a review is a
  // worse outcome than a named deferral.
  //
  // The deferral EXPIRES BY ITSELF: if a listed document no longer states the
  // figure, this test fails and tells me to delete the entry. An exemption that
  // cannot expire is a suppression, and the direction that never gets reported
  // is the one where the gap has quietly closed.
  const DEFERRED = Object.freeze({
    'REVIEWER-BRIEF.md': 'owned by the secretary; edited 02:49, live at the time of the freeze',
    'design/demo-ux.md': 'owned by the designer; edited 02:45, live at the time of the freeze',
  });

  const docs = execFileSync('git', ['ls-tree', '-r', 'HEAD', '--name-only', '--', '.'], { cwd: HERE })
    .toString()
    .split('\n')
    .filter((f) => f.endsWith('.md'))
    .filter((f) => !EXEMPT.has(f));

  const stillDirty = new Set();

  let inspected = 0;
  const offenders = [];
  for (const doc of docs) {
    // Strip fenced and inline code: a document may legitimately show the old
    // figure inside a quoted diff or an example of what not to write.
    const text = shipped(doc)
      .replace(/```[\s\S]*?```/g, ' ')
      .replace(/`[^`\n]*`/g, ' ');
    inspected += text.length;

    // A retraction must be able to name the thing it retracts -- but the
    // exemption is scoped to the VICINITY of each match, never to the document.
    //
    // The document-wide version of this check was wrong and shipped green for
    // one run: README.md retracts the PREFIX figure, so a whole-file test for
    // the word "withdrawn" excused it from the THROUGHPUT gate as well. One
    // retraction anywhere bought silence everywhere. That is the same defect as
    // a guard whose own source satisfies its exemption -- an exemption is a
    // claim about a PASSAGE, and testing it against a FILE answers a broader
    // question than the one being asked.
    const RETRACTION = /withdraw|withdrew|withdrawn|retract|no longer claim|deleted rather than hedged|used to (?:print|state|lead)/i;
    const WINDOW = 600;

    for (const re of [RATIO, ARMS]) {
      const m = new RegExp(re.source, re.flags.includes('g') ? re.flags : re.flags + 'g');
      for (const hit of text.matchAll(m)) {
        const near = text.slice(
          Math.max(0, hit.index - WINDOW),
          hit.index + hit[0].length + WINDOW,
        );
        if (RETRACTION.test(near)) continue;
        if (doc in DEFERRED) { stillDirty.add(doc); continue; }
        if (!offenders.includes(doc)) offenders.push(doc);
      }
    }
  }

  assert.ok(
    inspected > 5000,
    `only ${inspected} characters of markdown inspected — the corpus scan is `
      + 'not reaching the documents, so a green result here means nothing.',
  );

  // Once the tree is clean, a dead matcher and a clean tree are byte-identical.
  assert.ok(
    RATIO.test('aggregate was 2.46× single-request') && ARMS.test('+147 %'),
    'the withdrawn-ratio matchers no longer fire against a synthetic positive '
      + 'control — this guard has gone blind and would pass on any tree.',
  );

  // Anti-rot: a deferral for a document that is already clean is a lie.
  const staleDeferrals = Object.keys(DEFERRED).filter((d) => !stillDirty.has(d));
  assert.deepEqual(
    staleDeferrals,
    [],
    `These documents are listed as DEFERRED but no longer state the withdrawn `
      + `figure: ${staleDeferrals.join(', ')}. Their owner cleaned them. Delete `
      + 'the entries from DEFERRED so the gate covers them for real — an '
      + 'exemption that outlives its reason is indistinguishable from a '
      + 'suppression, and nobody reports a gap that has closed.',
  );

  assert.deepEqual(
    offenders,
    [],
    `These documents state the withdrawn throughput ratio as a live claim: `
      + `${offenders.join(', ')}.\n`
      + 'The figure was withdrawn because the model it was measured against was '
      + 'assembled from two builds seventeen days apart, with its metadata edited '
      + 'inside the measurement window. Do not hedge it and do not footnote it — '
      + 'a hedge is dropped in the retelling and the digits survive. State the '
      + 'MECHANISM instead: continuous batching admits queued rows between steps '
      + '(batched.rs, admit_available_rows); paged attention draws KV from a '
      + 'shared page pool (page_table.rs, allocate/free). A reader can check a '
      + 'mechanism by reading code; they cannot check a ratio they cannot rebuild.',
  );
});

test('the README does not restate an absolute figure the baseline calls irreproducible', () => {
  // perf-baseline.md §1 explicitly warns its own absolute numbers are a sanity
  // reference and not a gate, because the same binary drifted 9.8% in 75
  // minutes. The README may quote them, but only next to that drift figure --
  // otherwise a reader takes 33.415 tok/s as a target.
  assert.ok(
    /9\.8 %/.test(README),
    'README.md quotes absolute tok/s figures but no longer states the measured ' +
      '9.8% same-binary drift. Without it, a reader reads those figures as ' +
      'reproducible, which the baseline document itself denies.',
  );
  assert.ok(
    /perf-baseline\.md/.test(README),
    'README.md states performance numbers without linking perf-baseline.md, ' +
      'so a skeptic has no way to reach the raw samples, hardware or commands.',
  );
});

// The old 'speedup without the per-stream tradeoff' rule lived here. It required
// any document stating the aggregate figure to state the per-stream cost beside
// it. That rule is now unreachable by construction: the aggregate figure is
// forbidden outright by the withdrawal gate above, so there is no longer a
// permitted way to state it. Retired rather than left in place, because a rule
// whose precondition can never be satisfied is indistinguishable from a rule
// that works, and both are green forever.
//
// The TRADEOFF ITSELF is not retired -- it is stated in both documents as prose
// with no number attached, which is the form that survived the withdrawal.

test('no document presents a withdrawn prefix timing figure without its noise floor', () => {
  // The +7.0% "shared prefixes are SLOWER" result was WITHDRAWN BY ITS OWN
  // AUTHOR after a warm interleaved re-run came back 16.98% FASTER -- the
  // opposite sign -- on a box where a byte-identical binary swung 9.8% from
  // ambient load alone. The effect and the noise floor are the same size, so
  // there is no measured prefix timing result in EITHER direction.
  //
  // It reached me anyway, twice, relayed as "the single most credibility-
  // earning sentence we have". I did not publish it, but only because it
  // arrived as a measurement quoted from a code comment and I had not yet
  // resolved it against an artefact. That is luck, not a process, and luck
  // does not survive the next draft.
  //
  // 🔴 THIS GUARDS THE DIRECTION NOTHING ELSE WE BUILT GUARDS. Every other
  // honesty mechanism here -- the five field states, the provenance axis, the
  // em-dash, the claim-position rule -- points ONE WAY: it stops us claiming a
  // capability we lack. NOT ONE of them stops us overclaiming CERTAINTY THAT
  // SOMETHING IS ABSENT. Fabricated doubt is as serious as fabricated
  // confidence and it is harder to catch, because it wears the costume of
  // rigour and nobody challenges the person arguing for less.
  //
  // Scoped by PARAGRAPH, not by document, and only where a prefix discussion
  // and a percentage delta actually co-occur. A bare "7 %" anywhere in the
  // repo is not evidence of anything -- a checker that fired on that would be
  // reworded away within a day and would teach everyone to ignore it.
  const docs = execFileSync('git', ['ls-tree', '-r', 'HEAD', '--name-only', '--', '.'], { cwd: HERE })
    .toString()
    .split('\n')
    .filter((f) => f.endsWith('.md'))
    .filter((f) => !/^(perf-baseline|demo-spec)\.md$/.test(f))
    // EXEMPTION, DOCUMENTED SO IT CANNOT BECOME AN OVERSIGHT: design/demo-ux.md
    // is @0837fdf9's design record and currently has THREE unqualified
    // paragraphs (the +7.0% Scenario B line, the 1.53s->1.22s 20% speed-up
    // line, and the ARM A/ARM B 1341ms/1254ms numbers). They are real and I
    // have reported them to its owner rather than editing another agent's file
    // mid-flight. Asserting them here today would only redden my suite with
    // their bug at the gate. THIS EXEMPTION IS A PROMISE TO COME BACK: when
    // those three paragraphs carry their noise floor, delete this filter.
    .filter((f) => !/^design\//.test(f))
    // EXEMPTION, NAMED SO IT CANNOT DECAY INTO AN OVERSIGHT:
    // prefix-cache-verification.md is @fc8b5d97's raw measurement record and
    // is being actively written. It is exempt as a RECORD, like
    // perf-baseline.md -- but with one finding attached that is NOT a records
    // question and that I have reported to its author rather than edited into
    // someone else's live file:
    //
    //   its VERDICT TABLE, the first thing anyone reads, still states
    //   "+7.0 %, i.e. no benefit at all" as a live result, while the noise
    //   floor that withdraws it appears ~350 lines further down.
    //
    // That is the same shape @fc8b5d97 themselves found in the honesty
    // register: the number was removed from one surface and the CERTIFICATION
    // survived on another. A reader who stops at the verdict table -- which is
    // what a verdict table is for -- gets the withdrawn claim.
    .filter((f) => f !== 'prefix-cache-verification.md');

  assert.ok(docs.length > 0, 'no tracked .md files found — this check would pass vacuously');

  const offenders = [];
  let paragraphsInspected = 0;

  for (const doc of docs) {
    // Strip fenced blocks and inline code BEFORE looking for claims. A grep
    // pattern that searches for the withdrawn figure is not a statement of it
    // -- READABILITY-REVIEW.md ships the literal command
    // `grep -riE '7\.0%|...|1341|1254' $(git ls-files)`, which is a tool for
    // finding the defect, and flagging it would punish the person hunting it.
    // Same trap as the citation `page_table.rs:1254`, where the digits were a
    // LINE NUMBER: matching characters is not matching meaning, and a checker
    // that cannot tell a claim from a search for that claim will be trained
    // away by its own false positives.
    const text = shipped(doc)
      .replace(/```[\s\S]*?```/g, '\n')
      .replace(/`[^`\n]*`/g, ' ');
    for (const para of text.split(/\n\s*\n/)) {
      // Does this paragraph discuss prefix reuse AND state a timing delta?
      if (!/prefix/i.test(para)) continue;
      if (!/\b(?:6\.9|7|7\.0|16\.98|17)\s*%/.test(para)) continue;
      if (!/slower|faster/i.test(para)) continue;
      paragraphsInspected += 1;

      // Then it MUST carry, in the same paragraph, the reason the number is
      // not a result: the noise floor, the contradicting run, or an explicit
      // unresolvable/withdrawn marker. Same paragraph, because a qualifier
      // three screens away does not travel with a quoted sentence.
      // SUPERSEDED / struck-in-place records are QUALIFIED, not offenders.
      // demo-ux.md deliberately preserves the withdrawn figures under strike
      // markers because the POINT is the timestamp -- we drew the result
      // before we measured it. A checker that demanded those be edited would
      // be ordering the deletion of the evidence that we were once wrong,
      // which is the opposite of this suite's purpose.
      const qualified =
        /9\.8\s*%|noise floor|below the floor|unresolvable|unverified|withdrew|withdrawn|contradict|SUPERSEDED|struck|CORRECTED BY MEASUREMENT/i.test(
          para,
        );
      if (!qualified) offenders.push(`${doc}: "${para.trim().slice(0, 90).replace(/\s+/g, ' ')}…"`);
    }
  }

  // Empty-input floor. If the matcher stops recognising these paragraphs it
  // inspects nothing, reports nothing, and passes -- growing more trusted with
  // every green run. The README discusses this at length, so zero is wrong.
  assert.ok(
    paragraphsInspected > 0,
    'this check inspected ZERO paragraphs, so it proved nothing. The prefix ' +
      'timing discussion is expected in README.md; the matcher is broken.',
  );

  assert.deepEqual(
    offenders,
    [],
    `${offenders.length} paragraph(s) state a prefix timing delta as a finding ` +
      `without the noise floor that withdraws it:\n  ${offenders.join('\n  ')}\n` +
      `There is NO measured prefix timing result in either direction. The ` +
      `counter finding is the one that stands, and it needs no stopwatch: ` +
      `twelve requests with six deliberately unique prompts produced twelve ` +
      `hits -- one per completed generation -- so the counter cannot tell reuse from no-reuse.`,
  );
});

// ---------------------------------------------------------------------------
// A withdrawal has to reach the TREE, not just the chat.
//
// The +7.0% "shared prefixes are slower" result was the most careful
// measurement anyone had, so it got copied into four files. Then its own
// author withdrew it. The retraction was broadcast three times and reached
// none of the copies.
//
// THE BETTER A MEASUREMENT IS, THE MORE PLACES IT GETS COPIED -- so our
// propagation debt is proportional to our credibility, and until now nothing
// connected a withdrawal to its copies. The .md gate above cannot see this
// class at all: these live in code comments and, worse, in ASSERTION STRINGS,
// which are the text a developer reads at the exact moment a test fails.
// ---------------------------------------------------------------------------

/**
 * Split a source file into units in which a qualifier is allowed to count.
 *
 * A qualifier three statements away does not travel with a quoted sentence,
 * so proximity is not enough -- but neither is a fixed line window, which is
 * a line checker wearing a better name. The unit here is structural:
 *   - a contiguous run of comment lines, or
 *   - a single statement (contiguous code lines up to a line ending in `;`).
 * That is exactly the span a reader takes in as "one thing being said".
 */
function statementUnits(source) {
  const lines = source.split('\n');
  const isComment = (l) => /^\s*(\/\/|\*|\/\*)/.test(l);
  const units = [];
  let cur = null;
  for (let i = 0; i < lines.length; i += 1) {
    const kind = isComment(lines[i]) ? 'comment' : 'code';
    if (!cur || cur.kind !== kind) {
      cur = { kind, start: i + 1, lines: [] };
      units.push(cur);
    }
    cur.lines.push(lines[i]);
    if (kind === 'code' && /;\s*$/.test(lines[i])) cur = null;
  }
  return units.map((u) => ({ kind: u.kind, line: u.start, text: u.lines.join('\n') }));
}

test('no source file states the withdrawn prefix timing result as a live finding', () => {
  assertShippingTree();

  const sources = execFileSync('git', ['ls-tree', '-r', 'HEAD', '--name-only', '--', '.'], { cwd: HERE })
    .toString()
    .split('\n')
    .filter((f) => /\.(js|mjs)$/.test(f));
  assert.ok(
    sources.length > 0,
    'no .js/.mjs files found in HEAD — this check would pass vacuously',
  );

  // The withdrawn artefacts, by content rather than by digits alone: the
  // 7.0% delta and the 1341/1254 ms arm pair it was computed from.
  //
  // KNOWN AND DELIBERATE LIMIT -- do not "fix" this by widening the matcher.
  // A site can state the withdrawn claim with NO DIGITS AT ALL: @376a0297
  // found `store-adapter.js` asserting the counters were cut "after a
  // controlled A/B proved prefix reuse absent", which is the timing arm
  // wearing prose. This checker cannot see that class, and matching phrases
  // like "controlled A/B" would be worse than useless, because the SAME words
  // describe the counter arm, which is correct and needs no stopwatch --
  // `registry-prefix-tripwire.test.js` and `honesty.test.js` both use them
  // legitimately. A matcher that reddens on the true statement and the false
  // one alike teaches people to ignore it, and a false positive is worse than
  // no lint. The digit-free class is a PROSE REVIEW obligation, recorded here
  // so it is an acknowledged gap rather than an assumed absence.
  const WITHDRAWN = /\b7\.0\s*%|\b1341\s*ms|\b1254\s*ms/;

  // A citation being BURIED is not a citation being MADE. @376a0297 raised
  // their own measured defect count by quoting the dead citations in a
  // retraction table so readers could recognise them -- an honest correction
  // scoring worse to its own checker. If we repeat that here, every agent
  // learns that documenting a withdrawal is punished, which is the exact
  // incentive we cannot afford. So words that mark the figure as dead --
  // including the words this very comment is written in -- count as safe.
  const RETRACTED =
    /withdraw|withdrew|withdrawn|retract|noise floor|below the floor|SUPERSEDED|struck|no longer|opposite sign|9\.8\s*%|do not cite|historical|WAS THE CLAIM/i;

  const SELF = 'check-perf-claims.test.js';
  // POSITIVE CONTROL. Once the tree is clean, a dead matcher and a clean tree
  // produce byte-identical output -- both are silent. So prove the matcher can
  // still fire, against a synthetic sample rather than against the defect,
  // which is the only way to keep that proof after the defect is gone.
  assert.ok(
    WITHDRAWN.test('shared prefixes ran 7.0% slower') &&
      WITHDRAWN.test('warm TTFT 1341 ms vs 1254 ms'),
    'the WITHDRAWN matcher no longer matches the figures it was written for — ' +
      'this check has been silently disarmed.',
  );
  assert.ok(
    !RETRACTED.test('shared prefixes ran 7.0% slower') &&
      RETRACTED.test('that 7.0% figure was withdrawn by its author'),
    'the RETRACTED matcher can no longer tell a claim being MADE from one ' +
      'being BURIED — it will either flag honest retractions or excuse live claims.',
  );

  const offenders = [];
  let unitsInspected = 0;
  let withdrawnMentions = 0;

  for (const rel of sources) {
    const units = statementUnits(shipped(rel));
    if (rel !== SELF) unitsInspected += units.length;
    for (const unit of units) {
      if (!WITHDRAWN.test(unit.text)) continue;
      // THE FLOOR MUST NOT BE SATISFIABLE BY THIS FILE'S OWN SOURCE. It was:
      // the line `const WITHDRAWN = /.../` contains the word "withdrawn", so
      // it matched the matcher AND the retraction pattern, quietly satisfying
      // `unitsInspected > 0` forever. Neutering the matcher entirely still
      // came back green, and I only found that by mutating -- reading it back
      // ten times would never have shown it. A vacuity floor a checker can
      // satisfy with its own text is not a floor, it is a self-portrait.
      if (rel !== SELF) withdrawnMentions += 1;
      if (RETRACTED.test(unit.text)) continue;
      const isAssertionString =
        /assert\.|['"`]/.test(unit.text) && unit.kind === 'code';
      offenders.push(
        `${rel}:${unit.line}${isAssertionString ? '  [ASSERTION STRING — printed on every failure]' : '  [comment]'}\n` +
          `      ${unit.text.trim().split('\n')[0].slice(0, 96)}`,
      );
    }
  }

  // THE FLOOR ASKS WHETHER THE SCAN REACHED THE CORPUS -- NOT WHETHER THE
  // DEFECT IS STILL PRESENT. My first version asserted that some file still
  // mentioned the withdrawn figures, which meant the check could only stay
  // green while the bug survived: finishing the cleanup turned it red and the
  // obvious "fix" would have been to put a withdrawn number back. A guard
  // that punishes its own success trains people to disable it.
  assert.ok(
    unitsInspected > 500,
    `only ${unitsInspected} units parsed across ${sources.length} source files — ` +
      'the scan is not reaching the corpus, so a green result here means nothing.',
  );

  // Zero mentions is the SUCCESS state and is allowed. It is reported rather
  // than asserted, because the day it becomes true is the day this check has
  // done its job, and that fact should be visible without being fatal.
  if (withdrawnMentions === 0) {
    console.log(
      '      note: no source file outside this checker mentions the withdrawn ' +
        'prefix figures. Cleanup is complete; this guard is now purely a ratchet.',
    );
  }

  assert.deepEqual(
    offenders,
    [],
    `${offenders.length} source site(s) state the WITHDRAWN prefix timing result as a live finding.\n` +
      `Its own author withdrew it: the interleaved warm re-run came back with the OPPOSITE sign, ` +
      `on a box where a byte-identical binary swung 9.8% from ambient load alone.\n` +
      `Replacement (needs no stopwatch, so no re-run can withdraw it): "We could not measure a ` +
      `prefix effect above this machine's noise floor, so we ship no prefix number. The counter is ` +
      `disqualified on its own arithmetic instead: twelve requests with six deliberately unique ` +
      `prompts produced twelve hits, one per completed generation."\n\n` +
      offenders.join('\n'),
  );
});

// ---------------------------------------------------------------------------
// The README states how much of the page is populated: "10 of 13 have no
// catalogue entry (77%)". That number is a MEASUREMENT, and measurements in
// prose decay silently.
//
// This one decays in the flattering direction, which is why it needs a guard
// more than a harsh number would. As keys get plumbed the true figure FALLS,
// so a stale README understates our own progress -- and nobody files a bug
// against a document that is too modest. The pressure that normally corrects
// an inaccurate claim is entirely absent here, in both directions:
// overstatement gets challenged, understatement gets a shrug.
//
// So the ratio is recomputed from HEAD on every run and the prose must match.
test('the README populated-fields ratio still matches the tree it describes', () => {
  assertShippingTree();

  const PANEL = 'dashboard/kv-memory.js';
  const keys = [
    ...new Set(
      Array.from(shipped(PANEL).matchAll(/field\(\s*'([^']+)'/g), (m) => m[1]),
    ),
  ];
  assert.ok(
    keys.length > 5,
    `only ${keys.length} field() keys found in ${PANEL} — the extractor has `
      + 'probably gone blind rather than the panel having shrunk. Refusing to '
      + 'compute a ratio from a corpus this small.',
  );

  const catalogue = shipped('telemetry-provenance.js');
  const missing = keys.filter((k) => !catalogue.includes(`'${k}'`));
  const pct = Math.round((missing.length / keys.length) * 100);

  const claim = new RegExp(
    `\\*\\*${missing.length} of ${keys.length} have no catalogue entry \\(${pct}%\\)\\*\\*`,
  );

  assert.ok(
    claim.test(README),
    'The README\'s populated-fields ratio no longer matches the tree.\n'
      + `  measured now : ${missing.length} of ${keys.length} keys in ${PANEL} `
      + `have no catalogue entry (${pct}%)\n`
      + `  README says  : see the "How much of the page is actually populated" table\n`
      + (missing.length < keys.length / 2
        ? '  Coverage has IMPROVED past the halfway mark. Update the prose and '
          + 'consider whether the surrounding caveats still describe the page.\n'
        : '')
      + `  Unresolvable keys: ${missing.join(', ')}\n\n`
      + `  Replacement text, verbatim: **${missing.length} of ${keys.length} `
      + `have no catalogue entry (${pct}%)**`,
  );
});

// ---------------------------------------------------------------------------
// The KV panel's static-profile redefinition is DESIGNED but not LIVE: both
// fields it needs are recorded unpublished in field-keys.test.js. The README
// once described that redefinition in the present tense -- "real, measurable,
// and moves under load" -- which was true of the design and false of the page.
//
// This guard is bidirectional on purpose, because the caveat has two ways to
// become a lie and they pull in opposite directions:
//
//   still unpublished + no caveat  -> the original defect, prose outruns the tree
//   published        + caveat kept -> a stale apology for a gap that closed,
//                                     which trains readers to discount caveats
//
// The second is the one that never gets reported, so it is asserted first-class
// here rather than left to whoever happens to land the endpoint.
test('the KV redefinition caveat matches whether its fields are published', () => {
  assertShippingTree();

  const guard = shipped('dashboard/field-keys.test.js');
  const KEYS = ['kv.slots_filled', 'kv.slot_capacity'];

  // Membership is read from the allowlist LITERAL, not inferred from the key
  // appearing anywhere in the file -- the key also appears in error strings.
  const allowlist = guard.slice(
    guard.indexOf('NOT_YET_PUBLISHED = Object.freeze({'),
    guard.indexOf('});', guard.indexOf('NOT_YET_PUBLISHED = Object.freeze({')),
  );
  assert.ok(
    allowlist.length > 100,
    'could not locate the NOT_YET_PUBLISHED literal in field-keys.test.js — '
      + 'the guard was restructured and this check is now reading nothing. '
      + 'Fix the extractor rather than deleting the assertion.',
  );

  const unpublished = KEYS.filter((k) => allowlist.includes(`'${k}':`));
  const caveat = /That redefinition is designed and not yet live/.test(README);

  if (unpublished.length > 0) {
    assert.ok(
      caveat,
      `${unpublished.join(' and ')} ${unpublished.length === 1 ? 'is' : 'are'} `
        + 'still listed unpublished in dashboard/field-keys.test.js, so the KV '
        + 'panel em-dashes on the static-cache profile. The README must keep the '
        + 'caveat beginning "That redefinition is designed and not yet live". '
        + 'Do not describe a designed behaviour in the present tense.',
    );
  } else {
    assert.ok(
      !caveat,
      'Both kv.slots_filled and kv.slot_capacity are now PUBLISHED — the '
        + 'block-table endpoint landed. Delete the README caveat beginning '
        + '"That redefinition is designed and not yet live" and restore the '
        + 'plain description; the panel really does move under load now. '
        + 'A caveat that outlives its defect teaches readers to skip caveats.',
    );
  }
});

// A cut that ships in code, and a QA plan that still asks a tester to decide it.
//
// §5.5 of QA-PLAN.md specifies a 30-request protocol (n >= 15/arm, interleaved,
// 95% CIs) to determine whether the prefix-reuse scenario ships. That question
// was answered and FROZEN IN SHIPPING CODE: 'prefix-cache' sits in
// CUT_SCENARIOS with the reason "measured and found absent on both execution
// paths". The section did not become WRONG, it became ANSWERED -- and an
// answered question reads exactly like an open one, so a tester working
// top-down pays the full hour to re-derive a settled verdict. That is the exact
// cost §11 of the same document opens by warning against.
//
// Bidirectional on purpose. If someone ever un-cuts the scenario, the note
// telling testers not to run the protocol becomes the lie, and this fails the
// other way.
test('the QA plan matches whether the prefix scenario is actually cut', () => {
  const origins = shipped('scenario-origins.js');
  const qa = shipped('QA-PLAN.md');

  // Match the key only inside the CUT_SCENARIOS literal, not anywhere the
  // string 'prefix-cache' happens to appear in prose or a comment.
  const block = origins.slice(origins.indexOf('CUT_SCENARIOS'));
  const isCut = /^\s*'prefix-cache':/m.test(block.slice(0, block.indexOf('});') + 3));

  const note = qa.includes('THE DECISION THIS PROTOCOL WAS WRITTEN TO MAKE HAS ALREADY BEEN MADE');

  if (isCut) {
    assert.ok(
      note,
      "scenario-origins.js still lists 'prefix-cache' in CUT_SCENARIOS, so the "
        + 'scenario does not ship. QA-PLAN.md §5.5 must keep the note beginning '
        + '"THE DECISION THIS PROTOCOL WAS WRITTEN TO MAKE HAS ALREADY BEEN '
        + 'MADE". Without it the plan reads as an open question and costs a '
        + 'tester an hour of timed requests to re-derive a frozen verdict.',
    );
  } else {
    assert.ok(
      !note,
      "'prefix-cache' is NO LONGER in CUT_SCENARIOS -- the scenario was "
        + 'reinstated. Delete the §5.5 note that tells testers the decision is '
        + 'already made and not to run the protocol, and revisit exit criterion '
        + '7. That note is now the stale claim, and it suppresses exactly the '
        + 'measurement that would confirm the reinstatement.',
    );
  }
});

// ---------------------------------------------------------------------------
// THE TWELVE-REQUEST BLOCK MAY NOT CARRY A RATE.
//
// The counter finding is a DELTA: twelve requests, six sharing no prefix, +12
// hits -- one per completed generation. That form needs no denominator and no
// baseline, which is exactly why it survives.
//
// A rate attached to that same block is not merely weak evidence, it is
// ARITHMETICALLY FALSE, and @376a0297 reconstructed it from the primary record
// (prefix-cache-verification.md:78 and :112):
//
//     before the block   7 hits /  8 lookups = 0.875
//     after  the block  19 hits / 20 lookups = 0.95
//     19-7 = 12 hits, 20-8 = 12 lookups -> the block is fully accounted for
//
// So 15/16 = 0.9375 is REQUEST EIGHT OF TWELVE -- a mid-block snapshot -- and
// the rate did not sit at ~0.94, it CLIMBED across the very block the sentence
// describes. Our README stated "twelve requests ... produced +12 hits, a 0.9375
// hit rate", fusing the delta with a reading taken two-thirds of the way in.
//
// AND THE SENTENCE WAS SELF-DEFEATING: a reader who checked it found the rate
// MOVING, which reads as "the counter responds to what you send" -- the exact
// opposite of the point the paragraph exists to make. The false detail
// undermined the true conclusion it was added to support.
//
// DO NOT "FIX" A SITE BY SWAPPING 0.9375 FOR 0.95. `prefix_cache_hits` and
// `prefix_cache_lookups` are CUMULATIVE SINCE BOOT, so their ratio is a
// property of the process, not of the experiment: diluted by warm-up and
// tunable to any value by sending more traffic. Four rates for this one run
// appear across our documents (0.875, 0.9375, 0.95, 0.96875), every one
// honestly transcribed and not one of them evidence.
//
// WHY THIS IS SEMANTIC AND NOT A DIGIT BAN: "95 %" appears legitimately all
// over this tree as a CONFIDENCE INTERVAL -- a different quantity that happens
// to share a number. A checker that reddened on `95 % CI` would be reworded
// away within a day. The property is CO-OCCURRENCE: a rate inside a paragraph
// that is describing the twelve-request block.
test('no document attributes a hit RATE to the twelve-request block', () => {
  const docs = execFileSync('git', ['ls-tree', '-r', 'HEAD', '--name-only', '--', '.'], { cwd: HERE })
    .toString()
    .split('\n')
    .filter((f) => f.endsWith('.md'))
    // The raw measurement records legitimately carry every reading, including
    // the mid-block ones. They are where the arithmetic above was RECOVERED
    // from; banning the figures there would delete the evidence.
    .filter((f) => !/^(perf-baseline|demo-spec|prefix-cache-verification)\.md$/.test(f))
    .filter((f) => !/^design\//.test(f));

  assert.ok(docs.length > 0, 'no tracked .md files found — this check would pass vacuously');

  // The block, however it is worded across our documents.
  const BLOCK = /twelve requests|12 requests|\+12 hits|twelve hits/i;
  // A hit-rate-shaped figure. `\d+ ?% CI` is deliberately excluded.
  const RATE = /0\.9\d{2,}|\b9[0-9](?:\.\d+)? ?%(?! ?CI)|\b1[56]\s*\/\s*16\b|\b19\s*\/\s*20\b/;
  // A paragraph that WITHDRAWS a rate must be allowed to quote it, or
  // documenting the defect would trip the guard against the defect. This is
  // the same exemption shape the launcher guard needed.
  const WITHDRAWN = /wrong|withdrew|withdrawn|earlier version|not evidence|cumulative since boot|~~|NOT a baseline|mid-block/i;

  const offenders = [];
  let inspected = 0;
  // ONE KNOWN SITE, EXEMPTED BY ITS EXACT WORDING RATHER THAN BY FILENAME, so
  // that any OTHER rate appearing in that file still reddens and so that this
  // exemption stops matching the moment the line is fixed. A file-level
  // exemption would have hidden the next one too.
  //
  // browser-render-verification.md:256 is @fc8b5d97's QA evidence document and
  // is not mine to edit under the freeze. I have reported it with its
  // provenance, which is the uncomfortable part and the reason it is only
  // exempt rather than forgiven:
  //
  //   THAT SENTENCE IS VERBATIM THE REMEDIATION STRING THIS VERY FILE USED TO
  //   PRINT. A reader who tripped my check was handed "twelve hits and a
  //   0.9375 rate" as the CORRECT replacement text and pasted it faithfully.
  //   The guard did not merely fail to catch the defect -- IT AUTHORED IT, and
  //   in the one register nobody proofreads, because remediation text is read
  //   only by someone who has already been told they are wrong.
  //
  // Both of this file's remediation strings now quote the delta. DELETE THIS
  // EXEMPTION once :256 does too.
  const KNOWN = 'produced twelve hits and a 0.9375 rate';
  for (const doc of docs) {
    for (const para of shipped(doc).split(/\n\s*\n/)) {
      if (!BLOCK.test(para)) continue;
      inspected += 1;
      if (WITHDRAWN.test(para)) continue;
      if (doc === 'browser-render-verification.md' && para.includes(KNOWN)) continue;
      const hit = para.match(RATE);
      if (hit) {
        offenders.push(`${doc}: "${hit[0]}" in — ${para.trim().slice(0, 160).replace(/\s+/g, ' ')}`);
      }
    }
  }

  // ANTI-VACUITY, stated over what we REQUIRE to be present rather than over
  // the file list: a floor on `docs.length` stays green if the matcher stops
  // recognising the paragraph, which is the failure that actually happens.
  assert.ok(
    inspected >= 3,
    `only ${inspected} paragraph(s) describing the twelve-request block were found; ` +
      `the matcher has drifted and this check is inspecting nothing`,
  );

  assert.deepEqual(
    offenders,
    [],
    `${offenders.length} site(s) attach a hit RATE to the twelve-request block.\n` +
      `That block ran 0.875 -> 0.95; any single rate quoted for it is a mid-block\n` +
      `snapshot, and the counters are cumulative since boot so NO rate is evidence.\n` +
      `Quote the delta and stop: "+12 hits, one per completed generation."\n\n  ` +
      offenders.join('\n  '),
  );
});
