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
import {
  assertShippingTree,
  SHIPPING_REF,
  announceShippingRef,
  shippedPaths,
} from './shipping-tree.mjs';

announceShippingRef();

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
  // The `./` is load-bearing: `git show <ref>:<path>` resolves from the repo
  // root, not the cwd, so bare relative paths silently resolve to nothing.
  //
  // SHIPPING_REF, not the literal `HEAD`: this file reads several inputs and
  // compares them against each other, and `HEAD` is a pointer that moves on
  // this branch mid-run. Two reads through it can land in different trees and
  // report a contradiction that existed in no commit.
  return execFileSync('git', ['show', `${SHIPPING_REF}:./${rel}`], {
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
  //
  // `[×xX]` and not `×` alone. The typographic multiplication sign was the only
  // form matched for most of this branch's life, and a plain ASCII `2.46x` --
  // which is what anyone types who is not copying from an existing document --
  // slipped straight through. `\b2\.46\b` does not save it either: in `2.46x`
  // the digit and the letter are both word characters, so there is no boundary
  // between them and the alternative never fires.
  //
  // Measured before widening, so this is not a speculative hardening: five
  // shipped documents already carry the ASCII form, all five with a retraction
  // beside them, and widening the pattern produced ZERO new offenders. The
  // blindness was real and was being masked by the fact that every existing
  // ASCII occurrence happened to be already withdrawn.
  // The `-fold` arm was added after a probe showed `2.5-fold` evading a digit
  // this matcher already claimed to cover. "fold" is how a figure gets written
  // when someone is deliberately avoiding the × they know is contested.
  const RATIO = /\b2\.4[5-9]\s*(?:[×xX]|-?fold)|\b2\.5\s*(?:[×xX]|-?fold)|\b2\.46\b|\bratio[^.\n]{0,40}2\.[45]/i;
  const ARMS = /\b0\.62\s*×|\b82\.\d{3}\s*tok|\b33\.\d{3}\s*tok|\+147\s*%/;

  // THE ENVELOPE BOUNDS -- a laundering route that OPENED WHEN THE FIX LANDED.
  //
  // perf-baseline.md §12 (a81a6d54, @fc8b5d97) nulled this box at a worst-case
  // pair spread of +58.41 %, and drew the correct narrower conclusion: the
  // envelope does not forbid a 2.46× effect from EXISTING, it forbids the
  // DIGITS. It then prescribes the honest rendering: "roughly 1.6× to 3.9×".
  //
  // That prescription is right, and it handed this gate a hole. RATIO is pinned
  // to 2.45-2.5. It cannot see EITHER bound. So "throughput gains of up to
  // 3.9×" passes clean -- a BIGGER number than the withdrawn one, wearing §12's
  // authority, and unlike a bare 2.46× it looks like it came from an error
  // analysis. That is the form most likely to survive a retelling intact.
  //
  // THE DISTINCTION THIS ENCODES: the envelope is honest ONLY AS AN ENVELOPE.
  // Both bounds together are §12's prescribed rendering and are excused. ONE
  // BOUND ALONE IS NOT A RANGE, IT IS A HEADLINE -- "up to 3.9×" and "at least
  // 1.6× faster" are the two ways to quote an uncertainty interval as a result.
  // A guard that banned the range outright would forbid the very rendering §12
  // prescribes, so this cannot be a matcher alone; it needs the pairing rule.
  const ENVELOPE = /\b1\.6\s*[×xX]|\b3\.9\s*[×xX]/;
  // The separator forbids a PARAGRAPH break but permits a single newline:
  // markdown reflows, and PR-DESCRIPTION.md:340 states the range with the two
  // bounds on adjacent lines. A pairing rule that only recognises a range on
  // one physical line would call that honest rendering a bare bound.
  // ONE constant decides how far apart the two bounds may sit. PAIR_WINDOW used
  // to be a second, independent knob, and widening it from 120 to 100000 changed
  // no outcome and no test noticed -- because SEP's period-free limit was doing
  // all the real work. Two spellings of one decision cannot be mutation-tested:
  // they agree in every reachable state. The window is now DERIVED, so there is
  // exactly one number to get wrong.
  const PAIR_MAX_GAP = 60;
  const SEP = `(?:[^.\\n]|\\n(?!\\n)){0,${PAIR_MAX_GAP}}`;
  const ENVELOPE_BOTH = new RegExp(
    `1\\.6\\s*[×xX]${SEP}3\\.9\\s*[×xX]|3\\.9\\s*[×xX]${SEP}1\\.6\\s*[×xX]`,
  );
  // Wide enough that a legal pair is never cut in half by the slice itself.
  const PAIR_WINDOW = PAIR_MAX_GAP + 60;

  // PERMANENT exclusions, SCOPED TO A PATTERN CLASS RATHER THAN TO A FILE.
  //
  // The bar is a reason that is true of the FILE'S CONTENT and can never stop
  // being true. "Somebody else owns it" does not clear that bar: ownership is
  // fixed, content is not, so an ownership reason produces an entry that
  // outlives the condition it was written for.
  //
  // WHY THIS IS NO LONGER A WHOLE-FILE EXCLUSION. It used to remove the file
  // from `docs` entirely, which meant the ONE bucket with no expiry and no
  // vicinity discipline was also the STRONGEST bucket -- `DEFERRED`, the weaker
  // notation, had both. The lab notebook was therefore the only document in the
  // corpus that could state the withdrawn CONCLUSION with nothing nearby, and
  // no instrument would ever say so.
  //
  // Measured before narrowing it, because the exemption is load-bearing and
  // deleting it outright would have destroyed the evidence it protects:
  //   perf-baseline.md   RATIO (the conclusion) : 5 matches, 0 uncovered
  //                      ARMS  (raw samples)    : 10 matches, 9 uncovered
  // The raw samples are what must survive -- a lab notebook that cannot keep
  // its own numbers stops being evidence that the claim was unsafe. The
  // conclusion is already withdrawn at every site. So the exemption buys
  // exactly nothing on RATIO today, and narrowing it costs nothing today and
  // catches the next person who writes the ratio into the notebook bare.
  //
  // The rule this encodes is the QA owner's, in their words: RAW SAMPLES ARE
  // EXEMPT, A CONCLUSION NEVER IS.
  const SAMPLE_ONLY_EXEMPT = new Map([
    [
      'perf-baseline.md',
      'The lab notebook. It records what was run and must keep its raw per-arm '
        + 'samples, or we destroy the evidence that the claim was unsafe. This '
        + 'excuses the ARMS figures ONLY -- the withdrawn ratio itself is held '
        + 'to the same vicinity rule as every other document.',
    ],
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
  //
  // ONE HONEST LIMIT ON THAT PROMISE, MEASURED 07:57 AND NOT THEORISED. The
  // expiry fires on the figure being ABSENT. A document that fixes itself
  // WELL -- striking the old value in place and saying what it used to read --
  // ends up containing MORE occurrences than before, not fewer: demo-ux.md went
  // 14 -> 31 while the defect was being repaired. So this deferral CANNOT
  // expire on a well-documented fix, only on a silent one. That is the exact
  // inversion of what we want to reward, and it is the same blind spot named in
  // the demo-spec.md row: a strike record quotes the thing it strikes, so
  // presence stops tracking currency the moment anyone writes a good obituary.
  // The count RISING is therefore a prompt to READ, never evidence of a
  // regression -- and the row above records what reading it found.
  const DEFERRED = Object.freeze({
    'REVIEWER-BRIEF.md': 'owned by the secretary; edited 02:49, live at the time of the freeze',
    'design/demo-ux.md':
      'The designer STRUCK the withdrawn hero (verified in HEAD 07:57: the '
      + '`AC50/D85 compliant` badge is gone, 1 -> 0; the sketch reads '
      + '`2.46-2.72x`, the sanctioned interval; the two residual bare hits are '
      + 'past-tense strike records reading "PREVIOUSLY READ ... WERE STRUCK"). '
      + 'SO THE ORIGINAL REASON -- "owned by the designer; live at the time of '
      + 'the freeze" -- IS SPENT, AND THIS ENTRY IS NO LONGER HELD BY IT. It is '
      + 'held by the same content property as demo-spec.md above: the file '
      + 'states the figure as the SUBJECT of a retraction, which no pattern '
      + 'here can tell from an assertion. I MEASURED THIS RATHER THAN ASSUMING '
      + 'IT -- deleting this line yields 18/19 exit 1, "states the withdrawn '
      + 'throughput ratio as a live claim: design/demo-ux.md", against a '
      + 'document that is correctly fixed. THAT WOULD BE A FALSE RED ON A '
      + 'REPAIRED FILE. Do not drop it on the strike condition; the strikes '
      + 'have ALREADY landed and it is not what holds this row.',
    // Moved here from EXEMPT by @376a0297, who audited MY exemption list against
    // their own file -- the check nobody runs on their own work. It sat in the
    // permanent bucket on the reason "not mine to edit", which is a fact about
    // AUTHORSHIP, while the two entries above are in the expiring bucket on the
    // identical situation. Same condition, two notations, and the notation was
    // chosen by which day I wrote the line.
    //
    // The cost was not a false pass. It is that a permanent entry naming the
    // SPEC reads, to every later maintainer, as "the spec is allowed to state
    // the ratio" -- and a spec is where people go to learn what is permitted.
    'demo-spec.md': 'owned by the product manager; states the figure as the SUBJECT of a retraction, which no pattern here can tell from an assertion',
  });

  const docs = shippedPaths()
    .filter((f) => f.endsWith('.md'));

  // ---------------------------------------------------------------------
  // DECLARE THE CORPUS.
  //
  // The scope above is CORRECT and it is UNDECLARED, and those are different
  // problems. `-- .` with `cwd: HERE` restricts this guard to ONE DIRECTORY.
  // Everything else in the repository -- PROGRESS.md, the decisions archive,
  // every design note -- is invisible to it and always has been.
  //
  // That is deliberate: widening this corpus would redden it permanently on
  // documents nobody is willing to edit tonight, and a guard that cannot be
  // satisfied gets deleted within a day. A narrow guard that runs beats a
  // total guard that gets switched off.
  //
  // The DEFECT is what the exemption lists imply. Four hand-written entries
  // read as "we examine everything except these four" -- an exemption list is
  // a claim about coverage, and a SHORT one claims almost everything. The two
  // accidentally-visible keys in NOT_YET_PUBLISHED read as a deliberate survey
  // for exactly this reason. So state the denominator out loud on every run.
  //
  // Everything below is DERIVED, never enumerated. A hand-maintained list of
  // what we do not cover would rot the same way, and rot into a MORE
  // authoritative-looking artefact -- centralising a fact does not verify it,
  // it makes it wrong in one place instead of several.
  //
  // This block cannot fail. It asserts nothing; it only refuses to let the
  // scope stay implicit.
  const prefix = execFileSync('git', ['rev-parse', '--show-prefix'], { cwd: HERE })
    .toString()
    .trim();
  // `--full-tree` is LOAD-BEARING AND THE FIRST VERSION OF THIS BLOCK OMITTED
  // IT. Without it, `git ls-tree` silently restricts itself to the cwd and
  // prints paths RELATIVE to the cwd -- so this listing returned only this
  // directory, nothing matched `prefix`, and the "not examined" figure came out
  // as 15 when the true answer is 546.
  //
  // A declaration written to expose under-coverage under-reported it by a
  // factor of thirty-six, and it did so SILENTLY and in the FLATTERING
  // direction, which is the only direction nobody re-checks. `--full-tree`
  // makes the listing independent of where it is called from, which is the
  // same property this whole module exists to guarantee.
  const repoMd = execFileSync(
    'git',
    // `shippedPaths()` is directory-scoped and this arm needs the whole
    // repository, so it stays a direct call -- but it is pinned to the same
    // SHIPPING_REF as every read it feeds. Literal `HEAD` here would let the
    // corpus and the contents come from different commits.
    ['ls-tree', '-r', '--full-tree', SHIPPING_REF, '--name-only'],
    { cwd: HERE, maxBuffer: 64 * 1024 * 1024 },
  )
    .toString()
    .split('\n')
    .filter((f) => f.endsWith('.md'));
  const unexamined = repoMd.filter((f) => !f.startsWith(prefix));
  const say = (s) => console.log(`CORPUS-SCOPE: ${s}`);
  say(`this guard reads ${docs.length} .md file(s), and ONLY under ${prefix}`);
  say(`  raw-sample exempt  : ${SAMPLE_ONLY_EXEMPT.size} (${[...SAMPLE_ONLY_EXEMPT.keys()].join(', ')}) — ARMS figures only; the withdrawn RATIO is scanned in these files like everywhere else`);
  say(`  in-scope, deferred : ${Object.keys(DEFERRED).length} (${Object.keys(DEFERRED).join(', ')})`);
  say(`  NOT EXAMINED AT ALL: ${unexamined.length} of ${repoMd.length} .md file(s) tracked in this repository`);
  for (const f of unexamined.slice(0, 5)) say(`      e.g. ${f}`);
  if (unexamined.length > 5) say(`      ... and ${unexamined.length - 5} more`);
  say('  the exemptions above are exclusions from a ONE-DIRECTORY corpus, NOT');
  say('  from the repository. A green result here is silence about the rest.');
  // ---------------------------------------------------------------------

  const stillDirty = new Set();
  // How MANY live-claim sites each deferred document carries, not merely
  // whether it carries any. A Set answers "is this document deferred", which is
  // the question the deferral already answered; it cannot see a deferred
  // document acquiring ten new claims, because the set membership never changes.
  const deferredLiveClaims = new Map();
  // Which arms actually ran. The form controls below prove the MATCHERS see a
  // bare bound; they cannot prove the loop still ASKS. Today's corpus contains
  // no bare bound, so deleting 'ENVELOPE' from the label list would remove the
  // scan and change no result -- a guard that stops looking and stays green.
  const scannedLabels = new Set();
  const sampleExemptionUsed = new Set();

  let inspected = 0;
  const offenders = [];
  // One shared scan, called by the corpus loop AND by synthetic-text
  // assertions below. That sharing is the point, not tidiness: today's corpus
  // contains no BARE envelope bound, so mutating the pairing rule to `if (true)`
  // or widening its window to the whole document changed no result and no test
  // noticed. A branch only reachable when a document misbehaves cannot be
  // proven by a corpus of well-behaved documents -- so the hazard is passed in
  // as DATA instead of waited for.
  const RETRACTION = /withdraw|withdrew|withdrawn|retract|no longer claim|deleted rather than hedged|used to (?:print|state|lead)/i;
  const WINDOW = 600;

  const scanHits = (text, { sampleExempt }) => {
    const sites = [];
    const labels = new Set();
    let armsExcused = 0;
    // The raw-sample exemption applies HERE and only here: to the ARMS figures,
    // in the documents that must retain them. RATIO and ENVELOPE fall through
    // to the ordinary vicinity rule in every document without exception.
    for (const [label, re] of [['RATIO', RATIO], ['ARMS', ARMS], ['ENVELOPE', ENVELOPE]]) {
      labels.add(label);
      const excuseArms = label === 'ARMS' && sampleExempt;
      const m = new RegExp(re.source, re.flags.includes('g') ? re.flags : re.flags + 'g');
      for (const hit of text.matchAll(m)) {
        if (excuseArms) { armsExcused += 1; continue; }
        // An envelope bound is excused when its PARTNER sits alongside it --
        // that is §12's prescribed range, stated whole. A bound with no partner
        // nearby is a bound being used as a figure.
        if (label === 'ENVELOPE') {
          const tight = text.slice(
            Math.max(0, hit.index - PAIR_WINDOW),
            hit.index + hit[0].length + PAIR_WINDOW,
          );
          if (ENVELOPE_BOTH.test(tight)) continue;
        }
        const near = text.slice(
          Math.max(0, hit.index - WINDOW),
          hit.index + hit[0].length + WINDOW,
        );
        if (RETRACTION.test(near)) continue;
        sites.push({ label, index: hit.index });
      }
    }
    return { sites, armsExcused, labels };
  };

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
    const { sites, armsExcused, labels } = scanHits(text, {
      sampleExempt: SAMPLE_ONLY_EXEMPT.has(doc),
    });
    for (const l of labels) scannedLabels.add(l);
    // The matches are still COUNTED before being excused. Skipping the scan
    // outright would mark the exemption "used" on a file that no longer
    // contains a single raw sample, which is precisely the stale-entry failure
    // the DEFERRED bucket already guards against.
    if (armsExcused > 0) sampleExemptionUsed.add(doc);
    for (const _site of sites) {
      if (doc in DEFERRED) {
        stillDirty.add(doc);
        deferredLiveClaims.set(doc, (deferredLiveClaims.get(doc) ?? 0) + 1);
        continue;
      }
      if (!offenders.includes(doc)) offenders.push(doc);
    }
  }

  assert.ok(
    inspected > 5000,
    `only ${inspected} characters of markdown inspected — the corpus scan is `
      + 'not reaching the documents, so a green result here means nothing.',
  );

  // Once the tree is clean, a dead matcher and a clean tree are byte-identical.
  //
  // The ASCII arm is asserted SEPARATELY from the typographic one on purpose:
  // a single control containing `2.46×` passed for this guard's whole life
  // while `2.46x` was invisible, so one probe per FORM is the only version of
  // this control that can detect the gap it is here to detect.
  assert.ok(
    RATIO.test('aggregate was 2.46× single-request') && ARMS.test('+147 %'),
    'the withdrawn-ratio matchers no longer fire against a synthetic positive '
      + 'control — this guard has gone blind and would pass on any tree.',
  );
  assert.ok(
    RATIO.test('aggregate was 2.46x single-request') && RATIO.test('a 2.46X gain'),
    'the withdrawn-ratio matcher no longer fires on the plain ASCII form. That '
      + 'is the form a person TYPES rather than copies, so this is the arm most '
      + 'likely to carry a fresh claim, and it is the arm that was missing.',
  );

  // THE HAZARD AS DATA. Each of these was a surviving mutation before it was
  // written -- the pairing rule could be replaced with `if (true)` and its
  // window widened to 100000 with every test still green, because no real
  // document contains a bare bound to notice.
  const envLabels = (text) => scanHits(text, { sampleExempt: false })
    .sites.filter((x) => x.label === 'ENVELOPE').length;

  assert.equal(
    envLabels('throughput gains of up to 3.9× with continuous batching'),
    1,
    'a BARE upper bound is no longer reported. This is the laundered form of '
      + 'the withdrawn claim: a bigger number than 2.46×, wearing §12\'s error '
      + 'analysis, which it earned only as one END of an interval.',
  );
  assert.equal(
    envLabels('at least 1.6× faster than single-request decoding'),
    1,
    'a BARE lower bound is no longer reported. "At least" reads as a floor and '
      + 'is the more persuasive half of the interval, not the more modest one.',
  );
  assert.equal(
    envLabels('the honest envelope is roughly 1.6× to 3.9× on this hardware'),
    0,
    'the pairing rule stopped excusing §12\'s prescribed rendering. This guard '
      + 'must not forbid the sentence the baseline tells authors to write, or '
      + 'the next person to hit it deletes the guard instead of the claim.',
  );
  assert.equal(
    envLabels(`up to 3.9× and then${' padding '.repeat(30)}later a 1.6× mention`),
    2,
    'two bounds separated by 240 period-free characters are now pairing. '
      + 'PAIR_MAX_GAP has been widened past a PASSAGE into a FILE, which is the '
      + 'same widening this file was split to fix for the sample exemption.',
  );
  assert.equal(
    envLabels(`up to 3.9× in the headline.${' filler.'.repeat(400)} elsewhere 1.6× appears`),
    2,
    'two bounds thousands of characters apart are now excusing each other as a '
      + '"range". That is the pairing window widened from a PASSAGE to a FILE — '
      + 'the same widening this file was split to fix for the sample exemption, '
      + 'rebuilt in the newest arm.',
  );
  assert.equal(
    envLabels('the 3.9× figure was withdrawn once the box was nulled'),
    0,
    'the ordinary retraction rule stopped applying to envelope bounds. A bound '
      + 'quoted in order to strike it must stay legal, or documenting the '
      + 'withdrawal becomes an offence and authors stop writing strike records.',
  );

  assert.deepEqual(
    [...scannedLabels].sort(),
    ['ARMS', 'ENVELOPE', 'RATIO'],
    'an arm stopped being scanned. This is asserted separately from the matcher '
      + 'controls because they answer different questions: a control proves the '
      + 'regex can still SEE a bare bound, this proves the loop still ASKS it. '
      + 'No document currently holds a bare bound, so dropping an arm would '
      + 'change no offender list and this file would stay green while blind.',
  );

  // One control per FORM, the rule this file already learned the hard way when
  // a single `2.46×` probe kept the guard green while `2.46x` was invisible.
  // These four are the four ways to quote an uncertainty interval as a result.
  assert.ok(
    ENVELOPE.test('gains of up to 3.9× with batching')
      && ENVELOPE.test('at least 1.6× faster than single-request'),
    'the envelope matcher no longer fires on a BARE bound. §12 prescribes '
      + '"roughly 1.6× to 3.9×" as the honest rendering, which means both bounds '
      + 'are now quotable digits — and a lone bound is the laundered form of the '
      + 'withdrawn claim, wearing an error analysis it did not earn.',
  );
  assert.ok(
    ENVELOPE_BOTH.test('roughly 1.6× to 3.9× on this hardware')
      && ENVELOPE_BOTH.test('somewhere between roughly 1.6x and\n3.9x survives'),
    'the pairing rule no longer recognises §12\'s prescribed range, including '
      + 'the form that wraps across a line break. Without this the guard forbids '
      + 'the exact rendering the baseline tells authors to use, and the next '
      + 'person to hit it will delete the guard rather than the claim.',
  );
  assert.ok(
    !ENVELOPE_BOTH.test('up to 3.9× today.  Unrelated: 1.6× fewer allocations'),
    'the pairing rule now spans a sentence boundary, so two unrelated figures '
      + 'in adjacent sentences excuse each other as a "range". That turns the '
      + 'pairing rule into a blanket exemption for both bounds.',
  );
  assert.ok(
    RATIO.test('a 2.5-fold improvement') && RATIO.test('a 2.47-fold improvement'),
    'the fold-form arm has gone blind. "fold" is how a contested figure gets '
      + 'rewritten when the author knows the × is contested — it was found '
      + 'evading a digit this matcher already claimed to cover.',
  );

  // Anti-rot, the SAME rule the deferrals get, now applied to the bucket that
  // never had it. A raw-sample exemption for a document with no raw samples
  // left is an entry that outlived its reason -- and the direction that never
  // gets reported is the one where the gap has quietly closed.
  const staleSampleExemptions = [...SAMPLE_ONLY_EXEMPT.keys()]
    .filter((d) => docs.includes(d))
    .filter((d) => !sampleExemptionUsed.has(d));
  assert.deepEqual(
    staleSampleExemptions,
    [],
    `These documents hold a raw-sample exemption but no longer contain a single `
      + `per-arm figure: ${staleSampleExemptions.join(', ')}. The reason the `
      + 'exemption was granted is gone, so delete the entry and let the gate '
      + 'cover the file for real.',
  );

  // And the exemption must not have quietly become a whole-file pass. If the
  // notebook stops being scanned for the RATIO, this guard is back to where it
  // started and nothing else in the file would say so.
  assert.ok(
    [...SAMPLE_ONLY_EXEMPT.keys()].every((d) => docs.includes(d)),
    'a raw-sample-exempt document is missing from the scanned corpus — the '
      + 'exemption has widened from a pattern class back to the whole file, '
      + 'which is the defect this bucket was split to fix.',
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

  // THE DRAINED CORPUS, RATCHETED. @12e42da8's rule, and the reason a plain
  // intersection assertion is the wrong shape for it.
  //
  // THE RULE: an exemption is a statement about RAW EVIDENCE; the moment an
  // exempt file states a CONCLUSION, the exemption is a suppression. Every
  // document that still states the withdrawn ratio is exempt or deferred, so
  // this guard is green because its corpus drained to exactly the set of files
  // that no longer make the claim. No single decision was an error and the
  // aggregate is a suppression -- there is no bad commit to find.
  //
  // WHY NOT `assert stillDirty is empty`, WHICH IS THE THREE-LINE VERSION:
  // it is RED THE MOMENT IT IS WRITTEN, on three documents owned by three other
  // agents, mid-review. A guard that reddens instantly gets an exemption bolted
  // onto it within the hour, and we have rebuilt the disease one level up. The
  // suppression is real; the emergency is not.
  //
  // WHY A COUNT AND NOT A SET: `stillDirty` is a Set, so it answers "is this
  // document deferred" -- a question the deferral already answered. IT CANNOT
  // SEE A DEFERRED DOCUMENT ACQUIRING TEN NEW CLAIMS, because membership never
  // changes. That is the actual hole: today a deferral is an unmetered licence.
  //
  // Numbers measured at 53e5e7d9, not guessed. They count MATCH SITES that are
  // NOT within +/-600 characters of retraction language -- so a struck figure,
  // or one stated as the subject of a retraction, is already excused above and
  // never reaches this map. Every site counted here is a bare live claim.
  const DEFERRED_CLAIM_CEILING = Object.freeze({
    'REVIEWER-BRIEF.md': 2,
    'demo-spec.md': 1,
    'design/demo-ux.md': 6,
  });

  for (const doc of Object.keys(DEFERRED)) {
    const seen = deferredLiveClaims.get(doc) ?? 0;
    const ceiling = DEFERRED_CLAIM_CEILING[doc];

    assert.notEqual(
      ceiling,
      undefined,
      `${doc} is DEFERRED but has no entry in DEFERRED_CLAIM_CEILING. A deferral `
        + 'without a number is an unmetered licence: the document may acquire any '
        + 'number of new claims and this gate will stay green, because set '
        + 'membership never changes. Measure it and pin it.',
    );
    assert.ok(
      seen <= ceiling,
      `${doc} now states the withdrawn ratio as a LIVE claim at ${seen} site(s), `
        + `up from the pinned ${ceiling}. THIS IS A REGRESSION AND THE GATE COULD `
        + 'NOT HAVE SEEN IT BEFORE: the document is deferred, so its claims are '
        + 'not reported as offences. Somebody added a bare statement of a '
        + 'withdrawn figure to a document the gate is not covering. Remove the new '
        + 'site, or state it next to its retraction so the vicinity rule excuses '
        + 'it honestly.',
    );
    assert.equal(
      seen,
      ceiling,
      `${doc} is down to ${seen} live claim site(s) from the pinned ${ceiling}. `
        + 'THIS IS GOOD NEWS AND IT IS STILL A FAILURE, DELIBERATELY: lower the '
        + `number to ${seen} in DEFERRED_CLAIM_CEILING in the same commit that `
        + 'earned it, or delete the DEFERRED entry entirely if it is now 0 and '
        + 'let the gate cover the file for real. A ceiling that is never lowered '
        + 'stops being a ratchet and becomes a permanent allowance -- which is '
        + 'the exact drain this guard exists to stop, one level up.',
    );
  }

  // No ceiling entry may outlive its deferral, or the numbers accumulate as a
  // record of documents nobody is watching any more.
  assert.deepEqual(
    Object.keys(DEFERRED_CLAIM_CEILING).filter((d) => !(d in DEFERRED)),
    [],
    'a DEFERRED_CLAIM_CEILING entry names a document that is no longer deferred. '
      + 'Delete it: a ceiling on a document the gate now covers in full is a '
      + 'number that can only mislead.',
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
  // reference and not a gate. The README may quote them, but only next to the
  // measured instability -- otherwise a reader takes 33.415 tok/s as a target.
  //
  // This assertion used to REQUIRE the README to state "9.8 %". That figure was
  // retracted as evidence by its own author (perf-baseline.md §6f -- the run
  // window overlapped two CPU-heavy ONNX exports, so the swing has a cause and
  // is not ambient noise), and the clean replacement (§8.1's null A/B, true
  // delta zero by construction) is ~5x larger. A GUARD THAT REQUIRES A
  // SUPERSEDED NUMBER DOES NOT MERELY PERMIT THE STALE TEXT -- IT MANDATES IT,
  // and would have gone RED on the correction.
  assert.ok(
    /\+?52\.30\s*%/.test(README),
    'README.md quotes absolute tok/s figures but no longer states the measured ' +
      'noise floor from perf-baseline.md §8.1 (worst single-pair excursion ' +
      '+52.30 % on a null A/B whose true delta is ZERO by construction). ' +
      'Without it a reader takes those figures as reproducible, which the ' +
      'baseline document itself denies. Do NOT satisfy this with the older ' +
      '9.8 % figure: §6f retracted it as evidence.',
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
  // opposite sign -- on a box whose MEASURED null-A/B noise floor reaches
  // +52.30% / -40.17% between paired arms whose true delta is ZERO by
  // construction (perf-baseline.md §8.1). The effect is far SMALLER than the
  // noise floor, so there is no measured prefix timing result in EITHER
  // direction.
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
  const docs = shippedPaths()
    .filter((f) => f.endsWith('.md'))
    .filter((f) => !/^(perf-baseline|demo-spec)\.md$/.test(f))
    // EXEMPTION RETIRED by @0837fdf9. It was written as an explicit promise to
    // come back, and the condition it named is now satisfied: all three
    // unqualified paragraphs (+7.0% Scenario B, the 1.53s->1.22s speed-up, and
    // the ARM A/ARM B 1341ms/1254ms pair) now carry their noise floor. Verified
    // by re-running this suite with the filter gone, not by accepting the
    // report: PARAGRAPHS INSPECTED WENT UP, 9 -> 10. That direction is the
    // whole check. A rewrite that made the paragraphs stop matching would also
    // return zero offenders, and it would look identical to a repair.
    // EXEMPTION RETIRED by @fc8b5d97. It was written as an explicit promise to
    // come back: the verdict table stated "+7.0 %, i.e. no benefit at all" as a
    // live result while the noise floor withdrawing it sat ~350 lines below.
    // That condition is now satisfied -- the figure appears NOWHERE in the file
    // (0 occurrences of "+7.0" and of "PROVEN ABSENT" at HEAD) -- so the filter
    // is deleted rather than left as a suppression nothing tracks. The document
    // is now scanned like every other. Its RAW-RECORD exemption in the
    // hit-RATE check below is a different argument and correctly survives:
    // raw readings are exempt, a verdict never is.
    ;

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

  const sources = shippedPaths()
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
      `on a box where a null A/B -- same binary, same prompt, TRUE DELTA ZERO BY CONSTRUCTION -- ` +
      `swung +52.30 % / -40.17 % across six pairs (perf-baseline.md §8.1).\n` +
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
// A table row carrying a counter reading: `| … | 15 / 16 |`. Deliberately
// matches the SHAPE of a hits/lookups pair, not any specific figure, so the
// check keeps working after the numbers are re-measured.
const RATE_IN_ROW = /\|\s*\d+\s*\/\s*\d+\s*\|/;

test('no counter reading is labelled as a baseline when it is a mid-run snapshot', () => {
  // THE DEFECT THIS PINS, AND WHY IT IS DELIBERATELY TINY.
  //
  // The README's counter table read `| before | 15 / 16 |`. Every number in it
  // was correct and the argument it supports -- +12 hits for twelve requests,
  // six of which shared nothing -- is airtight. The single word `before` was
  // the defect: it claims PROVENANCE. It reads as "the state before the
  // experiment began", i.e. a baseline, when 15/16 is a mid-run snapshot off
  // an already-warm counter. @376a0297's finding is that the only two false
  // statements in this whole class were a STABILITY claim and a PROVENANCE
  // claim -- never a rate.
  //
  // SCOPED TO README.md, AND ROW LABELS ONLY, ON PURPOSE. The broad form of
  // this check ("every quoted rate must carry its counter readings") would
  // redden four files that use these figures correctly as EVIDENCE -- both
  // tripwires, telemetry-provenance.js, and my own signed retraction. A guard
  // that fires on correct files is one somebody deletes by Friday, and they
  // take the real check with it. The tell is grammar, not arithmetic, and
  // grammar is not mechanisable -- but THIS much is: a one-word row label
  // asserting priority over a counter reading.
  const rows = shipped('README.md')
    .split('\n')
    .filter((l) => /^\|/.test(l) && RATE_IN_ROW.test(l));

  assert.ok(rows.length > 0, 'no counter-reading table rows found in README.md — this check would pass vacuously; the hits/lookups table is expected here');

  const offenders = rows.filter((l) => /^\|\s*(before|baseline|start(?:ing)?|initial|at rest)\s*\|/i.test(l));
  assert.deepEqual(
    offenders,
    [],
    'a counter-reading row is labelled as if it were a baseline:\n' +
      offenders.map((o) => `  ${o.trim()}`).join('\n') +
      '\nThese readings come off a cumulative counter on a warm server, so the ' +
      'first row is not a starting state -- it is wherever the counter happened ' +
      'to be. Labelling it `before` turns a correct delta into a false claim ' +
      'about provenance, which is the one thing the surrounding section exists ' +
      'to disprove. Say what the row IS ("immediately before these two probes, ' +
      'counter already warm"), not when it happened.',
  );
});

test('no document attributes a hit RATE to the twelve-request block', () => {
  const docs = shippedPaths()
    .filter((f) => f.endsWith('.md'))
    // The raw measurement records legitimately carry every reading, including
    // the mid-block ones. They are where the arithmetic above was RECOVERED
    // from; banning the figures there would delete the evidence.
    .filter((f) => !/^(perf-baseline|demo-spec|prefix-cache-verification)\.md$/.test(f));
  // A SECOND `design/` FILTER USED TO SIT HERE WITH NO REASON ATTACHED. The
  // comment above justifies the raw-record names and stops; the design/ line
  // was appended silently and almost certainly copied from the sibling check,
  // whose exemption was real, specific, and about a DIFFERENT property
  // (unqualified figures, not hit rates). Measured before deleting: design/
  // contains 0 paragraphs that mention the twelve-request block at all, so
  // this filter never suppressed anything -- it only made the corpus look
  // deliberately narrowed. An exemption nobody can date and nobody can defend
  // is indistinguishable from one that is load-bearing, which is exactly how
  // a suppression survives review.

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
  // HISTORY, KEPT BECAUSE THE DEFECT CLASS IS THE POINT AND THE EXEMPTION IS NOW GONE:
  //
  //   This block used to exempt one paragraph of @fc8b5d97's QA evidence
  //   document by its exact wording. THAT SENTENCE WAS VERBATIM THE REMEDIATION
  //   STRING THIS VERY FILE USED TO PRINT. A reader who tripped my check was
  //   handed "twelve hits and a 0.9375 rate" as the CORRECT replacement text and
  //   pasted it faithfully. The guard did not merely fail to catch the defect --
  //   IT AUTHORED IT, and in the one register nobody proofreads, because
  //   remediation text is read only by someone who has already been told they
  //   are wrong.
  //
  // The owner fixed their line at 603d2b68 ("quote the +12 delta, not a
  // cumulative-counter rate"), so the exemption was removed here. MEASURED
  // BEFORE DELETING: the exempted wording has ZERO occurrences in that document
  // at HEAD.
  //
  // ⛔ AND THE RESIDUAL DEFECT WAS MINE, NOT THEIRS. The exemption was a bare
  // `continue` with NO staleness assertion, while this file's DEFERRED block
  // insists an exemption that cannot expire is a suppression. So it would have
  // sat here silently exempting a paragraph that no longer existed, for as long
  // as the file lived, and nothing would ever have gone red to say so. An
  // exemption whose removal condition lives only in a COMMENT is not expiring --
  // it is waiting for someone to happen to read it.
  for (const doc of docs) {
    for (const para of shipped(doc).split(/\n\s*\n/)) {
      if (!BLOCK.test(para)) continue;
      inspected += 1;
      if (WITHDRAWN.test(para)) continue;
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

// ---------------------------------------------------------------------------
// THE README QUOTES THE SERVER'S OWN LOG TO IDENTIFY THE EXECUTION PATH.
//
// A performance figure without its execution path is not a measurement: this
// runtime has two decode paths, chooses one at startup, and the same binary on
// the same machine produces non-comparable numbers depending on which it took.
// So the README tells the reader how to check which one ran -- and it does that
// by quoting two `tracing::info!` strings verbatim.
//
// A QUOTED LOG LINE IS A CITATION INTO A DIFFERENT LANGUAGE'S STRING LITERAL,
// which is the least likely thing in this repo to be updated in step with the
// docs: a Rust refactor that reworded the message would leave this README
// telling readers to grep for text the server has stopped emitting, and every
// markdown checker we own would stay green because the sentence around it is
// still perfectly well-formed English.
//
// This is also the claim the LEAD had inverted when ordering the section
// written -- the instruction was that the benchmark ran on the PER-REQUEST
// FALLBACK "confirmed from the server's own log". The lab record says the
// opposite in four places, and its harness protocol (perf-baseline.md) refuses
// to measure at all unless the ENABLED line is present. Hence a test: the
// distinction is load-bearing enough that an order was issued backwards on it.
test('every driver log line the README quotes still exists in the server source', () => {
  const readme = shipped('README.md');
  const driver = execFileSync(
    'git',
    ['show', `${SHIPPING_REF}:crates/onnx-genai-server/src/driver.rs`],
    { cwd: HERE, maxBuffer: 64 * 1024 * 1024 },
  ).toString();

  // The messages as Rust emits them. `max_batch` is a structured field rather
  // than part of the literal, so the quoted log line carries a `max_batch=4`
  // suffix that will NOT be found in source -- match the literal only.
  const quoted = [...readme.matchAll(/continuous batch driver [a-z-]+(?:; using per-request engine path)?/g)]
    .map((m) => m[0]);

  assert.ok(
    quoted.length >= 2,
    `found ${quoted.length} quoted driver log line(s) in the README; the section ` +
      `that names the execution path has been reworded or removed, and this ` +
      `check is inspecting nothing`,
  );

  const missing = [...new Set(quoted)].filter((line) => !driver.includes(line));
  assert.deepEqual(
    missing,
    [],
    `the README tells readers to look for ${missing.length} log line(s) that ` +
      `driver.rs no longer emits:\n  ${missing.join('\n  ')}\n` +
      `Either the message was reworded in Rust and the README was not updated, ` +
      `or the branch was removed. A reader following this instruction would ` +
      `grep for text that never appears and conclude the demo is broken.`,
  );

  // The capacity-1 claim is the part a reader can check WITHOUT the log, so it
  // has to stay true independently of the message strings.
  assert.ok(
    /published_capacity\s*=\s*if continuous_batch_supported/.test(driver),
    'the README states that the per-request path publishes a batch capacity of ' +
      '1 rather than max_batch, and offers that field as the execution-path ' +
      'witness. The branch that makes it true is gone from driver.rs.',
  );
});

// ---------------------------------------------------------------------------
// THE README'S PER-PANE ROW COUNTS ARE COPIED FROM perf-baseline.md §11.
//
// A number maintained in two places is a number that will disagree with itself.
// The house rule is COMPUTED FROM THE ARTIFACT OR DELETED, and these two counts
// cannot be computed at doc-build time -- they came off a live server. So the
// next best thing: pin them to the record they were copied from, and fail when
// the record moves.
//
// These particular counts matter more than most, because they are what licenses
// the README's warning that the two panes are NOT COMPARABLE. If the record is
// ever revised to show both panes batching, the warning becomes false and the
// demo's central side-by-side becomes legitimate again -- a change nobody would
// think to propagate into a paragraph phrased as a caveat.
test('the README per-pane row counts match the measurement record', () => {
  const readme = shipped('README.md');
  const record = shipped('perf-baseline.md');

  // The reproduce block is the most stable anchor in §11: it is executable, so
  // it cannot drift from the table above it without someone noticing.
  const expectations = [...record.matchAll(/qa-batch-width\.py\s+\d+\s+(\S+)\s*#\s*expect peak in_flight (\d+)/g)]
    .map((m) => ({ pane: m[1], peak: m[2] }));

  assert.equal(
    expectations.length,
    2,
    `expected two reproduce lines in perf-baseline.md §11, found ${expectations.length}. ` +
      `The record's shape changed and the README's row counts are no longer pinned ` +
      `to anything.`,
  );

  const scatter = expectations.find((e) => /scatter/.test(e.pane));
  const dynamic = expectations.find((e) => /dynamic/.test(e.pane));
  assert.ok(scatter && dynamic, 'could not identify the scatter and dynamic arms in the record');

  // The README states these as "peaks at N rows" / "at N". Assert the DIGITS the
  // record expects actually appear in the paragraph that draws the conclusion,
  // rather than anywhere in the file -- a match elsewhere would prove nothing.
  const para = readme
    .split(/\n\s*\n/)
    .find((p) => /batch_in_flight/.test(p) && /peaks at/.test(p));

  assert.ok(
    para,
    'the README paragraph reporting sampled batch_in_flight peaks is gone; the ' +
      'confound warning below it is now unsupported by any stated measurement',
  );

  for (const [name, e] of [['scatter', scatter], ['dynamic', dynamic]]) {
    assert.ok(
      new RegExp(`\\*\\*${e.peak}(?: rows?)?\\*\\*`).test(para),
      `the record expects the ${name} pane to peak at ${e.peak} rows, and the ` +
        `README paragraph does not state that number:\n  ${para.replace(/\s+/g, ' ').slice(0, 200)}`,
    );
  }

  // The confound warning is the reader-facing consequence of those two counts.
  // It must not survive them silently.
  assert.ok(
    scatter.peak !== dynamic.peak,
    'the record now shows both panes running the same width, so the README\'s ' +
      '"any comparison across the panes is confounded" warning is FALSE and must ' +
      'be withdrawn. A caveat that outlives its cause is a claim, not a caution.',
  );
});

// ---------------------------------------------------------------------------
// The README now states a CAUSE, and a cause is a claim about source code that
// can be refactored out from under the prose without touching the prose. These
// three guards exist because the paragraph they protect corrects an UNDERclaim
// -- it says more than the previous version, not less -- and an underclaim
// corrected in the direction of confidence is the one nobody re-audits.
//
// Note what is deliberately NOT pinned: line numbers. The arms move; the
// semantics are what the README describes.

// Rust source lives above this directory, so it is addressed from the repo
// root. `<ref>:./x` resolves relative to cwd and would need `../../`; the
// root-anchored form says what it means.
function shippedFromRoot(path) {
  return execFileSync('git', ['show', `${SHIPPING_REF}:${path}`], {
    cwd: HERE,
    maxBuffer: 64 * 1024 * 1024,
  }).toString();
}

const BATCHED_RS = 'crates/onnx-genai-engine/src/batched.rs';
const DECODE_META_RS = 'crates/onnx-genai-engine/src/decode/metadata.rs';

// Slice out `continuous_batch_manager`'s body rather than scanning the whole
// file: `ModelDecodePath::StaticCache` appears in a dozen unrelated matches,
// and a whole-file grep would score GREEN off any one of them while the arm the
// README describes had been deleted.
function continuousBatchManagerBody() {
  const src = shippedFromRoot(BATCHED_RS);
  const start = src.indexOf('pub fn continuous_batch_manager(');
  assert.ok(
    start !== -1,
    `${BATCHED_RS} no longer defines \`continuous_batch_manager\`. The README ` +
      'names it as the decision point for batch capability; if it has been ' +
      'renamed or removed, that section describes a function that is not there.',
  );
  const next = src.indexOf('\n    pub fn ', start + 1);
  const body = src.slice(start, next === -1 ? src.length : next);
  // Vacuity floor. A zero-length or truncated slice makes every `includes`
  // below trivially false, which would read as a loud failure -- but a slice
  // that is merely SHORT could pass a negative check by accident.
  assert.ok(
    body.length > 400,
    `extracted \`continuous_batch_manager\` body is only ${body.length} bytes; ` +
      'the slicing heuristic has drifted and these assertions are not ' +
      'inspecting what they claim to inspect.',
  );
  return body;
}

test('continuous batching still accepts exactly the two decode paths the README enumerates', () => {
  const body = continuousBatchManagerBody();

  // The README's load-bearing word is "sufficient, not necessary": a static
  // cache is ONE of two accepting shapes. If the shared-buffer arm is ever
  // removed, the README's correction of the crew's "only static_cache batches"
  // reading becomes the wrong one, and it must go red HERE rather than in a
  // reader's head.
  assert.ok(
    /ModelDecodePath::StaticCache\s*\{/.test(body),
    'the static-cache accepting arm is gone from `continuous_batch_manager`. ' +
      'README: "`ModelDecodePath::StaticCache { .. }` -- batches."',
  );
  // ORDER-INDEPENDENT BY CONSTRUCTION. An earlier form of this pattern required
  // `shared_buffer: true` to be the FIRST field after the brace. Rust does not
  // care about field order in a match pattern and rustfmt may insert a comment,
  // so a purely cosmetic edit made this assertion fail while the arm was fully
  // present -- and the failure message then told the author, with total
  // confidence, that the capability had been removed. MUTATION-PROVEN over six
  // shapes: as-shipped, reordered, comment-inserted and single-line all match;
  // a genuine deletion does not, and neither does an arm that offers only
  // `shared_buffer: false`, which is the anti-vacuity arm proving this pattern
  // did not simply become permissive.
  assert.ok(
    /ModelDecodePath::PastPresent\s*\{[^}]*shared_buffer:\s*true/.test(body),
    'the SHARED-BUFFER accepting arm no longer MATCHES in ' +
      '`continuous_batch_manager`. This is the arm that makes the README say a ' +
      'static cache is SUFFICIENT BUT NOT NECESSARY.\n\n' +
      'FIRST establish WHICH of these happened -- this check cannot tell them ' +
      'apart and must not be read as if it could:\n' +
      '  (a) the arm was genuinely removed, or\n' +
      '  (b) the arm is still there and only its FORMATTING changed.\n\n' +
      'If (b), fix this pattern and change NO prose. If (a), the README ' +
      'paragraph that refutes the static-cache predicate has to be re-derived ' +
      'from `batched.rs` -- state the accepting arms this function actually ' +
      'has, and name them. Deliberately NOT supplying a replacement sentence ' +
      'here: a sentence pasted from a failing check asserts a capability that ' +
      'nobody re-measured, and this message cannot see which case it is in.',
  );

  // And the refusal. The README quotes this string verbatim inside a fenced
  // block; a reader will search their own logs for it.
  const BAIL =
    'continuous batching requires a STATIC-CACHE or shared-buffer past/present model';
  assert.ok(
    body.includes(BAIL),
    `the README quotes this refusal verbatim and it is no longer in ${BATCHED_RS}:\n` +
      `  ${BAIL}\n` +
      'A quoted error message is the one piece of documentation a reader ' +
      'matches CHARACTER BY CHARACTER against their own terminal, so a stale ' +
      'one fails them at their least sceptical moment.',
  );
  assert.ok(
    shipped('README.md').includes(BAIL),
    'the README no longer quotes the refusal message that this section is ' +
      'built around.',
  );
});

test('the refused arms are still exactly the two the README says are refused', () => {
  const body = continuousBatchManagerBody();

  // WHY THIS NO LONGER MATCHES A SINGLE COMBINED ARM.
  //
  // This assertion used to be one regex requiring the literal source shape
  // `PastPresent { .. } | Legacy => { bail!`. That went RED on a refactor that
  // split the combined arm in two -- and the claim it exists to protect was
  // never touched. Still exactly two refusing paths, still the same two.
  //
  // A guard that pins the SYNTAX of a claim rather than the CLAIM fails in the
  // most corrosive way available: it reds on a correct change. A red nobody
  // can act on is not caution, it is an invitation to delete the guard, and
  // whoever deletes it also deletes the check on the thing that mattered.
  //
  // So: assert the SET of refusing arms. A third arm starting to bail still
  // fires this. Merging or splitting the two that already do, does not.
  const REFUSING_ARMS = ['ModelDecodePath::PastPresent { .. }', 'ModelDecodePath::Legacy'];
  const bailingArms = [...body.matchAll(/(ModelDecodePath::[A-Za-z]+(?:\s*\{[^}]*\})?)[^=]*=>\s*\{([\s\S]*?)\n(\s{12}\})/g)]
    .filter(([, , arm]) => /anyhow::bail!/.test(arm))
    .map(([, head]) => head.replace(/\s+/g, ' ').trim());

  assert.deepEqual(
    bailingArms,
    REFUSING_ARMS,
    'the set of match arms that REFUSE continuous batching has changed.\n' +
      `  now refusing: ${JSON.stringify(bailingArms)}\n` +
      `  README says : ${JSON.stringify(REFUSING_ARMS)}\n` +
      'The README narrows the unresolved question to exactly two ways ("which ' +
      'of those two it lands on has not been observed"). If a third arm bails, ' +
      'that stated width claims MORE certainty than we have. If one stops ' +
      'bailing, it claims less.',
  );

  // AND THE PART THE SPLIT ACTUALLY CHANGED, WHICH THE ARM-SET CANNOT SEE.
  //
  // The two arms now carry DIFFERENT messages, and that was the entire point
  // of splitting them -- batched.rs says so in a comment: collapsing them
  // "tells an operator to change the model when the real fix may be an
  // environment variable". The README quotes ONE of the two verbatim in a
  // block a reader matches against their own terminal. An operator on the
  // other path searches for that string and does not find it.
  //
  // So both messages must be quoted, or the README's fenced refusal is a
  // 50% chance of sending someone to replace a model they did not need to.
  const SHARED_BUFFER_BAIL = 'continuous batching requires a shared KV buffer';
  assert.ok(
    body.includes(SHARED_BUFFER_BAIL),
    `the PastPresent arm no longer emits its own distinct refusal (${SHARED_BUFFER_BAIL}). ` +
      'If the two arms were re-merged into one message, the README section that ' +
      'documents them separately is now overspecified -- fix them together.',
  );
  assert.ok(
    shipped('README.md').includes(SHARED_BUFFER_BAIL),
    'batched.rs emits TWO distinct refusal messages and the README quotes only ' +
      'one. The unquoted one is the shared-KV-buffer case, whose fix may be a ' +
      'launch flag rather than a different model -- so a reader who hits it and ' +
      'searches for the quoted string finds nothing, and the string they DO find ' +
      'tells them to change the model. Quote both.',
  );
});

test('shared-buffer batching is still gated on an execution-provider capability, not on the model file', () => {
  const src = shippedFromRoot(DECODE_META_RS);
  // This is the single most transferable sentence in that README section --
  // batch capability is a property of (model, EP, environment), not of a
  // directory -- and it is TRUE ONLY BECAUSE this predicate is consulted here.
  assert.ok(
    src.includes('supports_fixed_capacity_present_binding()'),
    `${DECODE_META_RS} no longer consults ` +
      '`supports_fixed_capacity_present_binding()`. The README uses this exact ' +
      'call to justify its central claim that batch capability CANNOT be ' +
      'predicted by reading `inference_metadata.yaml`, because it depends on ' +
      'the execution provider and an env opt-in. If the gate is gone, the ' +
      'capability may now be a pure function of the model directory -- which ' +
      'would make a metadata-reading check CORRECT and the README wrong.',
  );
  assert.ok(
    /shared_buffer:\s*true/.test(src) && src.includes('DecodeKvMode::SharedBuffer'),
    `${DECODE_META_RS} no longer resolves a shared-buffer decode path, so the ` +
      'README\'s "sufficient, not necessary" argument has no second path to ' +
      'point at.',
  );
});

// WHY THIS GUARD EXISTS, AND WHY IT IS A *CORPUS* WIDENING RATHER THAN A NEW
// MECHANISM.
//
// Everything above pins the two-qualifying-classes claim in `README.md`, and it
// pins it well -- including the sentence that batch capability is a property of
// (model, EP, launch) and NOT of a model directory. All of it was green while
// `.github/skills/build-static-cache-model/SKILL.md` said the opposite in its
// first paragraph: that batching engages "only for static-cache models", gated
// by a `.is_ok()` call that `1e1b2a82` had already removed, cited to a line
// range holding an unrelated function.
//
// The assertions were never wrong. The FILE SET was. This checker had never
// opened `.github/skills`, so the claim was pinned in the document humans read
// and unpinned in the documents AGENTS read -- which are instructions, acted on
// without a second reader. A skill file is the highest-leverage place in the
// repo to be wrong.
//
// Scope is deliberately narrow: this asserts only the mutual-exclusivity claim,
// because that is the one the Rust source contradicts outright. It is not a
// general prose linter.
function shippedSkillDocs() {
  // `:(top)` and `--full-name` are both load-bearing, and their absence is why
  // the anti-vacuity floor below exists -- each was wrong on a separate run of
  // this guard. `git ls-tree <ref> -- <path>` resolves the pathspec relative to
  // CWD and prints names relative to CWD, and this file runs from
  // examples/serving-dashboard: without `:(top)` the pathspec silently matched
  // `examples/serving-dashboard/.github/skills` and returned NOTHING, so every
  // check below would have passed over an empty corpus. This is the same
  // root-vs-cwd trap `shippedFromRoot` documents for `git show`.
  const paths = execFileSync(
    'git',
    [
      'ls-tree',
      '-r',
      '--full-name',
      '--name-only',
      SHIPPING_REF,
      '--',
      ':(top).github/skills',
      ':(top).agents/skills',
    ],
    { cwd: HERE, maxBuffer: 16 * 1024 * 1024 },
  )
    .toString()
    .split('\n')
    .filter((p) => p.endsWith('SKILL.md'));
  return paths.map((path) => ({ path, body: shippedFromRoot(path) }));
}

test('no agent-facing skill doc claims static-cache is the ONLY way to get batching', () => {
  const docs = shippedSkillDocs();

  // ANTI-VACUITY FLOOR. Without this, a rename of the skill trees -- or the
  // `ls-tree` pathspec silently matching nothing, which is EXACTLY what it did
  // on the first run of this guard -- turns every assertion below into a loop
  // over an empty array and reports GREEN. An empty corpus satisfies a
  // universal claim about its members, and that is the defect this guard was
  // written in response to.
  assert.ok(
    docs.length >= 10,
    `expected at least 10 SKILL.md files under the skill trees at ${SHIPPING_REF}, ` +
      `found ${docs.length}. A short corpus makes the checks below vacuously ` +
      'true, which is how the original defect survived.',
  );

  // Not an inventory of expected files -- a requirement that the corpus contain
  // the ONE doc whose entire subject is this claim. A guard about static-cache
  // claims that does not read the static-cache skill is decorative.
  const SUBJECT_DOC = '.github/skills/build-static-cache-model/SKILL.md';
  assert.ok(
    docs.some((d) => d.path === SUBJECT_DOC),
    `${SUBJECT_DOC} is not in this guard's corpus. That is the document whose ` +
      'subject IS the batching gate, and it is where the false claim was found.',
  );

  // The claim is false because of THIS arm, so anchor to it rather than to
  // prose: if `shared_buffer: true` ever stops qualifying, static-cache really
  // does become the only route and these docs become correct again.
  const managerBody = continuousBatchManagerBody();
  assert.ok(
    /shared_buffer:\s*true/.test(managerBody),
    '`continuous_batch_manager` no longer has a qualifying `shared_buffer: true` ' +
      'arm. If static-cache is now genuinely the ONLY path to continuous ' +
      'batching, this guard is obsolete and the skill docs it polices were ' +
      'right all along -- delete it deliberately rather than loosening it.',
  );

  // Each pattern states what is wrong with it, because a skill doc is read by
  // someone who is already lost.
  const FALSE_EXCLUSIVITY = [
    {
      re: /engages\s+\*\*only for static-cache models\*\*|only for static-cache models/i,
      why: 'a shared-buffer past/present model with a known max_len also qualifies',
    },
    {
      re: /\bOnly\s+these\s+`?-scatter`?\s+static-cache models engage continuous batching/i,
      why: 'a shared-buffer past/present model also engages it',
    },
    {
      re: /continuous_batch_manager\([^)]*\)\.is_ok\(\)/,
      why: '`1e1b2a82` replaced `.is_ok()` with a match that keeps the reason in ' +
        '`BatchDriver::PerRequest { reason }`; documenting `.is_ok()` describes a ' +
        'mechanism that no longer exists and implies the reason is unavailable',
    },
  ];

  const offenders = [];
  for (const { path, body } of docs) {
    for (const { re, why } of FALSE_EXCLUSIVITY) {
      const hit = body.match(re);
      if (hit) offenders.push(`  ${path}: ${JSON.stringify(hit[0])}\n      -> ${why}`);
    }
  }

  assert.deepEqual(
    offenders,
    [],
    'an agent-facing skill doc states that continuous batching requires a ' +
      'static-cache model, or describes a gate that was removed:\n' +
      `${offenders.join('\n')}\n` +
      'batched.rs refuses with "continuous batching requires a STATIC-CACHE or ' +
      'shared-buffer past/present model" -- TWO classes qualify, and which one ' +
      'applies depends on the execution provider and launch flags, not on the ' +
      'model directory. A skill doc is executed as instruction, so this sends an ' +
      'agent to rebuild a model when the real fix may be an environment variable.',
  );
});


// behaviour it removed. The prose was false the moment it was committed, the
// suite was green throughout, and the existing checker -- which pins the
// quoted log MESSAGE -- stayed green, because the message did not change. Only
// the LEVEL changed, and a field appeared.
//
// PINNING A QUOTED STRING DOES NOT PIN THE SENTENCE YOU WRAPPED AROUND IT.
test('the README describes the CURRENT batch-decision mechanism, not the one it replaced', () => {
  const driver = shippedFromRoot('crates/onnx-genai-server/src/driver.rs');
  const readme = shipped('README.md');

  // 1. The decision must still keep the reason. If anyone regresses to
  //    `.is_ok()`, the README's "the log tells you why" becomes false.
  //    Anchored to the CONSTRUCTION that retains the error, not to the word
  //    `match`. `match engine.continuous_batch_manager(max_batch)` appears
  //    TWICE in driver.rs, so a regression at the startup site is fully
  //    concealed by the surviving second call -- a mutation of the real site
  //    left a `match`-shaped assertion GREEN. Pin the semantics: the Err arm
  //    must still carry the reason into the driver.
  assert.ok(
    /Err\(err\)\s*=>\s*BatchDriver::PerRequest\s*\{\s*\n?\s*reason:/.test(driver),
    'the batch decision no longer carries the error into ' +
      '`BatchDriver::PerRequest { reason }`. If it has regressed to ' +
      '`.is_ok()`, the reason is discarded again and the README now ' +
      'OVERSTATES what the log can tell an operator.',
  );
  assert.ok(
    !/continuous_batch_manager\(max_batch\)\.is_ok\(\)/.test(driver),
    'driver.rs is back to `.is_ok()`. The README says the fallback is logged ' +
      'WITH a reason; it would no longer be.',
  );

  // 2. The fallback must still be a WARN carrying a `reason` field -- the two
  //    details the README now prints in its sample output.
  assert.ok(
    /tracing::warn!\(\s*\n?\s*reason = %/.test(driver),
    'the fallback is no longer logged at WARN with a `reason =` field. The ' +
      'README prints a sample line reading `WARN ... reason=<why>`; an ' +
      'operator greps for exactly that.',
  );
  assert.ok(
    readme.includes('WARN onnx_genai_server::driver: continuous batch driver disabled'),
    'the README no longer shows the fallback line at WARN. It was shipped as ' +
      'INFO for one commit after the server started emitting WARN -- a reader ' +
      'filtering their logs at INFO would have seen it, then not found it.',
  );

  // 3. The vacuity floor for THIS test: both refused arms must still share one
  //    `bail!`, because that shared message is precisely why the README says
  //    the remaining ambiguity is two-way and unclosable by better logging.
  const bailCount = (
    shippedFromRoot(BATCHED_RS)
      .slice(shippedFromRoot(BATCHED_RS).indexOf('pub fn continuous_batch_manager('))
      .match(/anyhow::bail!/g) || []
  ).length;
  assert.ok(
    bailCount >= 1,
    'no `bail!` remains in `continuous_batch_manager`; the README\'s account ' +
      'of WHY the refusal reason cannot discriminate the two refused decode ' +
      'paths no longer has a mechanism behind it.',
  );
});

// The README's null result rests on a SENSITIVITY argument: the effect we
// looked for (~90 %) is larger than the floor we could resolve, so failing to
// see it is evidence of absence rather than absence of evidence. That argument
// is a RATIO, and a ratio silently decays when either side is restated
// somewhere else. It already did: the README carried "9x the floor" computed
// against a floor its own source had retracted, and the true margin was 1.7x.
//
// Nothing caught that, because the existing checker asks whether a noise floor
// is PRESENT, never whether it is the CURRENT one. Presence is not currency.
test('the README noise floor is the one perf-baseline currently measures, and the margin arithmetic holds', () => {
  const readme = shipped('README.md');
  const baseline = shipped('perf-baseline.md');

  // Anchor on §8.1's null A/B -- the only floor evidence on a KNOWN-ZERO truth.
  // Read the extremes out of the source document rather than hardcoding them
  // here, so this test cannot become the next stale publisher of a number.
  const lo = /−?-?40\.17\s*%/;
  const hi = /\+?52\.30\s*%/;
  assert.ok(
    hi.test(baseline) && lo.test(baseline),
    'perf-baseline.md no longer reports the §8.1 null-A/B extremes ' +
      '(+52.30 % / -40.17 %). The README cites them as the measured noise ' +
      'floor; if they have been restated, the README is quoting a figure its ' +
      'own source no longer carries.',
  );
  assert.ok(
    hi.test(readme),
    'README.md no longer states the measured noise floor (+52.30 %). Its null ' +
      'prefix result is only admissible WITH a floor: "we did not observe it" ' +
      'becomes "it is not there" only once the instrument is shown capable of ' +
      'seeing the alternative.',
  );

  // The margin claim must be arithmetically consistent with the floor. 90/52.30
  // = 1.72. Anything asserting a substantially larger multiple is the old,
  // flattering number returning.
  // `\*{0,2}` not `\*\*?`: the bold marker opens BEFORE "~90", so there is no
  // asterisk between the arrow and the digits. `\*\*?` requires at least one
  // and matched nothing -- a guard that would have thrown on the very text it
  // was written to accept. Probed against the real string before trusting it.
  const margin = readme.match(/~?90\s*%\s*→\s*\*{0,2}([\d.]+)×\s*the floor/);
  assert.ok(
    margin,
    'the sensitivity table no longer states the margin as "<N>x the floor". ' +
      'That multiple is the entire sensitivity argument; without it the table ' +
      'lists two magnitudes and leaves the reader to divide.',
  );
  const stated = Number(margin[1]);
  const truth = 90 / 52.3;
  assert.ok(
    Math.abs(stated - truth) < 0.25,
    `the README claims the predicted effect is ${stated}x the noise floor, but ` +
      `90 / 52.30 = ${truth.toFixed(2)}x. This is how the defect appeared the ` +
      `first time: the floor was corrected upward ~5x somewhere else and the ` +
      `ratio was left behind, overstating our own resolving power by 5x -- IN ` +
      `THE FLATTERING DIRECTION, which is the one nobody re-derives.`,
  );

  // And the conclusion must still be stated as surviving. If the floor ever
  // rises past the effect, this suite must not let the page keep claiming a
  // sensitivity argument it no longer has.
  assert.ok(
    truth > 1,
    'the measured noise floor now EXCEEDS the effect a working prefix cache ' +
      'would produce. The README\'s null result is no longer supported by a ' +
      'sensitivity argument and must be restated as simply unmeasured.',
  );
});

// ---------------------------------------------------------------------------
// A GUARD SCOPED BY FILE EXTENSION IS A GUARD SCOPED BY GUESSWORK.
//
// Every noise-floor check above this line filters the tracked tree with
// `.endsWith('.md')`. That is a guess about where a claim can live, and it was
// wrong. When this test was written the retracted 9.8 % figure was stated as
// live fact in FIVE tracked non-markdown files -- two inside TEST ASSERTION
// MESSAGES (the remediation text handed to whoever is fixing a failure) and
// one in `scenario-origins.js`, which SHIPS TO THE BROWSER. The markdown-only
// guards were green over all five, and could not have been anything else.
//
// SCOPE, STATED SO IT CANNOT BE OVERSOLD: this covers exactly the complement
// of what the guards above cover -- tracked NON-markdown files. The markdown
// corpus already has three tuned checks with their own exemptions; a second
// opinion over the same prose would double-report and get reworded away. This
// is the blind spot, not a replacement.
//
// A first draft of this check scanned EVERYTHING and reported fifteen regions,
// of which most were correct retractions. A detector that fires on everything
// is indistinguishable from one that works, so it was cut down rather than
// tuned -- the corpus was the defect, not the threshold.
test('the retracted 9.8 % floor is not stated as live fact outside markdown', () => {
  // Prose-carrying source. Deliberately NOT an extension guess about where a
  // claim lives -- it is the set of tracked text files the .md guards skip.
  const RETRACTED = /(?<![\d.])9\.8\s*%/;
  const FLOOR_FRAMING = /noise floor|swung|swings|ambient load|byte-identical binary/i;
  // Anchored on the NAMED SECTION that performs the retraction, never on a
  // withdrawal vocabulary. A sibling prose guard in this repo exempted any line
  // containing `measured`, `no`, or `not`; appending "and measured" to a false
  // claim flipped it GREEN, so the strictly stronger lie was the one that
  // passed. An exemption clause is a hole shaped like the words honest people
  // use. `§6f` cannot be worn by accident, and a claim that does wear it has
  // cited the document that refutes it.
  //
  // MEASURED, NOT ASSUMED: the first draft of this line also accepted
  // /retract|withdrew|withdrawn/. `scenario-origins.js` withdraws the 7 %
  // TIMING ARM in the same breath as it cites the stale 9.8 % FLOOR that
  // supports the withdrawal -- so a withdrawal of one claim bought a free
  // exemption for a different, still-live one. The guard demonstrated the
  // vocabulary hole on itself within one run of being written.
  const NAMES_THE_RETRACTION = /§ ?6f|older 9\.8/i;

  // A guard must quote what it forbids. Excluded BY PATH -- checkable -- rather
  // than by a pattern that would also excuse anyone who looked like a guard.
  const SELF = 'check-perf-claims.test.js';

  // Expiring, named, owner-attributed. Under a freeze, reddening another
  // agent's file is worse than a deferral that reports itself. If a listed file
  // stops offending, this test FAILS and tells me to delete the row: an
  // exemption that cannot expire is a suppression, and the direction nobody
  // reports is the one where the gap has quietly closed.
  // EMPTY, AND IT EMPTIED ITSELF. The one entry -- `scenario-origins.js`, shipped
  // browser code owned by @bb2ee824, deferred because reddening another agent's
  // file under freeze is worse than a deferral that reports itself -- was removed
  // when its owner landed `cd22dcb7` ("stop citing a withdrawn load-drift figure
  // in shipped browser code") at 07:51:27. The retracted figure went to 0
  // occurrences in that file and THIS TEST WENT RED WITHIN ONE MINUTE, naming the
  // row to delete. That is the whole point of the design: the exemption I retired
  // earlier tonight recorded its removal condition in a COMMENT and sat dead for
  // hours; this one recorded it in an ASSERTION and expired the moment it was
  // obsolete. An exemption that cannot expire is a suppression.
  //
  // Note the ORDERING, because it is the safe half of @12e42da8's new rule: the
  // CONTENT fix landed first and the guard relaxation follows. A stale exemption
  // only ever REDDENS the branch -- it cannot green a defect. The dangerous order
  // is the reverse, a guard landing ahead of the prose that satisfies it.
  //
  // With no rows, the stale-row check below is vacuously true. That is the correct
  // terminal state and not a hole: the assertion that carries the real weight is
  // the offender scan, whose non-vacuity is pinned by named corpus members below.
  const DEFERRED = Object.freeze({});

  const corpus = shippedPaths()
    .filter(Boolean)
    .filter((f) => !f.endsWith('.md'))
    .filter((f) => f !== SELF)
    .filter((f) => /\.(js|mjs|cjs|sh|css|html|json|py)$/i.test(f));

  // NON-VACUITY BY NAMED MEMBER, not by count. A count is satisfied by twenty
  // files of the wrong kind; this proves the corpus reaches the two file shapes
  // the `.md` filter structurally excluded -- a shipped module and a test.
  assert.ok(
    corpus.includes('scenario-origins.js') && corpus.includes('dashboard/honesty.test.js'),
    `the corpus (${corpus.length} files) no longer reaches shipped .js and test .js — ` +
      'the extension blind spot has returned',
  );

  const offenders = [];
  const deferredSeen = new Set();

  for (const file of corpus) {
    let text;
    try {
      text = shipped(file);
    } catch {
      continue;
    }
    if (!RETRACTED.test(text)) continue;

    // Scoped by REGION, never by document: a file may retract the figure in one
    // place and must not assert it in another.
    for (const region of text.split(/\n\s*\n/)) {
      if (!RETRACTED.test(region) || !FLOOR_FRAMING.test(region)) continue;
      if (NAMES_THE_RETRACTION.test(region)) continue;

      if (Object.hasOwn(DEFERRED, file)) {
        deferredSeen.add(file);
        continue;
      }
      const hit = region.split('\n').find((l) => RETRACTED.test(l)) ?? region;
      offenders.push(`${file}: ${hit.trim().slice(0, 110)}`);
    }
  }

  const stale = Object.keys(DEFERRED).filter((f) => !deferredSeen.has(f));
  assert.deepEqual(
    stale,
    [],
    `DEFERRED lists ${stale.join(', ')}, which no longer states the retracted figure as ` +
      'a live floor. Delete the entry rather than leaving a suppression nothing tracks.',
  );

  assert.deepEqual(
    offenders,
    [],
    'These regions present the RETRACTED 9.8 % swing as the current noise floor:\n' +
      offenders.map((o) => `  - ${o}`).join('\n') +
      '\n\nperf-baseline.md §6f withdrew that figure as evidence: its window overlapped ' +
      'two CPU-heavy ONNX exports, so the swing had a CAUSE and was not ambient. The ' +
      'measured null-A/B floor (§8.1, true delta ZERO by construction) reaches ' +
      '+52.30 % / -40.17 %. Cite that instead — it is ~5x larger, so every argument of ' +
      'the form "the effect is smaller than the noise" gets STRONGER. To mention the ' +
      'old figure at all, name §6f in the same region.',
  );
});
