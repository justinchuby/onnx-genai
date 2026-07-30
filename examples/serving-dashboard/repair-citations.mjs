#!/usr/bin/env node
// Recompute the line numbers in README.md citations from the symbols the prose
// names. Read-only by default; pass --write to apply.
//
//   node repair-citations.mjs           # report what would change
//   node repair-citations.mjs --write   # apply, then re-run the checker
//
// WHY THIS EXISTS. check-source-citations.test.js proves a citation has gone
// stale; it cannot fix one, so every drift cost a manual archaeology pass.
// `driver.rs` moved THREE times in a single session -- :678 -> :717 -> :753 ->
// :850 -- and each move meant grepping for the definition and hand-editing the
// prose. That is a number maintained by hand in two places, which is the exact
// thing this project forbids everywhere else: a surviving number must be
// COMPUTED FROM THE ARTIFACT OR DELETED. A line number in prose is a copy of a
// fact that lives in a source file, and copies rot.
//
// ⚠️ THIS SCRIPT WRITES CITATIONS, SO IT IS SUBJECT TO THE RULE IT SERVES: AN
// AUTOMATED SUGGESTION IS A CLAIM. A repair tool that guesses converts a
// correct citation into a wrong one and leaves it looking freshly verified,
// which is worse than the drift it fixes -- stale citations are at least
// DETECTED by the checker, whereas a confidently-wrong repair passes it. So it
// is deliberately timid:
//
//   * It only rewrites when the symbol has EXACTLY ONE definition-shaped
//     occurrence. Two candidates means a judgement call, and it does not make
//     judgement calls -- it reports and leaves the citation alone.
//   * It only considers DEFINITIONS (fn/struct/enum/impl/const/static/let
//     bindings), never call sites. The first version of the checker's error
//     message named a call site as a definition's home and was wrong; this
//     tool must not repeat that in a form that EDITS THE FILE.
//   * It never invents a file. If the cited path does not resolve to exactly
//     one file, it declines.
//   * It reports declines as loudly as repairs. A tool that silently skips
//     what it cannot handle produces a clean run that means "I fixed
//     everything I felt like", and the reader hears "everything is fixed."
//   * It refuses to WRITE into a dirty README for the same reason it refuses
//     to READ a dirty source: the file then has two states, and the rewrite
//     is a whole-file overwrite from a snapshot taken before the repairs were
//     computed.
//
// It is not wired into the test suite and must not be: a check that repairs
// its own subject can never fail.

import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join, relative } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = execFileSync('git', ['rev-parse', '--show-toplevel'], { cwd: HERE })
  .toString()
  .trim();
const README_PATH = join(HERE, 'README.md');
const README_REL = relative(REPO, README_PATH);
const WRITE = process.argv.includes('--write');

// `path/to/file.rs:123` or bare `file.rs:123`, inside backticks.
const CITATION = /`([A-Za-z0-9_./-]+\.(?:rs|js|css|html|sh|toml|md)):(\d+)`/g;
// A backticked identifier, optionally with (), near the citation.
const SYMBOL = /`([A-Za-z_][A-Za-z0-9_]{2,})(?:\(\))?`/g;

/**
 * A CITATION IS A PROMISE TO A READER WHO HAS THE SHIPPED TREE, NOT MY DESK.
 *
 * Every read below resolves against `HEAD` rather than the working copy. Under
 * `--write` this tool BAKES A LINE NUMBER INTO A COMMITTED DOCUMENT, so if it
 * counts lines in an uncommitted edit it emits a citation that is correct on
 * exactly one machine in the world and wrong for every reviewer. That is the
 * same defect already fixed in `check-perf-claims.test.js` ("Nobody clones my
 * working tree") and in `scenario-routes.test.js`; this is its third site, and
 * the only one where the wrong answer gets WRITTEN DOWN rather than merely
 * mis-reported.
 *
 * Returns null when the path is not in HEAD, which is the honest answer for a
 * citation target: a reviewer cannot follow a link to a file they do not have.
 */
function shipped(rel) {
  try {
    return execFileSync('git', ['show', `HEAD:./${rel}`], {
      cwd: REPO,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
      stdio: ['ignore', 'pipe', 'ignore'],
    });
  } catch {
    return null;
  }
}

/** Paths tracked in HEAD. `ls-files` would also report staged-but-uncommitted. */
let headFilesCache = null;
function headFiles() {
  if (headFilesCache === null) {
    headFilesCache = execFileSync('git', ['ls-tree', '-r', 'HEAD', '--name-only'], {
      cwd: REPO,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
    })
      .split('\n')
      .filter(Boolean);
  }
  return headFilesCache;
}

function isDirty(rel) {
  return (
    execFileSync('git', ['status', '--porcelain', '--', rel], { cwd: REPO, encoding: 'utf8' })
      .trim().length > 0
  );
}

function resolveFile(cited) {
  // NOT `existsSync`. An untracked file exists on disk and is invisible to a
  // reviewer, so resolving to it would anchor a citation to a file nobody else
  // has -- the failure this tool is supposed to repair, committed by the tool.
  if (shipped(cited) !== null) return cited;
  const base = cited.split('/').pop();
  const hits = headFiles().filter((f) => f === base || f.endsWith('/' + base));
  return hits.length === 1 ? hits[0] : null;
}

// Definition-shaped only. A call site is not a definition, and this tool edits
// files, so the distinction is the whole safety margin.
function definitionsOf(symbol, lines) {
  const patterns = [
    new RegExp(`\\bfn\\s+${symbol}\\b`),
    new RegExp(`\\b(?:struct|enum|trait|type|union)\\s+${symbol}\\b`),
    new RegExp(`\\b(?:const|static)\\s+${symbol}\\b`),
    new RegExp(`\\bimpl\\b[^{]*\\b${symbol}\\b`),
    new RegExp(`^\\s*(?:export\\s+)?(?:async\\s+)?function\\s+${symbol}\\b`),
    new RegExp(`^\\s*(?:export\\s+)?(?:const|let|var)\\s+${symbol}\\b`),
    // Rust field DECLARATION: `pub eviction_order: Vec<EvictionTier>,`. The
    // type must start uppercase or be a primitive, which is what separates a
    // declaration from a struct-literal INITIALISER (`eviction_order: if ...`).
    // The dry run caught this: without the type constraint the tool proposed
    // moving a correct citation off the declaration and onto a construction
    // site 342 lines away.
    new RegExp(`^\\s*(?:pub(?:\\([^)]*\\))?\\s+)?${symbol}\\s*:\\s*(?:&|\\*)?(?:mut\\s+)?(?:[A-Z]|u8|u16|u32|u64|usize|i8|i16|i32|i64|isize|f32|f64|bool|str|char)`),
  ];
  const out = [];
  lines.forEach((l, i) => {
    if (patterns.some((p) => p.test(l))) out.push(i + 1);
  });
  return out;
}

const readme = readFileSync(README_PATH, 'utf8');
const repairs = [];
const declines = [];

for (const m of readme.matchAll(CITATION)) {
  const [full, citedPath, citedLineText] = m;
  const citedLine = Number(citedLineText);

  const file = resolveFile(citedPath);
  if (!file) {
    declines.push(`${full} — cited path does not resolve to exactly one tracked file.`);
    continue;
  }

  // Symbols named in the same neighbourhood as the citation.
  const window = readme.slice(Math.max(0, m.index - 240), m.index + 240);
  const named = [...new Set([...window.matchAll(SYMBOL)].map((s) => s[1]))];

  // A file with uncommitted changes has TWO line numberings and neither is the
  // one the reviewer will read: HEAD's is about to change, the desk's will
  // never be published. Refusing is the same judgement the tool already makes
  // for ambiguous symbols -- it declines rather than guesses.
  if (isDirty(file)) {
    declines.push(
      `${full} — ${file} has uncommitted changes, so a line number counted ` +
        `from it describes neither HEAD nor what the reviewer will clone. ` +
        `Commit it and re-run; this tool will not guess.`,
    );
    continue;
  }

  const lines = shipped(file).split('\n');
  const anchored = named.filter((s) => definitionsOf(s, lines).length > 0);

  if (anchored.length === 0) continue; // nothing to anchor to; checker ignores it too
  if (anchored.length > 1) {
    declines.push(
      `${full} — prose names ${anchored.length} symbols with definitions here ` +
        `(${anchored.join(', ')}); which one the line is meant to point at is a ` +
        `judgement call, so this tool will not guess.`,
    );
    continue;
  }

  const symbol = anchored[0];

  // AGREE WITH THE CHECKER ABOUT WHAT "ANCHORED" MEANS, OR FIGHT IT FOREVER.
  // The checker asks only whether the symbol TEXT appears within its window of
  // the cited line -- it does not care whether that occurrence is a definition.
  // This tool ranks definitions, which is a different question, and asking the
  // different question is how it proposed moving `governor.rs:166` (the field
  // declaration `pub eviction_order: Vec<EvictionTier>`, exactly what the prose
  // discusses) onto :508, a struct-literal initialiser. A repair tool that
  // "fixes" citations the checker is already happy with does not reduce drift;
  // it manufactures churn and eventually breaks something correct.
  const alreadyAnchored = lines
    .slice(Math.max(0, citedLine - 1 - 6), citedLine - 1 + 7)
    .some((l) => l.includes(symbol));
  if (alreadyAnchored) continue;
  const defs = definitionsOf(symbol, lines);
  if (defs.length > 1) {
    declines.push(
      `${full} — \`${symbol}\` has ${defs.length} definition-shaped lines ` +
        `(:${defs.join(', :')}). Ambiguous; not guessing.`,
    );
    continue;
  }

  const target = defs[0];
  if (Math.abs(target - citedLine) <= 6) continue; // already anchored

  repairs.push({
    index: m.index,
    full,
    replacement: `\`${citedPath}:${target}\``,
    symbol,
    citedLine,
    target,
  });
}

// ⚠️ APPLIED BY OFFSET, NEWEST-FIRST -- NEVER BY STRING REPLACEMENT.
// The first draft did `out.split(full).join(replacement)`, and the dry run
// caught it: `governor.rs:166` is cited TWICE, and the two occurrences sit in
// different prose. One resolved to a single symbol and the other named two, so
// the tool DECLINED one occurrence and repaired the other -- and then the
// global replace stamped the repair over the occurrence it had just refused to
// judge. A tool that declines to guess, and then silently applies its guess
// anyway, is worse than one that never declined: the decline message makes it
// look careful. Offsets keep each occurrence's verdict attached to that
// occurrence.
const applied = [...repairs].sort((a, b) => b.index - a.index);
let out = readme;
for (const r of applied) {
  out = out.slice(0, r.index) + r.replacement + out.slice(r.index + r.full.length);
}

// Decided BEFORE the verdicts are printed, not at the write site. The label
// below is the tool's claim about what it did, and a run that prints
// `REPAIRED` and then refuses to write has told the reader the opposite of
// what happened -- exactly the silent-skip failure the header forbids.
const refusedWrite = WRITE && repairs.length > 0 && isDirty(README_REL);

if (repairs.length === 0 && declines.length === 0) {
  console.log('All README citations are anchored to their symbols. Nothing to repair.');
}
for (const r of repairs) {
  console.log(
    `${WRITE && !refusedWrite ? 'REPAIRED' : 'WOULD REPAIR'}  ${r.full} -> :${r.target}   (${r.symbol})`,
  );
}
for (const d of declines) console.log(`DECLINED  ${d}`);

if (WRITE && repairs.length > 0) {
  // THE RULE THIS TOOL ALREADY APPLIES TO EVERY FILE IT CITES, APPLIED TO THE
  // ONE FILE IT EDITS. `isDirty` refuses a cited source with uncommitted
  // changes because it has two line numberings and neither is the reviewer's.
  // README.md was exempt from that argument for no reason other than that it is
  // the output rather than an input, and it is subject to the identical one:
  // it is read from the working copy at the top of this file, and the numbers
  // written into it are counted from HEAD, so repairing a dirty README produces
  // a document that matches neither tree.
  //
  // It is also a lost update. The snapshot was taken before the repairs were
  // computed and every offset in `repairs` indexes into it, so writing the
  // whole file back silently discards anything that landed here in between --
  // which, on a branch several agents are editing, is the ordinary case rather
  // than the unlucky one.
  if (refusedWrite) {
    console.log(
      `\nREFUSED TO WRITE  ${README_REL} has uncommitted changes. This tool rewrites ` +
        `the whole file from a snapshot taken before the repairs above were computed, ` +
        `so writing now would discard that work and anchor HEAD-derived line numbers ` +
        `into a document that matches neither HEAD nor the tree the reviewer clones. ` +
        `Commit or stash it and re-run -- the ${repairs.length} repair(s) above are ` +
        `unaffected and will be found again.`,
    );
  } else {
    writeFileSync(README_PATH, out);
    console.log(`\nWrote ${repairs.length} repair(s). Now run the checker — this tool is ` +
      `not evidence, it is a suggestion that happens to have edited the file:\n` +
      `  node --test check-source-citations.test.js`);
  }
}
if (declines.length > 0) {
  console.log(`\n${declines.length} citation(s) need a human. They are NOT repaired and NOT safe to ignore.`);
  process.exitCode = WRITE ? 0 : 1;
}
// After the block above, which sets 0 under --write: a refusal is a failure to
// do the job it was asked to do, and must not be reported as success merely
// because some citation elsewhere also needed a human.
if (refusedWrite) process.exitCode = 1;
