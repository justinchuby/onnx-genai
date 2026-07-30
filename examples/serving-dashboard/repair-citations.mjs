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
//
// It is not wired into the test suite and must not be: a check that repairs
// its own subject can never fail.

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = execFileSync('git', ['rev-parse', '--show-toplevel'], { cwd: HERE })
  .toString()
  .trim();
const README_PATH = join(HERE, 'README.md');
const WRITE = process.argv.includes('--write');

// `path/to/file.rs:123` or bare `file.rs:123`, inside backticks.
const CITATION = /`([A-Za-z0-9_./-]+\.(?:rs|js|css|html|sh|toml|md)):(\d+)`/g;
// A backticked identifier, optionally with (), near the citation.
const SYMBOL = /`([A-Za-z_][A-Za-z0-9_]{2,})(?:\(\))?`/g;

function resolveFile(cited) {
  const direct = join(REPO, cited);
  if (existsSync(direct)) return cited;
  const base = cited.split('/').pop();
  const hits = execFileSync('git', ['ls-files', '*/' + base, base], { cwd: REPO })
    .toString()
    .split('\n')
    .filter(Boolean);
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

  const lines = readFileSync(join(REPO, file), 'utf8').split('\n');
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

if (repairs.length === 0 && declines.length === 0) {
  console.log('All README citations are anchored to their symbols. Nothing to repair.');
}
for (const r of repairs) {
  console.log(`${WRITE ? 'REPAIRED' : 'WOULD REPAIR'}  ${r.full} -> :${r.target}   (${r.symbol})`);
}
for (const d of declines) console.log(`DECLINED  ${d}`);

if (WRITE && repairs.length > 0) {
  writeFileSync(README_PATH, out);
  console.log(`\nWrote ${repairs.length} repair(s). Now run the checker — this tool is ` +
    `not evidence, it is a suggestion that happens to have edited the file:\n` +
    `  node --test check-source-citations.test.js`);
}
if (declines.length > 0) {
  console.log(`\n${declines.length} citation(s) need a human. They are NOT repaired and NOT safe to ignore.`);
  process.exitCode = WRITE ? 0 : 1;
}
