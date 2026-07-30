// Copyright (c) Microsoft Corporation.
//
// Telling a COMMAND from PROSE ABOUT A COMMAND, in markdown.
//
// Every documentation guard in this repository needs the same discrimination
// and every one of them has had to rediscover it. The defect it prevents is the
// one that makes a guard unfixable:
//
//   DOCUMENTING A DEFECT TRIPS THE GUARD AGAINST THE DEFECT.
//
// A well-written fix quotes the bug it killed, so the hit and the proof-of-fix
// are byte-identical. A guard that cannot tell them apart forces its own
// documentation to stay silent, and then reports a confident positive off a
// tombstone.
//
// This module was extracted from `check-launch-command.test.js`, where it lived
// as a closure with three call sites, all its own. An unexported discriminator
// is a fix that applies exactly once.
//
// WHAT THIS DOES NOT DO, stated here rather than discovered later: it does not
// parse markdown. It is a line classifier. It cannot see indented (four-space)
// code blocks, and it does not attempt to. Callers that need those must say so.

/** Fence delimiters open and close on their own line, ``` or ~~~. */
const FENCE = /^\s*(?:```|~~~)/;

/**
 * A blockquote line. Prose QUOTING a command is quoted, by convention, in this
 * repository's review documents -- the history blockquotes name superseded
 * commands on purpose, and grading them as prescriptions is the defect above.
 *
 * @param {string} line
 * @returns {boolean}
 */
export function isBlockquote(line) {
  return /^\s*>/.test(line);
}

/**
 * Lines inside fenced code blocks, with their 1-based line numbers.
 *
 * ONLY FENCED LINES COUNT, AND THAT IS LOAD-BEARING. A prose sentence may quote
 * a command in order to explain why it is WRONG, and a whole-file scan makes a
 * cautionary counter-example indistinguishable from a prescription. The
 * counter-example is usually the defective command quoted verbatim -- which is
 * exactly the shape these guards hunt.
 *
 * @param {string} source
 * @returns {Array<{ line: string, lineNumber: number }>}
 */
export function fencedLines(source) {
  const inside = [];
  let inFence = false;
  let lineNumber = 0;
  for (const line of source.split('\n')) {
    lineNumber += 1;
    if (FENCE.test(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) inside.push({ line, lineNumber });
  }
  return inside;
}

/**
 * A line a reader could paste into a shell.
 *
 * Deliberately a SHAPE test rather than a keyword test: a line MENTIONING
 * `run-tests.sh` while explaining that it used to be spelled differently is
 * prose, and banning the topic rather than the claim is what makes a guard
 * impossible to document around.
 *
 * BLOCKQUOTES ARE HANDLED BY THE PREFIX RULE, NOT BY A SECOND CHECK. The first
 * version of this function called `isBlockquote()` first and the doc comment
 * claimed the exclusion was explicit rather than incidental. IT WAS NOT, AND THE
 * MUTATION PROVED IT: deleting that call left every test green, because `>` is
 * not one of the accepted prefixes and never can be. It was dead code asserting
 * its own importance -- a second mechanism enforcing what the first already
 * holds by construction, which is a divergence waiting to happen and, until it
 * diverges, a line that makes a reader think a case was considered.
 *
 * `isBlockquote` remains exported for callers that scan PROSE, where the
 * distinction is real.
 *
 * @param {string} line
 * @returns {boolean}
 */
export function isCommandLine(line) {
  const trimmed = line.trim().replace(/^\$\s*/, '');
  return (
    trimmed.startsWith('node --test') || trimmed.startsWith('./') || trimmed.startsWith('cd ')
  );
}
