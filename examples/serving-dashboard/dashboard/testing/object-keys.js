// Copyright (c) Microsoft Corporation.
//
// Read the keys an object literal DECLARES, duplicates preserved.
//
// This exists because a duplicate key is invisible everywhere else. JS keeps
// the last definition and silently discards the earlier one -- no error, no
// warning, and nothing to lint against in a repo with no linter. The executed
// object cannot be asked, because execution is what destroys the evidence: by
// the time you hold `PROVENANCE`, the loser is gone. Only the SOURCE knows.
//
// WHY THIS IS NOT A REGEX, stated as a measurement rather than a preference.
// The guard that shipped first matched `/^ {2}'([A-Za-z0-9_.]+)': \{/gm`. It
// catches a duplicate written the way the file happens to write them today,
// and it is BLIND to `"batch.capacity"` -- the same key, double quoted, which
// JavaScript treats as identical. Proven by mutation: injecting that form left
// the whole suite green while the catalogue silently lost an entry. Its
// anti-vacuity control could not save it either, because an unmatched line
// never enters the count it reconciles, so both halves agree and both are
// wrong. A shape-matching guard can only ever see the shapes it was taught.
//
// WHAT THIS DELIBERATELY DOES NOT DO: it is not a JavaScript parser and must
// not be sold as one. It tracks strings, template literals, comments, regex
// literals and brace depth -- enough to read a data table honestly -- and it
// refuses to guess beyond that. The first draft omitted regex literals and
// ternaries and reported two confident duplicates that were neither: a regex
// containing `ModelDecodePath::PastPresent {` and the expression
// `discrete ? top : null`, whose ternary colon reads exactly like a key colon.
// Both are handled below, and `object-keys.test.js` pins both by name, because
// a scanner's false POSITIVES are as damaging as its blind spots: they teach
// the crew to disbelieve the instrument.

/** Characters after which a `/` begins a regex literal rather than division. */
const REGEX_PRECEDERS = new Set(['(', ',', '=', ':', '[', '!', '&', '|', '?', '{', '}', ';', '\n', '+', '-', '*', '%', '<', '>', '~', '^']);

/** A key is only a key where a key may appear: right after `{` or `,`. */
const KEY_PRECEDERS = new Set(['{', ',']);

const KEY_PATTERN = /^(?:'([^'\\]*)'|"([^"\\]*)"|([A-Za-z_$][\w$]*))\s*:/;

/**
 * Index of the `{` opening the object literal that follows `marker`.
 *
 * @param {string} source
 * @param {string} marker exact text preceding the literal, e.g. `export const PROVENANCE =`
 * @returns {number}
 */
export function findLiteralOpener(source, marker) {
  const at = source.indexOf(marker);
  if (at === -1) {
    throw new Error(
      `findLiteralOpener: "${marker}" does not appear in the source. A scanner aimed at `
        + 'nothing reports no duplicates, which is byte-identical to a clean file, so this '
        + 'throws rather than returning an empty and reassuring answer.',
    );
  }
  const brace = source.indexOf('{', at + marker.length);
  if (brace === -1) throw new Error(`findLiteralOpener: no "{" follows "${marker}".`);
  return brace;
}

/**
 * Every key the literal at `openerIndex` declares directly, in written order,
 * WITH duplicates preserved. Keys of nested objects are not returned -- they
 * belong to a different scope and cannot collide with these.
 *
 * @param {string} source
 * @param {number} openerIndex index of the opening `{`
 * @returns {Array<{ name: string, line: number }>}
 */
export function declaredKeys(source, openerIndex) {
  if (source[openerIndex] !== '{') {
    throw new Error(`declaredKeys: index ${openerIndex} is "${source[openerIndex]}", not "{".`);
  }

  const found = [];
  let depth = 0;
  let line = 1 + [...source.slice(0, openerIndex)].filter((c) => c === '\n').length;
  let previous = '{';
  let i = openerIndex;

  while (i < source.length) {
    const c = source[i];
    const next = source[i + 1];

    if (c === '\n') { line++; i++; continue; }
    if (c === ' ' || c === '\t' || c === '\r') { i++; continue; }

    if (c === '/' && next === '/') { while (i < source.length && source[i] !== '\n') i++; continue; }
    if (c === '/' && next === '*') {
      i += 2;
      while (i < source.length && !(source[i] === '*' && source[i + 1] === '/')) { if (source[i] === '\n') line++; i++; }
      i += 2;
      continue;
    }

    // A key is tested BEFORE its opening quote can be eaten as a string. The
    // first draft checked strings first, so every quoted key in the file was
    // consumed as a string literal and the scanner found nothing -- a clean
    // report from an instrument that was not looking. Only a positive control
    // caught it.
    if (depth === 1 && KEY_PRECEDERS.has(previous)) {
      const match = KEY_PATTERN.exec(source.slice(i));
      if (match) {
        found.push({ name: match[1] ?? match[2] ?? match[3], line });
        const consumed = match[0];
        line += [...consumed].filter((ch) => ch === '\n').length;
        i += consumed.length;
        previous = ':';
        continue;
      }
    }

    if (c === "'" || c === '"' || c === '`') {
      const quote = c;
      i++;
      while (i < source.length && source[i] !== quote) {
        if (source[i] === '\\') i++;
        else if (source[i] === '\n') line++;
        i++;
      }
      i++;
      previous = quote;
      continue;
    }

    if (c === '/' && REGEX_PRECEDERS.has(previous)) {
      i++;
      let inClass = false;
      while (i < source.length && (inClass || source[i] !== '/')) {
        if (source[i] === '\\') i++;
        else if (source[i] === '[') inClass = true;
        else if (source[i] === ']') inClass = false;
        else if (source[i] === '\n') break;
        i++;
      }
      i++;
      previous = '/';
      continue;
    }

    if (c === '{') { depth++; previous = '{'; i++; continue; }
    if (c === '}') {
      depth--;
      previous = '}';
      i++;
      if (depth === 0) return found;
      continue;
    }

    previous = c;
    i++;
  }

  throw new Error('declaredKeys: the literal never closed; the scanner lost sync and must not report a result.');
}

/**
 * Keys declared more than once, in first-seen order.
 *
 * @param {Array<{ name: string, line: number }>} keys
 * @returns {Array<{ name: string, lines: number[] }>}
 */
export function duplicatesAmong(keys) {
  const byName = new Map();
  for (const { name, line } of keys) {
    if (!byName.has(name)) byName.set(name, []);
    byName.get(name).push(line);
  }
  return [...byName.entries()]
    .filter(([, lines]) => lines.length > 1)
    .map(([name, lines]) => ({ name, lines }));
}
