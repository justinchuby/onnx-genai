// Copyright (c) Microsoft Corporation.
//
// The honesty lint.
//
// CONTRACT.md §2 says a reviewer can grep for a `.value` access not preceded by
// a guard. This file does that grep, so the check runs on every commit instead
// of depending on a reviewer being thorough on a tired afternoon. It is the
// project's single most important rule, and the only rule cheap enough to
// enforce mechanically.
//
// It is a lint, not a proof: it checks that a guard is applied in the same
// small neighbourhood as the access. That is enough to catch the realistic
// failure — someone adding a line to an existing render function and reaching
// straight for `.value` because everything around it already works.

import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';

const DASHBOARD_DIR = fileURLToPath(new URL('.', import.meta.url));

/** Modules whose whole job is to interpret the envelope; they read state directly. */
const GUARD_MODULES = new Set(['field-state.js', 'panel-kit.js']);

/** Guards that count as having read the state before the value. */
const GUARD_PATTERN =
  /\b(isRenderable|hasValue|numericValueOf|isUnavailable|isPending|isStale|renderStateOf|renderField|metricRow|Array\.isArray)\b/;

/** @returns {string[]} */
function sourceFiles() {
  return readdirSync(DASHBOARD_DIR)
    .filter((name) => name.endsWith('.js'))
    .filter((name) => !name.endsWith('.test.js'))
    .filter((name) => !GUARD_MODULES.has(name));
}

describe('honesty lint — no unguarded value reads', () => {
  it('never reads .value without a guard nearby', () => {
    /** @type {string[]} */
    const offences = [];

    for (const name of sourceFiles()) {
      const lines = readFileSync(`${DASHBOARD_DIR}${name}`, 'utf8').split('\n');
      lines.forEach((line, index) => {
        const code = line.replace(/\/\/.*$/, '');
        if (!/\.value\b/.test(code)) return;

        // A guard anywhere in the enclosing few lines counts. Widening this
        // window trades false negatives for false positives; four lines is
        // where a value read and its guard stop being one thought.
        const window = lines.slice(Math.max(0, index - 4), index + 1).join('\n');
        if (GUARD_PATTERN.test(window)) return;

        offences.push(`${name}:${index + 1}  ${line.trim()}`);
      });
    }

    assert.deepEqual(
      offences,
      [],
      `unguarded field.value reads — a documented zero would render as a measurement:\n${offences.join('\n')}`,
    );
  });

  it('has something to check, so a passing run means something', () => {
    // Guarding against the lint quietly becoming a no-op if the glob or the
    // directory layout changes — a green check that inspects nothing is worse
    // than no check, because it is trusted.
    const files = sourceFiles();
    assert.ok(files.length >= 6, `expected to lint the panels, found ${files.length} files`);
    const anyValueReads = files.some((name) =>
      /\.value\b/.test(readFileSync(`${DASHBOARD_DIR}${name}`, 'utf8')),
    );
    assert.ok(anyValueReads, 'no .value reads found at all — the lint is inspecting nothing');
  });

  it('never renders a bare zero for a metric the server cannot measure', () => {
    // The documented-zero keys, named. Binding any of them as a plain field is
    // the mistake the whole project exists to prevent: the state reads
    // 'measured' and the value is fabricated at the source, so no amount of
    // state-checking in the panel saves you.
    const DOCUMENTED_ZEROS = [
      'throughput.tokens_per_second',
      'kv.usage',
      'batch.utilization',
    ];

    /** @type {string[]} */
    const offences = [];
    for (const name of sourceFiles()) {
      const source = readFileSync(`${DASHBOARD_DIR}${name}`, 'utf8');
      for (const key of DOCUMENTED_ZEROS) {
        const pattern = new RegExp(`field\\('${key.replace('.', '\\.')}'\\)`);
        if (pattern.test(source)) {
          offences.push(`${name} binds ${key}, which the server reports as a hardcoded zero`);
        }
      }
    }

    assert.deepEqual(offences, [], offences.join('\n'));
  });

  it('never calls fetch from a panel', () => {
    // CONTRACT.md §4: one polling loop for the page. Independent polling drifts,
    // and two panels showing different instants make the dashboard contradict
    // itself.
    const offenders = sourceFiles().filter((name) =>
      /\bfetch\s*\(/.test(readFileSync(`${DASHBOARD_DIR}${name}`, 'utf8')),
    );
    assert.deepEqual(offenders, [], `panels must not fetch: ${offenders.join(', ')}`);
  });

  it('never assigns innerHTML with anything but a literal', () => {
    // Server messages are shown verbatim and must not be interpretable as
    // markup. The server's own error text is the most likely thing to end up
    // here, and it is exactly the string we must not trust.
    /** @type {string[]} */
    const offences = [];
    for (const name of sourceFiles()) {
      const lines = readFileSync(`${DASHBOARD_DIR}${name}`, 'utf8').split('\n');
      lines.forEach((line, index) => {
        if (!/innerHTML\s*=/.test(line)) return;
        if (/innerHTML\s*=\s*''/.test(line)) return;
        offences.push(`${name}:${index + 1}  ${line.trim()}`);
      });
    }
    assert.deepEqual(offences, [], `use textContent:\n${offences.join('\n')}`);
  });
});

/**
 * Modules that legitimately touch the raw painter: sparkline.js defines it and
 * panel-kit.js is the wrapper that adds the table. Everything else is a panel.
 *
 * @returns {string[]}
 */
function panelSources() {
  const infrastructure = new Set(['sparkline.js', 'store-adapter.js', 'index.js']);
  return sourceFiles().filter((name) => !infrastructure.has(name));
}

describe('accessibility cannot be skipped one panel at a time', () => {
  it('routes every panel chart through renderSparkline', () => {
    // AC28 requires a table alternative for every canvas. renderSparkline is
    // the only call site that builds one, so a panel that reaches for
    // paintSparkline directly gets a chart with no readable alternative and
    // still looks perfectly correct on screen. prefix-cache.js did exactly
    // that, and nothing caught it until this rule existed: the accessible
    // path has to be the ONLY path, not the recommended one.
    const offenders = [];
    for (const file of panelSources()) {
      const source = readFileSync(`${DASHBOARD_DIR}${file}`, 'utf8');
      if (/\bpaintSparkline\s*\(/.test(source)) {
        offenders.push(`${file}: calls paintSparkline directly`);
      }
      if (/from '\.\/sparkline\.js'/.test(source)) {
        offenders.push(`${file}: imports sparkline.js instead of using panel-kit`);
      }
    }
    assert.deepEqual(offenders, [], offenders.join('\n'));
  });

  it('never makes an annotation its own tab stop', () => {
    // AC29. A panel of unavailable values must cost a keyboard user ONE tab
    // stop, not one per em-dash. Read-only annotations join the roving cursor
    // with tabindex -1; only genuinely operable controls (buttons, the chart
    // figure) may be reached by Tab directly.
    const offenders = [];
    for (const file of panelSources()) {
      const source = readFileSync(`${DASHBOARD_DIR}${file}`, 'utf8');
      const lines = source.split('\n');
      lines.forEach((line, index) => {
        if (!/tabindex['"]?\s*:\s*'0'|['"]tabindex['"],\s*'0'/.test(line)) return;
        offenders.push(`${file}:${index + 1}  ${line.trim()}`);
      });
    }
    assert.deepEqual(offenders, [], offenders.join('\n'));
  });

  it('gives every sparkline slot a label to name its table', () => {
    const offenders = [];
    for (const file of panelSources()) {
      const source = readFileSync(`${DASHBOARD_DIR}${file}`, 'utf8');
      for (const [, args] of source.matchAll(/createSparklineSlot\(\{([^}]*)\}/g)) {
        if (!/\blabel\s*:/.test(args)) {
          offenders.push(`${file}: createSparklineSlot without a label`);
        }
      }
    }
    assert.deepEqual(offenders, [], offenders.join('\n'));
  });
});
