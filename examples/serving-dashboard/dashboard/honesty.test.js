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
/**
 * Source with comments removed.
 *
 * These lints scan text, so without this a comment SAYING "panels never call
 * fetch()" would itself trip the rule forbidding fetch() — punishing the
 * documentation of a rule for describing it. Block comments first, then line
 * comments.
 *
 * @param {string} source
 * @returns {string}
 */
function stripComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1');
}

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

  it('never opens its own connection or its own clock', () => {
    // CONTRACT.md §4 and ARCHITECTURE.md I4: ONE polling loop for the page.
    // Independent polling drifts, and two panels showing different instants
    // make the dashboard contradict itself — the failure is not an error, it
    // is two numbers that cannot both be true with nothing to say which is.
    //
    // fetch() is only the obvious route. A panel could just as easily open an
    // EventSource, a WebSocket or an XMLHttpRequest, or run its own setInterval
    // and drift its own cadence, and every one of those looks entirely normal
    // in review. Relative URLs are a second trap: the page is served at /demo/
    // WITH a trailing slash, so a relative fetch resolves somewhere surprising.
    // Forbidding the whole class is cheaper than remembering the list.
    const FORBIDDEN = [
      [/\bfetch\s*\(/, 'fetch()'],
      [/\bnew\s+EventSource\b/, 'EventSource'],
      [/\bnew\s+WebSocket\b/, 'WebSocket'],
      [/\bnew\s+XMLHttpRequest\b/, 'XMLHttpRequest'],
      [/\bnavigator\s*\.\s*sendBeacon\b/, 'sendBeacon'],
      [/\bsetInterval\s*\(/, 'setInterval()'],
    ];

    const offences = [];
    for (const name of sourceFiles()) {
      const source = stripComments(readFileSync(`${DASHBOARD_DIR}${name}`, 'utf8'));
      for (const [pattern, what] of FORBIDDEN) {
        if (pattern.test(source)) {
          offences.push(`${name} uses ${what} — panels subscribe, they never poll`);
        }
      }
    }

    assert.deepEqual(offences, [], offences.join('\n'));
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

// ---------------------------------------------------------------------------
// §13(b): every panel declares the engine capability it needs.
//
// The shell uses this to decide what to mount. The reason it is asserted for
// EVERY panel rather than only for the ones that need a capability is that the
// dangerous value is the MISSING one: an undeclared panel is indistinguishable
// from a panel declared as universal, so a KV panel that forgot to declare
// would mount on a profile that cannot feed it and fill with em-dashes. Making
// the declaration mandatory turns that from a silent default into a build
// failure.
// ---------------------------------------------------------------------------

describe('panels declare their capability requirement', () => {
  const VALID = new Set(['continuous-batch', 'paged-kv', null]);

  it('every panel declares `requires` explicitly, including the universal ones', async () => {
    for (const file of panelSources()) {
      const module = await import(`./${file}`);
      const meta = module.meta;

      assert.ok(meta, `${file} exports no meta`);
      assert.ok(
        'requires' in meta,
        `${file} does not declare meta.requires. Declare it as null if the panel ` +
          'works on every profile — an absent declaration reads as universal, which ' +
          'is exactly how a panel ends up mounted on an engine that cannot feed it.',
      );
      assert.ok(
        VALID.has(meta.requires),
        `${file} declares meta.requires = ${JSON.stringify(meta.requires)}, which is ` +
          `not a capability the shell knows. A typo here silently means "universal".`,
      );
    }
  });

  it('keeps the KV panel universal, because it adapts rather than disappearing', async () => {
    // §13(d). Guards a specific and tempting mistake: declaring 'paged-kv'
    // here would look obviously correct and would DELETE the KV story from the
    // static-cache profile — which is the profile the demo actually runs on.
    // The panel adapts instead: a paged block table on one, decode-row
    // occupancy on the other.
    const { meta } = await import('./kv-memory.js');
    assert.equal(meta.requires, null);
  });

  it('keeps the prefix panel universal, because it ships showing whatever is true', async () => {
    const { meta } = await import('./prefix-cache.js');
    assert.equal(
      meta.requires,
      null,
      'The prefix panel ships unconditionally, including a stark 0%. Gating it ' +
        'behind a capability would hide the panel exactly where its answer is ' +
        'least flattering, which is the one genuinely dishonest move here.',
    );
  });
});
