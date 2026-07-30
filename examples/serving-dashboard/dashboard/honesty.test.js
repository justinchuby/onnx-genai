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

});

describe('honesty lint — the measured state is never compared as a literal', () => {
  // WHY THIS LINT EXISTS.
  //
  // The wire spelling of the measured state has been ruled three times in one
  // session ('measured' -> 'ok' -> 'measured') and it lives in a file this
  // dashboard does not own. `field-state.js` resolves BOTH spellings, so any
  // code going through `renderStateOf`/`normaliseState` survives a flip.
  //
  // A raw `field.state === 'ok'` does not. It fails in the worst available
  // direction: silently, and toward showing LESS than we measured. A flipped
  // enum turns every live sparkline into a hatched NOT MEASURABLE YET panel
  // over a server answering perfectly — no error, no 404, nothing in DevTools,
  // and the dashboard's own honesty machinery testifying that the data is not
  // there. That is close to the most expensive bug this project could ship,
  // because every instinct would send someone to debug the server.
  //
  // Both spellings are banned as comparison literals, including the one that
  // is currently correct: pinning today's spelling is exactly the mistake.
  const MEASURED_LITERAL = /(?:[.\w]*state\s*[!=]==?\s*['"](?:ok|measured)['"])/;
  const MEASURED_LITERAL_REVERSED = /['"](?:ok|measured)['"]\s*[!=]==?\s*[.\w]*[Ss]tate/;

  it('compares through field-state.js rather than against a spelling', () => {
    /** @type {string[]} */
    const offenders = [];

    for (const name of sourceFiles()) {
      // field-state.js owns the vocabulary; the literals live in its alias
      // table by design. Every other module must go through it.
      if (name === 'field-state.js') continue;

      const source = stripComments(readFileSync(`${DASHBOARD_DIR}${name}`, 'utf8'));
      source.split('\n').forEach((line, index) => {
        if (MEASURED_LITERAL.test(line) || MEASURED_LITERAL_REVERSED.test(line)) {
          offenders.push(`${name}:${index + 1}: ${line.trim()}`);
        }
      });
    }

    assert.deepEqual(
      offenders,
      [],
      'Compare states via normaliseState()/renderStateOf() and RENDER_STATES, never against ' +
        `a spelling. A wire rename then blanks these panels silently:\n${offenders.join('\n')}`,
    );
  });

  it('MUTATION TEST — the lint actually catches a raw comparison', () => {
    // An unenforced lint is worse than no lint: it advertises a guarantee it
    // does not provide. Both orderings and both spellings must be caught.
    const shouldCatch = [
      "if (field.state === 'ok') {",
      "if (base.state !== 'measured') {",
      "return f.state == 'ok';",
      "if ('ok' === field.state) {",
      "  const live = someField.state === 'measured';",
    ];
    for (const line of shouldCatch) {
      assert.ok(
        MEASURED_LITERAL.test(line) || MEASURED_LITERAL_REVERSED.test(line),
        `lint missed a raw measured-state comparison: ${line}`,
      );
    }

    // ...and must not fire on the legitimate shapes, or it gets disabled.
    const shouldPass = [
      'if (normaliseState(base.state) !== RENDER_STATES.OK) {',
      "if (field.state === 'pending') {",
      "if (typeof field.state !== 'string') {",
      "const label = 'ok';",
    ];
    for (const line of shouldPass) {
      assert.ok(
        !MEASURED_LITERAL.test(line) && !MEASURED_LITERAL_REVERSED.test(line),
        `lint fired on a legitimate line: ${line}`,
      );
    }
  });
});

describe('honesty lint — poisoned server fields must never be bound', () => {
  // Some fields on the wire are not merely unmeasured — they are ACTIVELY
  // MISLEADING, because they carry a plausible number computed from the wrong
  // thing. Those are more dangerous than a missing field: a missing field
  // renders an em-dash and asks a question, while a wrong-denominator rate
  // renders a confident percentage that survives into a screenshot.
  //
  // Each entry names the field, the file:line that makes it wrong, and what to
  // show instead. Nothing here is a style preference; every one has been
  // measured or read out of the server source.
  const FORBIDDEN = [
    {
      // DELIBERATELY THE WHOLE NAMESPACE, NOT THE THREE NAMED COUNTERS.
      //
      // The ruling named prefix_cache_hits, _lookups and _hit_rate. Banning
      // exactly those three would still admit `prefix_cache.tokens_reused`,
      // `prefill_tokens_skipped` and `time_saved_ms` — and those are WORSE,
      // not better. They derive from `prefix_cache_hit_len`, which QA proved
      // counts tokens that merely MATCHED rather than prefill that was
      // actually skipped: engine/runtime.rs:1017-1024 never sets
      // `loaded_prompt_prefix`, so the full prompt is re-prefilled anyway. A
      // "prefill skipped: 812 tokens" row would assert saved work that the
      // measured TTFT proves did not happen.
      //
      // A guard shaped like the incident catches only the incident. The
      // architectural fact is that NO number in this namespace is backed by
      // observed reuse on either execution path, so the namespace is the
      // correct unit to ban.
      pattern: /\.(?:field|series|rate)\(\s*['"`][\w.]*prefix[_.]?cache[._][\w.]+['"`]/i,
      what: 'any prefix_cache.* value',
      why:
        'a controlled A/B (n=6 per arm, with a sensitivity control proving the instrument ' +
        'could resolve a 90% effect) found shared-prefix requests ran 7.0% SLOWER than a ' +
        'zero-sharing control, while the hit counter fired on every request including all ' +
        'six controls. The reuse is PROVEN ABSENT, not merely unmeasured — so every counter ' +
        'in this namespace describes work that never happened.',
      instead:
        'render the verified gap itself, with citations and no numbers. There is no ' +
        'client-side substitute either: the TTFT delta measures the same absent effect.',
    },
    {
      // `prefix_hashes` is NOT in the prefix_cache.* namespace, so the ban
      // above never covered it. It is the same story wearing a different name:
      // /v1/status publishes a `prefix_hashes` array, and a panel showing how
      // many prefixes are "cached" asserts reuse that the controlled A/B
      // disproved. Named separately because the namespace ban was shaped like
      // the counters we found, not like the fault.
      pattern: /\.(?:field|series|rate)\(\s*['"`][\w.]*prefix_hashes[\w.]*['"`]/i,
      what: 'status.prefix_hashes',
      why:
        'it names the same disproven mechanism from outside the prefix_cache namespace. ' +
        'A count of tracked prefixes reads as evidence of reuse, and reuse was measured ' +
        'and found absent on both execution paths.',
      instead: 'nothing — no prefix field ships in any form, on either server.',
    },
    {
      // A bare `hit_rate` on any namespace. The prefix ratio is the one we
      // caught, but the shape is what is dangerous: every hit-rate on this
      // server divides a real count by a denominator that counts something
      // else, and the ruling is to audit numerator and denominator separately
      // because they usually have different provenance.
      pattern: /\.(?:field|series|rate)\(\s*['"`][\w.]*hit_rate[\w.]*['"`]/i,
      what: 'any hit_rate field',
      why:
        'the only hit-rate on the wire divides hits (a statically-dead numerator) by ' +
        'lookups (which increments once per COMPLETED GENERATION, so it would read 135 ' +
        'with the cache deleted). A ratio of two differently-wrong numbers is the most ' +
        'authoritative-looking value we could put on screen.',
      instead: 'nothing. If a future rate is real, bind its numerator and denominator separately.',
    },
    {
      pattern: /\.(?:field|series|rate)\(\s*['"`][\w.]*tokens_per_second['"`]/,
      what: 'status.tokens_per_second',
      why: 'the server hardcodes it to a literal 0.0 (routes/admin.rs:63) — it is a stub, not a measurement.',
      instead: 'differentiate metrics.tokens_generated_total client-side, badged `derived`.',
    },
  ];

  it('binds no field whose published value is known to be wrong', () => {
    /** @type {string[]} */
    const offenders = [];

    for (const name of sourceFiles()) {
      // Comments MUST be stripped: every one of these fields is discussed by
      // name in a comment explaining why it is not bound. A lint that fires on
      // its own documentation trains people to delete the documentation.
      //
      // The patterns match a VALUE READ specifically — `.field(...)`,
      // `.series(...)`, `.rate(...)` — not any mention of the key. That
      // distinction is load-bearing: `capability()` legitimately names
      // throughput.tokens_per_second in CAPABILITY_KEYS to read its STATE and
      // decide whether a panel can populate at all, and it never touches the
      // value. Forbidding the mention rather than the read flagged that
      // correct code on the first run of this lint.
      const source = stripComments(readFileSync(`${DASHBOARD_DIR}${name}`, 'utf8'));
      source.split('\n').forEach((line, index) => {
        for (const entry of FORBIDDEN) {
          if (entry.pattern.test(line)) {
            offenders.push(
              `${name}:${index + 1} binds ${entry.what} — ${entry.why} Instead: ${entry.instead}`,
            );
          }
        }
      });
    }

    assert.deepEqual(offenders, [], `poisoned field bound:\n${offenders.join('\n')}`);
  });

  it('MUTATION TEST — the tripwire fires on a real binding', () => {
    const wouldBind = [
      // The three counters named in the ruling.
      "const rate = telemetryStore.field('prefix_cache.hit_rate');",
      'const rate = store.field(`prefix_cache_hit_rate`);',
      "const hits = telemetryStore.field('prefix_cache.hits');",
      "const gens = telemetryStore.field('prefix_cache.lookups');",
      // The derived savings fields the three-name version of this lint would
      // have let through. These are the dangerous ones: they assert saved work.
      "telemetryStore.field('prefix_cache.tokens_reused')",
      "telemetryStore.field('prefix_cache.prefill_tokens_skipped')",
      "telemetryStore.field('prefix_cache.time_saved_ms')",
      "telemetryStore.series('prefix_cache.hits', WINDOW_MS)",
      "const tps = telemetryStore.field('status.tokens_per_second');",
      // prefix_hashes — lives OUTSIDE the prefix_cache namespace and was
      // uncovered until now. It is on the live /v1/status payload, so it is
      // the single most likely poisoned field for someone to bind by accident.
      "const hashes = telemetryStore.field('status.prefix_hashes');",
      'const hashes = store.field(`prefix_hashes`);',
      "telemetryStore.series('status.prefix_hashes', WINDOW_MS)",
      "telemetryStore.field('kv.prefix_hashes_tracked')",
      // A hit_rate on any namespace, not just prefix_cache.
      "telemetryStore.field('cache.hit_rate')",
      "telemetryStore.field('kv.block_hit_rate')",
    ];
    for (const line of wouldBind) {
      assert.ok(
        FORBIDDEN.some((entry) => entry.pattern.test(line)),
        `tripwire missed a poisoned binding: ${line}`,
      );
    }

    // Must NOT fire on prose, labels, or unrelated namespaces. The panel still
    // EXPLAINS the prefix cache at length; only reading a value is forbidden.
    const legitimate = [
      "telemetryStore.rate('metrics.tokens_generated_total')",
      "const label = 'Prefix cache hit rate';",
      "telemetryStore.field('kv.pages_allocated')",
      "telemetryStore.capability('prefix-cache')",
    ];
    for (const line of legitimate) {
      assert.ok(
        !FORBIDDEN.some((entry) => entry.pattern.test(line)),
        `tripwire fired on a legitimate binding: ${line}`,
      );
    }
  });
});

// AC59 — THE WORDS "BATCH SIZE" MUST NEVER REACH THE SCREEN.
//
// Not a style rule. `onnx_genai_batch_size_current` is NAMED "batch size" and
// documented "Current generation batch size", and it is neither: it is
// fetch_add(1) on generation start, decremented in Drop, so it counts HTTP
// requests in flight and never consults ContinuousBatchManager. Fire 8
// concurrent requests at max_batch=4 and it reads 8.
//
// So the phrase is ambiguous between a number we can measure (sequences the
// engine stepped) and a number we cannot, and a viewer cannot tell which one a
// label means. Every on-screen string names what it actually counts instead.
// This lint reads STRING LITERALS ONLY — comments (including this one) and
// identifiers are unaffected, because explaining the trap is the panel's job.
describe('AC59 — "batch size" never appears in UI copy', () => {
  const BANNED = /batch[\s_-]*size/i;

  // Extract single/double/backtick literals from comment-stripped source.
  const STRING_LITERAL = /'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*"|`(?:[^`\\]|\\.)*`/g;

  // `${fields.batchSize.value}` is an IDENTIFIER, not copy — it renders as the
  // number, never as the word. Interpolations are stripped so the lint reads
  // what reaches the screen rather than the code that produces it. Without
  // this, the lint's own first run flagged two honest strings, which is how it
  // was found: the check must be aimed at the rendered text, not the source.
  const stringLiteralsOf = (source) =>
    (stripComments(source).match(STRING_LITERAL) ?? []).map((literal) =>
      literal.replace(/\$\{[^}]*\}/g, ''),
    );

  it('no rendered string in any panel says "batch size"', () => {
    const offenders = [];
    for (const file of sourceFiles()) {
      const source = readFileSync(`${DASHBOARD_DIR}${file}`, 'utf8');
      for (const literal of stringLiteralsOf(source)) {
        if (BANNED.test(literal)) offenders.push(`${file}: ${literal}`);
      }
    }
    assert.deepEqual(
      offenders,
      [],
      `AC59: "batch size" is ambiguous between the engine batch and the ` +
        `in-flight request gauge. Name what the number counts:\n${offenders.join('\n')}`,
    );
  });

  it('MUTATION TEST — the lint fires on every spelling it must catch', () => {
    const wouldFail = [
      "label: 'Maximum batch size'",
      "parts.push('Batch size is not measurable yet.')",
      '`current batch size is real`',
      "'BATCH SIZE'",
      "'batch-size'",
      "'batch_size'",
    ];
    for (const literal of wouldFail) {
      assert.ok(BANNED.test(literal), `AC59 lint missed: ${literal}`);
    }

    // Must NOT fire on the permitted vocabulary — these are the replacements
    // actually shipping, and a lint that also banned them would be unusable.
    const permitted = [
      "'Batch limit'",
      "'Sequences in the current batch'",
      "'Batch occupancy'",
      "'Generations in flight'",
      "'The engine does not report how many sequences it stepped together.'",
      // Interpolated identifiers are code, not copy. These two are real
      // strings from scheduling.js that the lint flagged on its first run.
      '`Batch occupancy  of  slots.`',
      '` sequences in the current batch; the server does not report a `',
    ];
    for (const literal of permitted) {
      assert.ok(!BANNED.test(literal), `AC59 lint over-fired on: ${literal}`);
    }
  });
});
