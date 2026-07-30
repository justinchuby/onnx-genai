// Copyright (c) Microsoft Corporation.
//
// A flag named in a document must exist in the binary.
//
// `--max-batch` circulated for roughly an hour as "approved and surfaced",
// was reasoned from twice in design decisions, and became load-bearing for a
// committed panel design -- while not existing. It does exist now. The point
// is that nobody ran `grep` for it, because "approved" reads like "done", and
// an approved-but-absent flag has no symptom until a visitor pastes the
// command and the server exits with an unrecognised-argument error.
//
// This is the mechanical form of that check: every flag this demo tells a
// visitor to type must be a real argument of the binary they are typing it
// into.
//
// SCOPE, and why it is narrow on purpose:
//
// A bare `--foo` token is not evidence of a CLI flag. This codebase also
// contains BEM modifier classes (`value__num--unavailable`), CSS custom
// properties (`--og-fg-muted`), flags belonging to OTHER binaries (`cargo
// build --release`, `node --test`, `scripts/build_qwen.sh --runtime`), and
// prose that quotes all of the above. Auditing every `--token` would produce
// an alarm nobody reads, which is worse than no audit: a check that cries
// wolf gets suppressed, and then it is not there for the one case it was
// written for.
//
// So flags are attributed to a binary by INVOCATION CONTEXT: a flag counts
// against the server only if it appears in a command that actually runs the
// server. Note that `cargo build --release -p onnx-genai-server` mentions the
// server without invoking it -- an earlier draft of this file attributed
// `--release` to the server and would have failed on a correct README.
//
// Run: node --test examples/serving-dashboard/check-cli-flags.test.js

import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve, relative } from 'node:path';
import { assertShippingTree } from './shipping-tree.mjs';

// Provenance before content. Every path below is resolved from import.meta.url,
// so this file would read a parked worktree self-consistently and pass. Assert
// which tree we are in BEFORE asserting anything about what is in it.
assertShippingTree();

const demoDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(demoDir, '..', '..');

// READ WHAT SHIPS, NOT WHAT IS ON MY DESK.
//
// This guard used to readFileSync the working tree. A documented flag repaired
// only on disk scored GREEN and then evaporated on the next checkout, and the
// two readings are byte-identical whenever the tree is clean -- which is
// exactly when you are most likely to trust the result. A reviewer clones HEAD.
// So does the demo. Nobody clones my working tree.
const read = (relPath) =>
  execFileSync('git', ['show', `HEAD:${relPath}`], {
    cwd: repoRoot,
    maxBuffer: 64 * 1024 * 1024,
  }).toString();

const SERVER_CLI = 'crates/onnx-genai-server/src/cli.rs';

/** snake_case field name -> the kebab-case flag clap derives from it. */
const toKebab = (field) => field.replace(/_/g, '-');

/**
 * Parse the clap `ServeArgs` struct into the set of flags the server accepts.
 *
 * Handles the three forms present in the file:
 *   #[arg(long)]                     -> derived from the field name
 *   #[arg(long = "explicit-name")]   -> taken verbatim
 *   #[arg(                           -> multi-line attribute
 *       long, env = "...",
 *   )]
 * and records `#[cfg(feature = "...")]` gating, because a flag that only
 * exists under a non-default feature is not one a README may hand a visitor
 * unconditionally.
 */
export function parseServerFlags(source) {
  const lines = source.split('\n');
  const flags = new Map();

  for (let i = 0; i < lines.length; i += 1) {
    if (!/^\s*#\[arg\(/.test(lines[i])) continue;

    // Accumulate the attribute, which may span lines until its closing `)]`.
    let attr = '';
    let j = i;
    let depth = 0;
    do {
      attr += lines[j];
      depth += (lines[j].match(/\(/g) || []).length;
      depth -= (lines[j].match(/\)/g) || []).length;
      j += 1;
    } while (depth > 0 && j < lines.length);

    if (!/\blong\b/.test(attr)) continue;

    // A `#[cfg(feature = "x")]` may sit immediately above the `#[arg(..)]`.
    let feature = null;
    const cfgMatch = /#\[cfg\(feature\s*=\s*"([^"]+)"\)\]/.exec(lines[i - 1] || '');
    if (cfgMatch) [, feature] = cfgMatch;

    const explicit = /long\s*=\s*"([^"]+)"/.exec(attr);
    if (explicit) {
      flags.set(`--${explicit[1]}`, { feature });
      i = j - 1;
      continue;
    }

    // Otherwise the flag name comes from the field the attribute decorates.
    for (let k = j; k < Math.min(j + 4, lines.length); k += 1) {
      const field = /^\s*(?:pub(?:\([^)]*\))?\s+)?([a-z_][a-z0-9_]*)\s*:/.exec(lines[k]);
      if (field) {
        flags.set(`--${toKebab(field[1])}`, { feature });
        break;
      }
    }
    i = j - 1;
  }

  return flags;
}

/**
 * Extract the flags used in commands that actually INVOKE the server binary.
 *
 * A command begins on a line that runs `onnx-genai-server` -- as a path
 * (`./target/release/onnx-genai-server`) or a bare command -- and continues
 * across shell line-continuations. Lines that merely NAME the binary as an
 * argument to another tool (`cargo build -p onnx-genai-server`) are not
 * invocations and are skipped.
 */
export function serverFlagsUsedIn(text) {
  const lines = text.split('\n');
  const used = [];

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];

    const names = /(^|[\s`"'(=])(?:[\w./${}-]*\/)?onnx-genai-server(\s|$|\\)/.test(line);
    const commented = /^\s*(?:#|\/\/|\||>)/.test(line);
    if (!names || commented) continue;

    // `cargo run -p onnx-genai-server -- --model x` IS an invocation, and the
    // previous blanket `cargo`/`-p` exclusion silently dropped every one of
    // them. That exclusion was added to stop `cargo build --release -p
    // onnx-genai-server` attributing `--release` to the server -- a real false
    // positive -- but it over-corrected into a false NEGATIVE across the
    // benchmark runbooks, where `--model-id` and `--addr` sit unaudited after a
    // passthrough separator. Widening the corpus is what exposed this: the
    // files became visible and the matcher still declined to look at them.
    //
    // The precise rule, which needs no exclusion at all: in a cargo passthrough
    // EVERYTHING BEFORE THE BARE `--` IS CARGO'S AND EVERYTHING AFTER IT IS THE
    // SERVER'S. That is the shell's own rule, so it cannot drift from reality.
    const cargoPassthrough =
      /\bcargo\s+run\b/.test(line) && /-p\s+onnx-genai-server\b/.test(line);
    const namesOnly =
      (/\b(?:cargo|rustup)\b/.test(line) || /-p\s+onnx-genai-server\b/.test(line)) &&
      !cargoPassthrough;
    if (namesOnly) continue;

    const block = [{ n: i + 1, text: line }];
    let k = i;
    while (/\\\s*$/.test(lines[k]) && k + 1 < lines.length) {
      k += 1;
      block.push({ n: k + 1, text: lines[k] });
    }
    i = k;

    const command = block.map((b) => b.text).join('\n').trim();

    // In a passthrough we are not scanning until we have crossed the separator.
    let scanning = !cargoPassthrough;
    for (const { n, text: lineText } of block) {
      let segment = lineText;
      if (!scanning) {
        const sep = lineText.match(/(^|\s)--(?=\s|\\\s*$|$)/);
        if (!sep) continue;
        scanning = true;
        segment = lineText.slice(sep.index + sep[0].length);
      }
      for (const m of segment.matchAll(/(?<![\w-])--([a-z][a-z0-9-]*)/g)) {
        used.push({ flag: `--${m[1]}`, line: n, command });
      }
    }
  }

  return used;
}

/**
 * Every tracked surface that could hand a visitor a server command.
 *
 * THIS USED TO BE A HARDCODED LIST OF SIX PATHS. That is the same defect this
 * repository's test runner warns about in its own header -- A HARDCODED LIST
 * STOPS COVERING WHATEVER WAS ADDED LAST, WHICH IS ALWAYS THE THING MOST LIKELY
 * TO BE WRONG -- and I shipped it here while writing that warning there. The
 * repository tracks 581 .md/.sh files; this guard was reading six of them, so
 * it covered roughly 1% of the surface and reported green over the rest.
 *
 * DISCOVERY IS SAFE HERE BECAUSE THE MATCHER IS PRECISE, AND THAT IS THE WHOLE
 * DESIGN: `serverFlagsUsedIn` only counts flags inside a line that actually
 * INVOKES `onnx-genai-server`, so widening the corpus cannot produce false
 * positives from prose, from other binaries' flags, from CSS custom properties
 * or from BEM modifiers. A precise predicate is what buys you a broad corpus.
 *
 * The `git grep -l` prefilter is EXACT rather than a heuristic: an invocation
 * requires the literal `onnx-genai-server` on the line, so a file that does not
 * contain that string cannot contain an invocation. It is an optimisation that
 * cannot change the answer.
 */
function surfaceFiles() {
  let listed = '';
  try {
    listed = execFileSync(
      'git',
      ['grep', '-l', '--fixed-strings', 'onnx-genai-server', 'HEAD', '--', '.'],
      { cwd: repoRoot, maxBuffer: 64 * 1024 * 1024 },
    ).toString();
  } catch {
    listed = '';
  }

  return listed
    .split('\n')
    .filter(Boolean)
    // `git grep HEAD` prefixes every path with `HEAD:`.
    .map((l) => l.replace(/^HEAD:/, ''))
    .filter((p) => /\.(md|sh)$/.test(p) || (/\.js$/.test(p) && !/\.test\.js$/.test(p)))
    // NAMED REGION, NOT A VOCABULARY. `.squad/` is this crew's internal
    // coordination state -- charters, decision archives, agent histories. No
    // visitor is ever shown one, so a stale command in an agent's notebook is
    // not a defect in the demo. Auditing it would let any agent's append-only
    // history redden the release branch, and a guard that random notes can
    // redden is a guard somebody eventually deletes.
    //
    // THIS IS A SCOPE LIMIT AND IT IS PRINTED ON PASSING RUNS BELOW, because an
    // OK that hides what it declined to look at does the lying for you.
    .filter((p) => !p.startsWith('.squad/'));
}

/** Paths deliberately outside the audit, reported alongside every result. */
const EXCLUDED_REGION = '.squad/ (internal crew state, never shown to a visitor)';

test('the server CLI parses into a non-empty, plausible flag set', () => {
  const flags = parseServerFlags(read(SERVER_CLI));

  // Guard the PARSER, not just the flags. A regex that silently matches
  // nothing would make every downstream assertion vacuously pass -- the
  // failure mode where a green check means the check stopped working.
  assert.ok(
    flags.size >= 10,
    `parsed only ${flags.size} flags from cli.rs; the parser has probably broken`,
  );
  for (const known of ['--model', '--addr', '--max-batch', '--demo-assets-dir']) {
    assert.ok(flags.has(known), `expected ${known} in the parsed flag set`);
  }
});

test('the invocation matcher ignores commands that merely NAME the server', () => {
  // Regression guard for the false positive this file was almost shipped with.
  const notAnInvocation = 'cargo build --release -p onnx-genai-server';
  assert.deepEqual(serverFlagsUsedIn(notAnInvocation), []);

  const isAnInvocation = './target/release/onnx-genai-server --model ./m --addr 127.0.0.1:8123';
  assert.deepEqual(
    serverFlagsUsedIn(isAnInvocation).map((u) => u.flag),
    ['--model', '--addr'],
  );

  // A cargo passthrough is an invocation, and the separator is the whole rule.
  // BOTH directions are pinned here because each one alone is satisfiable by a
  // broken matcher: dropping the block entirely passes the first, and scanning
  // the whole line passes the second.
  const passthrough =
    'ONNX_GENAI_EP=cpu cargo run --release --features cuda -p onnx-genai-server -- \\\n' +
    '  --model models/q --model-id q --addr 127.0.0.1:8080';
  const seen = serverFlagsUsedIn(passthrough).map((u) => u.flag);
  assert.deepEqual(seen, ['--model', '--model-id', '--addr']);
  for (const cargosOwn of ['--release', '--features']) {
    assert.ok(!seen.includes(cargosOwn), `${cargosOwn} belongs to cargo, not to the server`);
  }

  // Line continuations must be followed, or a multi-line README command would
  // be audited only as far as its first line.
  const continued = './target/release/onnx-genai-server \\\n  --model ./m \\\n  --max-batch 4';
  assert.deepEqual(
    serverFlagsUsedIn(continued).map((u) => u.flag),
    ['--model', '--max-batch'],
  );
});

test('every flag in a documented server command exists in cli.rs', () => {
  const flags = parseServerFlags(read(SERVER_CLI));
  const problems = [];

  for (const file of surfaceFiles()) {
    for (const use of serverFlagsUsedIn(read(file))) {
      const known = flags.get(use.flag);
      if (!known) {
        problems.push(
          `${file}:${use.line} uses ${use.flag}, ` +
            `which is not an argument of onnx-genai-server`,
        );
      } else if (known.feature) {
        problems.push(
          `${file}:${use.line} uses ${use.flag}, which only ` +
            `exists under the non-default feature "${known.feature}"`,
        );
      }
    }
  }

  assert.deepEqual(
    problems,
    [],
    `documented server flags that the binary does not accept:\n  ${problems.join('\n  ')}`,
  );
});

test('the parser does not accept a SUPERSET of the real flags', () => {
  // @732c7548's drift test stayed green against an injected bogus flag: it
  // collected both a field's name AND its `long = ".."` override as valid
  // spellings, but clap uses the override INSTEAD of the field name. A test
  // asserting a superset of reality cannot fail, and sits there looking like
  // coverage indefinitely.
  //
  // The real cli.rs currently contains no `long = ".."` override, so this
  // branch of the parser is never exercised by the production input -- which
  // is precisely why it needs a synthetic case. Untested code that only runs
  // once someone edits cli.rs is a trap armed for a future contributor.
  const synthetic = `
    pub struct ServeArgs {
        /// Explicit override.
        #[arg(long = "cors-allow-origin", env = "X")]
        pub cors_allow_origins: Vec<String>,

        /// Derived from the field name.
        #[arg(long)]
        pub max_batch: usize,

        /// Short-only: clap exposes no long flag at all.
        #[arg(short)]
        pub verbose: bool,

        /// A positional argument, not a flag.
        pub model_path: PathBuf,
    }
  `;

  const flags = parseServerFlags(synthetic);

  assert.ok(flags.has('--cors-allow-origin'), 'the explicit override must be accepted');
  assert.ok(
    !flags.has('--cors-allow-origins'),
    'the FIELD NAME must be rejected when an explicit long override is present — ' +
      'accepting both is the superset bug that cannot fail',
  );
  assert.ok(flags.has('--max-batch'), 'a bare `long` must derive the flag from the field');
  assert.ok(!flags.has('--verbose'), 'a short-only arg exposes no long flag');
  assert.ok(!flags.has('--model-path'), 'a positional is not a flag');
  assert.equal(flags.size, 2, `expected exactly 2 flags, got ${[...flags.keys()].join(', ')}`);
});

test('a cfg-gated flag is reported as unavailable, not as available', () => {
  // `--native-device` exists only under the `native-backend` feature. A README
  // that hands a visitor a cfg-gated flag is wrong for every default build,
  // and the failure is an unrecognised-argument exit at first run.
  const synthetic = `
    pub struct ServeArgs {
        #[cfg(feature = "native-backend")]
        #[arg(long, env = "ONNX_GENAI_NATIVE_DEVICE")]
        pub native_device: Option<NativeDecodeDevice>,
    }
  `;
  const flags = parseServerFlags(synthetic);
  assert.equal(flags.get('--native-device')?.feature, 'native-backend');
});

test('a documented command is audited at all — the audit has real subject matter', () => {
  // Without this, deleting every command from the README would make the test
  // above pass perfectly. An audit that can be satisfied by having nothing to
  // audit is not an audit.
  //
  // THE COUNT ALONE IS NOT ENOUGH NOW THAT THE CORPUS IS DISCOVERED. A failing
  // `git grep` is caught here and returns [], and an empty corpus is
  // byte-indistinguishable from a repository with no invocations in it -- the
  // vacuous green this crew has hit repeatedly tonight. So assert the corpus by
  // NAMED MEMBER: these two files are the demo's front door and its launcher,
  // and if discovery ever stops finding them it is discovery that broke.
  const files = surfaceFiles();
  for (const required of [
    'examples/serving-dashboard/README.md',
    'examples/serving-dashboard/run-demo.sh',
  ]) {
    assert.ok(
      files.includes(required),
      `corpus discovery did not find ${required}; the prefilter or git grep has broken`,
    );
  }

  const total = files.reduce((n, f) => n + serverFlagsUsedIn(read(f)).length, 0);
  assert.ok(total > 0, 'found no server invocations in any demo surface to audit');

  // State the population, on green as well as on red. Every green on this board
  // is a claim about a population and almost none of them name it.
  console.log(
    `      scope: ${files.length} tracked surfaces at HEAD, ${total} server ` +
      `invocations audited; EXCLUDED: ${EXCLUDED_REGION}`,
  );
});
