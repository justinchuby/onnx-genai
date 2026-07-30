/**
 * THE REVERSE DIRECTION: a fact the server MEASURES, SERVES, and gets right —
 * that no pixel on the dashboard ever shows.
 *
 * Every honesty guard in this repository asks the same question in the same
 * direction: does this rendered number claim more than it measures? That
 * direction has thirteen instruments. The opposite population — fields we
 * measure correctly and never display — has had NONE, and it is invisible by
 * construction:
 *
 *   AN OVERCLAIM PUTS A WRONG NUMBER ON THE SCREEN, WHERE SOMEBODY SEES IT.
 *   AN UNDERCLAIM PUTS NOTHING ON THE SCREEN, AND NOTHING IS WHAT AN EMPTY
 *   PANEL ALREADY LOOKS LIKE.
 *
 * This found `batch_driver` and `batch_driver_detail` on /v1/status: served
 * since 1e1b2a82, referenced by zero shipped modules. It is the same defect as
 * the latency table requesting `latency.e2e_server_p50` — a key the store
 * cannot serve — while `metrics.e2e_latency` sits live and unrendered.
 * telemetry-key-namespace.test.js guards the panel->store half. This guards
 * the server->panel half, which nothing else reads.
 *
 * WHY IT MATTERS ON THE NIGHT: two of the four demo origins are per-request
 * drivers, 1 wide by construction, while advertising `batch_capacity: 4`. The
 * server can now say which driver is running. If nothing renders it, the
 * server gets fixed and the stage looks identical.
 *
 * The served surface is DERIVED from the Rust source at the shipping ref, never
 * hand-listed: a hand-listed expectation is a claim about a file, and this file
 * would then be auditing its own memory instead of the server.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
// git resolves pathspecs relative to the working directory. Running this from
// the dashboard directory with a repo-relative pathspec silently returns
// nothing, and an empty corpus makes every assertion below vacuously true.
const TOPLEVEL = execFileSync('git', ['rev-parse', '--show-toplevel'], {
  cwd: HERE,
  encoding: 'utf8',
}).trim();

const REF = 'HEAD';
const ROUTES_SRC = 'crates/onnx-genai-server/src/routes/mod.rs';

function git(...args) {
  return execFileSync('git', args, { cwd: TOPLEVEL, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
}

const routesSource = git('show', `${REF}:${ROUTES_SRC}`);

/**
 * The serialised field names of a `#[derive(Serialize)]` struct, with the type
 * of each, so containers can be expanded rather than reported as missing.
 *
 * @param {string} structName
 * @returns {Array<{name: string, type: string}>|null} null when absent
 */
function structFields(structName) {
  const start = routesSource.indexOf(`struct ${structName} {`);
  if (start < 0) return null;
  const end = routesSource.indexOf('\n}\n', start);
  const body = routesSource.slice(start, end === -1 ? undefined : end);

  const fields = [];
  let renamedTo = null;
  for (const raw of body.split('\n').slice(1)) {
    const line = raw.trim();
    const rename = line.match(/#\[serde\(rename\s*=\s*"([^"]+)"/);
    if (rename) {
      renamedTo = rename[1];
      continue;
    }
    if (!line || line.startsWith('//') || line.startsWith('#[')) continue;
    const field = line.match(/^(?:pub(?:\([^)]*\))?\s+)?([a-z_][a-z0-9_]*)\s*:\s*(.+?),?$/);
    if (!field) continue;
    fields.push({ name: renamedTo ?? field[1], type: field[2] });
    renamedTo = null;
  }
  return fields;
}

/** The innermost type argument, so `Option<Vec<ResourceTier>>` yields `ResourceTier`. */
function unwrapType(type) {
  let inner = type.trim();
  let previous = null;
  while (inner !== previous) {
    previous = inner;
    inner = inner.replace(/^(?:Option|Vec|Box|Arc)<(.+)>$/, '$1').trim();
  }
  return inner.split(',').pop().replace(/>+$/, '').trim();
}

/**
 * Served leaf field names for a route, expanding nested response structs.
 *
 * A container whose own name never appears in the dashboard is NOT a defect if
 * the panel renders its inner fields — `configured_limits` is exactly that, and
 * reporting it would be a false positive of the predicate rather than a finding
 * about the product.
 */
function servedLeafFields(structName, seen = new Set()) {
  if (seen.has(structName)) return [];
  seen.add(structName);
  const fields = structFields(structName);
  if (!fields) return [];

  const leaves = [];
  for (const field of fields) {
    const nested = structFields(unwrapType(field.type));
    if (nested) leaves.push(...servedLeafFields(unwrapType(field.type), seen));
    else leaves.push(field.name);
  }
  return leaves;
}

const PUBLIC_ROUTES = new Map([
  ['/v1/status', 'NodeStatus'],
  ['/v1/models', 'ModelObject'],
  ['/v1/resources', 'ResourcesResponse'],
]);

/**
 * Shipped dashboard JavaScript at the shipping ref.
 *
 * `.test.js` is a SUFFIX, not a directory, so a `dashboard/*.js` glob matches
 * every test file. A census that includes tests reports a field as rendered
 * because a FIXTURE mentions it — which is the exact opposite of this file's
 * question, since a fixture is the one place a served field is guaranteed to
 * appear whether or not any panel uses it.
 */
const shippedFiles = git('ls-tree', '-r', '--name-only', REF, 'examples/serving-dashboard')
  .split('\n')
  .filter((file) => /\.(?:js|mjs)$/.test(file))
  .filter((file) => !/\.test\.js$/.test(file) && !/\/testing\//.test(file));

const shippedSource = shippedFiles.map((file) => git('show', `${REF}:${file}`)).join('\n');

/**
 * Served fields that no shipped module names, and which are not exempt.
 *
 * `owned_by` is a `&'static str` OpenAI-compatibility constant, not a
 * measurement of this server — rendering it would state nothing about the run.
 * It is excluded by NAME AND REASON rather than by silence, because an
 * undocumented exclusion is indistinguishable from an oversight.
 */
const NOT_A_MEASUREMENT = new Set(['owned_by']);

function unrenderedFieldsOf(structName) {
  return servedLeafFields(structName)
    .filter((field) => !NOT_A_MEASUREMENT.has(field))
    .filter((field) => !shippedSource.includes(field));
}

/**
 * The pinned set, as an EXACT-SET RATCHET rather than an allowlist: adding a
 * served field that nothing renders fails, AND rendering one of these without
 * updating the pin also fails. A one-directional allowlist drains — each removal
 * is individually justified and the aggregate becomes a suppression.
 *
 * Every entry is a real, measured, correctly-served fact that no pixel shows.
 */
const KNOWN_UNRENDERED = [
  // ── /v1/status ────────────────────────────────────────────────────────────
  // The driver actually in use, and why. The demo cannot currently tell a
  // visitor which driver is running, while advertising batch_capacity: 4 — and
  // two of the four demo origins are per-request drivers, 1 wide by
  // construction. Rendering this PER PANE is the right shape: it makes each
  // pane state its own capability instead of leaving the visitor to infer a
  // comparison from geometry.
  'batch_driver',
  'batch_driver_detail',
  // Its three siblings (batch_utilization, batch_in_flight, batch_capacity) are
  // all rendered. A whole family displayed except one member is the strongest
  // available signal that this is an oversight rather than a decision.
  'batch_queued',
  // Which model the batch driver is running, as distinct from the node's name.
  'batch_model_id',

  // ── /v1/resources ─────────────────────────────────────────────────────────
  // Resolved ceilings. `disk_spill_bytes` IS rendered, so the struct is
  // reached and these two are a PARTIAL rendering, not an unused response.
  'vram_bytes',
  'host_ram_bytes',
  // DerivedKvBudget. `bytes` reads as rendered, so this is partial too.
  'total_pages',
  'max_total_tokens',
  'reserved_bytes',
  // ResourceTier, reached through vram/host_ram/disk_spill.
  'headroom',
  // SessionStatus, reached through NodeStatus.sessions.
  'priority',

  // -- /v1/status, build provenance ------------------------------------------
  // THESE TWO APPEARED WHEN THE CAPTURES WERE RE-RECORDED, AND THAT IS THE
  // FINDING. They are served by the binary this branch ships and were absent
  // from the previous captures, so the committed fixtures had fallen behind the
  // server: this guard was reasoning about a response shape the server had
  // already stopped sending. Recorded here rather than rendered because build
  // provenance is not a measurement any panel claims, and putting a commit SHA
  // on screen is a product call, not a plumbing one.
  //
  // Worth a look by whoever owns /v1/status: `build_dirty` is serialised as the
  // STRING "false", not the boolean false, so a client writing the obvious
  // `if (build_dirty)` reads every clean build as dirty. Not changed here --
  // it is a server-side serde question and outside this change.
  'build_sha',
  'build_dirty',

  // -- /v1/debug/kv -----------------------------------------------------------
  // NOT A REGRESSION, AND WORTH SAYING WHY, BECAUSE THE COUNT LOOKS LIKE ONE.
  // The FACT is on screen; this particular SPELLING of it is not. kv.pages_shared
  // used to be read from /v1/debug/kv's `kv_pages_shared`, and is now derived
  // from /v1/debug/kv/blocks' `pages_shared` instead, so no shipped module names
  // the older field any more. The block table was preferred because it reports
  // the window it actually scanned (`window.scanned`, `window.truncated`), so a
  // truncated read is visible rather than silently reported as a whole-pool
  // figure. Two endpoints serving one fact is a server-side tidy-up, not a
  // dashboard defect, and it is left to whoever owns /v1/debug/kv.
  'kv_pages_shared',
];

/**
 * KNOWN LIMITATION, stated because an undocumented one is indistinguishable
 * from a bug: a field counts as rendered when any shipped module NAMES it, so
 * short generic leaf names — `used`, `limit`, `bytes` — read as rendered from
 * any incidental mention, including a comment. The guard therefore UNDER-reports
 * on `/v1/resources`, whose leaves are the generic ones.
 *
 * It does NOT under-report on the fields this was built for: every /v1/status
 * name above is distinctive enough that a substring hit means a real reference.
 * The generous direction is deliberate — this guard exists to find facts nobody
 * asked for, and a stricter predicate would report plumbing as product defects.
 */
const UNDER_REPORTS_ON_GENERIC_NAMES = ['used', 'limit', 'bytes'];

test('the known-weak field names are still the generic ones, and still few', () => {
  // If the server grows more short generic names, this guard quietly loses
  // resolution. That is worth failing on, because losing resolution looks
  // exactly like the defect being fixed.
  for (const name of UNDER_REPORTS_ON_GENERIC_NAMES) {
    assert.ok(name.length <= 6, `${name} is not a short generic name — reclassify it`);
    assert.equal(shippedSource.includes(name), true, `${name} no longer reads as rendered`);
  }
});

test('CAN RUN: the Rust source and the shipped corpus are both readable', () => {
  assert.ok(routesSource.length > 5000, `routes/mod.rs came back ${routesSource.length} bytes`);
  assert.ok(
    shippedFiles.length >= 20,
    `only ${shippedFiles.length} shipped JS files — the ls-tree pathspec is resolving wrong`,
  );
  assert.ok(shippedSource.length > 100_000, `corpus is only ${shippedSource.length} bytes`);

  // The predicate must discriminate in BOTH directions, or a corpus that
  // matches everything and one that matches nothing both read as a pass.
  assert.equal(shippedSource.includes('kv_pages_used'), true, 'a rendered field reads as missing');
  assert.equal(shippedSource.includes('zz_nosuch_field_c8'), false, 'the corpus matches anything');
});

test('every public route struct is still found in the Rust source', () => {
  for (const [route, structName] of PUBLIC_ROUTES) {
    const fields = structFields(structName);
    assert.ok(fields, `${structName} (${route}) is gone from ${ROUTES_SRC} — this guard is blind`);
    assert.ok(fields.length >= 3, `${structName} parsed only ${fields.length} fields`);
  }
});

test('a container field is not reported missing when its inner fields render', () => {
  // Regression pin for the predicate's own false positive. `configured_limits`
  // appears nowhere in the dashboard, but vram/host_ram/disk_spill all render.
  // Reporting the container would be a finding about the parser, not the panel.
  const leaves = servedLeafFields('ResourcesResponse');
  assert.ok(!leaves.includes('configured_limits'), 'the container leaked into the leaf set');
  assert.ok(leaves.includes('vram'), `expansion failed — leaves were ${leaves.join(' ')}`);
});

test('no served measurement is left unrendered, beyond the pinned set', () => {
  const unrendered = [];
  for (const structName of PUBLIC_ROUTES.values()) unrendered.push(...unrenderedFieldsOf(structName));

  assert.deepEqual(
    [...new Set(unrendered)].sort(),
    KNOWN_UNRENDERED.slice().sort(),
    'The set of served-but-unrendered fields changed.\n\n' +
      'MORE: the server measures and serves a fact that no pixel shows. This is ' +
      'the silent half of the honesty problem — an overclaim puts a wrong number ' +
      'where somebody sees it, an underclaim puts nothing where nothing is ' +
      'already expected. Render it, or pin it here with the reason.\n\n' +
      'FEWER: one was rendered. Remove it from KNOWN_UNRENDERED in the same ' +
      'commit so the pin always states the true remaining size.\n\n' +
      'A field counts as rendered if any shipped module NAMES it. That is ' +
      'deliberately generous: this guard finds facts nobody asked for, and a ' +
      'stricter predicate would report plumbing changes as product defects.',
  );
});
