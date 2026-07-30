// Copyright (c) Microsoft Corporation.
//
// The telemetry capture recorder.
//
// A panel binding is a CLAIM THAT A PRODUCER EMITS SOMETHING. Until these
// fixtures existed nothing in this repository verified that claim against a
// producer: `telemetry-provenance.js` says a field is `MEASURED`, a panel binds
// it, every test goes green, and the field is absent from the wire forever. The
// visitor sees a permanently degraded panel and the suite never notices.
//
// This script records what the servers actually emit so `check-binding-liveness.test.js`
// can hold every binding to it OFFLINE, in CI, with no server running.
//
// Refresh with:
//   node capture-telemetry-fixtures.mjs                       (both origins)
//   node capture-telemetry-fixtures.mjs --origin dynamic
//
// ---------------------------------------------------------------------------
// WHAT THE MANIFEST CAN AND CANNOT PROVE. Read this before trusting a capture.
//
// It records the REPOSITORY's HEAD at capture time. That is NOT the identity of
// the binary that answered, and nothing in this repo can supply that: no
// `onnx-genai-server` binary carries its commit. There is no vergen, no
// `env!("GIT_*")`, no stamping build.rs, and no version endpoint -- the only
// 40-hex string in the binary is rustc's own. A port answering 200 proves a
// server is there, not which one.
//
// So `repo_head` means "the tree the RECORDER ran from", and it is recorded
// because a capture without provenance is the stale-baseline trap waiting
// eighteen months -- not because it identifies the producer.
//
// AND THERE IS NO FRESHNESS SIGNAL ON THE WIRE AT ALL. I looked for one to put
// here and there is none: `/v1/status` carries no uptime, no start time and no
// build id, and neither does any other endpoint. So the staleness of a capture
// is UNKNOWABLE FROM THE SERVER SIDE -- `captured_at` is the recorder's clock
// and nothing corroborates it. That is recorded in the manifest as a stated
// absence rather than left as a null field, because a null would read as "we
// forgot to fill this in" instead of "this does not exist".
// ---------------------------------------------------------------------------

import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { ENDPOINTS, TEXT_ENDPOINTS } from './telemetry-provenance.js';

const HERE = dirname(fileURLToPath(import.meta.url));

/** Where captures live. The guard reads this directory and nothing else. */
export const CAPTURE_DIR = join(HERE, 'fixtures', 'captures');

/**
 * The demo's two origins, by the same labels `PROVENANCE.byOrigin` uses. These
 * ports are the topology `run-demo.sh` actually serves; capturing from anything
 * else would record a tree we do not ship.
 */
export const ORIGINS = Object.freeze({
  scatter: 'http://127.0.0.1:8133',
  dynamic: 'http://127.0.0.1:8134',
});

/** Every endpoint the store polls, captured whether or not a panel binds it. */
const CAPTURED_ENDPOINTS = Object.freeze(Object.values(ENDPOINTS));

/**
 * Fetch one endpoint, returning its body and status. A non-200 is recorded
 * rather than thrown: "this endpoint 404s on this origin" is a fact the guard
 * needs, and dropping it would make an absent endpoint indistinguishable from
 * an endpoint nobody captured.
 *
 * @param {string} base
 * @param {string} endpoint
 */
async function capture(base, endpoint) {
  const response = await fetch(`${base}${endpoint}`, {
    signal: AbortSignal.timeout(5000),
  });
  const text = await response.text();
  if (!response.ok) {
    return { status: response.status, body: null, error: text.slice(0, 200) };
  }
  if (TEXT_ENDPOINTS.includes(endpoint)) {
    return { status: response.status, body: text };
  }
  return { status: response.status, body: JSON.parse(text) };
}

/** The recorder's own tree. See the header: this is not the server's identity. */
function repoHead() {
  return execFileSync('git', ['rev-parse', 'HEAD'], {
    cwd: HERE,
    encoding: 'utf8',
  }).trim();
}

/**
 * Record every endpoint on one origin.
 *
 * @param {string} label
 * @param {string} base
 */
export async function captureOrigin(label, base) {
  /** @type {Record<string, object>} */
  const endpoints = {};
  for (const endpoint of CAPTURED_ENDPOINTS) {
    endpoints[endpoint] = await capture(base, endpoint);
  }
  return { label, base, endpoints };
}

async function main() {
  const argv = process.argv.slice(2);
  const only = argv.includes('--origin') ? argv[argv.indexOf('--origin') + 1] : null;
  const selected = only ? { [only]: ORIGINS[only] } : ORIGINS;
  if (only && !ORIGINS[only]) {
    throw new Error(`unknown origin '${only}'; expected one of ${Object.keys(ORIGINS).join(', ')}`);
  }

  mkdirSync(CAPTURE_DIR, { recursive: true });
  const capturedAt = new Date().toISOString();
  const head = repoHead();
  /** @type {Record<string, object>} */
  const manifestOrigins = {};

  for (const [label, base] of Object.entries(selected)) {
    const result = await captureOrigin(label, base);
    writeFileSync(
      join(CAPTURE_DIR, `${label}.json`),
      `${JSON.stringify(result, null, 2)}\n`,
    );
    const health = result.endpoints[ENDPOINTS.HEALTH]?.body;
    manifestOrigins[label] = {
      base,
      model_id: health?.model ?? null,
      endpoints_2xx: Object.entries(result.endpoints)
        .filter(([, v]) => v.status >= 200 && v.status < 300)
        .map(([k]) => k),
      endpoints_not_2xx: Object.entries(result.endpoints)
        .filter(([, v]) => v.status < 200 || v.status >= 300)
        .map(([k, v]) => `${k} -> ${v.status}`),
    };
    console.log(`captured ${label} (${base}): ${Object.keys(result.endpoints).length} endpoints`);
  }

  writeFileSync(
    join(CAPTURE_DIR, 'manifest.json'),
    `${JSON.stringify(
      {
        captured_at: capturedAt,
        repo_head: head,
        no_server_freshness_signal:
          'Checked, not assumed: no endpoint exposes uptime, start time or build id, ' +
          'so nothing on the wire can date this capture. captured_at is the ' +
          "recorder's clock and nothing corroborates it.",
        repo_head_is_not_the_server_identity:
          'No onnx-genai-server binary carries its commit: no vergen, no env!("GIT_*"), ' +
          'no stamping build.rs, no version endpoint. This SHA is the tree the RECORDER ' +
          'ran from. It does not tell you which binary answered.',
        conditions:
          'Idle-to-light load on a developer machine shared by concurrent agents. ' +
          'Fields that only appear under sustained load may legitimately be absent ' +
          'here; that is what DECLARED_ABSENT exists for.',
        recorder: 'examples/serving-dashboard/capture-telemetry-fixtures.mjs',
        origins: manifestOrigins,
      },
      null,
      2,
    )}\n`,
  );
  console.log(`manifest written at repo_head ${head.slice(0, 8)}`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await main();
}
