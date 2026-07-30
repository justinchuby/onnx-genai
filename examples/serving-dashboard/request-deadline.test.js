// Copyright (c) Microsoft Corporation.
//
// Tests for the shared request deadline, and for the census that makes it
// unrepeatable.
//
// THE DEFECT THIS CLOSES WAS NOT "NO TIMEOUT EXISTS". A correct timeout was
// written in telemetry-store.js and shipped. The defect was that it did not
// TRAVEL: app.js probed /health with a bare `fetch` on the boot path, before
// any panel mounts, and never heard about it. One of two call sites learned
// the lesson.
//
// So the interesting test here is not the behavioural one — it is the census
// at the bottom. A fix that cannot be imported applies exactly once, and a
// fix that is not enforced applies exactly until the next call site.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  DEFAULT_REQUEST_TIMEOUT_MS,
  RequestTimeoutError,
  fetchWithDeadline,
} from './request-deadline.js';

const demoDir = dirname(fileURLToPath(import.meta.url));

/**
 * A server that accepts the request and never answers — the failure mode the
 * reconnect ladder cannot see, because it handles servers that FAIL and this
 * one STALLS.
 *
 * It settles only when the abort signal fires, which is what `fetch` does with
 * a signal attached. A fake that ignored the signal would HANG this file
 * rather than fail it — which is exactly what a browser does when the deadline
 * is removed, and is why no fixture in this repository could observe the bug.
 */
function stallingFetch({ onRequest } = {}) {
  return (_input, options = {}) => {
    onRequest?.();
    return new Promise((_resolve, reject) => {
      const { signal } = options;
      if (!signal) {
        reject(new Error('fetch was called with no abort signal — the request has no deadline'));
        return;
      }
      signal.addEventListener('abort', () => reject(signal.reason ?? new Error('aborted')));
    });
  };
}

test('a stalling server produces a rejection instead of a hang', async () => {
  const started = Date.now();
  await assert.rejects(
    () => fetchWithDeadline('http://stalled.invalid/health', {
      fetchImpl: stallingFetch(),
      timeoutMs: 40,
    }),
    RequestTimeoutError,
    'a request that never settles must reject, not pend forever',
  );
  // Bounded, not exact: the assertion is that it FINISHED, which is the whole
  // property. A wall-clock threshold tight enough to be interesting would be a
  // flake on a loaded machine.
  assert.ok(Date.now() - started < 5_000, 'the deadline did not fire');
});

test('the timeout error says the SERVER went quiet, not that we aborted', async () => {
  // The runtime's own wording is "This operation was aborted", which reads as
  // though the dashboard did something wrong. A visitor needs to know which
  // side went silent, because that distinguishes a hung generation from a dead
  // port — and those have different remedies on stage.
  const error = await fetchWithDeadline('http://stalled.invalid/health', {
    fetchImpl: stallingFetch(),
    timeoutMs: 20,
  }).catch((e) => e);

  assert.ok(error instanceof RequestTimeoutError);
  assert.match(error.message, /never replied/);
  assert.match(error.message, /20 ms/);
  assert.doesNotMatch(
    error.message,
    /operation was aborted/i,
    'the runtime abort text leaked to the visitor',
  );
});

test('a healthy server is not aborted — the control that must stay green', async () => {
  // Without this, "make everything time out immediately" would pass the suite
  // above. Trading a hang for a flake is the de-flaking spiral that disarmed
  // another guard on this branch tonight.
  const response = await fetchWithDeadline('http://healthy.invalid/health', {
    fetchImpl: async () => ({ ok: true, status: 200 }),
    timeoutMs: 20,
  });
  assert.equal(response.ok, true);
});

test('a real rejection passes through unchanged, not relabelled as a timeout', async () => {
  // A refused connection already worked before the deadline existed. If it now
  // arrives as a RequestTimeoutError, we have made a working failure mode
  // report the wrong cause — and the visitor is told to wait for a server that
  // is not running.
  const refused = new Error('connection refused');
  const error = await fetchWithDeadline('http://refused.invalid/health', {
    fetchImpl: async () => {
      throw refused;
    },
    timeoutMs: 5_000,
  }).catch((e) => e);

  assert.equal(error, refused);
  assert.ok(!(error instanceof RequestTimeoutError));
});

test('the caller options reach the underlying fetch untouched', async () => {
  let seen = null;
  await fetchWithDeadline('http://healthy.invalid/health', {
    fetchImpl: async (_input, init) => {
      seen = init;
      return { ok: true };
    },
    headers: { accept: 'application/json' },
    cache: 'no-store',
  });

  assert.deepEqual(seen.headers, { accept: 'application/json' });
  assert.equal(seen.cache, 'no-store');
  assert.ok(seen.signal, 'no signal was attached');
  // The two options this module consumes must NOT be forwarded — `fetch`
  // ignores unknown keys silently, so a leak here would never be noticed.
  assert.equal(seen.fetchImpl, undefined);
  assert.equal(seen.timeoutMs, undefined);
});

test('there is exactly one deadline value in the product', () => {
  // Two constants of the same name in two modules is how a duplicate
  // provenance key silently shipped a misnomer on this branch: JS raises no
  // error, it just picks one.
  assert.equal(typeof DEFAULT_REQUEST_TIMEOUT_MS, 'number');
  assert.ok(DEFAULT_REQUEST_TIMEOUT_MS > 0);

  const store = readFileSync(join(demoDir, 'telemetry-store.js'), 'utf8');
  assert.doesNotMatch(
    store,
    /^export const DEFAULT_REQUEST_TIMEOUT_MS/m,
    'telemetry-store.js declares its own deadline constant again; it must re-export the shared one',
  );
});

/** Every shipping module (no tests, no fixtures), as [relative path, source]. */
function shippingModules() {
  const found = [];
  const walk = (dir, prefix) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.name === 'node_modules' || entry.name === 'testing') continue;
      const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory()) walk(join(dir, entry.name), rel);
      else if (entry.name.endsWith('.js') && !entry.name.includes('.test.')) {
        found.push([rel, readFileSync(join(dir, entry.name), 'utf8')]);
      }
    }
  };
  walk(demoDir, '');
  return found;
}

// THE CENSUS, AND IT IS THE POINT OF THIS FILE.
//
// The behavioural tests above would all have passed on the night the bug
// shipped, because the module they exercise was already correct. What was
// missing was any statement that it is the ONLY way to make a request.
//
// The denominator is small enough to state exactly: at the time of writing
// there are two fetch sites in shipped dashboard JS, and one of them was
// unguarded for the entire life of the branch. The count is asserted non-zero
// so this cannot pass by finding nothing — a scan that matches no files and a
// tree with no defects are byte-identical from here.
test('every fetch in shipped dashboard code carries a deadline', () => {
  const offenders = [];
  let fetchSites = 0;

  for (const [path, src] of shippingModules()) {
    if (path === 'request-deadline.js') continue; // the module that owns the idiom

    // A bare `fetch(` or `fetchImpl(` that is not the guarded helper. The
    // helper's own name ends in `fetchWithDeadline`, so it is excluded by the
    // preceding-character check rather than by a name blocklist that would
    // rot.
    for (const match of src.matchAll(/(?<![\w.])(fetch|fetchImpl)\s*\(/g)) {
      fetchSites += 1;
      const line = src.slice(0, match.index).split('\n').length;
      offenders.push(`${path}:${line} — ${match[1]}(`);
    }
    for (const _ of src.matchAll(/fetchWithDeadline\s*\(/g)) fetchSites += 1;
  }

  assert.ok(
    fetchSites > 0,
    'found no request sites at all in shipped code; this census is broken, not the tree clean',
  );
  assert.deepEqual(
    offenders,
    [],
    `these requests can pend forever against a server that accepts the socket and never ` +
      `answers. Use fetchWithDeadline from request-deadline.js:\n  ${offenders.join('\n  ')}`,
  );
});

// The tests above prove the PRIMITIVE rejects. None of them proves the POLL
// LOOP survives, and that is the defect @f6527cc9 actually measured: against a
// stalling server the attempt count read 7,7,7,7,7 forever, while the control
// (a server that refuses the socket) climbed 9,11,11,13,13. A dead loop and a
// healthy one are indistinguishable from the rendered page -- it keeps showing
// its last good numbers with their original timestamps.
//
// This drives the real store through repeated cycles and asserts the attempt
// count GROWS. Deliberately not a threshold: a threshold encodes today's timing
// and goes flaky on a loaded machine, and tonight's load average reached 121.
// Growth is the property; any particular count is an accident of the box.
test('the poll loop survives a stalling server — attempts GROW, they never freeze', async () => {
  const { createTelemetryStore } = await import('./telemetry-store.js');

  let requests = 0;
  const store = createTelemetryStore({
    baseUrl: 'http://stalled.invalid',
    requestTimeoutMs: 20,
    fetchImpl: stallingFetch({ onRequest: () => { requests += 1; } }),
  });

  /** @type {number[]} */
  const ladder = [];
  for (let cycle = 0; cycle < 4; cycle += 1) {
    await store.pollOnce();
    ladder.push(requests);
  }

  // Anti-vacuity: a loop that never issued a request would produce [0,0,0,0]
  // and satisfy nothing below, but say so explicitly rather than relying on the
  // growth check to imply it.
  assert.ok(ladder[0] > 0, `the first cycle issued no requests at all (ladder ${ladder})`);

  for (let index = 1; index < ladder.length; index += 1) {
    assert.ok(
      ladder[index] > ladder[index - 1],
      `the poll loop stopped issuing requests after cycle ${index}: ${ladder}. ` +
        'pollInFlight is stuck true, the finally never ran, and the page will show ' +
        'its last good numbers forever while looking healthy.',
    );
  }
});
