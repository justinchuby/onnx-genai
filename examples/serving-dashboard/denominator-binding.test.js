/**
 * D239 / D258 TRIPWIRE — THE DENOMINATOR MUST BE READ, NEVER TYPED.
 *
 * `max_batch` defaults to 4 (cli.rs:76) and `page_size` is asserted to be 16
 * (kv/src/telemetry.rs:364). So on every machine we will ever demo on, the
 * FORBIDDEN HARDCODE AND THE CORRECT VALUE ARE THE SAME NUMBER. A panel that
 * types `4` renders identically to one that reads `scheduler.max_batch`:
 * review cannot see it, a screenshot cannot see it, and a visual check passes.
 * It only becomes visible on a machine launched with `--max-batch 8`, which is
 * to say: in front of someone else.
 *
 * This suite exists because `dashboard/scheduling.test.js:247-258` — the test
 * that certifies occupancy rendering — supplies `scheduler.max_batch: 4` and
 * asserts `/Batch occupancy 3 of 4 slots/`. Both halves are correct and the
 * pair CANNOT DISCRIMINATE: it passes identically whether the panel read the
 * field or typed the literal, because the fixture equals the default. That is
 * D261 in miniature — a green assertion that cannot redden is a reassurance
 * machine — and it is not a criticism of that suite, which tests rendering and
 * tests it well. Nothing was checking BINDING. Now something is.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const DASHBOARD = join(HERE, 'dashboard');

/** The field the batch denominator must come from, not a number. */
const DENOMINATOR_FIELD = 'scheduler.max_batch';

/** Strip line and block comments: prose may discuss `4`, code may not use it. */
function executableSource(text) {
  return text.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
}

function dashboardModules() {
  return readdirSync(DASHBOARD)
    .filter((f) => f.endsWith('.js') && !f.endsWith('.test.js'))
    .map((f) => ({ name: f, text: readFileSync(join(DASHBOARD, f), 'utf8') }));
}

describe('the batch denominator is bound, never typed', () => {
  it('found dashboard modules to check', () => {
    // ANTI-VACUITY. Every other assertion here is a NEGATIVE one, and a
    // negative assertion over an empty set is the most confident lie a test
    // can tell. If the directory moves, this fails first and says so.
    const modules = dashboardModules();
    assert.ok(
      modules.length >= 5,
      `expected the dashboard module directory at ${DASHBOARD}; found ${modules.length} modules`,
    );
  });

  it('reads the limit from the telemetry store', () => {
    const occupancyPanels = dashboardModules().filter((m) =>
      /renderOccupancy|batch occupancy/i.test(m.text),
    );

    assert.ok(
      occupancyPanels.length > 0,
      'no module renders batch occupancy; if the panel was renamed, this ' +
        'tripwire is pointing at nothing and must be re-aimed rather than deleted',
    );

    for (const panel of occupancyPanels) {
      assert.ok(
        panel.text.includes(DENOMINATOR_FIELD),
        `${panel.name} renders batch occupancy but never reads '${DENOMINATOR_FIELD}'. ` +
          'The denominator must arrive from the payload. A typed limit is correct ' +
          'on this machine and wrong on any server started with --max-batch.',
      );
    }
  });

  it('assigns no numeric literal into a batch limit', () => {
    // The mutation this exists to catch, stated exactly (AC85):
    //   const maxBatch = 4;                       -> RED
    //   const maxBatch = { value: 4 };            -> RED
    //   batchSize.value / 4                       -> RED
    // and the panel goes on rendering `3 of 4` perfectly through all three.
    const offenders = [];
    for (const { name, text } of dashboardModules()) {
      const source = executableSource(text);
      const patterns = [
        /\b(?:max_?[Bb]atch|batch_?[Cc]apacity|limit)\s*[:=]\s*\{?\s*(?:value\s*:\s*)?\d+/g,
        /\bbatch[A-Za-z]*\.?value?\s*\/\s*\d+/g,
      ];
      for (const pattern of patterns) {
        for (const hit of source.match(pattern) ?? []) {
          offenders.push(`${name}: ${hit.trim()}`);
        }
      }
    }

    assert.deepEqual(
      offenders,
      [],
      'A batch limit is assigned from a numeric literal. It renders correctly ' +
        'today because cli.rs:76 defaults --max-batch to 4, which is why no ' +
        'screenshot and no reviewer will ever catch this.\nFOUND:\n  ' +
        offenders.join('\n  '),
    );
  });
});
