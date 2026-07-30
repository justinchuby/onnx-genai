import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';

import { SCENARIOS, CUT_SCENARIOS } from './scenario-origins.js';

/**
 * D282 — EVERY HONESTY MECHANISM WE BUILT INSPECTS A FIELD. NONE INSPECTS A ROUTE.
 *
 * The provenance envelope, the five-state vocabulary, the classification
 * taxonomy, `field-keys.test.js`, the citation auditor -- all of them inspect
 * values INSIDE a page the visitor has already chosen. THE CHOICE ITSELF IS
 * UNGOVERNED, and it is made before the visitor sees a single number.
 *
 * The concrete defect (@376a0297's AC190, executed here rather than quoted):
 * `run-demo.sh` prints `?scenario=prefix-cache` as one of three co-equal
 * headline URLs, and `prefix-cache` was deliberately cut. `currentScenarioId`
 * falls through to a local default, so the operator is not shown an error --
 * they are shown a beautiful, fully honest, correctly labelled page for a
 * scenario they did not ask for. A 404 WOULD HAVE BEEN KINDER. Every field on
 * that page carries its correct provenance badge; the page is telling the
 * truth about paged KV to someone who clicked prefix caching, and nothing on
 * it says so. That is a SILENT SUBSTITUTION -- the exact failure our whole
 * apparatus exists to refuse -- occurring one layer above where the apparatus
 * operates.
 *
 * The cut was enforced in JavaScript and undone in Bash. `scenario-origins.js`
 * is the file that got it right: its `CUT_SCENARIOS` entry deliberately has no
 * `id` field, commented "an id is what makes a scenario addressable, and this
 * one must not be addressable." A DECISION IS ONLY AS ENFORCED AS ITS
 * LEAST-AUDITED LANGUAGE, and every instrument we own is single-language.
 * This guard is the first one that reads a shell script and a markdown file
 * against a JavaScript source of truth.
 *
 * Scope is deliberately OPERATOR-FACING SHIPPED FILES ONLY. Design notes,
 * specs and review documents QUOTE the broken URL in order to retire it --
 * a guard that reddened on its own explanation would be reworded away within
 * a day (@73e77d95's claim-versus-explanation split).
 */
describe('every advertised scenario route resolves to a real scenario', () => {
  const OPERATOR_FACING = ['run-demo.sh', 'README.md', 'QA-PLAN.md'];

  /** Tracked-only: an untracked orphan on one desk is not a thing we ship. */
  function trackedOperatorFiles() {
    const tracked = execFileSync('git', ['ls-files', ...OPERATOR_FACING], {
      cwd: import.meta.dirname,
      encoding: 'utf8',
    })
      .split('\n')
      .filter(Boolean);
    return tracked;
  }

  function advertisedRoutes() {
    const hits = [];
    for (const file of trackedOperatorFiles()) {
      const text = readFileSync(new URL(`./${file}`, import.meta.url), 'utf8');
      text.split('\n').forEach((line, i) => {
        for (const m of line.matchAll(/scenario=([a-z0-9-]+)/g)) {
          hits.push({ file, line: i + 1, id: m[1] });
        }
      });
    }
    return hits;
  }

  it('finds the routes it is meant to audit', () => {
    // ANTI-VACUITY. A pattern that matched nothing would certify every launcher
    // forever, and would be byte-identical to a clean run.
    const routes = advertisedRoutes();
    assert.ok(
      routes.length >= 2,
      `expected several advertised scenario routes; found ${routes.length}. ` +
        'If the launcher stopped advertising scenarios this guard is no longer ' +
        'measuring anything and must be re-pointed, not deleted.',
    );
    assert.ok(
      routes.some((r) => r.file === 'run-demo.sh'),
      'run-demo.sh advertises no scenario route; the banner is the first thing an operator reads',
    );
  });

  it('advertises no scenario that the shipping resolver cannot serve', () => {
    const broken = advertisedRoutes()
      .filter((r) => !Object.hasOwn(SCENARIOS, r.id))
      .map((r) => {
        const why = Object.hasOwn(CUT_SCENARIOS, r.id)
          ? 'DELIBERATELY CUT'
          : 'UNKNOWN ID (typo?)';
        return `${r.file}:${r.line} advertises ?scenario=${r.id} -- ${why}`;
      });

    assert.deepEqual(
      broken,
      [],
      'An operator-facing file advertises a scenario the client cannot serve. ' +
        'The visitor is NOT shown an error: currentScenarioId falls through to ' +
        'a local default, so they get a correct, honest page for a scenario ' +
        'they did not ask for, with nothing saying so. Remedy depends on the ' +
        'reason. DELIBERATELY CUT: delete the link and state the finding ' +
        'instead -- "prefix reuse: measured and found absent on both paths" is ' +
        'stronger copy than the link ever was. UNKNOWN ID: it is a typo, and a ' +
        'typo and a cut scenario are indistinguishable to every other check we ' +
        `own.\n${broken.join('\n')}`,
    );
  });

  it('keeps cut scenarios unaddressable by construction', () => {
    // The real ratchet. Deleting one bad link fixes today; keeping CUT_SCENARIOS
    // free of `id` is what stops the route being re-created tomorrow, because
    // `currentScenarioId` gates on Object.hasOwn(SCENARIOS, ...) alone.
    for (const [key, entry] of Object.entries(CUT_SCENARIOS)) {
      assert.equal(
        Object.hasOwn(entry, 'id'),
        false,
        `CUT_SCENARIOS['${key}'] has an 'id'. An id is what makes a scenario ` +
          'addressable, and a cut scenario must not be addressable.',
      );
      assert.equal(
        Object.hasOwn(SCENARIOS, key),
        false,
        `'${key}' is listed as cut AND present in SCENARIOS; it would resolve.`,
      );
    }
  });
});
