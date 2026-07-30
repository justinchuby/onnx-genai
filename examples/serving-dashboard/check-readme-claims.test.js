// A README has claim positions and explanation positions, and they are not held
// to the same standard.
//
// The opening paragraph, the capability tables and the model table are CLAIM
// POSITIONS: a reader takes what they say at face value, and they are what gets
// quoted, skimmed and pasted into a summary. The body is an EXPLANATION
// position, where discussing a feature that was cut -- with its control arm and
// its detection floor -- is not merely allowed, it is the most valuable content
// in the document.
//
// This distinction is not academic. The README's first sentence described
// prefix caching as one of "three things the onnx-genai runtime actually does"
// for hours after the feature was measured PROVEN ABSENT and Scenario B was
// re-scoped. Every honesty mechanism this project built -- five field states,
// the provenance axis, the em-dash treatment, the staleness ceiling -- operates
// on PANELS, and not one of them can reach a sentence in a README. A reader can
// take the claim and never load the page that would correct it.
//
// The capability table was worse than the headline: it asserted a "genuine
// measurement" of prefix caching on Profile D, which is precisely the fabricated
// confidence the entire demo exists to argue against.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const demoDir = dirname(fileURLToPath(import.meta.url));
const readme = readFileSync(join(demoDir, 'README.md'), 'utf8');

// Features measured absent or cut. Naming one in a claim position is a promise
// the runtime cannot keep.
const CUT_FEATURES = [
  {
    pattern: /prefix cach/i,
    name: 'prefix caching',
    evidence:
      'QA n=20 with a control arm sharing nothing from token 0 found NO reuse, ' +
      'against a floor where a working cache collapses TTFT ~1380ms -> ~140ms. ' +
      'Proven absent, not unobserved. Scenario B was re-scoped.',
  },
  {
    pattern: /preemption counter|preempted_total/i,
    name: 'preemption',
    evidence:
      'ContinuousBatchManager holds no scheduler at all, and the dynamic path ' +
      'runs generations serially. There is nothing to preempt.',
  },
];

// The lead: everything before the first `---` rule. This is the part a reader
// is guaranteed to see and the part that gets quoted.
function leadSection() {
  const end = readme.indexOf('\n---');
  assert.ok(end > 0, 'No `---` rule found; the README structure changed.');
  return readme.slice(0, end);
}

test('the lead section was located and is substantial', () => {
  // Without this, a structural change could shrink the inspected region to
  // nothing and every assertion below would pass over an empty string.
  assert.ok(
    leadSection().length > 400,
    `The lead section is only ${leadSection().length} characters. The README ` +
      `structure changed and this check is now inspecting almost nothing.`,
  );
});

test('the lead section claims no feature this runtime does not have', () => {
  const lead = leadSection();

  for (const { pattern, name, evidence } of CUT_FEATURES) {
    const match = lead.match(pattern);
    if (!match) continue;

    // Naming it is allowed only where the text is plainly reporting its
    // ABSENCE. Anything else in the lead reads as a capability claim.
    const line = lead.slice(0, match.index).split('\n').length;
    const context = lead.split('\n')[line - 1] ?? '';
    const isDisclaimer = /not\b|no reuse|absent|turns out|expected|cut|re-scoped/i.test(context);

    assert.ok(
      isDisclaimer,
      `README.md's lead section names "${name}" at line ${line} in what reads ` +
        `as a capability claim:\n\n  ${context.trim()}\n\n` +
        `EVIDENCE: ${evidence}\n\n` +
        `The lead is a CLAIM POSITION -- no room for a control arm, a detection ` +
        `floor or an em-dash, and none of our panel honesty machinery reaches ` +
        `it. Explaining the null result in the body is required; claiming the ` +
        `feature up front is not.`,
    );
  }
});

test('no capability table asserts a prefix-cache measurement', () => {
  // Table rows are claim positions too, and this one asserted a "genuine
  // measurement" of prefix caching on Profile D long after it was disproven.
  for (const row of readme.split('\n').filter((l) => l.trim().startsWith('|'))) {
    if (!/prefix cach/i.test(row)) continue;

    assert.ok(
      !/genuine measurement|measured|live\b/i.test(row),
      `A README table row presents prefix caching as measured:\n\n  ${row.trim()}\n\n` +
        `No prefix field ships on either profile, in any form. The row must ` +
        `read as not applicable.`,
    );
  }
});

// Every honesty mechanism in this project guards one direction: don't claim a
// capability you don't have. Nothing guarded the other direction, and the
// README crossed it -- it asserted prefix reuse was "proven absent" on the
// strength of two timing runs that contradicted each other by more than the
// effect they were measuring, taken on a machine at load average 22.
//
// That felt safe to write, which is exactly the problem. Understating a
// capability reads as scrupulous, so nobody challenges it, and an honesty
// process that only ever ratchets toward understating is not calibrated -- it
// is just a differently-biased claim. "Proven absent" is a strong claim about
// the world and it needs evidence like any other.
//
// The counter finding is airtight and stays: it involves no timing at all.
// The timing verdict is "unverified" until an interleaved TTFT run on a quiet
// machine says otherwise.
const OVERCONFIDENT_ABSENCE = [
  /proven absent/i,
  /does not measurably happen/i,
  /conclusively (?:absent|disproven)/i,
  /impossible to miss\.?\s*The measured effect/i,
];

test('the README does not overclaim CERTAINTY OF ABSENCE either', () => {
  for (const pattern of OVERCONFIDENT_ABSENCE) {
    // Global, because the first hit may be a legitimate quoted mention.
    for (const match of readme.matchAll(new RegExp(pattern.source, 'gi'))) {
    // Naming the phrase in order to REJECT it is explanation, not assertion.
    // The passage that retires "proven absent" has to be able to say it.
    const quoted =
      readme[match.index - 1] === '"' && readme[match.index + match[0].length] === '"';
    if (quoted) continue;

    const line = readme.slice(0, match.index).split('\n').length;
    assert.fail(
      `README.md asserts certainty of absence at line ${line}: "${match[0]}".\n\n` +
        `QA downgraded the prefix-reuse timing result from RED to INCONCLUSIVE: ` +
        `one controlled run put the shared arm 7% SLOWER, another put it 17% ` +
        `FASTER, on a box at load average 22 where a byte-identical binary ` +
        `swings 9.8%. Spread exceeds effect size.\n\n` +
        `Report the counter finding (airtight, no timing involved) and call the ` +
        `timing result UNVERIFIED. Understating a capability is still a claim, ` +
        `and it is the one nobody challenges.`,
    );
    }
  }
});
