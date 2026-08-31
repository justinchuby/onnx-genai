### 2026-08-19T14-16-47: Mutation-testing harnesses fail toward false confidence: guard every edit AND every restore with an exact occurrence count, never sed by line number
**By:** coordinator
**What:** Mutation-testing harnesses fail toward false confidence: guard every edit AND every restore with an exact occurrence count, never sed by line number
**References:** issue #1186, PR #1454, PR #1465, PR #1468, measurement-discipline, test-discipline
**Why:** Three separate incidents in the memory refactor produced verification results that were WRONG rather than merely incomplete. In each case a test run was reported against code the runner believed was in one state and was actually in another. Two produced phantom survivors (a mutation that was never applied, reported as "the tests did not catch this"); one produced a phantom baseline (a "clean" run against a file still in a mutated state). None were caught by the suite, because the suite was answering a different question than the operator thought they had asked.

INCIDENT 1 — line-number substitution. `sed -i '' '<N>s/old/new/'` was used to apply a mutation. An earlier edit in the same session had shifted the file by one line, so the substitution hit the wrong line. The intended mutation was never applied; the run was reported as evidence.

INCIDENT 2 — a delimiter inside the payload. A shell loop split before/after strings on `|`. The target expression was `served > 0 || committed > 0`, so the split landed inside it. The resulting "mutation" edited only whitespace. The suite stayed green and a survivor was recorded that did not exist.

INCIDENT 3 — the RESTORE step, not the mutation step. After applying a mutation, the naive restore needle occurred twice: once in the code, and once in a newly added doc comment that QUOTED the mutation for documentation purposes. A `count == 1` guard on the restore refused to act rather than silently reverting the wrong occurrence. This one was caught, and only because the guard was applied to the restore as well as to the mutation.

THE RULE: the tooling that judges the fixtures needs the same discipline as the fixtures. Specifically:

1. NEVER use `sed` with line numbers. Use exact-string replacement (Python or equivalent).
2. Guard every edit with an occurrence count and REFUSE on anything other than exactly the expected count. Do not proceed on "at least one".
3. Apply the guard to the RESTORE as well as the mutation. Incident 3 shows restores are not the safe half — and documenting your own mutation is a normal, virtuous act that creates the second occurrence.
4. PRINT the changed line and run `git diff -U1` BEFORE running the suite, and again after restoring. A mutation you did not visually confirm is not evidence.
5. Verify the tree is clean after restore (`git status --porcelain`) before recording any baseline number.

WHY THIS MATTERS MORE THAN ORDINARY TOOL BUGS: mutation testing is used precisely where the suite passing is not trusted. A broken harness fails in the direction of FALSE CONFIDENCE — either "the tests missed this" when nothing was changed, or "the baseline is green" when it was not. Both mislead in the direction the exercise exists to correct. A silent no-op edit and a passing suite are indistinguishable from a genuine survivor unless the edit itself is verified.

Note the incidents were spread across three different agents including the reviewer. This is not an individual failing; it is a property of the technique, and the guards must be mechanical rather than remembered.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
