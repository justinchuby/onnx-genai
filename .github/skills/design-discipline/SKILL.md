---
name: "design-discipline"
description: "Derive the rule from first principles instead of appending special cases"
domain: "quality"
confidence: "high"
source: "earned (CUDA-EP mask-capacity predicate, #1703)"
---

## Context

Backs Rule 10 in `RULES.md`. Load it when a predicate, dispatch table, allowlist, or capability gate rejects a case it should accept, and the tempting fix is one more branch.

## Patterns

- **Diagnose before extending.** Ask what property makes the *accepted* cases correct. Write it down. Only then touch the mechanism.
- **One question, one mechanism.** Two paths answering the same question are duplicated state, not defence in depth — a new case gets decided by whichever branch runs first.
- **Enumerate properties, not syntax.** Op names, model names and shapes encode what has been *seen*, so they silently mis-answer everything unseen.
- **Keep the guards.** A reorganisation that drops an existing test is a regression however much cleaner it reads.
- **Carry the reason.** Return *why* a case was rejected; that is what makes the next one diagnosable.

## Worked example (#1703)

The CUDA-EP mask-capacity predicate had an op allowlist **plus** a separate `Shape`-consumption policy. A third model, HY-MT1.5, then needed `Expand`. A third branch would have invited a fourth.

`Expand` is not decidable by op type at all: it takes its output width from its *shape operand*, not from the mask. It is safe only when that operand carries the mask length axis from `Shape(mask)` itself.

The principle underneath all three mechanisms was one sentence:

> Freezing the mask is a uniform substitution of `max_len` for the logical length `L`. It is sound exactly when no consumer sources that axis from outside the substitution, and the padded lanes are neutralised before reaching a consumer that is not padding-aware.

So the walk now classifies each consumer edge once — sink / invariant leaf / propagate / mixes-with-reason. GLM-5.2's indexer `Add`, previously excluded by an allowlist omission, now falls out of the general rule: harder to get wrong, not merely shorter.

Result: capture went from declined to `captures=1 replays=125`, decode ~3.0–3.5x faster, token stream byte-identical, and the DeepSeek-V2-Lite and GLM-5.2 guards still passing.

## Anti-Patterns

- Adding a branch per model, vendor, or op name (violates Rule 2)
- "Defence in depth" that is really two mechanisms disagreeing about the same question
- Deleting a test because the new abstraction "obviously" covers it
- Returning bare `false` from a gate, leaving the next engineer to re-derive the cause from the graph
