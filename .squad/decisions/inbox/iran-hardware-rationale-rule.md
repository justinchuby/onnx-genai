# Standing Directive: Hardware-Rationale Accuracy in Dispatch Comments

**Date**: 2025-07-28
**Author**: Iran (Mac CPU Optimization)
**Status**: Standing directive

## The Rule

Any source comment justifying a dispatch threshold with a hardware figure (cache size, SIMD width, core count, bandwidth, SLC capacity) must satisfy all three of the following:

1. **Cite a verified figure.** The hardware number must come from Apple's published specs, a reputable teardown/die-shot analysis (Notebookcheck, Anandtech, Wikipedia with cited sources), or an on-device measurement. Never cite a figure from memory alone — look it up, every time.

2. **State whether the constant is *derived* or *fitted*.** A derived constant follows mechanistically from the hardware figure (e.g., "tile size = L1 / 2 to leave room for output accumulator"). A fitted constant was chosen empirically and then rationalised against hardware figures after the fact (e.g., "crossover measured at M=16–24, 16 chosen conservatively"). Both are valid; mislabelling a fitted constant as derived is not.

3. **If fitted, state the measured bracket.** Give the range over which the crossover was observed, on which part, and whether the bracket was confirmed on more than one chip. This lets the next engineer know exactly how much slack exists before the threshold needs revisiting.

## Why This Matters

A wrong rationale is worse than no rationale. If a comment says "16 MB is below the 8 MB SLC" and the next engineer reads it literally, they may "fix" the constant downward to match the false premise — silently breaking a working dispatch. The code was right in every case below; only the stated reason was wrong, and a wrong reason actively invites wrong edits.

Three defects of this exact class shipped into review in this campaign:

- **PR #347 (twice):** Justified a constant via "64 KB L1" when 64 KB is the E-core L1 — the P-core L1 is 192 KB (128 KB data). Separately, presented an empirically fitted reuse factor as mechanistically derived.
- **PR #353:** Stated "16 MB is well below the smallest SLC (8 MB)" — the arithmetic is inverted (16 > 8). The mechanism and threshold were correct; the stated size relationship was backwards.

In all three cases the code was correct and the review caught the comment, not a bug. But each false rationale was one careless merge away from becoming a false constraint on future work.

## Reviewer Obligation

Reviewers (including Chew) must check the **arithmetic** of any stated hardware relationship, not just the plausibility of the conclusion. "This seems reasonable" is not sufficient when the comment contains a specific numerical claim.
