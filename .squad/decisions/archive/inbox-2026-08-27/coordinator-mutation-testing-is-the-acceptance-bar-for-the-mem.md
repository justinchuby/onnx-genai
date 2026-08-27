### 2026-08-19T09-08-45: Mutation testing is the acceptance bar for the memory stack; two review heuristics adopted from the Phase 6 rejection
**By:** coordinator
**What:** Mutation testing is the acceptance bar for the memory stack; two review heuristics adopted from the Phase 6 rejection
**References:** issue #1186, PR #1440, PR #1426, skill: reviewer-protocol, skill: test-discipline, skill: measurement-discipline
**Why:** ## Context

PR #1440 (Phase 6 plugin memory ABI, issue #1186) was rejected after independent review. The author's self-report was verified accurate on every checkable claim, and the author volunteered three of the gaps himself. It was still rejected, because mutation testing showed four separate mutations (M1, M3, M7, M9) that turned no integration test red — including M7, which replaced the bounded read in `copy_prefix` with a full-size read, reintroducing exactly the out-of-bounds read that prefix negotiation exists to prevent, and left all 80 tests green.

## Decision

Two review heuristics are adopted as standing practice for this repo, both articulated most precisely by the rejected author in his own post-mortem:

**1. A test can defend its name and nothing else.** `open_allocator` had seven post-`Ok` rejection paths and honored its documented release obligation on two. The single rejection path that was tested was the one whose fixture the author had made non-leaky, so the test named for that behavior could not have observed the leak on any path. When a test covers one instance of an N-way branch, ask which instance was chosen and why — the answer is often "the one the fixture supports," which is backwards.

**2. Coverage of a mechanism's branches is not coverage of its memory safety.** The clamp/negotiation logic in `read_prefix` was well covered on its *decision* branches (min-clamping, self-contradiction refusal, slot nulling), and that coverage created justified confidence that the whole mechanism was tested. The *bounded read* underneath it is a separate property and had zero coverage. Enumerate the safety properties separately from the behavioral branches.

**Corollary on fixtures:** a fixture must be physically capable of exhibiting the failure it is named for. Every "short struct" fixture in the suite was a full-size struct that merely lied in its `struct_size` field, so the memory behind the pointer was always valid and an over-read was unobservable even under ASan. Under-sized inputs need genuinely under-sized backing, ideally page-tail-aligned so the next page is unmapped.

**Enforcement:** mutation testing is now the acceptance bar for safety-critical code in the memory stack, not a nice-to-have. Break the production check, confirm red, restore, report what was mutated. This was already used to accept Phase 5 (two mutations, both correctly caught) and to reject Phase 6 (four mutations, none caught). Authors are expected to mutation-test their own work and report results; reviewers verify independently.

## Also recorded

Honesty in a self-report is necessary but not sufficient. This author's report was exemplary — it measurably shortened the review — and the work was still rejected, because two of the three gaps he flagged were more severe than his own assessment. Do not let report quality substitute for artifact quality.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
