### 2026-08-19T13-45-27: In environments that cannot exercise the code, written claims are the acceptance surface and need a higher evidentiary bar, not a lower one
**By:** coordinator
**What:** In environments that cannot exercise the code, written claims are the acceptance surface and need a higher evidentiary bar, not a lower one
**References:** issue #1186, PR #1465, measurement-discipline, test-discipline, reviewer-protocol
**Why:** Phase 7 of the memory refactor (PR #1465) removed the built-in CUDA eager allocator on a macOS/arm64 host with no CUDA device. Every behavioural CUDA path was GPU-gated; 459 tests could not run. The PR was rejected with **no production defect found**. All three blocking findings were false statements in prose: a capability probe documented as working in three places whose relevant branch is unreachable because the underlying function silently substitutes a 2 MiB default on driver failure; a shipped-constraints table stating a retained memory pool is off by default when it is on at 256 MiB on two of three paths; and a disclosure asserting a test helper was anchored against vacuity by naming an anchor that covers a different helper.

The author's own diagnosis, which is the transferable part: **"where a claim substituted for verification I could not perform, I did not hold it to the standard I held the tests to."**

THE RULE: when work is done in an environment that cannot exercise the code — no GPU, no target OS, no production data, no network — the written claims become the acceptance surface, because they are carrying the weight the tests cannot. They must therefore be held to a HIGHER evidentiary standard than usual, not a lower one. The instinct runs the opposite way: unverifiable areas feel like the place to relax, precisely because nothing will contradict you.

Three specific failure shapes observed, all in the same PR:
1. Reasoning from the EXISTENCE of a guard to its REACHABILITY. The author saw `if granularity == 0 { return Err(...) }` in `build()` and never opened the function producing that value, which ends `else { 2 << 20 }`. Trace every claimed check to the value it reads.
2. Using an unverified mechanism as the JUSTIFICATION FOR DECLINING to build a verified one. The author correctly declined to add a capability probe that could not be exercised, on the argument that an existing two-leg exercise sufficed — while one leg did not exist. Declining untestable code is right; the substitute you cite must itself be checked.
3. Generalising from the path you read to the paths you did not. The author read `auto_dynamic_lending.then_some(256 << 20)` on the governed path, generalised "off by default", and did not carry it back to the standalone path whose `Some(DEFAULT_STANDALONE_PHYSICAL_POOL_BYTES)` it had itself preserved.

A fourth shape, self-correcting only if caught: an error of this kind WILL NOT self-correct downstream when the verifying party follows the author's own checklist. A CUDA host following "what a CUDA host must check" item 3 would observe the diagnostic working through a different call path and conclude the documented boundary was accurate.

WHAT TO DO INSTEAD: state exactly which call detects the condition; mark unverified claims as unverified rather than describing the mechanism as though observed; and when declining to build something because it cannot be tested here — which remains the right call — verify the substitute you name in its place.

WHAT WAS RIGHT AND SHOULD BE COPIED: the same PR left the benchmark criterion explicitly unticked with no number invented, and split a criterion into a met half and an unmet half rather than rounding up. The reviewer endorsed both. Honest grading of what did not run was never the problem; the problem was describing unrun mechanisms in the confident register of run ones.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
