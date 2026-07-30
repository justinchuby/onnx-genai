# Make green mean something

**A green test proves nothing until it has failed for the right reason.**

## Seven-step pre-commit check

1. State the defect as a **falsifiable predicate**, not an intent.
2. Add the smallest counterexample the current code mishandles. Run it before the fix and
   require the intended assertion and accurate diagnostic to go red. A red exit code proves
   only that something failed. A positive fixture without a reachable negative is incomplete.
3. Trace the shipping entrypoint and test entrypoint to the same implementation and ordering.
4. Name the production consumer and implementation symbol; resolve both full paths.
5. Assert the invariant's real dimensions directly: identity, ordering, freshness, scope,
   cardinality, confidentiality -- not a correlated proxy.
6. Apply the fix. The same counterexample must turn green while its positive control stays green.
7. After committing, prove the object and ancestry, inspect `git show HEAD:<path>` for the
   counterexample, assertion, and implementation, then verify the disk hash equals HEAD before
   reporting results.

Run anti-vacuity controls before assertions that depend on them. For a goal state of zero findings,
anchor the control on either synthetic data outside the scanned corpus or a definitional occurrence
that must exist while the guard remains relevant. Never anchor it on a repairable defect.

## Six false-green mechanisms observed here

1. **The fixture derives what the assertion checks.** `syntheticBlockTable()` computed
   `pages_shared` from the scanned `refCounts`, making window/whole-pool disagreement
   impossible to represent.
2. **The test drives a helper while production drives an orchestrator.** Tests awaited
   `pollOnce()`; production ran `start()`. The tested execution model did not ship.
3. **Coupled inputs force one branch.** A format test paired a path-bearing warning with
   a path-bearing value, so cleanup always ran and the warning-only bypass was unreachable.
4. **A self-inspecting guard silently narrows its corpus.** It stops looking at relevant
   files and reports clean. This failure direction produces reassurance, not noise.
5. **An absence assertion has no anti-vacuity control.** "Zero offenders" passes both when
   the code is clean and when nothing was scanned.
6. **The oracle asserts a proxy rather than the invariant.** Older content `111@3000`
   overwrote newer content `222@2000`: the value rewound while the timestamp looked fresher.
   `fetchedAtMs` measured arrival, not content provenance.

## Oracle-dimension check

1. Write the invariant as a relation over observable variables.
2. Name each semantic dimension.
3. Map every dimension to an assertion reading that exact observable.
4. Change the thing that must never change while keeping everything merely correlated with it
   unchanged. A correct oracle goes red.

Example: keep both snapshots `measured`, but change value `222 -> 111` while the timestamp advances.
If the test asserts only timestamp monotonicity, it is blind.

**ARRIVAL TIME IS NOT CONTENT PROVENANCE.**

## Structural-fix check

Hold a routing observable constant. The path-leak repair kept `safeSame` true while both path flags
became false. If `safeSame` had flipped, cleanup rerouting -- not an intrinsic fix -- removed the leak.

## Corollaries

- **An anti-vacuity control anchored on a real defect fails when the defect is repaired.**
  Use a synthetic control outside the corpus or a permanent definitional occurrence.
- **Test data must not be indistinguishable from the thing it samples.** A guard scanning its
  own file can report its fixtures as production findings.
- **Recognising an error shape does not prevent recurrence.** After finding ambiguous `metrics.rs`
  basenames, the same developer used a basename predicate for a full-path question. Re-review structure.

## Pinned-review warning

Never use `git archive` for a pinned review. It can omit tests and disarm self-inspecting
guards. An archive failure is useful evidence; an archive pass is not.
