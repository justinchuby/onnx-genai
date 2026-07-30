# Make green mean something

**A green test proves nothing until it has failed for the right reason.**

## Seven-step pre-commit check

1. State the defect as a **falsifiable predicate**, not an intent.
2. Add the smallest counterexample. Before fixing, require the intended assertion **and accurate
   diagnostic** to go red; an exit code proves only that something failed. A positive fixture
   without a reachable negative is incomplete.
   Hash or diff the mutated file first: an unchanged file means the mutation never happened, so a
   green result proves nothing. This is more insidious than a false red because it looks robust.
3. Trace test and production entrypoints to the same implementation and ordering.
4. Resolve the production consumer and implementation by full path, not basename.
5. Assert the real dimensions -- identity, order, freshness, scope -- not a correlated proxy.
6. Apply the fix; the counterexample turns green and its positive control stays green.
7. Commit, prove object and ancestry, inspect `git show HEAD:<path>` for the counterexample,
   assertion, and implementation, then verify disk and HEAD blob hashes agree.

Run controls first. For zero findings, use synthetic data outside the corpus or a definitional
occurrence guaranteed while the guard matters. It must differ from findings and cross an explicit
test boundary, not a predicate exclusion. Never anchor it on a repairable defect.

## Six false-green mechanisms observed here

1. **Fixture derives assertion data.** `syntheticBlockTable()` derived `pages_shared` from
   `refCounts`; another test shared one variable between input and expectation, comparing a value
   with itself. Both made divergence inexpressible without a manual probe.
2. **Test drives a helper; production drives an orchestrator.** Tests awaited `pollOnce()`;
   production ran `start()`.
3. **Coupled inputs force one branch.** Pairing a path-bearing warning and value made the
   warning-only bypass unreachable.
4. **A self-inspecting guard silently narrows its corpus.** It stops looking and reports clean:
   reassurance, not noise.
5. **An absence assertion lacks anti-vacuity.** "Zero offenders" also passes when nothing ran.
6. **The oracle asserts a proxy.** `111@3000` overwrote newer `222@2000`; value rewound while
   `fetchedAtMs` looked fresher because it measured arrival.

## Oracle-dimension check

Write the invariant over observables, name each dimension, and assert those exact values. Change
what must never change while holding proxies constant. A correct oracle
goes red. Example: keep both snapshots `measured`, rewind value `222 -> 111`, and advance the
timestamp. A timestamp-only oracle is blind.

**ARRIVAL TIME IS NOT CONTENT PROVENANCE.**

## Structural-fix check

Hold routing constant. The path repair kept `safeSame` true while both path flags became false.
Had it flipped, cleanup rerouting -- not an intrinsic fix -- removed the leak.

## Corollaries

- **An anti-vacuity control anchored on a real defect fails when the defect is repaired.**
  Use synthetic data outside the corpus or a permanent definitional occurrence.
- **Test data must not be indistinguishable from the thing it samples.** A guard scanning its
  own file can report its fixtures as production findings.
- **Recognition does not prevent recurrence.** After ambiguous `metrics.rs` basenames, the same
  developer used a basename predicate for a full-path question. Review structure.
