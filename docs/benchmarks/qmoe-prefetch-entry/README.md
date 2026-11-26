# QMoE prefetch proof belongs at the funding entry

The route-first caller previously checked shared-bank capacity before calling
`HostCacheOperation::prefetch_memory`. That method could still return a pipeline
scope when called directly on an ineligible operation (including local,
ungoverned memory). Move the existing eligibility check into the scope-preparation
entry, before its local-source fast return, selected-expert traversal, payload
locking, reader pinning or grants. The caller retains only enabled/multiple-group
scheduling checks; it creates the worker only after receiving an eligible scope with its original
mode-specific funding contract. Local sources retain their on-demand mode.

Production scheduling still performs the same one eligibility query and uses
the original serial path on None. No capacity policy, bank registry, mutex,
worker, weight owner, pool, quota API, split/refund rule or arithmetic changes.
The returned scope still borrows the operation that freezes registry membership.
There is no per-row/tile/expert workspace admission and no new tracing/formatting
operation on the refusal path.

## Behavior covered

- Direct capacity refusal covers governed and local sources, both a current
  three-expert bank over budget and a two-expert bank whose retained foreign
  bank makes the shared requirement exceed budget. A held payload mutex and a
  panic-on-consumption iterator prove refusal completes before those boundaries.
  Request/refund counts, physical charge, cache fields and complete offload
  statistics remain unchanged (excluding unrelated dynamic Linux process data).
  The parent always unlocks before asserting/joining, making a misplaced guard
  a bounded failure rather than a stranded test worker.
- Governed zero-retention streaming prepares one eight-byte/two-slot window with
  one issuer request and one final refund while the registry write lock is held.
  It does not inspect selected experts, and the registry remains empty.
- A fully warm retained three-expert set pins its existing payload readers with
  zero new issuer callbacks. Clearing cache membership preserves all 12 charged
  bytes until the scope drops; the three independent payload grants then refund.

Existing shared/foreign-bank proof, writer exclusion/known-bank concurrency,
serial/prefetch byte-and-stat identity, cache pressure, numerical/IEEE, selected
payload split and workspace issuer-count tests remain in scope. No caller-side
eligibility guard is retained to duplicate the entry's decision.

## Negative control and verification boundary

A temporary mutation removes only the new entry guard. The direct-refusal test
must compile and fail as an executed single test on the local-source case;
a compile error or zero selected tests is not accepted. The mutation is restored
byte-for-byte against the final validated source. Original/mutant sources and
complete logs are retained. No mutation or diagnostic API is shipped.

This verifies resource behavior, not inference latency, CPU time, allocation-free
scope construction or GPU behavior. Reader-pin metadata still has its existing
allocations; positive cold/transient weight admissions remain accounting events.
No CUDA device was used. Heavy checks used the shared host flock and were
coordinated with the peer chat. Reproduction drivers retain original local paths
and need adaptation on other checkouts.

## Final validation

The exact final source passed native QMoE65 tests (one existing ignored timing
benchmark), MLAS research-reference QMoE65 (one ignored), host QMoE oracle1 and
strict CPU all-target Clippy in both configurations. Test listing confirms the
three new cases and original ownership/prefetch cases execute. The initial
Clippy bool-assertion diagnostic is retained; it was corrected before final
validation. Both held-lock tests unlock before scope join even on timeout.

Removing only the entry guard compiled and executed the direct-refusal test,
which failed with zero passed/one failed on the local-source case. The complete
log and both source bodies are retained; a compiler failure is rejected as a
negative control. Source reconstruction also proves that the only runtime edit
is moving this existing check from caller to scope preparation.

Run `python3 -B docs/benchmarks/qmoe-prefetch-entry/verify.py .` from the repository
root to check sealed logs, selected tests, the exact guard mutation, source
restoration and base/runtime reconstruction. Omit the checkout argument to
verify historical evidence after later source changes.
