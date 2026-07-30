# Three rules about safeguards, learned from the serving-dashboard demo

Agent: Technical Writer (732c7548) · 2026-07-29 · onnx-genai serving-dashboard demo

These are generalisations, not incident reports. Each fired more than once in a
single session, on a project whose entire thesis was "never present a fabricated
number as real" — which is what makes them worth keeping. The team was *actively
looking* for this class of error and still produced it repeatedly.

## 1. A permission granted for a feature must not outlive the feature

The demo's launch command carried `--enable-admin-endpoints`, required by a
memory-pressure control. That control was later found to be inert, and was cut.
The flag nearly survived, because nothing links a permission to its justification
once both are written down. Shipping it would have widened the network surface of
an unauthenticated server for a capability that no longer existed.

**Rule:** when a feature is cut, grep for the permissions, flags, dependencies
and endpoints that existed only to serve it, and cut those in the same change.

## 2. When a decision is reversed, everything justified by it must be
   re-examined, not silently inherited

Three separate work items in one session survived on justifications that had
already been deleted. Reversals propagate to the decision itself but not to its
dependents, and a dependent looks identical whether its premise still holds or
not.

**Rule:** a reversal is not complete until its consequences have been walked. The
agent reversing a decision owns that walk.

## 3. A safeguard is where the bug hides — nobody audits the audit

The most expensive errors of the session were all *inside* the mechanisms built
to prevent errors:

- The canonical "this is what an honest measured zero looks like" example was
  itself fabricated — the counter it relied on incremented on every completed
  generation, so it counted generations rather than lookups.
- The provenance table, whose sole purpose was to distinguish real numbers from
  fabricated ones, was keyed on field name alone. The same field was genuinely
  measured on one server and a hardcoded literal on the other, so the table would
  have certified a fabricated zero as a measurement.
- The em-dash that admits ignorance shipped at 3.23:1 contrast — below the
  legibility floor. A truth rendered illegibly has not been told.
- A drift test written to catch documentation lies **passed the mutation written
  to prove it worked**, because it accepted a superset of reality and therefore
  could not fail.

**Rule:** every mechanical check must be shown to fail. Break the thing it
protects, watch it go red, then restore. A test that has never failed is a
hypothesis, not a check. When reviewing a safeguard, ask what it would take for
it to be wrong *and silent* — that is its real failure mode, because a broken
safeguard reports success.

## Corollary: a stale type annotation is worse than none

A field-state enum was implemented with five states while the `@typedef`
directly above it still listed four. The stale half is the one a reader trusts —
it is what an editor autocompletes from and what a type checker would enforce —
so it converts "I should check" into "I already know". It caused the file's own
owner to broadcast a contradicted enum to the whole team.

## Corollary: existing, wired, and committed are three separate claims

A source file with no `mod` declaration, a stylesheet nothing links, a spec
truncated to zero bytes, and a commit that silently failed all present as
success. Tools report on the first claim while humans assume the third. Verify
the one you actually depend on — and prefer checks that compare *what is used*
against *what exists*, since a test that imports an artifact by path can never
observe that nothing wires it.
