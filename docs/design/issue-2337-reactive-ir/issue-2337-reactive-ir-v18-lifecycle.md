# Cyclic reactive dataflow IR - v18 lifecycle ownership

v18 keeps lifecycle ownership deliberately small.

## One rule

Core v1 permits zero or one reactor:

- with a reactor, every state node belongs to that reactor's working phase;
- without a reactor, every state node belongs to the one implicit
  `invocation.run` working phase.

There is no authored `owner`, region, phase, clock, or transaction-group field.
There is also no topology algorithm that tries to choose among owners: v1 never
has more than one candidate.

## Reactor graph

For each `<reactor>.pulse` occurrence:

1. expose each present `state.current` snapshot;
2. execute demanded reaction work;
3. sample each state's `next` once;
4. stage update or retain;
5. complete every pulse-scoped effect root;
6. atomically perform the working commit;
7. sample reactor `continue` and either emit the next pulse or
   `<reactor>.completed`.

`state.final` is first exposed with `<reactor>.completed`, after the last
successful working commit. When the reaction limit is zero, it equals the
admission baseline, including the original presence bit.

## One-shot graph

A graph without a reactor has one implicit working occurrence:

```text
invocation.run -> working commit -> invocation.completed
```

Every state exposes `current` during `invocation.run`, samples `next` once, and
then exposes its tentative `final` value with `invocation.completed`.

An absent initializer plus an absent `next` remains absent. A required
completion consumer must prove final presence on every path; completion does
not synthesize a default value.

After the completion postlude succeeds, durable commit emits
`invocation.committed`. Postlude failure aborts state, transactional effects,
and publications to the admission baseline. After-commit effects cannot roll
that commit back. They may read the same retained immutable `state.final`
snapshot, now durable, but that retention does not produce another value event.

## Demand and dead nodes

Execution is demand-driven from semantic sinks:

- workflow output publications;
- live state operations, including initializer, `next`, and every transition
  operand;
- effect sinks;
- reactor `continue`;

Their transitive value, event, and effect dependencies are live. Any authored
node outside that closure is rejected as unused with its node ID and nearest
unconsumed output.

This ordinary liveness check replaces a separate "disconnected
reaction-valued region" concept. Event-valued work is valid when it descends
from the graph's sole lifecycle source and reaches a semantic sink. The loader
does not need an additional region declaration or ownership proof.

## Stable work

A demanded pure node over stable inputs may be evaluated once before its first
consumer. Writing `when` anchors it to a working or completion event when that
lifecycle position is semantically required.

Hoisting never moves:

- state sampling or updates;
- effect-token operations;
- publication decisions;
- fallible work across a commit boundary when doing so would change the
  reported failure phase.

This is a lowering rule, not authored phase syntax.

## Validation

The lifecycle checks are:

1. reject more than one reactor;
2. select the reactor working phase or implicit one-shot working phase;
3. assign every state to that sole phase;
4. resolve event/value/effect presence;
5. compute liveness backward from semantic sinks;
6. reject unused nodes and instantaneous cycles;
7. prove state first-write-before-read and final presence;
8. lower the live graph to prebound blocks and commit latches.

No lifecycle-owner syntax or region inference is required. A separate storage
ownership policy may still declare who manages a state's backing resource; it
does not select a working phase.
