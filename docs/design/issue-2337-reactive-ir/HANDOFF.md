# Issue #2337 reactive IR handoff

Status: design exploration in progress. None of these drafts is an accepted
schema or implementation plan.

Public checkpoint:
[issue comment 5502807990](https://github.com/justinchuby/onnx-genai/issues/2337#issuecomment-5502807990).

## Where to resume

Read these in order:

1. `issue-2337-reactive-ir-v12.md` for the core invocation versus optional
   continuous-batching addon boundary.
2. `issue-2337-reactive-ir-v13-schema.md` for the first complete tagged-union
   grammar. Its top-level clock and gate syntax are superseded below.
3. `issue-2337-reactive-ir-v14-schema.md` for replacing the top-level clock
   block with a graph-local reactor node.
4. `issue-2337-reactive-ir-v15-events.md` for the current firing model:
   first-class presence events, switches, event/value merge arms, and no gate
   node.
5. `issue-2337-qwen-reactive-ir-v11.md` for the latest Qwen-specific cache,
   padding, and continuation decisions. Its boolean gate syntax must be
   translated to the v15 event grammar.

The next unresolved question was:

> Should a component's firing event be inferred from the common presence event
> of its required inputs, with `when` written only when the inputs are stable or
> their event cannot be inferred; or should every component and bundle always
> declare `when`?

The recommendation before the session was interrupted was to infer firing from
required input presence and require explicit `when` only when necessary.

## Locked design decisions

- The core is synchronous reactive dataflow for one isolated workflow
  invocation, not authored `sequence`, `branch`, or `loop` control flow.
- The core has value, event, and linear effect edges. Events may fan out;
  effects may not.
- Every instantaneous cycle is invalid. A legal recurrence crosses state or
  reactor delay.
- A reactor is an ordinary graph node. It owns the irreducible decision to emit
  another reaction event and a hard reaction bound. Pure DAGs have no reactor.
- Reactor outputs are presence events for first, steady, pulse, and completed,
  plus a typed reaction index value.
- Boolean control becomes events through a switch node. Node `when` consumes an
  event, not a boolean expression.
- Merge nodes have explicit event/value arms. Arms must be mutually exclusive,
  complete for their parent event, and type-compatible. Gate nodes are removed.
- State is generic and carries no model-semantic kind. It delays one typed
  value, which may itself be a structural bundle.
- State exposes only `current` and `final`. `next` is an input binding and does
  not create a second SSA alias.
- Initially absent state is allowed when presence analysis proves it cannot be
  read before its first successful write.
- Structural bundles are zero-copy typed records. State bundles must be
  transition-homogeneous, but their member tensor signatures may differ.
- ONNX component signatures come from graph `ValueInfo`; metadata does not
  repeat dtype, rank, or shape. Graph-visible ONNX I/O must be sufficiently
  typed.
- Binding components have explicit signatures for static analysis.
- Workflow input/output boundary contracts remain explicit.
- State transition kinds are `replace`, `append`, and `indexed_scatter`.
- Append/scatter candidate representation is a separate `full | delta` axis.
  A full source contains materialized old state plus a declared candidate
  suffix; a delta source contains only new candidate values.
- Candidate production and commit selection are separate. Commit selection is
  `prefix` or logical `gather`, enabling speculative and candidate-tree
  execution without first-class state-version forks.
- State transition plans contain logical positions only, never physical pages,
  slots, request IDs, or allocator addresses.
- Reaction working commits advance tentative invocation state. Invocation
  durable commit persists session state, effects, and output heads only after
  the reactor and postlude succeed.
- `state.final` and reactor `completed` expose the tentative final snapshot to
  the postlude. Postlude failure aborts to the admission baseline.
- Output streams have one binding to an ordered typed publication batch rather
  than multiple independent emit nodes.
- Effects separately declare commit behavior, retry behavior, and speculation
  safety. Commit behavior is pure, transactional, or
  after-invocation-commit.
- `max_reactions = 0` is a successful no-op: no prefill, mutation, or
  publication.
- Dynamic firing is valid. Runtimes compile it to finite prebound blocks/FSM
  transitions; they do not reject it merely for being slower.
- No hot-path YAML traversal, string lookup, dependency discovery, graph-object
  allocation, or per-reaction heap allocation is allowed.

## Core versus optional continuous batching

Continuous batching was deliberately removed from the core after its row
clock, compaction, transaction, and transform concepts made ordinary authoring
too difficult.

The core defines isolated, lockstep invocation semantics with a scalar reactor.
The optional versioned continuous-batching addon may prove:

- independent row ownership;
- component row independence, selection, expansion, or collective behavior;
- per-row state/effect/publication partitioning;
- compaction, release, and copy-on-write legality;
- lifting multiple invocation FSMs into serving batches.

Ignoring the addon must always leave a correct isolated execution. A runtime
without addon support falls back to isolated execution rather than rejecting
the core graph or guessing row independence.

## Current Qwen shape

The intended isolated execution is:

```text
reaction 0: prefill prompt -> select last valid logits -> sample -> decode token
            -> working commit

reaction 1+: carried logits/KV -> sample -> decode token -> working commit

completion: expose final state -> postlude -> durable invocation commit
```

This preserves the invariant that durable model state covers all committed
input and output tokens, including a published EOS token.

Qwen durable session state includes KV and logical cache lengths. Attention
mask is reconstructed on the first reaction from durable lengths and the new
prompt. A package with a non-reconstructible mask must persist it explicitly.

The Qwen package requires a non-empty prompt for a new invocation, including a
continuation invocation. It does not persist the full next-token logits tensor.
Suspending and resuming the same invocation retains working logits and is
unaffected.

Reaction-zero prefill logits must be gathered at each row's last valid prompt
position. Selecting the last physical column is not correct for arbitrary
padding.

The provisional state commit-plan binding produces:

- candidate count;
- logical gather indices relative to the candidate suffix/delta;
- committed length.

On reaction zero, it selects valid prompt positions plus the accepted sampled
token. On steady reactions, it selects the accepted sampled token only. This
component-produced plan was accepted provisionally; it still needs a final
comparison against transition-local authoring before the schema is frozen.

## Known inconsistencies in historical drafts

- v0-v4 use structured or boolean control ideas superseded by later drafts.
- v5-v11 use a top-level clock, boolean `when`, or gate nodes in places. Those
  must be translated to the v14 reactor and v15 event grammar.
- v10 briefly put continuous-batching transforms in the core. v12 moves them to
  an optional addon.
- v13 is the broadest schema inventory, but its clock section and gate node are
  superseded by v14-v15.
- v14 introduces `reactor.pulse`, but v15 is the first draft that makes reactor
  events directly consumable by `when`.
- The archived drafts are design history, not simultaneously valid schema
  alternatives.

## Remaining design work

1. Decide when firing events are inferred versus explicitly authored.
2. Rewrite the v13 tagged-union grammar as one clean document using v15 event
   semantics, with no clock/gate compatibility paths.
3. Specify event implication, exclusion, and coverage checking precisely.
4. Finalize one-shot DAG lifecycle events and state-final behavior.
5. Specify reactor/state ownership derivation and reject disconnected
   reaction-valued regions.
6. Finalize the full/delta candidate-source invariants for append and indexed
   scatter.
7. Decide the canonical dynamic commit-plan surface.
8. Define the continuous-batching addon separately from the core schema.
9. Expand a complete Qwen candidate YAML from the clean grammar.
10. Pressure-test the clean grammar against diffusion, recurrent/SSM, tools,
    speculative trees, and revision outputs.
11. Define Rust types, JSON Schema, validation diagnostics, and lowering.
12. Measure parity against the existing execution path; do not infer
    performance from the abstraction.

## Model constraint for continuing the design session

The design work in this session used GPT-5.6 Sol and GPT-5.6 Terra only. Do not
delegate follow-up work to Claude models.
