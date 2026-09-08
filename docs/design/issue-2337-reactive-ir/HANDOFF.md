# Issue #2337 reactive IR handoff

Status: design exploration in progress. None of these drafts is an accepted
schema or implementation plan.

Public checkpoint:
[issue comment 5502807990](https://github.com/justinchuby/onnx-genai/issues/2337#issuecomment-5502807990).

## Where to resume

Read these in order:

1. `issue-2337-qwen-reactive-ir-v22.md` for the graph-complete 24-layer Qwen
   candidate using the clean grammar.
2. `issue-2337-reactive-ir-v21-batching.md` for the derived continuous-batching
   certificate and removal of a second serialized batching authority.
3. `issue-2337-reactive-ir-v20-transitions.md` for finalized full/delta
   candidate invariants and the ordinary-component commit-policy surface.
4. `issue-2337-reactive-ir-v19-schema.md` for the first clean consolidated core
   grammar.
5. `issue-2337-reactive-ir-v18-lifecycle.md` for the simplified zero-or-one
   reactor lifecycle and state ownership rule.
6. `issue-2337-reactive-ir-v17-effects.md` for event-scoped effect roots,
   same-occurrence linearity, and commit-phase rules.
7. `issue-2337-reactive-ir-v16-firing.md` for the current firing-inference
   decision and precise event relation checks.
7. `issue-2337-qwen-reactive-ir-v11.md` for the latest Qwen-specific cache,
   padding, and continuation decisions. Its boolean gate syntax must be
   translated to the v19 grammar.

v12-v15 and the earlier Qwen drafts are design history. Read them only when
provenance for a decision is needed; they are no longer part of the active
schema assembly instructions.

The firing question is now resolved:

- components and bundles infer a named common event from required value/effect
  input presence;
- optional inputs do not participate in firing inference;
- `when` is written only to anchor stable inputs or restrict execution to a
  strict subevent;
- an equivalent explicit `when` is rejected as duplicate authority;
- switches keep explicit `when` because it names the parent event being
  partitioned.

The effect-root question is now resolved:

- each root is explicitly scoped to one named event;
- each event occurrence creates a fresh linear token instance;
- reaction and completion roots are distinct;
- every root occurrence reaches its sink before leaving that event;
- core v1 forbids cross-reaction effect carry;
- `after_invocation_commit` roots are scoped to the schema-provided
  `invocation.committed` event.

Completion identity is also resolved: a reactor graph exposes only
`<reactor>.completed`; a one-shot DAG exposes only `invocation.completed`.
There is no alias or parent/child pair in one graph. Both forms occupy the same
lifecycle position before durable commit.

Lifecycle ownership is intentionally simple in v1: a graph has zero or one
reactor. All state belongs to that reactor when present, otherwise to the
implicit `invocation.run` phase. There is no lifecycle-owner field or separate
region inference. A distinct storage-ownership policy only says who manages
the backing resource. Ordinary liveness from semantic sinks rejects unused
nodes.

Zero reaction limit skips only `first`/`steady`/`pulse` work. The completion
postlude, durable commit, and `invocation.committed` phase still run normally.
Completion-scoped transactional effects or publications therefore follow the
ordinary graph rather than a special zero-limit rule; state does not update
without a pulse.

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
- Components and bundles infer firing from the named common presence event of
  required inputs. Explicit `when` only anchors stable inputs or selects a
  strict subevent; equivalent annotations are invalid.
- Event implication, exclusion, and exact coverage are proven from lifecycle,
  switch, and event-join partitions. The compiler does not infer control
  correlation from ordinary tensor values.
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
- The same immutable `state.final` snapshot may be retained for a committed
  effect after durable commit. Retention does not produce a second firing.
- Output streams have one binding to an ordered typed publication batch rather
  than multiple independent emit nodes.
- Effects separately declare commit behavior, retry behavior, and speculation
  safety. Commit behavior is pure, transactional, or
  after-invocation-commit.
- Effect linearity is per event occurrence. Exclusive, complete branch
  consumers route one token family rather than fan it out.
- Effect roots are explicitly event-scoped and produce a fresh token per event
  occurrence. Reaction/completion roots are separate, and v1 has no
  cross-reaction effect carry.
- `invocation.committed` is emitted only after durable commit and is the only
  legal root event for `after_invocation_commit` domains.
- Reactor graphs use `<reactor>.completed`; one-shot DAGs use
  `invocation.completed`. The names never coexist in one graph.
- With one reactor, all state advances in its working phase. Without a reactor,
  all state advances once under `invocation.run`. Lifecycle ownership is not
  authored.
- `max_reactions = 0` is a successful zero-pulse execution: no reaction work
  runs, but completion, durable commit, and after-commit phases still do.
- Dynamic firing is valid. Runtimes compile it to finite prebound blocks/FSM
  transitions; they do not reject it merely for being slower.
- No hot-path YAML traversal, string lookup, dependency discovery, graph-object
  allocation, or per-reaction heap allocation is allowed.

## Core versus optional continuous batching

Continuous batching was deliberately removed from the core after its row
clock, compaction, transaction, and transform concepts made ordinary authoring
too difficult.

The core defines isolated, lockstep invocation semantics with a scalar reactor.
There is no serialized continuous-batching addon in v1. The runtime derives a
discardable grouping certificate from existing `batch_capacity`,
`batch_layout`, `row_scope`, state capabilities, publications, effects, and
backend evidence.

Missing or failed derivation leaves a correct isolated execution. A runtime
falls back per component call site rather than rejecting the core graph or
guessing row independence. Runtime row selections remain positional and never
serialize request, slot, page, or scheduler identity.

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
component-produced plan is now canonical: policy remains ordinary typed
dataflow, while the state transition consumes its count/indices outputs. There
is no special commit-plan node or transition-local conditional DSL.

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

1. Pressure-test the clean grammar against diffusion, recurrent/SSM, tools,
    speculative trees, and revision outputs.
2. Define Rust types, JSON Schema, validation diagnostics, and lowering.
3. Measure parity against the existing execution path; do not infer
    performance from the abstraction.

## Model constraint for continuing the design session

The design work in this session used GPT-5.6 Sol and GPT-5.6 Terra only. Do not
delegate follow-up work to Claude models.
