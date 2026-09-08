# Cyclic reactive dataflow IR - v19 consolidated core schema

This is the first clean schema after the v16-v18 decisions. It replaces the
clock, boolean-gate, and duplicated-`when` forms in the historical drafts.
It remains a design candidate, not an accepted standard.

## 1. Scope

The core describes one isolated workflow invocation:

```text
typed values + presence events + linear effects + delayed state
```

It does not serialize authored sequences, branches, loops, phases, transaction
groups, serving rows, or physical cache allocation. Core v1 permits zero or one
reactor. Continuous batching is an optional proof that several isolated
invocations may be lifted without changing their results.

## 2. Workflow shape

```yaml
pipeline:
  workflow:
    manifest: {...}
    inputs: {...}
    outputs: {...}
    components: {...}
    effects: {...}
    graph:
      nodes: {...}
      effect_roots: {...}
    interfaces: {...}
```

`pipeline.workflow` is the only serialized executable graph ABI. Schedules,
FSM blocks, decoder ABIs, memory plans, state plans, transaction plans, and
physical KV layouts are derived artifacts.

## 3. References

IDs are graph/package-local strings. Node, component, and effect-root IDs
cannot contain `.` because canonical references use dot-qualified generated
names. Artifact port names remain opaque and may contain dots; resolution uses
the resolved signature rather than blind string splitting.

After component signatures generate their output names, the complete value
namespace must be collision-free. In particular, a dotted workflow input ID
such as `request.input_ids` is invalid if its exact spelling also names any
generated node value. Resolution never relies on namespace precedence.

### Values

```text
workflow input              <InputId>
component output            <NodeId>.<PortName>
bundle output/projection    <NodeId>.value[.<MemberId>]
merge output                <NodeId>.value
state snapshot              <StateNodeId>.current[.<MemberId>]
                            <StateNodeId>.final[.<MemberId>]
reactor index               <ReactorNodeId>.index
```

State `next` is an input binding and creates no readable alias.

### Events

```text
reactor lifecycle           <ReactorNodeId>.pulse
                            <ReactorNodeId>.first
                            <ReactorNodeId>.steady
                            <ReactorNodeId>.completed
switch partition            <SwitchNodeId>.then
                            <SwitchNodeId>.else
event join                  <EventJoinNodeId>.event
one-shot lifecycle          invocation.run
                            invocation.completed
post-commit lifecycle       invocation.committed
```

`invocation.run` and `invocation.completed` exist only when the graph has no
reactor. A reactor graph exposes only its own `<id>.completed`.
`invocation.committed` exists in either form and is emitted only after durable
commit succeeds.

### Effects

```text
effect root                 effects.<EffectRootId>
node effect output          <NodeId>.effects.<EffectDomainId>
```

Resolved effect types retain domain and root identity.

## 4. Inputs and component signatures

Workflow inputs declare the caller boundary:

```yaml
inputs:
  request.input_ids:
    contract:
      dtype: int64
      rank: 2
      shape: [batch, sequence]
      batch_layout: {kind: request_aligned, axis: 0}
    role: {kind: runtime, version: '1', role: prompt_tokens}
    source: {kind: request}
    required: true
    constraints:
      valid_sequence_length: {min: 1}
```

An ONNX component takes tensor signatures from graph `ValueInfo`:

```yaml
components:
  model:
    implementation: {kind: onnx, artifact: model.onnx}
    contract:
      id: onnx-genai.autoregressive-decode
      version: '1'
      bindings: {...}
```

Metadata does not repeat or repair ONNX dtype, rank, or shape. Every
graph-visible ONNX port must have sufficient `ValueInfo`.

A runtime binding has no artifact signature, so it declares one:

```yaml
components:
  token_policy:
    implementation: {kind: binding}
    contract: {id: onnx-genai.token-policy, version: '1'}
    signature:
      inputs:
        logits: {required: true, dtype: float32, rank: 2}
        prompt_plan: {required: false, dtype: int64, rank: 1}
      outputs:
        token: {dtype: int64, rank: 1}
```

Signatures classify every input as required or optional. Conditional output or
optional-input presence rules must be statically expressible by the component
contract. Optional values never determine node firing.

## 5. Node union

```text
GraphNode =
    ComponentNode
  | BundleNode
  | StateNode
  | ReactorNode
  | SwitchNode
  | MergeNode
  | EventJoinNode
  | EffectMergeNode
  | EffectPassthroughNode
```

There is no `sequence`, `loop`, `branch`, `invoke`, `emit`, `gate`, authored
region, or top-level clock.

## 6. Firing

Components and bundles infer firing from required value and effect inputs.
Stable inputs add no event constraint.

An omitted `when` is valid when:

- every required input is stable and the node is pure, making it a demanded
  once-per-invocation computation; or
- the exact intersection of required-input presence equals one named event.

An explicit `when` is valid only when it adds information:

- it anchors otherwise-stable inputs to a lifecycle event; or
- it is a strict subevent on which every required input is available.

An explicit event equal to the inferred event is invalid redundant syntax.
Inference never invents an unnamed conjunction or reasons about correlation
between ordinary boolean tensor values.

## 7. Component

```yaml
sample:
  kind: component
  component: token_policy
  inputs:
    logits: effective_logits.value
  effects:
    rng: effects.rng_reaction
```

```text
ComponentNode {
  kind: "component",
  component: ComponentId,
  when?: EventRef,
  inputs: Map<PortName, ValueRef>,
  effects?: Map<EffectDomainId, EffectRef>
}
```

Outputs come from the resolved component signature and are named
`<NodeId>.<PortName>`.

## 8. Bundle

```yaml
decode_cache:
  kind: bundle
  members:
    key.0: decode.present.0.key
    value.0: decode.present.0.value
```

```text
BundleNode {
  kind: "bundle",
  when?: EventRef,
  members: NonEmptyMap<MemberId, ValueRef>
}
```

A bundle is an all-or-nothing structural record. Member order has no meaning;
identity is the exact key set and member contracts. Construction and projection
are zero-copy slot/handle operations.

## 9. Reactor

```yaml
generation:
  kind: reactor
  limit: request.max_iterations
  continue: termination.continue
```

```text
ReactorNode {
  kind: "reactor",
  limit: ValueRef,
  continue: ValueRef
}
```

`limit` is a stable non-negative int64 scalar. `continue` is a reaction-valued
rank-0 bool resolved once after each successful working commit.

For positive limits, the reactor emits `first` on reaction zero, `steady` on
later reactions, and `pulse` on every reaction. `index` is an int64 scalar
present with `pulse`.

After a working commit:

```text
continue && index + 1 < limit -> next pulse
otherwise                     -> completed
```

For limit zero, no pulse event exists; `completed` is emitted immediately.
Completion, durable commit, and after-commit processing still use their normal
semantics.

Every instantaneous cycle is invalid. A legal recurrence crosses state or the
reactor's working-commit delay. A reactor graph must contain a demanded
stateful feedback path.

## 10. Switch

```yaml
route:
  kind: switch
  when: generation.pulse
  predicate: policy.use_tool
```

```text
SwitchNode {
  kind: "switch",
  when: EventRef,
  predicate: ValueRef
}
```

The rank-0 bool predicate must be present whenever `when` is present.
`then` and `else` are mutually exclusive and exactly cover `when`.
Request-aligned `bool[batch]` remains ordinary value data.

A switch always declares `when` because that event is the parent partitioned
by the switch; predicate availability does not express that fact.

## 11. Value merge and event join

```yaml
effective_logits:
  kind: merge
  arms:
    - {when: generation.first, value: prefill_logits.selected}
    - {when: generation.steady, value: logits.current}
```

```text
MergeNode {
  kind: "merge",
  arms: NonEmptyList<{when: EventRef, value: ValueRef}>
}
```

Arm events must be pairwise exclusive and exactly cover one named parent
event. Values must be present on their arms and have identical resolved types.
The output is present on the parent. There is no priority or YAML-order rule.

```yaml
routed:
  kind: event_join
  inputs: [route.then, route.else]
```

```text
EventJoinNode {
  kind: "event_join",
  inputs: NonEmptyList<EventRef>
}
```

Inputs obey the same exclusion and exact-coverage rule. The output aliases
their parent event and normally erases during lowering.

## 12. State

```yaml
decoder_cache:
  kind: state
  initial: {kind: value, value: empty_cache.value}
  next: decode_cache.value
  transition: {...}
  storage_ownership: runtime
  scope: session
  aliasing: permitted
  capabilities:
    rollback_positions: 1
    snapshot: true
    fork: true
  interfaces: {...}
```

```text
StateNode {
  kind: "state",
  initial: {kind: "value", value: ValueRef} | {kind: "absent"},
  next: ValueRef,
  transition: StateTransition,
  storage_ownership: "workflow" | "runtime" | "external",
  scope: "invocation" | "session",
  release?: ReleasePolicy,
  aliasing?: "forbidden" | "permitted" | "required",
  reuse?: ReuseContract,
  capabilities?: StateCapabilities,
  interfaces?: Map<InterfaceId, InterfaceContract>
}
```

State has no firing event or lifecycle-owner field. `storage_ownership`
declares who manages the backing state resource; it does not select a clock,
phase, or reactor. With a reactor, every state samples `next` once per pulse.
Without a reactor, every state samples it once during `invocation.run`.

```text
next present -> stage transition(current, next)
next absent  -> stage retain(current)
```

Initially absent state is valid only when presence analysis proves every read
follows a successful total write. `final` is first exposed with the graph's
completion event. After durable commit, the same immutable snapshot may be
retained for a committed effect; retention does not fire value computation.

## 13. State transitions

```text
StateTransition =
    ReplaceTransition
  | AppendTransition
  | IndexedScatterTransition
```

Replace consumes one complete replacement:

```yaml
transition: {kind: replace}
```

Append adds a logically selected candidate sequence:

```yaml
transition:
  kind: append
  axis: 2
  candidate_source:
    kind: full
    suffix_length: cache_plan.candidate_count
  commit:
    kind: gather
    indices: cache_plan.indices
    length: cache_plan.commit_length
  bound: package.max_context
```

```text
AppendTransition {
  kind: "append",
  axis: unsigned integer,
  candidate_source: AppendCandidateSource,
  commit: CommitSelection,
  bound: ValueRef
}

AppendCandidateSource =
    {kind: "full", suffix_length: ValueRef}
  | {kind: "delta", length: ValueRef}

CommitSelection =
    {kind: "prefix", length: ValueRef}
  | {kind: "gather", indices: ValueRef, length: ValueRef}
```

A full candidate contains a representation of current state followed by the
declared candidate suffix. A delta contains only candidate positions. Commit
indices are logical and relative to that suffix/delta.

Indexed scatter uses logical destinations:

```yaml
transition:
  kind: indexed_scatter
  axis: 2
  candidate_source: {kind: delta, count: candidate.count}
  destinations: cursor.indices
  commit: {kind: prefix, length: accepted.count}
  logical_length: cache_lengths.current
  capacity: package.max_context
```

```text
ScatterCandidateSource =
    {kind: "full", count: ValueRef}
  | {kind: "delta", count: ValueRef}
```

Transition plans never contain physical pages, allocator slots, request IDs,
or device addresses. A bundled state uses one transition/candidate-source mode
for every member, although member tensor signatures may differ.

v20 defines the full-candidate proof and keeps dynamic commit policy in an
ordinary typed component rather than a new plan node or transition expression
language.

## 14. Effect domains and roots

```yaml
effects:
  tools:
    commit_behavior: transactional
    retry: idempotent
    speculation_safety: {kind: forbidden}

graph:
  effect_roots:
    tools_reaction:
      domain: tools
      when: generation.pulse
      sink: tools_done.effects.tools
```

```text
EffectRoot {
  domain: EffectDomainId,
  when: EventRef,
  sink: EffectRef
}
```

Each event occurrence creates one fresh token for the root. Exactly one token
must reach the nested sink within that same occurrence.

Same-reference branch consumers are valid only when their events are an
exclusive, complete partition:

```yaml
tool_call:
  kind: component
  component: invoke_tool
  when: route.then
  effects: {tools: effects.tools_reaction}

no_tool:
  kind: effect_passthrough
  when: route.else
  effects: {tools: effects.tools_reaction}

tools_done:
  kind: effect_merge
  arms:
    - {when: route.then, effect: tool_call.effects.tools}
    - {when: route.else, effect: no_tool.effects.tools}
```

```text
EffectPassthroughNode {
  kind: "effect_passthrough",
  when: EventRef,
  effects: Map<EffectDomainId, EffectRef>
}

EffectMergeNode {
  kind: "effect_merge",
  arms: NonEmptyList<{when: EventRef, effect: EffectRef}>
}
```

Merge arms must preserve one domain and root identity and exactly reconstruct
the parent support.

Core v1 forbids cross-reaction effect carry. A pulse-scoped root reaches its
sink before that pulse's working commit. Transactional effects persist only at
invocation durable commit and abort to the admission baseline on pre-commit
failure. `after_invocation_commit` domains root only at
`invocation.committed`; they cannot influence state, continuation,
publications, or commit success.

## 15. Outputs and publications

Each logical output stream has one ordered publication binding:

```yaml
outputs:
  tokens:
    contract: {...}
    role: tokens
    family: {kind: materialized}
    publication:
      operations:
        - kind: append
          value: acceptance.tokens
          valid_length: acceptance.length
        - kind: finalize
          value: response.value
```

```text
PublicationOperation = Replace | Append | Event | Retract | Finalize
```

Operation list order is payload order, not node execution order. Value
availability determines whether an operation is reaction- or
completion-scoped. Family/operation compatibility and transaction behavior are
validated. Publications become durable only at invocation durable commit.

## 16. Lifecycle and liveness

With a reactor:

```text
pulse -> demanded work -> state/effect working commit -> repeat or completed
completed -> tentative final postlude -> durable commit -> invocation.committed
```

Without a reactor:

```text
invocation.run -> demanded work -> working commit
               -> invocation.completed -> durable commit
               -> invocation.committed
```

Semantic sinks are publications, live state operations, effect-root sinks, and
reactor `continue`. A live state operation demands its initializer, `next`, and
every `ValueRef` embedded in its transition, including candidate counts,
selection indices/lengths, destinations, logical length, bounds, and capacity.
The compiler computes all transitive dependencies. An authored node outside
that live closure is invalid unused graph content; no separate region or
lifecycle-owner mechanism exists.

## 17. Event proof

The compiler derives event relations only from schema lifecycle partitions,
switches, and event joins:

```text
implies(A, B)   = unsatisfiable(A and not B)
excludes(A, B)  = unsatisfiable(A and B)
covers(P, arms) = equivalent(P, union(arms))
                  and pairwise_exclusive(arms)
```

It does not infer control correlation from equal boolean values. These are
load-time proofs, not serialized expressions or runtime event objects.

## 18. Validation order

A conforming loader:

1. validates strict tagged unions and unknown fields;
2. checks identifiers and generated-name collisions;
3. resolves artifacts, component contracts, and signatures;
4. resolves value, event, and effect references;
5. unifies tensor, symbolic-shape, and structural bundle types;
6. builds lifecycle/switch/event-join relations;
7. infers or validates component and bundle firing;
8. checks conditional and optional port presence;
9. removes state/reactor delay edges and rejects instantaneous cycles;
10. solves state initialization, first-write-before-read, and final presence;
11. validates transition candidate/current compatibility and bounds;
12. proves merge exclusion, coverage, and value availability;
13. proves root-to-sink effect linearity and transaction-phase legality;
14. validates publication availability, order, family, and commit behavior;
15. computes liveness and rejects unused nodes;
16. validates a selected optional interface/addon;
17. lowers the graph to dense prebound blocks, FSM transitions, commit latches,
    and memory plans.

Every rejection names the node, port/member or effect root, resolved reference,
event path, failed invariant, provenance, and a corrective action.

## 19. Lowering contract

Events become block membership and finite transitions. Merges become slot
aliases, bundles become port tables, state versions become handles, and effect
tokens become block-local dependency slots.

Steady execution performs:

```text
YAML traversal                  0
string/hash lookup              0
dependency discovery           0
graph-object allocation         0
per-reaction heap allocation    0
implicit tensor copies          0
device-wide synchronization     0
```

Dynamic firing is valid and lowers to precompiled FSM blocks. Continuous
batching may lift those blocks only when its optional interface proves
observational equivalence to isolated execution.
