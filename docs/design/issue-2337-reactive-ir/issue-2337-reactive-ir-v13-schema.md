# Cyclic reactive dataflow IR — v13 core schema grammar

This document defines the candidate canonical surface. It intentionally avoids compatibility aliases and duplicate shorthand forms.

## 1. Workflow shape

```yaml
pipeline:
  workflow:
    manifest: {...}
    inputs: {...}
    outputs: {...}
    components: {...}
    effects: {...}
    graph:
      clock: {...}       # absent for a one-shot acyclic workflow
      nodes: {...}
      effect_sinks: {...}
    interfaces: {...}    # optional optimization proofs/addons
    serving: {...}
```

`pipeline.workflow` remains the only serialized executable graph ABI. Compiled regions, schedules, decoder ABI, state plans, memory plans, and physical KV plans are derived artifacts.

## 2. Identifier and value-reference grammar

```text
InputId      := package-local string
ComponentId  := package-local string without '.'
NodeId       := package-local string without '.'
MemberId     := package-local string
PortName     := artifact-defined opaque string
ValueRef     := canonical value name
```

Canonical values are inserted into one symbol table:

```text
workflow input                  <InputId>
component output                <NodeId>.<PortName>
state current                   <NodeId>.current
state final                     <NodeId>.final
bundle projection               <NodeId>.value.<MemberId>
bundled-state projection       <NodeId>.current.<MemberId>
                                <NodeId>.final.<MemberId>
clock values                    reaction.index
                                reaction.first
                                clock.completed
effect output                   <NodeId>.effects.<EffectDomain>
```

Resolution uses the declared node/component signature, not blind string splitting. Any collision between an input ID, virtual clock value, or generated node output is invalid. Node IDs containing `.` are forbidden so artifact output ports may retain dots without ambiguity.

State `next:` is an input binding and is **not** inserted into the value symbol table.

## 3. Inputs

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

Input contracts remain explicit because they define the caller boundary. Existing request/application/literal/artifact source variants remain valid concepts. Constraints are typed schema objects, not free-form expressions.

## 4. Components

### ONNX component

```yaml
components:
  model:
    implementation: {kind: onnx, artifact: model.onnx}
    roles:
      input_ids: token_ids
      attention_mask: attention_mask
      logits: logits
    contract:
      id: onnx-genai.autoregressive-decode
      version: '1'
      bindings: {...}
```

Rules:

- ONNX input/output dtype, rank, and shape come only from graph `ValueInfo`.
- Every graph-visible ONNX port must have sufficient ValueInfo for static checking.
- Metadata does not patch or duplicate deficient ONNX types.
- Roles/contracts declare non-ONNX semantic facts and substitution bounds.

### Binding component

```yaml
components:
  token_policy:
    implementation: {kind: binding}
    contract: {id: onnx-genai.token-policy, version: '1'}
    signature:
      inputs:
        logits: {dtype: float32, rank: 2, shape: [batch, vocabulary]}
        active: {dtype: bool, rank: 1, shape: [batch]}
      outputs:
        token: {dtype: int64, rank: 1, shape: [batch]}
        done: {dtype: bool, rank: 1, shape: [batch]}
      presence:
        outputs: always
```

A binding has no artifact signature, so `signature` is required and explicit. Conditional input/output presence is part of the signature and statically checked.

## 5. Graph node union

```text
GraphNode = ComponentNode
          | StateNode
          | BundleNode
          | GateNode
          | MergeNode
          | EffectMergeNode
```

There is no `sequence`, `loop`, `branch`, `invoke`, `emit`, or authored execution region.

## 6. Component node

```yaml
nodes:
  sample:
    kind: component
    component: token_policy
    when:                         # optional scalar presence condition
      value: request.use_policy
      equals: true
    inputs:
      logits: effective_logits.value
      active: active.current
```

```text
ComponentNode {
  kind: "component",
  component: ComponentId,
  when?: PresencePredicate,
  inputs: Map<PortName, ValueRef>,
  effects?: Map<EffectDomain, EffectRef>
}
```

Outputs are implicit from the ONNX/binding signature and appear as `<NodeId>.<PortName>`.

`when` accepts scalar bool presence control in the core invocation model. Request-aligned tensor predicates remain ordinary data.

## 7. Bundle node

```yaml
nodes:
  decode_cache:
    kind: bundle
    when: {value: reaction.first, equals: true}
    members:
      key.0: prefill.present.0.key
      value.0: prefill.present.0.value
```

```text
BundleNode {
  kind: "bundle",
  when?: PresencePredicate,
  members: Map<MemberId, ValueRef>
}
```

A bundle is an unordered structural record keyed by member ID. Identity is the exact key set plus pairwise member contracts; YAML/map order has no semantic meaning. Construction/projection is zero-copy and erases to port slots.

Bundle presence is all-or-nothing unless a future bundle type explicitly introduces optional members.

## 8. Gate node

```yaml
nodes:
  carried_cache:
    kind: gate
    when: {value: reaction.first, equals: false}
    value: decoder_cache.current
```

```text
GateNode {
  kind: "gate",
  when: PresencePredicate,
  value: ValueRef
}
```

The node exposes `carried_cache.value` with the same type as its input. False means absent; it never synthesizes a zero/stale/default value.

## 9. Merge node

```yaml
nodes:
  effective_cache:
    kind: merge
    inputs: [prefill_cache.value, carried_cache.value]
    require: exactly_one_present
```

```text
MergeNode {
  kind: "merge",
  inputs: NonEmptyList<ValueRef>,
  require: "exactly_one_present"
}
```

All inputs must have the same tensor/bundle type. Output is `effective_cache.value`. Zero or multiple present inputs fail the reaction with provenance.

No priority/first-present variant exists in core.

## 10. State node

A state always delays exactly one typed value. That value may be a tensor or structural bundle.

```yaml
nodes:
  decoder_cache:
    kind: state
    initial: {kind: value, value: empty_cache.value}
    next: decode_cache.value
    transition: {...}
    ownership: runtime
    scope: session
    release: session_end
    aliasing: permitted
    reuse: {prefix: allowed, evict_prefix: forbidden}
    capabilities: {rollback_positions: 1, snapshot: true, fork: true}
    interfaces: {...}
```

Initially absent state is explicit:

```yaml
initial: {kind: absent}
```

```text
StateNode {
  kind: "state",
  initial: StateInitial,
  next: ValueRef,
  transition: StateTransition,
  when?: PresencePredicate,
  ownership: "workflow" | "runtime" | "external",
  scope: "invocation" | "session",
  release?: "invocation_end" | "session_end" | "row_release",
  aliasing?: "forbidden" | "permitted" | "required",
  reuse?: ReuseContract,
  capabilities?: StateCapabilities,
  interfaces?: Map<InterfaceId, InterfaceContract>
}
```

Readable outputs are `state.current` and `state.final`. For a bundled value, member projections append `.<MemberId>`.

A false state `when` retains current. `next` still names one producer; multiple writers do not exist.

## 11. State transition union

```text
StateTransition = ReplaceTransition
                | AppendTransition
                | IndexedScatterTransition
```

### Replace

```yaml
transition: {kind: replace}
```

The producer supplies the complete replacement value.

### Append

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
  candidate_source: CandidateSource,
  commit: CommitSelection,
  bound: ValueRef
}

CandidateSource = {
  kind: "full",
  suffix_length: ValueRef
} | {
  kind: "delta",
  length: ValueRef
}

CommitSelection = {
  kind: "prefix",
  length: ValueRef
} | {
  kind: "gather",
  indices: ValueRef,
  length: ValueRef
}
```

For `full`, the producer emits materialized old-prefix plus candidate suffix. For `delta`, it emits only candidate positions. Commit indices are relative to the candidate suffix/delta and define logical append order.

### Indexed scatter

```yaml
transition:
  kind: indexed_scatter
  axis: 2
  candidate_source: {kind: delta, length: candidate.count}
  destinations: cursor.indices
  commit: {kind: prefix, length: accepted.count}
  logical_length: cache_lengths.current
  capacity: package.max_context
```

`destinations` are logical positions. They are never runtime page IDs, allocator slots, or physical addresses.

## 12. Clock

The clock is absent for a one-shot acyclic DAG.

```yaml
graph:
  clock:
    kind: synchronous
    max_reactions: request.max_iterations
    repeat_while: termination.continue
```

```text
Clock {
  kind: "synchronous",
  max_reactions: ValueRef,
  repeat_while: ValueRef
}
```

Virtual outputs are fixed by the schema:

```text
reaction.index   zero-based scalar ordinal for the current reaction
reaction.first   scalar bool equal to (reaction.index == 0)
clock.completed  final unit/event after the last working commit
```

`max_reactions = 0` is a successful no-op. Repeat is sampled after working commit. State final becomes present after clock completion.

## 13. Output publication batch

Each logical output stream has one binding to an ordered batch of operations.

```yaml
outputs:
  tokens:
    contract: {...}
    role: tokens
    family: {kind: materialized}
    stage: pre_adapter
    publication:
      operations:
        - kind: append
          value: acceptance.tokens
          valid_length: acceptance.length
        - kind: append
          value: grammar.token
          valid_length: grammar.forced_length
```

```text
PublicationOperation = Replace | Append | Event | Retract | Finalize
```

Operation-list order is publication payload order, not execution-node order. Availability determines whether an operation occurs per reaction or once at finalization. Family/operation compatibility is validated.

## 14. Effect contract and edges

```yaml
effects:
  external_store:
    commit_behavior: transactional
    retry: idempotent
    speculation_safety: {...}
```

```text
CommitBehavior = "pure" | "transactional" | "after_invocation_commit"
```

Node effect input:

```yaml
nodes:
  write:
    kind: component
    component: external_write
    inputs: {value: producer.value}
    effects:
      external_store: read.effects.external_store

graph:
  effect_sinks:
    external_store: write.effects.external_store
```

Every effect token has one consumer. Fan-out, unordered same-domain writers, missing sinks, unsafe speculative use, and commit-behavior violations fail validation.

## 15. Optional continuous-batching addon

Continuous batching is outside the core grammar:

```yaml
interfaces:
  onnx-genai.continuous-batching:
    version: '1'
    contract: {...}
```

The addon may prove row independence, selection/expansion, compaction, per-row transaction partitioning, and cross-invocation scheduling. Ignoring it always leaves a correct isolated core workflow.

## 16. Validation passes

A conforming loader performs, in order:

1. strict schema/unknown-field validation;
2. identifier and generated-value collision checks;
3. component artifact/contract resolution;
4. ONNX/binding signature resolution;
5. exact component input-port binding checks;
6. value-reference and effect-reference resolution;
7. tensor/bundle type and symbolic-shape unification;
8. state-transition-specific current/candidate compatibility;
9. instantaneous-cycle rejection after removing state delays;
10. stable/reaction/final availability inference;
11. initializer, first-write-before-read, and presence proof;
12. merge and effect-linearity validation;
13. clock bound/predicate validation;
14. reaction working-transaction and invocation-transaction validation;
15. publication-family/ordering/commit validation;
16. optional interface/addon validation when selected;
17. lowering to dense prebound blocks/FSM and memory plan.

Every rejection names the node, port/member, resolved value, artifact/contract provenance, failed invariant, and corrective action.
