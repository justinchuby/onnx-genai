# Cyclic reactive dataflow IR — v14 reactor grammar

v14 removes the top-level clock DSL. Repetition is represented by one ordinary graph node.

## Reactor node

```yaml
graph:
  nodes:
    cycle:
      kind: reactor
      limit: request.max_iterations
      continue: termination.continue
```

Generated outputs:

```text
cycle.pulse       presence event for the current reaction
cycle.index       int64 scalar, zero-based current reaction ordinal
cycle.first       bool scalar, equivalent to index == 0
cycle.completed   final presence event emitted after the final working commit
```

`index`, `first`, and `completed` are outputs of this node rather than global virtual names.

## Semantics

Invocation admission supplies the reactor's implicit initial trigger.

```text
if limit == 0:
  emit cycle.completed
  run no reaction
else:
  emit cycle.pulse(index=0, first=true)
```

After reaction `i` reaches a successful working commit:

```text
if continue == true and i + 1 < limit:
  emit cycle.pulse(index=i+1, first=false)
else:
  emit cycle.completed
```

The reactor cannot emit the next pulse before the current reaction transaction resolves.

## Why repetition is not fully implicit

The cyclic value graph determines **what belongs to one reaction**, but topology cannot determine **whether another reaction should occur**.

Without `continue`, every state with a present `next` would fire forever. Treating missing state writes as termination would make ordinary gating, failure, and completion ambiguous. The reactor carries only this irreducible event-generation decision and hard safety bound.

## Region derivation

The compiler derives the reaction region from dataflow:

1. Start at consumers of `cycle.pulse`, `cycle.index`, `cycle.first`, and every `state.current` participating in feedback.
2. Follow instantaneous data/effect dependencies through `state.next`, publications, and `cycle.continue`.
3. Remove state/reactor delay edges and require the resulting instantaneous graph to be acyclic.
4. Values depending only on workflow inputs are stable and may be hoisted.
5. Consumers of `state.final` or `cycle.completed` form the final postlude.

No authored setup/body/finalize region exists.

## Qwen example

```yaml
graph:
  nodes:
    generation:
      kind: reactor
      limit: request.max_iterations
      continue: termination.continue

    prefill:
      kind: component
      component: model
      when: {value: generation.first, equals: true}
      inputs:
        input_ids: request.input_ids
        # ...

    carried_logits:
      kind: gate
      when: {value: generation.first, equals: false}
      value: logits.current

    termination:
      kind: component
      component: termination
      inputs:
        iteration: generation.index
        # ...

    finalizer:
      kind: component
      component: invocation_finalizer
      inputs:
        completed: generation.completed
        final_cache: decoder_cache.final
```

Common reaction nodes need not all consume `generation.pulse` explicitly. Their dependency on `generation.first/index`, gated merge outputs, or feedback state places them in the derived reaction region. A reaction subgraph with no path from the reactor or feedback state is stable and runs once.

## Reactor grammar

```text
ReactorNode {
  kind: "reactor",
  limit: ValueRef,
  continue: ValueRef
}
```

Validation:

- v1 permits at most one reactor per workflow;
- `limit` resolves to a stable, non-negative int64 scalar;
- `continue` resolves to a reaction-valued bool scalar;
- the reactor participates in at least one stateful feedback region;
- every reaction-valued state write/publication/effect is reachable from the reactor's region;
- `continue` is resolved exactly once per successful reaction;
- `cycle.completed` cannot feed reaction state or `continue`;
- a pure acyclic workflow has no reactor;
- a feedback cycle without a reactor is invalid in v1.

## Updated node union

```text
GraphNode = ComponentNode
          | StateNode
          | ReactorNode
          | BundleNode
          | GateNode
          | MergeNode
          | EffectMergeNode
```

## Updated canonical values

```text
workflow input                  <InputId>
component output                <NodeId>.<PortName>
state current/final             <StateNodeId>.current[.<MemberId>]
                                <StateNodeId>.final[.<MemberId>]
bundle projection               <BundleNodeId>.value.<MemberId>
gate/merge output               <NodeId>.value
reactor lifecycle               <ReactorNodeId>.pulse
                                <ReactorNodeId>.index
                                <ReactorNodeId>.first
                                <ReactorNodeId>.completed
effect output                   <NodeId>.effects.<EffectDomain>
```

## Performance lowering

The reactor is not interpreted as a generic event object in the hot path. It lowers to the compiled FSM's loop latch:

```text
working commit -> evaluate prebound continue slot -> bound check -> next block/final block
```

For Qwen the compiler emits one first-reaction block and one branch-free steady block. Reactor outputs become dense scalar/control slots or compile-time block facts; no string lookup, heap allocation, or YAML traversal occurs per reaction.
