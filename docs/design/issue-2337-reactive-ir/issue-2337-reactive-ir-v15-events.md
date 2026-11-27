# Cyclic reactive dataflow IR — v15 event grammar

v15 replaces boolean-expression firing and gate nodes with first-class presence events.

## Three distinct edge categories

| Edge | Meaning | Fan-out |
|---|---|---|
| value | immutable typed tensor/bundle data | allowed |
| event | presence/firing clock with no payload | allowed |
| effect | ordered access to an external/linear resource | forbidden |

An event is not an asynchronous message queue. Within one synchronous reaction it is either present once or absent.

## Reactor

```yaml
nodes:
  generation:
    kind: reactor
    limit: request.max_iterations
    continue: termination.continue
```

Outputs:

```text
generation.pulse       present once in every reaction
generation.first       present only in reaction 0
generation.steady      present only in reaction 1+
generation.index       int64 scalar value present with pulse
generation.completed   present once after the final working commit
```

`first` and `steady` are mutually exclusive and together equal `pulse` whenever a reaction exists.

## Component firing

```yaml
nodes:
  prefill:
    kind: component
    component: model
    when: generation.first
    inputs:
      input_ids: request.input_ids
      # ...

  sample:
    kind: component
    component: token_sampler
    when: generation.pulse
    inputs:
      logits: effective_logits.value
```

`when` is an `EventRef`, never a boolean expression.

A component fires iff:

1. its event is present;
2. every required input is present;
3. its effect inputs are present and owned by this firing.

If `when` is absent for a stable pure node, the node evaluates once on invocation demand. A reaction-local node whose data inputs are all stable must explicitly consume a reaction event through `when`; otherwise it is hoisted and runs once.

## Dynamic boolean branch

A switch converts a typed scalar boolean value into mutually exclusive events:

```yaml
nodes:
  route:
    kind: switch
    when: generation.pulse
    predicate: policy.use_tool
```

Outputs:

```text
route.then    present iff generation.pulse is present and predicate is true
route.else    present iff generation.pulse is present and predicate is false
```

```text
SwitchNode {
  kind: "switch",
  when: EventRef,
  predicate: ValueRef
}
```

The predicate must be a present rank-0 bool whenever `when` is present. Request-aligned `bool[batch]` remains ordinary tensor data in core; continuous-batching lifting belongs to its addon.

## Merge with event/value arms

```yaml
nodes:
  effective_logits:
    kind: merge
    arms:
      - when: generation.first
        value: prefill_logits.selected
      - when: generation.steady
        value: logits.current
```

```text
MergeNode {
  kind: "merge",
  arms: NonEmptyList<{
    when: EventRef,
    value: ValueRef
  }>,
  require: "exactly_one_present"
}
```

Output:

```text
<NodeId>.value
```

Rules:

- every arm value must be present whenever its arm event is present;
- arm value types must unify exactly, including structural bundle members;
- arm events must be statically mutually exclusive;
- for every firing of the merge's inferred parent event, exactly one arm must be present;
- the output is present on the union of arm events;
- no priority or YAML-order selection exists.

The compiler lowers merge to block-level phi/slot selection. It does not copy tensors.

## Gate node removed

This v14 shape:

```yaml
carried:
  kind: gate
  when: generation.steady
  value: logits.current
```

is removed. The value is named directly in a merge arm:

```yaml
- {when: generation.steady, value: logits.current}
```

For many values with the same lifecycle, first build one structural bundle and merge the bundle once.

## Structural bundle under events

```yaml
nodes:
  prefill_cache:
    kind: bundle
    when: generation.first
    members:
      key.0: prefill.present.0.key
      value.0: prefill.present.0.value

  effective_cache:
    kind: merge
    arms:
      - {when: generation.first, value: prefill_cache.value}
      - {when: generation.steady, value: decoder_cache.current}
    require: exactly_one_present
```

Bundle presence is all-or-nothing. Each member retains its own tensor signature.

## State update presence

A state has one candidate input binding:

```yaml
nodes:
  logits:
    kind: state
    initial: {kind: absent}
    next: decode_logits.selected
    transition: {kind: replace}
```

State semantics per reaction:

```text
if next is present:
  stage transition(current, next)
else:
  stage retain(current)
```

Therefore state needs no separate `when` field. Update presence is the presence of its `next` value, and the reactor—not missing writes—decides whether another reaction occurs.

Readable endpoints remain:

```text
logits.current
logits.final
```

`next:` creates no alias.

Presence validation rejects any read of an absent `state.current` on an event where the state has not yet been initialized or written. In Qwen, `logits.current` is read only in the `generation.steady` merge arm, after reaction 0 has committed `decode_logits.selected`.

## Event join

Control-only branches sometimes reconverge without merging a payload. `event_join` is the event analogue of merge:

```yaml
nodes:
  routed:
    kind: event_join
    inputs: [route.then, route.else]
    require: exactly_one_present
```

Output:

```text
routed.event
```

The inputs must be mutually exclusive children of a common parent event. The output is present when exactly one input is present. For the direct children of one switch, the output is equivalent to the switch input event; an optimizer may erase it.

## Effects under event branches

Effectful nodes use event firing plus linear effect tokens:

```yaml
nodes:
  tool_call:
    kind: component
    component: invoke_tool
    when: route.then
    inputs: {...}
    effects:
      tools: effects.tools.start

  no_tool:
    kind: effect_passthrough
    when: route.else
    effects:
      tools: effects.tools.start

  tools_done:
    kind: effect_merge
    arms:
      - {when: route.then, effect: tool_call.effects.tools}
      - {when: route.else, effect: no_tool.effects.tools}
```

Events may fan out; effect tokens may not. Branch effect arms must merge before their domain sink.

## Final lifecycle

```yaml
nodes:
  vae:
    kind: component
    component: vae_decoder
    when: generation.completed
    inputs:
      latent: latent.final
```

After the final reaction working commit:

1. state exposes a tentative `final` snapshot;
2. reactor emits `completed`;
3. completed-event postlude runs;
4. successful postlude performs invocation durable commit;
5. postlude failure discards the tentative final snapshot and aborts transactional effects/publications to the admission baseline;
6. `after_invocation_commit` effects run only after durable commit and cannot affect graph state or reactor continuation.

For `limit = 0`, `state.final` equals the admission baseline with its original presence bit. An initially absent state therefore remains absent; consumers must respect that presence.

## One-shot DAG

A graph without a reactor has one implicit invocation event:

```text
invocation.run
invocation.completed
```

- stable/demanded DAG nodes without `when` run once;
- nodes that need an explicit firing anchor may use `when: invocation.run`;
- state updates stage during this one working phase;
- `state.final` and `invocation.completed` appear after it;
- the postlude then runs before invocation durable commit.

These are schema-provided lifecycle events, not an authored clock block.

## Updated node union

```text
GraphNode = ComponentNode
          | StateNode
          | ReactorNode
          | SwitchNode
          | BundleNode
          | MergeNode
          | EventJoinNode
          | EffectMergeNode
          | EffectPassthroughNode
```

There is no gate node and no boolean-expression `when`.

## Presence validation

The compiler constructs an event-relation graph and proves:

- parent/child implication (`route.then -> generation.pulse`);
- sibling exclusion (`route.then` excludes `route.else`);
- sibling coverage (`then union else = parent`);
- reactor partition (`first union steady = pulse`);
- value availability under every consuming event;
- state first-write-before-read;
- merge/event/effect arm exclusion and coverage;
- final values are consumed only under completed/final events;
- no event depends instantaneously on itself.

A failure reports the consuming node/port, event path, unavailable value, and the missing switch/merge/state initialization needed to make the graph total.

## Performance lowering

Events become compile-time block membership and dense FSM transitions. They are not heap-allocated runtime objects. For Qwen:

```text
generation.first  -> precompiled prefill+sample+decode block
generation.steady -> precompiled sample+decode block
generation.completed -> finalization block
```

Merge nodes become slot aliases/phi selections, bundles become port tables, and event joins normally erase.
