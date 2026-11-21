# Cyclic reactive dataflow IR - v17 event-scoped effects

v17 defines the effect-root model left open by v15 and v16. It refines those
drafts; it is not yet the consolidated schema.

## Decision

Each declared effect root produces one fresh linear token for each occurrence
of one named event:

```text
effect root + event occurrence -> one linear token instance
```

The token must be consumed and returned to that root's sink within the same
event occurrence. Reactor ordering determines that reaction `i` finishes its
working commit before reaction `i + 1` starts, but no effect token is implicitly
carried between those reactions.

Core v1 has no cross-reaction effect delay. A future use case that requires one
must introduce an explicit delayed effect construct with its own initialization,
zero-reaction, failure, and final-sink semantics. It must not reinterpret an
ordinary effect edge as a loop-carried capability.

The principle is:

```text
event support decides when a linear capability exists
effect edges decide ordering within that occurrence
transaction policy decides when its externally visible work commits
```

These are separate questions. A token is not a transaction object, event queue,
or persistent resource handle.

## Domain declarations

An effect domain declares externally observable behavior, not its graph entry
points:

```yaml
effects:
  tools:
    commit_behavior: transactional
    retry: idempotent
    speculation_safety:
      kind: forbidden

  audit:
    commit_behavior: after_invocation_commit
    retry: idempotent
    speculation_safety:
      kind: not_applicable
```

```text
EffectDomain {
  commit_behavior: CommitBehavior,
  retry: RetryBehavior,
  speculation_safety: SpeculationSafety
}

CommitBehavior =
    "pure"
  | "transactional"
  | "after_invocation_commit"
```

Domain policy is declared once even when the graph has several disjoint roots
for that domain.

## Root and sink grammar

Roots and sinks are graph-local because their events are graph-local:

```yaml
graph:
  effect_roots:
    tools_reaction:
      domain: tools
      when: generation.pulse
      sink: tools_reaction_done.effects.tools

    tools_completion:
      domain: tools
      when: generation.completed
      sink: tools_completion_done.effects.tools

    audit_committed:
      domain: audit
      when: invocation.committed
      sink: audit_write.effects.audit
```

```text
EffectRootId := graph-local string

EffectRoot {
  domain: EffectDomainId,
  when: EventRef,
  sink: EffectRef
}
```

Each root generates one canonical effect reference:

```text
effects.<EffectRootId>
```

Node effect outputs remain:

```text
<NodeId>.effects.<EffectDomainId>
```

There is exactly one sink inside every root declaration. A sink names the final
token for the same root instance; it does not merge different roots. Keeping
the terminal reference with its source avoids a second table keyed by the same
root ID.

The root ID, rather than only the domain, is preserved in the resolved effect
type:

```text
EffectToken<domain, root, event-occurrence>
```

Consequently, tokens from two roots in the same domain cannot be accidentally
merged or substituted.

## Lifecycle events

Effect roots may be scoped to any explicit named event whose support is proven
by the v16 event-relation graph. Typical roots are:

```text
invocation.run
generation.first
generation.steady
generation.pulse
generation.completed
invocation.completed
invocation.committed
```

Completion has one public spelling per graph:

- a reactor graph uses `<ReactorNodeId>.completed`;
- a one-shot DAG uses the schema-provided `invocation.completed`.

The two forms have the same lifecycle position but are never simultaneously
present in one graph. After the final working commit, the applicable completion
event exposes tentative final state and runs the transactional postlude.
Postlude failure can still abort to the admission baseline.

`invocation.committed` is emitted exactly once after the invocation durable
commit succeeds. It is not a child occurrence of `invocation.completed`; it is
a later transaction phase. It cannot reach state `next`, reactor `continue`,
publication decisions, or any value/effect that is required to perform the
already completed durable commit.

There is no completion alias or child relation to prove in single-reactor v1.
This avoids two public names for the same event support.

## Linearity within an event occurrence

For one root occurrence, exactly one live token exists along every reachable
event path.

An ordinary same-event chain is single-consumer:

```yaml
first_write:
  kind: component
  component: write_a
  effects:
    tools: effects.tools_reaction

second_write:
  kind: component
  component: write_b
  effects:
    tools: first_write.effects.tools
```

A branch may name the same incoming reference in several textual consumers
only when their firing events are a statically proven exclusive and complete
partition of the token's support:

```yaml
tool_call:
  kind: component
  component: invoke_tool
  when: route.then
  inputs: {...}
  effects:
    tools: effects.tools_reaction

no_tool:
  kind: effect_passthrough
  when: route.else
  effects:
    tools: effects.tools_reaction

tools_reaction_done:
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
  arms: NonEmptyList<{
    when: EventRef,
    effect: EffectRef
  }>
}
```

An effect merge has no node-level `when`. Its arm events must be pairwise
exclusive, exactly cover one named parent support, and carry the same resolved
`EffectToken<domain, root>`. Its output support is that parent.

`effect_passthrough.when` is explicit because it intentionally restricts the
incoming token to one branch. It cannot be inferred from the effect input,
whose support is the wider parent event.

This conditional textual reuse is static routing, not runtime fan-out. If two
consumer events overlap, if the arms do not cover the input support, or if
there is no reconverged token at the sink, validation fails.

## Several roots in one domain

One domain may declare several roots only when their event supports are
pairwise exclusive or transaction-phase ordered:

- `generation.first` and `generation.steady` are exclusive occurrences;
- `generation.pulse` and `generation.completed` are lifecycle-exclusive;
- `invocation.completed` precedes `invocation.committed` by the durable-commit
  boundary.

Two roots for the same domain on overlapping events would provide unordered
access to the same external domain and are invalid. Authors who need two
operations on the same event use one token chain, not two roots.

Roots of different domains are independent unless a component explicitly
consumes both tokens. Consuming both imposes a join at that component; their
supports must have a named common event under the v16 firing rules.

## Firing inference

A root effect reference has the availability of its declared `when` event.
Effect outputs have the resolved firing support of their producing node.

Required effect inputs participate in v16 firing inference exactly like
required value inputs:

- stable values plus `effects.tools_reaction` infer the root event;
- a node with `when: route.then` may consume a pulse-scoped token because the
  child event implies the parent;
- a node cannot consume a first-scoped token on `generation.pulse`;
- optional linear effect ports are forbidden in core v1.

An ordinary effectful component may omit `when` when its required value/effect
inputs infer the exact intended event. It uses an explicit `when` only for a
strict subevent. `effect_passthrough` is the structural exception described
above.

## No cross-reaction token edges in v1

Every effect reference has both an event support and an occurrence identity.
Validation rejects an edge that would make a token produced in one occurrence
available in another occurrence, including:

- feeding an effect output through state;
- using a first-reaction token on a steady reaction;
- using a reaction `i` output as reaction `i + 1` input;
- sinking one root occurrence only after reactor completion;
- treating one `generation.pulse` root token as a single invocation-wide token.

For a root on `generation.pulse`, each pulse independently executes the
root-to-sink chain before that reaction's working commit resolves:

```text
pulse(i)
  -> fresh root token(i)
  -> exclusive/complete effect chain(i)
  -> sink token(i)
  -> working commit(i)
```

The compiled plan may reuse the same dense token slot across reactions because
occurrences do not overlap. That storage reuse does not change the logical
token identity.

## Commit behavior

### Pure

A `pure` domain has no externally visible mutation to commit or roll back, but
its token may order access to a runtime capability that cannot be duplicated.
Its chain must still satisfy linearity. Failure aborts the current phase like
an ordinary component failure.

### Transactional

A `transactional` domain may have roots under working or completion events, but
not under `invocation.committed`.

- Work performed under a reaction event joins that reaction's working
  savepoint.
- A successful reaction sink is required before the reactor may sample
  `continue` and advance.
- Reaction working commit advances tentative effect state together with graph
  state.
- Work under the graph's completion event joins the tentative postlude.
- Invocation durable commit atomically persists all tentative transactional
  effect work.
- Any pre-commit failure or cancellation aborts transactional work to the
  admission baseline.

The token orders operations; the domain's transaction implementation owns the
savepoint/commit/abort mechanism. The IR does not serialize database handles,
transaction IDs, or physical resource identities.

### After invocation commit

An `after_invocation_commit` domain may root only at
`invocation.committed`. Every node in that phase is anchored by the committed
effect root; retaining a value does not independently cause another firing.
Its nodes:

- may consume stable values, publications already durably selected, and
  immutable `state.final` snapshots retained for post-commit use;
- cannot produce state updates, reactor control, transactional effects, or
  publication operations;
- cannot affect whether durable commit occurs;
- cannot roll back durable state or publications on failure.

Failure is reported as a post-commit effect failure with the domain, root,
node, and retry contract. A retry-capable runtime may retry according to the
declared policy without rerunning the invocation.

## Zero-reaction behavior

When a reactor limit is zero:

- no `first`, `steady`, or `pulse` root occurrence exists;
- no working effect chain runs;
- `generation.completed` occurs once;
- completion-scoped transactional chains may run against the admission
  baseline and tentative final presence;
- completion-scoped publications and transactional effects follow their
  ordinary graph semantics;
- no state samples `next` or stages an update because there is no pulse;
- `invocation.committed` occurs only after that postlude and the durable
  commit succeed;
- after-commit roots then occur once.

Thus "zero reactions" means no pulse work, not skipping the completion or
commit lifecycle. This follows event presence directly; roots need no special
zero-limit flag.

For a one-shot DAG, `invocation.run` and `invocation.completed` each occur once.

## Validation

After event and node firing resolution, the loader:

1. resolves every effect domain and root;
2. assigns each root a domain, event support, transaction phase, and occurrence
   class;
3. rejects root/event combinations incompatible with commit behavior;
4. rejects overlapping unordered roots in one domain;
5. verifies every effect input and output preserves domain and root identity;
6. proves each conditional consumer set is exclusive and exactly covers the
   incoming token support;
7. proves every merge combines one root identity and exactly reconstructs its
   parent support;
8. proves every root-to-sink path has exactly one live token at each event
   point;
9. rejects any edge across a reactor, state, or durable-commit boundary unless
   the boundary is the declared source of a fresh root;
10. verifies after-commit non-interference;
11. lowers each root chain to dense block-local dependency slots.

Required diagnostics include:

```text
effect root `tools_reaction` is scoped to `generation.pulse`, but consumers
cover only `route.then`. Add an exclusive `route.else` passthrough and merge
both arms before the root sink.
```

```text
effect token `tool_call.effects.tools` belongs to root `tools_reaction` at
reaction occurrence i, but node `carry` would make it available at occurrence
i+1. Core v1 does not support cross-reaction effect carry.
```

```text
effect roots `write_a` and `write_b` both access domain `store` on
`generation.pulse`. Chain the operations from one root or place them on proven
exclusive events.
```

```text
effect root `audit` uses commit behavior `after_invocation_commit` but is
scoped to `generation.completed`, before durable commit. Scope it to
`invocation.committed`.
```

```text
effect merge `tools_done` combines roots `tools_reaction` and
`tools_completion`. Merge arms must preserve one root identity; sink each root
separately.
```

## Performance lowering

Effect roots and tokens are compile-time dependency identities. A prebound
block receives a fixed token slot for each root active on that event. Branches
select statically compiled successor blocks, and effect merges become control
joins.

The hot path performs no root lookup, graph traversal, event allocation,
dependency discovery, or per-reaction heap allocation. A pulse-scoped root may
reuse preallocated storage after each occurrence reaches its sink and working
commit.
