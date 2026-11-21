# Cyclic reactive dataflow IR - v16 inferred firing

v16 decides when `when` is authored and makes the event proofs used by that
decision precise. It refines v15; it is not yet the consolidated schema.

## Decision

A component or bundle normally fires on the common presence event of its
required inputs. Authors write `when` only to:

- anchor otherwise-stable inputs to a lifecycle event; or
- intentionally restrict execution to a strict subevent of the event on which
  all required inputs are available.

An explicit `when` equal to the inferred event is invalid redundant syntax.
Inference never invents an unnamed control event or chooses one of several
non-equivalent events.

The principle is:

```text
required inputs constrain when work can run
when adds information that data availability does not contain
```

Requiring both the inputs and an equivalent `when` would serialize the same
fact twice. Inferring an event not represented by the event graph would instead
hide authored control. The canonical grammar does neither.

## Event relations

Every named event has a normalized symbolic support set. The compiler proves
three relations:

```text
A implies B       every occurrence of A is an occurrence of B
A excludes B      A and B cannot be present in the same reaction
cover(P, arms)    the arms are pairwise exclusive and their union equals P
```

The proof graph starts with schema-defined lifecycle facts:

```text
generation.first union generation.steady = generation.pulse
generation.first excludes generation.steady
generation.completed excludes generation.pulse
```

For a one-shot DAG, `invocation.run` and `invocation.completed` are distinct
ordered lifecycle events. Completion is not treated as a child occurrence of
`invocation.run`.

A switch introduces one fresh partition:

```text
route.then implies route parent
route.else implies route parent
route.then excludes route.else
route.then union route.else = route parent
```

The compiler does not infer that two switches are correlated merely because
their predicates reference the same value. Event equivalence comes from event
constructors, not arbitrary reasoning about tensor values.

An `event_join` is valid only when its inputs are pairwise exclusive and their
union is statically equal to one parent support set. Its output is an event
alias for that union. A value `merge` uses the same exclusion and coverage
proof, in addition to its value-type checks.

Implementations may normalize these facts with a reduced decision diagram or
an equivalent finite representation. The required checks are:

```text
implies(A, B) = unsatisfiable(A and not B)
excludes(A, B) = unsatisfiable(A and B)
covers(P, arms) =
    equivalent(P, union(arms))
    and pairwise_exclusive(arms)
```

These formulas are load-time proof objects only. They are not serialized
expression syntax and are not interpreted in the reaction hot path.

## Value availability

Each resolved value has one availability:

```text
stable
event support
statically absent
unresolved conditional presence
```

- Workflow inputs and pure closures over only workflow inputs are stable.
- An ordinary component output is available on its component's resolved firing
  event, unless its signature declares a narrower conditional presence rule.
- A bundle output has the firing event inferred from all of its members.
- A merge output is available on the union of its arm events.
- Reactor values such as `index` are available on the lifecycle event specified
  by the reactor contract.
- State endpoints additionally require the temporal presence proof below.

Binding signatures must classify every input as required or optional. A
conditional output rule must be stated in the binding signature in terms of
port-presence relationships that can be instantiated at the call site. ONNX
outputs are present whenever their node fires unless an applicable semantic
contract says otherwise.

An unresolved conditional output cannot silently become a scheduling event.
It must be normalized by an explicit event-producing construct or consumed as
an optional value under an already known event.

## Firing inference

For a component, collect the availability of:

1. every required value input; and
2. every required effect input whose producer support is already resolved.

For a bundle, every member is required.

Optional value inputs never determine firing. Their presence at the resolved
firing event is checked against the component signature and passed explicitly
as present or absent.

Stable inputs impose no event constraint. For the remaining required inputs,
compute their exact intersection `M`.

### Omitted `when`

An omitted `when` is valid when exactly one of these cases holds:

1. Every required input is stable, and the node is pure. The node is stable and
   evaluates at most once on invocation demand.
2. `M` is satisfiable and is equivalent to a named event support set. The node
   fires on `M`.

Multiple names that are proven aliases of the same support set are not
ambiguous. The resolved plan stores the normalized support, not a preferred
spelling.

An omitted `when` is invalid when:

- required inputs are mutually exclusive, making `M` empty;
- `M` is not equal to any named event; or
- a required input has unresolved conditional presence.

The compiler does not create an implicit conjunction event. If the intended
intersection is not named, the author must make the event topology explicit,
for example by nesting the relevant switch under the other event. A future
conjunction node, if justified by real models, would be an explicit event
constructor rather than a special case in firing inference.

### Explicit `when`

For `when: W`, the compiler proves:

```text
W is satisfiable
W implies availability(input) for every required value and effect input
```

The annotation is valid only when:

- all required inputs are stable and `W` anchors the work to a lifecycle event;
  or
- `W` is a strict subevent of `M`.

If `W` is equivalent to `M`, validation rejects it as redundant. If `W` does
not imply every required input's availability, the node could observe an absent
required input and is invalid.

This makes `when` a restriction, never a fallback value, priority rule, or
second description of the same schedule.

## Node-specific rules

### Components

Components use the general inference rule. A component with stable inputs that
must execute in reaction zero writes:

```yaml
prefill:
  kind: component
  component: model
  when: generation.first
  inputs:
    input_ids: request.input_ids
```

A component whose required input is already pulse-valued omits `when`:

```yaml
sample:
  kind: component
  component: token_sampler
  inputs:
    logits: effective_logits.value
```

An effectful component must bind every effect domain it uses. It cannot be
classified as a stable pure node merely because its value inputs are stable.

### Bundles

A bundle applies the same inference to all members. Construction remains
all-or-nothing and zero-copy:

```yaml
prefill_cache:
  kind: bundle
  members:
    key.0: prefill.present.0.key
    value.0: prefill.present.0.value
```

If both members are present on `generation.first`, the bundle infers that event.
A mixed-lifecycle bundle must have a named common event or first normalize its
members through merges.

### Switches

A switch must continue to declare `when`.

```yaml
route:
  kind: switch
  when: generation.pulse
  predicate: policy.use_tool
```

The switch is an event constructor: `when` names the parent event that its
outputs partition. Predicate availability only proves that the partition can
be evaluated; it does not itself declare which event the switch is intended to
partition. The predicate must be present whenever `when` is present.

### Merge and event join

`merge` and `event_join` do not accept node-level `when`. Their arm/input events
define their support structurally. Arm exclusion and exact parent coverage are
mandatory; YAML order is never a priority rule.

### State

State does not accept `when`. Presence of `next` decides update versus retain:

```text
next present -> stage the declared transition
next absent  -> stage retain(current)
```

A stable `next` is sampled in every occurrence of the graph's sole working
phase. For a one-shot DAG it is sampled once during `invocation.run`.

State availability is a temporal proof layered on top of event support:

- a present initializer makes `current` available on the owning working event;
- an absent initializer contributes no initial presence;
- a total successful write on every reaction-zero path makes `current`
  available on a later `steady` event;
- once present, an absent later `next` retains presence;
- `final` is available on completion only when every completing path, including
  the zero-reaction path, has a present state.

The compiler solves these facts to a fixed point across state-delay edges after
instantaneous event/value inference. A required read cannot use an unproven
state presence to bootstrap its own firing event.

After durable commit, the same immutable `state.final` snapshot may be retained
for an `after_invocation_commit` effect. This retained readability is not a new
value-production event and never causes a component to fire again. The
component's committed-phase effect root supplies its firing event.

An initially absent state consumed under `completed` therefore still fails when
`limit = 0` can complete without a write. The package must make the input
optional, provide an initializer, or establish an admission constraint that
makes presence total.

### Reactor

The reactor does not accept `when`. Its stable `limit`, reaction-valued
`continue`, admission trigger, and working-commit latch define its lifecycle
events. Reactor events are roots and partitions in the proof graph.

### Effects

`effect_passthrough` uses an explicit `when` because it intentionally restricts
a parent effect token to one event branch. `effect_merge` derives support from
its arms and accepts no node-level `when`.

Linearity is per event occurrence. Two textual consumers of an effect family
are not illegal fan-out when their events are proven mutually exclusive and
collectively cover the producer event. For each event occurrence, exactly one
consumer receives the unique token. Missing coverage, overlapping consumers,
or an unordered same-event pair is invalid.

v17 resolves the effect-root question: each declared root produces a fresh
token per occurrence of one named event, and reaction/completion roots are
separate. Core v1 forbids cross-reaction effect carry. If a future use case
requires it, the delay must be explicit rather than implied by an ordinary
effect reference.

## One-shot DAG consequence

For a graph without a reactor:

- a pure component or bundle over stable inputs runs once on demand;
- `when: invocation.run` is required when stable-input work belongs to the
  working transaction rather than the stable preheader;
- a node consuming a guaranteed `state.final` may infer
  `invocation.completed`;
- writing an equivalent `when: invocation.completed` is redundant;
- state updates stage during the one working phase and become tentative final
  state before the postlude.

The exact admission-versus-working failure boundary and zero-work state-final
contract remain to be finalized in the consolidated schema.

## Qwen consequence

The common path becomes less repetitive:

```yaml
effective_logits:
  kind: merge
  arms:
    - {when: generation.first, value: prefill_logits.selected}
    - {when: generation.steady, value: logits.current}
  require: exactly_one_present

sample:
  kind: component
  component: token_sampler
  inputs:
    logits: effective_logits.value

decode:
  kind: component
  component: model
  inputs:
    input_ids: sample.token

decode_cache:
  kind: bundle
  members:
    key.0: decode.present.0.key
    value.0: decode.present.0.value
```

The merge output is present on `generation.pulse`, so `sample`, `decode`, and
`decode_cache` infer `generation.pulse`. Prefill still declares
`when: generation.first` because its prompt inputs are stable. A postlude
consumer of guaranteed final state similarly infers `generation.completed`.

The Qwen commit-plan component's prompt inputs are optional on steady
reactions; they do not determine firing. A required pulse-valued input, such as
the accepted-length result, anchors the component. Its binding signature must
state that prompt fields are accepted only on the first-event subset.

## Validation order

The relevant loader passes are:

1. resolve component signatures and required/optional ports;
2. build lifecycle and explicit switch/event-join relations;
3. resolve instantaneous value availability in topological order;
4. infer or validate component and bundle firing;
5. instantiate conditional port-presence rules;
6. solve temporal state presence across delay edges;
7. revalidate every state read and final consumer;
8. prove merge and effect exclusion, coverage, and linearity;
9. lower normalized event supports to prebound blocks and FSM transitions.

Lowering records dense block membership, slot presence, and transition IDs.
There is no runtime event-formula evaluation, YAML traversal, string lookup, or
per-reaction graph allocation.

## Required diagnostics

Examples of actionable failures:

```text
node combine: required inputs `left` (route.then) and `right` (audit.then)
overlap, but their intersection is not a named event. Nest one switch under the
other event or add an explicit event constructor for the intended conjunction.
```

```text
node invalid: required inputs `a` (route.then) and `b` (route.else) are
mutually exclusive. Merge the alternative values before consuming them.
```

```text
node prefill: `when: generation.pulse` does not imply required input `logits`
availability `generation.first`. Use `generation.first` or merge a steady
alternative.
```

```text
node sample: `when: generation.pulse` duplicates the event inferred from
required input `effective_logits.value`. Remove `when`.
```

```text
node policy: optional input `prompt_plan` cannot establish firing. Make the
port required or declare a lifecycle `when`.
```

```text
state `logits.current` is read on `generation.steady`, but no successful write
is proven on every preceding first-reaction path. Initialize the state or make
the reaction-zero write total.
```

```text
effect `tools.start` is available on `generation.pulse`, but its consumers
cover only `route.then`. Add the else passthrough and merge both effect arms.
```

## Superseded v15 examples

The following v15 annotations are no longer canonical:

- `sample.when: generation.pulse` when `effective_logits.value` is pulse-valued;
- `prefill_cache.when: generation.first` when all members are first-valued;
- `vae.when: generation.completed` when its required final input is guaranteed
  completed-valued.

Prefill's explicit first-event anchor remains canonical because its required
request inputs are stable. Switch parents and merge arms also remain explicit
because they define event topology rather than repeat value availability.
