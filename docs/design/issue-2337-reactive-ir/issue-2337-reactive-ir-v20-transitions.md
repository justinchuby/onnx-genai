# Cyclic reactive dataflow IR - v20 state-transition invariants

v20 finalizes the logical meaning of `full`/`delta` candidates and the dynamic
commit-plan surface.

## Decision

State transition syntax declares only:

- how committed state relates to current state;
- what representation the candidate producer returned; and
- which logical candidate ordinals commit.

Dynamic policy remains ordinary typed dataflow. A binding or ONNX component
computes counts and indices; the transition references those outputs. Core v1
has no `commit_plan` graph node, opaque plan handle, conditional transition
arms, or embedded expression language.

## Common selection

Append and indexed scatter select candidate ordinals with one of:

```text
CommitSelection =
    {kind: "prefix", length: ValueRef}
  | {kind: "gather", indices: ValueRef, length: ValueRef}
```

Let `C` be candidate count and `K` be committed count:

```text
0 <= K <= C
```

Prefix selects:

```text
[0, 1, ..., K - 1]
```

Gather reads the first `K` logical entries of `indices`. Every selected index
must satisfy:

```text
0 <= indices[j] < C
```

Gather order is commit order. Repeated indices are permitted for append because
ordinary gather semantics may intentionally duplicate a candidate. A runtime
substitution must preserve that behavior or reject the capability.

Counts and indices are typed component outputs. Their tensor contracts must
align with the participating state under normal symbolic-shape rules. Core may
execute request-aligned lengths in lockstep; the continuous-batching addon, not
the transition grammar, proves independent row lifting.

## Replace

```yaml
transition: {kind: replace}
```

The candidate is the complete next logical state. There is no candidate-source
or commit-selection axis because partial replacement would be append/scatter or
an explicit component computation.

## Append

Append commits an ordered candidate sequence after current logical state.

```text
AppendCandidateSource =
    {kind: "full", suffix_length: ValueRef}
  | {kind: "delta", length: ValueRef}

AppendTransition {
  kind: "append",
  axis: unsigned integer,
  candidate_source: AppendCandidateSource,
  commit: CommitSelection,
  bound: ValueRef
}
```

Let current logical state be `A`, candidate ordinals be `X[0..C)`, and selected
ordinals be `S`:

```text
result = A ++ gather(X, S)
```

### Delta append source

For `kind: delta`, the producer returns exactly `X`, and `length` is `C`.
No concatenation with current state is implied at the component boundary.

### Full append source

For `kind: full`, the producer returns:

```text
A ++ X
```

and `suffix_length` is `C`. The old prefix is logically identical to current
state. Rejected candidate suffix positions never become committed.

The loader does not compare tensor contents. The prefix relationship must be
proven by one of:

- a resolved component contract that relates an output to a bound current-state
  input;
- a state interface that declares an equivalent runtime-managed representation;
  or
- structural alias provenance produced by an already validated adapter.

The proof includes axis, member mapping, prefix preservation, and candidate
extent. An unproven author's assertion is not sufficient.

For either representation:

```text
logical_length(result) = logical_length(A) + K
logical_length(result) <= bound
```

The implementation may adopt/alias a full buffer or append a delta in place
only when state aliasing and rollback capabilities permit it. Representation
does not change logical results.

## Indexed scatter

Indexed scatter replaces existing logical positions without changing logical
length:

```text
ScatterCandidateSource =
    {kind: "full", count: ValueRef}
  | {kind: "delta", count: ValueRef}

IndexedScatterTransition {
  kind: "indexed_scatter",
  axis: unsigned integer,
  candidate_source: ScatterCandidateSource,
  destinations: ValueRef,
  commit: CommitSelection,
  logical_length: ValueRef,
  capacity: ValueRef
}
```

`destinations[j]` is the logical destination for candidate ordinal `j`.
For every candidate:

```text
0 <= destinations[j] < logical_length <= capacity
```

Scatter does not extend state. A workflow that adds logical positions uses
append and updates any separate logical-length state explicitly.

### Delta scatter source

A delta source returns `X[0..C)`. Selected candidate `j` writes:

```text
result[destinations[j]] = X[j]
```

Selected destinations must be unique. This rejects ambiguous multiple writes
to one logical position in a single transition.

### Full scatter source

A full source returns one complete materialized candidate buffer `F`:

- outside all declared destinations, `F` is logically identical to current;
- at `destinations[j]`, `F` contains candidate `X[j]`.

All declared destinations must be unique, not only selected ones, because one
materialized buffer cannot represent two distinct candidates for the same
position.

For selected candidate `j`:

```text
result[destinations[j]] = F[destinations[j]]
```

Unselected positions retain current state even if `F` materialized candidate
values there. Full-source preservation is proven through the same contract,
interface, or alias provenance used by full append; it is never established by
runtime content comparison.

## Structural bundles

A state bundle remains transition-homogeneous:

- every member uses the same transition kind;
- every member uses the same full/delta representation;
- selection counts and ordinals are shared;
- append/scatter semantics apply member-wise.

Member tensor signatures, axes, physical handles, and layouts may differ when a
resolved state interface maps the shared logical transition to each member.
Without such an interface, the declared axis must be valid and semantically
equivalent for every member.

Bundle construction performs no concatenation, gather, or scatter. The state
implementation performs the transition on member slots/handles.

## Canonical dynamic plan

Qwen uses an ordinary component:

```yaml
components:
  cache_commit_policy:
    implementation: {kind: binding}
    contract: {id: onnx-genai.state-commit-policy, version: '1'}
    signature:
      inputs:
        prompt_candidate_count: {required: false, dtype: int64, rank: 0}
        prompt_indices: {required: false, dtype: int64, rank: 1}
        prompt_length: {required: false, dtype: int64, rank: 0}
        accepted_length: {required: true, dtype: int64, rank: 0}
      outputs:
        candidate_count: {dtype: int64, rank: 0}
        indices: {dtype: int64, rank: 1}
        commit_length: {dtype: int64, rank: 0}

graph:
  nodes:
    cache_plan:
      kind: component
      component: cache_commit_policy
      inputs:
        prompt_candidate_count: prefill_context.prompt_candidate_count
        prompt_indices: prefill_context.prompt_commit_indices
        prompt_length: prefill_context.prompt_commit_length
        accepted_length: acceptance.length

    decoder_cache:
      kind: state
      initial: {kind: value, value: admission_cache.value}
      next: decode_cache.value
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

`acceptance.length` is pulse-valued and anchors `cache_plan`. Prompt fields are
optional and present only on `generation.first`, as declared by the binding
contract.

Reaction zero computes valid prompt candidate ordinals followed by the sampled
token ordinal when accepted. Steady reactions compute `[0]` when the sampled
token is accepted. The transition remains one static gather form in both
blocks; the policy component changes only its typed outputs.

This separation keeps:

```text
candidate production    component semantics
accept/reject policy    ordinary typed dataflow
state mutation          transition semantics
physical pages/slots    runtime lowering
```

## Validation diagnostics

Failures identify the state, transition, candidate producer, event, and bad
operand. Examples:

```text
state `decoder_cache`: full append candidate
`decode_cache.value` has no proven prefix relationship to
`decoder_cache.current`. Bind a component contract/interface that declares the
extension or produce a delta candidate.
```

```text
state `cache`: gather commit length is 4 but candidate count is 3 on
`generation.first`. Commit length must be in [0, candidate_count].
```

```text
state `buffer`: selected scatter candidates 1 and 3 both target logical
position 7. One transition cannot commit multiple writes to the same
destination.
```

```text
state `buffer`: scatter destination 12 is outside logical length 10. Use append
to extend state or correct the logical destination.
```

```text
state bundle `decoder_cache`: member `value.7` is delta while member `key.7`
is full. Split the state or adapt members to one candidate representation.
```
