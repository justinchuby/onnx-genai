# Cyclic reactive dataflow IR — v10 bundles and batching

v10 formalizes structural bundles and row semantics. It supplements the Qwen v9 graph.

## Structural bundle

A bundle is a compile-time structural record of independently typed values. It is not a tensor operation, concatenation, packed memory layout, or allocation request.

```yaml
nodes:
  prefill_cache:
    kind: bundle
    when: {value: reaction.first, equals: true}
    members:
      key.0: prefill.present.0.key
      value.0: prefill.present.0.value
      # ... through layer 23 ...

  carried_cache:
    kind: gate
    when: {value: reaction.first, equals: false}
    value: decoder_cache.current

  effective_cache:
    kind: merge
    inputs: [prefill_cache.value, carried_cache.value]
    require: exactly_one_present

  decode:
    kind: component
    component: model
    inputs:
      past_key_values.0.key: effective_cache.value.key.0
      past_key_values.0.value: effective_cache.value.value.0
```

Rules:

1. Bundle identity is its ordered/local member schema, not an artifact filename or model family.
2. Each member retains its own dtype, rank, shape constraints, batch mapping, and producer provenance.
3. Bundle construction and projection are zero-copy and erase to resolved port-slot tables.
4. Bundle presence is all-or-nothing. Partial member presence is invalid unless the bundle type explicitly marks that member optional.
5. Gate forwards or suppresses the whole bundle.
6. Merge requires identical member keys and pairwise compatible member contracts.
7. Bundle member names are package-local identifiers. No semantics are inferred from `key.0`, `value.0`, or other spelling.

## Transition-homogeneous state bundle

Members may share a state node only when these execution properties are identical:

- temporal presence and initialization behavior;
- ownership, scope, release boundary, and row partition;
- transition kind and its plan values;
- transition axis interpretation;
- aliasing legality;
- reuse and rollback/snapshot/fork behavior.

Member dtype/rank/non-transition dimensions may differ.

```yaml
nodes:
  decoder_cache:
    kind: state
    ownership: runtime
    scope: session
    release: session_end
    transition:
      kind: append
      axis: 2
      candidate_increment: cache_plan.candidate_count
      commit:
        kind: gather
        indices: cache_plan.indices
        length: cache_plan.commit_length
      bound: package.max_context
    aliasing: permitted
    reuse: {prefix: allowed, evict_prefix: forbidden}
    capabilities: {rollback_positions: 1, snapshot: true, fork: true}
    members:
      key.0:
        initial: empty_state.past_key_values.0.key
        next: decode.present.0.key
      value.0:
        initial: empty_state.past_key_values.0.value
        next: decode.present.0.value
```

KV, logical lengths, RNG, masks, grammar state, and token history remain separate state nodes when their transition behavior differs. Reaction/invocation transactions already provide atomicity across those nodes; bundling is not needed merely to make them commit together.

## Optional substitution interface

```yaml
interfaces:
  onnx-genai.attention-state:
    version: '1'
    bindings:
      layers:
        - {index: 0, key: key.0, value: value.0}
```

The interface labels canonical bundle members only to prove a legal optimized substitution. It cannot define the bundle schema, edges, transition, ordering, or physical storage.

## Batch transform contract

ONNX ValueInfo supplies tensor type and shape, but not request-row identity or row dependence. Every component that touches request-aligned values declares how rows transform.

```yaml
components:
  model:
    implementation: {kind: onnx, artifact: model.onnx}
    batching:
      kind: row_independent
      default_axis: 0
      unbatched_ports: []
      axis_overrides: {}
```

### `row_independent`

For every output row `i`, the value depends only on input row `i` and stable/unbatched inputs. The component neither mixes nor renumbers rows.

Consequences:

- rows may be independently admitted, cancelled, committed, compacted, and released;
- runtime may vectorize independent row transactions;
- request-aligned output mapping is propagated through the component;
- graph/state edges must agree on the resolved axis.

### `select`

```yaml
batching:
  kind: select
  source: tokens
  indices: row_selection
  output_axis: 0
```

Output row `j` comes from the source row named by the integer position `row_selection[j]`. Repeated source rows are allowed only if downstream state/effect ownership supports copy-on-write or independent cloning.

### `expand`

```yaml
batching:
  kind: expand
  source: tokens
  mapping: candidate_parent_rows
  output_axis: 0
```

One source row may produce several candidate rows. The explicit mapping is tensor data; expansion factor is not inferred from shape or model identity.

### `collective`

```yaml
batching:
  kind: collective
  input_axis: 0
  output_axis: 0
```

An output row may depend on multiple input rows. All related rows form one transaction partition for that component/state dependency. A runtime may not commit, cancel, reorder, or compact them independently across the collective boundary.

## Port mapping

`default_axis` is a component contract, not a guess from a dimension called `batch`. Exceptions are explicit:

```yaml
batching:
  kind: row_independent
  default_axis: 0
  unbatched_ports:
    - max_iterations
    - schedule
  axis_overrides:
    channels_last_output: 1
```

Validation requires every graph-visible tensor port to resolve to exactly one of:

- request-aligned with a concrete axis and transform provenance;
- stable/unbatched;
- explicitly collective/global.

No unresolved row ownership is admitted to shared batching.

## State row partition

A state member inherits row mapping from its `initial`, current consumers, and `next` producer. Those mappings must agree.

```text
request-aligned state
  -> one logical transaction per row

collective/global state
  -> one transaction partition covering all dependent rows
```

A transition-homogeneous bundle additionally requires every member to have the same row partition. A K/V member that is row-independent cannot share a bundle with a global scheduler scalar.

## Dynamic commit-plan binding

The provisional `state_commit_plan` is a row-independent binding:

```yaml
components:
  state_commit_plan:
    implementation: {kind: binding}
    contract: {id: onnx-genai.state-commit-plan, version: '1'}
    batching:
      kind: row_independent
      default_axis: 0
      unbatched_ports: [first]
    ports:
      inputs:
        first: {dtype: bool, rank: 0, shape: []}
        prompt_candidate_count: {dtype: int64, rank: 1, shape: [batch]}
        prompt_indices: {dtype: int64, rank: 2, shape: [batch, max_candidates]}
        prompt_length: {dtype: int64, rank: 1, shape: [batch]}
        accepted_length: {dtype: int64, rank: 1, shape: [batch]}
      outputs:
        candidate_count: {dtype: int64, rank: 1, shape: [batch]}
        indices: {dtype: int64, rank: 2, shape: [batch, max_candidates_plus_one]}
        commit_length: {dtype: int64, rank: 1, shape: [batch]}
```

`prompt_*` inputs are dynamically optional and absent after reaction 0. The binding contract must declare that presence relation explicitly; `first=false` selects the steady-state result without reading absent prompt inputs.

The binding may lower to constant-folded descriptors, host metadata arithmetic, device kernels, or fused planner logic. Its graph-visible outputs and validation rules are invariant across those implementations.

## Performance consequence

Bundles and batch contracts are compile-time planning information:

- bundle construction/projection creates no tensor copies;
- row transforms lower to precomputed gather/selection descriptors;
- row-independent components retain static batching and graph-capture eligibility;
- collective boundaries are visible before scheduling rather than discovered after an unsafe optimization;
- no artifact-port prefix matching or model-specific batch rule is required.
