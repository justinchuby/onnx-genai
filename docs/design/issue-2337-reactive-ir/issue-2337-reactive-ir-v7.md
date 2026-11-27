# Cyclic reactive dataflow IR — v7 execution model

v7 consolidates the decisions after temporal, batching, output, and performance review.

## 1. Synchronous firing rule

Within one availability class (`stable`, `reaction`, or `final`):

1. The compiler removes state-delay edges and obtains an instantaneous DAG.
2. Each node may fire at most once for that invocation/reaction/finalization.
3. A component fires when every required data/effect input is present and its optional scalar `when` is present and true.
4. A false `when` produces absent outputs; absence is not a zero tensor, stale tensor, or implicit hold.
5. Optional component ports may be statically absent if their ONNX/binding signature permits it.
6. A maybe-absent value may feed only an equally gated consumer, an optional port, or an explicit `merge`/`effect_merge`.
7. The reaction is complete only when every required `state.next`, publication batch, effect sink, and repeat predicate is resolved.
8. Unresolved required values are an actionable error, not silent node skipping.

This is synchronous reactive dataflow—not a Kahn process network and not an asynchronous event bus. Nodes do not repeatedly consume tokens within one reaction.

## 2. Clock

```yaml
graph:
  clock:
    kind: synchronous
    max_reactions: request.max_iterations
    index: reaction.index
    repeat_while: termination.continue
    sample_repeat: after_commit
```

Normative behavior:

- `max_reactions` is a required non-negative int64 scalar hard bound.
- If it is zero, no reaction runs; initialized state becomes `state.final` and `clock.completed` becomes present.
- Reaction 0 observes `reaction.index = 0`.
- After reaction `i` commits, another reaction runs iff `repeat_while` is true and `i + 1 < max_reactions`.
- Reaching the bound is successful clock completion, not an implicit error.
- The repeat predicate is computed during reaction `i` and sampled after its commit.
- Per-row active/EOS/limits remain tensor data; the global hard bound does not replace them.

## 3. Lifecycle without authored phases

```text
stable values:
  request/package inputs and pure closures depending only on them

reaction values:
  reaction.index, state.current, and dependent node outputs

final values:
  state.final, clock.completed, and dependent node outputs
```

- `state.initial` accepts stable values only.
- `state.next` accepts reaction values only.
- `state.current` is visible during every reaction.
- `state.final` becomes present once after the final commit—or immediately from initialized state when `max_reactions = 0`.
- Consuming `state.final`/`clock.completed` derives the postlude automatically.

```yaml
nodes:
  vae:
    kind: component
    component: vae_decoder
    inputs:
      latent: latent.final
```

## 4. Transaction scope

The single clock defines one logical reaction transaction, partitioned by ownership:

- Every request-aligned row commits or aborts all of its state, effects, and publications atomically.
- Runtime vectorization does not merge independent row transactions.
- A row-local validation/cancellation failure rolls back only that row.
- A physical execution failure that prevents determining any row result aborts every affected row.
- Mutable state without a request-aligned axis is invocation-global and explicitly couples every dependent row.
- A runtime unable to preserve global-state atomicity must execute the invocation without shared batching; it may not silently weaken semantics.
- Transaction visibility is logical. Implementations may update independent descriptors/pages in parallel and need no device-wide barrier across unrelated requests.

## 5. State transitions

```yaml
# Whole-value replacement
transition: {kind: replace}

# Linear/prefix speculative append
transition:
  kind: append
  axis: 2
  candidate_increment: verifier.evaluated
  commit: {kind: prefix, length: acceptance.length}
  bound: package.max_context

# Tree/beam selection
transition:
  kind: append
  axis: 2
  candidate_increment: tree.evaluated
  commit:
    kind: gather
    indices: acceptance.path_indices
    length: acceptance.length
  bound: package.max_context

# Fixed-capacity update
transition:
  kind: indexed_scatter
  axis: 2
  indices: cursor.current
  candidate_count: verifier.evaluated
  commit: {kind: prefix, length: acceptance.length}
  logical_length: length.current
  capacity: package.max_context
```

Candidate production and commit selection are distinct. Unselected speculative values never become committed state. Candidate trees remain tensor/component dataflow; v1 adds no state-version fork node.

## 6. Output publication batches

Every logical output stream has one binding. The binding produces an ordered typed publication batch, not multiple independent `emit` statements.

```yaml
outputs:
  tokens:
    contract: {dtype: int64, rank: 2, shape: [batch, generated_sequence]}
    family: {kind: materialized}
    publication:
      operations:
        - kind: append
          value: acceptance.tokens
          valid_length: acceptance.length
        - kind: append
          value: grammar.token
          valid_length: grammar.forced_length
```

- Operation list order is payload/event order, not node scheduling order.
- The boundary stages one batch per row/reaction and makes it visible after that row commits.
- A single append may instead consume an upstream concatenated tensor.
- Typed revision batches may contain ordered append/replace/retract/finalize envelopes.
- Reaction-valued operations publish after each successful reaction; final-valued operations publish after clock completion.
- Mixing reaction and final values in one operation is invalid; separate operations retain their inferred availability.
- Workflow-level commit-only/provisional-revision policy still applies to the complete transaction.

## 7. Static and dynamic presence lowering

Presence does not imply graph interpretation.

### Static reaction

If every gate is stable or absent:

- specialize stable gates at admission;
- hoist stable pure nodes;
- prebind component instances, ports, buffers, state descriptors, and publication slots;
- compile a fixed reaction schedule;
- permit fusion and CUDA graph capture.

### Dynamic firing

Reaction-valued scalar gates are valid and mandatory to support correctly. Every runtime compiles them to a finite block/FSM plan:

```text
block id -> prebound node sequence + successor decision
```

The runtime may use device conditional execution, captured variants, or a scalar host decision. It must not traverse YAML maps, resolve names, allocate graph objects, or rediscover dependencies per reaction.

A runtime does not reject a semantically valid graph merely because dynamic firing is slower on that backend. Performance reporting must distinguish static-reaction and dynamic-firing paths.

## 8. Zero-cost representation requirements

Compiled steady-state execution must use:

- dense node/block indices, not string lookups;
- pre-resolved typed port slots;
- preallocated state/output staging;
- state handles/descriptors rather than copied snapshots;
- zero-sized or erased effect-token dependencies;
- a fixed memory/liveness plan per compiled block;
- no per-reaction heap allocation;
- no generic graph traversal in the hot path.

Static Qwen decode additionally targets zero extra tensor copies, zero device-wide synchronization, and successful graph capture where the backend supports it. These are measurable acceptance conditions, not consequences asserted from the schema alone.

## 9. Node outputs remain implicit

```yaml
nodes:
  sample:
    kind: component
    component: token_sampler
    inputs: {...}

  token_state:
    kind: state
    initial: initializer.token
    next: sample.token
```

`sample.token` refers to output `token` in the component signature. ONNX components obtain that signature from graph ValueInfo; binding components declare it explicitly. Node instances do not repeat an outputs map.

## 10. Required full-type conformance

Every graph-visible ONNX input/output must contain sufficient ValueInfo to prove edge and state-transition compatibility. The validator rejects deficient ports and tells the producer to repair the ONNX graph; metadata never patches or duplicates the missing type.
