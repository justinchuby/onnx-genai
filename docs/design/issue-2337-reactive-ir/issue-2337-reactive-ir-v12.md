# Cyclic reactive dataflow IR — v12 core/addon split

v12 removes serving-scheduler concepts from the semantic core.

## Design principle

```text
core IR = one workflow invocation, executable in isolation
continuous-batching addon = proof that many invocations may be lifted safely
```

The addon can improve scheduling and memory behavior but cannot change isolated execution results.

## Core execution model

The core contains only:

- typed workflow inputs and outputs;
- component instances with signature-defined outputs;
- generic scalar/bundled state;
- structural bundle/gate/merge nodes;
- one optional synchronous clock;
- scalar presence and repeat predicates;
- reaction working commits and invocation durable commit;
- typed linear effects and publication batches.

```yaml
graph:
  clock:
    kind: synchronous
    max_reactions: request.max_iterations
    index: reaction.index       # scalar logical invocation ordinal
    first: reaction.first       # scalar presence control
    repeat_while: termination.continue
    sample_repeat: after_working_commit
```

A tensor batch inside one invocation executes lockstep. Per-row `active`, `done`, accepted lengths, masks, and valid lengths are ordinary tensor data. Policy components preserve inactive rows. The scalar repeat predicate normally reduces row activity to “continue while any row remains active.”

Core correctness never depends on row compaction, serving slots, request IDs, or combining unrelated invocations.

## Core transactions

```text
reaction working commit:
  atomically advances this invocation's complete working state

invocation durable commit:
  persists final session state/effects/output heads after clock/postlude success
```

Without the addon, the invocation is the transaction unit. A failure aborts the invocation to its admission baseline. This is simple, deterministic, and always executable.

## Continuous-batching addon

```yaml
interfaces:
  onnx-genai.continuous-batching:
    version: '1'
    rows:
      source: request.input_ids
      axis: 0
    state_partition: row
    components:
      model:
        transform: row_independent
      token_sampler:
        transform: row_independent
      termination:
        transform: row_independent
      row_selector:
        transform:
          kind: select
          indices: row_selection
```

The exact surface remains to be finalized, but the module owns these advanced facts:

- which tensor axis represents independently owned logical rows;
- component row independence or collective behavior;
- row selection, expansion, and parent mappings;
- state/effect/publication row partitioning;
- per-row commit/abort and cancellation;
- compaction and release legality;
- copy-on-write requirements for repeated selection;
- lifting scalar per-invocation clocks/FSM blocks into serving batches;
- grouping compatible prefill, steady-decode, dynamic-gate, and final blocks;
- reconstructing invocation-local ordering after batched component execution.

## Addon fallback rule

A runtime that does not implement the addon—or cannot prove one declared transform for its chosen executor—runs each invocation in isolated/lockstep mode.

It must not:

- reject an otherwise valid core workflow solely because batching is unavailable;
- silently assume row independence;
- partially apply the addon across a state/effect boundary that requires joint ownership;
- change generated values, state transitions, publication order, or failure semantics.

An unknown addon version is ignored for core execution and rejected only if the caller/deployment explicitly requires that optimization.

## Performance layering

### Core requirement

Every graph is precompiled to fixed blocks/FSM transitions. Even isolated dynamic execution performs no YAML traversal, name resolution, dependency discovery, or per-reaction graph allocation.

### Addon optimization

A continuous-batching runtime may:

- combine compatible component nodes from different invocation FSM instances;
- compact completed rows;
- retain paged state per independent row;
- commit completed invocations without waiting for unrelated ones;
- run different workflow blocks separately while sharing model execution islands where legal.

The addon is not permission to reinterpret the graph. It is a proof that vectorized execution is observationally equivalent to isolated execution.

## Qwen consequence

Qwen core stays understandable:

```text
reaction 0: prefill -> sample -> decode -> working commit
reaction 1+: sample -> decode -> working commit
final: durable commit
```

The addon may batch:

- prefill blocks with compatible prefill blocks;
- steady decode model calls from invocations at different logical ordinals;
- policy blocks only when their scalar/control ABI can be lifted safely;
- finalization independently after each invocation completes.

`reaction.index` remains scalar in the core. If a vectorized policy implementation accepts per-row ordinals, that is an addon lowering detail or an equivalent substituted binding—not a new core value type.

## What moves out of v10 core

The following are no longer mandatory concepts for ordinary package authors:

- `row_independent | select | expand | collective` component transforms;
- per-row reaction index or row-presence control;
- per-row transaction partitions;
- serving-batch compaction/release rules.

They remain available in the versioned addon for packages/runtimes seeking continuous batching.

## What remains core despite the split

- Tensor contracts still declare ordinary shape and batch layout at workflow boundaries.
- ONNX graph-visible ports still require sufficient ValueInfo.
- Binding signatures remain explicit.
- State transition plans remain logical and contain no physical slot/page identifiers.
- `bool[batch]` remains ordinary data and cannot become node firing control in the core.
- Scalar dynamic gates still compile to finite invocation-local FSM blocks.

## Conformance strategy

1. Execute the core workflow in isolated mode as the reference semantics.
2. Execute the addon-lifted schedule with the same inputs and state baseline.
3. Require identical publications and final state under the declared equivalence class.
4. If the proof/capability is unavailable, use isolated execution.

Continuous batching is therefore removable optimization metadata, not part of the meaning of the model workflow.
