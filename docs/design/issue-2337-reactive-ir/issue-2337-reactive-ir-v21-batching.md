# Cyclic reactive dataflow IR - v21 derived continuous batching

v21 resolves the core/addon boundary by removing a second authored batching
authority.

## Decision

Core v1 does not serialize an
`interfaces.onnx-genai.continuous-batching` block.

Continuous batching is a derived runtime certificate built from ordinary,
already authoritative facts:

- core event, value, effect, state, and transaction semantics;
- per-value `batch_layout` and padding/ownership companions;
- per-component `batch_capacity`;
- component-private `row_scope`;
- state selection, snapshot, fork, aliasing, and rollback capabilities;
- output publication shape/validity;
- effect commit and retry contracts;
- selected implementation/backend capabilities.

The certificate may be discarded at any time. Absence or failed derivation
means isolated execution, not an invalid package.

The principle is:

```text
metadata states semantic facts once
runtime proves a grouped schedule from those facts
```

An authored `row_independent`, `select`, `expand`, `collective`,
`state_partition`, compaction, or serving-slot table would duplicate facts
already present in component and tensor contracts.

## Authored component facts

`batch_capacity` remains the sole authored claim that one component invocation
may combine independent contributions:

```yaml
components:
  model:
    implementation: {kind: onnx, artifact: model.onnx}
    contract:
      id: onnx-genai.autoregressive-decode
      version: '1'
      bindings: {...}
    port_layouts:
      input_ids:
        batch_layout: {kind: request_aligned, axis: 0}
      logits:
        batch_layout: {kind: request_aligned, axis: 0}
    batch_capacity:
      uniform_dimensions: [vocabulary]
      budgets:
        - {dimensions: [batch], max_total: 256}
```

Absence of `batch_capacity` means the component is called separately for each
contribution. It does not prevent the surrounding workflow from using
continuous batching for other component calls.

For ONNX components, dtype, rank, and shape still come only from `ValueInfo`.
`port_layouts` is a non-tensor-signature overlay containing only facts ONNX
cannot express:

```text
PortLayout {
  batch_layout: BatchLayout,
  padding?: List<PaddedDimension>
}
```

Every named port must exist in the resolved artifact signature. Axis and shape
symbol references are checked against `ValueInfo`; the overlay cannot add,
replace, or repair an ONNX tensor type.

Binding components keep the same facts in their explicit signatures:

```yaml
signature:
  inputs:
    logits:
      required: true
      dtype: float32
      rank: 2
      shape: [batch, vocabulary]
      batch_layout: {kind: request_aligned, axis: 0}
```

There is one batch-layout vocabulary in both cases:

```text
shared
request_aligned
request_expanded
token_packed
runtime_sequence_state
```

If a component retains runtime-private row state, its existing `row_scope`
declares the request axis and requires positional `compact(selection)` and
`release(row)` behavior. This is a component ABI invariant, not a
continuous-batching capability flag.

## What `batch_capacity` proves

`batch_capacity` asserts:

> For every assembled call satisfying the declared layouts, uniform dimensions,
> padding/ownership contracts, and static budgets, each contribution's
> observable outputs and effects are equivalent to executing that contribution
> separately.

It does not select:

- target batch width;
- queue delay or fairness;
- device or execution provider;
- compaction timing;
- physical state layout;
- CUDA graph bucket;
- page size or allocator;
- admission policy.

Those remain runtime policy/evidence.

## Derived grouping islands

The compiler finds maximal call-site islands that may be physically grouped and
then split back into isolated invocation results.

Two live component occurrences may share a call only when:

1. they name the same resolved implementation or a proven equivalent
   substitution;
2. they occur in compatible compiled blocks;
3. the component declares `batch_capacity`;
4. all uniform dimensions agree;
5. every input can be assembled under its declared layout;
6. every static capacity budget holds;
7. outputs can be split/rebased back to their contributions;
8. component-private state can follow the same positional row plan;
9. effect results can be attributed to their original invocation transaction;
10. the selected backend reports support for the assembled contracts.

Grouping is local to the component call. Values are split back into each
invocation's logical slots before core state transitions, effects,
publications, and working commits proceed unless those later calls independently
qualify for grouping.

This allows a batched model call followed by isolated sampling or tool effects.
It is not unsafe "partial addon application": every grouped island has a
complete assemble/call/split equivalence boundary.

## Events remain scalar

Core lifecycle events stay scalar per isolated invocation. The batching runtime
does not turn `generation.pulse`, `route.then`, or `completed` into authored
`bool[batch]` values.

Instead:

1. each invocation's compiled FSM determines which call site is ready;
2. the scheduler queues ready occurrences by a compatibility key;
3. one grouped call is assembled from occurrences whose firing event is
   present;
4. results are returned to the corresponding FSM instances.

Invocations at different logical reaction indices may share a call when the
component does not consume the index and every call contract is otherwise
compatible. If the index is a component input, it must have an assemblable
per-contribution representation or the runtime leaves that component isolated.

Dynamic switch paths therefore reduce grouping opportunities but do not require
dynamic event tensors or graph interpretation.

## Value assembly

The existing `batch_layout` determines assembly and splitting:

- `shared`: every grouped contribution supplies one representation-compatible
  value; differing values may group only when the component contract says the
  port is safely broadcast/ignored, otherwise the runtime requires equality;
- `request_aligned`: concatenate contribution rows on the declared axis and
  split by recorded positional spans;
- `request_expanded`: move the declared fixed-size contiguous row group as one
  logical request contribution;
- `token_packed`: concatenate the packed axis, rebuild/rebase the declared
  ownership chain, and split/rebase companions on return;
- `runtime_sequence_state`: pass runtime-owned per-sequence handles through a
  backend ABI that explicitly supports grouped access.

Right padding is permitted only on declared padded dimensions with named
`valid_lengths`. The runtime never guesses validity from sentinel values or
attention masks.

Runtime contribution positions and spans are ephemeral. Metadata serializes no
request IDs, scheduler slots, epochs, page IDs, block tables, or parent request
identities.

## State and working commits

Core state remains logically per invocation even when a physical backend
processes several rows together.

After a grouped component returns:

- each invocation receives its own candidate values and transition operands;
- each state transition validates/commits against that invocation's current
  snapshot;
- each reactor samples its own `continue`;
- each working commit succeeds or fails independently.

A runtime-managed state service may keep several logical states in one paged or
contiguous allocation. That storage choice does not merge their core
transactions.

To compact, release, or clone physical rows, the runtime creates one positional
row selection and applies it atomically to every participating carrier:

- request-aligned/expanded values;
- packed ownership companions;
- runtime sequence state;
- component-private row state;
- pending effect/publication bookkeeping.

Selection entry `i` names the old source position copied to new position `i`.
Repeated sources express beam/speculative cloning. A mutable state must support
copy-on-write or an equivalent independent clone before repeated selection is
legal. Missing compaction/release/clone support removes the affected island
from shared batching; it does not invalidate isolated execution.

## Effects

Effect roots remain per invocation and per event occurrence. Grouping an
effectful component is legal only when:

- `batch_capacity` covers the grouped call's complete observable behavior;
- every returned effect result maps to exactly one input root token;
- transactional savepoints and failures remain attributable per invocation;
- retry and speculation contracts remain valid after splitting.

The runtime never combines several invocation effect tokens into one logical
token. If it cannot split success/failure without changing core transaction
semantics, the effectful call stays isolated.

An `after_invocation_commit` call may group only committed invocations and may
not make one invocation's retry or failure affect another's durable result.

## Publications and completion

Grouped outputs are split into invocation-local values before publication
operations become durable. Valid lengths, packed ownership, and output order
are reconstructed from the same layout contracts used for assembly.

One invocation may complete and durably commit without waiting for unrelated
invocations that happened to share an earlier physical call. Compaction then
removes its row from every carrier before a new grouped call uses the slot.

## Derived certificate

A runtime may materialize an internal certificate such as:

```text
ContinuousBatchCertificate {
  call_sites: Map<CallSiteId, GroupingProof>,
  carriers: List<RowCarrierPlan>,
  state_services: List<StateSelectionCapability>,
  split_plans: Map<CallSiteId, OutputSplitPlan>,
  effect_partitions: Map<CallSiteId, EffectPartitionProof>,
  compatibility_key: CompiledKeyProgram
}
```

This is explanatory, not serialized schema. Concrete Rust types may differ.
The certificate contains dense IDs and precomputed plans; it never causes YAML
traversal or dependency discovery in the scheduling hot path.

The compatibility key is derived from facts that affect correctness:

- call site and implementation/substitution;
- device/backend and compiled artifact;
- resolved dtypes/ranks/layouts;
- uniform dimension extents;
- event block where required by the component ABI;
- state-service representation/capabilities;
- effect partition requirements.

Queue age, target width, memory pressure, and performance estimates are runtime
policy and are not part of semantic compatibility.

## Fallback

Failure to derive or execute one grouping proof causes that call site to use
isolated execution. A runtime may still group other independently proven call
sites.

It must not:

- reject the core workflow because batching is unavailable;
- assume row independence from an axis alone;
- group a component without `batch_capacity`;
- move only some carriers during compaction;
- expose one invocation's state/effect/publication to another;
- silently change output values, ordering, retry, or failure boundaries.

An implementation failure after grouped execution begins is reported with the
call site, assembled shape/layout, backend, failed proof/capability, affected
invocations, and whether safe isolated retry is possible.

## Qwen lowering

For Qwen, the likely certificate contains:

- one prefill model-call grouping proof;
- one steady decode model-call grouping proof;
- request-aligned token/logit/mask/length assembly;
- a runtime-sequence-state KV selection plan;
- invocation-local sampling, state transitions, effects, publications, and
  working commits unless those call sites independently declare capacity.

Prefill and steady calls need not share one group or compiled artifact. Ready
prefill occurrences group with compatible prefills; ready steady occurrences
group with compatible steady calls. Different reaction indices may share the
steady call because the model does not consume the ordinal.

The same core Qwen graph remains executable when every certificate entry is
absent.

## Conformance

For each derived grouping proof:

1. run the participating invocations in isolated mode;
2. run the same invocations through the grouped island;
3. compare invocation-local component outputs under the contract's equivalence
   class;
4. require identical final state, publication order/content, effect outcomes,
   and failure attribution;
5. exercise selection, repeated-source cloning, release, cancellation,
   zero-row drain, and one-row fallback;
6. decline grouping when any proof or backend capability is absent.

Performance is measured only after semantic equivalence is established.
