# GraphView Lens Design

**Status:** Revised proposal; no implementation in this change
**Date:** 2026-07-17
**Audience:** IR, session, and execution-provider maintainers

## Decisions for Justin

| Choice | Recommendation | Tradeoff |
|---|---|---|
| Partition representation | Model each accepted claim as a `PartitionId` and compile a `CompiledPartitionView`; do not flatten assignment by EP. | More plan objects and conflict resolution, but preserves ORT claim atomicity, boundaries, connectivity, and `meta_def`. |
| Capability API | Migrate to iterator/view-based `supports_node` before promising allocation-free EP capability coverage. | EP API migration work, but avoids cloned `Shape`/`TensorLayout` arrays per node. |
| Freeze versus placement | Build the structural lens before partitioning, but store placement and schedule in the immutable frozen plan—not `Graph`. | New plan metadata, but no post-freeze mutable IR writes or stale view. |
| Reproducibility scope | V1 guarantees determinism for the same finalized IR artifact; size reproducible cache bytes by maximum live ID if bytes are serialized/hashed. | Does not canonicalize semantically equal graphs made through different mutation histories. |
| Assignment identity | Use `PartitionTarget`, richer than `EpId`, from the first partitioned-plan API. | Slightly more plumbing, but represents EP instance, device, session, and expert shard without a breaking redesign. |

## 1. Summary

Introduce an immutable, cached `GraphView` as the structural consumption lens
over a finalized `onnx_runtime_ir::Graph`.

The view compacts live arena entries into dense indices and caches topology and
edge relationships. It borrows graph payloads; it does not copy nodes, values,
or weights.

The view is only the structural layer. A `FrozenGraph` owns the graph plus its
cache, while a separate immutable `FrozenPlan` owns placement, scheduling,
partition decisions, and compiled-partition descriptors.

Execution providers (EPs) consume individual partition views. A partition
corresponds to one accepted ORT-style subgraph claim, not to all nodes selected
for one EP.

This document recommends a staged migration. It deliberately does not specify
all implementation details or change Rust source in this revision.

## 2. Verified constraints in the current implementation

### 2.1 IR and arena facts

* `Graph` is an SSA graph with node/value typed arenas, ordered graph I/O, and
  graph-level initializer membership.
* Nodes record positional optional inputs and mandatory outputs
  (`node.rs:25-45`).
* A node's `device` and `exec_order` are mutable fields filled by placement and
  scheduling (`node.rs:42-45`).
* Values own their `shape`, `layout`, and mutable `device` placement
  (`value.rs:83-103`).
* A value's `Consumers` stores uses in a set. `Consumers::uses()` allocates a
  vector and sorts it by `(NodeId, input_index)` on every call
  (`value.rs:61-65`).
* Arena capacity includes tombstones (`arena.rs:55-57`). Removal recycles a raw
  slot through the free list (`arena.rs:68-76`, `98-106`).
* Arena iteration skips tombstones and is ascending raw ID (`arena.rs:109-114`).

### 2.2 Existing deterministic topology

`Graph::topological_order()` uses Kahn's algorithm and breaks ready-node ties by
ascending raw `NodeId` (`graph.rs:327-375`). The topology is deterministic for
one finalized graph artifact.

That guarantee does not mean IDs are mutation-history independent: deleting a
node and later inserting one can reuse its raw ID. Capacity and raw-ID lookup
table length can also differ for semantically equal live graphs built through
different histories.

### 2.3 EP capability and claim facts

The current EP API is:

```rust
fn supports_op(
    &self,
    op: &Node,
    opset: u64,
    shapes: &[Shape],
    layouts: &[TensorLayout],
) -> KernelMatch;
```

(`provider.rs:269-281`).

Shapes and layouts are properties of separate input `Value`s, not contiguous
arrays on a node. The CUDA coverage check consequently clones them into two
owned vectors for every examined node before calling `supports_op`
(`executor.rs:845-866`).

The ABI-side `SubgraphClaim` already describes atomic claim data:

```rust
pub struct SubgraphClaim {
    pub ep_id: EpId,
    pub node_ids: Vec<NodeId>,
    pub input_values: Vec<ValueId>,
    pub output_values: Vec<ValueId>,
    pub meta_def: Option<String>,
}
```

(`onnx-runtime-ep-api/src/abi.rs:17-25`).

It is therefore incorrect to treat `EpId -> Vec<NodeIndex>` as the partition
model. One EP can make several disconnected claims, perhaps separated by
another EP, and each claim's declared boundary and metadata are meaningful to
compilation.

### 2.4 Build and nested-graph facts

Session validation currently occurs before `Executor::build`
(`onnx-runtime-session/src/lib.rs:751-767`). The executor then mutates the
graph with fusion and EP-scoped passes and only obtains a topological order
(`executor.rs:972-983`).

Optimizer postconditions call `Graph::validate()`, but only under
`debug_assertions` (`onnx-runtime-optimizer/src/pass.rs:100-115`). A final
release-mode validation after executor mutations is therefore new work, not an
existing invariant.

Nested-body compilation clones its template, adds captures as inputs, updates
external input shape/dtype, and reruns inference before `Executor::build`
(`executor.rs:2949-3001`). A root-time recursive freeze of every body would
freeze the wrong shape specialization.

## 3. Ownership, lifetime, and finalization

The cache cannot be owned by a `GraphView` that also borrows a graph and then be
returned as a self-reference from an owning `FrozenGraph`. Use this shape:

```rust
pub struct FrozenGraph {
    graph: Graph,
    cache: GraphViewCache,
}

pub struct GraphView<'a> {
    graph: &'a Graph,
    cache: &'a GraphViewCache,
}

impl FrozenGraph {
    pub fn view(&self) -> GraphView<'_> {
        GraphView { graph: &self.graph, cache: &self.cache }
    }
}
```

`FrozenGraph::build(graph)` consumes a mutable graph after its structural
rewrites. It performs explicit release-mode validation, builds the cache, and
returns the owner.

`FrozenPlan` follows the same ownership rule: it stores only owned placement,
schedule, and partition descriptors. It never stores a `GraphView<'a>` or
`CompiledPartitionView<'a>`. Accessors borrow the plan and construct those views
on demand, so every view lifetime is scoped to `&self` and safely borrows the
plan's `FrozenGraph` and owned partition data.

The ordinary borrowed view prevents simultaneous `&mut Graph` access. An API
that needs to reopen optimization consumes `FrozenGraph`, drops its cache, and
returns the graph. No lens survives that transition.

`GraphView` belongs in `onnx-runtime-ir`. Its cache is general graph structure;
the session and EP crates consume it without owning a second topology cache.

## 4. Freeze boundary and immutable plan state

### 4.1 Recommended boundary

Build `FrozenGraph` after all *structural* graph mutations and final validation,
but before multi-EP capability resolution, placement, scheduling, and compile:

```text
load
  -> generic optimization
  -> EP-scoped structural rewrites
  -> NEW: Graph::validate() in release mode
  -> FrozenGraph { graph, cache }
  -> capability discovery / claim normalization
  -> placement + schedule + partition resolution
  -> FrozenPlan
  -> compile / run
```

This is intentionally different from freezing after writing `Node.device`,
`Node.exec_order`, and `Value.device`. Those fields are graph mutations, so a
borrowed frozen view cannot safely coexist with them.

### 4.2 Placement and schedule move into the plan

The recommended design moves post-freeze state out of the mutable graph:

```rust
pub struct FrozenPlan {
    frozen: Arc<FrozenGraph>,
    node_placement: Vec<Option<PartitionTarget>>, // NodeIndex-indexed
    value_placement: Vec<Option<PartitionTarget>>, // ValueIndex-indexed
    execution_order: Vec<PlanStep>,
    partitions: Vec<PartitionDescriptor>,
}

pub struct PartitionDescriptor {
    id: PartitionId,
    target: PartitionTarget,
    nodes_topo: Vec<NodeIndex>,
    membership: PartitionMembership,
    inputs: Vec<ValueIndex>,
    outputs: Vec<ValueIndex>,
    meta_def: Option<String>,
}
```

`PlanStep` references a `PartitionId` or a host/node step. It must not require
writing `Node::exec_order`.

The descriptor is plain owned data. `FrozenPlan::partition_view` borrows the
descriptor and `self.frozen`, constructing a `CompiledPartitionView<'_>` only
for the duration of the plan borrow; the plan therefore has no self-reference.

This permits several plans over one frozen structural graph when policy differs.
It also keeps placement identity precise without making the IR graph a
session-specific mutable artifact.

An alternative is to retain graph fields and move the freeze point after
placement/scheduling. That is simpler for short-term code reuse, but capability
queries and partitioning would then be coupled to mutable graph state. It is
not the recommended direction.

## 5. Structural cache

### 5.1 Dense identity

```text
topo_nodes: Vec<NodeId>                     // NodeIndex -> NodeId
live_values: Vec<ValueId>                   // ValueIndex -> ValueId
node_index_by_raw_id: Vec<Option<NodeIndex>>
value_index_by_raw_id: Vec<Option<ValueIndex>>
node_inputs: Vec<Range<usize>>
flat_node_inputs: Vec<Option<ValueIndex>>
node_outputs: Vec<Range<usize>>
flat_node_outputs: Vec<ValueIndex>
value_producer: Vec<Option<NodeIndex>>
value_consumer_uses: Vec<Range<usize>>
consumer_uses: Vec<ConsumerUse>             // NodeIndex + input slot
initializer_bits: BitSet<ValueIndex>
```

`NodeIndex` and `ValueIndex` are opaque view-local `u32` newtypes. They are
not persisted as IR IDs and a consumer never observes tombstones.

The view borrows ordered graph input/output lists and node/value/initializer
payloads. A value-to-index conversion occurs once at build; public traversal
thereafter uses dense indices and slices.

### 5.2 Consumer-range construction

Do not build cache ranges by calling `Consumers::uses()` per value. Each call
allocates and sorts, yielding a worst case of up to `O(E log E)` aggregate
sorting work (more exactly, `sum_v O(deg(v) log deg(v))`).

Instead:

1. allocate/count one consumer slot per live value;
2. prefix-sum counts into `value_consumer_uses`;
3. scan live nodes in ascending `(NodeId, input-slot)` order;
4. for each present input, write `ConsumerUse { node, input_slot }` into that
   value's next prefix-filled slot.

Arena iteration provides ascending live raw `NodeId`; scanning each node's
positional input slice provides ascending input slot. The result has exactly
the current public consumer order without per-value sorting.

### 5.3 Lookup-table sizing and reproducible bytes

For ordinary in-memory use, lookup vectors may be sized to arena capacity for
constant-time raw-ID translation. Their content is deterministic for the same
finalized artifact, but their capacity-derived byte representation is not
history independent.

If cache bytes are serialized, hashed, or compared for reproducibility, size
the lookup vectors to `max(live raw ID) + 1` (or zero when there are no live
entries), not arena capacity. This removes trailing tombstone capacity from the
representation. It does not erase differences in live IDs themselves.

## 6. Determinism contract

V1 reproducibility means: given the **same finalized IR artifact**, cache
contents, topology, ordered enumeration, and partition-resolution input are
deterministic.

The structural order is:

1. Kahn topological order;
2. ascending finalized `NodeId` among ready nodes;
3. positional node input/output order;
4. consumer uses ascending `(consumer NodeId, input slot)`;
5. partition ordering by explicit `PartitionId`, then each partition's cached
   topological node sequence.

No public API exposes `HashMap` or `HashSet` iteration.

Semantically equal graphs produced by different edit histories are outside the
V1 canonicalization guarantee. Their recycled IDs can alter topology tie breaks
and IDs, even with max-live-ID table sizing. Supporting that stronger contract
would need a durable semantic node key and a defined canonicalization pass.

## 7. Partition model

### 7.1 Claim normalization

Normalize each accepted claim against the frozen view:

```rust
pub struct PartitionId(u32);

pub struct PartitionTarget {
    ep_id: EpId,
    ep_instance: u32,
    device: DeviceId,
    session: Option<SessionId>,
    expert_shard: Option<ExpertShardId>,
}

pub struct PartitionClaim {
    id: PartitionId,
    target: PartitionTarget,
    nodes_topo: Vec<NodeIndex>,
    input_values: Vec<ValueIndex>,
    output_values: Vec<ValueIndex>,
    meta_def: Option<String>,
}
```

Validation rejects dead IDs, duplicate node IDs in one claim, and missing
declared boundaries. A claim's `node_ids` are treated as an unordered set:
normalization canonicalizes the accepted membership into base topological
order, regardless of the supplied order. It does not merge separate claims
merely because targets share an `EpId`.

The partitioner resolves overlaps/cost policy over claims, then produces
non-overlapping selected partitions plus host fallbacks. Claim groups remain
atomic during that resolution unless an EP explicitly provides a legal split
contract.

### 7.2 Compiled partition view

```rust
pub struct CompiledPartitionView<'a> {
    id: PartitionId,
    target: &'a PartitionTarget,
    graph: GraphView<'a>,
    nodes_topo: &'a [NodeIndex],
    membership: &'a PartitionMembership,
    inputs: &'a [ValueIndex],
    outputs: &'a [ValueIndex],
    meta_def: Option<&'a str>,
}
```

It describes one compilation unit. Its nodes can be disconnected only if the
EP's claim semantics explicitly permit that; otherwise claim normalization
requires connectivity. Its `inputs` and `outputs` preserve declared
`SubgraphClaim` boundaries after validating them against crossing edges.

The view classifies every incident edge as internal, boundary input, or boundary
output using membership data. It does not materialize a cloned `Graph`.

`EpId -> Vec<NodeIndex>` may exist only as a derived diagnostic index. It must
not be accepted by `compile`, used to reconstruct claims, or treated as a
partition identity.

### 7.3 Boundary derivation

For each selected claim, derive crossing edges from cached producer/consumer
ranges and compare them with its declared inputs/outputs. The policy for
additional legal boundary values must be explicit:

* reject mismatches for claims whose ABI boundary is authoritative; or
* deterministically augment from crossing edges when the EP API permits it.

Use first occurrence in partition topological traversal as the ordering rule
for derived boundaries. Cache the normalized boundary slices with the partition
so repeated `compile` calls do not rescan the graph.

## 8. Capability API migration and allocation claims

### 8.1 Why the legacy adapter allocates

A borrowed graph lens can yield `&Shape` and `&TensorLayout` for one input at a
time. It cannot manufacture contiguous `&[Shape]` and `&[TensorLayout]`
covering values stored separately in `Graph::values`.

Therefore a `supports_node` implementation that merely adapts to legacy
`supports_op` must allocate/clone per-node arrays, exactly like CUDA coverage
does today. The base structural view is allocation-free to query; legacy
capability coverage is not.

### 8.2 Recommended native surface

Migrate in-tree EPs to an iterator/view contract:

```rust
trait ExecutionProvider: Send + Sync {
    fn supports_node(
        &self,
        view: &GraphView<'_>,
        node: NodeIndex,
        opset: u64,
    ) -> KernelMatch;

    fn compile(
        &self,
        partition: CompiledPartitionView<'_>,
    ) -> Result<CompiledPartition>;
}
```

`supports_node` queries input metadata through indexed access or an
`ExactSizeIterator<Item = InputValueRef<'_>>`. It does not need contiguous
owned arrays.

Keep `supports_op` temporarily for source compatibility. The compatibility
adapter documents per-node allocation and is not used for the allocation-free
performance claim.

An alternative is to cache cloned `Vec<Shape>` and `Vec<TensorLayout>` for
every node in `GraphViewCache`. That makes the old API query allocation-free at
the cost of cache memory, clone time, and duplicated mutable-IR metadata. It is
not recommended while a native API migration is viable.

### 8.3 Correct performance model

Let `V` be live nodes, `W` live values, `E` present input-use edges, `C` claims,
and `A` nodes in one partition.

| Operation | Cost | Query allocation |
|---|---:|---|
| Build topology | `O(V log V + E)` with heap tie break | bounded build phase |
| Build dense edge/cache data | `O(V + W + E)` | bounded build phase |
| Build consumer ranges by prefix fill | `O(V + W + E)` | bounded build phase |
| Normalize claims | `O(C + claimed nodes + boundary edges)` plus policy work | plan-build only |
| Node/value/edge lookup | `O(1)` | none |
| Topology or partition iteration | `O(V)` / `O(A)` | none |
| Native `supports_node` metadata traversal | `O(node input arity)` | none |
| Legacy `supports_op` adapter | `O(node input arity)` | owned shape/layout arrays per call |

The `O(V log V + E)` topology term reflects the existing binary heap. Do not
state an unqualified linear build bound while retaining that tie-break.

## 9. API sketch

`NodeIndex`/`ValueIndex` are opaque ordered `u32` newtypes. `GraphView`
provides topology, `node_index`, node/value payload lookup, positional input
iteration, producer lookup, and a borrowed `&[ConsumerUse]`. `FrozenPlan`
provides node/value placement lookup and constructs borrowed partition views
from its owned descriptors:

```rust
impl FrozenPlan {
    pub fn partition_view(&self, id: PartitionId) -> CompiledPartitionView<'_> {
        let partition = self.descriptor(id);
        CompiledPartitionView {
            id: partition.id,
            target: &partition.target,
            graph: self.frozen.view(),
            nodes_topo: &partition.nodes_topo,
            membership: &partition.membership,
            inputs: &partition.inputs,
            outputs: &partition.outputs,
            meta_def: partition.meta_def.as_deref(),
        }
    }

    pub fn partitions(
        &self,
    ) -> impl Iterator<Item = CompiledPartitionView<'_>> + '_ {
        self.partitions
            .iter()
            .map(|partition| self.partition_view(partition.id))
    }
}
```

The private `descriptor` lookup may use a dense `PartitionId` index or a
deterministic ID-to-index table. Neither accessor returns a view that can
outlive `&self`.

The ORT ABI facade borrows the same view/cache:

```rust
pub struct OrtGraphView<'view, 'graph> {
    view: &'view GraphView<'graph>,
}
```

Phase-2 ABI callbacks enumerate topology and metadata from the cached lens.
They must not rebuild topology or reintroduce unordered arena/collection
traversal.

## 10. Nested and shape-specialized graphs

Do not recursively freeze all body graphs at root finalization.

For a control-flow invocation, first clone the body, expose captures, seed
external types/shapes, and run inference—the current child-executor flow already
does this. Then perform the same explicit final validation and build a
shape-specialized `FrozenGraph` for that prepared child.

Cache such child plans by their existing input signature. A signature miss builds
a new prepared child/frozen view; a hit reuses the already compiled plan.

Parent/body capture values require an explicit boundary mapping in the child
plan. They are not assumed to share dense `ValueIndex` identity across views.

## 11. Migration and proof plan

1. **Characterize:** test topology, I/O, initializers, consumers, tombstones,
   and separated claims for one EP.
2. **Structural lens:** add IR cache/indices, explicit release-mode
   `Graph::validate()`, topology comparison, and prefix-filled consumers.
3. **Capability API:** add and migrate in-tree coverage to `supports_node`;
   retain an explicitly allocating `supports_op` adapter for one release.
4. **Frozen plans:** normalize claims into partition objects, resolve conflicts
   without flattening, and compile a partition view through a temporary
   node-ID adapter.
5. **ABI:** make `OrtGraphView` borrow the lens and version-deprecate direct
   mutable-graph capability/partition inputs.

### Required differential checks

For real models and synthetic wide-fanout, 10k-node, tombstone-heavy, and
multi-claim graphs:

* compare topology and sorted consumer uses with existing graph behavior;
* compare native and legacy capability outcomes;
* compare selected claims, their boundaries, and `meta_def`, not merely
  `NodeId -> EpId`;
* compile/run old and new plans and compare outputs;
* test nested shape signatures to ensure each prepared child freezes separately;
* benchmark cache build, native capability queries, partition normalization, and
  compilation independently.

## 12. Final recommendation

Adopt `FrozenGraph { graph, cache }` plus borrowed `GraphView`, and build it
after structural mutation plus a new release-mode final validation.

Keep placement and scheduling as immutable `FrozenPlan` metadata. Preserve each
accepted `SubgraphClaim` as an owned `PartitionDescriptor`, and expose it for
compilation as a borrowed `PartitionId`/`CompiledPartitionView` with its
boundaries and `meta_def`; never compile an EP-wide flattened node list.

Migrate to view-based `supports_node` before claiming allocation-free EP
coverage. Define determinism for one finalized IR artifact and use max-live-ID
lookup sizing whenever cache bytes themselves must be reproducible.
