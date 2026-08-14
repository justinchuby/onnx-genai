# Graph IR node/edge container refactor

## Problem

At baseline commit `c89e61f`, graph nodes and values lived in
`Arena<K, T>` (`crates/onnx-runtime-ir/src/arena.rs`): a
`Vec<Option<T>>` plus a recycled-slot free list. Arena lookup, insertion, and
removal were constant-time, but IDs were non-generational and iteration crossed
tombstones.

The edge representation was the larger removal bottleneck:

- `Value.consumers` was `Vec<NodeId>`
  (`crates/onnx-runtime-ir/src/value.rs:42-43`).
- `disconnect_edges` removed one node with
  `retain(|&consumer| consumer != id)`
  (`crates/onnx-runtime-ir/src/graph.rs:757-763`).
- `gc_value_if_orphan` scanned both graph-I/O vectors with `contains`
  (`graph.rs:776-783`).

Removing `N` consumers of a shared weight, attention mask, or RoPE table one at
a time therefore scanned approximately `N + (N-1) + ... + 1` consumer entries.
The existing `remove_nodes` batching optimization avoided that cost only when
every caller opted into the batch API; `remove_node` remained quadratic.

## Lessons from onnx/ir-py

The Python IR stores uses as dictionary keys containing both the node and input
index, and stores graph-input/output/initializer membership on each value.
Those choices are important:

1. `(node, input_index)` preserves multiplicity when one node consumes the same
   value in more than one slot.
2. A keyed use can be added or removed without scanning all consumers.
3. Per-value I/O membership makes orphan checks independent of graph-I/O width.

## Rust design

### Keyed uses

`Value.consumers` is now `Consumers`, backed by
`HashSet<(NodeId, u32)>`. `Consumers` deliberately exposes no unordered
iterator. Its public snapshots are:

- `uses() -> Vec<(NodeId, u32)>`, sorted by `(NodeId, input_index)`;
- `nodes() -> Vec<NodeId>`, deduplicated and sorted by `NodeId`;
- `len()` and `is_empty()` for order-independent queries.

`Graph::uses(value)` and `Graph::consumers(value)` are the normal traversal
boundaries. `Consumers` has a custom `Debug` implementation that prints sorted
uses, so randomized hash iteration cannot leak into `{:#?}` output.

`Graph::replace_input(node, input_index, new_value)` updates the old keyed use,
the node slot, and the new keyed use as one operation. Passing `None`
disconnects a slot. `disconnect_edges` uses this primitive, making a
single-node removal proportional to that node's input/output arity rather than
the fanout of its inputs. `replace_all_uses` consumes a sorted use snapshot and
rewires exact slots, so duplicate inputs remain correct.

The old special-case batching machinery is no longer needed:
`remove_nodes` is sequential `remove_node`, and `replace_node_groups` preserves
its documented sequential semantics directly. Both remain linear in removed
edges because the single-node primitive is now efficient.

### Determinism at boundaries

Internal hash order is never observable:

- edge snapshots are sorted before returning;
- `predecessors` and `successors` return ascending `NodeId`;
- optimizer, loader-writer, session, memory-planner, and EP readers use the
  sorted graph accessors;
- debug output sorts `(NodeId, input_index)`;
- serialization traverses deterministic node/value order and sorted edge
  boundaries;
- topological ordering retains the exact
  `BinaryHeap<Reverse<u32>>` ascending-ID tie-break.

Tests build the same graph independently, disconnect/reinsert all hub uses in a
shuffled order, and require identical debug text, ONNX protobuf bytes,
topological order, sorted uses, and sorted consumers.

### O(1) orphan membership

`Value` now carries `is_graph_input` and `is_graph_output`. `add_input`,
`add_output`, `insert_output`, `remove_input`, `remove_output`, `set_inputs`,
`set_outputs`, and graph-output replacement keep the flags synchronized with
the ordered vectors. All in-workspace direct graph-I/O mutations were moved to
these helpers. `Graph::validate` has debug assertions in both directions
between flags and vectors.

`gc_value_if_orphan` now checks the two flags and the initializer `HashMap`;
there are no graph-I/O vector scans in the removal hot path.

### Topological-order scratch

`Arena::capacity()` exposes the raw slot span. `topological_order` allocates
`in_degree: Vec<usize>` and `adjacency: Vec<Vec<NodeId>>`, indexed directly by
raw `NodeId`, with `usize::MAX` marking tombstones. This removes two per-call
`HashMap`s while preserving duplicate dependency edges and the previous
ascending-ID ready-queue tie-break.

## Measured benefit

Command:

```text
CARGO_TARGET_DIR=/home/justinchu/target-ir-refactor \
  cargo run --locked --release -q -p onnx-runtime-ir \
  --example remove_node_bench
```

Host: 96-vCPU Intel Xeon Platinum 8480C. Graph construction is outside the
timed region. Sequential removal is the median of three graphs. The
single-edge measurement removes one node from a full hub graph 100 times,
cloning outside the timed region.

| Hub consumers | Baseline sequential `remove_node` | Refactored | Speedup | Baseline one disconnect | Refactored |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 17.137 ms | 1.589 ms | 10.8x | 3.467 us | 0.683 us |
| 20,000 | 58.792 ms | 2.927 ms | 20.1x | 6.366 us | 0.815 us |

Doubling fanout made the baseline sequential removal 3.43x slower, consistent
with its quadratic scan. The refactored path grew 1.84x. The isolated
disconnect stayed nearly flat instead of doubling.

## Equivalence and reproducibility proof

- A test-only reference implements the old vector-era removal semantics:
  remove all uses belonging to the node, clear producers, remove the node, and
  perform the original graph-I/O vector membership orphan test.
- 2,000 randomized DAGs with duplicate input uses, graph outputs, shuffled
  removal order, duplicate IDs, and non-live IDs compare the reference and new
  single-node path using byte-identical `{:#?}`, topological order, sorted uses,
  and sorted consumers.
- Another 2,000 randomized DAGs, including arena tombstones, compare the
  Vec-indexed topological order against a copy of the old `HashMap` algorithm.
- The loader integration test requires byte-identical encoded ONNX protobuf for
  independently built graphs and for shuffled hub-use insertion order.
- Existing 10,000-trial batch-versus-sequential and node-group equivalence
  tests remain enabled.

## Risks and follow-ups

- `HashSet` adds hashing and allocation overhead to very-low-fanout values in
  exchange for bounded high-fanout mutation. A small-set optimization can be
  considered only with equivalent asymptotic guarantees and determinism tests.
- `Arena` IDs are still non-generational. A stale ID can alias a later occupant
  after slot reuse. Generational IDs are a separate API/serialization design.
- Arena iteration still scans tombstones. An ordered live-node container or
  compaction strategy is a follow-up; compaction must not silently renumber
  externally held IDs.
- `Graph.inputs` and `Graph.outputs` remain ordered vectors for ONNX contract
  fidelity. Mutations must go through the synchronization helpers; validation
  debug assertions detect in-workspace drift.
