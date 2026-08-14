# Graph partition, EP assignment, fusion, and planning performance

Audit date: 2026-07-17

## Verdict

**No — the pipeline as a whole is not yet reliably performant for very large
(10k–20k+ node) models.**

The important qualification is that the **default, optimization-disabled,
single-EP executor build is fast and scales approximately linearly**: a 20k-node
static `Relu` chain built in 41.6 ms in the release benchmark below. The SiLU
rewrite plus build also scaled linearly (21.3 ms at 20k original nodes).

However:

1. There is **no active heterogeneous graph partition / GetCapability
   implementation** in the session path. The session selects one EP, CUDA
   requires complete graph coverage, and `claim_nodes` is never overridden.
2. Shape re-inference has an **O(NV), normally O(N²), full-value scan for every
   node** and took 1.22 s at 20k nodes even on an artificially favorable graph
   whose interior values were anonymous.
3. Constant folding's full-graph fixpoint is **O(N²)** in legal reverse-NodeId
   dependency order and took 11.62 s at 20k nodes.
4. Match-heavy fusion, wide-fanout deletion, and many-partition EPContext
   splicing have proven super-linear/quadratic paths.
5. Static execution allocates one device buffer for every backed value, with no
   liveness-based reuse. Compile latency is acceptable for tiny synthetic
   tensors, but realistic 10k-node model memory can be the limiting failure.

Therefore 10k+ nodes are viable today only on the narrow path of optimization
off, one EP covering the whole graph, no large EPContext partition export, and
a model whose simultaneous per-value buffers fit memory.

## Scope and notation

- `N`: live graph nodes.
- `E`: present node-input edges (input slots with a `ValueId`).
- `V`: live values.
- `I`, `O`: graph inputs and outputs.
- `P`: partitions; `K_i`: nodes covered by partition `i`.
- `R`: kernel-registry entries (currently 145 `reg.register` calls for CPU and
  33 for CUDA).
- `F`: successful fusion matches.
- `B`: initializer or EP-context blob bytes copied/hashed.

Hash-table operations below are expected O(1), absent adversarial hashing.
Operator arity and tensor rank are not assumed unbounded unless stated.

## Pipeline map

| Stage | Current implementation | Complexity and allocation behavior |
|---|---|---|
| IR storage and lookup | `Graph` owns node/value `Arena`s, vector graph I/O, and hash maps for initializers, symbols, opsets, and subgraphs (`crates/onnx-runtime-ir/src/graph.rs:19-39`). `Arena` is `Vec<Option<T>>` plus a free-list (`crates/onnx-runtime-ir/src/arena.rs:19-24`). | Node/value lookup, insert, and remove are O(1) amortized (`arena.rs:55-101`). Iteration scans all arena slots, including tombstones (`arena.rs:104-123`), so repeated scans after many rewrites still cost the high-water slot count. Producer lookup is O(1); consumers are adjacency `Vec<NodeId>`s (`crates/onnx-runtime-ir/src/value.rs:39-43`). |
| Model validation before executor | `InferenceSession::from_parts` validates, scans I/O metadata, loads EPContext nodes, then builds the executor (`crates/onnx-runtime-session/src/lib.rs:740-764`). Loader checks opsets, control flow, dangling inputs, and initializer producers (`crates/onnx-runtime-loader/src/lib.rs:415-420`). | O(N + E + V) for ordinary graphs. Control-flow attribute names are sorted per node (`loader/src/lib.rs:596-623`). EPContext enumeration is another O(N) scan. |
| Generic optimization selection | Optimization defaults to `None`; Basic adds constant folding and DCE, All adds fusion (`session/src/lib.rs:287-337`). Passes run sequentially; debug builds perform full `Graph::validate` after every pass (`crates/onnx-runtime-optimizer/src/pass.rs:100-116`). | Release cost is the sum of passes. Debug cost adds one validation/topological sort per pass. |
| Constant folding | Repeatedly collects/scans all live node IDs until a pass makes no change (`optimizer/src/constant_folding.rs:43-98`). | O(QN + folded-data work), where `Q` is fixpoint rounds and can be N: **O(N²)**. Reverse dependency order folds only one node per round. Each round allocates a `Vec<NodeId>`; each candidate clones its `Node` (`constant_folding.rs:48-53`). Measured 11.62 s at 20k. |
| Dead-node elimination | Backward liveness is a hash-set traversal, then dead nodes are removed individually (`optimizer/src/dead_node.rs:23-62`). | Liveness is O(N+E). Removal can be **O(N²)**: disconnecting each node calls `retain` on every input value's consumer vector (`ir/src/graph.rs:496-513`), so deleting N siblings sharing one input repeatedly scans a shrinking N-element vector. Orphan GC also uses linear `inputs.contains` / `outputs.contains` (`graph.rs:515-525`). Measured 57.3 ms at 20k wide dead nodes. |
| Device-independent fusion | Five default patterns are applied one at a time to fixpoint (`optimizer/src/fusion.rs:1265-1318`). `find_match` scans nodes from the arena start (`fusion.rs:219-239`). | A no-match pass is O(patterns × N) and measured linear. With F matches, restarting from the beginning is O(FN), **O(N²)** when F=Theta(N). Match safety repeatedly uses linear `graph.outputs.contains`; graph mutation inherits consumer-`retain` costs. Match-heavy MatMul+Add measured 364.8 ms at 20k. |
| Generic post-optimization shape inference | Any non-empty generic optimizer pipeline reruns whole-graph inference (`session/src/lib.rs:644-679`). Inference topologically walks nodes (`crates/onnx-runtime-shape-inference/src/infer.rs:60-121`). | The topological walk should be O(N+E), but each node calls `visible_scope`, which clones imported scope, rebuilds the formal-input set, and scans **every graph value** (`shape-inference/src/infer.rs:427-467`). This is **O(NV)** time, O(V) transient allocation per node, and O(N²) for V=Theta(N). Loader-created interior ONNX values are named (`loader/src/graph_builder.rs:246-255`), causing additional String/type/shape cloning. |
| Executor SiLU fusion | `Executor::build` first lowers `x * Sigmoid(x)` (`session/src/executor.rs:649-710`, called at `executor.rs:849-858`). | Initial collection is O(N). Normal bounded-fanout matching is O(N), but `graph.outputs.contains` is O(O), consumers are cloned, and mutation can hit repeated `retain`; worst case is O(NO + fanout²). The normal chain benchmark remained linear. |
| EP-scoped passes | `run_ep_scoped_passes` asks the selected EP for passes, runs them, then always reruns shape inference (`executor.rs:721-739`). CPU optionally supplies `ProjectionFusion`; CUDA supplies none (`crates/onnx-runtime-ep-cpu/src/provider.rs:170-172`). | With no passes: O(1). ProjectionFusion first scans all SiLU nodes (`ep-cpu/src/optimizer.rs:52-81`), then does local matching, but uses linear graph-output membership (`optimizer.rs:219-255`) and copies matched weight/scale bytes (`optimizer.rs:176-196`). If enabled, even a no-change pass triggers the O(NV) shape inference. A changed pass also calls full `graph.validate` (`optimizer.rs:78-80`). |
| Topological planning | Kahn's algorithm constructs `HashMap` indegrees and adjacency, then uses a deterministic `BinaryHeap` (`ir/src/graph.rs:232-278`). | O(N+E+N log N) worst case because each node enters/leaves the heap. The two HashMaps are not pre-sized; adjacency allocates one Vec per producer. Measured near-linear through 20k: 3.48 ms chain, 1.26 ms independent fan-out. |
| EP assignment / capability | The builder selects exactly one EP (`session/src/lib.rs:530-544`), passes a one-element EP list to EPContext loading (`session/src/lib.rs:758-764`), and the executor stores one EP. `claim_nodes` is only a default empty trait method (`crates/onnx-runtime-ep-api/src/provider.rs:388-397`); no implementation exists. CUDA scans the whole graph and rejects any unsupported node (`executor.rs:742-830`). | **No graph partition algorithm currently executes.** CUDA coverage is O(NR + E×rank) and allocates shape/layout Vecs per node. For static nodes it also creates a kernel only to validate it, discards it, then `compile_all` creates it again. |
| Kernel registry lookup | `OpRegistry::supports` and `lookup` linearly scan all registry entries (`ep-api/src/registry.rs:58-81`). CPU/CUDA call them from `supports_op` and `get_kernel` (`ep-cpu/src/provider.rs:123-167`; `ep-cuda/src/provider.rs:113-168`). | O(R) per support check and O(R) per kernel lookup, hence O(NR). R is fixed today, so asymptotically linear in N, but 10k CPU nodes perform roughly 2.9 million registry-key comparisons during static compile. Index by `(domain, op_type)` to make this expected O(1). |
| Value metadata and plan construction | Executor creates shape/dtype/buffer maps, scans initializers/inputs/outputs, then builds a topological `NodePlan` per node (`executor.rs:860-989`). It also builds input/name indexes (`executor.rs:991-1011`). | O(V+E+N) expected. The four primary HashMaps and indexes start with no capacity. Every node clones inputs and outputs and allocates dtype Vecs. This is linear but allocation-heavy. |
| Static buffer materialization and kernel compile | Static graphs resolve all values, allocate buffers, then compile every node (`executor.rs:1036-1045`). `size_buffers` visits every value (`executor.rs:1123-1148`); `compile_all` visits every plan node (`executor.rs:1164-1198`). | O(V×rank + N(R+arity×rank)) plus allocation/copy cost. Critically, `ensure_buffer` calls the EP allocator separately for every non-initializer value (`executor.rs:1049-1065`). There is no liveness/reuse plan, so memory is the sum of all value buffers, not peak live tensors. |
| EPContext load | The loader scans all nodes; the session builds source maps, performs primary/reference passes, hashes copied payloads, and invokes `load_context` (`session/src/epcontext.rs:94-164`). | O(N + P + B) expected, but embedded payloads are copied into the dedup key and copied again into `EpContext` (`epcontext.rs:120-135`), increasing peak memory. |
| EPContext partition splice/export | Export deep-clones the graph, then calls `splice_partition` for every partition (`loader/src/writer.rs:137-181`). Each splice builds a covered set, derives boundary I/O, removes covered nodes, and inserts one node (`writer.rs:185-254`). Boundary detection sorts covered nodes and rebuilds a graph-output HashSet for every partition (`writer.rs:279-319`). | Clone is O(N+V+E+B) and may duplicate inline initializer bytes. Boundary work is `sum O(K_i log K_i + boundary_edges_i) + O(PO)`. Rebuilding graph outputs and repeated graph mutation make many small partitions **O(PN)/O(N²)**. Measured 4.82 s for 20k one-node partitions. |

### Repeated topological work

Topological order is not recomputed inside the executor's per-node plan loop, but
it is recomputed across pipeline stages:

- shape inference calls `topological_order`;
- `Executor::build` calls it again;
- debug pass postconditions call `Graph::validate`, which calls it
  (`ir/src/graph.rs:449-452`);
- changed CPU ProjectionFusion calls `validate` itself before shape inference.

Thus optimized release builds perform at least two sorts; debug and EP-scoped
rewrite paths can perform several more. Topological sorting itself is not the
dominant problem, but the repetitions are unnecessary.

## Empirical proof

Benchmark source:
`crates/onnx-runtime-session/tests/graph_perf_audit.rs:1-396`.

Command:

```bash
export CARGO_TARGET_DIR=/home/justinchu/target-wallace
cargo test -p onnx-runtime-session --release \
  --test graph_perf_audit graph_partition_performance_audit \
  -- --ignored --nocapture
```

Environment: Intel Xeon Platinum 8480C, Linux x86-64, Rust/Cargo 1.97.0.
Graph construction and object destruction were outside timed regions. Small
cases used medians of 2–7 runs; 10k/20k expensive cases used one run. A complete
second run before adding constant folding reproduced the principal results
(20k shape inference 1.179 s, fusion 377 ms, EPContext 4.861 s).

### Measured wall time (release, milliseconds)

| Stage / synthetic topology | 1k | 5k | 10k | 20k | 20k / 1k |
|---|---:|---:|---:|---:|---:|
| `Graph::topological_order`, chain | 0.161 | 0.863 | 1.604 | 3.477 | 21.6× |
| `Graph::topological_order`, independent fan-out | 0.063 | 0.317 | 0.723 | 1.256 | 19.8× |
| `InferenceSession::from_graph` / Executor build, Relu chain | 1.781 | 8.675 | 19.400 | 41.599 | 23.4× |
| Executor build, SiLU-pattern chain (original nodes) | 0.950 | 4.857 | 10.850 | 21.251 | 22.4× |
| Shape inference, anonymous-value chain | 2.028 | 31.824 | 122.154 | 1218.607 | 600.9× |
| OpFusion, no matches | 0.012 | 0.056 | 0.159 | 0.529 | 45.8×* |
| OpFusion, independent MatMul+Add matches | 1.095 | 19.804 | 75.731 | 364.826 | 333.3× |
| DCE, dead siblings sharing one input | 0.280 | 4.179 | 17.053 | 57.306 | 204.3× |
| Constant folding, reverse-NodeId constant chain | 27.475 | 684.227 | 2810.138 | 11620.438 | 423.0× |
| EPContext dump, one-node partitions sharing one input | 14.456 | 320.520 | 1219.542 | 4822.435 | 333.6× |

\* The no-match fusion times are sub-millisecond and timer/cache noise is a
large fraction; absolute time and code structure show a fixed number of linear
scans.

### Scaling interpretation

- A 20× node increase produced 19.8–23.4× time for topological order and default
  executor build: empirically linear in this range.
- Constant folding and EPContext export are close to the expected 400× increase
  for quadratic work.
- Match-heavy fusion and DCE are strongly super-linear for exactly the
  code-predicted repeated scans/mutations.
- Shape inference is worse than quadratic scaling at the largest point due to
  its O(NV) scan crossing cache/allocation thresholds. The benchmark is a
  **lower bound**: interior values were anonymous, whereas loader-created ONNX
  values are named and `visible_scope` clones their names and type information.

## Ranked hotspots

### 1. High — per-node whole-value scan in shape inference

**Location:** `crates/onnx-runtime-shape-inference/src/infer.rs:78-101`,
`427-467`.

`visible_scope` is built for every node even when that node has no subgraph.
It scans all V values, rebuilds the formal-input HashSet, and clones named
bindings. This turns every generic or EP-scoped re-inference into O(NV).

**Fix:** only construct a visible scope when `child_keys` is non-empty. Maintain
an incremental name-to-`NodeIo` environment as outputs become available, and
pass a persistent/copy-on-write scope into actual child graphs. Ordinary
non-control-flow inference should be O(N+E+V).

### 2. High — constant folding fixpoint rescans the graph

**Location:** `crates/onnx-runtime-optimizer/src/constant_folding.rs:43-98`.

Legal node IDs need not be topological. Reverse IDs cause one newly foldable
node per round, yielding N full arena scans and 11.62 s at 20k nodes.

**Fix:** use a work queue. Track unresolved non-constant inputs per foldable
node; seed nodes whose inputs are initializers; when a node folds, enqueue its
consumers. Complexity becomes O(N+E+folded bytes).

### 3. High — EPContext partition boundaries are recomputed per partition

**Location:** `crates/onnx-runtime-loader/src/writer.rs:158-170`,
`187-253`, `279-319`.

Every partition rebuilds the graph-output set, sorts covered nodes, scans
boundary edges, and mutates shared consumer lists. One-node partitions took
1.22 s at 10k and 4.82 s at 20k.

**Fix:** assign every covered node a `PartitionId`, precompute graph-output
membership once, and make one global edge pass that appends each crossing edge
to the owning partition boundary. Splice all partitions in a batch and rebuild
edge metadata once.

### 4. High — no liveness-based buffer planner

**Location:** `crates/onnx-runtime-session/src/executor.rs:237-249`,
`1036-1065`, `1123-1148`.

The executor owns one allocation for every backed value. The synthetic
benchmark used one-element tensors, hiding realistic allocator and memory
pressure. Large transformer intermediates can make 10k-node static build OOM
even though compile time is only tens of milliseconds.

**Fix:** calculate value live intervals from the topological plan, pin graph
outputs/initializers/captures, and allocate from per-device reusable arenas by
size/alignment class. Peak memory should track simultaneously live values.

### 5. Medium-High — fusion restarts global matching after every rewrite

**Location:** `crates/onnx-runtime-optimizer/src/fusion.rs:219-239`,
`1265-1318`.

F successful matches trigger F scans from arena slot zero. Tombstones do not
shorten arena iteration. Match-heavy 20k graphs took 365 ms and scaled
quadratically.

**Fix:** build per-op anchor lists once and process a candidate worklist. After
a rewrite, invalidate matched IDs and enqueue only the fused node and adjacent
producers/consumers. Cache graph-output membership in a HashSet.

### 6. Medium — individual node deletion repeatedly filters consumer Vecs

**Location:** `crates/onnx-runtime-ir/src/graph.rs:496-525`;
`crates/onnx-runtime-optimizer/src/dead_node.rs:53-62`.

Wide-fanout deletion repeatedly runs `retain` over the same consumer vector.

**Fix:** add a bulk graph rewrite/removal API: mark all removals, compact each
value's consumers once, clear producers once, then garbage-collect values using
precomputed input/output membership sets.

### 7. Medium — duplicate capability/kernel work and linear registry lookup

**Location:** `crates/onnx-runtime-ep-api/src/registry.rs:58-81`;
`crates/onnx-runtime-session/src/executor.rs:778-825`, `1164-1198`.

CUDA coverage creates static kernels for validation and discards them; static
compile later repeats support and creation. Both CPU and CUDA registry lookups
scan all R entries.

**Fix:** index registry entries by normalized `(domain, op_type)` with a sorted
small version vector. Make CUDA coverage return/cache validated kernels or
combine coverage and `compile_all`.

### 8. Low-Medium — avoidable map and per-node allocation churn

**Location:** `crates/onnx-runtime-ir/src/graph.rs:237-260`;
`crates/onnx-runtime-session/src/executor.rs:860-1011`.

Topological and executor maps are created without capacities. Each `NodePlan`
owns four vectors. At run time, those vectors are cloned again per node
(`executor.rs:1609-1621`), and shape maps are rebuilt over V values.

**Fix:** reserve from N/V/E, use dense `Vec<Option<T>>` keyed by `ValueId` where
IDs are arena indices, store plan slots in compact flat arrays/`SmallVec`, and
borrow plan slices rather than cloning them in the run loop.

## Recommendations

### Must fix before claiming general 10k+ optimization/partition scalability

1. Remove the O(NV) `visible_scope` construction from ordinary per-node shape
   inference.
2. Replace constant folding's global fixpoint scans with a dependency worklist.
3. Implement an actual heterogeneous EP assignment/partition phase; define its
   target complexity as O(N+E) or O((N+E) log N), not repeated capability scans.
4. Batch EPContext boundary computation and graph splicing in one global pass.
5. Add liveness-based device-buffer reuse before testing realistic 10k-node
   transformer memory.
6. Convert fusion and bulk deletion to local worklists/batched edge updates.

### Nice to have

1. Pre-size HashMaps/HashSets and use dense ID-indexed vectors for transient
   analyses.
2. Index `OpRegistry` by op/domain and combine CUDA validation with compilation.
3. Hoist graph input/output membership into cached HashSets during rewrite
   pipelines.
4. Avoid deep-copying inline initializers during EPContext export; use shared
   immutable weight storage or copy-on-write graph metadata.
5. Add this benchmark (or Criterion equivalents) to performance CI with
   slope/ratio guards, not only absolute thresholds.
6. Add a 10k-node named-value shape-inference case after fixing
   `visible_scope`; the current anonymous-value result is deliberately a lower
   bound.

## Bottom line

Topological ordering and the default executor plan are not the problem. The
current scaling risks are optional optimization/re-inference, rewrite mutation,
EPContext partition export, and static memory planning. Those must be corrected
before “10k+ nodes” can be treated as a supported general case rather than a
fast-path exception.
