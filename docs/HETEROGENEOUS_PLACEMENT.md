# Heterogeneous CPU+CUDA Placement for the Native Runtime

> **Status: AWAITING USER GREENLIGHT**
>
> This document is a design proposal. The current implementation remains a
> single-EP executor and fails CUDA model load when the selected CUDA EP cannot
> serve every node.
>
> **Primary targets:** GLM-5.2/DeepSeek-scale sub-4-bit decoder models.
>
> **Date:** 2026-07-16

## 1. Executive recommendation

Make `DevicePreference::Gpu` a CUDA-preferred, CPU-fallback native session:

```text
ONNX graph
    |
    v
shape-aware placement
    |
    +---- CUDA partitions: supported operators and M=1 BlockQuantizedMatMul
    |
    +---- CPU partitions: unsupported operators and M>1 quantized prefill
                         |
                         v
              explicit device-copy edges
```

Register CUDA at highest priority and the CPU EP as the universal fallback.
Assign each node to an EP by calling `ExecutionProvider::supports_op` with the
node's concrete input shapes and layouts. Coalesce adjacent nodes with the same
assignment into partitions, then insert explicit transfers where producer and
consumer partitions use different devices.

`BlockQuantizedMatMul` makes placement shape-dependent. Its CUDA implementation
is a GEMV kernel and accepts only `M=1`; prefill with `M>1` must use CPU, while
single-token decode should remain on CUDA. Placement therefore cannot be a
one-time operator-name decision. It must be cached by node plus resolved shape
class.

The first implementation phase should prioritize correctness:

1. CPU fallback for every unsupported CUDA node.
2. Synchronous host-staged transfers at partition boundaries.
3. Shape-keyed dispatch so quantized prefill runs on CPU and decode runs on CUDA.
4. Tests and diagnostics before transfer fusion, asynchronous copies, or cost
   tuning.

## 2. Why the current CUDA-only session is insufficient

The session builder currently selects one
`Arc<dyn ExecutionProvider>` for the whole executor. The comment in
[`onnx-runtime-session/src/lib.rs`](../crates/onnx-runtime-session/src/lib.rs#L529-L545)
explicitly keeps CUDA opt-in “until heterogeneous placement and fallback exist.”
`Executor` likewise owns one EP and one device allocation per value
([`executor.rs`](../crates/onnx-runtime-session/src/executor.rs#L209-L245)).

That architecture works only when the chosen EP supports every graph node and
every runtime shape. A real sub-4-bit decoder violates both conditions:

- standard operators such as `Transpose` may have no CUDA kernel;
- CUDA `BlockQuantizedMatMul` accepts only an activation whose leading
  dimensions multiply to one
  ([`block_quantized_matmul.rs`](../crates/onnx-runtime-ep-cuda/src/kernels/block_quantized_matmul.rs#L601-L640));
- a multi-token prompt produces `M>1`, so quantized prefill must run on CPU;
- later `M=1` decode is suitable for the CUDA GEMV kernel.

Adding a CPU EP to an error message or retrying the whole request on CPU is not
heterogeneous execution. It would either fail after partial work or discard the
CUDA decode path. Until the design below is implemented, CUDA-only load must
fail before generation when full CUDA coverage is unavailable.

## 3. Existing foundations

### 3.1 EP capability and registry APIs

`ExecutionProvider::supports_op` already accepts a node, inferred input shapes,
and layouts. `get_kernel` specializes a supported node for concrete shapes
([`provider.rs`](../crates/onnx-runtime-ep-api/src/provider.rs#L270-L281)).

`EpRegistry` already stores ordered providers and returns all candidates for a
node in priority order
([`registry.rs`](../crates/onnx-runtime-ep-api/src/registry.rs#L94-L153)).
CUDA-first/CPU-second placement should extend this registry rather than add
model-specific operator lists to the session.

The EP API also has two claim mechanisms that must remain authoritative:

- `ExecutionProvider::claim_nodes` for unconditional native claims
  ([`provider.rs`](../crates/onnx-runtime-ep-api/src/provider.rs#L399-L405));
- `SubgraphClaim` for plugin-EP graph capabilities
  ([`abi.rs`](../crates/onnx-runtime-ep-api/src/abi.rs#L17-L36)).

Explicit claims take precedence over ordinary cost-based candidates, but every
claimed partition must still identify its EP and device.

### 3.2 Partition boundary machinery

The EPContext writer already computes partition boundary inputs and outputs and
replaces covered nodes while preserving external edges
([`writer.rs`](../crates/onnx-runtime-loader/src/writer.rs#L185-L279)).
The runtime partitioner should reuse the same boundary definition and extract it
into a shared, non-serialization utility. The writer remains responsible only
for encoding compiled partitions; it must not become the execution planner.

### 3.3 Device memory and copies

`DeviceBuffer` records its owning `DeviceId`, and allocation/deallocation must
return to that owner. EPs expose host upload/download, device copy, asynchronous
copy, and fences
([`provider.rs`](../crates/onnx-runtime-ep-api/src/provider.rs#L24-L54),
[`provider.rs`](../crates/onnx-runtime-ep-api/src/provider.rs#L283-L365)).

Today CUDA `copy_async` is synchronous and returns an already-signalled fence.
That is sufficient for a correctness-first phase, but not for final
performance.

## 4. Proposed session architecture

### 4.1 Provider set

Replace the executor's single `ep` with a session-owned provider set:

```rust
struct SessionProviders {
    registry: EpRegistry,
    cuda: EpId,
    cpu: EpId,
}
```

For `DevicePreference::Gpu { index }`:

1. initialize CUDA on the requested index;
2. initialize the CPU EP;
3. register CUDA first and CPU second;
4. require CPU to support every non-executor-handled node that CUDA rejects;
5. report an actionable load error only if neither EP can serve a node.

`DevicePreference::Cpu` remains CPU-only. `Auto` should remain CPU-only until
the user explicitly approves changing its placement policy.

### 4.2 Placement result

Placement produces an immutable structural plan plus shape-keyed variants:

```rust
struct NodePlacement {
    node: NodeId,
    ep: EpId,
    input_devices: Vec<DeviceId>,
    output_device: DeviceId,
}

struct PlacementKey {
    node: NodeId,
    shapes: Vec<Vec<usize>>,
    layouts: Vec<TensorLayout>,
}

struct Partition {
    ep: EpId,
    nodes: Vec<NodeId>,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
}
```

The initial structural pass handles unconditional claims and nodes whose
support is shape-independent. Dynamic nodes are finalized after input symbols
are resolved. The resulting `PlacementKey` is cached alongside the existing
shape-keyed kernel cache.

### 4.3 Candidate selection

For each runnable leaf node:

1. honor a valid explicit EP claim;
2. call `EpRegistry::candidates_for_op(node, shapes, layouts)`;
3. discard candidates whose required input layouts cannot be satisfied;
4. prefer an already-resident producer device when costs are otherwise close;
5. choose CUDA before CPU for equal-cost candidates;
6. call `get_kernel` with the effective domain opset before committing;
7. fail load/run before dispatch if no candidate remains.

Control-flow and sequence operators remain executor-handled. Their nested
subgraphs are partitioned recursively using the parent model's opset imports.

### 4.4 Quantized prefill versus decode

For `com.github.onnxruntime.genai::BlockQuantizedMatMul`:

```text
M = product(activation.shape[..rank-1])

M == 1  and CUDA supports attrs/layout/format -> CUDA
M > 1                                 -> CPU
symbolic M at load                     -> defer to resolved run shape
```

The first multi-token invocation creates/caches the CPU kernel variant. Later
single-token invocations create/cache the CUDA variant. This is per-node
placement; it does not migrate the whole session.

Constant packed weights may need one CPU binding and one CUDA binding. Upload
the CUDA copy lazily on the first CUDA decode use rather than eagerly duplicating
all initializers at session load.

## 5. Executor and buffer changes

### 5.1 Values may have multiple device realizations

The current `HashMap<ValueId, DeviceBuffer>` becomes a device-aware table:

```rust
struct BufferKey {
    value: ValueId,
    device: DeviceId,
}

struct BufferEntry {
    owner: EpId,
    buffer: DeviceBuffer,
    shape: Vec<usize>,
    generation: u64,
}
```

Only one realization is authoritative for a mutable activation generation.
Copied realizations record the same generation and become stale when a producer
writes a newer value. Initializers are immutable and may retain multiple valid
realizations.

Deallocation always goes through `BufferEntry.owner`. This preserves the
current no-cross-EP-free invariant.

### 5.2 Transfer edges

Before dispatching a consumer, materialize each input on the consumer EP:

- CPU to CUDA: CPU-readable bytes → CUDA `copy_from_host`;
- CUDA to CPU: CUDA `copy_to_host` → CPU-owned buffer;
- CUDA device to the same CUDA device: reuse the existing realization;
- future cross-GPU: peer copy when available, otherwise pinned host staging.

Phase 1 may synchronize each transfer. A later phase can return fences, retain
staging buffers until completion, and overlap copies with independent
partitions.

Views need special treatment. A cross-device view cannot copy only its base
pointer metadata; the transfer planner must either:

1. copy/materialize the logical strided tensor on the destination, or
2. copy the backing allocation and recreate a valid destination view.

Correctness-first should materialize non-contiguous boundary values.

### 5.3 Partition execution

Coalesce adjacent nodes assigned to one EP when doing so does not cross a claim
boundary. Execute:

```text
materialize partition inputs
  -> await transfer fences
  -> dispatch cached kernels in topological order
  -> mark partition outputs authoritative on that EP
```

Partitioning reduces repeated transfers and creates the future unit for CUDA
graph capture and EPContext compilation. Node-level fallback remains valid when
coalescing would be unsafe.

### 5.4 KV cache interaction

Native CUDA decode currently binds KV storage to CUDA. Heterogeneous placement
must preserve one authoritative KV realization:

- CPU prefill may produce present KV on CPU;
- transfer the completed present tensors once to the CUDA bindings before
  single-token decode;
- CUDA decode appends in place and keeps subsequent KV on device;
- a later CPU fallback consumer downloads only the required logical range;
- rewind/reset update the authoritative logical length without duplicating
  stale KV state.

The first phase may transfer whole active KV tensors at the prefill/decode
boundary. Range-based and overlapped copies are later optimizations.

## 6. Correctness and failure invariants

1. Every leaf node has exactly one selected EP for a resolved shape.
2. `supports_op` is checked before `get_kernel`; `get_kernel` succeeds before
   any node in that placement variant dispatches.
3. Every input tensor is resident on the selected EP and has the expected
   dtype, shape, layout, and generation.
4. A `DeviceBuffer` is freed only by its owning EP.
5. A cross-device copy completes before the destination kernel reads it.
6. Mutable outputs invalidate older device realizations.
7. Request failure occurs before partial token streaming when placement cannot
   be completed.
8. CPU-only output remains bit-identical to the current CPU executor.
9. CUDA `M=1` quantized decode remains enabled; fallback does not collapse the
   whole request to CPU.
10. Diagnostics name the node, domain/opset, resolved shapes, attempted EPs,
    and remediation.

## 7. Observability

Expose per-session and per-request counters:

- nodes and partitions assigned to each EP;
- shape-keyed placement cache hits/misses;
- CPU↔CUDA transfer count and bytes;
- transfer wait time and kernel time by EP;
- `BlockQuantizedMatMul` CPU-prefill and CUDA-decode dispatch counts;
- initializer uploads and duplicate resident bytes;
- fallback reasons grouped by operator/domain and shape.

An optional placement dump should show:

```text
node 71 BlockQuantizedMatMul shape=[1,17,4096] -> cpu_ep (M=17)
node 71 BlockQuantizedMatMul shape=[1,1,4096]  -> cuda_ep (M=1)
node 84 Transpose                              -> cpu_ep
```

## 8. Validation plan

### 8.1 Unit tests

- CUDA-first/CPU-second candidate ordering.
- Unsupported CUDA node selects CPU.
- Node unsupported by both EPs fails before execution.
- Shape-keyed `BlockQuantizedMatMul`: `M>1` CPU, `M=1` CUDA.
- Initializer bindings can coexist on CPU and CUDA.
- Mutable writes invalidate stale copies.
- Non-contiguous boundary values materialize correctly.
- Nested subgraphs inherit provider placement and model opsets.

### 8.2 Integration tests

- Compact decoder with a reachable `M>1 BlockQuantizedMatMul` and `Transpose`:
  CPU generation succeeds; heterogeneous CUDA generation matches CPU.
- Real 144-`BlockQuantizedMatMul` sub-4-bit model with a multi-token prompt:
  prefill uses CPU, decode uses CUDA, and coherent tokens are produced.
- Single-token prompt still exercises unsupported CPU fallback nodes without
  losing CUDA quantized decode.
- CUDA-unavailable and CUDA-feature-disabled paths remain clear and non-panicking.
- Streaming emits no token before placement and initial transfers succeed.

### 8.3 Performance gates

Correctness is necessary but not sufficient. Before enabling the mode by
default, measure:

- prefill latency versus native CPU;
- decode tokens/s versus CUDA-only fully supported graphs;
- transfer bytes/token and boundary count;
- peak host RAM and VRAM;
- impact of partition coalescing and lazy initializer upload;
- H200-class real-model traces with CUDA `M=1` dispatch proven by counters.

## 9. Phased rollout

### Phase 0 — current safety gate

- Keep the CUDA-only executor.
- Probe every model node with CUDA `supports_op` at load.
- Reject unsupported nodes and symbolic/`M>1` quantized prefill before
  generation.
- Direct users to native CPU or the ORT backend.

### Phase 1 — synchronous CPU fallback

- Register CUDA and CPU in `EpRegistry`.
- Add shape-keyed per-node placement.
- Add device-aware value buffers and synchronous host-staged copies.
- Run `M>1 BlockQuantizedMatMul` and unsupported nodes on CPU.
- Preserve `M=1 BlockQuantizedMatMul` on CUDA.
- Ship only after compact and real-model token tests pass.

### Phase 2 — partitioning and residency

- Coalesce same-EP nodes using shared partition-boundary machinery.
- Lazily materialize initializers per EP.
- Transfer KV once at the prefill/decode boundary where possible.
- Add placement/transfer observability and memory accounting.

### Phase 3 — performance

- Implement real asynchronous CUDA copies and fences.
- Add pinned staging pools, transfer reuse, and overlap.
- Minimize CPU↔CUDA ping-pong with cost-aware partition selection.
- Integrate CUDA graph capture for stable CUDA partitions.

### Phase 4 — advanced placement

- Multi-GPU and peer copies.
- Memory-budget-aware placement and weight offload integration.
- Plugin-EP claims and compiled partition context export.
- Adaptive placement based on measured transfer and kernel costs.

## 10. Open questions requiring owner approval

1. Should explicit `DevicePreference::Gpu` mean CUDA-preferred with mandatory CPU
   fallback, or should a separate strict CUDA-only mode remain available?
2. Should placement cache by exact shapes or by bounded shape classes such as
   `M=1` versus `M>1`?
3. Is whole-KV transfer at the prefill/decode boundary acceptable for Phase 1,
   or must range-based KV copies land at the same time?
4. Should shared partition-boundary extraction live in IR, loader, or session?
5. How should cost selection balance fewer transfers against faster individual
   CUDA kernels?
6. Which counters and trace events are release-blocking evidence that real
   sub-4-bit decode remains on CUDA?

## 11. Decision

**AWAITING USER GREENLIGHT.** Do not implement heterogeneous placement as an
unreviewed extension of the current CUDA device-routing change.

Once approved, build CPU fallback first in `onnx-runtime-session` using the
existing EP registry/capability contracts, shape-keyed placement, and explicit
cross-device copies. Preserve CUDA `M=1 BlockQuantizedMatMul` decode while
routing `M>1` prefill and unsupported operators to CPU. Optimize partitioning,
copy overlap, and residency only after real-model correctness is demonstrated.
