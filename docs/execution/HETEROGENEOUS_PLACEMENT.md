# Heterogeneous CPU+CUDA Placement

> **Status:** opt-in static execution slice, merged in #2095
> (`f6d893fd7`, 2026-08-25). Enable with `ONNX_GENAI_HETERO=1`.
>
> The default remains the existing single-EP executor and whole-session
> fallback. Heterogeneous execution is not yet the default and is not broad
> stateful-decode support.

## What exists today

`onnx-runtime-session::hetero` plans a mixed-provider graph and executes its
topologically ordered convex partitions through one ordinary child `Executor`
per partition. The existing `HeterogeneousPlan` remains the authority for node
assignment, partition boundaries, and cross-provider transfers.

The highest-priority provider that supports a node owns it. Unsupported-by-all
errors report the node/domain/opset, provider order, and each provider's decline
reason. Quantized GEMM prefill is no longer a forced CPU case: implemented CUDA
prefill kernels remain on CUDA, and CPU receives only nodes that the selected
accelerator provider genuinely declines.

### Supported graph subset

The opt-in executor accepts:

- acyclic top-level DAGs;
- fully static tensor shapes;
- tensor values only;
- ordinary named tensor inputs and outputs;
- independently owned kernel outputs rather than zero-copy views;
- model-local functions already legalized by the bounded fixpoint from #602.

It fails closed **before execution** for:

- symbolic or runtime-dynamic shapes and runtime-sized outputs;
- `If`, `Loop`, `Scan`, subgraph bodies, and sequence operators/values;
- view-producing kernels or strided/view partition boundaries;
- EPContext repartitioning;
- persistent external bindings, KV/state aliases, or other cross-EP state;
- mixed-provider CUDA graph capture, replay, or reset;
- attribute-parameterized model-local functions that still need overload-safe
  identity and call-site attribute binding.

These are explicit unsupported states, not deferred validation inside a
partially executed request.

## Ownership and transfer semantics

Each boundary value has one authoritative provider-owned `DeviceBuffer`.
Destination realizations are allocated through the destination EP and its
existing governor, only for planned cross-provider edges, and released by that
owner after the last consuming partition.

- H2D and D2D use `copy_async`, then `wait_fence` before the consumer.
- D2H is the documented synchronous edge:
  `copy_to_host` writes directly into a governed CPU `DeviceBuffer`.
- There is no intermediate host `Vec` and no whole-session synchronization.
- Realizations are deduplicated by `(value, destination EP)`.
- Each child executor owns its kernel cache, workspace sizing, initializer
  materialization, and memory accounting. One EP does not size or free another
  EP's storage.

`InferenceSession::heterogeneous_placement_report()` exposes assigned node
counts by EP and planned cross-provider transfer counts.

## Correctness invariants

1. Every accepted leaf node has exactly one selected EP.
2. Planning validates the entire supported subset before dispatch.
3. Every partition input is materialized on the selected EP before its kernel
   reads it.
4. Transfer fences complete before destination consumption.
5. A buffer is released exactly once by the EP that allocated it.
6. Mixed execution never masquerades as a single-EP capture key.
7. The flag-off path remains unchanged.
8. Unsupported state/control-flow/sequence/dynamic/view/capture cases fail
   closed with actionable diagnostics.

The merged acceptance tests lock a fake accelerator/CPU
`Relu -> Abs -> Neg` chain to two accelerator nodes, one CPU node, three
partitions, two transfers, and byte-identical output. A delayed fake async copy
proves that removing `wait_fence` corrupts the output. The real CUDA test locks
the same placement and transfer shape on the RTX 4060 path.

## Remaining phases

The remaining work is explicit:

1. Dynamic/shape-keyed placement and safe runtime-sized tensor outputs.
2. Persistent external binding, KV/state authority, and alias semantics across
   EPs.
3. Child heterogeneous executors for `If`, `Loop`, `Scan`, and sequence values.
4. Materialization of strided/view boundaries instead of pre-execution refusal.
5. Overload-safe function identity and correct call-site/`ref_attr_name`
   attribute binding.
6. Mixed-partition capture/cache re-keying, counters, and copy/compute overlap.
7. Cost-aware partitioning, pinned D2H staging, peer/multi-GPU copies, and
   real-model measurements before any default flip.

Issue #603 is **closed**. It tracked the older function-inlining phases: #602
landed the bounded legalization fixpoint, and #2095 exposed the first public
opt-in execution slice. The remaining list above is the current scope; the
closed issue's old “public wiring is deferred” wording is no longer current.

## Validation commands

The CPU-only planner/executor contract:

```powershell
cargo test --locked -p onnx-runtime-session hetero::tests -- --nocapture
```

The real CUDA acceptance test on a configured CUDA 13.1 host:

```powershell
$env:ONNX_GENAI_HETERO = "1"
$env:NXRT_REQUIRE_CUDA = "1"
cargo test --locked -p onnx-runtime-session --features gpu-tests `
  --test hetero_cuda_gpu -- --test-threads=1 --nocapture
```

Do not infer default enablement, dynamic graph support, stateful decode support,
or mixed capture support from either test.
