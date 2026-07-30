# Decision: every extension seam must expose a stable C ABI with dynamic loading

**Date**: 2026-07-30
**Decided by**: @justinchuby (owner)
**Source**: #524 Q1 (meta-decision raised by the contract audit)
**Status**: settled, standing directive

## Decision

The target form of out-of-tree extension is: **every extension seam gets a stable C ABI with full `.dll` / `.so` dynamic loading support.**

Providing only a compile-time Rust trait that requires third parties to link this workspace is not acceptable.

## Scope

All extension points, including but not limited to:

- Execution Providers (native EPs, not only the existing legacy plugin EPs)
- `DeviceAllocator` (bring your own memory management)
- `MemoryPlanner` (activation / workspace planning)
- `KvCacheStore` and `KvCacheConnector`
- `Sampler` / `LogitProcessor` / `SpeculativeProposer`
- `SchedulingPolicy`
- `OptimizationPass` (fusion / graph optimization)
- `Kernel` (adding or replacing a single kernel for an existing EP)
- `PlacementCostModel`, `WeightEvictionPolicy`, `ReclaimPolicy`
- `Communicator` (cross-device transport)

## Direct implications

1. **Rust traits are still required**, but they are the **in-process implementation layer, not the boundary**. Each seam needs a trait plus a C vtable shim in both directions (host→plugin and plugin→host).
2. **An ABI foundation becomes a prerequisite for every seam**: version negotiation, error propagation, panic fencing, and cross-boundary ownership must be unified first, or each seam will invent its own rules.
3. **The stability policy is raised from P2 to P0.** Committing to an ABI externally requires publishing stability tiers and a versioning mechanism at the same time.
4. **The existing plugin EP C ABI is the only validated exemplar** (`crates/onnx-runtime-ep-api/src/abi/runtime.rs:33-132`, with `ort_version_supported` negotiation; `registry.rs:220-226` for `libloading` + `CreateEpFactories` loading). It should be extracted into a pattern all seams reuse.
5. **The FFI requirements in RULES.md §1 become hard constraints**: C ABI calls must return a machine-parseable error code plus a retrievable rich message; never discard the Rust cause; never unwind across FFI.
6. **DLPack's role is likely settled by implication** (#524 Q3): with the boundary being a C ABI, DLPack is the natural choice as an existing C ABI standard for zero-copy cross-implementation tensor exchange. Pending explicit owner confirmation.
7. **Q2 becomes more urgent**: whether the EP ABI matches upstream ORT's plugin ABI or defines a native nxrt ABI now becomes the template for every other seam's ABI.

## Affected issues

- New: plugin ABI foundation (version negotiation / error propagation / panic fencing / ownership conventions / conformance tests)
- Raised: #512 stability policy, P2 → P0
- Shape change: #506, #508, #509, #510, #511, #513, #514, #515, #516, #517, #518, #519, #520 — each moves from "add a Rust trait" to "Rust trait + C ABI vtable + versioning"

## Note

This decision raises the effort for every seam substantially, but it is what the project's core goal requires: third parties optimizing for their own hardware without forking. Recommend freezing ABIs seam by seam rather than designing them all at once.
