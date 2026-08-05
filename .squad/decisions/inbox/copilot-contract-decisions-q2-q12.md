# Decision: contract questions Q2-Q12 settled

**Date**: 2026-07-30
**Decided by**: @justinchuby (owner)
**Source**: #524 (contract audit decision list)
**Status**: settled, standing directive

Companion to `copilot-plugin-c-abi-everywhere.md`, which records Q1.

## Q1 (amended) — C ABI *and* Rust trait, kept in sync

The earlier record stated "every seam must expose a stable C ABI". Amended: **both a C ABI and a Rust trait are required for every seam, and the two must stay in sync.** The Rust trait is not a transitional artifact — it is a first-class, permanently supported surface alongside the C ABI.

## Q2 — Ship both EP ABIs; evolve the ORT ABI toward the nxrt ABI

Provide both the upstream ORT plugin-EP ABI and a native nxrt ABI. The goal is to **gradually evolve the ORT ABI into the nxrt ABI**, with nxrt acting as the vanguard for ORT. Neither is discarded.

## Q3 — DLPack is the boundary, scoped to component seams

DLPack is the sanctioned interchange at **external / component boundaries**: bring-your-own-memory import and export, cross-implementation handoff, and Python or third-party interop. Use the versioned form `DLManagedTensorVersioned` (`crates/onnx-runtime-dlpack/src/lib.rs:142`), consistent with the Q1 versioning requirement.

DLPack is **not** the per-op currency inside the executor hot path. Per-op exchange stays on `ep-api::TensorView` / `TensorMut` (borrowed, non-owning, no allocation).

Rationale: DLPack is zero-copy by construction — `DLTensor` (`lib.rs:97`) carries only pointer and metadata. Per-exchange cost is roughly 50-200ns (one `DLManagedTensor` heap allocation, backing storage for `shape`/`strides`, and a deleter function-pointer call). Negligible per inference call; measurable if applied across the hundreds of ops in a decode step, where it would also discard information the internal types carry (device buffer handles, stride invariants, EP-private residency state).

Two hazards this creates, both in scope for the ABI foundation:
1. The DLPack deleter is a C function pointer supplied by the producer and invoked by the consumer. Across a dynamic-library boundary, an unloaded plugin makes this a use-after-free.
2. DLPack has no stream field and does not express async/device-stream semantics. An explicit stream and synchronization convention must be defined on top of it, or CUDA handoffs carry a latent race.

## Q4 — The ORT backend is permanent; both backends must be supported

The ORT backend is **not** a transitional bridge. Both backends are supported indefinitely and must remain at parity.

Context: this project is a demonstration to ONNX Runtime colleagues of what the runtime could evolve into — effectively **ORT 2.0**. Staying general-purpose is therefore a requirement, not a preference. Designs that specialize for the native backend at the expense of generality are not acceptable.

## Q5 — Placement is replaceable, and may also be supplied precomputed

The placement cost model must be replaceable, **and** callers must be able to supply a **precomputed** placement result directly.

Two modes:
1. **Computed** — an external scoring function the runtime consults per node;
2. **Precomputed** — an external node→EP placement plan the runtime executes as given, performing only legality checks (does the EP genuinely support this op / dtype / shape), with no scoring.

```rust
enum Placement {
    Computed(Arc<dyn PlacementCostModel>),
    Precomputed(PlacementPlan),   // serializable, for offline production and version diffing
}
```

`PlacementPlan` must be exportable: running the cost model once should dump a plan the user can edit and feed back. This also satisfies the "inspectable" requirement of RULES.md §5.

## Q6 — Experimental phase: no freezing, clarity instead

APIs and ABIs may change freely right now. No stability freeze, no compatibility shims (consistent with RULES.md §3). The requirement is that contracts be **clear** — naming, semantics, and ownership rules — not that they be stable.

Consequence: the extension-point stability policy work is reframed from "publish stability tiers" to "publish clear contract definitions", and is **not** a P0 blocker.

## Q7 — Open enums that select behavior; keep enums that define data semantics or safety invariants

Rationale: a third-party behavior variant at worst behaves differently. A third-party data-semantics or trust variant forces **every consumer** to handle unknown variants, and can change how bytes are interpreted or how security is judged.

**Open:**
- `PriorityPolicy` / `PreemptionPolicy` (`onnx-genai-scheduler/src/lib.rs:166-183`) → `SchedulingPolicy` trait, enums demoted to built-in factories
- `EngineDecodeBackend` (`onnx-genai-engine/src/config.rs:516`) → add `Custom(String)` resolved through a registry
- `KvConnectorBackend` (`onnx-genai-engine/src/config.rs:472`) → `Named(String, Value)`, dynamically loadable per Q8
- **Quantization type vocabulary** → `QuantCodec` registry keyed by `type_uri`. The design already exists at `docs/EXTENSIBLE_QUANT_TYPES.md:228-272` and is simply unimplemented. Quantization formats are inherently open-ended (nf4, mxfp4, and successors), so this must be open.

**Keep closed:**
- `HostTrust` (`onnx-model-package/src/lib.rs:65`) — a security boundary; a plugin must never be able to invent a trust level. The crate's own docs state packages are untrusted input.
- `PackageLayout` (`onnx-model-package/src/lib.rs:44`) — coupled to `HostTrust`; the `Installed` variant governs whether references may escape the package root. This is a security declaration, not a behavior selection.
- `onnx-runtime-ir::DataType` (`onnx-runtime-ir/src/dtype.rs:9`) — defines how bytes are interpreted. Opening it would require every kernel, EP, and serialization path to handle unknown dtypes. Extensibility for new numeric formats goes through the quant codec layer instead.

**Already open, to be improved:**
- `onnx-runtime-ir::DeviceType` (`device.rs:8-19`) already has `Custom(u32)`. A bare `u32` risks collisions between third parties; add a `name → id` registry so vendors receive stable ids while the ABI still carries a compact `u32`.

## Q8 — KV connector selection supports dynamic loading

`KvConnectorBackend` becomes an open, string-keyed registry supporting dynamically loaded connectors, rather than a curated in-tree enum.

## Q9 — Memory pressure: pull for policy, push for notification, never call plugin code under a lock

Three rules:

1. **Reclaim policy is pull, and must be a pure function over a snapshot.**
   ```rust
   fn arbitrate(&self, snapshot: &PressureSnapshot) -> ReclaimPlan;
   ```
   The governor invokes the policy, receives a plan, and executes it itself. The policy calls no external code. This makes it deterministically replayable, unit-testable, and free of allocator-lock reentrancy.

2. **Pressure notification is push, but asynchronous and advisory.** Subscribers are told the pressure level changed; delivery happens outside locks, and subscribers may not reclaim from within the callback. They may record, export metrics, or note intent for the next time they are asked.

3. **Actual reclamation is the governor calling each participant's `release(...)` directly**, in the order the `ReclaimPlan` specifies — a top-down explicit call, not a callback chain.

Rationale: because Q1 requires a C ABI for every seam, a push-callback policy would hit three problems at once — reentrancy and deadlock (callbacks typically fire while the allocator lock is held), panic/unwind across FFI (callback paths are the hardest place to fence, and RULES.md §1 forbids unwinding across FFI), and non-reproducibility (callback ordering depends on scheduling, and pressure scenarios are already hard to reproduce). A pull-based pure-function policy eliminates all three and is the shape easiest to get right across a C ABI.

Accepted cost: the policy cannot proactively trigger reclamation; it only responds when the governor asks. If checkpoint granularity proves too coarse in practice, the fix is **more frequent checkpoints, not a switch to push callbacks**.

## Q10 — Unify buffer and device types

Make `onnx-runtime-ir::DeviceType` the canonical device enum, consolidate `DeviceBuffer` in `ep-api` with `onnx-runtime-comm` depending on it, and keep higher-level types (`DevicePreference`, `CudaDevice`) as configuration-layer projections with lossless conversion to the canonical types. Layered projections are legitimate; same-layer duplicate definitions are not.

## Q11 — Merge the duplicate tensor types

`onnx-runtime-session::Tensor` and `onnx-runtime-eager::Tensor` are not intentional layering. Merge them.

## Q12 — `WeightEvictionPolicy` belongs in `ep-api`

So that CPU and other EPs can reuse the residency concept, rather than it being CUDA-specific.
