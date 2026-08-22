# Native backend for the universal `pipeline.workflow` runtime

Status: **staged** — this document tracks the architecture and the incremental
landing plan. The first increment (a backend-neutral component-execution seam,
a native executor, and the `EngineDecodeBackend::Native` acceptance flip) lands
with parity tests; the follow-up boundaries below are called out explicitly.

Read [`RULES.md`](../../RULES.md) first. The two rules that shape every decision
here are **Rule 2** (no model/vendor/EP identity conditionals — behaviour is
driven by metadata and declared capability) and **Rule 10** (reduce entropy:
two code paths answering the same question are duplicated state; collapse them
into one classification, and a rejecting rule returns *why*).

## 1. The duplication we are collapsing

Before this work there were four places that "run a component graph", and they
did not share an implementation:

1. **The universal workflow interpreter** — `pipeline/workflow.rs`
   (`run_workflow_node`). This is the one place that already knows how to
   interpret SSA values, loops, branches, emits, loop-carried state cells,
   session-scoped state, adapters, effects, and session checkpointing. It is
   *universal over workflows* but **not** over backends: it reaches an ORT
   `Session` directly through `self.models.session(name)` and threads
   `onnx_genai_ort::Value` as its only tensor currency
   (`type PipelineTensors = HashMap<String, Value>`). `run_stable_component`
   additionally owns the ORT `IoBinding` / CUDA-graph execution-island path.

2. **`ComponentSession` / `ComponentTensor`** — `onnx-genai-metadata`. A second,
   *backend-neutral* component seam that was introduced to let the engine drive
   a component through either backend. It is **host-resident only**
   (`ComponentTensor` carries raw little-endian bytes), it has exactly one
   implementor (`OrtComponentSession` in `onnx-genai-ort`), and — critically —
   **the interpreter never calls it.** It is duplicated state: a second answer
   to "how do I invoke a component" that forces a host round-trip and is wired
   to nothing.

3. **The direct `Engine`** — `engine/`. Non-workflow generation keeps its own
   native session lifecycle (`native_session`, `native_sessions`,
   `native_active_session`, `default_native_session`, `create_native_session`,
   `close_native_session`) and its own ORT-vs-native routing in
   `engine/runtime.rs::generate_with_callbacks`.

4. **The autoregressive token loop** — this one is **already unified**.
   `decode_loop.rs::run_decode_loop` + the `DecodeLoopBackend` trait own the
   single token loop, and both `SessionDecodeLoopBackend` (ORT) and
   `NativeLoopAdapter` (native) implement it. Sampling, stopping, constraint
   application and KV commit have one authoritative home
   (`processors.rs::select_next_token*`, `finish_reason_after_token`,
   `ensure_constrained_finish`, `decode_loop.rs::commit_selected_token`,
   `native_decode/kv_commit.rs`). **We must not add a third loop.**

`PipelineEngine::validate_pipeline_backend_request` rejected
`EngineDecodeBackend::Native` outright, so the universal interpreter was
unreachable from the native backend even though (4) proves native execution of
component graphs already exists.

## 2. Target ownership boundaries

One interpreter, one value currency, one component-execution seam.

```
            ┌────────────────────────────────────────────┐
            │        universal workflow interpreter        │  pipeline/workflow.rs
            │  SSA · loops · branches · emits · state ·    │  (ONE implementation,
            │  adapters · effects · session checkpoints    │   backend-agnostic)
            └───────────────┬──────────────────────────────┘
                            │ invoke_onnx_component(name, &[(&str,&Value)], …)
                            ▼
            ┌────────────────────────────────────────────┐
            │      WorkflowComponentBackend (seam)         │  pipeline/component_backend.rs
            │  fn output_names / outputs / device_id       │
            │  fn run(&[(&str,&Value)]) -> Vec<(_, Value)> │
            └───────┬───────────────────────────┬──────────┘
                    │                            │
        ┌───────────▼─────────┐      ┌───────────▼────────────────────┐
        │ Ort backend         │      │ Native backend                  │
        │ · Session::run      │      │ · InferenceSession::run         │
        │ · IoBinding /       │      │ · Value ⇄ native Tensor bridge  │
        │   CUDA-graph island │      │   (CPU: bytes; CUDA: device     │
        │   (run_stable_…)    │      │    Value via external memory)   │
        └─────────────────────┘      └─────────────────────────────────┘
```

**Value currency stays `onnx_genai_ort::Value`.** This is deliberate and is the
reason the interpreter does *not* need a rewrite. `Value` is already a
device-capable *handle*, not an ORT-compute-specific container: it exposes
`is_host_resident`, `device_id`, `copy_from_cuda`, aliasing (`try_alias_clone`,
`into_alias_with_shape`) and, importantly, `from_external_memory`, which lets a
native device allocation be exposed as a device-resident `Value` **without a
host round-trip**. So every requirement about keeping loop-carried and shared
state backend/device-resident is met by the currency the interpreter already
threads — the seam abstracts *execution*, not the value type. This is the
opposite of the `ComponentTensor` seam, which forced host bytes; that seam is
therefore removed (§5), not extended.

Host materialization stays confined to the boundaries that already require it:
branch-predicate inspection, emit slicing, row selection, adapters, and the
package/API boundary. No new host round-trip is introduced on a recurring
component edge.

### Autoregressive decode

A canonical AR workflow (the `decoder` fixture) already expresses the token
loop declaratively: a `Loop` node re-invokes the decoder component, with
sampling and stopping expressed as ordinary policy components
(`token_sampler.onnx`, `termination.onnx`) and KV/length carried by
loop-carried state cells. Running that workflow on the native backend therefore
runs AR decode **through the one interpreter loop** — no second or third loop is
introduced. The optimized `run_decode_loop` / `NativeDecodeSession` path
(device sampling fast paths, in-place KV) remains the specialized executor for
the direct `Engine`; §6 describes folding it under the interpreter's decoder
`Loop` node as a *specialized component executor* rather than a parallel loop.

## 3. The seam

`WorkflowComponentBackend` is engine-internal (it speaks `onnx_genai_ort::Value`,
which the metadata crate must not depend on, so it cannot live where the old
`ComponentSession` did). It exposes only what `invoke_onnx_component` needs:

* `output_names()` / `outputs()` — to publish results into the value pool and to
  resolve output shapes/dtypes.
* `device_id()` — so the interpreter can keep its existing "stable eligible on
  CUDA" classification (Rule 2: a capability, not a backend name).
* `run(&[(&str, &Value)]) -> Vec<(String, Value)>` — named-tensor execution.

The ORT implementation is a behaviour-preserving wrapper over the existing
`Session::run` / `run_stable_component` paths. The native implementation wraps an
`onnx-runtime-session::InferenceSession` and bridges values at the boundary.

### Fail-closed, never silent fallback (Rule 4)

When `EngineDecodeBackend::Native` is selected, an unsupported construct fails
with an actionable diagnostic naming the component/adapter/optimization — it does
**not** silently fall back to ORT:

* CUDA-graph execution islands and the `IoBinding` stable path are an ORT-only
  optimization; under Native they are not planned (the compiled graph keeps its
  individual component nodes), so correctness is preserved with no island.
* Adapters and contracts the native backend cannot honor return
  `Err(…)` describing the exact ABI/version/port, with ORT/native error parity.

## 4. Backend/device residency

| Edge | ORT | Native CPU | Native CUDA (delivered) |
|------|-----|-----------|--------------------------|
| component output → value pool | `Value` (device or host) | `Value::from_raw_bytes` (host; CPU has no device) | device `Value` via `from_external_memory_with_owner` over a device output binding (zero-copy) |
| loop-carried / shared state | `Value` alias | `Value` (host) reused, no re-serialization | device `Value` alias (`try_alias_clone` shares the owning `Arc`), no host round-trip |
| host-policy boundary | `to_raw_bytes` | `to_raw_bytes` | `to_host_from_cuda` at the package/API boundary |

The native executor holds its `InferenceSession`s for the life of the engine and
runs each per iteration, so a recurring component edge reuses one session — the
"native session actually used" and "no host round-trip on recurring edges"
invariants are observable through a per-backend run counter.

## 5. Staged deletion plan

> **Update:** the direct-`Engine` collapse (boundary C) is now the committed
> direction, tracked authoritatively in
> [`WORKFLOW_RUNTIME_UNIFICATION.md`](WORKFLOW_RUNTIME_UNIFICATION.md). Its
> Phase 0 (deleting the deprecated `*_native_*` session compatibility shims and
> migrating the one caller to the unified `create_session`/`generate_in_session`
> API) has landed; boundaries A/B/D below remain as written.

* **This increment**
  * Add `WorkflowComponentBackend` seam + `invoke_onnx_component`; route the
    interpreter's ONNX-component case through it (ORT path unchanged — proven by
    existing conformance).
  * Add the native executor (feature `native-backend`), the `Value ⇄ Tensor`
    bridge, and the run counter.
  * Accept `EngineDecodeBackend::Native` for workflow packages; fail closed for
    unsupported constructs.
  * **Remove** the redundant host-only `ComponentSession` / `ComponentTensor` /
    `ComponentIo` / `ComponentError` (metadata) and `OrtComponentSession`
    (onnx-genai-ort). They have no callers; keeping them would be exactly the
    "duplicate old symbol just in case" that Rule 3 forbids. The still-useful
    `DataType ⇔ ir::DataType` mapping is retained where the native bridge needs
    it.
  * **Rubber-duck fix 1 — no ORT sessions under Native.** `PipelineEngine::build`
    now loads component graphs with `PipelineModels::load_with_ort_session_filter(
    …, |_| false)` under Native, so **zero** ORT `Session`s are constructed — the
    package's I/O contract stays available as backend-neutral `graph_io_metadata`
    (read from the ONNX graph without instantiating ORT). This removes the ORT
    dependency / double load / misreported EP, and lets a package whose component
    ORT would reject at load run natively. `execution_provider_status()` reports
    the real native device. (`native_backend_builds_no_ort_sessions`,
    `ort_backend_builds_ort_sessions`.)
  * **Rubber-duck fix 2 — explicit device/provider + real device-resident bridge.**
    The native executor no longer calls `InferenceSession::load` (which
    auto-detects a CPU EP and ignores the requested device). It resolves the
    device via `resolve_native_decode_device(config.native_device, …)` and builds
    each session with the explicit EP (`CpuExecutionProvider` /
    `CudaExecutionProvider`), mirroring `native_decode/load.rs`. The `Value ⇄
    Tensor` bridge is residency-aware: on CPU it round-trips through bytes (no
    device round-trip); on CUDA it keeps tensors **device-resident end-to-end**
    (boundary A, below). It fails closed only for a genuinely unsupported
    device/provider (e.g. a plugin EP), never by silently host-copying or reading
    device memory as host bytes.
* **Delivered boundary A — native CUDA device-resident bridge.** Implemented
  behind `native-cuda`. A guard-carrying constructor
  `Value::from_external_memory_with_owner(ptr, …, owner)` lets a `Value` **own**
  the native `Tensor`/`DeviceIoBinding` behind the device pointer it wraps: the
  owner is an `Arc<dyn Any + Send + Sync>` in `TensorBacking::External`, freed by
  `Value`'s `Drop` *after* the `OrtValue` is released, so there is no leak and no
  use-after-free, and `try_alias_clone` hands out another `External` sharing the
  owner so a device value flows through the interpreter's `clone_value` fast path
  with no host read. In `native_component.rs` the CUDA path binds device-resident
  inputs zero-copy via `ExternalMemorySpec` + `device_binding_from_external_memory`,
  binds a device **output** buffer (`allocate_device_output_binding`) for every
  output whose concrete shape the interpreter already resolved from the bound
  input symbols (`resolve_component_output_shapes`), runs via
  `run_with_device_bindings`, and republishes each device output as an owning
  `Value` — so a recurring/loop-carried/state tensor (KV cache) never round-trips
  the host. Cross-session stream correctness uses the conservative
  `Tensor::sync()` / allocator-sync barrier before an output is handed on. Host
  materialization happens only at genuine host inputs (token ids) and the
  package/API boundary (`package_outputs`, `Emit`). Proven on H200 by
  `native_cuda_device_resident_multicomponent`: ORT/native bit-for-bit parity on
  a two-step static-cache decode **and** non-zero device-residency counters
  (device input bindings + device outputs) that prove the recurring KV stayed on
  the device. Dynamic-shape outputs (not resolvable ahead of the run) fall back
  to host materialization — the correct, safe fallback. Remaining: CUDA
  execution-island / graph-capture optimizations stay ORT-only (fail-closed under
  Native); reusing device output buffers across loop steps instead of allocating
  per step is a perf follow-up, not a correctness gap.
* **Follow-up boundary B — specialized AR executor.** Recognize the canonical
  single-decoder `Loop` and delegate it to `run_decode_loop` /
  `NativeDecodeSession` as a specialized component executor (device sampling
  fast paths, in-place KV) instead of re-entering the generic interpreter per
  token. Correctness is already delivered by the generic loop; this is a
  perf/entropy fold, not a behaviour change.
* **Follow-up boundary C — direct `Engine` façade.** Re-express the direct
  `Engine`'s bespoke native session lifecycle and ORT/native routing as a thin
  façade over a synthesized canonical single-decoder workflow, so
  `engine/runtime.rs` stops being a second orchestrator.
* **Follow-up boundary D — Gemma4 target+assistant — DONE.** Target-owned KV
  groups, read-only shared KV consumed by a cacheless assistant, heterogeneous
  full/sliding groups, target hidden-state handoff, and speculative
  accept/reject/rollback now run through the same seam with **no model-name
  conditional**: `pipeline::speculative` reads `speculative.proposal_execution`
  (`folded_carry_seed`, `folded_carry_output`, `token_embedding`, `recurrent[]`)
  and drives the proposer through `invoke_component_values`. The model-name-gated
  `Gemma4SharedKvSpec` / `Gemma4AssistantSignature` path and the whole
  `SharedKvProposerConfig` orchestration beneath it are deleted, not wrapped.
  See [`WORKFLOW_RUNTIME_UNIFICATION.md`](WORKFLOW_RUNTIME_UNIFICATION.md).

Each boundary is independently landable and independently testable; none require
re-opening the interpreter.
