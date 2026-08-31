//! `ExportedEp` — the heap object behind an opaque `OrtEp*`.
//!
//! Implements `GetCapability`, `Compile`, `ReleaseNodeComputeInfos`, and
//! `GetKernelRegistry` by delegating to the Rust `ExecutionProvider` trait.

use std::collections::HashSet;
use std::panic::AssertUnwindSafe;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use onnx_genai_ort_sys as ort;
use onnx_runtime_ep_api::provider::ExecutionProvider;
use onnx_runtime_ir::{DataType, DeviceType, NodeIndex, ValueId};

/// How an [`ExportedEp`] holds its backing Rust [`ExecutionProvider`].
///
/// Device EPs (CUDA) advertise a *shared* EP: the same instance backs the
/// factory's allocator, stream, and data-transfer, so `CreateEp` must reuse it
/// (rather than constructing a fresh EP whose stream/allocator would not match
/// the memory ORT allocated through the factory). CPU EPs own their instance.
pub enum EpHandle {
    /// The EP is owned exclusively by this `ExportedEp` (CPU plugin path).
    Owned(Box<dyn ExecutionProvider>),
    /// The EP is shared with the factory's allocator/stream/data-transfer
    /// (CUDA plugin path). `ExportedEp` must not shut it down on release.
    Shared(Arc<Mutex<Box<dyn ExecutionProvider + Send>>>),
}

impl EpHandle {
    /// Runs `f` with a shared reference to the backing EP. For the shared
    /// variant the mutex is locked only for the duration of `f`; callers must
    /// not re-enter ORT (allocator/stream) while inside `f`.
    pub fn with<R>(&self, f: impl FnOnce(&dyn ExecutionProvider) -> R) -> R {
        match self {
            EpHandle::Owned(ep) => f(ep.as_ref()),
            EpHandle::Shared(shared) => {
                let guard = shared.lock().unwrap_or_else(|p| p.into_inner());
                f(guard.as_ref())
            }
        }
    }

    /// Returns the EP name.
    pub fn name(&self) -> String {
        self.with(|ep| ep.name().to_string())
    }

    /// Shuts down the EP only if it is exclusively owned. Shared EPs are owned
    /// by the factory and are shut down when the factory is released.
    pub fn shutdown_if_owned(&mut self) {
        if let EpHandle::Owned(ep) = self {
            let _ = ep.shutdown();
        }
    }
}

/// Global counter of nodes compiled by this EP (across all subgraphs).
/// Incremented in `ep_compile` for each kernel entry. Used for test
/// observability: confirms our EP actually compiled nodes (not just that
/// ORT's built-in fallback produced correct output).
static COMPILED_NODE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Global counter of C-ABI `GetCapability` callback invocations.
///
/// Session-creation tests read this through the plugin cdylib to prove that a
/// rejection happened inside the real ORT capability path rather than during
/// model parsing before the EP was consulted.
static GET_CAPABILITY_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Returns the number of C-ABI `GetCapability` callback invocations.
pub fn get_capability_call_count() -> usize {
    GET_CAPABILITY_CALL_COUNT.load(Ordering::Relaxed)
}

/// Resets the C-ABI `GetCapability` callback counter.
pub fn reset_get_capability_call_count() {
    GET_CAPABILITY_CALL_COUNT.store(0, Ordering::Relaxed);
}

/// Returns the total number of nodes compiled by our EP since process start.
/// Reset with [`reset_compiled_node_count`] before a test session.
pub fn compiled_node_count() -> usize {
    COMPILED_NODE_COUNT.load(Ordering::Relaxed)
}

/// Resets the compiled-node counter to zero.
pub fn reset_compiled_node_count() {
    COMPILED_NODE_COUNT.store(0, Ordering::Relaxed);
}

/// Global counter of node inputs this EP told a kernel to treat as
/// session-lifetime constants (see [`constant_input_flags`]).
///
/// A weight that is not reported constant is not a correctness bug, which is
/// exactly why it needs a counter: the kernel still computes the right answer,
/// it just rebuilds its prepack on every `Run` — and `MatMulNBits` also drops
/// to a slower kernel, because its MLAS SQNBit path is gated on the same flag.
/// Nothing in an output comparison can see that, so the wiring is observed
/// here instead.
static CONSTANT_WEIGHT_INPUTS: AtomicUsize = AtomicUsize::new(0);

/// Number of node inputs reported to kernels as constant since process start.
pub fn constant_weight_inputs() -> usize {
    CONSTANT_WEIGHT_INPUTS.load(Ordering::Relaxed)
}

/// Resets the constant-weight-input counter to zero.
pub fn reset_constant_weight_inputs() {
    CONSTANT_WEIGHT_INPUTS.store(0, Ordering::Relaxed);
}

use crate::compute::ExportedComputeInfo;
use crate::graph_reader::OutboundGraphReader;
use crate::status::{fail_status, invalid_arg_status, ok_status};

// ─── Kernel registry entry type ─────────────────────────────────────────────

/// Describes a single operator kernel for ORT's `GetKernelRegistry` type-constraint
/// advertisement. Sourced from the Rust EP's real registry — do not hand-maintain.
#[derive(Clone, Debug)]
pub struct KernelRegistryEntry {
    /// ONNX operator type (e.g. `"Add"`, `"MatMul"`).
    pub op_type: &'static str,
    /// ONNX domain (empty string = default `ai.onnx`; or `"com.microsoft"` etc.).
    pub domain: &'static str,
    /// Starting opset version that is supported.
    pub since_version: i32,
    /// Ending opset version (inclusive). Set equal to `since_version` for single version.
    pub end_version: i32,
    /// Supported element types for the `"T"` type-constraint parameter.
    /// Import from `kernel_ctx::CPU_EP_SUPPORTED_DTYPES` to keep in sync.
    pub supported_dtypes: &'static [DataType],
    /// Per-input-slot dtype constraints, for ops whose edges are not uniform.
    ///
    /// `supported_dtypes` is a union, so on a mixed-dtype op it admits
    /// combinations no kernel accepts — a `MatMulNBits` with `float16`
    /// `zero_points` passes because both `float16` and `uint8` are somewhere in
    /// the union. Each `(slot, dtypes)` entry here overrides the union for that
    /// input position; unlisted and absent inputs keep the union rule. Empty
    /// means "the union is exact", which is true for every uniform op.
    pub input_dtype_constraints: &'static [(usize, &'static [DataType])],
    /// Per-output-slot dtype constraints for mixed-dtype ops.
    ///
    /// Each listed output position overrides `supported_dtypes`, exactly like
    /// `input_dtype_constraints`. This keeps capability admission aligned with
    /// kernels whose outputs have a narrower contract than their input union.
    pub output_dtype_constraints: &'static [(usize, &'static [DataType])],
}

/// A heap-allocated EP whose raw pointer is returned as `OrtEp*`.
///
/// The first field is `OrtEp` so the pointer can be cast directly.
#[repr(C)]
pub struct ExportedEp {
    /// The vtable ORT reads through the `OrtEp*` pointer.
    pub vtable: ort::OrtEp,
    /// The Rust EP instance (owned for CPU, shared for CUDA device EPs).
    pub ep: EpHandle,
    /// EP name kept alive for `GetName` callback.
    pub name_cstr: std::ffi::CString,
    /// ORT kernel registry built from [`KernelRegistryEntry`] slices.
    /// Remains valid for the lifetime of this EP (ORT requirement).
    /// `None` means the EP uses compile-only mode (no type-constraint metadata).
    pub kernel_registry: Option<OrtKernelRegistryHolder>,
    /// The same registry entries used to build `kernel_registry`, kept here so
    /// that `GetCapability` can dtype-filter claims against the same source of
    /// truth. Empty when no registry entries were provided.
    pub registry_entries: Vec<KernelRegistryEntry>,
    /// Whether ORT places this EP's tensors in host-accessible memory, copied
    /// from the factory's `DeviceSupport::host_accessible` in `CreateEp`.
    ///
    /// `Compile` hands it to each [`ExportedComputeInfo`] so `Compute` can tell
    /// host placement from device placement without querying ORT per `Run`.
    /// Defaults to `false` ("assume device"), the conservative direction: an EP
    /// built by a path that does not set it keeps the full memory-info scan.
    pub host_accessible: bool,
}

/// Owns an `OrtKernelRegistry*` allocated via ORT's EP API.
///
/// Releases it on drop via `ReleaseKernelRegistry`.
#[derive(Debug)]
pub struct OrtKernelRegistryHolder {
    ptr: *mut ort::OrtKernelRegistry,
}

// SAFETY: The kernel registry is read-only after construction.
unsafe impl Send for OrtKernelRegistryHolder {}
unsafe impl Sync for OrtKernelRegistryHolder {}

impl Drop for OrtKernelRegistryHolder {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let api = crate::status::host_api();
        if api.is_null() {
            return;
        }
        let ep_api = unsafe {
            let Some(get_ep_api) = (*api).GetEpApi else {
                return;
            };
            get_ep_api()
        };
        if ep_api.is_null() {
            return;
        }
        if let Some(release) = unsafe { (*ep_api).ReleaseKernelRegistry } {
            unsafe { release(self.ptr) };
        }
    }
}

impl ExportedEp {
    pub fn new(ep: Box<dyn ExecutionProvider>) -> Self {
        Self::new_with_registry(ep, None)
    }

    /// Construct with an optional pre-built kernel registry.
    ///
    /// When `registry` is `Some`, ORT uses its type constraints to validate
    /// node→EP routing (enables f16/bf16). When `None`, the EP is compile-only
    /// and ORT assumes all types are handled (per header: "If set to NULL, ORT
    /// assumes the EP compiles nodes").
    pub fn new_with_registry(
        ep: Box<dyn ExecutionProvider>,
        registry: Option<OrtKernelRegistryHolder>,
    ) -> Self {
        Self::new_with_registry_and_entries(ep, registry, Vec::new())
    }

    /// Construct with an optional pre-built kernel registry AND the source
    /// entries for dtype-aware claim filtering in `GetCapability`.
    ///
    /// `entries` are the same descriptors used to build `registry`. Keeping
    /// them here ensures the claim predicate and the advertised type constraints
    /// agree **by construction** — no independently maintained list to drift.
    pub fn new_with_registry_and_entries(
        ep: Box<dyn ExecutionProvider>,
        registry: Option<OrtKernelRegistryHolder>,
        entries: Vec<KernelRegistryEntry>,
    ) -> Self {
        let name = ep.name().to_string();
        Self::from_handle(EpHandle::Owned(ep), &name, registry, entries)
    }

    /// Construct an `ExportedEp` that reuses a *shared* EP (device/CUDA path).
    ///
    /// The shared EP is the same instance the factory hands to its allocator,
    /// stream, and data-transfer, so a graph compiled here allocates and runs
    /// on the exact context ORT already uses for memory transfers.
    pub fn new_shared(
        ep: Arc<Mutex<Box<dyn ExecutionProvider + Send>>>,
        name: &str,
        registry: Option<OrtKernelRegistryHolder>,
        entries: Vec<KernelRegistryEntry>,
    ) -> Self {
        Self::from_handle(EpHandle::Shared(ep), name, registry, entries)
    }

    fn from_handle(
        ep: EpHandle,
        name: &str,
        registry: Option<OrtKernelRegistryHolder>,
        entries: Vec<KernelRegistryEntry>,
    ) -> Self {
        let name_cstr = std::ffi::CString::new(name)
            .unwrap_or_else(|_| std::ffi::CString::new("nxrt_ep").unwrap());
        let has_registry = registry.is_some();
        Self {
            vtable: ort::OrtEp {
                ort_version_supported: ort::ORT_API_VERSION,
                GetName: Some(ep_get_name),
                GetCapability: Some(ep_get_capability),
                Compile: Some(ep_compile),
                ReleaseNodeComputeInfos: Some(ep_release_node_compute_infos),
                GetKernelRegistry: if has_registry {
                    Some(ep_get_kernel_registry)
                } else {
                    None
                },
                GetPreferredDataLayout: None,
                ShouldConvertDataLayoutForOp: None,
                SetDynamicOptions: None,
                OnRunStart: None,
                OnRunEnd: None,
                CreateAllocator: None,
                CreateSyncStreamForDevice: None,
                GetCompiledModelCompatibilityInfo: None,
                ..Default::default()
            },
            ep,
            name_cstr,
            kernel_registry: registry,
            registry_entries: entries,
            host_accessible: false,
        }
    }
}

// ─── OrtEp callbacks ────────────────────────────────────────────────────────

/// GetName: return the EP name as a C string.
unsafe extern "C" fn ep_get_name(ep: *const ort::OrtEp) -> *const std::ffi::c_char {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ep.is_null() {
            return c"unknown".as_ptr();
        }
        let exported = unsafe { &*(ep.cast::<ExportedEp>()) };
        exported.name_cstr.as_ptr()
    }));
    result.unwrap_or(c"unknown".as_ptr())
}

// ─── OrtEp callbacks ────────────────────────────────────────────────────────

/// GetCapability: read ORT's graph, ask our EP which nodes it supports, report
/// via `OrtEpApi.EpGraphSupportInfo_AddNodesToFuse`.
unsafe extern "C" fn ep_get_capability(
    ep: *mut ort::OrtEp,
    graph: *const ort::OrtGraph,
    support: *mut ort::OrtEpGraphSupportInfo,
) -> *mut ort::OrtStatus {
    GET_CAPABILITY_CALL_COUNT.fetch_add(1, Ordering::Relaxed);
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        ep_get_capability_inner(ep, graph, support)
    }));
    result.unwrap_or_else(|_| fail_status("GetCapability: internal panic"))
}

fn ep_get_capability_inner(
    ep: *mut ort::OrtEp,
    graph: *const ort::OrtGraph,
    support: *mut ort::OrtEpGraphSupportInfo,
) -> *mut ort::OrtStatus {
    if ep.is_null() || graph.is_null() || support.is_null() {
        return invalid_arg_status("GetCapability: null argument");
    }

    let exported = unsafe { &*(ep.cast::<ExportedEp>()) };

    // Read ORT's graph into our IR-compatible structures.
    let reader = match unsafe { OutboundGraphReader::from_ort_graph(graph) } {
        Ok(reader) => reader,
        Err(msg) => {
            return fail_status(&format!("GetCapability: failed to read ORT graph: {msg}"));
        }
    };

    // Build a GraphView and query capabilities using the shared
    // `OrtGraphView::query_capabilities`.
    let ir_graph = reader.to_ir_graph();
    let cache = match onnx_runtime_ir::GraphViewCache::build(ir_graph) {
        Ok(c) => c,
        Err(e) => {
            return fail_status(&format!("GetCapability: failed to build graph cache: {e}"));
        }
    };
    let view = onnx_runtime_ir::GraphView::new(ir_graph, &cache);
    let ort_view = onnx_runtime_ep_api::abi::OrtGraphView::new(&view);

    // Is this a non-CPU (GPU) EP? Two capability-time gates below are specific to
    // the GPU plugin and must not disturb the CPU plugin, whose paths already
    // work. Determined from the EP's own device type.
    let is_gpu_ep = exported
        .ep
        .with(|ep| ep.device_type() != onnx_runtime_ir::DeviceType::Cpu);

    // Claim-time routability gate (B2), GPU-only: the plugin's *multi-node*
    // fused-subgraph compile path (`build_subgraph_routing`, invoked only when a
    // claim has >1 node) can only thread a node input that is a graph input or a
    // prior node's output. A node that consumes a weight *initializer* (a value
    // with no producer that ORT did not surface as a graph input) is therefore
    // un-routable inside a multi-node subgraph. Declining such nodes *here* — at
    // capability time, before convex partitioning — lets ORT partition around
    // them instead of discovering the problem at Compile time, when ORT has
    // already committed the partition and can only surface a hard
    // session-creation failure (the CUDA decoder load regression).
    //
    // This gate is GPU-only. A *single-node* claim never calls
    // `build_subgraph_routing` and reads its initializer directly, so the CPU
    // plugin (e.g. `conformance_matmul_initializer_weights`: a lone MatMul with
    // initializer weights) must keep claiming such nodes — applying the gate to
    // it would wrongly decline them and break session creation with CPU fallback
    // disabled.
    // Claim-time *routing preference* gate: removed, permanently.
    //
    // An earlier design asked the EP, per node, whether it *wanted* a node it
    // could already run, and left the losing shape/dtype ranges to the host
    // runtime's own kernel. That is withdrawn as an architectural rule:
    // selecting this EP is a request for this EP, so a range where it is
    // slower than the host is a kernel to optimize, not a node to give away.
    // The `ClaimPreference` type, the `claim_preference`/`claim_preference_node`
    // trait methods and the `host_fallback_available` plumbing that switched
    // the gate off under `session.disable_cpu_ep_fallback=1` are all deleted,
    // so the deferral cannot be reintroduced by overriding a default — there is
    // nothing left to override.
    let claims = exported.ep.with(|ep| {
        ort_view.query_capabilities_filtered(ep, |node| {
            if is_gpu_ep && !node_inputs_all_routable(&view, node) {
                return false;
            }
            if is_gpu_ep {
                let ir_node = view.node(node);
                let input_shapes: Vec<Vec<Option<usize>>> = view
                    .node_inputs(node)
                    .iter()
                    .map(|input| {
                        input
                            .map(|value| {
                                view.value(value)
                                    .shape
                                    .iter()
                                    .map(|dimension| dimension.as_static())
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect();
                let strategy = crate::compute::ShapeInference::for_node(
                    ir_node,
                    &input_shapes,
                    ir_node.outputs.len(),
                );
                if shape_inference_reads_runtime_values(&strategy, ir_node) {
                    return false;
                }
            }
            true
        })
    });

    if claims.is_empty() {
        return ok_status();
    }

    // Fail-closed routing filter: drop any multi-node claim that
    // `build_subgraph_routing` would refuse to route, so we decline it here
    // (ORT then runs those nodes itself) instead of failing `Compile`, which
    // ORT surfaces as a hard session-creation error rather than a fallback.
    //
    // The one shape it cannot route is a value that is *both* an output of the
    // fused subgraph and consumed by a later node inside it: the producing slot
    // gets a `NodeOutputSink::Ort`, so no intermediate buffer is recorded, and
    // the internal consumer has nowhere to read it from. Found by
    // `com.microsoft::FastGelu` in float16, which ORT inlines as a function
    // body; once one node of that body is declined, `X_bias` becomes an output
    // of the remaining partition while three later `Mul`s still consume it.
    //
    // This is not a routing-preference artefact — any decline (dtype filter,
    // shape-inference filter, another EP) can produce the same partition. The
    // filter is therefore unconditional and applies to every EP.
    let claims: Vec<_> = claims
        .into_iter()
        .filter(|claim| !claim_has_unroutable_internal_output(ir_graph, &claim.node_ids))
        .collect();

    if claims.is_empty() {
        return ok_status();
    }

    // Fail-closed filter: remove any claim containing a node whose shape
    // inference returns `Declined`. This prevents over-claiming ops we cannot
    // correctly execute (e.g. NonZero with data-dependent output shape).
    let claims: Vec<_> = claims
        .into_iter()
        .filter(|claim| {
            claim.node_ids.iter().all(|&nid| {
                let node = ir_graph.nodes.get(nid);
                if node.is_none() {
                    return false;
                }
                let node = node.unwrap();
                let input_shapes: Vec<Vec<Option<usize>>> = node
                    .inputs
                    .iter()
                    .map(|input| {
                        input
                            .and_then(|vid| ir_graph.values.get(vid))
                            .map(|v| v.shape.iter().map(|d| d.as_static()).collect())
                            .unwrap_or_default()
                    })
                    .collect();
                let num_outputs = node.outputs.len();
                let si = crate::compute::ShapeInference::for_node(node, &input_shapes, num_outputs);
                !matches!(si, crate::compute::ShapeInference::Declined { .. })
            })
        })
        .collect();

    if claims.is_empty() {
        return ok_status();
    }

    // Fail-closed dtype filter: remove any claim containing a node whose
    // input/output element types are not in the registry's supported_dtypes
    // for that op. This ensures the claim predicate and the advertised type
    // constraints agree by construction — both are sourced from the same
    // `KernelRegistryEntry` data.
    //
    // Additionally, decline any node with an Undefined output dtype — we
    // cannot produce a tensor if we don't know its element type. This is
    // independent of the registry filter: even when no registry entries
    // exist, we refuse to claim nodes whose output types are unknown.
    // Absent optional outputs (tracked out-of-band via the reader's
    // `absent_outputs` set) are exempt — they represent known absence.
    let absent = reader.absent_outputs();
    let claims: Vec<_> = claims
        .into_iter()
        .filter(|claim| {
            claim.node_ids.iter().all(|&nid| {
                let Some(node) = ir_graph.nodes.get(nid) else {
                    return false;
                };
                // Fail-closed: every output must have a resolved, producible dtype
                // UNLESS it is an intentionally absent optional output slot.
                for &vid in &node.outputs {
                    let Some(value) = ir_graph.values.get(vid) else {
                        return false;
                    };
                    if value.dtype == DataType::Undefined {
                        // Absent optional outputs are a known absence, not an
                        // unresolved dtype — do not decline the node for these.
                        if !absent.contains(&vid) {
                            return false;
                        }
                    }
                }
                node_passes_dtype_filter(node, ir_graph, &exported.registry_entries, absent)
            })
        })
        .collect();

    if claims.is_empty() {
        return ok_status();
    }

    // Partial-GPU-claim gate (default OFF) — see #982.
    //
    // Executing an interspersed CPU/GPU partition (some claimed nodes on this
    // GPU EP, the rest on CPU, with tensors crossing the boundary) currently
    // deadlocks in a synchronous CUDA memcpy (#982). A hang is strictly worse
    // for a user than main's silent all-CPU fallback, so a non-CPU EP does not
    // form such a partition by default: unless a partial-claim env flag is set,
    // it keeps its claims only when they cover the ENTIRE graph — a pure
    // all-GPU graph (e.g. the smoke test) has no CPU boundary and runs fine.
    // Otherwise it declines everything and ORT runs the whole graph on CPU,
    // exactly like main. Setting the flag restores per-node claiming so #982 can
    // be reproduced and worked on. The gate is GPU-specific (`device_type !=
    // Cpu`) so the CPU plugin — whose partial claiming works — is unaffected.
    //
    // NOTE: on a real decoder the nodes this EP *can* claim are default-domain
    // elementwise ops (Add/Mul/Sigmoid) that consume only activations; the
    // `com.microsoft` weight-consuming ops are already declined at claim time by
    // the routability/GQA gates. So the hang is triggered by any interspersed
    // partition, not specifically by `com.microsoft` nodes — which is why the
    // gate is expressed as "whole-graph-or-nothing for GPU EPs" rather than
    // "don't advertise com.microsoft".
    if is_gpu_ep && !partial_gpu_claim_enabled() {
        let total_nodes = view.nodes().count();
        // Convex claims are disjoint, so summing lengths counts each claimed
        // node once.
        let claimed_nodes: usize = claims.iter().map(|c| c.node_ids.len()).sum();
        if claimed_nodes != total_nodes {
            return ok_status();
        }
    }

    // Report claims to ORT via EpGraphSupportInfo_AddNodesToFuse.
    let api = crate::status::host_api();
    if api.is_null() {
        return fail_status("GetCapability: host ORT API not available");
    }

    let ep_api = unsafe {
        match (*api).GetEpApi {
            Some(get_ep_api) => get_ep_api(),
            None => return fail_status("GetCapability: host has no OrtEpApi"),
        }
    };
    if ep_api.is_null() {
        return fail_status("GetCapability: OrtEpApi is null");
    }

    let add_nodes = unsafe {
        match (*ep_api).EpGraphSupportInfo_AddNodesToFuse {
            Some(f) => f,
            None => {
                return fail_status(
                    "GetCapability: OrtEpApi.EpGraphSupportInfo_AddNodesToFuse is null",
                );
            }
        }
    };

    // Map our NodeId claims back to ORT node pointers.
    for claim in &claims {
        let ort_node_ptrs: Vec<*const ort::OrtNode> = claim
            .node_ids
            .iter()
            .map(|id| reader.node_id_to_ort_ptr(*id))
            .collect();

        if ort_node_ptrs.is_empty() {
            continue;
        }

        // SAFETY: add_nodes is a function pointer from ORT's EpApi, the support
        // pointer is valid, and node pointers are from the same graph.
        let status = unsafe {
            add_nodes(
                support,
                ort_node_ptrs.as_ptr(),
                ort_node_ptrs.len(),
                ptr::null(), // fusion options (optional)
            )
        };

        if !status.is_null() {
            return status;
        }
    }

    ok_status()
}

/// Check whether a node's input/output element types are all supported by the
/// corresponding registry entry. Returns `true` if the node should be claimed.
///
/// Fail-closed: returns `false` if the op has no registry entry, or if any
/// value has `DataType::Undefined`.
pub(crate) fn node_passes_dtype_filter(
    node: &onnx_runtime_ir::Node,
    ir_graph: &onnx_runtime_ir::Graph,
    entries: &[KernelRegistryEntry],
    absent_outputs: &HashSet<ValueId>,
) -> bool {
    if entries.is_empty() {
        return true;
    }
    let domain = if node.domain.is_empty() {
        ""
    } else {
        node.domain.as_str()
    };
    let entry = entries
        .iter()
        .find(|e| e.op_type == node.op_type && e.domain == domain);
    let Some(entry) = entry else {
        return false;
    };
    for (slot, input) in node.inputs.iter().enumerate() {
        let Some(vid) = input else { continue };
        let Some(value) = ir_graph.values.get(*vid) else {
            continue;
        };
        if value.dtype == DataType::Undefined {
            return false;
        }
        // A per-slot constraint replaces the union for that position; without
        // it a mixed-dtype op is claimed for combinations its kernel rejects,
        // which turns a decline into a run-time failure.
        let allowed = entry
            .input_dtype_constraints
            .iter()
            .find(|(index, _)| *index == slot)
            .map_or(entry.supported_dtypes, |(_, dtypes)| *dtypes);
        if !allowed.contains(&value.dtype) {
            return false;
        }
    }
    for (slot, &vid) in node.outputs.iter().enumerate() {
        let Some(value) = ir_graph.values.get(vid) else {
            continue;
        };
        // Absent optional outputs are exempt from the dtype filter — the
        // kernel doesn't actually produce them, so their dtype doesn't need
        // to be in supported_dtypes.
        if absent_outputs.contains(&vid) {
            continue;
        }
        if value.dtype == DataType::Undefined {
            return false;
        }
        let allowed = entry
            .output_dtype_constraints
            .iter()
            .find(|(index, _)| *index == slot)
            .map_or(entry.supported_dtypes, |(_, dtypes)| *dtypes);
        if !allowed.contains(&value.dtype) {
            return false;
        }
    }
    true
}

fn shape_inference_reads_runtime_values(
    strategy: &crate::compute::ShapeInference,
    node: &onnx_runtime_ir::Node,
) -> bool {
    use crate::compute::ShapeInference;

    match strategy {
        ShapeInference::SharedNative { node, .. } if node.op_type == "DFT" => {
            node.inputs.get(1).is_some_and(Option::is_some)
                || node.inputs.get(2).is_some_and(Option::is_some)
        }
        // Every other shared rule in the current census consumes shape-data:
        // ConstantOfShape, Expand, STFT, and Tile.
        ShapeInference::ReductionFromInput { .. } | ShapeInference::SqueezeFromInput => {
            node.inputs.get(1).is_some_and(Option::is_some)
        }
        ShapeInference::SharedNative { .. }
        | ShapeInference::ReshapeData { .. }
        | ShapeInference::SliceData
        | ShapeInference::ConstantOfShape
        | ShapeInference::Expand
        | ShapeInference::Tile
        | ShapeInference::Window
        | ShapeInference::Compress { .. } => true,
        ShapeInference::Dft { .. } => {
            node.inputs.get(1).is_some_and(Option::is_some)
                || node.inputs.get(2).is_some_and(Option::is_some)
        }
        _ => false,
    }
}

/// Compile: for each claimed subgraph, create kernels and wrap them as
/// `OrtNodeComputeInfo` callbacks.
unsafe extern "C" fn ep_compile(
    ep: *mut ort::OrtEp,
    graphs: *mut *const ort::OrtGraph,
    _fused_nodes: *mut *const ort::OrtNode,
    count: usize,
    out_infos: *mut *mut ort::OrtNodeComputeInfo,
    _out_ep_context_nodes: *mut *mut ort::OrtNode,
) -> *mut ort::OrtStatus {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        ep_compile_inner(ep, graphs, count, out_infos)
    }));
    result.unwrap_or_else(|_| fail_status("Compile: internal panic"))
}

fn ep_compile_inner(
    ep: *mut ort::OrtEp,
    graphs: *mut *const ort::OrtGraph,
    count: usize,
    out_infos: *mut *mut ort::OrtNodeComputeInfo,
) -> *mut ort::OrtStatus {
    if ep.is_null() || out_infos.is_null() || count == 0 {
        return invalid_arg_status("Compile: null argument or zero count");
    }
    // Null-check graphs pointer (H2: would segfault otherwise).
    if graphs.is_null() {
        return invalid_arg_status("Compile: graphs pointer is null");
    }

    let exported = unsafe { &*(ep.cast::<ExportedEp>()) };

    // Capture the device staging context once (#982). `Some` only for a device
    // EP: it lets each subgraph's `Compute` upload host-resident boundary inputs
    // into device scratch on an interspersed CPU→device partition. A CPU EP
    // returns `None` and its subgraphs run with inputs verbatim, unchanged.
    let device_staging: Option<(
        Arc<dyn onnx_runtime_ep_api::HostToDeviceCopier>,
        String,
        ort::OrtMemoryInfoDeviceType,
        u32,
    )> = exported.ep.with(|ep| {
        ep.host_to_device_copier().map(|copier| {
            let mem_dev_type = match ep.device_type() {
                DeviceType::Cpu => ort::OrtMemoryInfoDeviceType_CPU,
                DeviceType::Qnn => ort::OrtMemoryInfoDeviceType_NPU,
                _ => ort::OrtMemoryInfoDeviceType_GPU,
            };
            (
                copier,
                ep.name().to_string(),
                mem_dev_type,
                ep.memory_vendor_id(),
            )
        })
    });

    for i in 0..count {
        let graph_ptr = unsafe { *graphs.add(i) };
        if graph_ptr.is_null() {
            // Write a null info for this slot and continue.
            unsafe { *out_infos.add(i) = ptr::null_mut() };
            continue;
        }

        let reader = match unsafe { OutboundGraphReader::from_ort_graph(graph_ptr) } {
            Ok(r) => r,
            Err(msg) => {
                // NEW-2 fix: free already-written out_infos[0..i] and null them
                // so that a subsequent ReleaseNodeComputeInfos (if ORT calls it)
                // is a safe no-op. This is safe under both "ORT frees on failure"
                // (all slots are null → no double-free) and "ORT does not free"
                // (we freed → no leak). Header lines 2179/2203–2207 do not
                // specify the failure-path contract.
                cleanup_partial_infos(out_infos, i);
                return fail_status(&format!("Compile: failed to read subgraph {i}: {msg}"));
            }
        };

        let ir_graph = reader.to_ir_graph();
        let cache = match onnx_runtime_ir::GraphViewCache::build(ir_graph) {
            Ok(c) => c,
            Err(e) => {
                cleanup_partial_infos(out_infos, i);
                return fail_status(&format!(
                    "Compile: failed to build graph cache for subgraph {i}: {e}"
                ));
            }
        };
        let view = onnx_runtime_ir::GraphView::new(ir_graph, &cache);

        // Collect kernels for all nodes in the subgraph.
        let mut entries: Vec<crate::compute::CompiledKernelEntry> = Vec::new();
        for node_idx in view.nodes() {
            let node = view.node(node_idx);
            // Preserve rank: map each dim to Option<usize> so symbolic dims
            // become None rather than being dropped (which would destroy rank).
            let shapes_opt: Vec<Vec<Option<usize>>> = view
                .node_inputs(node_idx)
                .iter()
                .map(|input| {
                    input
                        .map(|v| view.value(v).shape.iter().map(|d| d.as_static()).collect())
                        .unwrap_or_default()
                })
                .collect();

            // INVARIANT: The `get_kernel` trait takes `&[Vec<usize>]` — it cannot
            // express "unknown dimension" at the type level. We use the sentinel
            // value `DIM_UNKNOWN` (0) to represent symbolic/dynamic dims.
            //
            // This is safe because:
            //   (a) Valid static dims for non-empty tensors are always ≥ 1.
            //   (b) Kernels receive actual runtime shapes from OrtKernelContext
            //       and MUST NOT pre-allocate buffers based on compile-time shapes.
            //   (c) The `shapes_opt` vector (with full `Option<usize>` fidelity)
            //       is passed separately to `ShapeInference::for_node` below.
            //
            // If the trait is ever extended to accept optional dims, remove this.
            const DIM_UNKNOWN: usize = 0;
            let shapes: Vec<Vec<usize>> = shapes_opt
                .iter()
                .map(|s| s.iter().map(|d| d.unwrap_or(DIM_UNKNOWN)).collect())
                .collect();

            let opset = ir_graph.effective_opset(node).unwrap_or(0);

            match exported.ep.with(|ep| ep.get_kernel(node, &shapes, opset)) {
                Ok(mut kernel) => {
                    // Tell the kernel which of its inputs are session-lifetime
                    // constants. Without this every kernel sees all inputs as
                    // runtime tensors and re-does its one-time weight work on
                    // every `Run`: `MatMulNBits` repacks (or, worse, declines
                    // the MLAS SQNBit path, which is gated on the same flag)
                    // and `QLinearMatMul` re-packs B, turning a load-time cost
                    // into a per-token cost.
                    let constant_inputs =
                        constant_input_flags(&view, node_idx, reader.constant_initializer_names());
                    CONSTANT_WEIGHT_INPUTS.fetch_add(
                        constant_inputs.iter().filter(|c| **c).count(),
                        Ordering::Relaxed,
                    );
                    kernel.set_constant_inputs(&constant_inputs);
                    let num_inputs = view.node_inputs(node_idx).len();
                    let num_outputs = view.node_outputs(node_idx).len();

                    // Read per-output declared dtype from the ORT graph's
                    // value info — never inferred from inputs. This is the
                    // authoritative dtype for Cast, Where, Shape, etc.
                    // Absent optional outputs now carry their actual ORT-declared
                    // dtype (not Undefined) so scratch buffers are sized correctly.
                    let absent = reader.absent_outputs();
                    let node_output_vals = view.node_outputs(node_idx);
                    let mut output_dtypes: Vec<DataType> = node_output_vals
                        .iter()
                        .map(|&val_idx| view.value(val_idx).dtype)
                        .collect();
                    let absent_output_slots: std::collections::HashSet<usize> = node_output_vals
                        .iter()
                        .enumerate()
                        .filter(|&(_, &val_idx)| {
                            let vid = view.value_id(val_idx);
                            absent.contains(&vid)
                        })
                        .map(|(slot, _)| slot)
                        .collect();

                    // Resolve Undefined dtypes for absent output slots: if ORT
                    // didn't provide type info, propagate from the first present
                    // output of the same node (ONNX type constraints share types
                    // across outputs). This is NOT input-based inference — it uses
                    // present outputs of the same node.
                    if absent_output_slots
                        .iter()
                        .any(|&s| output_dtypes.get(s).copied() == Some(DataType::Undefined))
                    {
                        let present_dtype = output_dtypes
                            .iter()
                            .enumerate()
                            .find(|(i, dt)| {
                                !absent_output_slots.contains(i) && **dt != DataType::Undefined
                            })
                            .map(|(_, dt)| *dt);
                        if let Some(dt) = present_dtype {
                            for &slot in &absent_output_slots {
                                if output_dtypes[slot] == DataType::Undefined {
                                    output_dtypes[slot] = dt;
                                }
                            }
                        }
                        // If no present output has a known dtype, the fail-closed
                        // path in compute.rs will reject the scratch allocation.
                    }

                    // Determine shape inference strategy using full node
                    // attributes (wired to Deckard's 22 rules).
                    let shape_inference =
                        crate::compute::ShapeInference::for_node(node, &shapes_opt, num_outputs);
                    if matches!(
                        &shape_inference,
                        crate::compute::ShapeInference::KernelSizedOutputs
                    ) && !kernel.has_kernel_sized_outputs()
                    {
                        cleanup_partial_infos(out_infos, i);
                        return fail_status(&format!(
                            "Compile: node '{}' ({}) requires kernel-sized outputs, but its \
                             selected kernel did not opt into that contract",
                            node.name, node.op_type
                        ));
                    }

                    // Build input_slots: maps node input position → ORT index
                    // (None for absent inputs).
                    //
                    // Indices are assigned per *distinct value*, not per
                    // position. ORT's fused-node metadata carries a set of
                    // input names, so a node that names the same value twice
                    // is bound once and every later slot would otherwise be
                    // shifted — the last one past the end of the array ORT
                    // actually passes. That is not a corner case: a quantized
                    // `QLinearMatMul` routinely shares one zero-point or scale
                    // initializer between its `a`, `b` and `y` triples, and any
                    // `Mul(x, x)`-shaped node does the same.
                    let mut ort_input_idx = 0usize;
                    let mut value_to_ort_index: std::collections::HashMap<
                        onnx_runtime_ir::ValueIndex,
                        usize,
                    > = std::collections::HashMap::new();
                    let input_slots: Vec<Option<usize>> = view
                        .node_inputs(node_idx)
                        .iter()
                        .map(|input| {
                            input.map(|value| {
                                *value_to_ort_index.entry(value).or_insert_with(|| {
                                    let idx = ort_input_idx;
                                    ort_input_idx += 1;
                                    idx
                                })
                            })
                        })
                        .collect();

                    entries.push(crate::compute::CompiledKernelEntry {
                        kernel,
                        num_inputs,
                        num_outputs,
                        output_dtypes,
                        absent_output_slots,
                        shape_inference,
                        input_slots,
                    });
                    COMPILED_NODE_COUNT.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    cleanup_partial_infos(out_infos, i);
                    return fail_status(&format!(
                        "Compile: get_kernel failed for node '{}' ({}): {e}",
                        node.name.as_str(),
                        node.op_type
                    ));
                }
            }
        }

        if entries.is_empty() {
            unsafe { *out_infos.add(i) = ptr::null_mut() };
            continue;
        }

        // Wrap kernels in OrtNodeComputeInfo.
        let mut info = ExportedComputeInfo::new(entries);

        // Placement is a property of the EP, decided by the factory when it
        // registered the allocator; pass it down so `Compute` never has to ask
        // ORT where a routed subgraph's intermediates belong.
        info.set_host_accessible(exported.host_accessible);

        // Attach device staging so an interspersed CPU→device partition can
        // upload host-resident boundary inputs before launching (#982).
        if let Some((ref copier, ref alloc_name, mem_dev_type, vendor_id)) = device_staging {
            info.set_device_staging(Arc::clone(copier), alloc_name, mem_dev_type, vendor_id);
        }

        // For multi-node fused subgraphs, construct the SubgraphRouting so
        // intermediates are threaded correctly in topological order.
        // Fail at Compile (not Run) if the graph is unroutable — ORT can still
        // fall back cleanly at this stage, whereas a Run-time failure is much
        // harder for users to diagnose.
        if info.entries.len() > 1 {
            match build_subgraph_routing(&view, ir_graph, reader.absent_outputs()) {
                Some(routing) => info.set_routing(routing),
                None => {
                    cleanup_partial_infos(out_infos, i);
                    return fail_status(
                        "Compile: multi-node subgraph has unroutable graph \
                         (a node input is not reachable from graph inputs or \
                         prior node outputs). Declining subgraph so ORT can \
                         fall back.",
                    );
                }
            }
        }

        let info_ptr = Box::into_raw(Box::new(info));
        unsafe { *out_infos.add(i) = info_ptr.cast::<ort::OrtNodeComputeInfo>() };
    }

    ok_status()
}

/// Which inputs of `node` are session-lifetime constants.
///
/// A node input that ORT lists as a *constant* initializer is a weight: ORT
/// owns the buffer, materializes it once at session creation and cannot change
/// it between `Run` calls. An IR>=4 initializer that also appears as a graph
/// input is only a default value the caller may override per `Run`, and is
/// deliberately not in that set (see `read_constant_initializer_names`). That is the
/// contract
/// [`onnx_runtime_ep_api::kernel::Kernel::set_constant_inputs`] expresses, and
/// several kernels use it to decide whether a prepack may be built once and
/// kept in the kernel instance (which lives as long as the session) instead of
/// rebuilt on every call. `MatMulNBits` goes further and gates its MLAS SQNBit
/// path on the same flag, so leaving it false does not merely repeat work — it
/// selects a different, slower kernel.
///
/// The producer/`is_graph_input` test used elsewhere in this file cannot be
/// reused here. It is correct for the whole-model graph at capability time,
/// but ORT hands a *fused node's* subgraph over with the initializers it kept
/// inside listed as graph inputs of that subgraph, so at Compile time every
/// weight looks like an activation. `Graph_GetInitializers` still distinguishes
/// them, which is why the flags are keyed by name against that set — filtered
/// to the entries `ValueInfo_IsConstantInitializer` accepts, because
/// `Graph_GetInitializers` also lists IR>=4 defaults the caller may replace.
///
/// Absent optional inputs and unnamed values are never constant: there is
/// nothing to cache and nothing to key a cache on.
fn constant_input_flags(
    view: &onnx_runtime_ir::GraphView<'_>,
    node: NodeIndex,
    constant_initializer_names: &std::collections::HashSet<String>,
) -> Vec<bool> {
    view.node_inputs(node)
        .iter()
        .map(|input| {
            input.is_some_and(|value| {
                view.value(value)
                    .name
                    .as_deref()
                    .is_some_and(|name| constant_initializer_names.contains(name))
            })
        })
        .collect()
}

/// Whether every present input of `node` can be routed by the fused-subgraph
/// compile path (`build_subgraph_routing`).
///
/// That path threads a node input only when the value is a graph input or is
/// produced by another node in the graph. A value with no producer that is not
/// a graph input is a weight *initializer* (or other constant ORT keeps inside
/// the subgraph); the routing table has no ORT input index or intermediate
/// buffer for it and would decline the whole subgraph at Compile time. Gating
/// such nodes out at capability time keeps them out of every convex claim so
/// ORT partitions around them and the session still builds.
///
/// This intentionally mirrors `build_subgraph_routing`'s per-input decision so
/// the claim-time predicate and the compile-time router agree by construction.
fn node_inputs_all_routable(view: &onnx_runtime_ir::GraphView<'_>, node: NodeIndex) -> bool {
    view.node_inputs(node)
        .iter()
        .flatten()
        .all(|&input| view.producer(input).is_some() || view.value(input).is_graph_input)
}

/// Whether a multi-node claim contains a value that
/// [`build_subgraph_routing`] cannot route, namely one that is both an output
/// of the fused subgraph and consumed by another node inside it.
///
/// ORT surfaces such a value as a fused-node output, so the producing slot is
/// assigned a [`crate::compute::NodeOutputSink::Ort`] and no intermediate
/// buffer is recorded — leaving the internal consumer with nothing to read.
/// The router declines at Compile time, which ORT reports as a session-creation
/// failure instead of falling back, so we must catch it here while declining is
/// still free.
///
/// A value is an output of the subgraph when it is a graph output or is
/// consumed by a node outside the claim. Single-node claims are never routed
/// and are always accepted.
fn claim_has_unroutable_internal_output(
    graph: &onnx_runtime_ir::Graph,
    node_ids: &[onnx_runtime_ir::NodeId],
) -> bool {
    if node_ids.len() < 2 {
        return false;
    }
    let claimed: std::collections::HashSet<_> = node_ids.iter().copied().collect();

    for &nid in node_ids {
        let Some(node) = graph.nodes.get(nid) else {
            return true;
        };
        for &produced in &node.outputs {
            let Some(value) = graph.values.get(produced) else {
                return true;
            };
            let mut consumed_inside = false;
            let mut consumed_outside = value.is_graph_output;
            for consumer in value.consumers.nodes() {
                if claimed.contains(&consumer) {
                    consumed_inside |= consumer != nid;
                } else {
                    consumed_outside = true;
                }
            }
            if consumed_inside && consumed_outside {
                return true;
            }
        }
    }
    false
}

/// Environment variables that opt a GPU EP into partial (interspersed CPU/GPU)
/// claiming. Off by default because such partitions currently hang (#982).
///
/// The second name is the one proposed in the review thread; both are accepted
/// so either spelling enables the path.
const PARTIAL_GPU_CLAIM_ENV: [&str; 2] = [
    "ONNX_GENAI_PLUGIN_PARTIAL_GPU_CLAIM",
    "ONNX_GENAI_PLUGIN_CLAIM_MS_DOMAIN",
];

/// Whether partial (interspersed CPU/GPU) claiming is enabled for GPU EPs.
///
/// Off by default (see the partial-GPU-claim gate in
/// [`ep_get_capability_inner`]). Enabled when either environment variable in
/// [`PARTIAL_GPU_CLAIM_ENV`] is set to a truthy value.
fn partial_gpu_claim_enabled() -> bool {
    PARTIAL_GPU_CLAIM_ENV
        .iter()
        .any(|k| parse_bool_flag(std::env::var(k).ok().as_deref()))
}

/// Parse a boolean environment flag: truthy for `1`, `true`, `yes`, or `on`
/// (case-insensitive, surrounding whitespace ignored); falsy otherwise,
/// including unset, empty, `0`, and `false`.
fn parse_bool_flag(value: Option<&str>) -> bool {
    match value {
        Some(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => false,
    }
}

/// Build a `SubgraphRouting` table for a multi-node fused subgraph.
///
/// Determines which node inputs come from ORT kernel-context inputs (graph inputs)
/// vs. intermediate buffers, and which outputs go to ORT outputs vs. buffers.
fn build_subgraph_routing(
    view: &onnx_runtime_ir::GraphView<'_>,
    graph: &onnx_runtime_ir::Graph,
    absent_outputs: &std::collections::HashSet<onnx_runtime_ir::ValueId>,
) -> Option<crate::compute::SubgraphRouting> {
    use crate::compute::{NodeInputSource, NodeOutputSink};
    use std::collections::HashMap;

    // Build maps: ValueId → ORT input/output index.
    let input_index: HashMap<onnx_runtime_ir::ValueId, usize> = graph
        .inputs
        .iter()
        .enumerate()
        .map(|(i, &vid)| (vid, i))
        .collect();

    let output_index: HashMap<onnx_runtime_ir::ValueId, usize> = graph
        .outputs
        .iter()
        .enumerate()
        .map(|(i, &vid)| (vid, i))
        .collect();

    // Map ValueId → buffer index for intermediate values.
    let mut value_to_buffer: HashMap<onnx_runtime_ir::ValueId, usize> = HashMap::new();
    let mut next_buffer = 0usize;

    let nodes: Vec<_> = view.nodes().collect();

    let mut input_sources: Vec<Vec<NodeInputSource>> = Vec::with_capacity(nodes.len());
    let mut output_sinks: Vec<Vec<NodeOutputSink>> = Vec::with_capacity(nodes.len());

    for &node_idx in &nodes {
        // Build input sources.
        let node_inputs = view.node_inputs(node_idx);
        let mut sources = Vec::with_capacity(node_inputs.len());
        for input_slot in node_inputs {
            match input_slot {
                Some(val_idx) => {
                    let vid = view.value_id(*val_idx);
                    if let Some(&ort_idx) = input_index.get(&vid) {
                        sources.push(NodeInputSource::Ort(ort_idx));
                    } else if let Some(&buf_idx) = value_to_buffer.get(&vid) {
                        sources.push(NodeInputSource::Buffer(buf_idx));
                    } else {
                        // Value not from graph input or prior node output — decline.
                        return None;
                    }
                }
                None => {
                    sources.push(NodeInputSource::Absent);
                }
            }
        }
        input_sources.push(sources);

        // Build output sinks.
        let node_outputs = view.node_outputs(node_idx);
        let mut sinks = Vec::with_capacity(node_outputs.len());
        for &val_idx in node_outputs {
            let vid = view.value_id(val_idx);
            if absent_outputs.contains(&vid) {
                // Absent optional output — no buffer needed; scratch-allocated
                // at compute time via absent_output_slots.
                sinks.push(NodeOutputSink::Absent);
            } else if let Some(&ort_idx) = output_index.get(&vid) {
                sinks.push(NodeOutputSink::Ort(ort_idx));
            } else {
                // Intermediate — assign a buffer.
                let buf_idx = next_buffer;
                next_buffer += 1;
                value_to_buffer.insert(vid, buf_idx);
                sinks.push(NodeOutputSink::Buffer(buf_idx));
            }
        }
        output_sinks.push(sinks);
    }

    Some(crate::compute::SubgraphRouting {
        input_sources,
        output_sinks,
        num_intermediate_buffers: next_buffer,
    })
}

/// Release compiled kernel infos.
unsafe extern "C" fn ep_release_node_compute_infos(
    _ep: *mut ort::OrtEp,
    infos: *mut *mut ort::OrtNodeComputeInfo,
    count: usize,
) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if infos.is_null() {
            return;
        }
        for i in 0..count {
            let ptr = unsafe { *infos.add(i) };
            if !ptr.is_null() {
                unsafe { drop(Box::from_raw(ptr.cast::<ExportedComputeInfo>())) };
            }
        }
    }));
}

/// Free already-written `out_infos[0..written]` and null them out on compile
/// failure. This is safe under both possible ORT behaviors on a failed Compile:
///
/// 1. ORT calls `ReleaseNodeComputeInfos` on the partial array → all slots are
///    null → no double-free (our release callback skips nulls).
/// 2. ORT does NOT call `ReleaseNodeComputeInfos` → no leak because we freed.
///
/// Evidence: ORT header lines 2179 ("ORT calls ReleaseNodeComputeInfos() to
/// release multiple instances in a batch") and 2203–2207 do NOT specify whether
/// this applies on Compile failure. This cleanup-and-null strategy is safe under
/// both interpretations.
fn cleanup_partial_infos(out_infos: *mut *mut ort::OrtNodeComputeInfo, written: usize) {
    for j in 0..written {
        let ptr = unsafe { *out_infos.add(j) };
        if !ptr.is_null() {
            unsafe { drop(Box::from_raw(ptr.cast::<ExportedComputeInfo>())) };
            unsafe { *out_infos.add(j) = ptr::null_mut() };
        }
    }
}

// ─── GetKernelRegistry ──────────────────────────────────────────────────────

/// `GetKernelRegistry` callback: returns the EP's pre-built kernel registry.
///
/// ORT uses this for type-constraint metadata so f16/bf16 nodes are correctly
/// routed to our EP during `GetCapability`. The kernel registry coexists with
/// the compile-based path: ORT header line 1522 documents
/// `EpGraphSupportInfo_LookUpKernel` as "Used within OrtEp::GetCapability()".
unsafe extern "C" fn ep_get_kernel_registry(
    ep: *mut ort::OrtEp,
    out_registry: *mut *const ort::OrtKernelRegistry,
) -> *mut ort::OrtStatus {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ep.is_null() || out_registry.is_null() {
            return invalid_arg_status("GetKernelRegistry: null argument");
        }
        let exported = unsafe { &*(ep.cast::<ExportedEp>()) };
        match &exported.kernel_registry {
            Some(holder) => {
                unsafe { *out_registry = holder.ptr.cast_const() };
            }
            None => {
                unsafe { *out_registry = ptr::null() };
            }
        }
        ok_status()
    }));
    result.unwrap_or_else(|_| fail_status("GetKernelRegistry: internal panic"))
}

/// Build an ORT `OrtKernelRegistry` from a slice of [`KernelRegistryEntry`].
///
/// Outcome of building the ORT kernel registry, including any entry-level failures.
#[derive(Debug)]
pub struct RegistryBuildOutcome {
    pub registry: Option<OrtKernelRegistryHolder>,
    /// Diagnostic messages for entries that failed to register.
    pub failures: Vec<String>,
}

/// Requires the ORT host API to be set (called after `set_host_api`).
/// Returns `None` registry if the entries slice is empty or ORT API is unavailable.
/// Any per-entry failures are reported in `failures` rather than silently swallowed.
///
/// The returned registry is valid for the EP's lifetime. ORT never frees it;
/// we free it in [`OrtKernelRegistryHolder::drop`].
pub fn build_ort_kernel_registry(
    entries: &[KernelRegistryEntry],
    ep_name: &str,
) -> RegistryBuildOutcome {
    if entries.is_empty() {
        return RegistryBuildOutcome {
            registry: None,
            failures: vec![],
        };
    }
    let api = crate::status::host_api();
    if api.is_null() {
        return RegistryBuildOutcome {
            registry: None,
            failures: vec!["host API not set".into()],
        };
    }
    let ep_api = unsafe {
        let get_ep_api = match (*api).GetEpApi {
            Some(f) => f,
            None => {
                return RegistryBuildOutcome {
                    registry: None,
                    failures: vec!["GetEpApi unavailable".into()],
                };
            }
        };
        get_ep_api()
    };
    if ep_api.is_null() {
        return RegistryBuildOutcome {
            registry: None,
            failures: vec!["EP API is null".into()],
        };
    }

    // Create the kernel registry.
    let create_registry = match unsafe { (*ep_api).CreateKernelRegistry } {
        Some(f) => f,
        None => {
            return RegistryBuildOutcome {
                registry: None,
                failures: vec!["CreateKernelRegistry unavailable".into()],
            };
        }
    };
    let mut registry_ptr: *mut ort::OrtKernelRegistry = ptr::null_mut();
    let status = unsafe { create_registry(&mut registry_ptr) };
    if !status.is_null() || registry_ptr.is_null() {
        return RegistryBuildOutcome {
            registry: None,
            failures: vec!["CreateKernelRegistry call failed".into()],
        };
    }

    let create_builder = match unsafe { (*ep_api).CreateKernelDefBuilder } {
        Some(f) => f,
        None => {
            return RegistryBuildOutcome {
                registry: None,
                failures: vec!["CreateKernelDefBuilder unavailable".into()],
            };
        }
    };
    let set_op_type = match unsafe { (*ep_api).KernelDefBuilder_SetOperatorType } {
        Some(f) => f,
        None => {
            return RegistryBuildOutcome {
                registry: None,
                failures: vec!["SetOperatorType unavailable".into()],
            };
        }
    };
    let set_domain = match unsafe { (*ep_api).KernelDefBuilder_SetDomain } {
        Some(f) => f,
        None => {
            return RegistryBuildOutcome {
                registry: None,
                failures: vec!["SetDomain unavailable".into()],
            };
        }
    };
    let set_since_version = match unsafe { (*ep_api).KernelDefBuilder_SetSinceVersion } {
        Some(f) => f,
        None => {
            return RegistryBuildOutcome {
                registry: None,
                failures: vec!["SetSinceVersion unavailable".into()],
            };
        }
    };
    let set_ep = match unsafe { (*ep_api).KernelDefBuilder_SetExecutionProvider } {
        Some(f) => f,
        None => {
            return RegistryBuildOutcome {
                registry: None,
                failures: vec!["SetExecutionProvider unavailable".into()],
            };
        }
    };
    let add_type_constraint = match unsafe { (*ep_api).KernelDefBuilder_AddTypeConstraint } {
        Some(f) => f,
        None => {
            return RegistryBuildOutcome {
                registry: None,
                failures: vec!["AddTypeConstraint unavailable".into()],
            };
        }
    };
    let build_def = match unsafe { (*ep_api).KernelDefBuilder_Build } {
        Some(f) => f,
        None => {
            return RegistryBuildOutcome {
                registry: None,
                failures: vec!["Build unavailable".into()],
            };
        }
    };
    let release_builder = match unsafe { (*ep_api).ReleaseKernelDefBuilder } {
        Some(f) => f,
        None => {
            return RegistryBuildOutcome {
                registry: None,
                failures: vec!["ReleaseKernelDefBuilder unavailable".into()],
            };
        }
    };
    let add_kernel = match unsafe { (*ep_api).KernelRegistry_AddKernel } {
        Some(f) => f,
        None => {
            return RegistryBuildOutcome {
                registry: None,
                failures: vec!["AddKernel unavailable".into()],
            };
        }
    };
    let release_def = match unsafe { (*ep_api).ReleaseKernelDef } {
        Some(f) => f,
        None => {
            return RegistryBuildOutcome {
                registry: None,
                failures: vec!["ReleaseKernelDef unavailable".into()],
            };
        }
    };
    let get_tensor_data_type = match unsafe { (*ep_api).GetTensorDataType } {
        Some(f) => f,
        None => {
            return RegistryBuildOutcome {
                registry: None,
                failures: vec!["GetTensorDataType unavailable".into()],
            };
        }
    };

    let ep_name_c = match std::ffi::CString::new(ep_name) {
        Ok(c) => c,
        Err(_) => {
            return RegistryBuildOutcome {
                registry: None,
                failures: vec!["invalid ep_name".into()],
            };
        }
    };

    let mut failures: Vec<String> = Vec::new();

    for entry in entries {
        // Validate version range.
        if entry.end_version < entry.since_version || entry.since_version <= 0 {
            failures.push(format!(
                "{}/{}: invalid version range since={} end={}",
                entry.domain, entry.op_type, entry.since_version, entry.end_version
            ));
            continue;
        }

        let op_c = match std::ffi::CString::new(entry.op_type) {
            Ok(c) => c,
            Err(_) => {
                failures.push(format!("{}: invalid op_type", entry.op_type));
                continue;
            }
        };
        let domain_c = match std::ffi::CString::new(entry.domain) {
            Ok(c) => c,
            Err(_) => {
                failures.push(format!("{}: invalid domain", entry.op_type));
                continue;
            }
        };

        let mut builder: *mut ort::OrtKernelDefBuilder = ptr::null_mut();
        let s = unsafe { create_builder(&mut builder) };
        if !s.is_null() || builder.is_null() {
            failures.push(format!(
                "{}/{}: CreateKernelDefBuilder failed",
                entry.domain, entry.op_type
            ));
            continue;
        }

        let s = unsafe { set_op_type(builder, op_c.as_ptr()) };
        if !s.is_null() {
            unsafe { release_builder(builder) };
            failures.push(format!(
                "{}/{}: SetOperatorType failed",
                entry.domain, entry.op_type
            ));
            continue;
        }
        let s = unsafe { set_domain(builder, domain_c.as_ptr()) };
        if !s.is_null() {
            unsafe { release_builder(builder) };
            failures.push(format!(
                "{}/{}: SetDomain failed",
                entry.domain, entry.op_type
            ));
            continue;
        }
        let s = unsafe { set_since_version(builder, entry.since_version, entry.end_version) };
        if !s.is_null() {
            unsafe { release_builder(builder) };
            failures.push(format!(
                "{}/{}: SetSinceVersion({}, {}) failed",
                entry.domain, entry.op_type, entry.since_version, entry.end_version
            ));
            continue;
        }
        let s = unsafe { set_ep(builder, ep_name_c.as_ptr()) };
        if !s.is_null() {
            unsafe { release_builder(builder) };
            failures.push(format!(
                "{}/{}: SetExecutionProvider failed",
                entry.domain, entry.op_type
            ));
            continue;
        }

        // Build OrtDataType* array for the type constraint "T".
        let mut ort_dtypes: Vec<*const ort::OrtDataType> = Vec::new();
        for &dtype in entry.supported_dtypes {
            let onnx_elem = dtype_to_onnx_tensor_elem(dtype);
            let mut dt_ptr: *const ort::OrtDataType = ptr::null();
            let s = unsafe { get_tensor_data_type(onnx_elem, &mut dt_ptr) };
            if s.is_null() && !dt_ptr.is_null() {
                ort_dtypes.push(dt_ptr);
            }
        }

        if !ort_dtypes.is_empty() {
            let constraint_name = c"T";
            let s = unsafe {
                add_type_constraint(
                    builder,
                    constraint_name.as_ptr(),
                    ort_dtypes.as_ptr(),
                    ort_dtypes.len(),
                )
            };
            if !s.is_null() {
                unsafe { release_builder(builder) };
                failures.push(format!(
                    "{}/{}: AddTypeConstraint failed",
                    entry.domain, entry.op_type
                ));
                continue;
            }
        }

        let mut kernel_def: *mut ort::OrtKernelDef = ptr::null_mut();
        let s = unsafe { build_def(builder, &mut kernel_def) };
        unsafe { release_builder(builder) };
        if !s.is_null() || kernel_def.is_null() {
            failures.push(format!(
                "{}/{}: KernelDefBuilder_Build failed",
                entry.domain, entry.op_type
            ));
            continue;
        }

        // Register with a no-op kernel create function. For compile-based EPs,
        // ORT should never call it (nodes go through Compile). If it IS called,
        // returning null kernel signals unsupported, which is safe.
        let s = unsafe {
            add_kernel(
                registry_ptr,
                kernel_def,
                Some(noop_kernel_create),
                ptr::null_mut(),
            )
        };
        unsafe { release_def(kernel_def) };
        if !s.is_null() {
            failures.push(format!(
                "{}/{}: AddKernel failed",
                entry.domain, entry.op_type
            ));
            continue;
        }
    }

    RegistryBuildOutcome {
        registry: Some(OrtKernelRegistryHolder { ptr: registry_ptr }),
        failures,
    }
}

/// No-op kernel create function. For compile-based EPs using a kernel registry
/// purely for type-constraint advertisement, ORT should never invoke this.
/// If it does (unexpected), return null kernel → ORT falls back.
unsafe extern "C" fn noop_kernel_create(
    _state: *mut std::ffi::c_void,
    _info: *const ort::OrtKernelInfo,
    kernel_out: *mut *mut ort::OrtKernelImpl,
) -> *mut ort::OrtStatus {
    if !kernel_out.is_null() {
        unsafe { *kernel_out = ptr::null_mut() };
    }
    fail_status("kernel_create called on compile-based EP — unexpected; returning null kernel")
}

/// Map `DataType` to `ONNXTensorElementDataType` enum value.
fn dtype_to_onnx_tensor_elem(dtype: DataType) -> ort::ONNXTensorElementDataType {
    match dtype {
        DataType::Float32 => 1,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        DataType::Uint8 => 2,     // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
        DataType::Int8 => 3,      // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
        DataType::Uint16 => 4,    // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
        DataType::Int16 => 5,     // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
        DataType::Int32 => 6,     // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        DataType::Int64 => 7,     // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        DataType::Bool => 9,      // ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
        DataType::Float16 => 10,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
        DataType::Float64 => 11,  // ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
        DataType::Uint32 => 12,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
        DataType::Uint64 => 13,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
        DataType::BFloat16 => 16, // ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
        _ => 0,                   // ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    /// ep_get_capability_inner returns a (null) status — not a crash — when
    /// called with null ep pointer.
    #[test]
    fn get_capability_null_ep_returns_status() {
        // Reset host API so invalid_arg_status returns null safely.
        unsafe { crate::status::set_host_api(ptr::null()) };
        let status = ep_get_capability_inner(ptr::null_mut(), ptr::null(), ptr::null_mut());
        // With no ORT API loaded, invalid_arg_status returns null.
        // The important invariant: no panic, no segfault.
        let _ = status;
    }

    /// ep_compile_inner returns a status (not a crash) when graphs is null.
    #[test]
    fn compile_null_graphs_returns_status() {
        unsafe { crate::status::set_host_api(ptr::null()) };
        // Pass a non-null sentinel for ep and out_infos so we reach the
        // graphs null check, which should return before dereferencing either.
        let mut dummy_out: *mut ort::OrtNodeComputeInfo = ptr::null_mut();
        let status = ep_compile_inner(
            std::ptr::dangling_mut::<ort::OrtEp>(), // non-null sentinel; never dereferenced
            ptr::null_mut(),                        // null graphs → returns invalid_arg_status
            1,
            &raw mut dummy_out,
        );
        // With no ORT API, invalid_arg_status returns null — no segfault.
        let _ = status;
    }

    /// Panic inside an extern "C" callback wrapper is caught and does not
    /// unwind past the catch_unwind boundary.
    #[test]
    fn catch_unwind_in_callback_wrapper_works() {
        unsafe { crate::status::set_host_api(ptr::null()) };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || -> *mut ort::OrtStatus { panic!("simulated panic in ep callback") },
            ));
            r.unwrap_or_else(|_| crate::status::fail_status("internal panic"))
        }));
        assert!(result.is_ok(), "panic must be contained by catch_unwind");
    }

    /// ShapeInference::for_node returns Declined for NonZero (data-dependent
    /// output shape), ensuring the capability filter rejects it.
    #[test]
    fn shape_inference_declines_nonzero() {
        use onnx_runtime_ir::{Node, NodeId};
        let node = Node::new(NodeId(0), "NonZero", vec![None], vec![]);
        let si = crate::compute::ShapeInference::for_node(&node, &[vec![Some(2), Some(3)]], 1);
        assert!(
            matches!(si, crate::compute::ShapeInference::Declined { .. }),
            "NonZero must be Declined; got {si:?}"
        );
    }

    #[test]
    fn device_claims_decline_every_host_value_reading_shape_rule() {
        use onnx_runtime_ir::{Node, NodeId, ValueId};

        for (op, inputs) in [
            ("ConstantOfShape", 1usize),
            ("Expand", 2),
            ("STFT", 4),
            ("Tile", 2),
            ("Compress", 2),
            ("HannWindow", 1),
            ("Reshape", 2),
            ("Slice", 3),
        ] {
            let node = Node::new(
                NodeId(0),
                op,
                (0..inputs)
                    .map(|index| Some(ValueId(index as u32)))
                    .collect(),
                vec![ValueId(100)],
            );
            let shapes = vec![vec![Some(1)]; inputs];
            let strategy = crate::compute::ShapeInference::for_node(&node, &shapes, 1);
            assert!(
                shape_inference_reads_runtime_values(&strategy, &node),
                "{op} must not be claimed by a device plugin while its shape rule reads values"
            );
        }

        let mut dft = Node::new(NodeId(0), "DFT", vec![Some(ValueId(0))], vec![ValueId(100)]);
        dft.version = Some(20);
        let strategy =
            crate::compute::ShapeInference::for_node(&dft, &[vec![Some(1), Some(8), Some(1)]], 1);
        assert!(!shape_inference_reads_runtime_values(&strategy, &dft));
        dft.inputs.push(Some(ValueId(1)));
        let strategy = crate::compute::ShapeInference::for_node(
            &dft,
            &[vec![Some(1), Some(8), Some(1)], vec![]],
            1,
        );
        assert!(shape_inference_reads_runtime_values(&strategy, &dft));

        let mut squeeze = Node::new(
            NodeId(0),
            "Squeeze",
            vec![Some(ValueId(0)), Some(ValueId(1))],
            vec![ValueId(100)],
        );
        squeeze.version = Some(13);
        let strategy = crate::compute::ShapeInference::for_node(
            &squeeze,
            &[vec![Some(1), Some(1), Some(3)], vec![Some(1)]],
            1,
        );
        assert!(matches!(
            strategy,
            crate::compute::ShapeInference::SqueezeFromInput
        ));
        assert!(shape_inference_reads_runtime_values(&strategy, &squeeze));
        squeeze.inputs[1] = None;
        let strategy = crate::compute::ShapeInference::for_node(
            &squeeze,
            &[vec![Some(1), Some(1), Some(3)], vec![]],
            1,
        );
        assert!(matches!(
            strategy,
            crate::compute::ShapeInference::SqueezeAllUnitDims
        ));
        assert!(!shape_inference_reads_runtime_values(&strategy, &squeeze));

        let mut reduction = Node::new(
            NodeId(0),
            "ReduceSum",
            vec![Some(ValueId(0)), Some(ValueId(1))],
            vec![ValueId(100)],
        );
        reduction.version = Some(13);
        let strategy = crate::compute::ShapeInference::for_node(
            &reduction,
            &[vec![Some(2), Some(3), Some(4)], vec![Some(1)]],
            1,
        );
        assert!(matches!(
            strategy,
            crate::compute::ShapeInference::ReductionFromInput { .. }
        ));
        assert!(shape_inference_reads_runtime_values(&strategy, &reduction));
        reduction.inputs[1] = None;
        let strategy = crate::compute::ShapeInference::for_node(
            &reduction,
            &[vec![Some(2), Some(3), Some(4)], vec![]],
            1,
        );
        assert!(matches!(
            strategy,
            crate::compute::ShapeInference::ReductionFromInput { .. }
        ));
        assert!(!shape_inference_reads_runtime_values(&strategy, &reduction));
    }

    /// ShapeInference::for_node accepts Add (elementwise broadcast) without
    /// attributes.
    #[test]
    fn shape_inference_accepts_add() {
        use onnx_runtime_ir::{Node, NodeId};
        let node = Node::new(NodeId(0), "Add", vec![None, None], vec![]);
        let si = crate::compute::ShapeInference::for_node(
            &node,
            &[vec![Some(2), Some(3)], vec![Some(2), Some(3)]],
            1,
        );
        assert!(
            matches!(si, crate::compute::ShapeInference::ElementwiseBroadcast),
            "Add must be ElementwiseBroadcast; got {si:?}"
        );
    }

    /// ShapeInference::for_node reads the axis attribute for Concat.
    #[test]
    fn shape_inference_reads_concat_axis_attribute() {
        use onnx_runtime_ir::{Attribute, Node, NodeId};
        let mut node = Node::new(NodeId(0), "Concat", vec![None, None], vec![]);
        node.attributes
            .insert("axis".to_string(), Attribute::Int(1));
        let si = crate::compute::ShapeInference::for_node(
            &node,
            &[vec![Some(2), Some(3)], vec![Some(2), Some(5)]],
            1,
        );
        match si {
            crate::compute::ShapeInference::Concat { axis } => {
                assert_eq!(axis, 1, "axis attribute must be read as 1");
            }
            other => panic!("Expected Concat; got {other:?}"),
        }
    }

    /// ShapeInference::for_node handles opset-13 Unsqueeze when axes are
    /// injected from initializer data.
    #[test]
    fn shape_inference_unsqueeze_with_injected_axes() {
        use onnx_runtime_ir::{Attribute, Node, NodeId};
        let mut node = Node::new(NodeId(0), "Unsqueeze", vec![None, None], vec![]);
        node.version = Some(13);
        // Simulate axes injected from initializer.
        node.attributes
            .insert("axes".to_string(), Attribute::Ints(vec![0, 2]));
        let si = crate::compute::ShapeInference::for_node(&node, &[vec![Some(3), Some(4)]], 1);
        match si {
            crate::compute::ShapeInference::Unsqueeze { axes } => {
                assert_eq!(axes, vec![0, 2]);
            }
            other => panic!("Expected Unsqueeze; got {other:?}"),
        }
    }

    /// cleanup_partial_infos frees written entries and nulls them out.
    #[test]
    fn cleanup_partial_infos_nulls_freed_slots() {
        unsafe { crate::status::set_host_api(ptr::null()) };
        // Allocate an array of 3 pointers, writing non-null "sentinels" for 2.
        let mut infos: [*mut ort::OrtNodeComputeInfo; 3] = [ptr::null_mut(); 3];
        // Create real ExportedComputeInfo that cleanup_partial_infos will drop.
        let info0 = Box::into_raw(Box::new(ExportedComputeInfo::new(Vec::new())));
        let info1 = Box::into_raw(Box::new(ExportedComputeInfo::new(Vec::new())));
        infos[0] = info0.cast();
        infos[1] = info1.cast();
        // Simulate failure at index 2 — cleanup [0..2].
        cleanup_partial_infos(infos.as_mut_ptr(), 2);
        assert!(infos[0].is_null(), "slot 0 must be nulled after cleanup");
        assert!(infos[1].is_null(), "slot 1 must be nulled after cleanup");
    }

    /// dtype_to_onnx_tensor_elem maps all CPU_EP_SUPPORTED_DTYPES correctly.
    #[test]
    fn dtype_mapping_matches_ort_constants() {
        use crate::kernel_ctx::CPU_EP_SUPPORTED_DTYPES;
        for &dtype in CPU_EP_SUPPORTED_DTYPES {
            let elem = dtype_to_onnx_tensor_elem(dtype);
            assert_ne!(elem, 0, "dtype {dtype:?} mapped to UNDEFINED");
        }
        // Spot-check specific values.
        assert_eq!(dtype_to_onnx_tensor_elem(DataType::Float16), 10);
        assert_eq!(dtype_to_onnx_tensor_elem(DataType::BFloat16), 16);
        assert_eq!(dtype_to_onnx_tensor_elem(DataType::Float32), 1);
    }

    /// KernelRegistryEntry can be constructed with static data.
    #[test]
    fn kernel_registry_entry_construction() {
        use crate::kernel_ctx::CPU_EP_SUPPORTED_DTYPES;
        let entry = KernelRegistryEntry {
            op_type: "Add",
            domain: "",
            since_version: 7,
            end_version: 21,
            supported_dtypes: CPU_EP_SUPPORTED_DTYPES,
            input_dtype_constraints: &[],
            output_dtype_constraints: &[],
        };
        assert_eq!(entry.op_type, "Add");
        assert!(entry.supported_dtypes.contains(&DataType::Float16));
        assert!(entry.supported_dtypes.contains(&DataType::BFloat16));
    }

    /// build_ort_kernel_registry returns None when host API is not set.
    #[test]
    fn build_registry_without_host_api_returns_none() {
        unsafe { crate::status::set_host_api(ptr::null()) };
        use crate::kernel_ctx::CPU_EP_SUPPORTED_DTYPES;
        let entries = vec![KernelRegistryEntry {
            op_type: "Add",
            domain: "",
            since_version: 7,
            end_version: 21,
            supported_dtypes: CPU_EP_SUPPORTED_DTYPES,
            input_dtype_constraints: &[],
            output_dtype_constraints: &[],
        }];
        let result = build_ort_kernel_registry(&entries, "test_ep");
        assert!(
            result.registry.is_none(),
            "must return None registry without host API"
        );
    }

    // ─── dtype filter tests ─────────────────────────────────────────────────

    /// Helper to build a minimal Graph with a single node and typed values.
    fn graph_with_node(
        op_type: &str,
        domain: &str,
        input_dtypes: &[DataType],
        output_dtypes: &[DataType],
    ) -> (onnx_runtime_ir::Graph, onnx_runtime_ir::NodeId) {
        use onnx_runtime_ir::{Graph, Node, NodeId, Shape};
        let mut g = Graph::new();
        let inputs: Vec<Option<onnx_runtime_ir::ValueId>> = input_dtypes
            .iter()
            .map(|&dt| {
                let vid = g.create_named_value(format!("in_{dt:?}"), dt, Shape::default());
                Some(vid)
            })
            .collect();
        let outputs: Vec<onnx_runtime_ir::ValueId> = output_dtypes
            .iter()
            .map(|&dt| g.create_named_value(format!("out_{dt:?}"), dt, Shape::default()))
            .collect();
        let mut node = Node::new(NodeId(0), op_type, inputs, outputs);
        node.domain = domain.to_string();
        let nid = g.insert_node(node);
        (g, nid)
    }

    /// f32 node with matching registry entry is claimed.
    #[test]
    fn dtype_filter_claims_f32_node() {
        let entries = vec![KernelRegistryEntry {
            op_type: "Add",
            domain: "",
            since_version: 7,
            end_version: 21,
            supported_dtypes: &[DataType::Float32, DataType::Float16],
            input_dtype_constraints: &[],
            output_dtype_constraints: &[],
        }];
        let (g, nid) = graph_with_node(
            "Add",
            "",
            &[DataType::Float32, DataType::Float32],
            &[DataType::Float32],
        );
        let node = g.nodes.get(nid).unwrap();
        assert!(super::node_passes_dtype_filter(
            node,
            &g,
            &entries,
            &std::collections::HashSet::new()
        ));
    }

    /// Node with unsupported dtype (Int64 for Add that only supports f32/f16)
    /// is NOT claimed.
    #[test]
    fn dtype_filter_rejects_unsupported_dtype() {
        let entries = vec![KernelRegistryEntry {
            op_type: "Add",
            domain: "",
            since_version: 7,
            end_version: 21,
            supported_dtypes: &[DataType::Float32, DataType::Float16],
            input_dtype_constraints: &[],
            output_dtype_constraints: &[],
        }];
        let (g, nid) = graph_with_node(
            "Add",
            "",
            &[DataType::Int64, DataType::Int64],
            &[DataType::Int64],
        );
        let node = g.nodes.get(nid).unwrap();
        assert!(!super::node_passes_dtype_filter(
            node,
            &g,
            &entries,
            &std::collections::HashSet::new()
        ));
    }

    /// The union of an op's edge dtypes is not its kernel's rule.
    ///
    /// `com.microsoft::MatMulNBits` spec-allows `float16` `zero_points`, and
    /// ORT's own kernel builds a session for one, but this EP's kernel accepts
    /// only `uint8` there. Membership in the union would claim the node and
    /// then fail inside `Compute`, where the only outcome is a run-time error
    /// on a model that worked before. The per-slot list must decline it.
    #[test]
    fn dtype_filter_applies_per_slot_constraints() {
        const FLOATS: &[DataType] = &[DataType::Float32, DataType::Float16, DataType::BFloat16];
        let entries = vec![KernelRegistryEntry {
            op_type: "MatMulNBits",
            domain: "com.microsoft",
            since_version: 1,
            end_version: i32::MAX,
            supported_dtypes: &[
                DataType::Float32,
                DataType::Float16,
                DataType::BFloat16,
                DataType::Uint8,
                DataType::Int32,
            ],
            input_dtype_constraints: &[
                (0, FLOATS),
                (1, &[DataType::Uint8]),
                (2, FLOATS),
                (3, &[DataType::Uint8]),
            ],
            output_dtype_constraints: &[],
        }];
        let empty = std::collections::HashSet::new();

        // A, B, scales, zero_points — all as the kernel requires.
        let (good, good_id) = graph_with_node(
            "MatMulNBits",
            "com.microsoft",
            &[
                DataType::Float32,
                DataType::Uint8,
                DataType::Float32,
                DataType::Uint8,
            ],
            &[DataType::Float32],
        );
        assert!(super::node_passes_dtype_filter(
            good.nodes.get(good_id).unwrap(),
            &good,
            &entries,
            &empty
        ));

        // float16 zero_points: in the union, wrong for slot 3.
        let (bad, bad_id) = graph_with_node(
            "MatMulNBits",
            "com.microsoft",
            &[
                DataType::Float32,
                DataType::Uint8,
                DataType::Float32,
                DataType::Float16,
            ],
            &[DataType::Float32],
        );
        assert!(
            !super::node_passes_dtype_filter(
                bad.nodes.get(bad_id).unwrap(),
                &bad,
                &entries,
                &empty
            ),
            "a float16 zero_points node must not be claimed: the kernel rejects it"
        );

        // uint8 activation: also in the union, also wrong for slot 0.
        let (swapped, swapped_id) = graph_with_node(
            "MatMulNBits",
            "com.microsoft",
            &[
                DataType::Uint8,
                DataType::Uint8,
                DataType::Float32,
                DataType::Uint8,
            ],
            &[DataType::Float32],
        );
        assert!(
            !super::node_passes_dtype_filter(
                swapped.nodes.get(swapped_id).unwrap(),
                &swapped,
                &entries,
                &empty
            ),
            "a uint8 activation must not be claimed"
        );
    }

    #[test]
    fn dtype_filter_applies_per_output_slot_constraints() {
        const FLOATS: &[DataType] = &[DataType::Float32, DataType::Float16, DataType::BFloat16];
        let entries = vec![KernelRegistryEntry {
            op_type: "DsaIndexSelect",
            domain: "pkg.nxrt",
            since_version: 1,
            end_version: i32::MAX,
            supported_dtypes: &[
                DataType::Float32,
                DataType::Float16,
                DataType::BFloat16,
                DataType::Int64,
            ],
            input_dtype_constraints: &[
                (0, FLOATS),
                (1, FLOATS),
                (2, FLOATS),
                (3, &[DataType::Float32]),
            ],
            output_dtype_constraints: &[(0, &[DataType::Int64])],
        }];
        let empty = std::collections::HashSet::new();

        let (good, good_id) = graph_with_node(
            "DsaIndexSelect",
            "pkg.nxrt",
            &[
                DataType::Float32,
                DataType::Float32,
                DataType::Float32,
                DataType::Float32,
            ],
            &[DataType::Int64],
        );
        assert!(super::node_passes_dtype_filter(
            good.nodes.get(good_id).unwrap(),
            &good,
            &entries,
            &empty
        ));

        let (bad, bad_id) = graph_with_node(
            "DsaIndexSelect",
            "pkg.nxrt",
            &[
                DataType::Float32,
                DataType::Float32,
                DataType::Float32,
                DataType::Float32,
            ],
            &[DataType::Float32],
        );
        assert!(
            !super::node_passes_dtype_filter(
                bad.nodes.get(bad_id).unwrap(),
                &bad,
                &entries,
                &empty
            ),
            "a Float32 selected_indices output must be declined before Compile/Compute"
        );
    }

    /// Node with Undefined dtype is NOT claimed (fail closed).
    #[test]
    fn dtype_filter_rejects_undefined_dtype() {
        let entries = vec![KernelRegistryEntry {
            op_type: "Add",
            domain: "",
            since_version: 7,
            end_version: 21,
            supported_dtypes: &[DataType::Float32],
            input_dtype_constraints: &[],
            output_dtype_constraints: &[],
        }];
        let (g, nid) = graph_with_node("Add", "", &[DataType::Undefined], &[DataType::Float32]);
        let node = g.nodes.get(nid).unwrap();
        assert!(!super::node_passes_dtype_filter(
            node,
            &g,
            &entries,
            &std::collections::HashSet::new()
        ));
    }

    /// Node with no matching registry entry is NOT claimed (fail closed).
    #[test]
    fn dtype_filter_rejects_unknown_op() {
        let entries = vec![KernelRegistryEntry {
            op_type: "Add",
            domain: "",
            since_version: 7,
            end_version: 21,
            supported_dtypes: &[DataType::Float32],
            input_dtype_constraints: &[],
            output_dtype_constraints: &[],
        }];
        let (g, nid) = graph_with_node("UnknownOp", "", &[DataType::Float32], &[DataType::Float32]);
        let node = g.nodes.get(nid).unwrap();
        assert!(!super::node_passes_dtype_filter(
            node,
            &g,
            &entries,
            &std::collections::HashSet::new()
        ));
    }

    /// Empty registry entries → filter is bypassed (legacy compile-only mode).
    #[test]
    fn dtype_filter_bypassed_when_no_entries() {
        let (g, nid) = graph_with_node("Add", "", &[DataType::Int64], &[DataType::Int64]);
        let node = g.nodes.get(nid).unwrap();
        assert!(super::node_passes_dtype_filter(
            node,
            &g,
            &[],
            &std::collections::HashSet::new()
        ));
    }

    /// B1: A model-provided value whose name happens to start with
    /// `__absent_output_` must NOT be treated as absent — only ValueIds
    /// explicitly in the out-of-band `absent_outputs` set are exempt.
    #[test]
    fn forgeable_name_not_treated_as_absent() {
        use onnx_runtime_ir::{Graph, Node, NodeId};
        let mut g = Graph::new();
        // Create a value with the old sentinel name prefix — simulates a
        // model that contains such a name (untrusted input).
        let forgery = g.create_named_value("__absent_output_0_1", DataType::Undefined, vec![]);
        let input = g.create_named_value("x", DataType::Float32, vec![]);
        g.add_input(input);

        let node = Node::new(NodeId(0), "Add", vec![Some(input)], vec![forgery]);
        let _nid = g.insert_node(node);
        let node = g.nodes.iter().next().unwrap().1;

        let entries = vec![KernelRegistryEntry {
            op_type: "Add",
            domain: "",
            since_version: 7,
            end_version: 21,
            supported_dtypes: &[DataType::Float32],
            input_dtype_constraints: &[],
            output_dtype_constraints: &[],
        }];

        // Empty absent set — the forgery name should NOT grant exemption.
        let empty_absent = std::collections::HashSet::new();
        assert!(
            !super::node_passes_dtype_filter(node, &g, &entries, &empty_absent),
            "Value named '__absent_output_*' must NOT bypass dtype filter \
             unless its ValueId is in the out-of-band absent_outputs set"
        );

        // With the ValueId in the absent set — it should pass.
        let mut absent = std::collections::HashSet::new();
        absent.insert(forgery);
        // Still fails because Undefined is not in supported_dtypes and we
        // only skip the Undefined check for absent outputs — the entry
        // supported_dtypes filter is separate.
        // The point: absent membership is by ValueId, not by name.
        assert!(
            super::node_passes_dtype_filter(node, &g, &entries, &absent),
            "ValueId in absent_outputs set should be exempt from Undefined rejection"
        );
    }

    /// B2: Symbolic (dynamic) dimensions must preserve rank in shape inference.
    /// A shape [None, None, Some(768)] has rank 3, and for_node must not
    /// collapse it to rank 1.
    #[test]
    fn symbolic_dims_preserve_rank() {
        use onnx_runtime_ir::{Node, NodeId};
        // Add with one input having symbolic dims [batch, seq, 768].
        let node = Node::new(NodeId(0), "Add", vec![None, None], vec![]);
        let shapes: Vec<Vec<Option<usize>>> = vec![
            vec![None, None, Some(768)], // rank 3 with symbolic batch, seq
            vec![Some(1), Some(1), Some(768)],
        ];
        let si = crate::compute::ShapeInference::for_node(&node, &shapes, 1);
        // ElementwiseBroadcast — the op should not be declined just
        // because inputs have symbolic dims.
        assert!(
            matches!(si, crate::compute::ShapeInference::ElementwiseBroadcast),
            "Add with symbolic dims must not be Declined; got {si:?}"
        );
    }

    /// B2: Conv with all-symbolic spatial dims declines (fail-closed)
    /// rather than producing a wrong answer from truncated rank.
    #[test]
    fn conv_declines_with_symbolic_spatial_dims() {
        use onnx_runtime_ir::{Node, NodeId};
        let node = Node::new(NodeId(0), "Conv", vec![None, None], vec![]);
        // input[0] = [1, 3, None, None] (rank 4, but spatial dims unknown)
        // input[1] = [16, 3, None, None] (weight with unknown kernel)
        let shapes: Vec<Vec<Option<usize>>> = vec![
            vec![Some(1), Some(3), None, None],
            vec![Some(16), Some(3), None, None],
        ];
        let si = crate::compute::ShapeInference::for_node(&node, &shapes, 1);
        // Conv should decline because kernel dims are unknown and no
        // kernel_shape attribute is provided.
        assert!(
            matches!(si, crate::compute::ShapeInference::Declined { .. }),
            "Conv with unknown spatial dims and no kernel_shape attr must Decline; got {si:?}"
        );
    }

    /// B2 (claim-time routability gate): a node whose inputs are all either
    /// graph inputs or produced by another node is routable by the fused
    /// compile path and must be admitted.
    #[test]
    fn routable_when_inputs_are_graph_input_or_produced() {
        use onnx_runtime_ir::{Graph, GraphView, GraphViewCache, Node, NodeId, Shape};
        let mut g = Graph::new();
        let x = g.create_named_value("x", DataType::Float32, Shape::default());
        g.add_input(x);
        // producer: p_out = Identity(x)
        let p_out = g.create_named_value("p_out", DataType::Float32, Shape::default());
        g.insert_node(Node::new(NodeId(0), "Identity", vec![Some(x)], vec![p_out]));
        // consumer: y = Add(x, p_out) — both inputs routable
        let y = g.create_named_value("y", DataType::Float32, Shape::default());
        g.add_output(y);
        g.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(x), Some(p_out)],
            vec![y],
        ));

        let cache = GraphViewCache::build(&g).unwrap();
        let view = GraphView::new(&g, &cache);
        let add = view
            .nodes()
            .find(|&n| view.node(n).op_type == "Add")
            .expect("Add node present");
        assert!(
            super::node_inputs_all_routable(&view, add),
            "a node consuming only a graph input and a produced value is routable"
        );
    }

    /// Constant-input flags must come from ORT's initializer list, not from
    /// the producer/`is_graph_input` shape of the graph.
    ///
    /// This is the exact shape ORT hands a fused node's subgraph over in:
    /// the weight has no producer *and* is listed as a graph input of the
    /// subgraph, because it is an input of the fused node. Deriving the flag
    /// the way `node_inputs_all_routable` derives routability marks every
    /// weight non-constant, which costs `MatMulNBits` its prepack — and, since
    /// its MLAS SQNBit path is gated on the same flag, its fast kernel too.
    #[test]
    fn constant_initializers_are_flagged_even_when_the_subgraph_calls_them_inputs() {
        use onnx_runtime_ir::{Graph, GraphView, GraphViewCache, Node, NodeId, Shape};
        let mut g = Graph::new();
        let a = g.create_named_value("a", DataType::Float32, Shape::default());
        let w = g.create_named_value("w", DataType::Uint8, Shape::default());
        let scales = g.create_named_value("scales", DataType::Float32, Shape::default());
        let y = g.create_named_value("y", DataType::Float32, Shape::default());
        // Every input of the fused node is a graph input of the subgraph ORT
        // hands over, weights included.
        g.add_input(a);
        g.add_input(w);
        g.add_input(scales);
        g.add_output(y);
        g.insert_node(Node::new(
            NodeId(0),
            "MatMulNBits",
            vec![Some(a), Some(w), Some(scales), None],
            vec![y],
        ));

        let cache = GraphViewCache::build(&g).unwrap();
        let view = GraphView::new(&g, &cache);
        let node = view.nodes().next().expect("one node");
        let constant_initializers: std::collections::HashSet<String> =
            ["w".to_owned(), "scales".to_owned()].into_iter().collect();

        assert_eq!(
            super::constant_input_flags(&view, node, &constant_initializers),
            vec![false, true, true, false],
            "weights are constant, the activation is not, and an absent optional \
             input is not"
        );
    }

    /// With no initializers, nothing is constant — including values that have
    /// no producer. A subgraph whose weights ORT kept outside it must not have
    /// its activations cached as if they were weights.
    #[test]
    fn nothing_is_constant_without_an_initializer_list() {
        use onnx_runtime_ir::{Graph, GraphView, GraphViewCache, Node, NodeId, Shape};
        let mut g = Graph::new();
        let a = g.create_named_value("a", DataType::Float32, Shape::default());
        let b = g.create_named_value("b", DataType::Float32, Shape::default());
        let y = g.create_named_value("y", DataType::Float32, Shape::default());
        g.add_input(a);
        g.add_output(y);
        g.insert_node(Node::new(
            NodeId(0),
            "MatMul",
            vec![Some(a), Some(b)],
            vec![y],
        ));

        let cache = GraphViewCache::build(&g).unwrap();
        let view = GraphView::new(&g, &cache);
        let node = view.nodes().next().expect("one node");
        assert_eq!(
            super::constant_input_flags(&view, node, &std::collections::HashSet::new()),
            vec![false, false],
            "a producerless value is only constant when ORT says it is an initializer"
        );
    }

    /// B2 (claim-time routability gate): a node consuming a weight initializer
    /// (a value with no producer that is not a graph input) is NOT routable by
    /// the fused compile path and must be declined at claim time so ORT
    /// partitions around it — this is what keeps session creation from failing
    /// at Compile time on real decoders.
    #[test]
    fn not_routable_when_input_is_initializer() {
        use onnx_runtime_ir::{Graph, GraphView, GraphViewCache, Node, NodeId, Shape};
        let mut g = Graph::new();
        let x = g.create_named_value("x", DataType::Float32, Shape::default());
        g.add_input(x);
        // w is a weight: created, but neither a graph input nor produced by any
        // node — exactly how the plugin's graph reader represents initializers.
        let w = g.create_named_value("w", DataType::Float32, Shape::default());
        let y = g.create_named_value("y", DataType::Float32, Shape::default());
        g.add_output(y);
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(x), Some(w)], vec![y]));

        let cache = GraphViewCache::build(&g).unwrap();
        let view = GraphView::new(&g, &cache);
        let add = view
            .nodes()
            .find(|&n| view.node(n).op_type == "Add")
            .expect("Add node present");
        assert!(
            !super::node_inputs_all_routable(&view, add),
            "a node consuming a weight initializer must be declined at claim time"
        );
    }

    /// Routing filter: a claim is unroutable when one of its values is both an
    /// output of the fused subgraph (consumed outside the claim) and an input
    /// to a later node inside it. This is the `com.microsoft::FastGelu`
    /// float16 shape — `X_bias` feeds both the declined `Tanh` chain and three
    /// `Mul`s we keep — and `build_subgraph_routing` fails Compile on it, which
    /// ORT reports as a session-creation error rather than falling back.
    #[test]
    fn claim_with_internally_reused_subgraph_output_is_unroutable() {
        use onnx_runtime_ir::{Graph, Node, NodeId, Shape};
        let mut g = Graph::new();
        let x = g.create_named_value("x", DataType::Float32, Shape::default());
        g.add_input(x);
        let bias = g.create_named_value("bias", DataType::Float32, Shape::default());
        let t = g.create_named_value("t", DataType::Float32, Shape::default());
        let y = g.create_named_value("y", DataType::Float32, Shape::default());
        g.add_output(y);

        // bias = Identity(x)   ← claimed
        let n_id = g.insert_node(Node::new(NodeId(0), "Identity", vec![Some(x)], vec![bias]));
        // t = Tanh(bias)       ← NOT claimed (declined; runs on the host)
        g.insert_node(Node::new(NodeId(0), "Tanh", vec![Some(bias)], vec![t]));
        // y = Mul(bias, t)     ← claimed, and re-reads `bias` from inside
        let n_mul = g.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(bias), Some(t)],
            vec![y],
        ));

        assert!(
            super::claim_has_unroutable_internal_output(&g, &[n_id, n_mul]),
            "a value that is both a subgraph output and an internal input must be declined"
        );
        // A single-node claim is never routed, so it is always accepted.
        assert!(
            !super::claim_has_unroutable_internal_output(&g, &[n_id]),
            "single-node claims are not routed and must not be filtered"
        );
    }

    /// The routing filter must not fire on ordinary fusions: a purely internal
    /// intermediate gets a buffer, and a terminal subgraph output has no
    /// internal consumer. Both are routable, so a `MatMul`+`Add`+`Relu` chain
    /// (including one reading an initializer) keeps its fusion.
    #[test]
    fn ordinary_fusion_is_not_filtered() {
        use onnx_runtime_ir::{Graph, Node, NodeId, Shape};
        let mut g = Graph::new();
        let x = g.create_named_value("x", DataType::Float32, Shape::default());
        g.add_input(x);
        let w = g.create_named_value("w", DataType::Float32, Shape::default());
        let m = g.create_named_value("m", DataType::Float32, Shape::default());
        let y = g.create_named_value("y", DataType::Float32, Shape::default());
        g.add_output(y);
        let bias = g.create_named_value("bias", DataType::Float32, Shape::default());
        let a = g.create_named_value("a", DataType::Float32, Shape::default());
        let n_mm = g.insert_node(Node::new(
            NodeId(0),
            "MatMul",
            vec![Some(x), Some(w)],
            vec![m],
        ));
        let n_add = g.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(m), Some(bias)],
            vec![a],
        ));
        let n_relu = g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(a)], vec![y]));
        assert!(
            !super::claim_has_unroutable_internal_output(&g, &[n_mm, n_add, n_relu]),
            "an ordinary chain with internal-only intermediates must keep its fusion"
        );
    }

    /// Partial-GPU-claim gate: the flag is OFF unless explicitly set to a truthy
    /// value, so a GPU EP does not form an interspersed CPU/GPU partition (which
    /// currently hangs, #982) by default. Both accepted env names enable it.
    #[test]
    fn partial_gpu_claim_flag_parsing() {
        // Off for unset / empty / explicit falsy / unrecognized values.
        assert!(!super::parse_bool_flag(None), "unset must be off");
        assert!(!super::parse_bool_flag(Some("")), "empty must be off");
        assert!(!super::parse_bool_flag(Some("0")), "0 must be off");
        assert!(!super::parse_bool_flag(Some("false")), "false must be off");
        assert!(
            !super::parse_bool_flag(Some("maybe")),
            "unrecognized must be off"
        );
        // On for the accepted truthy spellings, case/space-insensitive.
        assert!(super::parse_bool_flag(Some("1")));
        assert!(super::parse_bool_flag(Some("true")));
        assert!(super::parse_bool_flag(Some("  On ")));
        assert!(super::parse_bool_flag(Some("YES")));
        // Both env names are wired into the gate.
        assert_eq!(super::PARTIAL_GPU_CLAIM_ENV.len(), 2);
        assert!(super::PARTIAL_GPU_CLAIM_ENV.contains(&"ONNX_GENAI_PLUGIN_CLAIM_MS_DOMAIN"));
    }
}
