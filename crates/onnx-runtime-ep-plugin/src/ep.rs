//! `ExportedEp` — the heap object behind an opaque `OrtEp*`.
//!
//! Implements `GetCapability`, `Compile`, `ReleaseNodeComputeInfos`, and
//! `GetKernelRegistry` by delegating to the Rust `ExecutionProvider` trait.

use std::panic::AssertUnwindSafe;
use std::ptr;

use onnx_genai_ort_sys as ort;
use onnx_runtime_ep_api::provider::ExecutionProvider;
use onnx_runtime_ir::DataType;

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
}

/// A heap-allocated EP whose raw pointer is returned as `OrtEp*`.
///
/// The first field is `OrtEp` so the pointer can be cast directly.
#[repr(C)]
pub struct ExportedEp {
    /// The vtable ORT reads through the `OrtEp*` pointer.
    pub vtable: ort::OrtEp,
    /// The Rust EP instance.
    pub ep: Box<dyn ExecutionProvider>,
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
        let name_cstr = std::ffi::CString::new(ep.name())
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
    let claims = ort_view.query_capabilities(exported.ep.as_ref());

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
                let input_shapes: Vec<Vec<usize>> = node
                    .inputs
                    .iter()
                    .map(|input| {
                        input
                            .and_then(|vid| ir_graph.values.get(vid))
                            .map(|v| v.shape.iter().filter_map(|d| d.as_static()).collect())
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
    let claims: Vec<_> = claims
        .into_iter()
        .filter(|claim| {
            claim.node_ids.iter().all(|&nid| {
                let Some(node) = ir_graph.nodes.get(nid) else {
                    return false;
                };
                // Fail-closed: every output must have a resolved, producible dtype.
                for &vid in &node.outputs {
                    let Some(value) = ir_graph.values.get(vid) else {
                        return false;
                    };
                    if value.dtype == DataType::Undefined {
                        return false;
                    }
                }
                node_passes_dtype_filter(node, ir_graph, &exported.registry_entries)
            })
        })
        .collect();

    if claims.is_empty() {
        return ok_status();
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
    for input in &node.inputs {
        let Some(vid) = input else { continue };
        let Some(value) = ir_graph.values.get(*vid) else {
            continue;
        };
        if value.dtype == DataType::Undefined {
            return false;
        }
        if !entry.supported_dtypes.contains(&value.dtype) {
            return false;
        }
    }
    for &vid in &node.outputs {
        let Some(value) = ir_graph.values.get(vid) else {
            continue;
        };
        if value.dtype == DataType::Undefined {
            return false;
        }
        if !entry.supported_dtypes.contains(&value.dtype) {
            return false;
        }
    }
    true
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
            let shapes: Vec<Vec<usize>> = view
                .node_inputs(node_idx)
                .iter()
                .map(|input| {
                    input
                        .map(|v| {
                            view.value(v)
                                .shape
                                .iter()
                                .filter_map(|d| d.as_static())
                                .collect()
                        })
                        .unwrap_or_default()
                })
                .collect();

            let opset = ir_graph.effective_opset(node).unwrap_or(0);

            match exported.ep.get_kernel(node, &shapes, opset) {
                Ok(kernel) => {
                    let num_inputs = view.node_inputs(node_idx).len();
                    let num_outputs = view.node_outputs(node_idx).len();

                    // Read per-output declared dtype from the ORT graph's
                    // value info — never inferred from inputs. This is the
                    // authoritative dtype for Cast, Where, Shape, etc.
                    let output_dtypes: Vec<DataType> = view
                        .node_outputs(node_idx)
                        .iter()
                        .map(|&val_idx| view.value(val_idx).dtype)
                        .collect();

                    // Determine shape inference strategy using full node
                    // attributes (wired to Deckard's 22 rules).
                    let shape_inference =
                        crate::compute::ShapeInference::for_node(node, &shapes, num_outputs);

                    entries.push(crate::compute::CompiledKernelEntry {
                        kernel,
                        num_inputs,
                        num_outputs,
                        output_dtypes,
                        shape_inference,
                    });
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

        // For multi-node fused subgraphs, construct the SubgraphRouting so
        // intermediates are threaded correctly in topological order.
        if info.entries.len() > 1
            && let Some(routing) = build_subgraph_routing(&view, ir_graph)
        {
            info.set_routing(routing);
        }

        let info_ptr = Box::into_raw(Box::new(info));
        unsafe { *out_infos.add(i) = info_ptr.cast::<ort::OrtNodeComputeInfo>() };
    }

    ok_status()
}

/// Build a `SubgraphRouting` table for a multi-node fused subgraph.
///
/// Determines which node inputs come from ORT kernel-context inputs (graph inputs)
/// vs. intermediate buffers, and which outputs go to ORT outputs vs. buffers.
fn build_subgraph_routing(
    view: &onnx_runtime_ir::GraphView<'_>,
    graph: &onnx_runtime_ir::Graph,
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
            if let Some(&ort_idx) = output_index.get(&vid) {
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
        let si = crate::compute::ShapeInference::for_node(&node, &[vec![2, 3]], 1);
        assert!(
            matches!(si, crate::compute::ShapeInference::Declined { .. }),
            "NonZero must be Declined; got {si:?}"
        );
    }

    /// ShapeInference::for_node accepts Add (elementwise broadcast) without
    /// attributes.
    #[test]
    fn shape_inference_accepts_add() {
        use onnx_runtime_ir::{Node, NodeId};
        let node = Node::new(NodeId(0), "Add", vec![None, None], vec![]);
        let si = crate::compute::ShapeInference::for_node(&node, &[vec![2, 3], vec![2, 3]], 1);
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
        let si = crate::compute::ShapeInference::for_node(&node, &[vec![2, 3], vec![2, 5]], 1);
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
        let si = crate::compute::ShapeInference::for_node(&node, &[vec![3, 4]], 1);
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
        }];
        let (g, nid) = graph_with_node(
            "Add",
            "",
            &[DataType::Float32, DataType::Float32],
            &[DataType::Float32],
        );
        let node = g.nodes.get(nid).unwrap();
        assert!(super::node_passes_dtype_filter(node, &g, &entries));
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
        }];
        let (g, nid) = graph_with_node(
            "Add",
            "",
            &[DataType::Int64, DataType::Int64],
            &[DataType::Int64],
        );
        let node = g.nodes.get(nid).unwrap();
        assert!(!super::node_passes_dtype_filter(node, &g, &entries));
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
        }];
        let (g, nid) = graph_with_node("Add", "", &[DataType::Undefined], &[DataType::Float32]);
        let node = g.nodes.get(nid).unwrap();
        assert!(!super::node_passes_dtype_filter(node, &g, &entries));
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
        }];
        let (g, nid) = graph_with_node("UnknownOp", "", &[DataType::Float32], &[DataType::Float32]);
        let node = g.nodes.get(nid).unwrap();
        assert!(!super::node_passes_dtype_filter(node, &g, &entries));
    }

    /// Empty registry entries → filter is bypassed (legacy compile-only mode).
    #[test]
    fn dtype_filter_bypassed_when_no_entries() {
        let (g, nid) = graph_with_node("Add", "", &[DataType::Int64], &[DataType::Int64]);
        let node = g.nodes.get(nid).unwrap();
        assert!(super::node_passes_dtype_filter(node, &g, &[]));
    }
}
