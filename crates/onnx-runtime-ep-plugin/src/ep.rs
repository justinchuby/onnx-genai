//! `ExportedEp` — the heap object behind an opaque `OrtEp*`.
//!
//! Implements `GetCapability`, `Compile`, and `ReleaseNodeComputeInfos` by
//! delegating to the Rust `ExecutionProvider` trait.

use std::panic::AssertUnwindSafe;
use std::ptr;

use onnx_genai_ort_sys as ort;
use onnx_runtime_ep_api::provider::ExecutionProvider;

use crate::compute::ExportedComputeInfo;
use crate::graph_reader::OutboundGraphReader;
use crate::status::{fail_status, ok_status};

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
}

impl ExportedEp {
    pub fn new(ep: Box<dyn ExecutionProvider>) -> Self {
        let name_cstr = std::ffi::CString::new(ep.name())
            .unwrap_or_else(|_| std::ffi::CString::new("nxrt_ep").unwrap());
        Self {
            vtable: ort::OrtEp {
                ort_version_supported: ort::ORT_API_VERSION,
                GetName: Some(ep_get_name),
                GetCapability: Some(ep_get_capability),
                Compile: Some(ep_compile),
                ReleaseNodeComputeInfos: Some(ep_release_node_compute_infos),
                GetPreferredDataLayout: None,
                ShouldConvertDataLayoutForOp: None,
                SetDynamicOptions: None,
                OnRunStart: None,
                OnRunEnd: None,
                CreateAllocator: None,
                CreateSyncStreamForDevice: None,
                GetCompiledModelCompatibilityInfo: None,
                ValidateCompiledModelCompatibilityInfo: None,
                GetKernelRegistry: None,
            },
            ep,
            name_cstr,
        }
    }
}

// ─── OrtEp callbacks ────────────────────────────────────────────────────────

/// GetName: return the EP name as a C string.
unsafe extern "C" fn ep_get_name(
    ep: *const ort::OrtEp,
) -> *const std::ffi::c_char {
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
        return fail_status("GetCapability: null argument");
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
                )
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
        return fail_status("Compile: null argument or zero count");
    }
    // B3: null-check graphs pointer
    if graphs.is_null() {
        return fail_status("Compile: graphs pointer is null");
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
                return fail_status(&format!(
                    "Compile: failed to read subgraph {i}: {msg}"
                ));
            }
        };

        let ir_graph = reader.to_ir_graph();
        let cache = match onnx_runtime_ir::GraphViewCache::build(ir_graph) {
            Ok(c) => c,
            Err(e) => {
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

            let opset = ir_graph
                .effective_opset(node)
                .unwrap_or(0);

            match exported.ep.get_kernel(node, &shapes, opset) {
                Ok(kernel) => {
                    let num_inputs = view.node_inputs(node_idx).len();
                    let num_outputs = view.node_outputs(node_idx).len();

                    // Determine output dtype from the first input's dtype
                    // (correct for elementwise ops; fail-closed for others).
                    let output_dtype = view
                        .node_inputs(node_idx)
                        .iter()
                        .find_map(|v| v.map(|val| view.value(val).dtype))
                        .unwrap_or(onnx_runtime_ir::DataType::Float32);

                    // Determine shape inference strategy from op_type.
                    let shape_inference =
                        crate::compute::ShapeInference::for_op(&node.op_type);

                    entries.push(crate::compute::CompiledKernelEntry {
                        kernel,
                        num_inputs,
                        num_outputs,
                        output_dtype,
                        shape_inference,
                    });
                }
                Err(e) => {
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
        let info = ExportedComputeInfo::new(entries);
        let info_ptr = Box::into_raw(Box::new(info));
        unsafe { *out_infos.add(i) = info_ptr.cast::<ort::OrtNodeComputeInfo>() };
    }

    ok_status()
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
