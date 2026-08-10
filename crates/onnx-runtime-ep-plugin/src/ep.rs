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
use crate::status::{fail_status, invalid_arg_status, ok_status};

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
                GetKernelRegistry: None,
                ..Default::default()
            },
            ep,
            name_cstr,
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
                return fail_status(&format!("Compile: failed to read subgraph {i}: {msg}"));
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

            let opset = ir_graph.effective_opset(node).unwrap_or(0);

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

                    // Determine shape inference strategy using full node
                    // attributes (wired to Deckard's 22 rules).
                    let shape_inference =
                        crate::compute::ShapeInference::for_node(node, &shapes, num_outputs);

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
                    sources.push(NodeInputSource::Ort(0));
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
}
