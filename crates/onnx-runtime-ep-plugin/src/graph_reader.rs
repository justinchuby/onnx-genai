//! `OutboundGraphReader` — reads ORT's `OrtGraph*` through the ORT C API.
//!
//! This is the inverse of `abi/host.rs` which projects our IR as a fake
//! `OrtGraph*`. Here we read a REAL `OrtGraph*` from upstream ORT and build
//! a nxrt `Graph` for capability discovery.

use std::collections::HashMap;
use std::ffi::CStr;
use std::ptr;

use onnx_genai_ort_sys as ort;
use onnx_runtime_ir::{DataType, Graph, Node, NodeId, Shape, ValueId};

/// A reader that extracts nxrt IR from an ORT `OrtGraph*`.
pub struct OutboundGraphReader {
    graph: Graph,
    /// Maps ORT node index (position in the `Graph_GetNodes` array) → our NodeId.
    ort_index_to_node_id: Vec<NodeId>,
    /// The original ORT node pointers, kept for passing back to
    /// `EpGraphSupportInfo_AddNodesToFuse`.
    ort_node_ptrs: Vec<*const ort::OrtNode>,
}

// OutboundGraphReader intentionally does NOT implement Send or Sync.
// The raw OrtNode pointers it holds are only valid within the ORT callback
// frame in which the reader was constructed. Granting Send/Sync would allow
// the reader (and its dangling pointers) to escape to other threads.
// Use it stack-locally inside a single GetCapability/Compile callback only.

impl OutboundGraphReader {
    /// Read an `OrtGraph*` and produce an IR `Graph`.
    ///
    /// # Safety
    ///
    /// `graph_ptr` must be a valid `OrtGraph*` from ORT, and the host ORT API
    /// must have been initialized via [`crate::status::set_host_api`].
    pub unsafe fn from_ort_graph(
        graph_ptr: *const ort::OrtGraph,
    ) -> Result<Self, String> {
        let api = crate::status::host_api();
        if api.is_null() {
            return Err("host ORT API not initialized".into());
        }

        let mut ir_graph = Graph::new();
        let mut ort_index_to_node_id = Vec::new();

        // Get nodes.
        let num_nodes = unsafe { Self::graph_num_nodes(api, graph_ptr)? };

        let mut ort_nodes: Vec<*const ort::OrtNode> = vec![ptr::null(); num_nodes];
        if num_nodes > 0 {
            unsafe { Self::graph_get_nodes(api, graph_ptr, ort_nodes.as_mut_ptr(), num_nodes)? };
        }

        // Get graph inputs (value infos).
        let num_inputs = unsafe { Self::graph_num_inputs(api, graph_ptr)? };
        let mut input_infos: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); num_inputs];
        if num_inputs > 0 {
            unsafe {
                Self::graph_get_inputs(api, graph_ptr, input_infos.as_mut_ptr(), num_inputs)?
            };
        }

        // Get graph outputs.
        let num_outputs = unsafe { Self::graph_num_outputs(api, graph_ptr)? };
        let mut output_infos: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); num_outputs];
        if num_outputs > 0 {
            unsafe {
                Self::graph_get_outputs(api, graph_ptr, output_infos.as_mut_ptr(), num_outputs)?
            };
        }

        // Build a name→ValueId map for wiring edges.
        let mut value_map: HashMap<String, ValueId> = HashMap::new();

        // Create graph input values.
        for info in &input_infos {
            if info.is_null() {
                continue;
            }
            let (name, dtype, shape) = unsafe { Self::read_value_info(api, *info)? };
            let vid = ir_graph.create_named_value(&name, dtype, shape);
            ir_graph.add_input(vid);
            value_map.insert(name, vid);
        }

        // First pass: create output values for all nodes.
        let mut node_output_names: Vec<Vec<String>> = Vec::with_capacity(num_nodes);
        let mut node_input_names: Vec<Vec<String>> = Vec::with_capacity(num_nodes);

        for ort_node in &ort_nodes {
            if ort_node.is_null() {
                node_output_names.push(Vec::new());
                node_input_names.push(Vec::new());
                continue;
            }

            // Read node outputs.
            let n_out = unsafe { Self::node_num_outputs(api, *ort_node)? };
            let mut out_infos: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); n_out];
            if n_out > 0 {
                unsafe {
                    Self::node_get_outputs(api, *ort_node, out_infos.as_mut_ptr(), n_out)?;
                }
            }

            let mut out_names = Vec::with_capacity(n_out);
            for info in &out_infos {
                if info.is_null() {
                    out_names.push(String::new());
                    continue;
                }
                let (name, dtype, shape) = unsafe { Self::read_value_info(api, *info)? };
                if !name.is_empty() && !value_map.contains_key(&name) {
                    let vid = ir_graph.create_named_value(&name, dtype, shape);
                    value_map.insert(name.clone(), vid);
                }
                out_names.push(name);
            }
            node_output_names.push(out_names);

            // Read node inputs.
            let n_in = unsafe { Self::node_num_inputs(api, *ort_node)? };
            let mut in_infos: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); n_in];
            if n_in > 0 {
                unsafe {
                    Self::node_get_inputs(api, *ort_node, in_infos.as_mut_ptr(), n_in)?;
                }
            }

            let mut in_names = Vec::with_capacity(n_in);
            for info in &in_infos {
                if info.is_null() {
                    in_names.push(String::new());
                    continue;
                }
                let (name, dtype, shape) = unsafe { Self::read_value_info(api, *info)? };
                if !name.is_empty() && !value_map.contains_key(&name) {
                    let vid = ir_graph.create_named_value(&name, dtype, shape);
                    value_map.insert(name.clone(), vid);
                }
                in_names.push(name);
            }
            node_input_names.push(in_names);
        }

        // Second pass: create nodes with proper edges.
        for (i, ort_node) in ort_nodes.iter().enumerate() {
            if ort_node.is_null() {
                continue;
            }

            let op_type = unsafe { Self::node_op_type(api, *ort_node)? };
            let domain = unsafe { Self::node_domain(api, *ort_node)? };
            let since_version = unsafe { Self::node_since_version(api, *ort_node)? };
            let name = unsafe { Self::node_name(api, *ort_node).unwrap_or_default() };

            let inputs: Vec<Option<ValueId>> = node_input_names[i]
                .iter()
                .map(|n| {
                    if n.is_empty() {
                        None
                    } else {
                        value_map.get(n).copied()
                    }
                })
                .collect();

            let outputs: Vec<ValueId> = node_output_names[i]
                .iter()
                .filter_map(|n| {
                    if n.is_empty() {
                        None
                    } else {
                        value_map.get(n).copied()
                    }
                })
                .collect();

            let mut node = Node::new(NodeId(0), &op_type, inputs, outputs);
            node.name = name;
            node.domain = domain;
            if since_version > 0 {
                node.version = Some(since_version as i64);
            }

            let nid = ir_graph.insert_node(node);
            ort_index_to_node_id.push(nid);
        }

        // Set graph outputs.
        for info in &output_infos {
            if info.is_null() {
                continue;
            }
            let (name, _dtype, _shape) = unsafe { Self::read_value_info(api, *info)? };
            if let Some(&vid) = value_map.get(&name) {
                ir_graph.add_output(vid);
            }
        }

        // Set default opset.
        ir_graph
            .opset_imports
            .insert(String::new(), ort::ORT_API_VERSION as u64);

        Ok(Self {
            graph: ir_graph,
            ort_index_to_node_id,
            ort_node_ptrs: ort_nodes,
        })
    }

    /// Get the IR graph for capability discovery.
    pub fn to_ir_graph(&self) -> &Graph {
        &self.graph
    }

    /// Map our internal `NodeId` back to the ORT node's index in the original
    /// `Graph_GetNodes` array, for reporting claims.
    pub fn node_id_to_ort_index(&self, node_id: NodeId) -> usize {
        self.ort_index_to_node_id
            .iter()
            .position(|&nid| nid == node_id)
            .unwrap_or(0)
    }

    /// Get the original ORT `OrtNode*` pointer for one of our `NodeId`s.
    pub fn node_id_to_ort_ptr(&self, node_id: NodeId) -> *const ort::OrtNode {
        let idx = self.node_id_to_ort_index(node_id);
        self.ort_node_ptrs.get(idx).copied().unwrap_or(ptr::null())
    }

    // ─── ORT API wrappers ───────────────────────────────────────────────

    unsafe fn graph_num_nodes(
        api: *const ort::OrtApi,
        graph: *const ort::OrtGraph,
    ) -> Result<usize, String> {
        let f = unsafe { (*api).Graph_GetNumNodes }
            .ok_or("OrtApi.Graph_GetNumNodes is null")?;
        let mut count = 0usize;
        let status = unsafe { f(graph, &mut count) };
        Self::check(status)?;
        Ok(count)
    }

    unsafe fn graph_get_nodes(
        api: *const ort::OrtApi,
        graph: *const ort::OrtGraph,
        out: *mut *const ort::OrtNode,
        count: usize,
    ) -> Result<(), String> {
        let f = unsafe { (*api).Graph_GetNodes }
            .ok_or("OrtApi.Graph_GetNodes is null")?;
        let status = unsafe { f(graph, out, count) };
        Self::check(status)
    }

    unsafe fn graph_num_inputs(
        api: *const ort::OrtApi,
        graph: *const ort::OrtGraph,
    ) -> Result<usize, String> {
        let f = unsafe { (*api).Graph_GetNumInputs }
            .ok_or("OrtApi.Graph_GetNumInputs is null")?;
        let mut count = 0usize;
        let status = unsafe { f(graph, &mut count) };
        Self::check(status)?;
        Ok(count)
    }

    unsafe fn graph_get_inputs(
        api: *const ort::OrtApi,
        graph: *const ort::OrtGraph,
        out: *mut *const ort::OrtValueInfo,
        count: usize,
    ) -> Result<(), String> {
        let f = unsafe { (*api).Graph_GetInputs }
            .ok_or("OrtApi.Graph_GetInputs is null")?;
        let status = unsafe { f(graph, out, count) };
        Self::check(status)
    }

    unsafe fn graph_num_outputs(
        api: *const ort::OrtApi,
        graph: *const ort::OrtGraph,
    ) -> Result<usize, String> {
        let f = unsafe { (*api).Graph_GetNumOutputs }
            .ok_or("OrtApi.Graph_GetNumOutputs is null")?;
        let mut count = 0usize;
        let status = unsafe { f(graph, &mut count) };
        Self::check(status)?;
        Ok(count)
    }

    unsafe fn graph_get_outputs(
        api: *const ort::OrtApi,
        graph: *const ort::OrtGraph,
        out: *mut *const ort::OrtValueInfo,
        count: usize,
    ) -> Result<(), String> {
        let f = unsafe { (*api).Graph_GetOutputs }
            .ok_or("OrtApi.Graph_GetOutputs is null")?;
        let status = unsafe { f(graph, out, count) };
        Self::check(status)
    }

    unsafe fn node_num_inputs(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
    ) -> Result<usize, String> {
        let f = unsafe { (*api).Node_GetNumInputs }
            .ok_or("OrtApi.Node_GetNumInputs is null")?;
        let mut count = 0usize;
        let status = unsafe { f(node, &mut count) };
        Self::check(status)?;
        Ok(count)
    }

    unsafe fn node_get_inputs(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
        out: *mut *const ort::OrtValueInfo,
        count: usize,
    ) -> Result<(), String> {
        let f = unsafe { (*api).Node_GetInputs }
            .ok_or("OrtApi.Node_GetInputs is null")?;
        let status = unsafe { f(node, out, count) };
        Self::check(status)
    }

    unsafe fn node_num_outputs(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
    ) -> Result<usize, String> {
        let f = unsafe { (*api).Node_GetNumOutputs }
            .ok_or("OrtApi.Node_GetNumOutputs is null")?;
        let mut count = 0usize;
        let status = unsafe { f(node, &mut count) };
        Self::check(status)?;
        Ok(count)
    }

    unsafe fn node_get_outputs(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
        out: *mut *const ort::OrtValueInfo,
        count: usize,
    ) -> Result<(), String> {
        let f = unsafe { (*api).Node_GetOutputs }
            .ok_or("OrtApi.Node_GetOutputs is null")?;
        let status = unsafe { f(node, out, count) };
        Self::check(status)
    }

    unsafe fn node_op_type(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
    ) -> Result<String, String> {
        let f = unsafe { (*api).Node_GetOperatorType }
            .ok_or("OrtApi.Node_GetOperatorType is null")?;
        let mut ptr: *const std::ffi::c_char = std::ptr::null();
        let status = unsafe { f(node, &mut ptr) };
        Self::check(status)?;
        if ptr.is_null() {
            return Err("Node_GetOperatorType returned null".into());
        }
        Ok(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
    }

    unsafe fn node_domain(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
    ) -> Result<String, String> {
        let f = unsafe { (*api).Node_GetDomain }
            .ok_or("OrtApi.Node_GetDomain is null")?;
        let mut ptr: *const std::ffi::c_char = std::ptr::null();
        let status = unsafe { f(node, &mut ptr) };
        Self::check(status)?;
        if ptr.is_null() {
            return Ok(String::new());
        }
        Ok(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
    }

    unsafe fn node_since_version(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
    ) -> Result<i32, String> {
        let f = unsafe { (*api).Node_GetSinceVersion }
            .ok_or("OrtApi.Node_GetSinceVersion is null")?;
        let mut version: i32 = 0;
        let status = unsafe { f(node, &mut version) };
        Self::check(status)?;
        Ok(version)
    }

    unsafe fn node_name(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
    ) -> Result<String, String> {
        let f = unsafe { (*api).Node_GetName }
            .ok_or("OrtApi.Node_GetName is null")?;
        let mut ptr: *const std::ffi::c_char = std::ptr::null();
        let status = unsafe { f(node, &mut ptr) };
        Self::check(status)?;
        if ptr.is_null() {
            return Ok(String::new());
        }
        Ok(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
    }

    /// Read dtype and shape from an `OrtValueInfo*`.
    unsafe fn read_value_info(
        api: *const ort::OrtApi,
        info: *const ort::OrtValueInfo,
    ) -> Result<(String, DataType, Shape), String> {
        // Get name.
        let get_name = unsafe { (*api).GetValueInfoName }
            .ok_or("OrtApi.GetValueInfoName is null")?;
        let mut name_ptr: *const std::ffi::c_char = ptr::null();
        let status = unsafe { get_name(info, &mut name_ptr) };
        Self::check(status)?;
        let name = if name_ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned()
        };

        // Get type info.
        let get_type_info = unsafe { (*api).GetValueInfoTypeInfo }
            .ok_or("OrtApi.GetValueInfoTypeInfo is null")?;
        let mut type_info: *const ort::OrtTypeInfo = ptr::null();
        let status = unsafe { get_type_info(info, &mut type_info) };
        Self::check(status)?;

        if type_info.is_null() {
            return Ok((name, DataType::Undefined, Vec::new()));
        }

        // Cast to tensor info.
        let cast_fn = unsafe { (*api).CastTypeInfoToTensorInfo }
            .ok_or("OrtApi.CastTypeInfoToTensorInfo is null")?;
        let mut tensor_info: *const ort::OrtTensorTypeAndShapeInfo = ptr::null();
        let status = unsafe { cast_fn(type_info, &mut tensor_info) };
        Self::check(status)?;

        if tensor_info.is_null() {
            return Ok((name, DataType::Undefined, Vec::new()));
        }

        // Get element type.
        let get_elem_type = unsafe { (*api).GetTensorElementType }
            .ok_or("OrtApi.GetTensorElementType is null")?;
        let mut elem_type: ort::ONNXTensorElementDataType = 0;
        let status = unsafe { get_elem_type(tensor_info, &mut elem_type) };
        Self::check(status)?;
        let dtype = DataType::from_onnx(elem_type as i32).unwrap_or(DataType::Undefined);

        // Get dimensions.
        let get_dims_count = unsafe { (*api).GetDimensionsCount }
            .ok_or("OrtApi.GetDimensionsCount is null")?;
        let mut dims_count = 0usize;
        let status = unsafe { get_dims_count(tensor_info, &mut dims_count) };
        Self::check(status)?;

        let shape = if dims_count > 0 {
            let get_dims = unsafe { (*api).GetDimensions }
                .ok_or("OrtApi.GetDimensions is null")?;
            let mut dims: Vec<i64> = vec![0; dims_count];
            let status = unsafe { get_dims(tensor_info, dims.as_mut_ptr(), dims_count) };
            Self::check(status)?;

            Shape::from(
                dims.iter()
                    .map(|&d| {
                        if d < 0 {
                            onnx_runtime_ir::Dim::Symbolic(onnx_runtime_ir::SymbolId(0))
                        } else {
                            onnx_runtime_ir::Dim::Static(d as usize)
                        }
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            Vec::new()
        };

        Ok((name, dtype, shape))
    }

    /// Check an OrtStatus and convert to Result, extracting the error message.
    fn check(status: *mut ort::OrtStatus) -> Result<(), String> {
        if status.is_null() {
            return Ok(());
        }
        // Extract the real error message before releasing.
        let api = crate::status::host_api();
        let message = if !api.is_null() {
            // SAFETY: api is valid (set during CreateEpFactories).
            let msg_ptr = unsafe {
                (*api).GetErrorMessage.map(|f| f(status))
            };
            let msg = msg_ptr
                .filter(|&p| !p.is_null())
                .map(|p| {
                    // SAFETY: ORT guarantees a null-terminated string.
                    unsafe { std::ffi::CStr::from_ptr(p) }
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_else(|| "ORT API call failed".into());
            // Release the status only after we've copied the message.
            if let Some(release) = unsafe { (*api).ReleaseStatus } {
                unsafe { release(status) };
            }
            msg
        } else {
            "ORT API call failed (host API not available)".into()
        };
        Err(message)
    }
}
